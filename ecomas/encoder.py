from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


class FrozenTextEncoder(nn.Module):
    @property
    def output_dim(self) -> int:
        raise NotImplementedError


class HashingTextEncoder(FrozenTextEncoder):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dim = dim
        for param in self.parameters():
            param.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return self.dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        vectors = []
        for text in texts:
            vec = torch.zeros(self.dim, dtype=torch.float32)
            for token in text.split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = vec.norm(p=2)
            vectors.append(vec / norm if norm > 0 else vec)
        return torch.stack(vectors, dim=0)


class HFMeanPoolingEncoder(FrozenTextEncoder):
    def __init__(self, model_path: Path, device: str = "auto") -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        device_map = "auto" if device == "auto" and torch.cuda.is_available() else None
        self.model = AutoModel.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self._output_dim = int(self.model.config.hidden_size)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to(device)
        with torch.inference_mode():
            output = self.model(**encoded)
        mask = encoded["attention_mask"].unsqueeze(-1).to(output.last_hidden_state.dtype)
        pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return pooled.float().cpu()


def build_encoder(backend: str, dim: int, model_path: Path, device: str) -> FrozenTextEncoder:
    if backend == "hash":
        return HashingTextEncoder(dim)
    if backend == "hf":
        return HFMeanPoolingEncoder(model_path, device)
    raise ValueError(f"Unknown encoder backend: {backend}")
