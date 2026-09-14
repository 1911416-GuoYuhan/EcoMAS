from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn


class FrozenTextEncoder(nn.Module):
    @property
    def output_dim(self) -> int:
        raise NotImplementedError


class PuppeteerStateEncoder(FrozenTextEncoder):
    """Qwen state encoder compatible with Puppeteer's policy checkpoints."""

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
        if device != "auto":
            self.model.to(torch.device(device))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        self._output_dim = int(self.model.config.hidden_size)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @staticmethod
    def truncate(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        messages = deepcopy(messages)
        length = sum(len(str(message.get("content", ""))) for message in messages)
        while length > 12000:
            for message in messages:
                content = str(message.get("content", ""))
                message["content"] = content[-int(len(content) * 0.75):]
            length = sum(len(str(message.get("content", ""))) for message in messages)
        return messages

    def _encode_messages(self, messages: list[dict[str, Any]]):
        messages = self.truncate(messages)
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=2048,
        )

    def forward(self, message_batches: list[list[dict[str, Any]]]) -> torch.Tensor:
        device = next(self.model.parameters()).device
        tokenized = [self._encode_messages(messages) for messages in message_batches]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [item["input_ids"][0] for item in tokenized],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(device)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [item["attention_mask"][0] for item in tokenized],
            batch_first=True,
            padding_value=0,
        ).to(device)
        base_model = getattr(self.model, "model", self.model)
        with torch.inference_mode():
            output = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        last_index = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_indices = torch.arange(output.last_hidden_state.size(0), device=device)
        states = output.last_hidden_state[batch_indices, last_index, :].detach()
        return states.float().cpu()


def build_encoder(backend: str, model_path: Path, device: str) -> FrozenTextEncoder:
    if backend != "hf":
        raise ValueError(
            "EcoMAS now requires the Puppeteer-compatible Hugging Face state encoder; "
            f"unsupported backend: {backend}"
        )
    return PuppeteerStateEncoder(model_path, device)
