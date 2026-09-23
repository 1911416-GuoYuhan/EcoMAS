from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ecomas.config import ENCODER_CONFIG, RUNTIME_CONFIG


def prepare_messages_for_template(
    messages: list[dict[str, Any]],
    chat_template: str | None,
) -> list[dict[str, Any]]:
    prepared = deepcopy(messages)
    template = str(chat_template or "")
    supports_system = any(
        marker in template
        for marker in (
            "message['role'] == 'system'",
            'message["role"] == "system"',
            "message.role == 'system'",
            'message.role == "system"',
        )
    )
    if supports_system:
        return prepared
    system_content = "\n\n".join(
        str(message.get("content", ""))
        for message in prepared
        if message.get("role") == "system"
    ).strip()
    non_system = [message for message in prepared if message.get("role") != "system"]
    if not system_content:
        return non_system
    prefix = f"System context:\n{system_content}"
    first_user = next(
        (message for message in non_system if message.get("role") == "user"),
        None,
    )
    if first_user is None:
        non_system.insert(0, {"role": "user", "content": prefix})
    else:
        first_user["content"] = f"{prefix}\n\n{first_user.get('content', '')}"
    return non_system


class FrozenTextEncoder(nn.Module):
    @property
    def output_dim(self) -> int:
        raise NotImplementedError


class HFStateEncoder(FrozenTextEncoder):
    def __init__(self, model_path: Path, device: str = RUNTIME_CONFIG["device"]) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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
        while length > ENCODER_CONFIG["max_context_characters"]:
            for message in messages:
                content = str(message.get("content", ""))
                message["content"] = content[-int(len(content) * ENCODER_CONFIG["truncation_ratio"]):]
            length = sum(len(str(message.get("content", ""))) for message in messages)
        return messages

    def _encode_messages(self, messages: list[dict[str, Any]]):
        messages = self.truncate(messages)
        messages = prepare_messages_for_template(
            messages, getattr(self.tokenizer, "chat_template", None)
        )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=ENCODER_CONFIG["max_length"],
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
            "EcoMAS requires a Hugging Face state encoder; "
            f"unsupported backend: {backend}"
        )
    return HFStateEncoder(model_path, device)


PuppeteerStateEncoder = HFStateEncoder
