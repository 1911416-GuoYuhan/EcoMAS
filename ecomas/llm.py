from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import torch


class LocalHFLLM:
    def __init__(
        self,
        model_path: Path,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        device: str = "auto",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map: Optional[str] = "auto" if device == "auto" and torch.cuda.is_available() else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.generate_messages(messages)

    def generate_messages(self, messages: list[dict[str, str]]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        device = next(self.model.parameters()).device
        original_padding_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "left"
        try:
            encoded = self.tokenizer(
                [text],
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
            ).to(device)
        finally:
            self.tokenizer.padding_side = original_padding_side
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=0.9,
                max_new_tokens=min(self.max_new_tokens, 96),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False,
            )
        generated = output[0][encoded["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class MockLLM:
    """Fast deterministic backend for wiring tests; real experiments use LocalHFLLM."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        options = re.findall(r"\b([A-J])\s*:", user_prompt)
        if "entailment" in user_prompt.lower() and "hypothesis" in user_prompt.lower():
            answer = "neutral"
        elif "boxed" in user_prompt.lower() or "problem:" in user_prompt.lower():
            answer = "0"
        else:
            answer = options[0] if options else "A"
        return (
            "ANALYSIS: Mock backend selected a deterministic placeholder so the "
            "MAS pipeline can be tested without loading the LLM.\n"
            f"CANDIDATE_ANSWER: {answer}"
        )

    def generate_messages(self, messages: list[dict[str, str]]) -> str:
        return self.generate(
            messages[0]["content"] if messages else "",
            messages[-1]["content"] if messages else "",
        )


def build_llm(backend: str, model_path: Path, max_new_tokens: int, temperature: float, device: str):
    if backend == "local_hf":
        return LocalHFLLM(model_path, max_new_tokens, temperature, device)
    if backend == "mock":
        return MockLLM()
    raise ValueError(f"Unknown LLM backend: {backend}")
