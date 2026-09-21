import json

import pytest
import torch

from ecomas.config import DEFAULT_ENCODER_MODEL_PATH, RuntimeConfig
from ecomas.encoder import prepare_messages_for_template
from ecomas.router import ArgmaxRouter


def test_default_encoder_is_nemotron_with_8192_hidden_size():
    config = json.loads((DEFAULT_ENCODER_MODEL_PATH / "config.json").read_text(encoding="utf-8"))
    assert RuntimeConfig().encoder_model_path == DEFAULT_ENCODER_MODEL_PATH
    assert config["model_type"] == "llama"
    assert config["hidden_size"] == 8192


def test_router_mlp_and_checkpoint_follow_encoder_dimension(tmp_path):
    router = ArgmaxRouter(
        "mmlu_pro",
        ["first", "second"],
        8192,
        encoder_model_path=DEFAULT_ENCODER_MODEL_PATH,
    )
    assert router.model.fc1.in_features == 8192
    assert router.model.fc1.out_features == 512

    checkpoint = tmp_path / "router.pt"
    router.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["input_dim"] == 8192
    assert payload["metadata"]["encoder_output_dim"] == 8192
    assert payload["metadata"]["encoder_model_path"] == str(DEFAULT_ENCODER_MODEL_PATH)


def test_old_encoder_checkpoint_is_rejected(tmp_path):
    old_router = ArgmaxRouter("mmlu_pro", ["first", "second"], 3584)
    checkpoint = tmp_path / "old-router.pt"
    old_router.save(checkpoint)

    nemotron_router = ArgmaxRouter("mmlu_pro", ["first", "second"], 8192)
    with pytest.raises(ValueError, match="input dimension"):
        nemotron_router.load(checkpoint)


def test_nemotron_template_keeps_system_context_visible():
    tokenizer_config = json.loads(
        (DEFAULT_ENCODER_MODEL_PATH / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    prepared = prepare_messages_for_template(
        [
            {"role": "system", "content": "the benchmark question"},
            {"role": "user", "content": "previous agent evidence"},
        ],
        tokenizer_config["chat_template"],
    )
    assert all(message["role"] != "system" for message in prepared)
    assert "the benchmark question" in prepared[0]["content"]
    assert "previous agent evidence" in prepared[0]["content"]
