from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.json"
APP_CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
PATH_CONFIG = APP_CONFIG["paths"]
ARTIFACT_CONFIG = APP_CONFIG["artifacts"]
RUNTIME_CONFIG = APP_CONFIG["runtime"]
ENCODER_CONFIG = APP_CONFIG["encoder"]
ROUTER_CONFIG = APP_CONFIG["router"]
SAMPLING_CONFIG = APP_CONFIG["sampling"]
COMMAND_CONFIG = APP_CONFIG["commands"]
METRIC_CONFIG = APP_CONFIG["metrics"]
DATASET_CONFIG = APP_CONFIG["datasets"]
PROMPT_CONFIG = APP_CONFIG["prompts"]
TASK_CONFIG = APP_CONFIG["tasks"]


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() or value.startswith("<") else PROJECT_ROOT / path


@dataclass(frozen=True)
class RuntimeConfig:
    benchmark_root: Path = project_path(PATH_CONFIG["benchmark_root"])
    output_root: Path = project_path(PATH_CONFIG["output_root"])
    checkpoint_root: Path = project_path(PATH_CONFIG["checkpoint_root"])
    llm_backend: str = RUNTIME_CONFIG["llm_backend"]
    llm_model_path: Path = project_path(PATH_CONFIG["generation_model"])
    encoder_model_path: Path = project_path(PATH_CONFIG["encoder_model"])
    max_new_tokens: int = RUNTIME_CONFIG["max_new_tokens"]
    temperature: float = RUNTIME_CONFIG["inference_temperature"]
    encoder_backend: str = RUNTIME_CONFIG["encoder_backend"]
    device: str = RUNTIME_CONFIG["device"]
