from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BENCHMARK_ROOT = Path("/home_bak/guoyuhan/benchmarks")
DEFAULT_QWEN_PATH = Path(
    "/data2/guoyuhan/qwen_semantic_clustering_feasibility/.hf_cache/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)


@dataclass(frozen=True)
class RuntimeConfig:
    benchmark_root: Path = DEFAULT_BENCHMARK_ROOT
    output_root: Path = PROJECT_ROOT / "runs"
    checkpoint_root: Path = PROJECT_ROOT / "checkpoints"
    llm_backend: str = "local_hf"
    llm_model_path: Path = DEFAULT_QWEN_PATH
    encoder_model_path: Path | None = None
    max_new_tokens: int = 768
    temperature: float = 0.0
    encoder_backend: str = "hf"
    device: str = "auto"
