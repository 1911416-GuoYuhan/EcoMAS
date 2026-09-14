"""ecoMAS: deterministic single-path MAS benchmark runner."""

__version__ = "0.1.0"

from ecomas.semantic_clustering import SemanticClusteringResult, cluster_answers, semantic_kernel
from ecomas.uncertainty import SystemEntropyEstimate, estimate_system_entropy

__all__ = [
    "SemanticClusteringResult", "cluster_answers", "semantic_kernel",
    "SystemEntropyEstimate", "estimate_system_entropy",
]
