from __future__ import annotations

from ecomas.agents import AgentSpec
from ecomas.config import TASK_CONFIG


TASK_REGISTRY = {
    task_name: {
        **task_config,
        "agents": [AgentSpec(**agent) for agent in task_config["agents"]],
    }
    for task_name, task_config in TASK_CONFIG.items()
}
