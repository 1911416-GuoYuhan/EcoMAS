from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn


class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, num_agents: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_agents),
        )

    def forward(self, encoded_state: torch.Tensor) -> torch.Tensor:
        return self.net(encoded_state)


@dataclass
class RouterDecision:
    agent_index: int
    agent_name: str
    probabilities: list[float]
    log_prob: torch.Tensor


class ArgmaxRouter:
    def __init__(self, task_name: str, agent_names: list[str], input_dim: int, device: str = "cpu") -> None:
        self.task_name = task_name
        self.agent_names = agent_names
        self.device = torch.device(device)
        self.model = RouterMLP(input_dim, len(agent_names)).to(self.device)

    def decide(self, encoded_state: torch.Tensor) -> RouterDecision:
        encoded_state = encoded_state.to(self.device)
        logits = self.model(encoded_state)
        probs = torch.softmax(logits, dim=-1)
        idx = int(torch.argmax(probs, dim=-1).item())
        return RouterDecision(
            agent_index=idx,
            agent_name=self.agent_names[idx],
            probabilities=[float(x) for x in probs.squeeze(0).detach().cpu().tolist()],
            log_prob=torch.log(probs.squeeze(0)[idx].clamp_min(1e-8)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "task_name": self.task_name,
                "agent_names": self.agent_names,
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    def load(self, path: Path) -> None:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        if payload["agent_names"] != self.agent_names:
            raise ValueError(
                f"Checkpoint agents {payload['agent_names']} do not match {self.agent_names}"
            )
        self.model.load_state_dict(payload["state_dict"])
