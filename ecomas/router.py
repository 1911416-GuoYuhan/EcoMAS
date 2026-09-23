from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from abc import ABC, abstractmethod

from ecomas.config import METRIC_CONFIG, ROUTER_CONFIG, RUNTIME_CONFIG


class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, num_agents: int) -> None:
        super().__init__()
        first, second, third = ROUTER_CONFIG["hidden_dimensions"]
        self.fc1 = nn.Linear(input_dim, first)
        self.fc2 = nn.Linear(first, second)
        self.fc3 = nn.Linear(second, third)
        self.fc4 = nn.Linear(third, num_agents)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoded_state: torch.Tensor) -> torch.Tensor:
        encoded_state = self.relu(self.fc1(encoded_state))
        encoded_state = self.relu(self.fc2(encoded_state))
        encoded_state = self.relu(self.fc3(encoded_state))
        return self.softmax(self.fc4(encoded_state))


@dataclass
class RouterDecision:
    agent_index: int
    agent_name: str
    probabilities: list[float]
    log_prob: torch.Tensor


class RouterPolicy(ABC):
    @abstractmethod
    def decide(self, encoded_state: torch.Tensor, mode: str = RUNTIME_CONFIG["router_modes"][0], seed: int | None = None) -> RouterDecision:
        raise NotImplementedError


class ArgmaxRouter(RouterPolicy):
    def __init__(
        self,
        task_name: str,
        agent_names: list[str],
        input_dim: int,
        device: str = ROUTER_CONFIG["default_device"],
        encoder_model_path: Path | None = None,
    ) -> None:
        self.task_name = task_name
        self.agent_names = agent_names
        self.device = torch.device(device)
        self.encoder_model_path = Path(encoder_model_path) if encoder_model_path else None
        self.model = RouterMLP(input_dim, len(agent_names)).to(self.device)

    def decide(self, encoded_state: torch.Tensor, mode: str = RUNTIME_CONFIG["router_modes"][0], seed: int | None = None) -> RouterDecision:
        if mode not in RUNTIME_CONFIG["router_modes"]:
            raise ValueError(f"Unknown router mode: {mode}")
        encoded_state = encoded_state.to(self.device)
        probs = self.model(encoded_state)
        if mode == RUNTIME_CONFIG["router_modes"][0]:
            idx = int(torch.argmax(probs, dim=-1).item())
        else:
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
            idx = int(torch.multinomial(probs.squeeze(0), num_samples=1, generator=generator).item())
        return RouterDecision(
            agent_index=idx,
            agent_name=self.agent_names[idx],
            probabilities=[float(x) for x in probs.squeeze(0).detach().cpu().tolist()],
            log_prob=torch.log(probs.squeeze(0)[idx].clamp_min(METRIC_CONFIG["router_probability_floor"])),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "input_dim": self.model.fc1.in_features,
                "output_dim": self.model.fc4.out_features,
                "metadata": {
                    "task_name": self.task_name,
                    "agent_names": self.agent_names,
                    "format": "ecomas_router_mlp_v2",
                    "encoder_model_path": str(self.encoder_model_path) if self.encoder_model_path else None,
                    "encoder_output_dim": self.model.fc1.in_features,
                },
            },
            path,
        )

    def load(self, path: Path) -> None:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location=self.device)

        if "model_state_dict" not in payload:
            raise ValueError(
                "Unsupported router checkpoint: expected a Puppeteer policy checkpoint "
                "with model_state_dict/input_dim/output_dim"
            )
        checkpoint_input_dim = int(payload["input_dim"])
        checkpoint_output_dim = int(payload["output_dim"])
        if checkpoint_output_dim != len(self.agent_names):
            raise ValueError(
                "Checkpoint output dimension does not match EcoMAS agents: "
                f"checkpoint={checkpoint_output_dim}, expected={len(self.agent_names)}"
            )
        if checkpoint_input_dim != self.model.fc1.in_features:
            raise ValueError(
                "Checkpoint input dimension does not match EcoMAS encoder: "
                f"checkpoint={checkpoint_input_dim}, expected={self.model.fc1.in_features}"
            )
        self.model.load_state_dict(payload["model_state_dict"])
