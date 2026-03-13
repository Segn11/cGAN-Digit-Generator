from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from torch import nn


IMAGE_SIZE = 28
IMAGE_SHAPE = (1, IMAGE_SIZE, IMAGE_SIZE)
LATENT_DIM = 100
NUM_CLASSES = 10
EMBEDDING_DIM = 10


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, EMBEDDING_DIM)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + EMBEDDING_DIM, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, IMAGE_SIZE * IMAGE_SIZE),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embedded_labels = self.label_embedding(labels)
        generator_input = torch.cat((noise, embedded_labels), dim=1)
        image = self.model(generator_input)
        return image.view(noise.size(0), *IMAGE_SHAPE)


def resolve_checkpoint_path() -> Path:
    backend_checkpoint = Path(__file__).resolve().parent / "generator_cgan.pth"
    if backend_checkpoint.exists():
        return backend_checkpoint

    workspace_checkpoint = Path(__file__).resolve().parents[2] / "generator_cgan.pth"
    if workspace_checkpoint.exists():
        return workspace_checkpoint

    raise FileNotFoundError(
        "generator_cgan.pth was not found in the backend folder or workspace root."
    )


def extract_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "generator_state_dict", "generator"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if all(isinstance(key, str) for key in checkpoint.keys()):
            return checkpoint  # type: ignore[return-value]

    raise TypeError("Unsupported generator checkpoint format.")


def _batch_norm_running_stats_are_default(layer: nn.BatchNorm1d) -> bool:
    if layer.running_mean is None or layer.running_var is None:
        return False

    zero_mean = torch.allclose(layer.running_mean, torch.zeros_like(layer.running_mean))
    unit_var = torch.allclose(layer.running_var, torch.ones_like(layer.running_var))
    return bool(zero_mean and unit_var)


def enable_batch_stat_fallback_if_needed(generator: Generator) -> bool:
    batch_norm_layers = [
        layer
        for layer in generator.modules()
        if isinstance(layer, nn.BatchNorm1d)
    ]

    if not batch_norm_layers:
        return False

    should_enable_fallback = all(
        _batch_norm_running_stats_are_default(layer) for layer in batch_norm_layers
    )

    if not should_enable_fallback:
        return False

    for layer in batch_norm_layers:
        layer.track_running_stats = False
        layer.running_mean = None
        layer.running_var = None
        layer.num_batches_tracked = None

    return True


def load_generator(
    checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> Generator:
    resolved_checkpoint = Path(checkpoint_path) if checkpoint_path else resolve_checkpoint_path()
    state_dict = extract_state_dict(torch.load(resolved_checkpoint, map_location=device))

    generator = Generator().to(device)
    generator.load_state_dict(state_dict)
    generator.eval()
    return generator