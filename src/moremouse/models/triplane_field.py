"""Triplane NeRF field components."""

import torch
import torch.nn.functional as functional


class TriplaneField(torch.nn.Module):
    """Query density, color, and deformation from triplane features."""

    def __init__(self, channels: int, hidden_dim: int) -> None:
        """Initialize field MLP heads."""
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(channels * 3 + 3, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
        )
        self.density = torch.nn.Linear(hidden_dim, 1)
        self.color = torch.nn.Linear(hidden_dim, 3)
        self.deformation = torch.nn.Linear(hidden_dim, 3)

    def forward(self, triplanes: torch.Tensor, points: torch.Tensor) -> dict[str, torch.Tensor]:
        """Query batched points in normalized [-1, 1] coordinates."""
        features = sample_triplanes(triplanes, points)
        hidden = self.mlp(torch.cat([features, points], dim=-1))
        return {
            "density": functional.softplus(self.density(hidden)),
            "color": torch.sigmoid(self.color(hidden)),
            "deformation": self.deformation(hidden),
        }


def sample_triplanes(triplanes: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample XY, XZ, and YZ planes at query points."""
    batch, plane_count, channels, _, _ = triplanes.shape
    if plane_count != 3:
        raise ValueError("triplanes must have exactly 3 planes")
    coords = (points[..., [0, 1]], points[..., [0, 2]], points[..., [1, 2]])
    samples = []
    for plane, coord in enumerate(coords):
        grid = coord.view(batch, -1, 1, 2).clamp(-1.0, 1.0)
        value = functional.grid_sample(triplanes[:, plane], grid, align_corners=True)
        samples.append(value.squeeze(-1).transpose(1, 2).view(batch, *points.shape[1:-1], channels))
    return torch.cat(samples, dim=-1)
