"""Hierarchical NeRF with the optional FreeNeRF frequency regularizer."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    width: int = 256
    depth: int = 8
    pos_bands: int = 10
    dir_bands: int = 4
    coarse_samples: int = 64
    fine_samples: int = 128
    anneal_fraction: float = 0.5

    def __post_init__(self):
        if self.width < 2 or self.depth < 1:
            raise ValueError("The field needs a width of at least two and a positive depth.")
        if min(self.pos_bands, self.dir_bands) < 0:
            raise ValueError("Frequency band counts must be nonnegative.")
        if self.coarse_samples < 3 or self.fine_samples < 1:
            raise ValueError("Use at least three coarse samples and one additional fine sample.")
        if not 0 < self.anneal_fraction <= 1:
            raise ValueError("anneal_fraction must be in (0, 1].")


class FourierEncoding(nn.Module):
    """Raw xyz followed by sin(xyz), cos(xyz) at successive dyadic frequencies."""

    def __init__(self, bands: int):
        super().__init__()
        self.groups = 1 + 2 * bands
        self.out_dim = 3 * self.groups
        self.register_buffer("frequencies", 2.0 ** torch.arange(bands), persistent=False)
        self.register_buffer("group_indices", torch.arange(self.groups), persistent=False)

    def forward(self, xyz: Tensor, fraction: float) -> Tensor:
        angles = xyz.unsqueeze(-2) * self.frequencies[:, None]
        features = torch.stack((angles.sin(), angles.cos()), dim=-2).flatten(-3)
        encoded = torch.cat((xyz, features), dim=-1)
        if fraction >= 1.0:
            return encoded

        # FreeNeRF get_freq_reg_mask works on xyz triplets, starting with raw xyz.
        # One partially visible triplet bridges fully visible and masked features.
        visible_groups = 1.0 + fraction * self.groups
        mask = (visible_groups - self.group_indices).clamp(1e-8, 1.0)
        return encoded * mask.repeat_interleave(3)


class RadianceField(nn.Module):
    """A position-dependent density trunk and a view-conditioned RGB head."""

    def __init__(self, config: ModelConfig, position_dim: int, direction_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        # In the original eight-layer trunk, append the position after layer 5.
        for index in range(config.depth):
            input_dim = position_dim if index == 0 else config.width
            if index == 5:
                input_dim += position_dim
            self.layers.append(nn.Linear(input_dim, config.width))
        self.density = nn.Linear(config.width, 1)
        self.feature = nn.Linear(config.width, config.width)
        self.view = nn.Linear(config.width + direction_dim, config.width // 2)
        self.color = nn.Linear(config.width // 2, 3)
        # Initialize the density near a small positive constant. An entirely empty
        # ReLU field would block gradients to the density and color heads.
        nn.init.uniform_(self.density.weight, -1e-4, 1e-4)
        nn.init.constant_(self.density.bias, 0.1)

    def forward(self, positions: Tensor, directions: Tensor) -> tuple[Tensor, Tensor]:
        hidden = positions
        for index, layer in enumerate(self.layers):
            if index == 5:
                hidden = torch.cat((hidden, positions), dim=-1)
            hidden = F.relu(layer(hidden))
        sigma = F.relu(self.density(hidden).squeeze(-1))
        views = directions[:, None, :].expand(-1, positions.shape[1], -1)
        color_input = torch.cat((self.feature(hidden), views), dim=-1)
        rgb = torch.sigmoid(self.color(F.relu(self.view(color_input))))
        return rgb, sigma


def _importance_samples(bins: Tensor, weights: Tensor, count: int, randomized: bool) -> Tensor:
    """Draw from the piecewise-constant density on the intervals between bins."""
    # An empty ray yields a uniform distribution, keeping the fine pass trainable.
    mass = weights.detach() + 1e-5
    cumulative = torch.cumsum(mass / mass.sum(dim=-1, keepdim=True), dim=-1)
    cdf = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative), dim=-1)
    if randomized:
        quantiles = torch.rand(len(bins), count, device=bins.device, dtype=bins.dtype)
    else:
        quantiles = (torch.arange(count, device=bins.device, dtype=bins.dtype) + 0.5) / count
        quantiles = quantiles.expand(len(bins), -1).contiguous()
    right = torch.searchsorted(cdf, quantiles, right=True).clamp(1, cdf.shape[-1] - 1)
    left = right - 1
    cdf_left = cdf.gather(1, left)
    span = cdf.gather(1, right) - cdf_left
    local = (quantiles - cdf_left) / span.clamp_min(torch.finfo(span.dtype).tiny)
    return torch.lerp(bins.gather(1, left), bins.gather(1, right), local.clamp(0.0, 1.0))


def _composite(rgb: Tensor, density: Tensor, depths: Tensor, ray_lengths: Tensor,
               white_background: bool) -> tuple[Tensor, Tensor, Tensor]:
    # Accumulate optical thickness in float32 even when an AMP MLP returns fp16.
    optical_depth = density[:, :-1].float() * depths.diff(dim=-1) * ray_lengths
    prefix = torch.cat((torch.zeros_like(depths[:, :1]), optical_depth.cumsum(dim=-1)), dim=-1)
    transmittance = torch.exp(-prefix)
    alpha = -torch.expm1(-optical_depth)
    # Original NeRF treats the final sample as extending to infinity.
    tail = (density[:, -1:] > 0).to(alpha.dtype)
    weights = transmittance * torch.cat((alpha, tail), dim=-1)
    background = transmittance[:, -1:] * (1.0 - tail)
    color = (weights[..., None] * rgb).sum(dim=1)
    if white_background:
        color = color + background
    return color, weights, background


class NeRFRenderer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.position_encoding = FourierEncoding(config.pos_bands)
        self.direction_encoding = FourierEncoding(config.dir_bands)
        self.coarse = RadianceField(config, self.position_encoding.out_dim, self.direction_encoding.out_dim)
        self.fine = RadianceField(config, self.position_encoding.out_dim, self.direction_encoding.out_dim)

    def forward(self, origins: Tensor, directions: Tensor, near: float, far: float, *,
                progress: float = 1.0, regularized: bool = False,
                randomized: bool = False, white_background: bool = True) -> dict[str, Tensor]:
        """Render rays o + z*d. Near, far and the returned depth use camera-axis z.

        progress is the fraction of the entire training run completed. At inference
        use the progress stored for the checkpoint, including when rendering early.
        """
        if origins.ndim != 2 or origins.shape != directions.shape or origins.shape[-1] != 3:
            raise ValueError("Expected matching origins and directions of shape [rays, 3].")
        if not (0 <= near < far and math.isfinite(far)):
            raise ValueError("Expected finite camera-axis bounds with 0 <= near < far.")
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be a fraction in [0, 1].")

        fraction = min(progress / self.config.anneal_fraction, 1.0) if regularized else 1.0
        ray_lengths = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
        encoded_direction = self.direction_encoding(F.normalize(directions, dim=-1), fraction)
        depths = torch.linspace(near, far, self.config.coarse_samples,
                                device=origins.device, dtype=origins.dtype).expand(len(origins), -1)
        if randomized:
            midpoints = 0.5 * (depths[:, :-1] + depths[:, 1:])
            lower = torch.cat((depths[:, :1], midpoints), dim=-1)
            upper = torch.cat((midpoints, depths[:, -1:]), dim=-1)
            depths = torch.lerp(lower, upper, torch.rand_like(depths))

        def query(field, sample_depths):
            points = origins[:, None, :] + sample_depths[..., None] * directions[:, None, :]
            rgb, sigma = field(self.position_encoding(points, fraction), encoded_direction)
            return _composite(rgb, sigma, sample_depths, ray_lengths, white_background)

        rgb_coarse, coarse_weights, _ = query(self.coarse, depths)
        midpoints = 0.5 * (depths[:, :-1] + depths[:, 1:])
        new_depths = _importance_samples(midpoints, coarse_weights[:, 1:-1],
                                         self.config.fine_samples, randomized)
        fine_depths = torch.cat((depths, new_depths), dim=-1).sort(dim=-1).values
        rgb, weights, background = query(self.fine, fine_depths)
        depth = (weights * fine_depths).sum(-1) + far * background.squeeze(-1)
        return {"rgb": rgb, "rgb_coarse": rgb_coarse, "depth": depth}
