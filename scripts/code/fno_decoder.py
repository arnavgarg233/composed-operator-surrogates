from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PeriodicSpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / math.sqrt(in_channels)
        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, 2)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        coefficients = torch.fft.rfft(inputs, dim=-1)
        active_modes = min(self.modes, coefficients.shape[-1])
        output_coefficients = coefficients.new_zeros(
            inputs.shape[0], self.out_channels, coefficients.shape[-1]
        )
        weight = torch.view_as_complex(self.weight[..., :active_modes, :].contiguous())
        output_coefficients[..., :active_modes] = torch.einsum(
            "bim,iom->bom", coefficients[..., :active_modes], weight
        )
        return torch.fft.irfft(output_coefficients, n=inputs.shape[-1], dim=-1)


class PeriodicResidualBlock(nn.Module):
    def __init__(self, width: int, modes: int) -> None:
        super().__init__()
        self.spectral = PeriodicSpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.spectral(inputs) + self.local(inputs)
        return inputs + F.gelu(update)


class ConditionedPeriodicResidualBlock(nn.Module):
    def __init__(self, width: int, modes: int, latent_dim: int) -> None:
        super().__init__()
        self.spectral = PeriodicSpectralConv1d(width, width, modes)
        self.local = nn.Conv1d(width, width, 1)
        film_width = 2 * width
        self.film = nn.Sequential(
            nn.Linear(latent_dim, film_width),
            nn.GELU(),
            nn.Linear(film_width, 2 * width),
        )

    def forward(self, inputs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        update = self.spectral(inputs) + self.local(inputs)
        scale, shift = self.film(z).chunk(2, dim=-1)
        conditioned = (1.0 + scale.unsqueeze(-1)) * update + shift.unsqueeze(-1)
        return inputs + F.gelu(conditioned)


def _periodic_features(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    points = reference.shape[-1]
    coordinates = (
        2.0
        * torch.pi
        * torch.arange(points, device=reference.device, dtype=reference.dtype)
        / points
    )
    return torch.sin(coordinates), torch.cos(coordinates)


def _canonical_mean(inputs: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.sort(inputs, dim=dim).values.mean(dim=dim)


class TransitionSetEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        width: int = 64,
        modes: int = 16,
        blocks: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.lift = nn.Conv1d(4, width, 1)
        self.blocks = nn.ModuleList(
            PeriodicResidualBlock(width, modes) for _ in range(blocks)
        )
        self.transition_projection = nn.Sequential(
            nn.Linear(2 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.trajectory_projection = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
        )
        self.latent_projection = nn.Sequential(
            nn.Linear(width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, latent_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.ndim != 5 or context.shape[-2] != 2:
            raise ValueError(
                "context must have shape (batch, trajectories, transitions, 2, points)"
            )
        if context.shape[1] == 0 or context.shape[2] == 0 or context.shape[-1] == 0:
            raise ValueError("context set dimensions must be nonempty")
        batch, trajectories, transitions, _, points = context.shape
        flattened = context.reshape(batch * trajectories * transitions, 2, points)
        sine, cosine = _periodic_features(flattened)
        coordinates = (
            torch.stack((sine, cosine)).unsqueeze(0).expand(flattened.shape[0], -1, -1)
        )
        features = self.lift(torch.cat((flattened, coordinates), dim=1))
        for block in self.blocks:
            features = block(features)
        spatial_mean = features.mean(dim=-1)
        spatial_rms = torch.sqrt(
            features.square().mean(dim=-1) + torch.finfo(features.dtype).eps
        )
        transition_features = self.transition_projection(
            torch.cat((spatial_mean, spatial_rms), dim=-1)
        ).reshape(batch, trajectories, transitions, -1)
        trajectory_inputs = _canonical_mean(transition_features, dim=2)
        trajectory_features = self.trajectory_projection(trajectory_inputs)
        context_features = _canonical_mean(trajectory_features, dim=1)
        return self.latent_projection(context_features)


class ConditionedFNOGenerator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        width: int = 128,
        modes: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.lift = nn.Conv1d(3, width, 1)
        self.blocks = nn.ModuleList(
            ConditionedPeriodicResidualBlock(width, modes, latent_dim) for _ in range(6)
        )
        self.project = nn.Sequential(
            nn.Conv1d(width, width, 1),
            nn.GELU(),
            nn.Conv1d(width, 1, 1),
        )

    def forward(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        unbatched = state.ndim == 1
        if unbatched:
            state = state.unsqueeze(0)
        if state.ndim != 2:
            raise ValueError("state must have shape (batch, points) or (points,)")
        if z.ndim == 1:
            z = z.unsqueeze(0)
        if z.ndim != 2 or z.shape != (state.shape[0], self.latent_dim):
            raise ValueError("z must have shape (batch, latent_dim)")
        sine, cosine = _periodic_features(state)
        inputs = torch.stack(
            (state, sine.expand_as(state), cosine.expand_as(state)), dim=1
        )
        features = self.lift(inputs)
        for block in self.blocks:
            features = block(features, z)
        output = self.project(features).squeeze(1)
        return output.squeeze(0) if unbatched else output


class PDEOperatorModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        encoder_width: int = 64,
        encoder_modes: int = 16,
        encoder_blocks: int = 2,
        generator_width: int = 128,
        generator_modes: int = 32,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.context_encoder = TransitionSetEncoder(
            latent_dim=latent_dim,
            width=encoder_width,
            modes=encoder_modes,
            blocks=encoder_blocks,
        )
        self.generator = ConditionedFNOGenerator(
            latent_dim=latent_dim,
            width=generator_width,
            modes=generator_modes,
        )

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(context)

    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.generator(state, self.encode_context(context))
