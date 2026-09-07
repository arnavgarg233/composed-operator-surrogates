from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fno_decoder import ConditionedFNOGenerator


class SpatialTrajectoryBlock(nn.Module):
    """A nonlinear periodic feature block for observed state transitions."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.first = nn.Conv1d(
            width,
            width,
            kernel_size=5,
            padding=2,
            padding_mode="circular",
        )
        self.second = nn.Conv1d(
            width,
            width,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.GroupNorm(1, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        update = self.first(inputs)
        update = self.second(F.gelu(update))
        return inputs + F.gelu(self.norm(update))


class TrajectoryContextEncoder(nn.Module):
    """Encode one unlabeled raw demonstration path into one latent flow code."""

    def __init__(
        self,
        latent_dim: int = 16,
        width: int = 24,
        temporal_width: int = 32,
        sampled_transitions: int = 16,
    ) -> None:
        super().__init__()
        if sampled_transitions < 2:
            raise ValueError("sampled_transitions must be at least two")
        self.latent_dim = latent_dim
        self.sampled_transitions = sampled_transitions
        self.lift = nn.Conv1d(3, width, kernel_size=1)
        self.spatial_blocks = nn.ModuleList(
            SpatialTrajectoryBlock(width) for _ in range(2)
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(2 * width, temporal_width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(temporal_width, temporal_width, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.project = nn.Sequential(
            nn.Linear(4 * temporal_width, 2 * temporal_width),
            nn.GELU(),
            nn.Linear(2 * temporal_width, latent_dim),
            nn.Tanh(),
        )

    def _transition_indices(self, path: torch.Tensor) -> torch.Tensor:
        available = path.shape[-2] - 1
        count = min(available, self.sampled_transitions)
        if count == available:
            return torch.arange(available, device=path.device)
        return torch.linspace(
            0,
            available - 1,
            count,
            device=path.device,
        ).round().to(torch.long)

    def forward(self, demonstration: torch.Tensor) -> torch.Tensor:
        unbatched = demonstration.ndim == 2
        if unbatched:
            demonstration = demonstration.unsqueeze(0)
        if demonstration.ndim != 3:
            raise ValueError(
                "demonstration must have shape (batch, frames, points) "
                "or (frames, points)"
            )
        if demonstration.shape[-2] < 2 or demonstration.shape[-1] < 2:
            raise ValueError("demonstration needs at least two frames and points")

        indices = self._transition_indices(demonstration)
        before = demonstration.index_select(-2, indices)
        after = demonstration.index_select(-2, indices + 1)
        transition = torch.stack((before, after, after - before), dim=-2)
        batch, count, _, points = transition.shape
        features = self.lift(transition.reshape(batch * count, 3, points))
        for block in self.spatial_blocks:
            features = block(features)
        mean = features.mean(dim=-1)
        rms = torch.sqrt(
            features.square().mean(dim=-1) + torch.finfo(features.dtype).eps
        )
        temporal_input = torch.cat((mean, rms), dim=-1)
        temporal_input = temporal_input.reshape(batch, count, -1).transpose(1, 2)
        temporal = self.temporal(temporal_input)
        summary = torch.cat(
            (
                temporal[..., 0],
                temporal[..., -1],
                temporal.mean(dim=-1),
                torch.sqrt(
                    temporal.square().mean(dim=-1)
                    + torch.finfo(temporal.dtype).eps
                ),
            ),
            dim=-1,
        )
        code = 4.0 * self.project(summary)
        return code.squeeze(0) if unbatched else code


class TrajectoryConditionedModel(nn.Module):
    """Predict state evolution from raw demonstrations and no identity input."""

    def __init__(
        self,
        *,
        latent_dim: int = 16,
        encoder_width: int = 24,
        temporal_width: int = 32,
        sampled_transitions: int = 16,
        decoder_width: int = 32,
        decoder_modes: int = 24,
        state_mean: float = 0.0,
        state_scale: float = 1.0,
        increment_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if state_scale <= 0.0:
            raise ValueError("state_scale must be positive")
        if increment_scale <= 0.0:
            raise ValueError("increment_scale must be positive")
        self.latent_dim = latent_dim
        self.context_encoder = TrajectoryContextEncoder(
            latent_dim=latent_dim,
            width=encoder_width,
            temporal_width=temporal_width,
            sampled_transitions=sampled_transitions,
        )
        self.decoder = ConditionedFNOGenerator(
            latent_dim=latent_dim,
            width=decoder_width,
            modes=decoder_modes,
        )
        self.register_buffer("state_mean", torch.tensor(float(state_mean)))
        self.register_buffer("state_scale", torch.tensor(float(state_scale)))
        self.register_buffer("increment_scale", torch.tensor(float(increment_scale)))

    def set_normalization(self, mean: float, scale: float) -> None:
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        with torch.no_grad():
            self.state_mean.fill_(float(mean))
            self.state_scale.fill_(float(scale))

    def encode_context(self, demonstration: torch.Tensor) -> torch.Tensor:
        normalized = (demonstration - self.state_mean) / self.state_scale
        return self.context_encoder(normalized)

    def step_encoded(self, state: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        normalized = (state - self.state_mean) / self.state_scale
        increment = self.decoder(normalized, code)
        next_normalized = normalized + self.increment_scale * increment
        return next_normalized * self.state_scale + self.state_mean

    def forward(
        self,
        state: torch.Tensor,
        demonstration: torch.Tensor,
    ) -> torch.Tensor:
        return self.step_encoded(state, self.encode_context(demonstration))

    def rollout(
        self,
        initial_state: torch.Tensor,
        ordered_demonstrations: torch.Tensor,
    ) -> torch.Tensor:
        unbatched = initial_state.ndim == 1
        if unbatched:
            if ordered_demonstrations.ndim != 3:
                raise ValueError(
                    "an unbatched rollout needs demonstrations shaped "
                    "(placements, frames, points)"
                )
            initial_state = initial_state.unsqueeze(0)
            ordered_demonstrations = ordered_demonstrations.unsqueeze(0)
        if initial_state.ndim != 2 or ordered_demonstrations.ndim != 4:
            raise ValueError(
                "batched inputs must have shapes (batch, points) and "
                "(batch, placements, frames, points)"
            )
        if ordered_demonstrations.shape[0] != initial_state.shape[0]:
            raise ValueError("state and demonstrations must share a batch size")
        if ordered_demonstrations.shape[-1] != initial_state.shape[-1]:
            raise ValueError("state and demonstrations must share point count")
        if ordered_demonstrations.shape[-2] < 2:
            raise ValueError("each demonstration needs at least two frames")

        batch, placements, frames, points = ordered_demonstrations.shape
        flattened = ordered_demonstrations.reshape(
            batch * placements,
            frames,
            points,
        )
        codes = self.encode_context(flattened).reshape(batch, placements, -1)
        state = initial_state
        trajectory = [state]
        for placement in range(placements):
            code = codes[:, placement]
            for _ in range(frames - 1):
                state = self.step_encoded(state, code)
                trajectory.append(state)
        result = torch.stack(trajectory, dim=1)
        return result.squeeze(0) if unbatched else result
