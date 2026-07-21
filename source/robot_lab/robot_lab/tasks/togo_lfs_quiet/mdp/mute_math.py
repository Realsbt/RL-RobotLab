"""Tensor operations shared by MUTE reward terms and tests."""

from __future__ import annotations

import torch


def contact_phase(
    current_air_time: torch.Tensor,
    current_contact_time: torch.Tensor,
    swing_duration: float,
    stance_duration: float,
) -> torch.Tensor:
    """Build a causal 0-to-1 swing and 1-to-0 stance phase from contact timing."""
    if swing_duration <= 0.0 or stance_duration <= 0.0:
        raise ValueError("swing_duration and stance_duration must be positive")

    swing_phase = torch.clamp(current_air_time / swing_duration, min=0.0, max=1.0)
    stance_phase = 1.0 - torch.clamp(current_contact_time / stance_duration, min=0.0, max=1.0)
    return torch.where(current_contact_time > 0.0, stance_phase, swing_phase)


def phase_weighted_vertical_velocity(
    foot_velocity_z: torch.Tensor,
    phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-environment MUTE drop penalties and raise rewards."""
    if foot_velocity_z.shape != phase.shape:
        raise ValueError(
            f"foot_velocity_z and phase must have the same shape, got {foot_velocity_z.shape} and {phase.shape}"
        )

    drop_velocity = torch.clamp(-foot_velocity_z, min=0.0)
    raise_velocity = torch.clamp(foot_velocity_z, min=0.0)
    drop_term = torch.sum(torch.exp(phase) * torch.square(drop_velocity), dim=-1)
    raise_term = torch.sum(torch.exp(-phase) * torch.square(raise_velocity), dim=-1)
    return drop_term, raise_term
