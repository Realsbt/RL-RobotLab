"""Pure tensor helpers for the ToGo_LFs backflip task."""

from __future__ import annotations

import math

import torch


def linear_clearance_scale(
    value: torch.Tensor,
    minimum_value: float,
    full_reward_value: float,
) -> torch.Tensor:
    """Map a safety quantity to a smooth 0-to-1 reward multiplier."""
    if full_reward_value <= minimum_value:
        raise ValueError("full_reward_value must be greater than minimum_value.")
    return torch.clamp(
        (value - minimum_value) / (full_reward_value - minimum_value),
        min=0.0,
        max=1.0,
    )


def centroidal_angular_velocity(
    masses: torch.Tensor,
    body_positions_w: torch.Tensor,
    body_linear_velocities_w: torch.Tensor,
    body_angular_velocities_w: torch.Tensor,
    body_inertias_w: torch.Tensor,
) -> torch.Tensor:
    """Compute whole-system angular velocity about its center of mass.

    The calculation includes both each body's spin momentum and the orbital
    momentum produced by its COM motion. Dividing the total angular momentum
    by the composite inertia rewards true whole-body rotation and also exposes
    useful inertia reduction from an aerial tuck.
    """
    expected_vector_shape = (*masses.shape, 3)
    expected_matrix_shape = (*masses.shape, 3, 3)
    if body_positions_w.shape != expected_vector_shape:
        raise ValueError("body_positions_w has an incompatible shape.")
    if body_linear_velocities_w.shape != expected_vector_shape:
        raise ValueError("body_linear_velocities_w has an incompatible shape.")
    if body_angular_velocities_w.shape != expected_vector_shape:
        raise ValueError("body_angular_velocities_w has an incompatible shape.")
    if body_inertias_w.shape != expected_matrix_shape:
        raise ValueError("body_inertias_w has an incompatible shape.")

    mass = masses.unsqueeze(-1)
    total_mass = torch.sum(mass, dim=1).clamp_min(1.0e-8)
    system_com = torch.sum(mass * body_positions_w, dim=1) / total_mass
    system_linear_velocity = (
        torch.sum(mass * body_linear_velocities_w, dim=1) / total_mass
    )

    relative_position = body_positions_w - system_com.unsqueeze(1)
    relative_velocity = body_linear_velocities_w - system_linear_velocity.unsqueeze(1)
    spin_momentum = torch.matmul(
        body_inertias_w, body_angular_velocities_w.unsqueeze(-1)
    ).squeeze(-1)
    orbital_momentum = torch.linalg.cross(
        relative_position, mass * relative_velocity, dim=-1
    )
    angular_momentum = torch.sum(spin_momentum + orbital_momentum, dim=1)

    identity = torch.eye(
        3, device=body_positions_w.device, dtype=body_positions_w.dtype
    ).view(1, 1, 3, 3)
    radius_squared = torch.sum(torch.square(relative_position), dim=-1)
    outer_product = relative_position.unsqueeze(-1) * relative_position.unsqueeze(-2)
    parallel_axis = masses[..., None, None] * (
        radius_squared[..., None, None] * identity - outer_product
    )
    composite_inertia = torch.sum(body_inertias_w + parallel_axis, dim=1)
    composite_inertia = composite_inertia + 1.0e-7 * identity[:, 0]
    return torch.linalg.solve(
        composite_inertia, angular_momentum.unsqueeze(-1)
    ).squeeze(-1)


def multiscale_phase_features(
    elapsed_s: torch.Tensor,
    episode_length_s: float,
    phase_cycles: float = 1.0,
) -> torch.Tensor:
    """Encode normalized episode time with three sine/cosine frequencies."""
    if episode_length_s <= 0.0:
        raise ValueError("episode_length_s must be positive.")
    if phase_cycles <= 0.0:
        raise ValueError("phase_cycles must be positive.")

    normalized_time = torch.clamp(elapsed_s / episode_length_s, min=0.0, max=1.0)
    phase = 2.0 * math.pi * phase_cycles * normalized_time
    return torch.stack(
        (
            torch.sin(phase),
            torch.cos(phase),
            torch.sin(phase / 2.0),
            torch.cos(phase / 2.0),
            torch.sin(phase / 4.0),
            torch.cos(phase / 4.0),
        ),
        dim=-1,
    )


def linear_rotation_projected_gravity(
    elapsed_s: torch.Tensor,
    rotation_start_s: float,
    rotation_end_s: float,
    backward_pitch_sign: float = -1.0,
) -> torch.Tensor:
    """Return a linear full-turn gravity reference used by Genesis-backflip.

    Unlike :func:`desired_projected_gravity`, this reference deliberately has
    constant scheduled pitch velocity between the two time boundaries.  It is
    kept separate so the existing smooth staged curriculum and its checkpoints
    retain exactly the same semantics.
    """
    if rotation_end_s <= rotation_start_s:
        raise ValueError("rotation_end_s must be greater than rotation_start_s.")
    if backward_pitch_sign not in (-1.0, 1.0):
        raise ValueError("backward_pitch_sign must be either -1.0 or 1.0.")

    progress = torch.clamp(
        (elapsed_s - rotation_start_s) / (rotation_end_s - rotation_start_s),
        min=0.0,
        max=1.0,
    )
    pitch = backward_pitch_sign * 2.0 * math.pi * progress
    zeros = torch.zeros_like(pitch)
    return torch.stack((torch.sin(pitch), zeros, -torch.cos(pitch)), dim=-1)


def rotation_schedule(
    elapsed_s: torch.Tensor,
    rotation_start_s: float,
    rotation_end_s: float,
) -> torch.Tensor:
    """Return a smooth 0-to-1 rotation progress over the requested time window."""
    if rotation_end_s <= rotation_start_s:
        raise ValueError("rotation_end_s must be greater than rotation_start_s.")

    progress = torch.clamp(
        (elapsed_s - rotation_start_s) / (rotation_end_s - rotation_start_s),
        min=0.0,
        max=1.0,
    )
    # Quintic smoothstep keeps angular velocity and acceleration continuous at
    # both ends of the reference window.
    return progress**3 * (progress * (progress * 6.0 - 15.0) + 10.0)


def desired_projected_gravity(
    elapsed_s: torch.Tensor,
    rotation_start_s: float,
    rotation_end_s: float,
    backward_pitch_sign: float = -1.0,
) -> torch.Tensor:
    """Return gravity in the body frame for a scheduled full pitch rotation.

    ``backward_pitch_sign`` maps positive task rotation progress to the robot's
    world-frame pitch convention. For the LFS/Isaac convention a backflip is
    expected to use negative world-Y angular velocity, hence the default -1.
    """
    if backward_pitch_sign not in (-1.0, 1.0):
        raise ValueError("backward_pitch_sign must be either -1.0 or 1.0.")

    progress = rotation_schedule(elapsed_s, rotation_start_s, rotation_end_s)
    pitch = backward_pitch_sign * 2.0 * math.pi * progress
    zeros = torch.zeros_like(pitch)
    return torch.stack((torch.sin(pitch), zeros, -torch.cos(pitch)), dim=-1)


def positive_rotation_increment(
    signed_rotation: torch.Tensor,
    previous_max_rotation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update monotonic rotation progress and return only newly gained angle."""
    if signed_rotation.shape != previous_max_rotation.shape:
        raise ValueError("signed_rotation and previous_max_rotation must have the same shape.")

    max_rotation = torch.maximum(previous_max_rotation, signed_rotation)
    increment = torch.clamp(max_rotation - previous_max_rotation, min=0.0)
    return max_rotation, increment


def unwrapped_projected_gravity_pitch_increment(
    projected_gravity_b: torch.Tensor,
    previous_wrapped_angle: torch.Tensor,
    backward_pitch_sign: float = -1.0,
    minimum_xz_norm: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure a signed pitch increment from the body-frame gravity winding.

    A genuine backflip makes the gravity vector wind once in the body's x-z
    plane.  Steps with a small x-z projection are rejected because the robot
    is too far into roll for a meaningful pitch angle.
    """
    if projected_gravity_b.shape != (*previous_wrapped_angle.shape, 3):
        raise ValueError("projected_gravity_b has an incompatible shape.")
    if backward_pitch_sign not in (-1.0, 1.0):
        raise ValueError("backward_pitch_sign must be either -1.0 or 1.0.")
    if not 0.0 < minimum_xz_norm <= 1.0:
        raise ValueError("minimum_xz_norm must be in (0, 1].")

    wrapped_angle = backward_pitch_sign * torch.atan2(
        projected_gravity_b[..., 0], -projected_gravity_b[..., 2]
    )
    raw_delta = wrapped_angle - previous_wrapped_angle
    wrapped_delta = torch.atan2(torch.sin(raw_delta), torch.cos(raw_delta))
    xz_norm = torch.linalg.vector_norm(
        projected_gravity_b[..., (0, 2)], dim=-1
    )
    valid = xz_norm >= minimum_xz_norm
    return wrapped_angle, torch.where(valid, wrapped_delta, 0.0)
