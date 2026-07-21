"""Termination terms that allow a complete pitch rotation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def torso_contact_after_warmup(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    warmup_time_s: float,
    force_threshold: float,
) -> torch.Tensor:
    """Terminate sustained torso contact, but never terminate on pitch angle."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.linalg.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1
    )
    has_contact = torch.any(force >= force_threshold, dim=-1)
    return has_contact & (env.episode_length_buf * env.step_dt >= warmup_time_s)


def base_too_low_after_warmup(
    env: "ManagerBasedRLEnv",
    minimum_height: float,
    warmup_time_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate unrecoverable low-base states after reset settling."""
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.root_pos_w[:, 2] < minimum_height) & (
        env.episode_length_buf * env.step_dt >= warmup_time_s
    )


def touchdown_before_rotation(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
) -> torch.Tensor:
    """Terminate when the first aerial touchdown occurs before a full flip.

    The command term freezes ``max_backward_rotation`` at first foot contact,
    so this condition cannot be satisfied later by a ground bounce or tumble.
    """
    if minimum_rotation <= 0.0:
        raise ValueError("minimum_rotation must be positive.")
    state = env.command_manager.get_term(command_name)
    return state.has_touched_down & (
        (state.max_backward_rotation < minimum_rotation)
        | state.has_invalid_rotation_axis
    )
