"""MUTE-specific reward terms for ToGo_LFs quiet locomotion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .mute_math import contact_phase, phase_weighted_vertical_velocity

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _phase_and_vertical_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    swing_duration: float,
    stance_duration: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    phase = contact_phase(
        contact_sensor.data.current_air_time[:, sensor_cfg.body_ids],
        contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids],
        swing_duration,
        stance_duration,
    )
    foot_velocity_z = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    return phase, foot_velocity_z


def mute_drop_foot_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    swing_duration: float,
    stance_duration: float,
) -> torch.Tensor:
    """Penalize downward foot velocity increasingly toward touchdown."""
    phase, foot_velocity_z = _phase_and_vertical_velocity(
        env, asset_cfg, sensor_cfg, swing_duration, stance_duration
    )
    drop_term, _ = phase_weighted_vertical_velocity(foot_velocity_z, phase)
    return drop_term


def mute_raise_foot_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    swing_duration: float,
    stance_duration: float,
) -> torch.Tensor:
    """Reward upward foot velocity primarily near the start of swing."""
    phase, foot_velocity_z = _phase_and_vertical_velocity(
        env, asset_cfg, sensor_cfg, swing_duration, stance_duration
    )
    _, raise_term = phase_weighted_vertical_velocity(foot_velocity_z, phase)
    return raise_term
