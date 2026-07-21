"""Observation terms for the ToGo_LFs backflip task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def action_pair(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Return current and previous policy actions in deployment order."""
    return torch.cat((env.action_manager.action, env.action_manager.prev_action), dim=-1)


def foot_contacts(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 2.0,
) -> torch.Tensor:
    """Return binary foot-contact flags as floating-point critic observations."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = torch.linalg.norm(
        contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1
    )
    return (force >= force_threshold).float()


def privileged_backflip_state(
    env: "ManagerBasedRLEnv",
    command_name: str = "backflip_phase",
) -> torch.Tensor:
    """Expose simulator-only skill state to the critic, never to the actor."""
    state = env.command_manager.get_term(command_name)
    return torch.stack(
        (
            state.max_backward_rotation / (2.0 * torch.pi),
            state.airborne.float(),
            state.has_landed.float(),
            state.contact_count.float() / 4.0,
        ),
        dim=-1,
    )
