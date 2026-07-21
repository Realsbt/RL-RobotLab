# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def illegal_contact_consecutive(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    consecutive_frames: int = 3,
) -> torch.Tensor:
    """Terminate when an illegal contact persists across consecutive sensor frames.

    IsaacLab's default illegal contact termination triggers when any frame in the
    contact history exceeds the threshold. This variant requires the selected body
    set to exceed the threshold on every frame in the recent window, which filters
    single-frame contact spikes while still terminating sustained torso contact.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    recent_forces = net_contact_forces[:, -consecutive_frames:, sensor_cfg.body_ids]
    recent_contact = torch.norm(recent_forces, dim=-1) > threshold
    frame_has_contact = torch.any(recent_contact, dim=-1)
    return torch.all(frame_has_contact, dim=1)
