"""Contact sensor extension that also records foot velocity at physics rate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors.contact_sensor.contact_sensor_data import ContactSensorData
from isaaclab.utils import configclass


@dataclass
class FootImpactSensorData(ContactSensorData):
    """Contact data augmented with world-frame linear velocity history."""

    lin_vel_w: torch.Tensor | None = None
    """Current foot linear velocity in world frame, shape ``(N, B, 3)``."""

    lin_vel_w_history: torch.Tensor | None = None
    """Velocity history ordered newest to oldest, shape ``(N, T, B, 3)``."""


class FootImpactSensor(ContactSensor):
    """Record contact force and foot velocity together at every physics step."""

    cfg: FootImpactSensorCfg

    def __init__(self, cfg: FootImpactSensorCfg):
        super().__init__(cfg)
        self._data = FootImpactSensorData()

    @property
    def data(self) -> FootImpactSensorData:
        self._update_outdated_buffers()
        return self._data

    def reset(self, env_ids: Sequence[int] | None = None):
        resolved_env_ids = slice(None) if env_ids is None else env_ids
        super().reset(env_ids)
        if self._data.lin_vel_w is not None:
            self._data.lin_vel_w[resolved_env_ids] = 0.0
        if self._data.lin_vel_w_history is not None:
            self._data.lin_vel_w_history[resolved_env_ids] = 0.0

    def _initialize_impl(self):
        super()._initialize_impl()
        self._data.lin_vel_w = torch.zeros(
            self._num_envs, self._num_bodies, 3, device=self._device
        )
        self._data.lin_vel_w_history = torch.zeros(
            self._num_envs, self.cfg.history_length, self._num_bodies, 3, device=self._device
        )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        resolved_env_ids = slice(None) if len(env_ids) == self._num_envs else env_ids
        super()._update_buffers_impl(env_ids)

        body_velocities = self.body_physx_view.get_velocities().view(
            -1, self._num_bodies, 6
        )[..., :3]
        self._data.lin_vel_w[resolved_env_ids] = body_velocities[resolved_env_ids]
        self._data.lin_vel_w_history[resolved_env_ids] = self._data.lin_vel_w_history[
            resolved_env_ids
        ].roll(1, dims=1)
        self._data.lin_vel_w_history[resolved_env_ids, 0] = self._data.lin_vel_w[resolved_env_ids]


@configclass
class FootImpactSensorCfg(ContactSensorCfg):
    """Configuration for synchronized foot force and velocity histories."""

    class_type: type = FootImpactSensor

    def __post_init__(self):
        if self.history_length < 2:
            raise ValueError("FootImpactSensorCfg.history_length must be at least two.")
