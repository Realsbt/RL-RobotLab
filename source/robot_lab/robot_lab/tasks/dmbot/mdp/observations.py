# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, Imu
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class _SharedImuErrorState:
    """Per-environment IMU calibration errors shared by all IMU observation terms."""

    def __init__(
        self,
        env: ManagerBasedEnv,
        gyro_bias_range: tuple[float, float],
        gyro_drift_std: float,
        gyro_drift_limit: float,
        gravity_bias_range: tuple[float, float],
        gravity_drift_std: float,
        gravity_drift_limit: float,
        mounting_rpy_range: tuple[float, float],
        sensor_name: str,
        delay_steps_range: tuple[int, int],
    ):
        self.env = env
        self.gyro_bias_range = gyro_bias_range
        self.gyro_drift_std = gyro_drift_std
        self.gyro_drift_limit = gyro_drift_limit
        self.gravity_bias_range = gravity_bias_range
        self.gravity_drift_std = gravity_drift_std
        self.gravity_drift_limit = gravity_drift_limit
        self.mounting_rpy_range = mounting_rpy_range
        self.sensor_name = sensor_name

        delay_min, delay_max = delay_steps_range
        if delay_min < 0 or delay_max < delay_min:
            raise ValueError(f"Invalid IMU delay_steps_range: {delay_steps_range}")
        self.delay_steps_range = delay_steps_range

        shape = (env.num_envs, 3)
        self.gyro_bias = torch.zeros(shape, device=env.device)
        self.gyro_drift = torch.zeros(shape, device=env.device)
        self.gravity_bias = torch.zeros(shape, device=env.device)
        self.gravity_drift = torch.zeros(shape, device=env.device)
        self.mounting_rpy = torch.zeros(shape, device=env.device)
        self.last_reset_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.last_drift_step = -1
        self.delay_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.delay_history = torch.zeros((delay_max + 1, env.num_envs, 6), device=env.device)
        self.delay_history_index = -1
        self.delayed_imu = torch.zeros((env.num_envs, 6), device=env.device)
        self.needs_delay_seed = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.last_observation_step = -1
        self._batch_ids = torch.arange(env.num_envs, device=env.device)
        self._resample_delay(self._batch_ids)

    def _resample_delay(self, env_ids: torch.Tensor) -> None:
        delay_min, delay_max = self.delay_steps_range
        self.delay_steps[env_ids] = torch.randint(
            delay_min, delay_max + 1, (env_ids.numel(),), device=self.env.device
        )

    def _resolve_env_ids(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        all_ids = torch.arange(self.env.num_envs, device=self.env.device)
        if env_ids is None:
            return all_ids
        if isinstance(env_ids, slice):
            return all_ids[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        ids = self._resolve_env_ids(env_ids)
        current_step = int(getattr(self.env, "common_step_counter", 0))
        ids = ids[self.last_reset_step[ids] != current_step]
        if ids.numel() == 0:
            return

        sample_shape = (ids.numel(), 3)
        self.gyro_bias[ids] = torch.empty(sample_shape, device=self.env.device).uniform_(*self.gyro_bias_range)
        self.gravity_bias[ids] = torch.empty(sample_shape, device=self.env.device).uniform_(*self.gravity_bias_range)
        self.mounting_rpy[ids] = torch.empty(sample_shape, device=self.env.device).uniform_(
            *self.mounting_rpy_range
        )
        self.gyro_drift[ids] = 0.0
        self.gravity_drift[ids] = 0.0
        self._resample_delay(ids)
        self.needs_delay_seed[ids] = True
        self.last_reset_step[ids] = current_step

    def update_drift(self) -> None:
        current_step = int(getattr(self.env, "common_step_counter", 0))
        if current_step == self.last_drift_step:
            return

        # The training counter jumps when a checkpoint is resumed, but this transient
        # sensor state is intentionally not checkpointed. Advance one observation period.
        sqrt_dt = math.sqrt(float(self.env.step_dt))
        self.gyro_drift.add_(torch.randn_like(self.gyro_drift), alpha=self.gyro_drift_std * sqrt_dt)
        self.gravity_drift.add_(torch.randn_like(self.gravity_drift), alpha=self.gravity_drift_std * sqrt_dt)
        self.gyro_drift.clamp_(-self.gyro_drift_limit, self.gyro_drift_limit)
        self.gravity_drift.clamp_(-self.gravity_drift_limit, self.gravity_drift_limit)
        self.last_drift_step = current_step

    def mounting_quat(self) -> torch.Tensor:
        return math_utils.quat_from_euler_xyz(
            self.mounting_rpy[:, 0], self.mounting_rpy[:, 1], self.mounting_rpy[:, 2]
        )

    def _current_imu(self) -> torch.Tensor:
        self.update_drift()
        sensor: Imu = self.env.scene[self.sensor_name]
        mounting_quat = self.mounting_quat()
        gyro = math_utils.quat_apply_inverse(mounting_quat, sensor.data.ang_vel_b)
        gravity = math_utils.quat_apply_inverse(mounting_quat, sensor.data.projected_gravity_b)
        gyro = gyro + self.gyro_bias + self.gyro_drift
        gravity = gravity + self.gravity_bias + self.gravity_drift
        return torch.cat((gyro, gravity), dim=-1)

    def delayed_observation(self) -> torch.Tensor:
        current_step = int(getattr(self.env, "common_step_counter", 0))
        if current_step != self.last_observation_step:
            current_imu = self._current_imu()
            self.delay_history_index = (self.delay_history_index + 1) % self.delay_history.shape[0]

            seed_ids = self.needs_delay_seed.nonzero(as_tuple=False).squeeze(-1)
            if seed_ids.numel() > 0:
                self.delay_history[:, seed_ids] = current_imu[seed_ids].unsqueeze(0)
                self.needs_delay_seed[seed_ids] = False

            self.delay_history[self.delay_history_index] = current_imu
            read_indices = (self.delay_history_index - self.delay_steps) % self.delay_history.shape[0]
            self.delayed_imu = self.delay_history[read_indices, self._batch_ids].clone()
            self.last_observation_step = current_step
        else:
            # A subset can reset after another observation group was evaluated in
            # the same policy step. Never expose history from the previous episode.
            seed_ids = self.needs_delay_seed.nonzero(as_tuple=False).squeeze(-1)
            if seed_ids.numel() > 0:
                current_imu = self._current_imu()
                self.delay_history[:, seed_ids] = current_imu[seed_ids].unsqueeze(0)
                self.delayed_imu[seed_ids] = current_imu[seed_ids]
                self.needs_delay_seed[seed_ids] = False
        return self.delayed_imu


class imu_with_calibration_error(ManagerTermBase):
    """Return gyro or gravity observations with bias, random-walk drift, and mounting error."""

    _STATE_ATTR = "_togo_lfs_imu_error_state"

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        state = getattr(env, self._STATE_ATTR, None)
        if state is None:
            state = _SharedImuErrorState(
                env=env,
                gyro_bias_range=cfg.params["gyro_bias_range"],
                gyro_drift_std=cfg.params["gyro_drift_std"],
                gyro_drift_limit=cfg.params["gyro_drift_limit"],
                gravity_bias_range=cfg.params["gravity_bias_range"],
                gravity_drift_std=cfg.params["gravity_drift_std"],
                gravity_drift_limit=cfg.params["gravity_drift_limit"],
                mounting_rpy_range=cfg.params["mounting_rpy_range"],
                sensor_name=cfg.params["sensor_name"],
                delay_steps_range=cfg.params["delay_steps_range"],
            )
            setattr(env, self._STATE_ATTR, state)
        self._state: _SharedImuErrorState = state

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        self._state.reset(env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        output: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        gyro_bias_range: tuple[float, float] = (-0.05, 0.05),
        gyro_drift_std: float = 0.002,
        gyro_drift_limit: float = 0.02,
        gravity_bias_range: tuple[float, float] = (-0.01, 0.01),
        gravity_drift_std: float = 0.001,
        gravity_drift_limit: float = 0.02,
        mounting_rpy_range: tuple[float, float] = (-0.034906585, 0.034906585),
        sensor_name: str = "imu",
        delay_steps_range: tuple[int, int] = (0, 1),
    ) -> torch.Tensor:
        del gyro_bias_range, gyro_drift_std, gyro_drift_limit
        del gravity_bias_range, gravity_drift_std, gravity_drift_limit, mounting_rpy_range
        del sensor_name, delay_steps_range, asset_cfg

        delayed_imu = self._state.delayed_observation()
        if output == "angular_velocity":
            return delayed_imu[:, :3]
        if output == "projected_gravity":
            return delayed_imu[:, 3:]
        raise ValueError(f"Unsupported IMU output: {output}")


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

def joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]

def foot_contact_force_norm(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history # [B, T_hist, num_bodies, 3]
    
    contact_force_norm = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1) # [B, T_hist, num_legs]
    max_contact_force_norm, _ = torch.max(contact_force_norm, dim=1)  # [B, num_legs]
    contact_force_norm = torch.concat([max_contact_force_norm.unsqueeze(1), contact_force_norm], dim=1)  # [B, T_hist+1, num_legs]
    
    return contact_force_norm.flatten(start_dim=-2)  # [B, (T_hist+1)*num_legs]
