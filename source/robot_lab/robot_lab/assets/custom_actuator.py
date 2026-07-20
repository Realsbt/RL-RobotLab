"""Custom actuator model combining delay, DC motor torque-speed curve, and first-order low-pass filter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils import configclass


class PhysicalMotor(DelayedPDActuator):
    """Physically-realistic motor model.

    Execution order each physics step:
    1. Delay the incoming position/velocity/effort commands (inherited)
    2. Compute PD torque from delayed commands (inherited)
    3. Clip torque by DC motor torque-speed curve (velocity-dependent saturation)
    4. Apply first-order low-pass filter on output torque

    DaMiao DM-J10010-2EC reference values (48V, output shaft after 10:1 gearbox):
        saturation_effort = 150.0   # stall torque [N·m]
        effort_limit      = 40.0    # continuous torque [N·m]  (150 for peak-only training)
        velocity_limit    = 15.7    # no-load output shaft speed [rad/s]  (150 RPM)
        filter_tau        = 0.005   # ~5 ms, covers CAN + current-loop lag
        physics_dt        = 0.005   # must match sim.dt
    """

    cfg: "PhysicalMotorCfg"

    def __init__(self, cfg: "PhysicalMotorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        if cfg.saturation_effort is None:
            raise ValueError("saturation_effort must be set for PhysicalMotor.")

        self._saturation_effort = cfg.saturation_effort
        filter_tau_range = cfg.filter_tau_range
        if filter_tau_range is None:
            filter_tau_range = (cfg.filter_tau, cfg.filter_tau)
        if filter_tau_range[0] < 0.0 or filter_tau_range[1] < filter_tau_range[0]:
            raise ValueError(f"Invalid filter_tau_range: {filter_tau_range}")
        self._filter_tau_range = filter_tau_range
        self._filter_enabled = filter_tau_range[1] > 0.0

        # corner velocity: where the torque-speed curve crosses effort_limit
        self._vel_at_effort_lim = self.velocity_limit * (1.0 + self.effort_limit / self._saturation_effort)

        # buffer for filtered torque, initialized to zero
        self._filtered_effort = torch.zeros_like(self.computed_effort)

        # One motor response constant is sampled per environment and actuator group.
        self._filter_tau = torch.zeros((self._num_envs, 1), device=self._device)
        self._filter_alpha = torch.ones_like(self._filter_tau)
        self._resample_filter_tau(slice(None))

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self._filtered_effort[env_ids] = 0.0
        self._resample_filter_tau(env_ids)

    def _resample_filter_tau(self, env_ids: Sequence[int] | slice) -> None:
        if env_ids is None or env_ids == slice(None):
            env_ids = slice(None)
            num_envs = self._num_envs
        else:
            num_envs = len(env_ids)

        tau_min, tau_max = self._filter_tau_range
        if tau_min == tau_max:
            sampled_tau = torch.full((num_envs, 1), tau_min, device=self._device)
        else:
            sampled_tau = torch.empty((num_envs, 1), device=self._device).uniform_(tau_min, tau_max)
        self._filter_tau[env_ids] = sampled_tau
        self._filter_alpha[env_ids] = self.cfg.physics_dt / (sampled_tau + self.cfg.physics_dt)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # save joint vel for torque-speed curve clipping
        self._current_joint_vel = joint_vel.clone()

        # run delay + PD computation from parent (sets self.applied_effort)
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # at this point control_action.joint_efforts = clipped PD torque
        torque = control_action.joint_efforts

        # apply first-order low-pass filter if tau > 0
        if self._filter_enabled:
            self._filtered_effort = self._filter_alpha * torque + (1.0 - self._filter_alpha) * self._filtered_effort
            control_action.joint_efforts = self._filtered_effort.clone()

        return control_action

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        """DC motor torque-speed curve: torque drops linearly with velocity."""
        vel = torch.clip(self._current_joint_vel, min=-self._vel_at_effort_lim, max=self._vel_at_effort_lim)

        # four-quadrant linear torque-speed model
        max_effort = torch.clip(
            self._saturation_effort * (1.0 - vel / self.velocity_limit),
            max=self.effort_limit,
        )
        min_effort = torch.clip(
            self._saturation_effort * (-1.0 - vel / self.velocity_limit),
            min=-self.effort_limit,
        )
        return torch.clip(effort, min=min_effort, max=max_effort)


@configclass
class PhysicalMotorCfg(DelayedPDActuatorCfg):
    """Configuration for PhysicalMotor."""

    class_type: type = PhysicalMotor

    saturation_effort: float = MISSING
    """Stall torque at zero velocity (N·m). For DM-J10010-2EC: 150 N·m."""

    filter_tau: float = 0.0
    """Time constant for first-order low-pass filter on output torque (seconds).
    Set to 0.0 to disable. For DM-J10010-2EC: ~0.005 s.
    """

    filter_tau_range: tuple[float, float] | None = None
    """Optional per-episode response time-constant range in seconds.

    When set, this overrides ``filter_tau`` and samples one value per environment and
    actuator group at reset.
    """

    physics_dt: float = 0.005
    """Physics simulation timestep (seconds). Must match sim.dt in your env config."""
