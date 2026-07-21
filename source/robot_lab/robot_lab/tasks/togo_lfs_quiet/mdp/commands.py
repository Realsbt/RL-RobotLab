"""Low-speed velocity command with MUTE impact diagnostics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

from .mute_math import contact_phase, phase_weighted_vertical_velocity

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MuteVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command that logs impact proxies without changing reward."""

    cfg: MuteVelocityCommandCfg

    def __init__(self, cfg: MuteVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.contact_sensor_name]
        self.asset_foot_ids, asset_foot_names = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)
        self.sensor_foot_ids, sensor_foot_names = self.contact_sensor.find_bodies(
            cfg.foot_body_names, preserve_order=True
        )
        if asset_foot_names != sensor_foot_names:
            raise ValueError(
                "Robot and contact sensor resolved different foot orders: "
                f"{asset_foot_names} != {sensor_foot_names}"
            )
        if not 0.0 < cfg.noise_quiet_threshold < cfg.noise_loud_threshold:
            raise ValueError(
                "Noise visualization thresholds must satisfy "
                "0 < noise_quiet_threshold < noise_loud_threshold."
            )
        if cfg.noise_max_visualized_speed <= 0.0 or cfg.noise_marker_decay_time <= 0.0:
            raise ValueError("Noise visualization speed and decay time must be positive.")

        num_feet = len(self.asset_foot_ids)
        self.previous_foot_velocity_z = torch.zeros(self.num_envs, num_feet, device=self.device)
        self.noise_proxy_speed = torch.zeros(self.num_envs, num_feet, device=self.device)
        self.touchdown_velocity_sum = torch.zeros(self.num_envs, device=self.device)
        self.touchdown_count = torch.zeros(self.num_envs, device=self.device)
        self.metrics["touchdown_velocity"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["phase_weighted_drop_velocity"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["peak_foot_contact_force"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["contact_foot_slip_speed"] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        self.previous_foot_velocity_z[env_ids] = 0.0
        self.noise_proxy_speed[env_ids] = 0.0
        self.touchdown_velocity_sum[env_ids] = 0.0
        self.touchdown_count[env_ids] = 0.0
        return super().reset(env_ids)

    def _update_metrics(self):
        super()._update_metrics()

        foot_velocity_w = self.robot.data.body_lin_vel_w[:, self.asset_foot_ids]
        foot_velocity_z = foot_velocity_w[..., 2]
        first_contact = self.contact_sensor.compute_first_contact(self._env.step_dt)[:, self.sensor_foot_ids]
        touchdown_velocity = torch.clamp(-self.previous_foot_velocity_z, min=0.0) * first_contact
        self.touchdown_velocity_sum += torch.sum(touchdown_velocity, dim=-1)
        self.touchdown_count += torch.sum(first_contact, dim=-1)
        self.metrics["touchdown_velocity"][:] = torch.where(
            self.touchdown_count > 0.0,
            self.touchdown_velocity_sum / torch.clamp(self.touchdown_count, min=1.0),
            0.0,
        )

        phase = contact_phase(
            self.contact_sensor.data.current_air_time[:, self.sensor_foot_ids],
            self.contact_sensor.data.current_contact_time[:, self.sensor_foot_ids],
            self.cfg.swing_duration,
            self.cfg.stance_duration,
        )
        weighted_drop_speed = torch.exp(0.5 * phase) * torch.clamp(-foot_velocity_z, min=0.0)
        touchdown_proxy_speed = torch.clamp(-self.previous_foot_velocity_z, min=0.0) * first_contact
        instantaneous_proxy_speed = torch.maximum(weighted_drop_speed, touchdown_proxy_speed)
        decay = math.exp(-self._env.step_dt / self.cfg.noise_marker_decay_time)
        self.noise_proxy_speed[:] = torch.maximum(
            instantaneous_proxy_speed, self.noise_proxy_speed * decay
        )

        drop_term, _ = phase_weighted_vertical_velocity(foot_velocity_z, phase)
        max_command_steps = self.cfg.resampling_time_range[1] / self._env.step_dt
        self.metrics["phase_weighted_drop_velocity"] += drop_term / len(self.asset_foot_ids) / max_command_steps

        force_history = self.contact_sensor.data.net_forces_w_history[:, :, self.sensor_foot_ids]
        peak_force = torch.linalg.norm(force_history, dim=-1).amax(dim=(1, 2))
        self.metrics["peak_foot_contact_force"][:] = torch.maximum(
            self.metrics["peak_foot_contact_force"], peak_force
        )

        contacts = torch.linalg.norm(self.contact_sensor.data.net_forces_w[:, self.sensor_foot_ids], dim=-1) > 1.0
        slip_speed = torch.linalg.norm(foot_velocity_w[..., :2], dim=-1)
        mean_contact_slip = torch.sum(slip_speed * contacts, dim=-1) / torch.clamp(
            torch.sum(contacts, dim=-1), min=1
        )
        self.metrics["contact_foot_slip_speed"] += mean_contact_slip / max_command_steps
        self.previous_foot_velocity_z.copy_(foot_velocity_z)

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "noise_proxy_visualizer"):
                self.noise_proxy_visualizer = VisualizationMarkers(self.cfg.noise_marker_cfg)
            self.noise_proxy_visualizer.set_visibility(True)
        elif hasattr(self, "noise_proxy_visualizer"):
            self.noise_proxy_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if not self.robot.is_initialized:
            return

        marker_positions = self.robot.data.body_pos_w[:, self.asset_foot_ids].clone()
        marker_positions[..., 2] += self.cfg.noise_marker_height

        marker_indices = torch.zeros_like(self.noise_proxy_speed, dtype=torch.long)
        marker_indices[self.noise_proxy_speed >= self.cfg.noise_quiet_threshold] = 1
        marker_indices[self.noise_proxy_speed >= self.cfg.noise_loud_threshold] = 2

        normalized_speed = torch.clamp(
            self.noise_proxy_speed / self.cfg.noise_max_visualized_speed, min=0.0, max=1.0
        )
        marker_scale = 0.7 + 1.3 * normalized_speed
        marker_scales = marker_scale.unsqueeze(-1).repeat(1, 1, 3)
        self.noise_proxy_visualizer.visualize(
            translations=marker_positions.reshape(-1, 3),
            scales=marker_scales.reshape(-1, 3),
            marker_indices=marker_indices.reshape(-1),
        )


@configclass
class MuteVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for low-speed commands and MUTE diagnostics."""

    class_type: type = MuteVelocityCommand
    contact_sensor_name: str = "contact_forces"
    foot_body_names: str = ".*_foot"
    swing_duration: float = 0.35
    stance_duration: float = 0.35
    noise_quiet_threshold: float = 0.15
    """Green-to-amber threshold for phase-weighted downward foot speed [m/s]."""
    noise_loud_threshold: float = 0.35
    """Amber-to-red threshold for phase-weighted downward foot speed [m/s]."""
    noise_max_visualized_speed: float = 0.6
    """Proxy speed that produces the maximum marker size [m/s]."""
    noise_marker_decay_time: float = 0.2
    """Time constant used to keep touchdown peaks visible [s]."""
    noise_marker_height: float = 0.06
    """Vertical marker offset above each foot [m]."""
    noise_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/MUTE/noise_proxy",
        markers={
            "quiet": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2)),
            ),
            "medium": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.65, 0.0)),
            ),
            "loud": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.05, 0.05)),
            ),
        },
    )
    """Green, amber, and red foot markers for the MUTE acoustic proxy."""
