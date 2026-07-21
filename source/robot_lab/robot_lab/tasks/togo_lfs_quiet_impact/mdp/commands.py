"""Low-speed command term with event-based touchdown diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass

from robot_lab.tasks.togo_lfs_quiet.mdp.commands import MuteVelocityCommand, MuteVelocityCommandCfg

from .impact_math import contact_hysteresis_step, normalized_impact_score
from .impact_sensor import FootImpactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ImpactEventVelocityCommand(MuteVelocityCommand):
    """Track each touchdown as a short event without changing the training reward."""

    cfg: ImpactEventVelocityCommandCfg

    def __init__(self, cfg: ImpactEventVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.impact_sensor: FootImpactSensor = env.scene.sensors[cfg.impact_sensor_name]
        sensor_foot_ids, sensor_foot_names = self.impact_sensor.find_bodies(
            cfg.foot_body_names, preserve_order=True
        )
        _, asset_foot_names = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)
        if asset_foot_names != sensor_foot_names:
            raise ValueError(
                "Robot and impact sensor resolved different foot orders: "
                f"{asset_foot_names} != {sensor_foot_names}"
            )
        self.impact_sensor_foot_ids = sensor_foot_ids

        sensor_dt = self.impact_sensor.cfg.update_period or env.physics_dt
        self.physics_samples_per_step = round(env.step_dt / sensor_dt)
        self.impact_window_samples = round(cfg.impact_window_s / sensor_dt)
        if self.physics_samples_per_step < 1 or self.impact_window_samples < 1:
            raise ValueError("Impact sensor and event window must contain at least one sample.")
        if self.impact_sensor.cfg.history_length <= self.physics_samples_per_step:
            raise ValueError(
                "Impact sensor history must include the current policy interval plus one previous sample."
            )

        shape = (self.num_envs, len(self.asset_foot_ids))
        self.contact_state = torch.zeros(shape, dtype=torch.bool, device=self.device)
        self.window_samples_left = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.event_preimpact_speed = torch.zeros(shape, device=self.device)
        self.event_peak_force = torch.zeros(shape, device=self.device)
        self.event_peak_force_rise_rate = torch.zeros(shape, device=self.device)
        self.event_impulse = torch.zeros(shape, device=self.device)
        self.event_force_variation_energy = torch.zeros(shape, device=self.device)
        self.event_display_score = torch.zeros(shape, device=self.device)
        self.event_display_time = torch.zeros(shape, device=self.device)

        self.completed_event_count = torch.zeros(self.num_envs, device=self.device)
        self.preimpact_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self.peak_force_sum = torch.zeros(self.num_envs, device=self.device)
        self.force_rise_rate_sum = torch.zeros(self.num_envs, device=self.device)
        self.impulse_sum = torch.zeros(self.num_envs, device=self.device)
        self.force_variation_energy_sum = torch.zeros(self.num_envs, device=self.device)
        self.event_score_sum = torch.zeros(self.num_envs, device=self.device)
        self.slip_power_sum = torch.zeros(self.num_envs, device=self.device)
        self.contact_sample_count = torch.zeros(self.num_envs, device=self.device)

        for metric_name in (
            "impact_event_count",
            "event_preimpact_speed",
            "event_peak_force",
            "event_peak_force_rise_rate",
            "event_impulse",
            "event_force_variation_energy",
            "event_impact_score",
            "peak_event_impact_score",
            "contact_slip_power_proxy",
        ):
            self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        resolved_env_ids = slice(None) if env_ids is None else env_ids
        for buffer in (
            self.contact_state,
            self.window_samples_left,
            self.event_preimpact_speed,
            self.event_peak_force,
            self.event_peak_force_rise_rate,
            self.event_impulse,
            self.event_force_variation_energy,
            self.event_display_score,
            self.event_display_time,
            self.completed_event_count,
            self.preimpact_speed_sum,
            self.peak_force_sum,
            self.force_rise_rate_sum,
            self.impulse_sum,
            self.force_variation_energy_sum,
            self.event_score_sum,
            self.slip_power_sum,
            self.contact_sample_count,
        ):
            buffer[resolved_env_ids] = 0
        return super().reset(env_ids)

    def _update_metrics(self):
        super()._update_metrics()
        self._process_impact_samples()
        self.event_display_time.sub_(self._env.step_dt).clamp_(min=0.0)

    def clear_impact_statistics(self, env_ids: Sequence[int] | None = None):
        """Clear accumulated benchmark statistics without changing contact state."""
        resolved_env_ids = slice(None) if env_ids is None else env_ids
        for buffer in (
            self.completed_event_count,
            self.preimpact_speed_sum,
            self.peak_force_sum,
            self.force_rise_rate_sum,
            self.impulse_sum,
            self.force_variation_energy_sum,
            self.event_score_sum,
            self.slip_power_sum,
            self.contact_sample_count,
        ):
            buffer[resolved_env_ids] = 0.0
        self.window_samples_left[resolved_env_ids] = 0
        self.event_display_time[resolved_env_ids] = 0.0
        for metric_name in (
            "impact_event_count",
            "event_preimpact_speed",
            "event_peak_force",
            "event_peak_force_rise_rate",
            "event_impulse",
            "event_force_variation_energy",
            "event_impact_score",
            "peak_event_impact_score",
            "contact_slip_power_proxy",
        ):
            self.metrics[metric_name][resolved_env_ids] = 0.0

    def get_impact_statistics(self) -> dict[str, float]:
        """Return event-weighted aggregate statistics across all environments."""
        event_count = torch.sum(self.completed_event_count)
        denominator = torch.clamp(event_count, min=1.0)
        contact_samples = torch.sum(self.contact_sample_count)
        return {
            "event_count": event_count.item(),
            "preimpact_speed_m_s": (torch.sum(self.preimpact_speed_sum) / denominator).item(),
            "peak_force_n": (torch.sum(self.peak_force_sum) / denominator).item(),
            "peak_force_rise_rate_n_s": (
                torch.sum(self.force_rise_rate_sum) / denominator
            ).item(),
            "impulse_n_s": (torch.sum(self.impulse_sum) / denominator).item(),
            "force_variation_energy_n2_s": (
                torch.sum(self.force_variation_energy_sum) / denominator
            ).item(),
            "impact_score": (torch.sum(self.event_score_sum) / denominator).item(),
            "peak_impact_score": torch.max(self.metrics["peak_event_impact_score"]).item(),
            "slip_power_proxy_w": torch.where(
                contact_samples > 0,
                torch.sum(self.slip_power_sum) / torch.clamp(contact_samples, min=1.0),
                0.0,
            ).item(),
        }

    def _process_impact_samples(self):
        sensor_data = self.impact_sensor.data
        force_history = torch.linalg.norm(
            sensor_data.net_forces_w_history[:, :, self.impact_sensor_foot_ids], dim=-1
        )
        velocity_history = sensor_data.lin_vel_w_history[:, :, self.impact_sensor_foot_ids]
        sensor_dt = self.impact_sensor.cfg.update_period or self._env.physics_dt

        previous_force = force_history[:, self.physics_samples_per_step]
        previous_velocity = velocity_history[:, self.physics_samples_per_step]
        for history_index in range(self.physics_samples_per_step - 1, -1, -1):
            force = force_history[:, history_index]
            velocity = velocity_history[:, history_index]
            entered, _, contact_state = contact_hysteresis_step(
                force,
                self.contact_state,
                self.cfg.contact_on_force,
                self.cfg.contact_off_force,
            )

            self.window_samples_left[:] = torch.where(
                entered,
                self.impact_window_samples,
                self.window_samples_left,
            )
            self.event_preimpact_speed[:] = torch.where(
                entered,
                torch.clamp(-previous_velocity[..., 2], min=0.0),
                self.event_preimpact_speed,
            )
            for event_buffer in (
                self.event_peak_force,
                self.event_peak_force_rise_rate,
                self.event_impulse,
                self.event_force_variation_energy,
            ):
                event_buffer[:] = torch.where(entered, 0.0, event_buffer)
            self.event_display_time[:] = torch.where(
                entered,
                self.cfg.event_marker_hold_time,
                self.event_display_time,
            )
            self.metrics["impact_event_count"] += torch.sum(entered, dim=-1)

            active = self.window_samples_left > 0
            force_delta = force - previous_force
            force_rise_rate = torch.clamp(force_delta / sensor_dt, min=0.0)
            self.event_peak_force[:] = torch.where(
                active, torch.maximum(self.event_peak_force, force), self.event_peak_force
            )
            self.event_peak_force_rise_rate[:] = torch.where(
                active,
                torch.maximum(self.event_peak_force_rise_rate, force_rise_rate),
                self.event_peak_force_rise_rate,
            )
            self.event_impulse += force * sensor_dt * active
            self.event_force_variation_energy += torch.square(force_delta) * sensor_dt * active

            current_score = self._current_impact_score()
            self.event_display_score[:] = torch.where(active, current_score, self.event_display_score)

            horizontal_speed = torch.linalg.norm(velocity[..., :2], dim=-1)
            slip_power = force * horizontal_speed * contact_state
            self.slip_power_sum += torch.sum(slip_power, dim=-1)
            self.contact_sample_count += torch.sum(contact_state, dim=-1)

            finishing = active & (self.window_samples_left == 1)
            self.window_samples_left[:] = torch.where(
                active, self.window_samples_left - 1, self.window_samples_left
            )
            self._finalize_events(finishing, current_score)

            self.contact_state[:] = contact_state
            previous_force = force
            previous_velocity = velocity

        self.metrics["contact_slip_power_proxy"][:] = torch.where(
            self.contact_sample_count > 0,
            self.slip_power_sum / torch.clamp(self.contact_sample_count, min=1.0),
            0.0,
        )

    def _current_impact_score(self) -> torch.Tensor:
        return normalized_impact_score(
            self.event_preimpact_speed,
            self.event_peak_force,
            self.event_peak_force_rise_rate,
            self.event_impulse,
            references=(
                self.cfg.preimpact_speed_reference,
                self.cfg.peak_force_reference,
                self.cfg.force_rise_rate_reference,
                self.cfg.impulse_reference,
            ),
            weights=self.cfg.impact_score_weights,
        )

    def _finalize_events(self, finishing: torch.Tensor, current_score: torch.Tensor):
        event_count = torch.sum(finishing, dim=-1)
        self.completed_event_count += event_count
        self.preimpact_speed_sum += torch.sum(self.event_preimpact_speed * finishing, dim=-1)
        self.peak_force_sum += torch.sum(self.event_peak_force * finishing, dim=-1)
        self.force_rise_rate_sum += torch.sum(self.event_peak_force_rise_rate * finishing, dim=-1)
        self.impulse_sum += torch.sum(self.event_impulse * finishing, dim=-1)
        self.force_variation_energy_sum += torch.sum(
            self.event_force_variation_energy * finishing, dim=-1
        )
        self.event_score_sum += torch.sum(current_score * finishing, dim=-1)
        self.metrics["peak_event_impact_score"][:] = torch.maximum(
            self.metrics["peak_event_impact_score"],
            torch.amax(current_score * finishing, dim=-1),
        )

        denominator = torch.clamp(self.completed_event_count, min=1.0)
        completed = self.completed_event_count > 0
        for metric_name, value_sum in (
            ("event_preimpact_speed", self.preimpact_speed_sum),
            ("event_peak_force", self.peak_force_sum),
            ("event_peak_force_rise_rate", self.force_rise_rate_sum),
            ("event_impulse", self.impulse_sum),
            ("event_force_variation_energy", self.force_variation_energy_sum),
            ("event_impact_score", self.event_score_sum),
        ):
            self.metrics[metric_name][:] = torch.where(completed, value_sum / denominator, 0.0)

    def _set_debug_vis_impl(self, debug_vis: bool):
        UniformVelocityCommand._set_debug_vis_impl(self, debug_vis)
        if debug_vis:
            if not hasattr(self, "impact_event_visualizer"):
                self.impact_event_visualizer = VisualizationMarkers(self.cfg.impact_event_marker_cfg)
            self.impact_event_visualizer.set_visibility(True)
        elif hasattr(self, "impact_event_visualizer"):
            self.impact_event_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        UniformVelocityCommand._debug_vis_callback(self, event)
        if not self.robot.is_initialized:
            return

        marker_positions = self.robot.data.body_pos_w[:, self.asset_foot_ids].clone()
        marker_positions[..., 2] += self.cfg.event_marker_height
        active = self.event_display_time > 0.0
        marker_indices = torch.zeros_like(self.event_display_score, dtype=torch.long)
        marker_indices[active & (self.event_display_score < self.cfg.event_quiet_score)] = 1
        marker_indices[
            active
            & (self.event_display_score >= self.cfg.event_quiet_score)
            & (self.event_display_score < self.cfg.event_loud_score)
        ] = 2
        marker_indices[active & (self.event_display_score >= self.cfg.event_loud_score)] = 3

        normalized_score = torch.clamp(
            self.event_display_score / self.cfg.event_loud_score, min=0.0, max=2.0
        )
        marker_scale = 0.7 + 0.65 * normalized_score
        marker_scales = marker_scale.unsqueeze(-1).repeat(1, 1, 3)
        self.impact_event_visualizer.visualize(
            translations=marker_positions.reshape(-1, 3),
            scales=marker_scales.reshape(-1, 3),
            marker_indices=marker_indices.reshape(-1),
        )


@configclass
class ImpactEventVelocityCommandCfg(MuteVelocityCommandCfg):
    """Configuration for event-only touchdown diagnostics."""

    class_type: type = ImpactEventVelocityCommand
    impact_sensor_name: str = "quiet_impact_sensor"
    contact_on_force: float = 1.0
    contact_off_force: float = 0.5
    impact_window_s: float = 0.05
    event_marker_hold_time: float = 0.3
    event_marker_height: float = 0.06

    preimpact_speed_reference: float = 0.2
    peak_force_reference: float = 500.0
    force_rise_rate_reference: float = 100000.0
    impulse_reference: float = 10.0
    impact_score_weights: tuple[float, float, float, float] = (0.4, 0.2, 0.3, 0.1)
    event_quiet_score: float = 0.5
    event_loud_score: float = 1.5

    impact_event_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/MUTE/impact_events",
        markers={
            "inactive": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
                visible=False,
            ),
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

    def __post_init__(self):
        if self.contact_on_force <= self.contact_off_force or self.contact_off_force < 0.0:
            raise ValueError("Contact thresholds must satisfy contact_on_force > contact_off_force >= 0.")
        if self.impact_window_s <= 0.0 or self.event_marker_hold_time <= 0.0:
            raise ValueError("Impact window and marker hold time must be positive.")
        if not 0.0 < self.event_quiet_score < self.event_loud_score:
            raise ValueError("Event score thresholds must satisfy 0 < quiet < loud.")
