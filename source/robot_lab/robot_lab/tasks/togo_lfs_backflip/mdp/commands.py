"""Phase command and episode-state tracking for the LFS backflip task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from .backflip_math import (
    multiscale_phase_features,
    positive_rotation_increment,
    unwrapped_projected_gravity_pitch_increment,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BackflipPhaseCommand(CommandTerm):
    """Provide a phase clock and track takeoff, rotation, and touchdown state.

    The six-dimensional command is deterministic and deployable: it depends
    only on time since the skill was triggered. Extra state kept by this term
    is used by rewards and episode diagnostics, not by the deployed policy.
    """

    cfg: "BackflipPhaseCommandCfg"

    def __init__(self, cfg: "BackflipPhaseCommandCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.contact_sensor_name]

        self.asset_foot_ids, asset_foot_names = self.robot.find_bodies(
            cfg.foot_body_names, preserve_order=True
        )
        self.sensor_foot_ids, sensor_foot_names = self.contact_sensor.find_bodies(
            cfg.foot_body_names, preserve_order=True
        )
        if asset_foot_names != sensor_foot_names:
            raise ValueError(
                "Robot and contact sensor resolved different foot orders: "
                f"{asset_foot_names} != {sensor_foot_names}"
            )
        if len(self.asset_foot_ids) != 4:
            raise ValueError(f"Backflip task requires four feet, resolved {asset_foot_names}.")
        all_asset_ids, _ = self.robot.find_bodies(".*", preserve_order=True)
        asset_foot_id_set = set(self.asset_foot_ids)
        self.asset_nonfoot_ids = [
            body_id for body_id in all_asset_ids if body_id not in asset_foot_id_set
        ]
        if not self.asset_nonfoot_ids:
            raise ValueError("Backflip task requires non-foot articulation bodies.")
        self.knee_joint_ids, knee_joint_names = self.robot.find_joints(
            ".*_knee", preserve_order=True
        )
        self.hip_pitch_joint_ids, hip_pitch_joint_names = self.robot.find_joints(
            ".*_hipp", preserve_order=True
        )
        if len(self.knee_joint_ids) != 4 or len(self.hip_pitch_joint_ids) != 4:
            raise ValueError(
                "Backflip diagnostics require four knee and four hip-pitch joints, "
                f"resolved {knee_joint_names} and {hip_pitch_joint_names}."
            )
        all_sensor_ids, _ = self.contact_sensor.find_bodies(".*", preserve_order=True)
        foot_id_set = set(self.sensor_foot_ids)
        self.sensor_nonfoot_ids = [
            body_id for body_id in all_sensor_ids if body_id not in foot_id_set
        ]
        if not self.sensor_nonfoot_ids:
            raise ValueError("Backflip task requires contact sensing on non-foot bodies.")

        self._command = torch.zeros(self.num_envs, 6, device=self.device)
        self.contact = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.nonfoot_contact = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.airborne = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.was_airborne = torch.zeros_like(self.airborne)
        self.has_support = torch.zeros_like(self.airborne)
        self.has_taken_off = torch.zeros_like(self.airborne)
        self.has_touched_down = torch.zeros_like(self.airborne)
        self.has_landed = torch.zeros_like(self.airborne)
        self.landing_success = torch.zeros_like(self.airborne)
        self.has_reached_apex = torch.zeros_like(self.airborne)
        self.has_invalid_rotation_axis = torch.zeros_like(self.airborne)
        self.signed_backward_rotation = torch.zeros(self.num_envs, device=self.device)
        self.max_backward_rotation = torch.zeros(self.num_envs, device=self.device)
        self.rotation_increment = torch.zeros(self.num_envs, device=self.device)
        self.wrapped_pitch_angle = torch.zeros(self.num_envs, device=self.device)
        self.max_base_height = torch.full(
            (self.num_envs,), cfg.initial_base_height, device=self.device
        )
        self.takeoff_time_s = torch.zeros(self.num_envs, device=self.device)
        self.apex_time_s = torch.zeros(self.num_envs, device=self.device)
        self.backward_rotation_at_apex = torch.zeros(
            self.num_envs, device=self.device
        )
        self.first_touchdown_time_s = torch.zeros(self.num_envs, device=self.device)
        self.flight_time_s = torch.zeros(self.num_envs, device=self.device)
        self.touchdown_upright_cosine = torch.zeros(self.num_envs, device=self.device)
        self.touchdown_vertical_speed = torch.zeros(self.num_envs, device=self.device)
        self.touchdown_linear_speed = torch.zeros(self.num_envs, device=self.device)
        self.touchdown_angular_speed = torch.zeros(self.num_envs, device=self.device)
        self.touchdown_contact_count = torch.zeros(self.num_envs, device=self.device)
        self.touchdown_nonfoot_contact = torch.zeros(
            self.num_envs, device=self.device
        )
        self.signed_backward_rotation_at_touchdown = torch.zeros(
            self.num_envs, device=self.device
        )
        self.aerial_backward_rotation_at_touchdown = torch.zeros(
            self.num_envs, device=self.device
        )
        self.backward_pitch_rate_at_takeoff = torch.zeros(
            self.num_envs, device=self.device
        )
        self.backward_rotation_at_takeoff = torch.zeros(
            self.num_envs, device=self.device
        )
        self.stable_landing_time_s = torch.zeros(
            self.num_envs, device=self.device
        )
        self.touchdown_mean_foot_drop_body_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self.touchdown_mean_foot_drop_world_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self.touchdown_joint_pose_error_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self.touchdown_knee_pose_error_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self.touchdown_hip_pitch_pose_error_rad = torch.zeros(
            self.num_envs, device=self.device
        )
        self.touchdown_first_foot_origin_margin_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self.touchdown_second_foot_origin_margin_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self.touchdown_third_foot_origin_margin_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self.touchdown_fourth_foot_origin_margin_m = torch.zeros(
            self.num_envs, device=self.device
        )
        self.maximum_post_touchdown_contact_count = torch.zeros(
            self.num_envs, device=self.device
        )
        self.clean_support_time_s = torch.zeros(self.num_envs, device=self.device)
        self.maximum_clean_support_time_s = torch.zeros(
            self.num_envs, device=self.device
        )
        self.touchdown_foot_origin_margin_by_leg_m = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.touchdown_foot_forward_by_leg_m = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.post_touchdown_foot_forward_by_leg_m = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self.touchdown_action_l2_by_leg = torch.zeros(
            self.num_envs, 4, device=self.device
        )

        for metric_name in (
            "takeoff_success",
            "rotation_success",
            "rotation_axis_success",
            "landing_rotation_success",
            "touchdown_success",
            "landing_success",
            "max_base_height_m",
            "max_backward_rotation_rad",
            "takeoff_time_s",
            "apex_time_s",
            "backward_rotation_at_apex_rad",
            "first_touchdown_time_s",
            "flight_time_s",
            "touchdown_upright_cosine",
            "touchdown_vertical_speed_m_s",
            "touchdown_linear_speed_m_s",
            "touchdown_angular_speed_rad_s",
            "touchdown_contact_count",
            "touchdown_nonfoot_contact",
            "signed_backward_rotation_at_touchdown_rad",
            "aerial_backward_rotation_at_touchdown_rad",
            "backward_pitch_rate_at_takeoff_rad_s",
            "backward_rotation_at_takeoff_rad",
            "stable_landing_time_s",
            "touchdown_mean_foot_drop_body_m",
            "touchdown_mean_foot_drop_world_m",
            "touchdown_joint_pose_error_rad",
            "touchdown_knee_pose_error_rad",
            "touchdown_hip_pitch_pose_error_rad",
            "touchdown_first_foot_origin_margin_m",
            "touchdown_second_foot_origin_margin_m",
            "touchdown_third_foot_origin_margin_m",
            "touchdown_fourth_foot_origin_margin_m",
            "maximum_post_touchdown_contact_count",
            "maximum_clean_support_time_s",
            "touchdown_fl_foot_origin_margin_m",
            "touchdown_fr_foot_origin_margin_m",
            "touchdown_rl_foot_origin_margin_m",
            "touchdown_rr_foot_origin_margin_m",
            "touchdown_fl_foot_forward_m",
            "touchdown_fr_foot_forward_m",
            "touchdown_rl_foot_forward_m",
            "touchdown_rr_foot_forward_m",
            "post_touchdown_fl_foot_forward_m",
            "post_touchdown_fr_foot_forward_m",
            "post_touchdown_rl_foot_forward_m",
            "post_touchdown_rr_foot_forward_m",
            "touchdown_fl_action_l2",
            "touchdown_fr_action_l2",
            "touchdown_rl_action_l2",
            "touchdown_rr_action_l2",
        ):
            self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)

        self._write_phase_command()

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def contact_count(self) -> torch.Tensor:
        return torch.sum(self.contact, dim=-1)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None or isinstance(env_ids, slice):
            resolved_env_ids = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            resolved_env_ids = env_ids.to(device=self.device, dtype=torch.long)
        else:
            resolved_env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        extras = super().reset(resolved_env_ids)
        for buffer in (
            self.contact,
            self.nonfoot_contact,
            self.airborne,
            self.was_airborne,
            self.has_support,
            self.has_taken_off,
            self.has_touched_down,
            self.has_landed,
            self.landing_success,
            self.has_reached_apex,
            self.has_invalid_rotation_axis,
            self.signed_backward_rotation,
            self.max_backward_rotation,
            self.rotation_increment,
            self.wrapped_pitch_angle,
            self.takeoff_time_s,
            self.apex_time_s,
            self.backward_rotation_at_apex,
            self.first_touchdown_time_s,
            self.flight_time_s,
            self.touchdown_upright_cosine,
            self.touchdown_vertical_speed,
            self.touchdown_linear_speed,
            self.touchdown_angular_speed,
            self.touchdown_contact_count,
            self.touchdown_nonfoot_contact,
            self.signed_backward_rotation_at_touchdown,
            self.aerial_backward_rotation_at_touchdown,
            self.backward_pitch_rate_at_takeoff,
            self.backward_rotation_at_takeoff,
            self.stable_landing_time_s,
            self.touchdown_mean_foot_drop_body_m,
            self.touchdown_mean_foot_drop_world_m,
            self.touchdown_joint_pose_error_rad,
            self.touchdown_knee_pose_error_rad,
            self.touchdown_hip_pitch_pose_error_rad,
            self.touchdown_first_foot_origin_margin_m,
            self.touchdown_second_foot_origin_margin_m,
            self.touchdown_third_foot_origin_margin_m,
            self.touchdown_fourth_foot_origin_margin_m,
            self.maximum_post_touchdown_contact_count,
            self.clean_support_time_s,
            self.maximum_clean_support_time_s,
            self.touchdown_foot_origin_margin_by_leg_m,
            self.touchdown_foot_forward_by_leg_m,
            self.post_touchdown_foot_forward_by_leg_m,
            self.touchdown_action_l2_by_leg,
        ):
            buffer[resolved_env_ids] = 0
        self.max_base_height[resolved_env_ids] = self.cfg.initial_base_height
        self._command[resolved_env_ids] = 0.0
        self._command[resolved_env_ids, 1] = 1.0
        self._command[resolved_env_ids, 3] = 1.0
        self._command[resolved_env_ids, 5] = 1.0
        return extras

    def _update_metrics(self):
        elapsed_s = self._env.episode_length_buf * self._env.step_dt
        net_force = self.contact_sensor.data.net_forces_w
        foot_force_norm = torch.linalg.norm(
            net_force[:, self.sensor_foot_ids], dim=-1
        )
        nonfoot_force_norm = torch.linalg.norm(
            net_force[:, self.sensor_nonfoot_ids], dim=-1
        )
        self.contact[:] = foot_force_norm >= self.cfg.contact_force_threshold
        self.nonfoot_contact[:] = torch.any(
            nonfoot_force_norm >= self.cfg.contact_force_threshold, dim=-1
        )
        any_contact = torch.any(self.contact, dim=-1)
        any_robot_contact = any_contact | self.nonfoot_contact
        self.airborne[:] = ~any_robot_contact

        base_height = self.robot.data.root_pos_w[:, 2]
        upward_speed = self.robot.data.root_lin_vel_w[:, 2]
        self.has_support |= self.contact_count >= self.cfg.minimum_support_contact_count

        new_takeoff = (
            self.airborne
            & self.has_support
            & (elapsed_s >= self.cfg.minimum_takeoff_time_s)
            & (base_height >= self.cfg.initial_base_height + self.cfg.minimum_takeoff_height_gain)
            & (upward_speed >= self.cfg.minimum_takeoff_vertical_speed)
            & ~self.has_taken_off
        )
        backward_pitch_rate = (
            self.cfg.backward_pitch_sign * self.robot.data.root_ang_vel_b[:, 1]
        )
        self.takeoff_time_s[:] = torch.where(
            new_takeoff, elapsed_s, self.takeoff_time_s
        )
        self.backward_pitch_rate_at_takeoff[:] = torch.where(
            new_takeoff, backward_pitch_rate, self.backward_pitch_rate_at_takeoff
        )
        self.has_taken_off |= new_takeoff

        # A backflip is the winding of body-frame gravity through upright,
        # inverted, and upright again. This is stricter than integrating one
        # angular-velocity component, which can be exploited by 3-D precession.
        wrapped_pitch, pitch_increment = (
            unwrapped_projected_gravity_pitch_increment(
                self.robot.data.projected_gravity_b,
                self.wrapped_pitch_angle,
                self.cfg.backward_pitch_sign,
                self.cfg.minimum_pitch_xz_norm,
            )
        )
        pitch_xz_norm = torch.linalg.vector_norm(
            self.robot.data.projected_gravity_b[:, (0, 2)], dim=-1
        )
        rotation_window_open = (
            self.has_support
            & (elapsed_s >= self.cfg.minimum_takeoff_time_s)
        )
        invalid_axis_now = (
            rotation_window_open
            & ~self.has_touched_down
            & (pitch_xz_norm < self.cfg.minimum_pitch_xz_norm)
        )
        self.has_invalid_rotation_axis |= invalid_axis_now
        # Count the supported launch from 0.20 s, then continue through flight.
        # First post-takeoff contact by any body is the hard aerial boundary.
        rotation_active = (
            rotation_window_open
            & ~self.has_touched_down
            & ~self.has_invalid_rotation_axis
        )
        self.signed_backward_rotation += pitch_increment * rotation_active
        self.wrapped_pitch_angle.copy_(wrapped_pitch)
        self.backward_rotation_at_takeoff[:] = torch.where(
            new_takeoff,
            self.signed_backward_rotation,
            self.backward_rotation_at_takeoff,
        )
        self.max_backward_rotation[:], self.rotation_increment[:] = positive_rotation_increment(
            self.signed_backward_rotation, self.max_backward_rotation
        )

        # Capture the first vertical-velocity zero crossing after takeoff.  The
        # resulting apex rotation is a diagnostic and reward target; it is not
        # exposed to the deployable actor observation.
        new_apex = (
            self.has_taken_off
            & self.airborne
            & ~self.has_reached_apex
            & (upward_speed <= 0.0)
        )
        self.apex_time_s[:] = torch.where(
            new_apex, elapsed_s, self.apex_time_s
        )
        self.backward_rotation_at_apex[:] = torch.where(
            new_apex, self.max_backward_rotation, self.backward_rotation_at_apex
        )
        self.has_reached_apex |= new_apex

        touchdown_now = self.has_taken_off & self.was_airborne & any_robot_contact
        first_touchdown_now = touchdown_now & ~self.has_touched_down
        self.first_touchdown_time_s[:] = torch.where(
            first_touchdown_now, elapsed_s, self.first_touchdown_time_s
        )
        self.flight_time_s[:] = torch.where(
            first_touchdown_now,
            elapsed_s - self.takeoff_time_s,
            self.flight_time_s,
        )
        self.touchdown_upright_cosine[:] = torch.where(
            first_touchdown_now,
            -self.robot.data.projected_gravity_b[:, 2],
            self.touchdown_upright_cosine,
        )
        self.touchdown_vertical_speed[:] = torch.where(
            first_touchdown_now,
            self.robot.data.root_lin_vel_w[:, 2],
            self.touchdown_vertical_speed,
        )
        self.touchdown_linear_speed[:] = torch.where(
            first_touchdown_now,
            torch.linalg.norm(self.robot.data.root_lin_vel_w, dim=-1),
            self.touchdown_linear_speed,
        )
        self.touchdown_angular_speed[:] = torch.where(
            first_touchdown_now,
            torch.linalg.norm(self.robot.data.root_ang_vel_w, dim=-1),
            self.touchdown_angular_speed,
        )
        self.touchdown_contact_count[:] = torch.where(
            first_touchdown_now,
            self.contact_count.float(),
            self.touchdown_contact_count,
        )
        self.touchdown_nonfoot_contact[:] = torch.where(
            first_touchdown_now,
            self.nonfoot_contact.float(),
            self.touchdown_nonfoot_contact,
        )
        self.signed_backward_rotation_at_touchdown[:] = torch.where(
            first_touchdown_now,
            self.signed_backward_rotation,
            self.signed_backward_rotation_at_touchdown,
        )
        relative_foot_w = (
            self.robot.data.body_pos_w[:, self.asset_foot_ids, :]
            - self.robot.data.root_pos_w.unsqueeze(1)
        )
        root_rotation_w = math_utils.matrix_from_quat(
            self.robot.data.root_quat_w
        )
        relative_foot_b = torch.einsum(
            "nfi,nij->nfj", relative_foot_w, root_rotation_w
        )
        mean_foot_drop_body = torch.mean(-relative_foot_b[:, :, 2], dim=-1)
        mean_foot_drop_world = torch.mean(-relative_foot_w[:, :, 2], dim=-1)
        self.touchdown_mean_foot_drop_body_m[:] = torch.where(
            first_touchdown_now,
            mean_foot_drop_body,
            self.touchdown_mean_foot_drop_body_m,
        )
        self.touchdown_mean_foot_drop_world_m[:] = torch.where(
            first_touchdown_now,
            mean_foot_drop_world,
            self.touchdown_mean_foot_drop_world_m,
        )
        joint_pose_error = self.robot.data.joint_pos - self.robot.data.default_joint_pos
        joint_pose_error_rms = torch.sqrt(torch.mean(torch.square(joint_pose_error), dim=-1))
        knee_pose_error_rms = torch.sqrt(
            torch.mean(torch.square(joint_pose_error[:, self.knee_joint_ids]), dim=-1)
        )
        hip_pitch_pose_error_rms = torch.sqrt(
            torch.mean(torch.square(joint_pose_error[:, self.hip_pitch_joint_ids]), dim=-1)
        )
        self.touchdown_joint_pose_error_rad[:] = torch.where(
            first_touchdown_now,
            joint_pose_error_rms,
            self.touchdown_joint_pose_error_rad,
        )
        self.touchdown_knee_pose_error_rad[:] = torch.where(
            first_touchdown_now,
            knee_pose_error_rms,
            self.touchdown_knee_pose_error_rad,
        )
        self.touchdown_hip_pitch_pose_error_rad[:] = torch.where(
            first_touchdown_now,
            hip_pitch_pose_error_rms,
            self.touchdown_hip_pitch_pose_error_rad,
        )
        sorted_foot_origin_z = torch.sort(
            self.robot.data.body_pos_w[:, self.asset_foot_ids, 2], dim=-1
        ).values
        lowest_nonfoot_origin_z = torch.min(
            self.robot.data.body_pos_w[:, self.asset_nonfoot_ids, 2], dim=-1
        ).values
        first_foot_origin_margin = lowest_nonfoot_origin_z - sorted_foot_origin_z[:, 0]
        second_foot_origin_margin = lowest_nonfoot_origin_z - sorted_foot_origin_z[:, 1]
        third_foot_origin_margin = lowest_nonfoot_origin_z - sorted_foot_origin_z[:, 2]
        fourth_foot_origin_margin = lowest_nonfoot_origin_z - sorted_foot_origin_z[:, 3]
        foot_origin_margin_by_leg = (
            lowest_nonfoot_origin_z.unsqueeze(-1)
            - self.robot.data.body_pos_w[:, self.asset_foot_ids, 2]
        )
        action_l2_by_leg = torch.mean(
            torch.square(
                self._env.action_manager.action.reshape(self.num_envs, 4, 3)
            ),
            dim=-1,
        )
        self.touchdown_first_foot_origin_margin_m[:] = torch.where(
            first_touchdown_now,
            first_foot_origin_margin,
            self.touchdown_first_foot_origin_margin_m,
        )
        self.touchdown_second_foot_origin_margin_m[:] = torch.where(
            first_touchdown_now,
            second_foot_origin_margin,
            self.touchdown_second_foot_origin_margin_m,
        )
        self.touchdown_third_foot_origin_margin_m[:] = torch.where(
            first_touchdown_now,
            third_foot_origin_margin,
            self.touchdown_third_foot_origin_margin_m,
        )
        self.touchdown_fourth_foot_origin_margin_m[:] = torch.where(
            first_touchdown_now,
            fourth_foot_origin_margin,
            self.touchdown_fourth_foot_origin_margin_m,
        )
        self.touchdown_foot_origin_margin_by_leg_m[:] = torch.where(
            first_touchdown_now.unsqueeze(-1),
            foot_origin_margin_by_leg,
            self.touchdown_foot_origin_margin_by_leg_m,
        )
        self.touchdown_foot_forward_by_leg_m[:] = torch.where(
            first_touchdown_now.unsqueeze(-1),
            relative_foot_b[:, :, 0],
            self.touchdown_foot_forward_by_leg_m,
        )
        self.touchdown_action_l2_by_leg[:] = torch.where(
            first_touchdown_now.unsqueeze(-1),
            action_l2_by_leg,
            self.touchdown_action_l2_by_leg,
        )
        self.aerial_backward_rotation_at_touchdown[:] = torch.where(
            first_touchdown_now,
            self.signed_backward_rotation - self.backward_rotation_at_takeoff,
            self.aerial_backward_rotation_at_touchdown,
        )
        self.has_touched_down |= touchdown_now
        landed_now = (
            first_touchdown_now
            & any_contact
            & ~self.nonfoot_contact
            & ~self.has_invalid_rotation_axis
            & (self.max_backward_rotation >= self.cfg.minimum_landing_rotation)
            & ~self.has_landed
        )
        self.has_landed |= landed_now
        self.post_touchdown_foot_forward_by_leg_m[:] = torch.where(
            self.has_landed.unsqueeze(-1),
            relative_foot_b[:, :, 0],
            self.post_touchdown_foot_forward_by_leg_m,
        )
        self.was_airborne.copy_(self.airborne)

        post_touchdown_contact_count = torch.where(
            self.has_landed,
            self.contact_count.float(),
            torch.zeros_like(self.touchdown_contact_count),
        )
        self.maximum_post_touchdown_contact_count[:] = torch.maximum(
            self.maximum_post_touchdown_contact_count,
            post_touchdown_contact_count,
        )
        clean_support_now = (
            self.has_landed
            & ~self.nonfoot_contact
            & (self.contact_count >= self.cfg.minimum_support_contact_count)
        )
        self.clean_support_time_s[:] = torch.where(
            clean_support_now,
            self.clean_support_time_s + self._env.step_dt,
            torch.zeros_like(self.clean_support_time_s),
        )
        self.maximum_clean_support_time_s[:] = torch.maximum(
            self.maximum_clean_support_time_s,
            self.clean_support_time_s,
        )

        self.max_base_height[:] = torch.maximum(self.max_base_height, base_height)

        upright = -self.robot.data.projected_gravity_b[:, 2]
        low_motion = (
            (torch.linalg.norm(self.robot.data.root_lin_vel_w, dim=-1) <= self.cfg.stable_linear_speed)
            & (torch.linalg.norm(self.robot.data.root_ang_vel_w, dim=-1) <= self.cfg.stable_angular_speed)
        )
        rotation_ok = (
            self.has_taken_off
            & (self.max_backward_rotation >= self.cfg.success_rotation_range[0])
            & (self.max_backward_rotation <= self.cfg.success_rotation_range[1])
            & ~self.has_invalid_rotation_axis
            & (
                self.backward_rotation_at_takeoff
                <= self.cfg.maximum_rotation_at_takeoff
            )
        )
        landing_rotation_ok = (
            (self.signed_backward_rotation >= self.cfg.success_rotation_range[0])
            & (self.signed_backward_rotation <= self.cfg.success_rotation_range[1])
            & ~self.has_invalid_rotation_axis
            & (
                self.backward_rotation_at_takeoff
                <= self.cfg.maximum_rotation_at_takeoff
            )
        )
        stable_now = (
            self.has_landed
            & landing_rotation_ok
            & ~self.nonfoot_contact
            & (self.contact_count >= self.cfg.success_contact_count)
            & (upright >= self.cfg.success_upright_cosine)
            & low_motion
        )
        self.stable_landing_time_s[:] = torch.where(
            stable_now,
            self.stable_landing_time_s + self._env.step_dt,
            torch.zeros_like(self.stable_landing_time_s),
        )
        success_now = (
            self.stable_landing_time_s
            >= self.cfg.minimum_stable_landing_time_s
        )
        self.landing_success |= success_now

        self.metrics["takeoff_success"][:] = self.has_taken_off.float()
        self.metrics["rotation_success"][:] = rotation_ok.float()
        self.metrics["rotation_axis_success"][:] = (
            ~self.has_invalid_rotation_axis
        ).float()
        self.metrics["landing_rotation_success"][:] = (
            self.has_landed & landing_rotation_ok
        ).float()
        self.metrics["touchdown_success"][:] = self.has_touched_down.float()
        self.metrics["landing_success"][:] = self.landing_success.float()
        self.metrics["max_base_height_m"][:] = self.max_base_height
        self.metrics["max_backward_rotation_rad"][:] = self.max_backward_rotation
        self.metrics["takeoff_time_s"][:] = self.takeoff_time_s
        self.metrics["apex_time_s"][:] = self.apex_time_s
        self.metrics["backward_rotation_at_apex_rad"][:] = (
            self.backward_rotation_at_apex
        )
        self.metrics["first_touchdown_time_s"][:] = self.first_touchdown_time_s
        self.metrics["flight_time_s"][:] = self.flight_time_s
        self.metrics["touchdown_upright_cosine"][:] = self.touchdown_upright_cosine
        self.metrics["touchdown_vertical_speed_m_s"][:] = self.touchdown_vertical_speed
        self.metrics["touchdown_linear_speed_m_s"][:] = self.touchdown_linear_speed
        self.metrics["touchdown_angular_speed_rad_s"][:] = self.touchdown_angular_speed
        self.metrics["touchdown_contact_count"][:] = self.touchdown_contact_count
        self.metrics["touchdown_nonfoot_contact"][:] = (
            self.touchdown_nonfoot_contact
        )
        self.metrics["signed_backward_rotation_at_touchdown_rad"][:] = (
            self.signed_backward_rotation_at_touchdown
        )
        self.metrics["aerial_backward_rotation_at_touchdown_rad"][:] = (
            self.aerial_backward_rotation_at_touchdown
        )
        self.metrics["backward_pitch_rate_at_takeoff_rad_s"][:] = (
            self.backward_pitch_rate_at_takeoff
        )
        self.metrics["backward_rotation_at_takeoff_rad"][:] = (
            self.backward_rotation_at_takeoff
        )
        self.metrics["stable_landing_time_s"][:] = self.stable_landing_time_s
        self.metrics["touchdown_mean_foot_drop_body_m"][:] = (
            self.touchdown_mean_foot_drop_body_m
        )
        self.metrics["touchdown_mean_foot_drop_world_m"][:] = (
            self.touchdown_mean_foot_drop_world_m
        )
        self.metrics["touchdown_joint_pose_error_rad"][:] = (
            self.touchdown_joint_pose_error_rad
        )
        self.metrics["touchdown_knee_pose_error_rad"][:] = (
            self.touchdown_knee_pose_error_rad
        )
        self.metrics["touchdown_hip_pitch_pose_error_rad"][:] = (
            self.touchdown_hip_pitch_pose_error_rad
        )
        self.metrics["touchdown_first_foot_origin_margin_m"][:] = (
            self.touchdown_first_foot_origin_margin_m
        )
        self.metrics["touchdown_second_foot_origin_margin_m"][:] = (
            self.touchdown_second_foot_origin_margin_m
        )
        self.metrics["touchdown_third_foot_origin_margin_m"][:] = (
            self.touchdown_third_foot_origin_margin_m
        )
        self.metrics["touchdown_fourth_foot_origin_margin_m"][:] = (
            self.touchdown_fourth_foot_origin_margin_m
        )
        self.metrics["maximum_post_touchdown_contact_count"][:] = (
            self.maximum_post_touchdown_contact_count
        )
        self.metrics["maximum_clean_support_time_s"][:] = (
            self.maximum_clean_support_time_s
        )
        for leg_index, leg_name in enumerate(("fl", "fr", "rl", "rr")):
            self.metrics[f"touchdown_{leg_name}_foot_origin_margin_m"][:] = (
                self.touchdown_foot_origin_margin_by_leg_m[:, leg_index]
            )
            self.metrics[f"touchdown_{leg_name}_foot_forward_m"][:] = (
                self.touchdown_foot_forward_by_leg_m[:, leg_index]
            )
            self.metrics[f"post_touchdown_{leg_name}_foot_forward_m"][:] = (
                self.post_touchdown_foot_forward_by_leg_m[:, leg_index]
            )
            self.metrics[f"touchdown_{leg_name}_action_l2"][:] = (
                self.touchdown_action_l2_by_leg[:, leg_index]
            )

    def _resample_command(self, env_ids: Sequence[int]):
        # This is a deterministic one-shot skill. Resampling only restarts the
        # timer, which is already handled by the environment episode reset.
        self._command[env_ids] = 0.0
        self._command[env_ids, 1] = 1.0
        self._command[env_ids, 3] = 1.0
        self._command[env_ids, 5] = 1.0

    def _update_command(self):
        self._write_phase_command()

    def _write_phase_command(self):
        elapsed_s = self._env.episode_length_buf * self._env.step_dt
        self._command[:] = multiscale_phase_features(
            elapsed_s,
            self.cfg.episode_length_s,
            self.cfg.phase_cycles,
        )


@configclass
class BackflipPhaseCommandCfg(CommandTermCfg):
    """Configuration for the deterministic backflip phase command."""

    class_type: type = BackflipPhaseCommand
    resampling_time_range: tuple[float, float] = (1000.0, 1000.0)
    debug_vis: bool = False

    asset_name: str = "robot"
    contact_sensor_name: str = "contact_forces"
    foot_body_names: str = ".*_foot"
    episode_length_s: float = 1.8
    phase_cycles: float = 1.0
    initial_base_height: float = 0.30
    minimum_takeoff_time_s: float = 0.20
    minimum_takeoff_height_gain: float = 0.03
    minimum_takeoff_vertical_speed: float = 0.25
    minimum_support_contact_count: int = 2
    contact_force_threshold: float = 2.0
    backward_pitch_sign: float = -1.0
    minimum_landing_rotation: float = 1.50 * math.pi
    success_rotation_range: tuple[float, float] = (1.75 * math.pi, 2.25 * math.pi)
    success_contact_count: int = 3
    success_upright_cosine: float = math.cos(math.radians(20.0))
    stable_linear_speed: float = 0.75
    stable_angular_speed: float = 2.0
    minimum_pitch_xz_norm: float = math.cos(math.radians(30.0))
    maximum_rotation_at_takeoff: float = 0.35 * math.pi
    minimum_stable_landing_time_s: float = 0.20

    def __post_init__(self):
        if self.episode_length_s <= 0.0 or self.minimum_takeoff_time_s < 0.0:
            raise ValueError("Episode length must be positive and takeoff time must be non-negative.")
        if self.phase_cycles <= 0.0:
            raise ValueError("phase_cycles must be positive.")
        if self.minimum_takeoff_height_gain < 0.0 or self.minimum_takeoff_vertical_speed < 0.0:
            raise ValueError("Takeoff height and vertical-speed thresholds must be non-negative.")
        if self.contact_force_threshold <= 0.0:
            raise ValueError("contact_force_threshold must be positive.")
        if self.backward_pitch_sign not in (-1.0, 1.0):
            raise ValueError("backward_pitch_sign must be either -1.0 or 1.0.")
        if self.minimum_landing_rotation <= 0.0:
            raise ValueError("minimum_landing_rotation must be positive.")
        if not 0.0 < self.minimum_pitch_xz_norm <= 1.0:
            raise ValueError("minimum_pitch_xz_norm must be in (0, 1].")
        if self.maximum_rotation_at_takeoff <= 0.0:
            raise ValueError("maximum_rotation_at_takeoff must be positive.")
        if self.minimum_stable_landing_time_s <= 0.0:
            raise ValueError("minimum_stable_landing_time_s must be positive.")
        if not 1 <= self.success_contact_count <= 4:
            raise ValueError("success_contact_count must be in [1, 4].")
        if not 1 <= self.minimum_support_contact_count <= 4:
            raise ValueError("minimum_support_contact_count must be in [1, 4].")
