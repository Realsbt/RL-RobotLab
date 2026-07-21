"""Reward terms for staged ToGo_LFs backflip learning."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

from .backflip_math import (
    centroidal_angular_velocity,
    desired_projected_gravity,
    linear_rotation_projected_gravity,
    linear_clearance_scale,
    rotation_schedule,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _state(env: "ManagerBasedRLEnv", command_name: str):
    return env.command_manager.get_term(command_name)


def _elapsed_s(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.episode_length_buf * env.step_dt


def jump_height(
    env: "ManagerBasedRLEnv",
    initial_height: float,
    maximum_height_gain: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward bounded root-height gain above the nominal standing height."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(
        asset.data.root_pos_w[:, 2] - initial_height,
        min=0.0,
        max=maximum_height_gain,
    )


def upward_velocity_window(
    env: "ManagerBasedRLEnv",
    start_s: float,
    end_s: float,
    maximum_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward upward root velocity only during the launch window."""
    asset: Articulation = env.scene[asset_cfg.name]
    elapsed_s = _elapsed_s(env)
    active = (elapsed_s >= start_s) & (elapsed_s <= end_s)
    return torch.clamp(asset.data.root_lin_vel_w[:, 2], min=0.0, max=maximum_speed) * active


def genesis_backward_pitch_velocity(
    env: "ManagerBasedRLEnv",
    start_s: float,
    end_s: float,
    maximum_rate: float,
    backward_pitch_sign: float = -1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reproduce the signed, time-windowed pitch signal from Genesis-backflip."""
    if end_s <= start_s or maximum_rate <= 0.0:
        raise ValueError("Invalid Genesis pitch window or maximum rate.")
    if backward_pitch_sign not in (-1.0, 1.0):
        raise ValueError("backward_pitch_sign must be either -1.0 or 1.0.")
    asset: Articulation = env.scene[asset_cfg.name]
    elapsed_s = _elapsed_s(env)
    active = (elapsed_s > start_s) & (elapsed_s < end_s)
    backward_rate = backward_pitch_sign * asset.data.root_ang_vel_b[:, 1]
    return torch.clamp(backward_rate, min=-maximum_rate, max=maximum_rate) * active


def genesis_backward_pitch_velocity_before_touchdown(
    env: "ManagerBasedRLEnv",
    command_name: str,
    start_s: float,
    end_s: float,
    maximum_rate: float,
    backward_pitch_sign: float = -1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Use Genesis' pitch signal, but never pay for ground tumbling."""
    if end_s <= start_s or maximum_rate <= 0.0:
        raise ValueError("Invalid Genesis pitch window or maximum rate.")
    if backward_pitch_sign not in (-1.0, 1.0):
        raise ValueError("backward_pitch_sign must be either -1.0 or 1.0.")
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    elapsed_s = _elapsed_s(env)
    active = (
        (elapsed_s > start_s)
        & (elapsed_s < end_s)
        & ~state.has_touched_down
    )
    backward_rate = backward_pitch_sign * asset.data.root_ang_vel_b[:, 1]
    return torch.clamp(backward_rate, min=-maximum_rate, max=maximum_rate) * active


def first_touchdown_rotation_quality(
    env: "ManagerBasedRLEnv",
    command_name: str,
    target_rotation: float = 2.0 * math.pi,
    takeoff_excess_scale: float = 0.25,
) -> torch.Tensor:
    """Score touchdown angle while discouraging supported pre-rotation.

    The command tracker freezes its maximum at first touchdown.  Restricting
    this reward to that event avoids both ground-tumble credit and the incentive
    to touch down early merely to collect the same quality for many steps.  A
    smooth multiplier reduces the score when too much of the revolution was
    completed before takeoff, without introducing a discontinuous success gate.
    """
    if target_rotation <= 0.0:
        raise ValueError("target_rotation must be positive.")
    if takeoff_excess_scale <= 0.0:
        raise ValueError("takeoff_excess_scale must be positive.")
    state = _state(env, command_name)
    elapsed_s = _elapsed_s(env)
    # The reward manager can observe the command event one control step after
    # contact-manager data changes, so keep a narrow 1.5-step event window.
    touchdown_event = state.has_touched_down & (
        elapsed_s <= state.first_touchdown_time_s + 1.5 * env.step_dt
    )
    quality = torch.clamp(
        state.signed_backward_rotation_at_touchdown / target_rotation,
        min=0.0,
        max=1.0,
    )
    takeoff_excess = torch.clamp(
        state.backward_rotation_at_takeoff
        - state.cfg.maximum_rotation_at_takeoff,
        min=0.0,
    )
    takeoff_quality = torch.exp(
        -torch.square(takeoff_excess) / takeoff_excess_scale
    )
    return (
        torch.square(quality)
        * takeoff_quality
        * touchdown_event
        * ~state.has_invalid_rotation_axis
    )


def first_touchdown_nonfoot_contact(
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> torch.Tensor:
    """Return a narrow event penalty when something other than a foot lands first."""
    state = _state(env, command_name)
    elapsed_s = _elapsed_s(env)
    touchdown_event = state.has_touched_down & (
        elapsed_s <= state.first_touchdown_time_s + 1.5 * env.step_dt
    )
    return state.touchdown_nonfoot_contact * touchdown_event


def first_touchdown_foot_contact_quality(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    target_rotation: float = 2.0 * math.pi,
    rotation_error_scale: float = 0.50,
) -> torch.Tensor:
    """Reward clean foot-first contact, scaled by foot count and flip angle.

    A simultaneous non-foot collision makes the event worth zero.  The narrow
    event window prevents a one-foot touchdown from being paid repeatedly as
    more feet arrive later; later support is handled by landing rewards.
    """
    if minimum_rotation <= 0.0 or target_rotation <= minimum_rotation:
        raise ValueError("Touchdown rotation thresholds must be positive and increasing.")
    if rotation_error_scale <= 0.0:
        raise ValueError("rotation_error_scale must be positive.")
    state = _state(env, command_name)
    elapsed_s = _elapsed_s(env)
    touchdown_event = state.has_touched_down & (
        elapsed_s <= state.first_touchdown_time_s + 1.5 * env.step_dt
    )
    rotation = state.signed_backward_rotation_at_touchdown
    rotation_quality = torch.exp(
        -torch.square(rotation - target_rotation) / rotation_error_scale
    )
    credible_rotation = rotation >= minimum_rotation
    clean_foot_contact = ~state.touchdown_nonfoot_contact.bool()
    foot_ratio = state.touchdown_contact_count / 4.0
    return (
        foot_ratio
        * rotation_quality
        * credible_rotation
        * clean_foot_contact
        * touchdown_event
    )


def genesis_vertical_velocity(
    env: "ManagerBasedRLEnv",
    start_s: float,
    end_s: float,
    maximum_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Use Genesis' signed vertical-velocity reward during the launch window."""
    if end_s <= start_s or maximum_speed <= 0.0:
        raise ValueError("Invalid Genesis vertical-velocity window or maximum speed.")
    asset: Articulation = env.scene[asset_cfg.name]
    elapsed_s = _elapsed_s(env)
    active = (elapsed_s > start_s) & (elapsed_s < end_s)
    return torch.clamp(asset.data.root_lin_vel_w[:, 2], max=maximum_speed) * active


def genesis_orientation_error_l2(
    env: "ManagerBasedRLEnv",
    rotation_start_s: float,
    rotation_end_s: float,
    backward_pitch_sign: float = -1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track Genesis' linear upright-to-upright full-pitch reference."""
    asset: Articulation = env.scene[asset_cfg.name]
    desired_gravity = linear_rotation_projected_gravity(
        _elapsed_s(env),
        rotation_start_s,
        rotation_end_s,
        backward_pitch_sign,
    )
    return torch.sum(
        torch.square(asset.data.projected_gravity_b - desired_gravity), dim=-1
    )


def genesis_base_height_l2(
    env: "ManagerBasedRLEnv",
    target_height: float,
    early_end_s: float,
    late_start_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize standing-height error before launch and after the flip."""
    if late_start_s <= early_end_s:
        raise ValueError("late_start_s must be greater than early_end_s.")
    asset: Articulation = env.scene[asset_cfg.name]
    elapsed_s = _elapsed_s(env)
    active = (elapsed_s < early_end_s) | (elapsed_s > late_start_s)
    height = asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return torch.square(target_height - height) * active


def genesis_feet_height_before_rotation(
    env: "ManagerBasedRLEnv",
    end_s: float,
    ground_clearance: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Discourage lifting the feet before Genesis' scheduled launch begins."""
    if end_s <= 0.0 or ground_clearance < 0.0:
        raise ValueError("Invalid Genesis early-foot-height parameters.")
    asset: Articulation = env.scene[asset_cfg.name]
    ground_z = env.scene.env_origins[:, 2].unsqueeze(-1)
    foot_height = (
        asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        - ground_z
        - ground_clearance
    )
    active = _elapsed_s(env) < end_s
    return torch.clamp(foot_height, min=0.0).sum(dim=-1) * active


def genesis_yaw_rate_l1(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize body-frame yaw rate without suppressing the pitch flip."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_ang_vel_b[:, 2])


def genesis_gravity_y_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll tilt using the lateral projected-gravity component."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.projected_gravity_b[:, 1])


def genesis_feet_lateral_distance_l2(
    env: "ManagerBasedRLEnv",
    stance_width: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize lateral foot-position error in the root frame."""
    if stance_width < 0.0:
        raise ValueError("stance_width must be non-negative.")
    asset: Articulation = env.scene[asset_cfg.name]
    relative_w = (
        asset.data.body_pos_w[:, asset_cfg.body_ids, :]
        - asset.data.root_pos_w.unsqueeze(1)
    )
    num_feet = relative_w.shape[1]
    root_quat = (
        asset.data.root_quat_w.unsqueeze(1)
        .expand(-1, num_feet, -1)
        .reshape(-1, 4)
    )
    feet_b = math_utils.quat_apply_inverse(
        root_quat, relative_w.reshape(-1, 3)
    ).reshape(env.num_envs, num_feet, 3)
    side_sign = torch.tensor(
        [1.0 if index % 2 == 0 else -1.0 for index in range(num_feet)],
        device=env.device,
        dtype=feet_b.dtype,
    )
    desired_y = 0.5 * stance_width * side_sign
    return torch.sum(torch.square(feet_b[:, :, 1] - desired_y), dim=-1)


def upright_outside_rotation_window(
    env: "ManagerBasedRLEnv",
    rotation_start_s: float,
    rotation_end_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward upright posture before launch and after the scheduled rotation."""
    asset: Articulation = env.scene[asset_cfg.name]
    elapsed_s = _elapsed_s(env)
    active = (elapsed_s < rotation_start_s) | (elapsed_s > rotation_end_s)
    upright = torch.clamp(-asset.data.projected_gravity_b[:, 2], min=0.0, max=1.0)
    return upright * active


def takeoff_success(
    env: "ManagerBasedRLEnv",
    command_name: str = "backflip_phase",
) -> torch.Tensor:
    """Reward episodes that have achieved a clean all-feet takeoff."""
    return _state(env, command_name).has_taken_off.float()


def takeoff_backward_pitch_quality(
    env: "ManagerBasedRLEnv",
    command_name: str,
    target_rate: float,
    reward_end_s: float,
) -> torch.Tensor:
    """Reward backward base pitch already present at takeoff.

    The signed quality remains active only through the early flight phase.  A
    forward pitch at takeoff therefore receives a penalty instead of being
    recoverable for free by a delayed aerial body/leg exchange.
    """
    if target_rate <= 0.0 or reward_end_s <= 0.0:
        raise ValueError("target_rate and reward_end_s must be positive.")
    state = _state(env, command_name)
    quality = torch.clamp(
        state.backward_pitch_rate_at_takeoff / target_rate,
        min=-1.0,
        max=1.0,
    )
    active = state.has_taken_off & (_elapsed_s(env) <= reward_end_s)
    return quality * active


def takeoff_rotation_excess_l2(
    env: "ManagerBasedRLEnv",
    command_name: str = "backflip_phase",
) -> torch.Tensor:
    """Penalize the squared supported rotation beyond the takeoff allowance.

    The takeoff angle is frozen at the first credible all-body takeoff and the
    penalty remains active only during that flight. This gives PPO a denser
    signal than a success gate while leaving the required takeoff angular
    velocity unconstrained.
    """
    state = _state(env, command_name)
    excess = torch.clamp(
        state.backward_rotation_at_takeoff
        - state.cfg.maximum_rotation_at_takeoff,
        min=0.0,
    )
    active = state.has_taken_off & ~state.has_touched_down
    return torch.square(excess) * active


def supported_rotation_l2(
    env: "ManagerBasedRLEnv",
    command_name: str = "backflip_phase",
) -> torch.Tensor:
    """Penalize body pitch accumulated while the launch is still supported.

    This immediate pre-takeoff signal complements the frozen takeoff-angle
    penalty. It encourages a late angular impulse and early release instead of
    slowly pivoting around feet that remain on the ground.
    """
    state = _state(env, command_name)
    active = (
        state.has_support
        & ~state.has_taken_off
        & ~state.has_touched_down
    )
    return torch.square(state.signed_backward_rotation) * active


def launch_backward_pitch_quality(
    env: "ManagerBasedRLEnv",
    command_name: str,
    start_s: float,
    end_s: float,
    target_rate: float,
    minimum_upward_speed: float,
    full_upward_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward backward pitch while supported feet are producing the jump.

    Multiplying by upward launch quality prevents the policy from earning this
    term by simply rocking backward on the ground.  Unlike the takeoff-state
    reward, this dense signal gives PPO direct credit during the few control
    steps in which ground reaction forces can create angular momentum.
    """
    if end_s <= start_s or target_rate <= 0.0:
        raise ValueError("Invalid launch pitch window or target rate.")
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    elapsed_s = _elapsed_s(env)
    supported = torch.any(state.contact, dim=-1)
    active = supported & (elapsed_s >= start_s) & (elapsed_s <= end_s)
    backward_rate = state.cfg.backward_pitch_sign * asset.data.root_ang_vel_w[:, 1]
    rate_quality = torch.clamp(backward_rate / target_rate, min=-1.0, max=1.0)
    upward_quality = linear_clearance_scale(
        asset.data.root_lin_vel_w[:, 2],
        minimum_upward_speed,
        full_upward_speed,
    )
    return rate_quality * upward_quality * active


def apex_rotation_quality(
    env: "ManagerBasedRLEnv",
    command_name: str,
    target_rotation: float,
    reward_end_s: float,
) -> torch.Tensor:
    """Reward reaching the requested backward rotation by the flight apex."""
    if target_rotation <= 0.0 or reward_end_s <= 0.0:
        raise ValueError("target_rotation and reward_end_s must be positive.")
    state = _state(env, command_name)
    quality = torch.clamp(
        state.backward_rotation_at_apex / target_rotation,
        min=0.0,
        max=1.0,
    )
    active = state.has_reached_apex & (_elapsed_s(env) <= reward_end_s)
    return quality * active


def backward_pitch_rate(
    env: "ManagerBasedRLEnv",
    command_name: str,
    start_s: float,
    end_s: float,
    maximum_rate: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward backward pitch rate while airborne in the rotation window."""
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    elapsed_s = _elapsed_s(env)
    active = state.airborne & (elapsed_s >= start_s) & (elapsed_s <= end_s)
    rate = state.cfg.backward_pitch_sign * asset.data.root_ang_vel_w[:, 1]
    return torch.clamp(rate, min=0.0, max=maximum_rate) * active


class centroidal_pitch_velocity(ManagerTermBase):
    """Reward the pitch component of whole-system centroidal velocity."""

    def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = env.scene[asset_cfg.name]
        # Physical properties are loaded lazily on the first reward call. This
        # occurs after startup mass/COM randomization in the robust stage.
        self._masses: torch.Tensor | None = None
        self._body_inertias_b: torch.Tensor | None = None

    def _load_physical_properties(self) -> None:
        device = self.asset.data.body_com_pos_w.device
        dtype = self.asset.data.body_com_pos_w.dtype
        self._masses = self.asset.root_physx_view.get_masses().to(
            device=device, dtype=dtype
        )
        inertias = self.asset.root_physx_view.get_inertias().to(
            device=device, dtype=dtype
        )
        self._body_inertias_b = inertias.reshape(*inertias.shape[:-1], 3, 3)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        command_name: str,
        start_s: float,
        end_s: float,
        maximum_rate: float,
        negative_rate_floor: float = -0.1,
        minimum_peak_height: float = 0.58,
        full_peak_height: float = 0.72,
        minimum_current_height: float = 0.25,
        full_current_height: float = 0.55,
        require_takeoff: bool = True,
        require_support: bool = False,
        minimum_upward_speed: float | None = None,
        full_upward_speed: float | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        del asset_cfg
        if end_s <= start_s:
            raise ValueError("end_s must be greater than start_s.")
        if maximum_rate <= 0.0 or negative_rate_floor > 0.0:
            raise ValueError("Invalid centroidal angular-velocity clamp.")
        if (minimum_upward_speed is None) != (full_upward_speed is None):
            raise ValueError("Both upward-speed gates must be provided together.")
        if (
            minimum_upward_speed is not None
            and full_upward_speed <= minimum_upward_speed
        ):
            raise ValueError("full_upward_speed must exceed minimum_upward_speed.")
        if self._masses is None or self._body_inertias_b is None:
            self._load_physical_properties()

        rotation_w = math_utils.matrix_from_quat(self.asset.data.body_link_quat_w)
        inertia_w = torch.matmul(
            torch.matmul(rotation_w, self._body_inertias_b),
            rotation_w.transpose(-1, -2),
        )
        centroidal_velocity_w = centroidal_angular_velocity(
            self._masses,
            self.asset.data.body_com_pos_w,
            self.asset.data.body_com_lin_vel_w,
            self.asset.data.body_com_ang_vel_w,
            inertia_w,
        )

        state = _state(env, command_name)
        elapsed_s = _elapsed_s(env)
        active = (
            ~state.has_landed
            & (elapsed_s >= start_s)
            & (elapsed_s <= end_s)
        )
        if require_takeoff:
            active &= state.has_taken_off
        if require_support:
            active &= torch.any(state.contact, dim=-1)
        pitch_rate = state.cfg.backward_pitch_sign * centroidal_velocity_w[:, 1]
        peak_quality = linear_clearance_scale(
            state.max_base_height, minimum_peak_height, full_peak_height
        )
        current_clearance = linear_clearance_scale(
            self.asset.data.root_pos_w[:, 2],
            minimum_current_height,
            full_current_height,
        )
        upward_quality = torch.ones_like(pitch_rate)
        if minimum_upward_speed is not None:
            upward_quality = linear_clearance_scale(
                self.asset.data.root_lin_vel_w[:, 2],
                minimum_upward_speed,
                full_upward_speed,
            )
        return torch.clamp(
            pitch_rate, min=negative_rate_floor, max=maximum_rate
        ) * active * peak_quality * current_clearance * upward_quality


def backward_rotation_progress(
    env: "ManagerBasedRLEnv",
    command_name: str = "backflip_phase",
) -> torch.Tensor:
    """Reward only newly achieved backward rotation, expressed as rad/s."""
    state = _state(env, command_name)
    return state.rotation_increment / env.step_dt


def capped_backward_rotation_progress(
    env: "ManagerBasedRLEnv",
    command_name: str = "backflip_phase",
    maximum_rewarded_rotation: float = 2.0 * math.pi,
) -> torch.Tensor:
    """Reward new backward rotation only until the requested revolution.

    Unlike the stage-2 clearance-gated reward, this landing-stage term has no
    prescribed height or apex timing. It stops paying once the robot reaches
    one revolution so that continued spinning cannot improve the return.
    """
    if maximum_rewarded_rotation <= 0.0:
        raise ValueError("maximum_rewarded_rotation must be positive.")
    state = _state(env, command_name)
    previous_max = state.max_backward_rotation - state.rotation_increment
    remaining = torch.clamp(maximum_rewarded_rotation - previous_max, min=0.0)
    rewarded_increment = torch.minimum(state.rotation_increment, remaining)
    active = state.has_taken_off & ~state.has_touched_down
    return rewarded_increment / env.step_dt * active


def safe_backward_rotation_progress(
    env: "ManagerBasedRLEnv",
    command_name: str,
    start_s: float,
    end_s: float,
    minimum_peak_height: float,
    full_peak_height: float,
    minimum_current_height: float,
    full_current_height: float,
    maximum_rewarded_rotation: float = 2.0 * math.pi,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward new rotation only after a high launch and with ground clearance.

    This prevents the stage-2 policy from trading away its vertical jump and
    collecting rotation reward by pitching onto its torso close to the ground.
    Progress before the scheduled rotation window and beyond one revolution is
    deliberately worth zero.
    """
    if end_s <= start_s:
        raise ValueError("end_s must be greater than start_s.")
    if maximum_rewarded_rotation <= 0.0:
        raise ValueError("maximum_rewarded_rotation must be positive.")

    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    elapsed_s = _elapsed_s(env)
    active = (
        state.has_taken_off
        & state.airborne
        & ~state.has_touched_down
        & (elapsed_s >= start_s)
        & (elapsed_s <= end_s)
    )

    # Cap the last increment at exactly one revolution so overspinning cannot
    # produce an unbounded dense reward.
    previous_max = state.max_backward_rotation - state.rotation_increment
    remaining = torch.clamp(maximum_rewarded_rotation - previous_max, min=0.0)
    rewarded_increment = torch.minimum(state.rotation_increment, remaining)

    peak_quality = linear_clearance_scale(
        state.max_base_height, minimum_peak_height, full_peak_height
    )
    current_clearance = linear_clearance_scale(
        asset.data.root_pos_w[:, 2], minimum_current_height, full_current_height
    )
    return (
        rewarded_increment
        / env.step_dt
        * peak_quality
        * current_clearance
        * active
    )


def airborne_height_clearance(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_height: float,
    full_reward_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward useful flight time above a safe root-height clearance."""
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    flight = state.has_taken_off & state.airborne & ~state.has_touched_down
    clearance = linear_clearance_scale(
        asset.data.root_pos_w[:, 2], minimum_height, full_reward_height
    )
    return clearance * flight


def rotation_completion_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
) -> torch.Tensor:
    """Reward a completed rotation for every remaining non-failed step.

    Persisting the bonus after the threshold turns rare exploratory full flips
    into a strong PPO learning signal and also favors surviving after completion.
    """
    if minimum_rotation <= 0.0:
        raise ValueError("minimum_rotation must be positive.")
    state = _state(env, command_name)
    return (
        (state.max_backward_rotation >= minimum_rotation)
        & ~state.has_invalid_rotation_axis
    ).float()


def rotation_range_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    maximum_rotation: float,
) -> torch.Tensor:
    """Reward retaining a completed, non-overrotated backflip."""
    if minimum_rotation <= 0.0 or maximum_rotation <= minimum_rotation:
        raise ValueError("Rotation range must be positive and increasing.")
    state = _state(env, command_name)
    rotation = state.signed_backward_rotation
    return (
        (rotation >= minimum_rotation)
        & (rotation <= maximum_rotation)
        & ~state.has_invalid_rotation_axis
    ).float()


def rotation_before_deadline_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    deadline_s: float,
) -> torch.Tensor:
    """Reward a rotation milestone only while its deadline has not passed.

    The ordinary completion bonus can be collected by rotating late in the
    flight.  This deadline-gated variant is used by the takeoff-retiming stage:
    reaching a milestone earlier earns the bonus for more control steps, while
    reaching it after the apex earns nothing.
    """
    if minimum_rotation <= 0.0 or deadline_s <= 0.0:
        raise ValueError("minimum_rotation and deadline_s must be positive.")
    state = _state(env, command_name)
    reached = state.max_backward_rotation >= minimum_rotation
    before_deadline = _elapsed_s(env) <= deadline_s
    return (reached & before_deadline).float()


def failure_before_rotation(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    term_keys: list[str],
) -> torch.Tensor:
    """Penalize hard failures only while the rotation skill is incomplete.

    Stage 2 is responsible for discovering a complete aerial revolution, not
    for absorbing the landing impact. Once the requested rotation is reached,
    landing failures are deliberately deferred to Stage 3. This preserves the
    sparse successful explorations that would otherwise receive the same large
    terminal penalty as an early torso fall.
    """
    if minimum_rotation <= 0.0:
        raise ValueError("minimum_rotation must be positive.")
    if not term_keys:
        raise ValueError("term_keys must contain at least one termination term.")

    failure = torch.zeros(env.num_envs, device=env.device)
    for term_key in term_keys:
        failure += env.termination_manager.get_term(term_key)
    incomplete = _state(env, command_name).max_backward_rotation < minimum_rotation
    return failure * incomplete * (~env.termination_manager.time_outs)


def premature_pitch_rate_l2(
    env: "ManagerBasedRLEnv",
    rotation_start_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize pitching before the vertical launch phase is complete."""
    asset: Articulation = env.scene[asset_cfg.name]
    active = _elapsed_s(env) < rotation_start_s
    return torch.square(asset.data.root_ang_vel_w[:, 1]) * active


def scheduled_orientation_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str,
    rotation_start_s: float,
    rotation_end_s: float,
    error_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track a smooth upright-to-upright full-pitch gravity reference."""
    if error_scale <= 0.0:
        raise ValueError("error_scale must be positive.")
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    desired_gravity = desired_projected_gravity(
        _elapsed_s(env),
        rotation_start_s,
        rotation_end_s,
        state.cfg.backward_pitch_sign,
    )
    error = torch.sum(torch.square(asset.data.projected_gravity_b - desired_gravity), dim=-1)
    return torch.exp(-error / error_scale)


def unwrapped_rotation_tracking(
    env: "ManagerBasedRLEnv",
    command_name: str,
    rotation_start_s: float,
    rotation_end_s: float,
    error_scale: float,
) -> torch.Tensor:
    """Track monotonic accumulated rotation, removing half-turn ambiguity."""
    if error_scale <= 0.0:
        raise ValueError("error_scale must be positive.")
    state = _state(env, command_name)
    desired_rotation = 2.0 * math.pi * rotation_schedule(
        _elapsed_s(env), rotation_start_s, rotation_end_s
    )
    error = torch.square(state.signed_backward_rotation - desired_rotation)
    active = state.has_taken_off & ~state.has_touched_down
    return torch.exp(-error / error_scale) * active


def wrong_pitch_direction(
    env: "ManagerBasedRLEnv",
    command_name: str,
    maximum_rate: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize forward pitch rate during flight."""
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    backward_rate = state.cfg.backward_pitch_sign * asset.data.root_ang_vel_w[:, 1]
    rotation_active = state.has_taken_off & ~state.has_touched_down
    return torch.clamp(-backward_rate, min=0.0, max=maximum_rate) * rotation_active


def off_axis_angular_velocity_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll and yaw angular velocity without suppressing pitch."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_w[:, (0, 2)]), dim=-1)


def mirrored_action_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize left/right asymmetry using the LFS mirrored joint convention."""
    actions = env.action_manager.action.reshape(env.num_envs, 4, 3)
    front_error = actions[:, 0] + actions[:, 1]
    rear_error = actions[:, 2] + actions[:, 3]
    return torch.mean(torch.square(front_error), dim=-1) + torch.mean(
        torch.square(rear_error), dim=-1
    )


def lateral_drift_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize horizontal displacement from the environment origin."""
    asset: Articulation = env.scene[asset_cfg.name]
    relative_xy = asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return torch.sum(torch.square(relative_xy), dim=-1)


def landing_contact_ratio(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
) -> torch.Tensor:
    """Reward multi-foot contact only after a credible full rotation."""
    state = _state(env, command_name)
    eligible = state.has_landed & (state.max_backward_rotation >= minimum_rotation)
    return state.contact_count.float() / 4.0 * eligible


def landing_angular_speed_excess_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    full_rotation: float,
    maximum_angular_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize residual angular speed only during the final airborne approach."""
    if minimum_rotation <= 0.0 or full_rotation <= minimum_rotation:
        raise ValueError("Landing rotation thresholds must be positive and increasing.")
    if maximum_angular_speed <= 0.0:
        raise ValueError("maximum_angular_speed must be positive.")

    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    active = (
        state.has_taken_off
        & state.airborne
        & ~state.has_touched_down
        & (asset.data.root_lin_vel_w[:, 2] < 0.0)
        & (state.signed_backward_rotation >= minimum_rotation)
    )
    rotation_quality = linear_clearance_scale(
        state.signed_backward_rotation, minimum_rotation, full_rotation
    )
    angular_speed = torch.linalg.norm(asset.data.root_ang_vel_w, dim=-1)
    excess = torch.clamp(angular_speed - maximum_angular_speed, min=0.0)
    return torch.square(excess) * rotation_quality * active


def post_touchdown_clean_support(
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> torch.Tensor:
    """Reward retained multi-foot support after a clean first touchdown."""
    state = _state(env, command_name)
    clean = state.has_landed & ~state.nonfoot_contact
    return state.contact_count.float() / 4.0 * clean


def post_touchdown_foot_origin_clearance(
    env: "ManagerBasedRLEnv",
    command_name: str,
    foot_rank: int,
    minimum_margin: float,
    full_margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continue deploying the remaining feet after a clean first contact."""
    if not 1 <= foot_rank <= 4:
        raise ValueError("foot_rank must be in [1, 4].")
    if full_margin <= minimum_margin:
        raise ValueError("full_margin must exceed minimum_margin.")
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    sorted_foot_origin_z = torch.sort(
        asset.data.body_pos_w[:, state.asset_foot_ids, 2], dim=-1
    ).values
    lowest_nonfoot_origin_z = torch.min(
        asset.data.body_pos_w[:, state.asset_nonfoot_ids, 2], dim=-1
    ).values
    clearance_margin = (
        lowest_nonfoot_origin_z - sorted_foot_origin_z[:, foot_rank - 1]
    )
    margin_quality = linear_clearance_scale(
        clearance_margin, minimum_margin, full_margin
    )
    return margin_quality * state.has_landed


def landing_default_action_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    full_rotation: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Drive residual actions toward zero, i.e. the nominal support pose.

    Unlike a joint-position penalty, this term gives the actor an immediate
    gradient before the delayed motors and leg inertia have moved the joints.
    """
    if minimum_rotation <= 0.0 or full_rotation <= minimum_rotation:
        raise ValueError("Landing rotation thresholds must be positive and increasing.")
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    descending_airborne = (
        state.has_taken_off
        & state.airborne
        & ~state.has_touched_down
        & (asset.data.root_lin_vel_w[:, 2] < 0.0)
        & (state.signed_backward_rotation >= minimum_rotation)
    )
    airborne_scale = linear_clearance_scale(
        state.signed_backward_rotation, minimum_rotation, full_rotation
    ) * descending_airborne
    active_scale = torch.where(
        state.has_landed,
        torch.ones_like(airborne_scale),
        airborne_scale,
    )
    return torch.mean(torch.square(env.action_manager.action), dim=-1) * active_scale


def landing_retracted_leg_action_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    full_rotation: float,
    minimum_margin: float,
    full_margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Drive only legs whose feet are still above the landing envelope to default."""
    if minimum_rotation <= 0.0 or full_rotation <= minimum_rotation:
        raise ValueError("Landing rotation thresholds must be positive and increasing.")
    if full_margin <= minimum_margin:
        raise ValueError("full_margin must exceed minimum_margin.")
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    descending_airborne = (
        state.has_taken_off
        & state.airborne
        & ~state.has_touched_down
        & (asset.data.root_lin_vel_w[:, 2] < 0.0)
        & (state.signed_backward_rotation >= minimum_rotation)
    )
    airborne_scale = linear_clearance_scale(
        state.signed_backward_rotation, minimum_rotation, full_rotation
    ) * descending_airborne
    active_scale = torch.where(
        state.has_landed,
        torch.ones_like(airborne_scale),
        airborne_scale,
    )

    foot_origin_z = asset.data.body_pos_w[:, state.asset_foot_ids, 2]
    lowest_nonfoot_origin_z = torch.min(
        asset.data.body_pos_w[:, state.asset_nonfoot_ids, 2], dim=-1
    ).values
    foot_margin = lowest_nonfoot_origin_z.unsqueeze(-1) - foot_origin_z
    deployment_need = 1.0 - linear_clearance_scale(
        foot_margin, minimum_margin, full_margin
    )
    leg_action_l2 = torch.mean(
        torch.square(env.action_manager.action.reshape(env.num_envs, 4, 3)),
        dim=-1,
    )
    return torch.mean(leg_action_l2 * deployment_need, dim=-1) * active_scale


def landing_front_leg_default_action_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    full_rotation: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Directly command the front legs back toward their nominal stance.

    On ToGo_LFs, zero residual action maps to the default standing pose.  This
    bypasses the delayed kinematic credit chain when the front feet are low
    enough to touch but folded behind the center of mass.
    """
    if minimum_rotation <= 0.0 or full_rotation <= minimum_rotation:
        raise ValueError("Landing rotation thresholds must be positive and increasing.")

    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    descending_airborne = (
        state.has_taken_off
        & state.airborne
        & ~state.has_touched_down
        & (asset.data.root_lin_vel_w[:, 2] < 0.0)
        & (state.signed_backward_rotation >= minimum_rotation)
    )
    airborne_scale = linear_clearance_scale(
        state.signed_backward_rotation, minimum_rotation, full_rotation
    ) * descending_airborne
    active_scale = torch.where(
        state.has_landed,
        torch.ones_like(airborne_scale),
        airborne_scale,
    )
    leg_action_l2 = torch.mean(
        torch.square(env.action_manager.action.reshape(env.num_envs, 4, 3)),
        dim=-1,
    )
    return torch.mean(leg_action_l2[:, :2], dim=-1) * active_scale


def post_touchdown_motion_excess_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    maximum_linear_speed: float,
    maximum_angular_speed: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Give non-saturating impact-braking gradients after clean foot contact."""
    if maximum_linear_speed <= 0.0 or maximum_angular_speed <= 0.0:
        raise ValueError("Post-touchdown speed limits must be positive.")
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    linear_speed = torch.linalg.norm(asset.data.root_lin_vel_w, dim=-1)
    angular_speed = torch.linalg.norm(asset.data.root_ang_vel_w, dim=-1)
    linear_excess = torch.clamp(linear_speed - maximum_linear_speed, min=0.0)
    angular_excess = torch.clamp(angular_speed - maximum_angular_speed, min=0.0)
    return (
        torch.square(linear_excess) + torch.square(angular_excess)
    ) * state.has_landed


def post_touchdown_upright_error_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize falling away from upright after a clean first touchdown."""
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    gravity = asset.data.projected_gravity_b
    upright_error = (
        torch.square(gravity[:, 0])
        + torch.square(gravity[:, 1])
        + torch.square(gravity[:, 2] + 1.0)
    )
    return upright_error * state.has_landed


def post_touchdown_joint_pose_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Guide the legs back to the nominal support pose after first contact."""
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    pose_error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.mean(torch.square(pose_error), dim=-1) * state.has_landed


def post_touchdown_nonfoot_contact(
    env: "ManagerBasedRLEnv",
    command_name: str,
) -> torch.Tensor:
    """Continuously mark body/leg collisions after a clean foot-first event."""
    state = _state(env, command_name)
    return state.nonfoot_contact.float() * state.has_landed


def landing_approach(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    target_rotation: float,
    orientation_error_scale: float,
    rotation_error_scale: float,
    angular_velocity_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continuously shape an upright, braked approach before first contact.

    The earlier landing reward only activated after a multi-foot stable state
    already existed. This term bridges that sparse gap while the robot is
    descending near one revolution. Using all three projected-gravity
    components distinguishes upright ``[0, 0, -1]`` from upside down
    ``[0, 0, 1]``.
    """
    if minimum_rotation <= 0.0 or target_rotation <= minimum_rotation:
        raise ValueError("Landing rotation thresholds must be positive and increasing.")
    if (
        orientation_error_scale <= 0.0
        or rotation_error_scale <= 0.0
        or angular_velocity_scale <= 0.0
    ):
        raise ValueError("Landing approach scales must be positive.")

    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    rotation = state.signed_backward_rotation
    descending = asset.data.root_lin_vel_w[:, 2] < 0.0
    active = (
        state.has_taken_off
        & state.airborne
        & ~state.has_touched_down
        & descending
        & (rotation >= minimum_rotation)
    )
    gravity = asset.data.projected_gravity_b
    upright_error = (
        torch.square(gravity[:, 0])
        + torch.square(gravity[:, 1])
        + torch.square(gravity[:, 2] + 1.0)
    )
    rotation_error = torch.square(rotation - target_rotation)
    pitch_rate_l2 = torch.square(asset.data.root_ang_vel_w[:, 1])
    upright_quality = torch.exp(-upright_error / orientation_error_scale)
    rotation_quality = torch.exp(-rotation_error / rotation_error_scale)
    braking_quality = torch.exp(-pitch_rate_l2 / angular_velocity_scale)
    # Keep a useful gradient even while the source policy is still far from
    # upright. Braking only pays in proportion to current one-turn accuracy,
    # so the policy cannot profit by stopping while inverted.
    return (
        0.5 * upright_quality
        + rotation_quality
        + rotation_quality * braking_quality
    ) * active


def landing_leg_extension(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    full_rotation: float,
    minimum_foot_drop: float,
    full_foot_drop: float,
    orientation_error_scale: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reward extending feet along the body's downward axis late in descent."""
    if minimum_rotation <= 0.0 or full_rotation <= minimum_rotation:
        raise ValueError("Landing rotation thresholds must be positive and increasing.")
    if orientation_error_scale <= 0.0:
        raise ValueError("orientation_error_scale must be positive.")

    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    descending = asset.data.root_lin_vel_w[:, 2] < 0.0
    active = (
        state.has_taken_off
        & state.airborne
        & ~state.has_touched_down
        & descending
        & (state.signed_backward_rotation >= minimum_rotation)
    )
    rotation_quality = linear_clearance_scale(
        state.signed_backward_rotation, minimum_rotation, full_rotation
    )
    relative_foot_w = (
        asset.data.body_pos_w[:, asset_cfg.body_ids, :]
        - asset.data.root_pos_w.unsqueeze(1)
    )
    root_rotation_w = math_utils.matrix_from_quat(asset.data.root_quat_w)
    relative_foot_b = torch.einsum(
        "nfi,nij->nfj", relative_foot_w, root_rotation_w
    )
    foot_drop = -relative_foot_b[:, :, 2]
    if full_foot_drop <= minimum_foot_drop:
        raise ValueError("full_foot_drop must exceed minimum_foot_drop.")
    # A sigmoid keeps a usable gradient even for the source policy's deeply
    # tucked pose.  The previous hard clamp returned exactly zero below the
    # minimum and made foot deployment undiscoverable by local PPO updates.
    extension_center = 0.5 * (minimum_foot_drop + full_foot_drop)
    extension_temperature = (full_foot_drop - minimum_foot_drop) / 6.0
    extension_quality = torch.mean(
        torch.sigmoid(
            (foot_drop - extension_center) / extension_temperature
        ),
        dim=-1,
    )
    gravity = asset.data.projected_gravity_b
    upright_error = (
        torch.square(gravity[:, 0])
        + torch.square(gravity[:, 1])
        + torch.square(gravity[:, 2] + 1.0)
    )
    upright_quality = torch.exp(-upright_error / orientation_error_scale)
    # Body-frame extension remains meaningful while pitched. A small upright
    # floor gives PPO credit before the robot is perfectly upright, while the
    # increasing factor still favors completing the revolution before impact.
    upright_blend = 0.25 + 0.75 * upright_quality
    return extension_quality * upright_blend * rotation_quality * active


def landing_foot_fore_aft_stance_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    full_rotation: float,
    front_target_x: float,
    rear_target_x: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Place front and rear feet on opposite sides of the body for landing.

    Vertical foot-clearance rewards alone can be exploited by putting every
    foot behind the center of mass.  The foot order is FL, FR, RL, RR, so this
    term directly restores the nominal fore-aft support polygon in body frame.
    It starts late in the descending rotation and remains active after a clean
    foot-first touchdown so the policy cannot retract the front legs on impact.
    """
    if minimum_rotation <= 0.0 or full_rotation <= minimum_rotation:
        raise ValueError("Landing rotation thresholds must be positive and increasing.")
    if front_target_x <= 0.0 or rear_target_x >= 0.0:
        raise ValueError("Front and rear landing targets must straddle the body origin.")

    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    descending_airborne = (
        state.has_taken_off
        & state.airborne
        & ~state.has_touched_down
        & (asset.data.root_lin_vel_w[:, 2] < 0.0)
        & (state.signed_backward_rotation >= minimum_rotation)
    )
    airborne_scale = linear_clearance_scale(
        state.signed_backward_rotation, minimum_rotation, full_rotation
    ) * descending_airborne
    active_scale = torch.where(
        state.has_landed,
        torch.ones_like(airborne_scale),
        airborne_scale,
    )

    relative_foot_w = (
        asset.data.body_pos_w[:, state.asset_foot_ids, :]
        - asset.data.root_pos_w.unsqueeze(1)
    )
    root_rotation_w = math_utils.matrix_from_quat(asset.data.root_quat_w)
    relative_foot_b = torch.einsum(
        "nfi,nij->nfj", relative_foot_w, root_rotation_w
    )
    target_x = relative_foot_b.new_tensor(
        [front_target_x, front_target_x, rear_target_x, rear_target_x]
    )
    fore_aft_error = torch.mean(
        torch.square(relative_foot_b[:, :, 0] - target_x), dim=-1
    )
    return fore_aft_error * active_scale


def landing_joint_pose_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    full_rotation: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize deviation from the nominal standing joint pose late in descent."""
    if minimum_rotation <= 0.0 or full_rotation <= minimum_rotation:
        raise ValueError("Landing rotation thresholds must be positive and increasing.")

    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    active = (
        state.has_taken_off
        & state.airborne
        & ~state.has_touched_down
        & (asset.data.root_lin_vel_w[:, 2] < 0.0)
        & (state.signed_backward_rotation >= minimum_rotation)
    )
    rotation_quality = linear_clearance_scale(
        state.signed_backward_rotation, minimum_rotation, full_rotation
    )
    pose_error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.mean(torch.square(pose_error), dim=-1) * rotation_quality * active


def landing_foot_origin_clearance(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    full_rotation: float,
    foot_rank: int,
    minimum_margin: float,
    full_margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward multiple foot origins clearing the lowest non-foot body origin."""
    if minimum_rotation <= 0.0 or full_rotation <= minimum_rotation:
        raise ValueError("Landing rotation thresholds must be positive and increasing.")
    if not 1 <= foot_rank <= 4:
        raise ValueError("foot_rank must be in [1, 4].")
    if full_margin <= minimum_margin:
        raise ValueError("full_margin must exceed minimum_margin.")

    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    active = (
        state.has_taken_off
        & state.airborne
        & ~state.has_touched_down
        & (asset.data.root_lin_vel_w[:, 2] < 0.0)
        & (state.signed_backward_rotation >= minimum_rotation)
    )
    rotation_quality = linear_clearance_scale(
        state.signed_backward_rotation, minimum_rotation, full_rotation
    )
    sorted_foot_origin_z = torch.sort(
        asset.data.body_pos_w[:, state.asset_foot_ids, 2], dim=-1
    ).values
    lowest_nonfoot_origin_z = torch.min(
        asset.data.body_pos_w[:, state.asset_nonfoot_ids, 2], dim=-1
    ).values
    clearance_margin = (
        lowest_nonfoot_origin_z - sorted_foot_origin_z[:, foot_rank - 1]
    )
    margin_quality = linear_clearance_scale(
        clearance_margin, minimum_margin, full_margin
    )
    return margin_quality * rotation_quality * active


def landing_stability(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    linear_velocity_scale: float,
    angular_velocity_scale: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward upright, low-motion support after a credible rotation."""
    if linear_velocity_scale <= 0.0 or angular_velocity_scale <= 0.0:
        raise ValueError("Landing velocity scales must be positive.")
    asset: Articulation = env.scene[asset_cfg.name]
    state = _state(env, command_name)
    eligible = state.has_landed & (state.max_backward_rotation >= minimum_rotation)
    gravity = asset.data.projected_gravity_b
    gravity_error = (
        torch.square(gravity[:, 0])
        + torch.square(gravity[:, 1])
        + torch.square(gravity[:, 2] + 1.0)
    )
    linear_speed_l2 = torch.sum(torch.square(asset.data.root_lin_vel_w), dim=-1)
    angular_speed_l2 = torch.sum(torch.square(asset.data.root_ang_vel_w), dim=-1)
    contact_quality = state.contact_count.float() / 4.0
    stability = (
        torch.exp(-gravity_error / 0.15)
        * torch.exp(-linear_speed_l2 / linear_velocity_scale)
        * torch.exp(-angular_speed_l2 / angular_velocity_scale)
        * contact_quality
    )
    return stability * eligible


def landing_rotation_accuracy(
    env: "ManagerBasedRLEnv",
    command_name: str,
    target_rotation: float = 2.0 * math.pi,
    error_scale: float = 0.5,
) -> torch.Tensor:
    """Reward touchdown near one complete rotation, not under/over-rotation."""
    if error_scale <= 0.0:
        raise ValueError("error_scale must be positive.")
    state = _state(env, command_name)
    error = torch.square(state.signed_backward_rotation - target_rotation)
    return torch.exp(-error / error_scale) * state.has_landed


def completed_backflip(
    env: "ManagerBasedRLEnv",
    command_name: str = "backflip_phase",
) -> torch.Tensor:
    """Sparse bonus for satisfying the explicit backflip success criteria."""
    return _state(env, command_name).landing_success.float()


def landing_foot_slip_l2(
    env: "ManagerBasedRLEnv",
    command_name: str,
    minimum_rotation: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 2.0,
) -> torch.Tensor:
    """Penalize horizontal foot speed after the robot has completed its rotation."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    state = _state(env, command_name)
    contact = (
        torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
        >= force_threshold
    )
    slip_l2 = torch.sum(torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]), dim=-1)
    eligible = state.max_backward_rotation >= minimum_rotation
    return torch.sum(slip_l2 * contact, dim=-1) * eligible
