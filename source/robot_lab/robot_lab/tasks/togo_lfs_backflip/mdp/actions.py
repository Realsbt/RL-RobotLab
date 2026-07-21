"""Phase-aware action transforms for the ToGo_LFs backflip landing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class TerminalFrontStanceAction(JointPositionAction):
    """Blend front-leg targets to nominal stance near the end of one rotation.

    PPO still supplies all twelve actions.  Only after the tracked backflip has
    nearly completed a revolution are the first six (FL/FR) processed joint
    targets smoothly blended toward their default offsets.  Keeping this
    deterministic terminal transform outside the shared actor prevents a
    landing-only update from changing the already successful launch policy.
    """

    cfg: "TerminalFrontStanceActionCfg"

    def __init__(self, cfg: "TerminalFrontStanceActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        if self.action_dim != 12:
            raise ValueError(
                f"Terminal front stance requires 12 ordered leg actions, got {self.action_dim}."
            )
        expected_front_prefixes = ("Jfl", "Jfl", "Jfl", "Jfr", "Jfr", "Jfr")
        if any(
            not joint_name.startswith(prefix)
            for joint_name, prefix in zip(self._joint_names[:6], expected_front_prefixes)
        ):
            raise ValueError(
                "Terminal front stance requires FL then FR as the first six actions, "
                f"resolved {self._joint_names}."
            )
        if len(self.cfg.front_target_offsets) != 6:
            raise ValueError("front_target_offsets must contain FL/FR's six joint offsets.")
        if len(self.cfg.post_touchdown_all_target_offsets) != 12:
            raise ValueError(
                "post_touchdown_all_target_offsets must contain all twelve joint offsets."
            )
        if self.cfg.post_touchdown_blend_duration_s <= 0.0:
            raise ValueError("post_touchdown_blend_duration_s must be positive.")
        self._front_target_offsets = torch.tensor(
            self.cfg.front_target_offsets,
            device=self.device,
            dtype=self._processed_actions.dtype,
        ).unsqueeze(0)
        self._post_touchdown_all_target_offsets = torch.tensor(
            self.cfg.post_touchdown_all_target_offsets,
            device=self.device,
            dtype=self._processed_actions.dtype,
        ).unsqueeze(0)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        state = self._env.command_manager.get_term(self.cfg.command_name)
        rotation_span = self.cfg.full_rotation - self.cfg.minimum_rotation
        blend = torch.clamp(
            (state.signed_backward_rotation - self.cfg.minimum_rotation)
            / rotation_span,
            min=0.0,
            max=1.0,
        )
        descending = self._asset.data.root_lin_vel_w[:, 2] < 0.0
        blend = blend * state.has_taken_off * descending
        blend = torch.where(state.has_landed, torch.ones_like(blend), blend)

        if not isinstance(self._offset, torch.Tensor):
            raise TypeError("Terminal front stance requires tensor default joint offsets.")
        front_blend = blend.unsqueeze(-1)
        front_target = self._offset[:, :6] + self._front_target_offsets
        self._processed_actions[:, :6] = torch.lerp(
            self._processed_actions[:, :6],
            front_target,
            front_blend,
        )
        support_target = self._offset + self._post_touchdown_all_target_offsets
        elapsed_s = self._env.episode_length_buf * self._env.step_dt
        support_blend = torch.clamp(
            (elapsed_s - state.first_touchdown_time_s)
            / self.cfg.post_touchdown_blend_duration_s,
            min=0.0,
            max=1.0,
        ) * state.has_landed
        self._processed_actions[:] = torch.lerp(
            self._processed_actions,
            support_target,
            support_blend.unsqueeze(-1),
        )


@configclass
class TerminalFrontStanceActionCfg(JointPositionActionCfg):
    """Configuration for the phase-gated front-leg stance transform."""

    class_type: type[ActionTerm] = TerminalFrontStanceAction
    command_name: str = "backflip_phase"
    minimum_rotation: float = 1.52 * math.pi
    full_rotation: float = 1.75 * math.pi
    front_target_offsets: tuple[float, ...] = (
        0.0,
        -0.45,
        -0.40,
        0.0,
        0.45,
        0.40,
    )
    post_touchdown_all_target_offsets: tuple[float, ...] = (0.0,) * 12
    post_touchdown_blend_duration_s: float = 0.30

    def __post_init__(self):
        if self.minimum_rotation <= 0.0:
            raise ValueError("minimum_rotation must be positive.")
        if self.full_rotation <= self.minimum_rotation:
            raise ValueError("full_rotation must exceed minimum_rotation.")
