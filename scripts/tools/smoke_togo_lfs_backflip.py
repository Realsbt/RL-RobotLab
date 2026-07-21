"""Create and step a small ToGo_LFs backflip environment."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke-test an LFS backflip task.")
parser.add_argument(
    "--task",
    default="RobotLab-ToGo-LFs-Backflip-Jump-v0",
    choices=(
        "RobotLab-ToGo-LFs-Backflip-Genesis-v0",
        "RobotLab-ToGo-LFs-Backflip-Genesis-Strict-v0",
        "RobotLab-ToGo-LFs-Backflip-Genesis-Strict-Consolidate-v0",
        "RobotLab-ToGo-LFs-Backflip-Genesis-Land-v0",
        "RobotLab-ToGo-LFs-Backflip-Genesis-Land-Energy-v0",
        "RobotLab-ToGo-LFs-Backflip-Genesis-Land-Assisted-v0",
        "RobotLab-ToGo-LFs-Backflip-Jump-v0",
        "RobotLab-ToGo-LFs-Backflip-Early-Rotate-v0",
        "RobotLab-ToGo-LFs-Backflip-Rotate-v0",
        "RobotLab-ToGo-LFs-Backflip-v0",
        "RobotLab-ToGo-LFs-Backflip-Robust-v0",
    ),
)
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--steps", type=int, default=10)
parser.add_argument(
    "--action_std",
    type=float,
    default=0.0,
    help="Standard deviation of independent Gaussian test actions.",
)
parser.add_argument(
    "--disable_self_collisions",
    action="store_true",
    help="Temporarily disable self-collision for an A/B diagnostic run.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab.utils import math as math_utils

import robot_lab.tasks  # noqa: F401


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.log_dir = "/tmp/togo_lfs_backflip_smoke"
    if args_cli.disable_self_collisions:
        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    try:
        observations, _ = env.reset()
        raw_env = env.unwrapped
        action_dim = raw_env.action_manager.total_action_dim
        observation_shapes = {
            name: tuple(value.shape) for name, value in observations.items()
        }
        print(
            f"[BACKFLIP_SMOKE] task={args_cli.task} observations={observation_shapes} "
            f"actions=({raw_env.num_envs}, {action_dim})",
            flush=True,
        )

        action_shape = (raw_env.num_envs, action_dim)
        actions = torch.zeros(*action_shape, device=raw_env.device, dtype=torch.float32)
        reward = torch.zeros(raw_env.num_envs, device=raw_env.device)
        terminated_count = 0
        truncated_count = 0
        for _ in range(args_cli.steps):
            if args_cli.action_std > 0.0:
                actions.normal_(mean=0.0, std=args_cli.action_std)
            observations, reward, terminated, truncated, _ = env.step(actions)
            terminated_count += int(terminated.sum().item())
            truncated_count += int(truncated.sum().item())
            if not torch.isfinite(reward).all():
                raise RuntimeError("Backflip smoke test produced a non-finite reward.")
            for name, value in observations.items():
                if not torch.isfinite(value).all():
                    raise RuntimeError(
                        f"Backflip smoke test produced non-finite '{name}' observations."
                    )

        phase_state = raw_env.command_manager.get_term("backflip_phase")
        robot = raw_env.scene["robot"]
        foot_body_names = [robot.body_names[index] for index in phase_state.asset_foot_ids]
        relative_foot_w = (
            robot.data.body_pos_w[:, phase_state.asset_foot_ids, :]
            - robot.data.root_pos_w.unsqueeze(1)
        )
        root_rotation_w = math_utils.matrix_from_quat(robot.data.root_quat_w)
        relative_foot_b = torch.einsum(
            "nfi,nij->nfj", relative_foot_w, root_rotation_w
        )
        action_joint_ids, action_joint_names = robot.find_joints(
            env_cfg.actions.joint_pos.joint_names,
            preserve_order=env_cfg.actions.joint_pos.preserve_order,
        )
        print(
            f"[BACKFLIP_SMOKE] foot_body_ids={phase_state.asset_foot_ids} "
            f"foot_body_names={foot_body_names} action_joint_ids={action_joint_ids} "
            f"action_joint_names={action_joint_names}",
            flush=True,
        )
        print(
            "[BACKFLIP_SMOKE] final_foot_positions_body_m="
            f"{relative_foot_b[0].detach().cpu().tolist()}",
            flush=True,
        )
        print(
            f"[BACKFLIP_SMOKE] completed_steps={args_cli.steps} "
            f"action_std={args_cli.action_std:.3f} "
            f"terminated={terminated_count} truncated={truncated_count} "
            f"mean_reward={reward.mean().item():.6f} "
            f"max_height={phase_state.max_base_height.max().item():.4f} "
            f"max_rotation={phase_state.max_backward_rotation.max().item():.4f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
