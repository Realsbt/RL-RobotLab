"""Measure fixed-command speed tracking on one Isaac Lab training terrain."""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py  # noqa: F401
import tensordict  # noqa: F401

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Benchmark a checkpoint on a selected training terrain.")
parser.add_argument("--task", type=str, default="RobotLab-ToGo-LFs-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--terrain", type=str, default="stairs_up")
parser.add_argument("--terrain-level", type=int, default=5)
parser.add_argument("--command-speed", type=float, default=1.0)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--warmup-s", type=float, default=0.5)
parser.add_argument("--eval-s", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if not args_cli.checkpoint:
    parser.error("--checkpoint is required.")
if args_cli.num_envs < 4 or args_cli.num_envs % 4 != 0:
    parser.error("MoE-CTS evaluation requires --num_envs to be a positive multiple of 4.")
if args_cli.warmup_s < 0.0 or args_cli.eval_s <= 0.0:
    parser.error("--warmup-s must be non-negative and --eval-s must be positive.")

sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunnerCTS

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from robot_lab.tasks.dmbot.mdp.utils import _get_terrain_column_range
import robot_lab.tasks  # noqa: F401


def _disable_randomization(env_cfg) -> None:
    for event_name in (
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_rigid_body_material",
        "randomize_actuator_gains",
        "randomize_motor_zero_offset",
        "randomize_apply_external_force_torque",
        "randomize_push_robot",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)


def _configure_env(env_cfg) -> None:
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.warmup_s + args_cli.eval_s + 1.0)
    env_cfg.curriculum.terrain_levels = None
    _disable_randomization(env_cfg)

    reset_joints = getattr(env_cfg.events, "reset_robot_joints", None)
    if reset_joints is not None:
        reset_joints.params["position_range"] = (1.0, 1.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)
    reset_base = getattr(env_cfg.events, "reset_base", None)
    if reset_base is not None:
        reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }


def _assign_terrain(raw_env) -> None:
    terrain = raw_env.scene.terrain
    terrain_cfg = terrain.cfg.terrain_generator
    if terrain_cfg is None or terrain.terrain_origins is None:
        raise RuntimeError("The selected task does not use generated curriculum terrain.")
    if args_cli.terrain_level < 0 or args_cli.terrain_level >= terrain.terrain_origins.shape[0]:
        raise ValueError(f"terrain level must be in [0, {terrain.terrain_origins.shape[0] - 1}]")

    column_range = _get_terrain_column_range(terrain_cfg, args_cli.terrain, raw_env.device)
    if column_range is None:
        raise ValueError(f"Unknown terrain {args_cli.terrain!r}; options: {list(terrain_cfg.sub_terrains)}")
    column_start, column_end = column_range
    env_ids = torch.arange(raw_env.num_envs, device=raw_env.device)
    terrain.terrain_levels[:] = args_cli.terrain_level
    terrain.terrain_types[:] = column_start + env_ids % (column_end - column_start)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]

    command_term = raw_env.command_manager.get_term("base_velocity")
    command_term.terrain_idxs[:] = command_term.terrain_type2idx[args_cli.terrain]
    raw_env.reset()


def _set_command(raw_env) -> None:
    command = raw_env.command_manager.get_term("base_velocity").commands
    command[:, 0] = args_cli.command_speed
    command[:, 1:] = 0.0


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    _configure_env(env_cfg)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.device = env_cfg.sim.device
    agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict["robogauge"] = {"enabled": False}

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = gym_env.unwrapped
    _assign_terrain(raw_env)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerCTS(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    checkpoint = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=raw_env.device)
    policy_module = runner.alg.policy

    def step_once():
        _set_command(raw_env)
        obs = env.get_observations()
        with torch.no_grad():
            actions = policy(obs)
            _, _, dones, _ = env.step(actions)
        policy_module.reset(dones)
        return dones

    for _ in range(round(args_cli.warmup_s / raw_env.step_dt)):
        step_once()

    speed_sum = torch.zeros((), device=raw_env.device)
    abs_error_sum = torch.zeros((), device=raw_env.device)
    squared_error_sum = torch.zeros((), device=raw_env.device)
    within_10pct_sum = torch.zeros((), device=raw_env.device)
    fall_count = torch.zeros((), device=raw_env.device)
    eval_steps = round(args_cli.eval_s / raw_env.step_dt)
    for _ in range(eval_steps):
        dones = step_once()
        speed = raw_env.scene["robot"].data.root_lin_vel_b[:, 0]
        error = speed - args_cli.command_speed
        speed_sum += speed.sum()
        abs_error_sum += error.abs().sum()
        squared_error_sum += error.square().sum()
        within_10pct_sum += (error.abs() <= max(0.1, abs(args_cli.command_speed) * 0.1)).sum()
        fall_count += dones.sum()

    sample_count = args_cli.num_envs * eval_steps
    result = {
        "checkpoint": checkpoint,
        "terrain": args_cli.terrain,
        "terrain_level": args_cli.terrain_level,
        "command_speed_m_s": args_cli.command_speed,
        "actual_forward_speed_m_s": (speed_sum / sample_count).item(),
        "tracking_abs_error_m_s": (abs_error_sum / sample_count).item(),
        "tracking_rmse_m_s": torch.sqrt(squared_error_sum / sample_count).item(),
        "samples_within_10pct_ratio": (within_10pct_sum / sample_count).item(),
        "falls_total": int(fall_count.item()),
        "falls_per_env_min": (fall_count / (args_cli.num_envs * args_cli.eval_s / 60.0)).item(),
        "num_envs": args_cli.num_envs,
        "warmup_s": args_cli.warmup_s,
        "eval_s": args_cli.eval_s,
        "domain_randomization": False,
    }
    print(json.dumps(result, indent=2))

    if args_cli.output:
        output = Path(args_cli.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Benchmark report: {output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
