# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Benchmark two MoE-CTS checkpoints with event-based touchdown diagnostics."""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import h5py  # noqa: F401
import tensordict  # noqa: F401

from isaaclab.app import AppLauncher

import cli_args  # isort: skip
from quiet_impact_report import compare_results


parser = argparse.ArgumentParser(description="Compare two checkpoints with quiet-impact event metrics.")
parser.add_argument("--task", type=str, default="RobotLab-ToGo-LFs-Quiet-Impact-v1")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--baseline_checkpoint", type=str, required=True)
parser.add_argument("--candidate_checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--speeds", type=float, nargs="+", default=[0.2, 0.4])
parser.add_argument("--warmup_s", type=float, default=2.0)
parser.add_argument("--eval_s", type=float, default=10.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default=None)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.num_envs < 2:
    parser.error("--num_envs must be at least 2 for the MoE-CTS runner.")
if args_cli.warmup_s < 0.0 or args_cli.eval_s <= 0.0:
    parser.error("--warmup_s must be non-negative and --eval_s must be positive.")
if any(speed < 0.0 for speed in args_cli.speeds):
    parser.error("--speeds must contain non-negative forward speeds.")

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunnerCTS

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401


def _set_range_to_value(range_cfg, name: str, value: float):
    setattr(range_cfg, name, (value, value))


def _configure_benchmark_env(env_cfg, first_speed: float):
    """Remove random confounders while preserving the trained physics model."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.episode_length_s = args_cli.warmup_s + args_cli.eval_s + 5.0
    env_cfg.observations.policy.enable_corruption = False

    command_cfg = env_cfg.commands.base_velocity
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.debug_vis = False
    command_cfg.resampling_time_range = (1.0e9, 1.0e9)
    _set_range_to_value(command_cfg.ranges, "lin_vel_x", first_speed)
    _set_range_to_value(command_cfg.ranges, "lin_vel_y", 0.0)
    _set_range_to_value(command_cfg.ranges, "ang_vel_z", 0.0)

    for event_name in (
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_rigid_body_material",
        "randomize_actuator_gains",
        "randomize_motor_zero_offset",
        "randomize_push_robot",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)

    if getattr(env_cfg.events, "reset_robot_joints", None) is not None:
        env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    if getattr(env_cfg.events, "reset_base", None) is not None:
        env_cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        env_cfg.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }

    env_cfg.terminations.time_out = None
    env_cfg.terminations.illegal_contact = None


def _load_policy(env, agent_cfg_dict: dict, checkpoint: str):
    runner = OnPolicyRunnerCTS(env, agent_cfg_dict, log_dir=None, device=agent_cfg_dict["device"])
    path = os.path.abspath(os.path.expanduser(checkpoint))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    runner.load(path, load_optimizer=False)
    return runner.get_inference_policy(device=env.unwrapped.device), runner.alg.policy, path


def _set_fixed_command(raw_env, speed: float):
    command = raw_env.command_manager.get_term("base_velocity")
    _set_range_to_value(command.cfg.ranges, "lin_vel_x", speed)
    _set_range_to_value(command.cfg.ranges, "lin_vel_y", 0.0)
    _set_range_to_value(command.cfg.ranges, "ang_vel_z", 0.0)


def _evaluate_policy(env, initial_state, policy, policy_module, speed: float) -> dict[str, float]:
    raw_env = env.unwrapped
    _set_fixed_command(raw_env, speed)
    raw_env.reset_to(initial_state, env_ids=None, seed=args_cli.seed, is_relative=True)
    obs = env.get_observations()
    policy_module.reset()

    warmup_steps = round(args_cli.warmup_s / raw_env.step_dt)
    eval_steps = round(args_cli.eval_s / raw_env.step_dt)
    # Isaac Lab mutates cached state tensors during the next reset. Using
    # inference_mode would turn those caches into immutable inference tensors.
    with torch.no_grad():
        for _ in range(warmup_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_module.reset(dones)

        impact_command = raw_env.command_manager.get_term("base_velocity")
        impact_command.clear_impact_statistics()
        velocity_sum = torch.zeros((), device=raw_env.device)
        tracking_error_sum = torch.zeros((), device=raw_env.device)

        for _ in range(eval_steps):
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_module.reset(dones)
            forward_velocity = raw_env.scene["robot"].data.root_lin_vel_b[:, 0]
            velocity_sum += torch.sum(forward_velocity)
            tracking_error_sum += torch.sum(torch.abs(forward_velocity - speed))

    sample_count = args_cli.num_envs * eval_steps
    result = impact_command.get_impact_statistics()
    result.update(
        {
            "command_speed_m_s": speed,
            "actual_forward_speed_m_s": (velocity_sum / sample_count).item(),
            "tracking_abs_error_m_s": (tracking_error_sum / sample_count).item(),
            "event_rate_hz_per_env": result["event_count"] / (args_cli.num_envs * args_cli.eval_s),
        }
    )
    return result


def _print_comparison(speed: float, baseline: dict, candidate: dict, reductions: dict):
    print(f"\n=== Command speed {speed:.2f} m/s ===")
    print(f"{'Metric':36s} {'Baseline':>12s} {'Candidate':>12s} {'Reduction':>11s}")
    print("-" * 75)
    metrics = (
        "actual_forward_speed_m_s",
        "tracking_abs_error_m_s",
        "event_rate_hz_per_env",
        "preimpact_speed_m_s",
        "peak_force_n",
        "peak_force_rise_rate_n_s",
        "impulse_n_s",
        "force_variation_energy_n2_s",
        "impact_score",
        "peak_impact_score",
        "slip_power_proxy_w",
    )
    for metric in metrics:
        reduction = reductions.get(metric)
        reduction_text = "n/a" if reduction is None else f"{reduction:+.2f}%"
        print(f"{metric:36s} {baseline[metric]:12.5f} {candidate[metric]:12.5f} {reduction_text:>11s}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlBaseRunnerCfg):
    first_speed = args_cli.speeds[0]
    _configure_benchmark_env(env_cfg, first_speed)
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg.device = env_cfg.sim.device
    agent_cfg_dict = agent_cfg.to_dict()
    agent_cfg_dict["robogauge"] = {"enabled": False}

    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    initial_state = gym_env.unwrapped.scene.get_state(is_relative=True)
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

    baseline_policy, baseline_module, baseline_path = _load_policy(
        env, copy.deepcopy(agent_cfg_dict), args_cli.baseline_checkpoint
    )
    candidate_policy, candidate_module, candidate_path = _load_policy(
        env, copy.deepcopy(agent_cfg_dict), args_cli.candidate_checkpoint
    )

    report = {
        "protocol": {
            "task": args_cli.task,
            "num_envs": args_cli.num_envs,
            "speeds_m_s": args_cli.speeds,
            "warmup_s": args_cli.warmup_s,
            "eval_s": args_cli.eval_s,
            "seed": args_cli.seed,
            "contact_on_force_n": 1.0,
            "contact_off_force_n": 0.5,
            "impact_window_s": 0.05,
            "domain_randomization": False,
        },
        "baseline_checkpoint": baseline_path,
        "candidate_checkpoint": candidate_path,
        "results": [],
    }

    for speed in args_cli.speeds:
        baseline = _evaluate_policy(env, initial_state, baseline_policy, baseline_module, speed)
        candidate = _evaluate_policy(env, initial_state, candidate_policy, candidate_module, speed)
        reductions = compare_results(baseline, candidate)
        report["results"].append(
            {"command_speed_m_s": speed, "baseline": baseline, "candidate": candidate, "reduction_pct": reductions}
        )
        _print_comparison(speed, baseline, candidate, reductions)

    output_path = args_cli.output
    if output_path is None:
        output_path = f"logs/benchmarks/quiet_impact_{time.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    output_path = os.path.abspath(os.path.expanduser(output_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nBenchmark report: {output_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
