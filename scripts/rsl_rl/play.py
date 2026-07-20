# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

# Preload native extensions before isaacsim/Kit modules are imported to avoid
# Windows DLL loader conflicts when Isaac Lab/RSL-RL import them later.
import h5py  # noqa: F401
import tensordict  # noqa: F401

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from utils import export_cts_policy_as_jit, export_cts_policy_as_onnx

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=int(1e9), help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop after this many policy steps. Useful for finite headless evaluation.",
)
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument("--fix_commands", action="store_true", default=False, help="Fix the velocity commands.")
parser.add_argument(
    "--quiet-noise-vis",
    action="store_true",
    default=False,
    help="Visualize the MUTE per-foot impact-noise proxy (not acoustic dB).",
)
parser.add_argument("--no-camera-follow", action="store_true", default=False, help="Disable camera follow during play.")
parser.add_argument(
    "--terrain-level",
    type=str,
    default=None,
    help="Force playback terrain level row, e.g. 9 for highest difficulty, or use --terrain-level all to spread levels.",
)
parser.add_argument(
    "--export-only",
    action="store_true",
    default=False,
    help="Export the checkpoint to TorchScript and ONNX, then exit without entering the simulation loop.",
)
parser.add_argument(
    "--export-name",
    type=str,
    default="policy",
    help="Base filename for exported models, without an extension.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.export_name or os.path.basename(args_cli.export_name) != args_cli.export_name:
    parser.error("--export-name must be a non-empty filename without directory components.")
if args_cli.export_name.endswith((".pt", ".onnx")):
    parser.error("--export-name must not include a .pt or .onnx extension.")
if args_cli.max_steps is not None and args_cli.max_steps <= 0:
    parser.error("--max_steps must be positive.")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import time
import torch
# from scripts.reinforcement_learning.utils import camera_follow
from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerCTS

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
# from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import robot_lab.tasks  # noqa: F401

def fix_commands(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg):
    """Fix commanded velocity to a constant target.

    Prefer locking the command generator to keep policy observations consistent
    with environment internal command state.
    """
    fixed_lin_x, fixed_lin_y, fixed_ang_z = 1.0, 0.0, 0.0

    base_velocity_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if base_velocity_cfg is None:
        return

    # Keep the MUTE command class so its impact diagnostics and visualization remain active.
    if hasattr(base_velocity_cfg, "noise_marker_cfg"):
        base_velocity_cfg.ranges.lin_vel_x = (fixed_lin_x, fixed_lin_x)
        base_velocity_cfg.ranges.lin_vel_y = (fixed_lin_y, fixed_lin_y)
        base_velocity_cfg.ranges.ang_vel_z = (fixed_ang_z, fixed_ang_z)
        base_velocity_cfg.heading_command = False
        base_velocity_cfg.rel_standing_envs = 0.0
        base_velocity_cfg.rel_heading_envs = 0.0
        return

    fixed_cfg = UniformVelocityCommandCfg(
        asset_name=getattr(base_velocity_cfg, "asset_name", "robot"),
        heading_command=False,
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        resampling_time_range=(5.0, 5.0),
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(fixed_lin_x, fixed_lin_x),
            lin_vel_y=(fixed_lin_y, fixed_lin_y),
            ang_vel_z=(fixed_ang_z, fixed_ang_z),
            heading=None,
        ),
        debug_vis=True,
    )
    env_cfg.commands.base_velocity = fixed_cfg

    # terrain_levels_vel_gym expects custom Go2RLGymCommand fields.
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None


def camera_follow(env, env_index: int = 0) -> None:
    """Keep the viewport camera close to the first robot during play."""
    try:
        robot = env.unwrapped.scene["robot"]
        base_pos = robot.data.root_pos_w[env_index].detach().cpu().tolist()
        target = [base_pos[0], base_pos[1], base_pos[2] + 0.15]
        eye = [base_pos[0] - 2.0, base_pos[1] - 2.4, base_pos[2] + 1.0]
        env.unwrapped.sim.set_camera_view(eye=eye, target=target)
    except Exception as exc:
        if not getattr(camera_follow, "_warned", False):
            print(f"[PLAY] Camera follow skipped: {exc}", flush=True)
            camera_follow._warned = True


def camera_overview(env) -> None:
    """Place the viewport camera once so many environments are visible."""
    try:
        scene = env.unwrapped.scene
        env_origins = getattr(scene, "env_origins", None)
        if env_origins is not None:
            origins = env_origins.detach().cpu()
            center_xy = origins[:, :2].mean(dim=0)
            min_xy = origins[:, :2].min(dim=0).values
            max_xy = origins[:, :2].max(dim=0).values
            span = float(torch.max(max_xy - min_xy).item())
            center = [float(center_xy[0]), float(center_xy[1]), 0.0]
        else:
            num_envs = int(getattr(env.unwrapped, "num_envs", 1))
            spacing = float(getattr(getattr(env.unwrapped.cfg, "scene", None), "env_spacing", 1.0))
            cols = max(1, math.ceil(math.sqrt(num_envs)))
            rows = max(1, math.ceil(num_envs / cols))
            center = [(cols - 1) * spacing * 0.5, (rows - 1) * spacing * 0.5, 0.0]
            span = max(cols, rows) * spacing

        distance = max(8.0, span * 1.15)
        eye = [center[0] - distance * 0.6, center[1] - distance * 0.8, distance * 0.65]
        target = [center[0], center[1], center[2]]
        env.unwrapped.sim.set_camera_view(eye=eye, target=target)
        print("[PLAY] Camera follow disabled; overview camera set once.", flush=True)
    except Exception as exc:
        print(f"[PLAY] Overview camera skipped: {exc}", flush=True)


def force_terrain_level(env, terrain_level_spec: str) -> None:
    """Force or spread playback environments across terrain difficulty levels."""
    terrain = getattr(env.unwrapped.scene, "terrain", None)
    if terrain is None or getattr(terrain, "terrain_origins", None) is None:
        print("[PLAY] Terrain level override skipped: no curriculum terrain origins found.", flush=True)
        return

    max_level = int(terrain.terrain_origins.shape[0] - 1)
    if terrain_level_spec == "all":
        env_ids = torch.arange(terrain.terrain_levels.numel(), device=terrain.terrain_levels.device)
        terrain.terrain_levels[:] = env_ids % (max_level + 1)
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        print(f"[PLAY] Spread playback across terrain levels 0-{max_level}.", flush=True)
        return

    terrain_level = int(terrain_level_spec)
    if terrain_level < 0 or terrain_level > max_level:
        raise ValueError(f"--terrain-level must be in [0, {max_level}], got {terrain_level}.")

    terrain.terrain_levels[:] = terrain_level
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    print(f"[PLAY] Forced playback terrain level to {terrain_level}/{max_level}.", flush=True)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Disable playback-only noise and perturbations when those terms exist. Not
    # every task is a locomotion task, so do not assume velocity curricula or
    # push events are present.
    policy_obs_cfg = getattr(getattr(env_cfg, "observations", None), "policy", None)
    if policy_obs_cfg is not None and hasattr(policy_obs_cfg, "enable_corruption"):
        policy_obs_cfg.enable_corruption = False

    event_cfg = getattr(env_cfg, "events", None)
    if event_cfg is not None:
        for event_name in ("randomize_apply_external_force_torque", "randomize_push_robot"):
            if hasattr(event_cfg, event_name):
                setattr(event_cfg, event_name, None)

    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    if curriculum_cfg is not None:
        for curriculum_name in ("command_levels_lin_vel", "command_levels_ang_vel"):
            if hasattr(curriculum_cfg, curriculum_name):
                setattr(curriculum_cfg, curriculum_name, None)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        # resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        # if not resume_path:
        #     print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
        #     return
        raise NotImplementedError("Pre-trained checkpoint retrieval is disabled temporarily.")
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # fix velocity commands if specified
    if args_cli.fix_commands:
        fix_commands(env_cfg)
    if args_cli.quiet_noise_vis:
        base_velocity_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
        if base_velocity_cfg is None or not hasattr(base_velocity_cfg, "noise_marker_cfg"):
            raise ValueError("--quiet-noise-vis requires a task using MuteVelocityCommandCfg.")
        base_velocity_cfg.debug_vis = True
        if hasattr(base_velocity_cfg, "impact_event_marker_cfg"):
            print(
                "[PLAY] Event-only touchdown visualization enabled: green=low, amber=medium, red=high; not dB.",
                flush=True,
            )
        else:
            print(
                "[PLAY] MUTE impact proxy visualization enabled: green=quiet, amber=medium, red=high; not dB.",
                flush=True,
            )
    if args_cli.terrain_level is not None and hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum.terrain_levels = None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.terrain_level is not None:
        force_terrain_level(env, args_cli.terrain_level)
        env.reset()

    # wrap for video recording
    if args_cli.video:
        import imageio
        video_path = os.path.join(log_dir, "videos", "play", time.strftime("%Y-%m-%d_%H-%M-%S") + ".mp4")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        writer = imageio.get_writer(video_path, fps=int(1/env.unwrapped.step_dt))
        # video_kwargs = {
        #     "video_folder": os.path.join(log_dir, "videos", "play"),
        #     "step_trigger": lambda step: step == 0,
        #     "video_length": args_cli.video_length,
        #     "disable_logger": True,
        # }
        # print("[INFO] Recording videos during training.")
        # print_dict(video_kwargs, nesting=4)
        # env = gym.wrappers.RecordVideo(env, **video_kwargs)  # Store all frames into list is slow and memory-consuming, use imageio

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerCTS":
        runner = OnPolicyRunnerCTS(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if agent_cfg.class_name == "OnPolicyRunnerCTS":
        export_cts_policy_as_jit(policy_nn, actor_obs_normalizer=policy_nn.actor_obs_normalizer, single_obs_normalizer=policy_nn.single_obs_normalizer, path=export_model_dir, filename=f"{args_cli.export_name}.pt")
        export_cts_policy_as_onnx(policy_nn, actor_obs_normalizer=policy_nn.actor_obs_normalizer, single_obs_normalizer=policy_nn.single_obs_normalizer, path=export_model_dir, filename=f"{args_cli.export_name}.onnx")
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename=f"{args_cli.export_name}.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename=f"{args_cli.export_name}.onnx")

    print(f"[INFO]: Exported TorchScript policy: {os.path.join(export_model_dir, args_cli.export_name + '.pt')}")
    print(f"[INFO]: Exported ONNX policy: {os.path.join(export_model_dir, args_cli.export_name + '.onnx')}")
    if args_cli.export_only:
        env.close()
        return

    dt = env.unwrapped.step_dt

    # env.unwrapped.eye = (1.1, 3.3, 0.9)
    # reset environment
    obs = env.get_observations()
    if args_cli.no_camera_follow:
        camera_overview(env)
    else:
        camera_follow(env)
    timestep = 0
    completed_episodes = 0
    episode_log_sums: dict[str, float] = {}
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, extras = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        done_count = int(dones.sum().item())
        if done_count > 0:
            completed_episodes += done_count
            episode_log = extras.get("log", {}) if isinstance(extras, dict) else {}
            for key, value in episode_log.items():
                if not key.startswith(("Metrics/", "Episode_Termination/")):
                    continue
                try:
                    scalar = float(value.detach().mean().item()) if torch.is_tensor(value) else float(value)
                except (TypeError, ValueError):
                    continue
                episode_log_sums[key] = episode_log_sums.get(key, 0.0) + scalar * done_count
        if args_cli.video:
            writer.append_data(env.env.render())
        if not args_cli.no_camera_follow:
            camera_follow(env)

        timestep += 1
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if args_cli.max_steps is not None:
        print(
            f"[PLAY_EVAL] steps={timestep} completed_episodes={completed_episodes}",
            flush=True,
        )
        if completed_episodes > 0:
            for key in sorted(episode_log_sums):
                print(
                    f"[PLAY_EVAL] {key}={episode_log_sums[key] / completed_episodes:.6f}",
                    flush=True,
                )

    # close the simulator
    env.close()

    # close the video writer
    if args_cli.video:
        writer.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
