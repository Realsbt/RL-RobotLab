"""Open Isaac Sim and hold DMBot in its default pose with zero actions."""

import argparse
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visual zero-action check for a RobotLab task.")
parser.add_argument("--task", type=str, default="RobotLab-DMBot-v0", help="Gym task id to inspect.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--duration", type=float, default=60.0, help="Seconds to keep the GUI running. Use 0 to run until closed.")
parser.add_argument("--flat-plane", action="store_true", help="Use a flat plane instead of the task terrain generator.")
parser.add_argument("--no-real-time", action="store_true", help="Do not throttle stepping to wall-clock real time.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import robot_lab.tasks  # noqa: F401


def _disable_event(events_cfg, name: str) -> None:
    if hasattr(events_cfg, name):
        setattr(events_cfg, name, None)


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.log_dir = "/tmp/dmbot_visual_check"

    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    if args_cli.flat_plane:
        env_cfg.scene.terrain.terrain_type = "plane"
        env_cfg.scene.terrain.terrain_generator = None
        if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None

    for event_name in (
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_rigid_body_material",
        "reset_robot_joints",
        "randomize_actuator_gains",
        "randomize_motor_zero_offset",
        "randomize_push_robot",
        "reset_base",
    ):
        _disable_event(env_cfg.events, event_name)

    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        env.unwrapped.sim.set_camera_view(eye=[2.0, -3.0, 1.2], target=[0.0, 0.0, 0.35])
    except Exception as exc:
        print(f"[VISUAL_CHECK] Camera setup skipped: {exc}", flush=True)

    env.reset()
    action_dim = env.unwrapped.action_manager.total_action_dim
    actions = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)

    print(
        f"[VISUAL_CHECK] task={args_cli.task} num_envs={args_cli.num_envs} "
        f"duration={args_cli.duration}s zero_action_dim={action_dim}",
        flush=True,
    )

    start_time = time.time()
    last_report_time = -5.0
    step_dt = env.unwrapped.step_dt

    while simulation_app.is_running():
        elapsed = time.time() - start_time
        if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
            break

        step_start = time.time()
        env.step(actions)

        if elapsed - last_report_time >= 5.0:
            robot = env.unwrapped.scene["robot"]
            base_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
            base_quat = robot.data.root_quat_w[0].detach().cpu().tolist()
            print(
                f"[VISUAL_CHECK] t={elapsed:.1f}s base_pos={base_pos} root_quat={base_quat}",
                flush=True,
            )
            last_report_time = elapsed

        if not args_cli.no_real_time:
            sleep_time = step_dt - (time.time() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
