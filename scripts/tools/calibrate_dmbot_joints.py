"""Interactive DMBot joint-position calibration tool.

Focus the terminal for keyboard input while watching the Isaac Sim window.
The script keeps the robot base fixed and sends joint-position actions around
the current default joint pose.
"""

import argparse
import re
import select
import sys
import termios
import time
import tty

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Interactively calibrate DMBot joint_pos values.")
parser.add_argument("--task", type=str, default="RobotLab-DMBot-v0", help="Gym task id to inspect.")
parser.add_argument("--step", type=float, default=0.05, help="Joint increment in radians.")
parser.add_argument("--duration", type=float, default=0.0, help="Seconds to run. Use 0 to run until closed or q is pressed.")
parser.add_argument("--start-joint", type=int, default=0, help="Initial selected joint index.")
parser.add_argument("--no-real-time", action="store_true", help="Do not throttle stepping to wall-clock real time.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import robot_lab.tasks  # noqa: F401


DMBOT_JOINT_NAMES = [
    "Jfl1_hipr", "Jfl2_hipp", "Jfl3_knee",
    "Jfr1_hipr", "Jfr2_hipp", "Jfr3_knee",
    "Jrl1_hipr", "Jrl2_hipp", "Jrl3_knee",
    "Jrr1_hipr", "Jrr2_hipp", "Jrr3_knee",
]


def _disable_event(events_cfg, name: str) -> None:
    if hasattr(events_cfg, name):
        setattr(events_cfg, name, None)


def _disable_term(terms_cfg, name: str) -> None:
    if hasattr(terms_cfg, name):
        setattr(terms_cfg, name, None)


def _configure_calibration_env(env_cfg) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.log_dir = "/tmp/dmbot_joint_calibration"

    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    # Keep the task terrain generator intact. DMBot commands and height scanners
    # read terrain metadata during reset, even when the generator is configured flat.
    if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None

    robot_spawn = env_cfg.scene.robot.spawn
    if getattr(robot_spawn, "articulation_props", None) is not None:
        robot_spawn.articulation_props.fix_root_link = True
    if getattr(robot_spawn, "rigid_props", None) is not None:
        robot_spawn.rigid_props.disable_gravity = True

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

    for term_name in ("illegal_contact", "time_out"):
        _disable_term(env_cfg.terminations, term_name)


def _scale_for_joint(joint_name: str, scale_cfg) -> float:
    if isinstance(scale_cfg, dict):
        for pattern, scale in scale_cfg.items():
            if re.fullmatch(pattern, joint_name):
                return float(scale)
    return float(scale_cfg)


def read_key_nonblocking() -> str | None:
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not ready:
        return None
    return sys.stdin.read(1)


def format_joint_pos(joint_names: list[str], values: torch.Tensor) -> str:
    lines = ["joint_pos={"]
    for joint_name, value in zip(joint_names, values.detach().cpu().tolist(), strict=True):
        lines.append(f'    "{joint_name}": {value:.4f},')
    lines.append("}")
    return "\n".join(lines)


def _print_controls() -> None:
    print(
        "\n[DMBOT_CALIBRATE] Focus this terminal for keys while watching Isaac Sim.\n"
        "  [ / ] : previous joint / next joint\n"
        "  a / d : decrease / increase selected joint\n"
        "  A / D : decrease / increase selected joint by 5x step\n"
        "  r     : reset all joints to default\n"
        "  0     : reset selected joint to default\n"
        "  p     : print current joint_pos={...}\n"
        "  q     : quit\n",
        flush=True,
    )


def _print_status(selected_index: int, joint_names: list[str], target_pos: torch.Tensor) -> None:
    print(
        f"[DMBOT_CALIBRATE] selected={selected_index:02d} "
        f"{joint_names[selected_index]} target={target_pos[selected_index].item():+.4f} rad",
        flush=True,
    )


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    _configure_calibration_env(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        env.unwrapped.sim.set_camera_view(eye=[1.6, -2.3, 1.0], target=[0.0, 0.0, 0.25])
    except Exception as exc:
        print(f"[DMBOT_CALIBRATE] Camera setup skipped: {exc}", flush=True)

    env.reset(seed=0)
    robot = env.unwrapped.scene["robot"]

    joint_ids = [robot.joint_names.index(joint_name) for joint_name in DMBOT_JOINT_NAMES]
    default_pos = robot.data.default_joint_pos[0, joint_ids].detach().clone()
    target_pos = default_pos.clone()
    scales = torch.tensor(
        [_scale_for_joint(joint_name, env_cfg.actions.joint_pos.scale) for joint_name in DMBOT_JOINT_NAMES],
        device=env.unwrapped.device,
        dtype=torch.float32,
    )
    default_pos = default_pos.to(env.unwrapped.device)
    target_pos = target_pos.to(env.unwrapped.device)

    selected_index = max(0, min(args_cli.start_joint, len(DMBOT_JOINT_NAMES) - 1))
    step_dt = env.unwrapped.step_dt
    start_time = time.time()

    _print_controls()
    _print_status(selected_index, DMBOT_JOINT_NAMES, target_pos)
    print(format_joint_pos(DMBOT_JOINT_NAMES, target_pos), flush=True)

    old_terminal_attrs = None
    if sys.stdin.isatty():
        old_terminal_attrs = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    try:
        while simulation_app.is_running():
            elapsed = time.time() - start_time
            if args_cli.duration > 0.0 and elapsed >= args_cli.duration:
                break

            key = read_key_nonblocking()
            if key == "q":
                break
            if key == "[":
                selected_index = (selected_index - 1) % len(DMBOT_JOINT_NAMES)
                _print_status(selected_index, DMBOT_JOINT_NAMES, target_pos)
            elif key == "]":
                selected_index = (selected_index + 1) % len(DMBOT_JOINT_NAMES)
                _print_status(selected_index, DMBOT_JOINT_NAMES, target_pos)
            elif key in ("a", "A", "d", "D"):
                direction = -1.0 if key in ("a", "A") else 1.0
                multiplier = 5.0 if key in ("A", "D") else 1.0
                target_pos[selected_index] += direction * multiplier * args_cli.step
                _print_status(selected_index, DMBOT_JOINT_NAMES, target_pos)
            elif key == "0":
                target_pos[selected_index] = default_pos[selected_index]
                _print_status(selected_index, DMBOT_JOINT_NAMES, target_pos)
            elif key == "r":
                target_pos[:] = default_pos
                print("[DMBOT_CALIBRATE] reset all joints to default", flush=True)
                _print_status(selected_index, DMBOT_JOINT_NAMES, target_pos)
            elif key == "p":
                print(format_joint_pos(DMBOT_JOINT_NAMES, target_pos), flush=True)

            step_start = time.time()
            action = ((target_pos - default_pos) / scales).unsqueeze(0)
            env.step(action)

            if not args_cli.no_real_time:
                sleep_time = step_dt - (time.time() - step_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        if old_terminal_attrs is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_terminal_attrs)
        print(format_joint_pos(DMBOT_JOINT_NAMES, target_pos), flush=True)
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
