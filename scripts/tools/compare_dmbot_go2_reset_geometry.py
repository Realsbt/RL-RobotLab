"""Compare Go2 and DMBot reset foot geometry in each robot base frame."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[2]
JSON_PREFIX = "[RESET_GEOMETRY_JSON]"
FOOT_ORDER = ("FL", "FR", "RL", "RR")


parser = argparse.ArgumentParser(description="Compare Go2 and DMBot reset foot positions in base coordinates.")
parser.add_argument("--go2-task", type=str, default="RobotLab-Go2-v0")
parser.add_argument("--dmbot-task", type=str, default="RobotLab-DMBot-v0")
parser.add_argument("--log-dir", type=str, default="/tmp/dmbot_go2_reset_geometry")
parser.add_argument("--child-timeout", type=float, default=180.0)
parser.add_argument("--robot", choices=("go2", "dmbot"), default=None, help=argparse.SUPPRESS)
parser.add_argument("--emit-json", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()


@dataclass(frozen=True)
class RobotSpec:
    task: str
    base: str
    feet: dict[str, str]


GO2_SPEC = RobotSpec(
    task="RobotLab-Go2-v0",
    base="base",
    feet={
        "FL": "FL_foot",
        "FR": "FR_foot",
        "RL": "RL_foot",
        "RR": "RR_foot",
    },
)

DMBOT_SPEC = RobotSpec(
    task="RobotLab-DMBot-v0",
    base="L0_torso",
    feet={
        "FL": "Lfl4_foot",
        "FR": "Lfr4_foot",
        "RL": "Lrl4_foot",
        "RR": "Lrr4_foot",
    },
)


def disable_randomization(env_cfg) -> None:
    """Disable reset/startup randomization so both robots are inspected at nominal reset."""
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    if hasattr(env_cfg.scene, "terrain"):
        env_cfg.scene.terrain.max_init_terrain_level = 0

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
        "randomize_apply_external_force_torque",
        "reset_base",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)


def _resolve_spec(robot_key: str) -> RobotSpec:
    if robot_key == "go2":
        return RobotSpec(args_cli.go2_task, GO2_SPEC.base, GO2_SPEC.feet)
    if robot_key == "dmbot":
        return RobotSpec(args_cli.dmbot_task, DMBOT_SPEC.base, DMBOT_SPEC.feet)
    raise ValueError(f"Unsupported robot key: {robot_key}")


def _make_env(task: str):
    import gymnasium as gym

    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import robot_lab.tasks  # noqa: F401

    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.log_dir = args_cli.log_dir
    disable_randomization(env_cfg)
    return gym.make(task, cfg=env_cfg)


def _body_index(body_names: list[str], body_name: str) -> int:
    try:
        return body_names.index(body_name)
    except ValueError as exc:
        available = ", ".join(body_names)
        raise RuntimeError(f"Body '{body_name}' not found. Available bodies: {available}") from exc


def _body_link_state(robot):
    if hasattr(robot.data, "body_link_state_w"):
        return robot.data.body_link_state_w[0].detach().clone()

    import torch

    body_pos_w = robot.data.body_link_pos_w[0]
    body_quat_w = robot.data.body_link_quat_w[0]
    return torch.cat((body_pos_w, body_quat_w), dim=-1).detach().clone()


def collect_reset_geometry(spec: RobotSpec) -> dict[str, list[float]]:
    from isaaclab.utils.math import quat_apply_inverse

    env = _make_env(spec.task)
    try:
        env.reset(seed=0)
        robot = env.unwrapped.scene["robot"]
        body_names = list(robot.body_names)
        body_link_state_w = _body_link_state(robot)

        base_index = _body_index(body_names, spec.base)
        base_pos_w = body_link_state_w[base_index, 0:3]
        base_quat_w = body_link_state_w[base_index, 3:7]

        foot_positions_b = {}
        for foot_label in FOOT_ORDER:
            foot_index = _body_index(body_names, spec.feet[foot_label])
            foot_pos_w = body_link_state_w[foot_index, 0:3]
            foot_delta_w = foot_pos_w - base_pos_w
            foot_pos_b = quat_apply_inverse(base_quat_w.unsqueeze(0), foot_delta_w.unsqueeze(0))[0]
            foot_positions_b[foot_label] = [float(value) for value in foot_pos_b.detach().cpu().tolist()]

        return foot_positions_b
    finally:
        env.close()


def _run_child(robot_key: str) -> dict[str, list[float]]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--robot",
        robot_key,
        "--emit-json",
        "--go2-task",
        args_cli.go2_task,
        "--dmbot-task",
        args_cli.dmbot_task,
        "--log-dir",
        args_cli.log_dir,
    ]
    if getattr(args_cli, "headless", False):
        cmd.append("--headless")

    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=args_cli.child_timeout,
        check=False,
    )
    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)

    for line in reversed(output.splitlines()):
        if line.startswith(JSON_PREFIX):
            return json.loads(line.removeprefix(JSON_PREFIX))

    tail = "\n".join(output.splitlines()[-80:])
    if completed.returncode != 0:
        raise RuntimeError(f"{robot_key} child process failed with exit code {completed.returncode}:\n{tail}")
    raise RuntimeError(f"{robot_key} child process did not emit reset geometry JSON:\n{tail}")


def _stance_summary(foot_positions_b: dict[str, list[float]]) -> dict[str, float]:
    front_x = 0.5 * (foot_positions_b["FL"][0] + foot_positions_b["FR"][0])
    rear_x = 0.5 * (foot_positions_b["RL"][0] + foot_positions_b["RR"][0])
    left_y = 0.5 * (foot_positions_b["FL"][1] + foot_positions_b["RL"][1])
    right_y = 0.5 * (foot_positions_b["FR"][1] + foot_positions_b["RR"][1])
    foot_z = sum(foot_positions_b[label][2] for label in FOOT_ORDER) / len(FOOT_ORDER)
    return {
        "front_x": front_x,
        "rear_x": rear_x,
        "length": front_x - rear_x,
        "left_y": left_y,
        "right_y": right_y,
        "width": left_y - right_y,
        "foot_z": foot_z,
    }


def _fmt_xyz(values: list[float]) -> str:
    return f"({values[0]: .4f}, {values[1]: .4f}, {values[2]: .4f})"


def print_comparison(go2_feet: dict[str, list[float]], dmbot_feet: dict[str, list[float]]) -> None:
    print("[RESET_GEOMETRY] Foot positions in base frame, meters: x forward, y left, z up", flush=True)
    print("foot | go2_xyz                 | dmbot_xyz               | dmbot_minus_go2", flush=True)
    print("-----+-------------------------+-------------------------+----------------", flush=True)

    deltas = []
    for foot_label in FOOT_ORDER:
        go2_xyz = go2_feet[foot_label]
        dmbot_xyz = dmbot_feet[foot_label]
        delta_xyz = [dmbot_xyz[index] - go2_xyz[index] for index in range(3)]
        deltas.append([abs(value) for value in delta_xyz])
        print(
            f"{foot_label:<4} | {_fmt_xyz(go2_xyz)} | "
            f"{_fmt_xyz(dmbot_xyz)} | {_fmt_xyz(delta_xyz)}",
            flush=True,
        )

    mean_abs_delta = [sum(delta[index] for delta in deltas) / len(deltas) for index in range(3)]
    print(
        "[RESET_GEOMETRY] mean_abs_delta xyz="
        f"{_fmt_xyz(mean_abs_delta)}",
        flush=True,
    )

    go2_summary = _stance_summary(go2_feet)
    dmbot_summary = _stance_summary(dmbot_feet)
    print("[RESET_GEOMETRY] stance summary, meters", flush=True)
    for key in ("front_x", "rear_x", "length", "left_y", "right_y", "width", "foot_z"):
        print(
            f"  {key:<8} go2={go2_summary[key]: .4f} "
            f"dmbot={dmbot_summary[key]: .4f} delta={dmbot_summary[key] - go2_summary[key]: .4f}",
            flush=True,
        )


def _child_main() -> None:
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    try:
        geometry = collect_reset_geometry(_resolve_spec(args_cli.robot))
        if args_cli.emit_json:
            print(JSON_PREFIX + json.dumps(geometry, sort_keys=True, separators=(",", ":")), flush=True)
        else:
            for foot_label in FOOT_ORDER:
                print(f"{foot_label}: {_fmt_xyz(geometry[foot_label])}", flush=True)
    finally:
        sys.stdout.flush()
        simulation_app.close()


def main() -> None:
    if args_cli.robot is not None:
        _child_main()
        return

    go2_feet = _run_child("go2")
    dmbot_feet = _run_child("dmbot")
    print_comparison(go2_feet, dmbot_feet)


if __name__ == "__main__":
    main()
