"""Bake the DMBot reset pose into static USD body transforms."""

from pathlib import Path
import argparse
import sys

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Gf, Usd, UsdGeom

import gymnasium as gym
import torch

from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import robot_lab.tasks  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGET_USD = ROOT / "resources/Robots/dm/OpenDog_novlnk_stance/OpenDog_novlnk.usd"
DEFAULT_SOURCE_USD = ROOT / "resources/Robots/dm/OpenDog_novlnk/OpenDog_novlnk.usd"
ROOT_PRIM_PATH = "/DaMiao_OpenDog_novlnk"
BASE_LINK_NAME = "L0_torso"


def _disable_event(events_cfg, name: str) -> None:
    if hasattr(events_cfg, name):
        setattr(events_cfg, name, None)


def _make_env(task: str, source_usd: Path):
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.log_dir = "/tmp/dmbot_usd_stance_bake"
    env_cfg.scene.robot.spawn.usd_path = str(source_usd)

    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

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

    return gym.make(task, cfg=env_cfg)


def collect_reset_body_transforms(task: str, source_usd: Path) -> dict[str, tuple[list[float], list[float]]]:
    env = _make_env(task, source_usd)
    try:
        env.reset(seed=0)
        robot = env.unwrapped.scene["robot"]
        body_names = list(robot.body_names)
        body_link_state_w = robot.data.body_link_state_w[0].detach().clone()

        base_index = body_names.index(BASE_LINK_NAME)
        base_pos_w = body_link_state_w[base_index : base_index + 1, 0:3].expand(len(body_names), -1)
        base_quat_w = body_link_state_w[base_index : base_index + 1, 3:7].expand(len(body_names), -1)
        body_pos_w = body_link_state_w[:, 0:3]
        body_quat_w = body_link_state_w[:, 3:7]

        body_pos_b, body_quat_b = subtract_frame_transforms(base_pos_w, base_quat_w, body_pos_w, body_quat_w)

        transforms = {}
        for index, body_name in enumerate(body_names):
            transforms[body_name] = (
                body_pos_b[index].detach().cpu().tolist(),
                body_quat_b[index].detach().cpu().tolist(),
            )
        return transforms
    finally:
        env.close()


def _matrix_from_pose(position: list[float], quat_wxyz: list[float]) -> Gf.Matrix4d:
    quat = Gf.Quatd(quat_wxyz[0], Gf.Vec3d(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Rotation(quat))
    matrix.SetTranslate(Gf.Vec3d(position[0], position[1], position[2]))
    return matrix


def _get_body_prim(stage: Usd.Stage, body_name: str) -> Usd.Prim:
    prim = stage.GetPrimAtPath(f"{ROOT_PRIM_PATH}/{body_name}")
    if not prim.IsValid():
        raise RuntimeError(f"Could not find body prim {ROOT_PRIM_PATH}/{body_name}")
    return prim


def write_body_transforms(target_usd: Path, transforms: dict[str, tuple[list[float], list[float]]]) -> None:
    stage = Usd.Stage.Open(str(target_usd))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {target_usd}")

    with Usd.EditContext(stage, stage.GetRootLayer()):
        for body_name, (position, quat_wxyz) in transforms.items():
            prim = _get_body_prim(stage, body_name)
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            transform_op = xformable.AddTransformOp(precision=UsdGeom.XformOp.PrecisionDouble)
            transform_op.Set(_matrix_from_pose(position, quat_wxyz))

    stage.GetRootLayer().Save()


def _matrices_close(actual: Gf.Matrix4d, expected: Gf.Matrix4d, tolerance: float = 1.0e-5) -> bool:
    for row in range(4):
        for col in range(4):
            if abs(actual[row][col] - expected[row][col]) > tolerance:
                return False
    return True


def verify_body_transforms(target_usd: Path, transforms: dict[str, tuple[list[float], list[float]]]) -> None:
    stage = Usd.Stage.Open(str(target_usd))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {target_usd}")

    mismatches = []
    for body_name, (position, quat_wxyz) in transforms.items():
        prim = _get_body_prim(stage, body_name)
        xformable = UsdGeom.Xformable(prim)
        ops = xformable.GetOrderedXformOps()
        if len(ops) != 1 or ops[0].GetOpType() != UsdGeom.XformOp.TypeTransform:
            mismatches.append(f"{body_name}: expected one transform op, found {[op.GetOpName() for op in ops]}")
            continue
        actual = ops[0].Get()
        expected = _matrix_from_pose(position, quat_wxyz)
        if not _matrices_close(actual, expected):
            mismatches.append(f"{body_name}: transform matrix mismatch")

    if mismatches:
        raise RuntimeError("Baked DMBot body transforms do not match reset pose:\n" + "\n".join(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake DMBot reset body transforms into the stance USD copy.")
    parser.add_argument("--task", type=str, default="RobotLab-DMBot-v0")
    parser.add_argument("--source-usd", type=Path, default=DEFAULT_SOURCE_USD)
    parser.add_argument("--target-usd", type=Path, default=DEFAULT_TARGET_USD)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    source_usd = args.source_usd.resolve()
    target_usd = args.target_usd.resolve()
    transforms = collect_reset_body_transforms(args.task, source_usd)

    if not args.verify_only:
        write_body_transforms(target_usd, transforms)
        print(f"Baked {len(transforms)} DMBot body transforms into {target_usd}")

    verify_body_transforms(target_usd, transforms)
    print(f"Verified baked DMBot body transforms in {target_usd}")


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        app.close()
