"""Bake a Go2-style leg pose into a separate DMBot USD preview copy."""

from pathlib import Path
import argparse
import math
import shutil
import sys

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import gymnasium as gym

from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import robot_lab.tasks  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = ROOT / "resources/Robots/dm/OpenDog_novlnk"
DEFAULT_TARGET_DIR = ROOT / "resources/Robots/dm/OpenDog_novlnk_go2_pose"
DEFAULT_SOURCE_USD = DEFAULT_SOURCE_DIR / "OpenDog_novlnk.usd"
DEFAULT_TARGET_USD = ROOT / "resources/Robots/dm/OpenDog_novlnk_go2_pose/OpenDog_novlnk.usd"
DEFAULT_FLATTENED_USD = ROOT / "resources/Robots/dm/OpenDog_novlnk_go2_pose/OpenDog_novlnk_go2_pose_flattened.usd"
ROOT_PRIM_PATH = "/DaMiao_OpenDog_novlnk"
BASE_LINK_NAME = "L0_torso"

# Go2 original reset pose from source/robot_lab/robot_lab/assets/unitree.py,
# mapped to DMBot joint names. Right-side hipp/knee signs follow DMBot's
# mirrored joint-axis convention.
GO2_STYLE_DMBOT_JOINT_POS = {
    "Jfl1_hipr": 0.1,
    "Jfr1_hipr": -0.1,
    "Jrl1_hipr": 0.1,
    "Jrr1_hipr": -0.1,
    "Jfl2_hipp": 0.8,
    "Jfr2_hipp": -0.8,
    "Jrl2_hipp": 1.0,
    "Jrr2_hipp": -1.0,
    "Jfl3_knee": -1.5,
    "Jfr3_knee": 1.5,
    "Jrl3_knee": -1.5,
    "Jrr3_knee": 1.5,
}


def _disable_event(events_cfg, name: str) -> None:
    if hasattr(events_cfg, name):
        setattr(events_cfg, name, None)


def ensure_preview_asset(source_dir: Path, target_dir: Path) -> None:
    if not target_dir.exists():
        shutil.copytree(source_dir, target_dir)
        return

    for source_path in source_dir.rglob("*"):
        relative_path = source_path.relative_to(source_dir)
        target_path = target_dir / relative_path
        if source_path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
        elif not target_path.exists():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)


def _make_env(task: str, source_usd: Path):
    env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = 1
    env_cfg.log_dir = "/tmp/dmbot_usd_go2_pose_bake"
    env_cfg.scene.robot.spawn.usd_path = str(source_usd)
    env_cfg.scene.robot.init_state.joint_pos = GO2_STYLE_DMBOT_JOINT_POS.copy()

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


def _find_joint_prim(stage: Usd.Stage, joint_name: str) -> Usd.Prim:
    matches = [prim for prim in stage.Traverse() if prim.GetName() == joint_name and prim.IsA(UsdPhysics.Joint)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one USD joint named {joint_name}, found {len(matches)}")
    return matches[0]


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


def write_joint_states(target_usd: Path) -> None:
    stage = Usd.Stage.Open(str(target_usd))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {target_usd}")

    with Usd.EditContext(stage, stage.GetRootLayer()):
        for joint_name, joint_pos_rad in GO2_STYLE_DMBOT_JOINT_POS.items():
            prim = _find_joint_prim(stage, joint_name)
            joint_state = UsdPhysics.JointStateAPI.Apply(prim, "angular")
            joint_state.CreatePositionAttr().Set(math.degrees(joint_pos_rad))
            joint_state.CreateVelocityAttr().Set(0.0)

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
        ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
        if len(ops) != 1 or ops[0].GetOpType() != UsdGeom.XformOp.TypeTransform:
            mismatches.append(f"{body_name}: expected one transform op, found {[op.GetOpName() for op in ops]}")
            continue
        if not _matrices_close(ops[0].Get(), _matrix_from_pose(position, quat_wxyz)):
            mismatches.append(f"{body_name}: transform matrix mismatch")

    if mismatches:
        raise RuntimeError("Baked DMBot Go2-pose body transforms do not match reset pose:\n" + "\n".join(mismatches))


def verify_joint_states(target_usd: Path) -> None:
    stage = Usd.Stage.Open(str(target_usd))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {target_usd}")

    mismatches = []
    for joint_name, joint_pos_rad in GO2_STYLE_DMBOT_JOINT_POS.items():
        prim = _find_joint_prim(stage, joint_name)
        attr = prim.GetAttribute("state:angular:physics:position")
        actual = attr.Get() if attr else None
        expected = math.degrees(joint_pos_rad)
        if actual is None or abs(actual - expected) > 1.0e-5:
            mismatches.append((joint_name, actual, expected))

    if mismatches:
        lines = [f"{joint}: actual={actual} expected={expected}" for joint, actual, expected in mismatches]
        raise RuntimeError("USD Go2-pose joint state mismatch:\n" + "\n".join(lines))


def flatten_usd(source_usd: Path, flattened_usd: Path) -> None:
    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {source_usd}")

    flattened_layer = stage.Flatten()
    if not flattened_layer.Export(str(flattened_usd), args={"format": "usdc"}):
        raise RuntimeError(f"Could not export flattened USD: {flattened_usd}")

    check_stage = Usd.Stage.Open(str(flattened_usd))
    if check_stage is None:
        raise RuntimeError(f"Could not reopen flattened USD: {flattened_usd}")
    prim = check_stage.GetPrimAtPath(f"{ROOT_PRIM_PATH}/{BASE_LINK_NAME}")
    if not prim.IsValid():
        raise RuntimeError(f"Flattened USD is missing {ROOT_PRIM_PATH}/{BASE_LINK_NAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bake a Go2-style DMBot leg pose into a separate USD preview copy.")
    parser.add_argument("--task", type=str, default="RobotLab-DMBot-v0")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--source-usd", type=Path, default=DEFAULT_SOURCE_USD)
    parser.add_argument("--target-usd", type=Path, default=DEFAULT_TARGET_USD)
    parser.add_argument("--flattened-usd", type=Path, default=DEFAULT_FLATTENED_USD)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-flatten", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    target_dir = args.target_dir.resolve()
    source_usd = args.source_usd.resolve()
    target_usd = args.target_usd.resolve()
    flattened_usd = args.flattened_usd.resolve()

    ensure_preview_asset(source_dir, target_dir)
    transforms = collect_reset_body_transforms(args.task, source_usd)

    if not args.verify_only:
        write_body_transforms(target_usd, transforms)
        write_joint_states(target_usd)
        print(f"Baked {len(transforms)} DMBot body transforms into {target_usd}")
        print(f"Wrote Go2-style DMBot joint states to {target_usd}")

    verify_body_transforms(target_usd, transforms)
    verify_joint_states(target_usd)
    print(f"Verified Go2-style DMBot pose in {target_usd}")

    if not args.no_flatten:
        flatten_usd(target_usd, flattened_usd)
        print(f"Flattened {target_usd} -> {flattened_usd}")


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        app.close()
