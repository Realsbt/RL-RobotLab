from pathlib import Path
import argparse
import math
import sys

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Usd, UsdPhysics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_USD_PATH = ROOT / "resources/Robots/dm/OpenDog_novlnk_stance/OpenDog_novlnk.usd"

DMBOT_STANCE_RAD = {
    "Jfl1_hipr": 0.0,
    "Jfr1_hipr": 0.0,
    "Jrl1_hipr": 0.0,
    "Jrr1_hipr": 0.0,
    "Jfl2_hipp": -0.8,
    "Jfr2_hipp": 0.8,
    "Jrl2_hipp": -1.0,
    "Jrr2_hipp": 1.0,
    "Jfl3_knee": -1.5,
    "Jfr3_knee": 1.5,
    "Jrl3_knee": -1.5,
    "Jrr3_knee": 1.5,
}


def _find_joint_prim(stage: Usd.Stage, joint_name: str) -> Usd.Prim:
    matches = [prim for prim in stage.Traverse() if prim.GetName() == joint_name and prim.IsA(UsdPhysics.Joint)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one USD joint named {joint_name}, found {len(matches)}")
    return matches[0]


def write_joint_states(usd_path: Path) -> None:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {usd_path}")

    with Usd.EditContext(stage, stage.GetRootLayer()):
        for joint_name, joint_pos_rad in DMBOT_STANCE_RAD.items():
            prim = _find_joint_prim(stage, joint_name)
            joint_state = UsdPhysics.JointStateAPI.Apply(prim, "angular")
            joint_state.CreatePositionAttr().Set(math.degrees(joint_pos_rad))
            joint_state.CreateVelocityAttr().Set(0.0)

    stage.GetRootLayer().Save()


def verify_joint_states(usd_path: Path) -> None:
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {usd_path}")

    mismatches = []
    for joint_name, joint_pos_rad in DMBOT_STANCE_RAD.items():
        prim = _find_joint_prim(stage, joint_name)
        attr = prim.GetAttribute("state:angular:physics:position")
        actual = attr.Get() if attr else None
        expected = math.degrees(joint_pos_rad)
        if actual is None or abs(actual - expected) > 1.0e-5:
            mismatches.append((joint_name, actual, expected))

    if mismatches:
        lines = [f"{joint}: actual={actual} expected={expected}" for joint, actual, expected in mismatches]
        raise RuntimeError("USD stance joint state mismatch:\n" + "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Write DMBot default joint state into the stance USD copy.")
    parser.add_argument("--usd", type=Path, default=DEFAULT_USD_PATH)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    usd_path = args.usd.resolve()
    if not args.verify_only:
        write_joint_states(usd_path)
        print(f"Wrote DMBot stance joint states to {usd_path}")
    verify_joint_states(usd_path)
    print(f"Verified DMBot stance joint states in {usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        app.close()
