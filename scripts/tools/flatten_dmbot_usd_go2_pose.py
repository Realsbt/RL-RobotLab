from pathlib import Path
import sys

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Usd


ROOT = Path(__file__).resolve().parents[2]
SOURCE_USD = ROOT / "resources/Robots/dm/OpenDog_novlnk_go2_pose/OpenDog_novlnk.usd"
TARGET_USD = ROOT / "resources/Robots/dm/OpenDog_novlnk_go2_pose/OpenDog_novlnk_go2_pose_flattened.usd"


def main():
    stage = Usd.Stage.Open(str(SOURCE_USD))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {SOURCE_USD}")

    flattened_layer = stage.Flatten()
    if not flattened_layer.Export(str(TARGET_USD), args={"format": "usdc"}):
        raise RuntimeError(f"Could not export flattened USD: {TARGET_USD}")

    check_stage = Usd.Stage.Open(str(TARGET_USD))
    if check_stage is None:
        raise RuntimeError(f"Could not reopen flattened USD: {TARGET_USD}")
    prim = check_stage.GetPrimAtPath("/DaMiao_OpenDog_novlnk/L0_torso")
    if not prim.IsValid():
        raise RuntimeError("Flattened USD is missing /DaMiao_OpenDog_novlnk/L0_torso")

    print(f"Flattened {SOURCE_USD} -> {TARGET_USD}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
