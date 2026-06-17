from pathlib import Path
import sys

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Usd, UsdPhysics


ROOT = Path(__file__).resolve().parents[2]
USD_PATH = ROOT / "resources/Robots/dm/OpenDog_novlnk_stance/OpenDog_novlnk.usd"


def main():
    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise SystemExit(f"Could not open USD: {USD_PATH}")

    articulation_roots = []
    rigid_bodies = []
    joints = []

    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(path)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(path)
        if prim.IsA(UsdPhysics.Joint):
            joints.append(path)

    print("USD:", USD_PATH)
    print("\nArticulation roots:")
    for item in articulation_roots:
        print(" ", item)
    print("\nRigid bodies:")
    for item in rigid_bodies:
        print(" ", item)
    print("\nJoints:")
    for item in joints:
        print(" ", item)


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        app.close()
