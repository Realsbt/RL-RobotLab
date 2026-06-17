from pathlib import Path
import sys

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Usd, UsdGeom, UsdPhysics


ROOT = Path(__file__).resolve().parents[2]
USD_PATH = ROOT / "resources/Robots/dm/OpenDog_novlnk_stance/OpenDog_novlnk.usd"


def main():
    stage = Usd.Stage.Open(str(USD_PATH))
    if stage is None:
        raise RuntimeError(f"Could not open USD: {USD_PATH}")

    print(f"USD={USD_PATH}", flush=True)
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith("/DaMiao_OpenDog_novlnk"):
            continue
        markers = []
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            markers.append("RigidBody")
        if prim.IsA(UsdGeom.Mesh):
            markers.append("Mesh")
        marker_text = " markers=" + ",".join(markers) if markers else ""
        print(f"{path} type={prim.GetTypeName()}{marker_text}", flush=True)



if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        app.close()
