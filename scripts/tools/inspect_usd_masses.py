from __future__ import annotations

import argparse
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect rigid-body masses authored in a USD file.")
    parser.add_argument(
        "usd_path",
        nargs="?",
        default="resources/Robots/dm/OpenDog_novlnk_stance/OpenDog_novlnk.usd",
        help="USD file to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = AppLauncher(headless=True).app

    from pxr import Usd, UsdPhysics

    usd_path = Path(args.usd_path).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise SystemExit(f"Could not open USD: {usd_path}")

    rows: list[tuple[str, float]] = []
    missing: list[str] = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        mass_api = UsdPhysics.MassAPI(prim)
        mass = mass_api.GetMassAttr().Get()
        if mass is None:
            missing.append(prim.GetPath().pathString)
        else:
            rows.append((prim.GetPath().pathString, float(mass)))

    total_mass = sum(mass for _, mass in rows)
    print(f"USD: {usd_path}")
    print(f"Rigid bodies with mass: {len(rows)}")
    print(f"Rigid bodies missing mass: {len(missing)}")
    print(f"Total authored mass: {total_mass:.6f} kg")
    print()
    for path, mass in rows:
        print(f"{mass:10.6f} kg  {path}")

    if missing:
        print()
        print("Missing mass attributes:")
        for path in missing:
            print(path)

    sys.stdout.flush()
    app.close()


if __name__ == "__main__":
    main()
