from pathlib import Path
import sys

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

from pxr import Usd, UsdGeom


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_USD = ROOT / "resources/Robots/dm/OpenDog_novlnk/OpenDog_novlnk.usd"
STANCE_USD = ROOT / "resources/Robots/dm/OpenDog_novlnk_stance/OpenDog_novlnk.usd"
BODY_NAMES = [
    "L0_torso",
    "Lfl1_hipr",
    "Lfl2_hipp",
    "Lfl3_knee",
    "Lfl4_foot",
    "Lfr1_hipr",
    "Lfr2_hipp",
    "Lfr3_knee",
    "Lfr4_foot",
    "Lrl1_hipr",
    "Lrl2_hipp",
    "Lrl3_knee",
    "Lrl4_foot",
    "Lrr1_hipr",
    "Lrr2_hipp",
    "Lrr3_knee",
    "Lrr4_foot",
]


def _body_info(stage: Usd.Stage, body_name: str):
    prim = stage.GetPrimAtPath(f"/DaMiao_OpenDog_novlnk/{body_name}")
    if not prim.IsValid():
        raise RuntimeError(f"Missing body prim: {body_name}")
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    local_matrix, _ = xformable.GetLocalTransformation()
    return [op.GetOpName() for op in ops], local_matrix


def _translation(matrix):
    return matrix.ExtractTranslation()


def main():
    print(f"original={ORIGINAL_USD}", flush=True)
    print(f"stance={STANCE_USD}", flush=True)
    original_stage = Usd.Stage.Open(str(ORIGINAL_USD))
    stance_stage = Usd.Stage.Open(str(STANCE_USD))
    if original_stage is None or stance_stage is None:
        raise RuntimeError("Could not open one of the USD stages")
    changed = 0
    for body_name in BODY_NAMES:
        original_ops, original_matrix = _body_info(original_stage, body_name)
        stance_ops, stance_matrix = _body_info(stance_stage, body_name)
        original_t = _translation(original_matrix)
        stance_t = _translation(stance_matrix)
        delta = stance_t - original_t
        moved = delta.GetLength() > 1.0e-5 or original_ops != stance_ops
        changed += int(moved)
        print(
            f"{body_name}: changed={moved} "
            f"original_ops={original_ops} stance_ops={stance_ops} "
            f"original_t=({original_t[0]:.4f},{original_t[1]:.4f},{original_t[2]:.4f}) "
            f"stance_t=({stance_t[0]:.4f},{stance_t[1]:.4f},{stance_t[2]:.4f}) "
            f"delta_len={delta.GetLength():.6f}",
            flush=True,
        )
    print(f"changed_bodies={changed}/{len(BODY_NAMES)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        app.close()
