from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tools/write_dmbot_usd_stance.py"
BAKE_SCRIPT = ROOT / "scripts/tools/bake_dmbot_usd_stance.py"
FLATTEN_SCRIPT = ROOT / "scripts/tools/flatten_dmbot_usd_stance.py"
INSPECT_SCRIPT = ROOT / "scripts/tools/inspect_dmbot_usd.py"
GO2_POSE_SCRIPT = ROOT / "scripts/tools/bake_dmbot_usd_go2_pose.py"
GO2_FLATTEN_SCRIPT = ROOT / "scripts/tools/flatten_dmbot_usd_go2_pose.py"


def test_dmbot_usd_stance_script_exists_and_uses_joint_state_api():
    text = SCRIPT.read_text()

    assert "AppLauncher" in text
    assert "UsdPhysics.JointStateAPI" in text
    assert "OpenDog_novlnk_stance/OpenDog_novlnk.usd" in text
    assert '"Jfl2_hipp": -0.8' in text
    assert '"Jfr2_hipp": 0.8' in text
    assert '"Jrl2_hipp": -1.0' in text
    assert '"Jrr2_hipp": 1.0' in text
    assert '"Jfl3_knee": -1.5' in text
    assert '"Jfr3_knee": 1.5' in text
    assert text.index("AppLauncher") < text.index("from pxr")


def test_dmbot_usd_inspector_defaults_to_stance_copy():
    text = INSPECT_SCRIPT.read_text()

    assert "OpenDog_novlnk_stance/OpenDog_novlnk.usd" in text


def test_dmbot_usd_bake_script_bakes_torso_relative_body_transforms():
    text = BAKE_SCRIPT.read_text()

    assert "AppLauncher" in text
    assert "body_link_state_w" in text
    assert "subtract_frame_transforms" in text
    assert "UsdGeom.Xformable" in text
    assert "OpenDog_novlnk_stance/OpenDog_novlnk.usd" in text
    assert "--verify-only" in text
    assert text.index("AppLauncher") < text.index("from pxr")


def test_dmbot_usd_flatten_script_exports_viewable_stance_usd():
    text = FLATTEN_SCRIPT.read_text()

    assert "AppLauncher" in text
    assert "OpenDog_novlnk_stance/OpenDog_novlnk.usd" in text
    assert "OpenDog_novlnk_stance_flattened.usd" in text
    assert "OpenDog_novlnk_stance_flattened.usdc" not in text
    assert 'args={"format": "usdc"}' in text
    assert "Flatten" in text
    assert ".identifier" not in text
    assert "app.close()" not in text
    assert text.index("AppLauncher") < text.index("from pxr")


def test_dmbot_usd_go2_pose_script_exports_separate_preview_usd():
    text = GO2_POSE_SCRIPT.read_text()

    assert "AppLauncher" in text
    assert "GO2_STYLE_DMBOT_JOINT_POS" in text
    assert "OpenDog_novlnk_go2_pose/OpenDog_novlnk.usd" in text
    assert "OpenDog_novlnk_go2_pose_flattened.usd" in text
    assert "OpenDog_novlnk_go2_pose_flattened.usdc" not in text
    assert '"Jfl1_hipr": 0.1' in text
    assert '"Jfr1_hipr": -0.1' in text
    assert '"Jfl2_hipp": 0.8' in text
    assert '"Jfr2_hipp": -0.8' in text
    assert '"Jrl2_hipp": 1.0' in text
    assert '"Jrr2_hipp": -1.0' in text
    assert '"Jfl3_knee": -1.5' in text
    assert '"Jfr3_knee": 1.5' in text
    assert "subtract_frame_transforms" in text
    assert "UsdPhysics.JointStateAPI" in text
    assert "Flatten" in text
    assert ".identifier" not in text
    assert 'args={"format": "usdc"}' in text
    assert text.index("AppLauncher") < text.index("from pxr")


def test_dmbot_usd_go2_pose_flatten_script_exports_viewable_preview_usd():
    text = GO2_FLATTEN_SCRIPT.read_text()

    assert "AppLauncher" in text
    assert "OpenDog_novlnk_go2_pose/OpenDog_novlnk.usd" in text
    assert "OpenDog_novlnk_go2_pose_flattened.usd" in text
    assert "OpenDog_novlnk_go2_pose_flattened.usdc" not in text
    assert 'args={"format": "usdc"}' in text
    assert "Flatten" in text
    assert ".identifier" not in text
    assert "app.close()" not in text
    assert text.index("AppLauncher") < text.index("from pxr")
