from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tools/compare_dmbot_go2_reset_geometry.py"


def test_compare_reset_geometry_script_exists_and_uses_both_tasks():
    text = SCRIPT.read_text()

    assert "AppLauncher" in text
    assert "RobotLab-Go2-v0" in text
    assert "RobotLab-DMBot-v0" in text
    assert "GO2_SPEC" in text
    assert "DMBOT_SPEC" in text
    assert "disable_randomization" in text
    assert "subprocess.run" in text
    assert "--robot" in text
    assert "quat_apply_inverse" in text
    assert "foot_positions_b" in text
    assert "FOOT_ORDER" in text
    assert "mean_abs_delta" in text
    assert text.index("AppLauncher") < text.index("import gymnasium")
