from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tools/calibrate_dmbot_joints.py"


def test_dmbot_joint_calibration_script_exists_and_launches_isaac_first():
    text = SCRIPT.read_text()

    assert "AppLauncher" in text
    assert "import torch" in text
    assert text.index("AppLauncher") < text.index("import torch")
    assert 'default="RobotLab-DMBot-v0"' in text


def test_dmbot_joint_calibration_script_has_all_dmbot_action_joints():
    text = SCRIPT.read_text()

    expected = [
        "Jfl1_hipr", "Jfl2_hipp", "Jfl3_knee",
        "Jfr1_hipr", "Jfr2_hipp", "Jfr3_knee",
        "Jrl1_hipr", "Jrl2_hipp", "Jrl3_knee",
        "Jrr1_hipr", "Jrr2_hipp", "Jrr3_knee",
    ]
    for joint_name in expected:
        assert joint_name in text


def test_dmbot_joint_calibration_script_freezes_base_and_reads_terminal_keys():
    text = SCRIPT.read_text()

    assert "fix_root_link" in text
    assert "disable_gravity" in text
    assert "termios" in text
    assert "select.select" in text
    assert "read_key_nonblocking" in text
    assert "format_joint_pos" in text
    assert "joint_pos={" in text
    assert "previous joint" in text
    assert "next joint" in text


def test_dmbot_joint_calibration_script_keeps_task_terrain_generator():
    text = SCRIPT.read_text()

    assert 'terrain_type = "plane"' not in text
    assert "terrain_generator = None" not in text
