from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tools/visual_check_dmbot.py"


def test_visual_check_script_exists_and_launches_isaac_before_torch():
    text = SCRIPT.read_text()

    assert "AppLauncher" in text
    assert "import torch" in text
    assert text.index("AppLauncher") < text.index("import torch")
    assert 'default="RobotLab-DMBot-v0"' in text
