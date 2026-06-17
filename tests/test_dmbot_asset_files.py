from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[1]
DMBOT_ASSET = ROOT / "source/robot_lab/robot_lab/assets/dmbot.py"
STANCE_USD = ROOT / "resources/Robots/dm/OpenDog_novlnk_stance/OpenDog_novlnk.usd"


def _literal_joint_pos_from_dmbot_asset():
    module = ast.parse(DMBOT_ASSET.read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "DM_CFG" for target in node.targets):
            for keyword in node.value.keywords:
                if keyword.arg != "init_state":
                    continue
                for init_keyword in keyword.value.keywords:
                    if init_keyword.arg == "joint_pos":
                        return ast.literal_eval(init_keyword.value)
    raise AssertionError("Could not find DM_CFG init_state.joint_pos")


def test_dmbot_asset_files_are_present():
    expected = [
        ROOT / "resources/Robots/dm/OpenDog_novlnk/OpenDog_novlnk.usd",
        ROOT / "resources/Robots/dm/OpenDog_novlnk/configuration/OpenDog_novlnk_base.usd",
        ROOT / "resources/Robots/dm/OpenDog_novlnk/configuration/OpenDog_novlnk_physics.usd",
        ROOT / "resources/Robots/dm/OpenDog_novlnk/configuration/OpenDog_novlnk_robot.usd",
        ROOT / "resources/Robots/dm/OpenDog_novlnk/configuration/OpenDog_novlnk_sensor.usd",
        ROOT / "resources/Robots/dm/dmgo/dmgo.usd",
        ROOT / "source/robot_lab/robot_lab/assets/dmbot.py",
        ROOT / "source/robot_lab/robot_lab/assets/custom_actuator.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
    assert missing == []


def test_dmbot_stance_usd_copy_is_present():
    expected = [
        STANCE_USD,
        ROOT / "resources/Robots/dm/OpenDog_novlnk_stance/configuration/OpenDog_novlnk_base.usd",
        ROOT / "resources/Robots/dm/OpenDog_novlnk_stance/configuration/OpenDog_novlnk_physics.usd",
        ROOT / "resources/Robots/dm/OpenDog_novlnk_stance/configuration/OpenDog_novlnk_robot.usd",
        ROOT / "resources/Robots/dm/OpenDog_novlnk_stance/configuration/OpenDog_novlnk_sensor.usd",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.exists()]
    assert missing == []


def test_dmbot_asset_config_references_project_asset_dir():
    text = DMBOT_ASSET.read_text()
    assert "DM_CFG" in text
    assert "ISAACLAB_ASSETS_DATA_DIR" in text
    assert "PhysicalMotorCfg" in text
    assert "OpenDog_novlnk_stance/OpenDog_novlnk.usd" in text


def test_dmbot_default_joint_pose_uses_go2_pitch_and_knee_magnitudes():
    joint_pos = _literal_joint_pos_from_dmbot_asset()

    assert joint_pos == {
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
