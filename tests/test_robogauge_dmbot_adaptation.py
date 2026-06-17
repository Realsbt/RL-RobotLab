from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mujoco


ROOT = Path(__file__).resolve().parents[1]
ROBOGAUGE_ROOT = ROOT / "external" / "RoboGauge"
DMBOT_XML = ROBOGAUGE_ROOT / "resources" / "robots" / "dmbot" / "dmbot.xml"
DMBOT_ASSETS = ROBOGAUGE_ROOT / "resources" / "robots" / "dmbot" / "assets"

DMBOT_JOINT_NAMES = [
    "Jfl1_hipr",
    "Jfl2_hipp",
    "Jfl3_knee",
    "Jfr1_hipr",
    "Jfr2_hipp",
    "Jfr3_knee",
    "Jrl1_hipr",
    "Jrl2_hipp",
    "Jrl3_knee",
    "Jrr1_hipr",
    "Jrr2_hipp",
    "Jrr3_knee",
]
DMBOT_DEFAULT_DOF_POS = [
    0.0,
    -0.8,
    -1.5,
    0.0,
    0.8,
    1.5,
    0.0,
    -1.0,
    -1.5,
    0.0,
    1.0,
    1.5,
]
DMBOT_FOOT_GEOMS = ["FL", "FR", "RL", "RR"]


sys.path.insert(0, str(ROBOGAUGE_ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "rsl_rl"))


def _object_names(model, obj_type, count: int) -> list[str]:
    return [mujoco.mj_id2name(model, obj_type, index) for index in range(count)]


def test_dmbot_mujoco_resource_loads_with_robogauge_required_interface():
    assert DMBOT_XML.exists()
    assert (DMBOT_ASSETS / "m0_torso.stl").exists()
    assert (DMBOT_ASSETS / "mfl4_foot.stl").exists()

    model = mujoco.MjModel.from_xml_path(str(DMBOT_XML))

    assert _object_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt) == DMBOT_JOINT_NAMES
    f_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "F_base")
    assert abs(float(model.body_pos[f_base_id][2]) - 0.35) < 1e-9
    assert model.nu == 12
    assert sorted(
        name for name in _object_names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom) if name is not None
    ) == DMBOT_FOOT_GEOMS

    sensor_types = [int(model.sensor_type[index]) for index in range(model.nsensor)]
    assert sensor_types.count(int(mujoco.mjtSensor.mjSENS_JOINTPOS)) == 12
    assert sensor_types.count(int(mujoco.mjtSensor.mjSENS_JOINTVEL)) == 12
    assert sensor_types.count(int(mujoco.mjtSensor.mjSENS_JOINTACTFRC)) == 12
    assert sensor_types.count(int(mujoco.mjtSensor.mjSENS_GYRO)) == 1
    assert sensor_types.count(int(mujoco.mjtSensor.mjSENS_ACCELEROMETER)) == 1
    assert sensor_types.count(int(mujoco.mjtSensor.mjSENS_FRAMEQUAT)) == 1
    assert sensor_types.count(int(mujoco.mjtSensor.mjSENS_FRAMEPOS)) == 1
    assert sensor_types.count(int(mujoco.mjtSensor.mjSENS_FRAMELINVEL)) == 1


def test_robogauge_exports_dmbot_robot_config():
    from robogauge.tasks.robots import DMBot, DMBotConfig, DMBotTerrainConfig

    cfg = DMBotConfig()

    assert cfg.robot_name == "dmbot"
    assert cfg.robot_class == "DMBot"
    assert cfg.assets.robot_xml == "{ROBOGAUGE_ROOT_DIR}/resources/robots/dmbot/dmbot.xml"
    assert cfg.assets.base_body_name == "L0_torso"
    assert cfg.assets.robot_spawn_height == 0.35
    assert cfg.assets.foot_geom_names == DMBOT_FOOT_GEOMS
    assert cfg.control.default_dof_pos == DMBOT_DEFAULT_DOF_POS
    assert cfg.control.p_gains == [40.0] * 12
    assert cfg.control.d_gains == [2.0] * 12
    assert cfg.control.scales.cmd == [1.0, 1.0, 1.0]
    assert issubclass(DMBotTerrainConfig, DMBotConfig)
    assert DMBot.__name__ == "DMBot"


def test_robogauge_registers_dmbot_tasks():
    import robogauge.tasks  # noqa: F401
    from robogauge.utils.task_register import task_register

    for task_name in (
        "dmbot.flat",
        "dmbot.slope_fd",
        "dmbot.slope_bd",
        "dmbot.wave",
        "dmbot.stairs_fd",
        "dmbot.stairs_bd",
        "dmbot.obstacle",
    ):
        _, _, robot_cfg = task_register.get_cfgs(task_name)
        assert robot_cfg.robot_name == "dmbot"


def test_mujoco_simulator_load_accepts_configurable_dmbot_base_body():
    from robogauge.tasks.simulator.mujoco_config import MujocoConfig
    from robogauge.tasks.simulator.mujoco_simulator import MujocoSimulator

    sim_cfg = MujocoConfig()
    sim_cfg.viewer.headless = True
    sim_cfg.noise.enabled = False
    sim = MujocoSimulator(sim_cfg)
    try:
        sim.load(
            terrain_xmls=["{ROBOGAUGE_ROOT_DIR}/resources/terrains/flat.xml"],
            robot_xml="{ROBOGAUGE_ROOT_DIR}/resources/robots/dmbot/dmbot.xml",
            terrain_spawn_pos=[0, 0, 0],
            default_dof_pos=DMBOT_DEFAULT_DOF_POS,
            invert_yaw=False,
            base_body_name="L0_torso",
        )
        assert sim.robot_name_prefix == "dmbot/"
        assert [name.rsplit("/", 1)[-1] for name in sim.dof_names] == DMBOT_JOINT_NAMES
    finally:
        if hasattr(sim, "close_viewer"):
            sim.close_viewer()
        if hasattr(sim, "close_video_writer"):
            sim.close_video_writer()


def test_training_cli_forwards_robogauge_task_name():
    import cli_args

    parser = argparse.ArgumentParser()
    cli_args.add_rsl_rl_args(parser)

    default_args = parser.parse_args([])
    custom_args = parser.parse_args(["--robogauge_task", "dmbot"])

    assert default_args.robogauge_task == "go2_lab"
    assert custom_args.robogauge_task == "dmbot"

    train_text = (ROOT / "scripts" / "rsl_rl" / "train.py").read_text()
    runner_text = (ROOT / "source" / "rsl_rl" / "rsl_rl" / "runners" / "on_policy_runner_cts.py").read_text()
    assert '"task_name": args_cli.robogauge_task' in train_text
    assert 'robogauge_cfg.get("task_name", "go2_lab")' in runner_text
