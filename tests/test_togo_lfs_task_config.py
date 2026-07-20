import csv
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
LFS_CFG_PATH = ROOT / "source/robot_lab/robot_lab/tasks/togo_lfs/env_cfg.py"
LFS_OBSERVATIONS_PATH = ROOT / "source/robot_lab/robot_lab/tasks/dmbot/mdp/observations.py"
DMBOT_COMMAND_PATH = ROOT / "source/robot_lab/robot_lab/tasks/dmbot/mdp/commands.py"
LFS_ASSET_PATH = ROOT / "source/robot_lab/robot_lab/assets/togo_lfs.py"
LFS_CUSTOM_ACTUATOR_PATH = ROOT / "source/robot_lab/robot_lab/assets/custom_actuator.py"
LFS_ASSET_DIR = ROOT / "resources/Robots/xtellar/ToGo_LFs_v0p1_new"
LFS_ACTUATOR_CSV_PATH = LFS_ASSET_DIR / "csv/actuator.csv"
LFS_SCHEMA_CSV_PATH = LFS_ASSET_DIR / "csv/schema.csv"
LFS_URDF_PATHS = (
    LFS_ASSET_DIR / "urdf/ToGo_LFs_v0p1_prototype.urdf",
    LFS_ASSET_DIR / "urdf/ToGo_LFs_v0p1_prototype_novlnk.urdf",
)


def test_lfs_has_an_isolated_aggressive_command_curriculum():
    cfg_text = LFS_CFG_PATH.read_text(encoding="utf-8")

    assert "class ToGoLFsCommandsCfg:" in cfg_text
    assert "commands: ToGoLFsCommandsCfg = ToGoLFsCommandsCfg()" in cfg_text
    assert '"iter": 10000' in cfg_text
    assert '"iter": 15000' in cfg_text
    assert '"iter": 20000' in cfg_text
    assert '"iter": 30000' in cfg_text


def test_lfs_curriculum_expands_forward_lateral_and_yaw_ranges():
    cfg_text = LFS_CFG_PATH.read_text(encoding="utf-8")

    for expected_range in (
        '"lin_vel_x": [-0.75, 0.75]',
        '"lin_vel_y": [-0.75, 0.75]',
        '"ang_vel_yaw": [-1.25, 1.25]',
        '"lin_vel_x": [-1.0, 1.0]',
        '"lin_vel_y": [-1.0, 1.0]',
        '"ang_vel_yaw": [-1.5, 1.5]',
        '"lin_vel_x": [-1.5, 1.5]',
        '"ang_vel_yaw": [-1.75, 1.75]',
        '"lin_vel_x": [-2.0, 2.0]',
        '"ang_vel_yaw": [-2.0, 2.0]',
    ):
        assert expected_range in cfg_text


def test_shared_dmbot_command_curriculum_is_unchanged():
    dmbot_text = DMBOT_COMMAND_PATH.read_text(encoding="utf-8")

    assert '"iter": 20000' in dmbot_text
    assert '"iter": 35000' in dmbot_text
    assert '"iter": 60000' in dmbot_text
    assert '"iter": 100000' in dmbot_text


def test_lfs_motor_speed_converts_268_rpm_without_dividing_by_gear_ratio():
    asset_text = LFS_ASSET_PATH.read_text(encoding="utf-8")

    assert "TOGO_LFS_MAX_JOINT_SPEED_RPM = 268.0" in asset_text
    assert "TOGO_LFS_MAX_JOINT_SPEED_RPM * 2.0 * math.pi / 60.0" in asset_text
    assert asset_text.count("velocity_limit=TOGO_LFS_MAX_JOINT_SPEED_RAD_S") == 3
    assert "268.0 / 17.0" not in asset_text


def test_lfs_motor_delay_and_response_cover_realistic_latency():
    asset_text = LFS_ASSET_PATH.read_text(encoding="utf-8")
    actuator_text = LFS_CUSTOM_ACTUATOR_PATH.read_text(encoding="utf-8")

    assert asset_text.count("min_delay=1") == 3
    assert asset_text.count("max_delay=4") == 3
    assert asset_text.count("filter_tau_range=(0.005, 0.015)") == 3
    assert "min_delay=2" not in asset_text
    assert "max_delay=6" not in asset_text
    assert "filter_tau_range: tuple[float, float] | None = None" in actuator_text
    assert "self._resample_filter_tau(env_ids)" in actuator_text


def test_lfs_policy_uses_persistent_imu_calibration_errors():
    cfg_text = LFS_CFG_PATH.read_text(encoding="utf-8")
    observations_text = LFS_OBSERVATIONS_PATH.read_text(encoding="utf-8")

    assert "class ToGoLFsObservationsCfg" in cfg_text
    assert cfg_text.count("func=mdp.imu_with_calibration_error") == 2
    assert '"gyro_bias_range": (-0.05, 0.05)' in cfg_text
    assert '"gyro_drift_std": 0.002' in cfg_text
    assert '"gyro_drift_limit": 0.02' in cfg_text
    assert '"gravity_bias_range": (-0.01, 0.01)' in cfg_text
    assert '"gravity_drift_std": 0.001' in cfg_text
    assert "math.radians(2.0)" in cfg_text
    assert "IMU_UPDATE_PERIOD_S = 0.01" in cfg_text
    assert "IMU_DELAY_STEPS_RANGE = (0, 1)" in cfg_text
    assert "update_period=IMU_UPDATE_PERIOD_S" in cfg_text
    assert "self.scene.lazy_sensor_update = False" in cfg_text
    assert "observations: ToGoLFsObservationsCfg = ToGoLFsObservationsCfg()" in cfg_text

    assert "class _SharedImuErrorState" in observations_text
    assert "class imu_with_calibration_error" in observations_text
    assert "math_utils.quat_apply_inverse" in observations_text
    assert "torch.randn_like(self.gyro_drift)" in observations_text
    assert "sqrt_dt = math.sqrt(float(self.env.step_dt))" in observations_text
    assert "self.delay_history" in observations_text
    assert "sensor.data.ang_vel_b" in observations_text
    assert "sensor.data.projected_gravity_b" in observations_text


def test_lfs_new_source_uses_rpm_and_generated_urdfs_use_rad_s():
    expected_speed = 268.0 * 2.0 * math.pi / 60.0
    with LFS_ACTUATOR_CSV_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    with LFS_SCHEMA_CSV_PATH.open(newline="", encoding="utf-8") as file:
        schema = next(csv.DictReader(file))

    assert len(rows) == 12
    assert schema["Revolute Joint Velocity Unit"] == "revolutions_per_minute"
    assert all(float(row["Velocity Limit"]) == 268.0 for row in rows)
    assert all(float(row["Gear Ratio"]) == 17.0 for row in rows)
    for urdf_path in LFS_URDF_PATHS:
        joints = ET.parse(urdf_path).getroot().findall("joint")
        velocities = [float(joint.find("limit").get("velocity")) for joint in joints if joint.get("type") == "revolute"]
        assert velocities == pytest.approx([expected_speed] * 12)


def test_lfs_new_asset_joint_convention_limits_and_motor_friction():
    asset_text = LFS_ASSET_PATH.read_text(encoding="utf-8")
    root = ET.parse(LFS_URDF_PATHS[1]).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}

    assert "ToGo_LFs_v0p1_new" in asset_text
    assert "TOGO_LFS_COULOMB_FRICTION = 0.194" in asset_text
    assert "TOGO_LFS_VISCOUS_FRICTION = 0.007" in asset_text
    assert asset_text.count("dynamic_friction=TOGO_LFS_COULOMB_FRICTION") == 3
    assert asset_text.count("viscous_friction=TOGO_LFS_VISCOUS_FRICTION") == 3

    expected_hip_axes = {
        "Jfl2_hipp": "0.000000000e+00 1.000000000e+00 0.000000000e+00",
        "Jfr2_hipp": "0.000000000e+00 -1.000000000e+00 0.000000000e+00",
        "Jrl2_hipp": "0.000000000e+00 1.000000000e+00 0.000000000e+00",
        "Jrr2_hipp": "0.000000000e+00 -1.000000000e+00 0.000000000e+00",
    }
    for name, axis in expected_hip_axes.items():
        assert joints[name].find("axis").get("xyz") == axis
        limit = joints[name].find("limit")
        assert float(limit.get("lower")) == pytest.approx(math.radians(-110.0))
        assert float(limit.get("upper")) == pytest.approx(math.radians(110.0))

    for joint in joints.values():
        if joint.get("type") != "revolute":
            continue
        dynamics = joint.find("dynamics")
        assert float(dynamics.get("damping")) == pytest.approx(0.007)
        assert float(dynamics.get("friction")) == pytest.approx(0.194)
