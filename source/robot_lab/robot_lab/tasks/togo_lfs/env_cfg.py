import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ImuCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.dmbot.mdp as mdp
from robot_lab.assets.togo_lfs import TOGO_LFS_CFG, TOGO_LFS_JOINT_NAMES
from robot_lab.tasks.dmbot.mdp.terrains import TERRAIN_CFG
from robot_lab.tasks.dmbot.env_cfg import (
    DMBotEnvCfg,
    DMBotSceneCfg,
    ObservationsCfg as DMBotObservationsCfg,
    RewardsCfg as DMBotRewardsCfg,
)


JOINT_NAMES = TOGO_LFS_JOINT_NAMES
BASE_LINK_NAME = "L0_torso"
FOOT_LINK_NAME = ".*_foot"
BASE_HEIGHT_TARGET = 0.30
IMU_MOUNTING_ERROR_RAD = math.radians(2.0)
IMU_UPDATE_PERIOD_S = 0.01
IMU_DELAY_STEPS_RANGE = (0, 1)


def _imu_error_params(output: str) -> dict:
    return {
        "output": output,
        "gyro_bias_range": (-0.05, 0.05),
        "gyro_drift_std": 0.002,
        "gyro_drift_limit": 0.02,
        "gravity_bias_range": (-0.01, 0.01),
        "gravity_drift_std": 0.001,
        "gravity_drift_limit": 0.02,
        "mounting_rpy_range": (-IMU_MOUNTING_ERROR_RAD, IMU_MOUNTING_ERROR_RAD),
        "sensor_name": "imu",
        "delay_steps_range": IMU_DELAY_STEPS_RANGE,
    }


@configclass
class ToGoLFsCommandsCfg:
    """Aggressive command curriculum for the corrected high-speed LFS actuators."""

    base_velocity = mdp.Go2RLGymCommandCfg(
        smooth_command_range_curriculum={
            "nodes": [
                {
                    "iter": 0,
                    "ranges": {
                        "lin_vel_x": [-0.5, 0.5],
                        "lin_vel_y": [-0.5, 0.5],
                        "ang_vel_yaw": [-1.0, 1.0],
                    },
                },
                {
                    "iter": 10000,
                    "ranges": {
                        "lin_vel_x": [-0.75, 0.75],
                        "lin_vel_y": [-0.75, 0.75],
                        "ang_vel_yaw": [-1.25, 1.25],
                    },
                },
                {
                    "iter": 15000,
                    "ranges": {
                        "lin_vel_x": [-1.0, 1.0],
                        "lin_vel_y": [-1.0, 1.0],
                        "ang_vel_yaw": [-1.5, 1.5],
                    },
                },
                {
                    "iter": 20000,
                    "ranges": {
                        "lin_vel_x": [-1.5, 1.5],
                        "lin_vel_y": [-1.0, 1.0],
                        "ang_vel_yaw": [-1.75, 1.75],
                    },
                },
                {
                    "iter": 30000,
                    "ranges": {
                        "lin_vel_x": [-2.0, 2.0],
                        "lin_vel_y": [-1.0, 1.0],
                        "ang_vel_yaw": [-2.0, 2.0],
                    },
                },
            ],
            "log_interval": 1000,
        }
    )


@configclass
class ToGoLFsObservationsCfg(DMBotObservationsCfg):
    """LFS policy observations with persistent IMU calibration errors."""

    @configclass
    class PolicyCfg(DMBotObservationsCfg.PolicyCfg):
        base_ang_vel = ObsTerm(
            func=mdp.imu_with_calibration_error,
            params=_imu_error_params("angular_velocity"),
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.imu_with_calibration_error,
            params=_imu_error_params("projected_gravity"),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )

    @configclass
    class SingleObsCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 1

    policy: PolicyCfg = PolicyCfg()
    single_obs: SingleObsCfg = SingleObsCfg()


@configclass
class ToGoLFsSceneCfg(DMBotSceneCfg):
    """Terrain scene using the Xtellar ToGo_LFs robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.45, 0.45, 0.45),
            roughness=0.8,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = TOGO_LFS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/L0_torso",
        update_period=IMU_UPDATE_PERIOD_S,
        debug_vis=False,
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/L0_torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_small = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/L0_torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.4, 0.3]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ToGoLFsRewardsCfg(DMBotRewardsCfg):
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        },
    )
    feet_regulation = RewTerm(
        func=mdp.feet_regulation,
        weight=-0.05,
        params={
            "base_height_target": BASE_HEIGHT_TARGET,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        },
    )


@configclass
class ToGoLFsTerminationsCfg:
    """Termination terms for ToGo_LFs."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact_consecutive,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_LINK_NAME),
            "threshold": 1.0,
            "consecutive_frames": 2,
        },
    )


@configclass
class ToGoLFsEnvCfg(DMBotEnvCfg):
    """MoE-CTS training configuration for Xtellar ToGo_LFs."""

    scene: ToGoLFsSceneCfg = ToGoLFsSceneCfg(num_envs=8192, env_spacing=0.5)
    observations: ToGoLFsObservationsCfg = ToGoLFsObservationsCfg()
    commands: ToGoLFsCommandsCfg = ToGoLFsCommandsCfg()
    rewards: ToGoLFsRewardsCfg = ToGoLFsRewardsCfg()
    terminations: ToGoLFsTerminationsCfg = ToGoLFsTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 25.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # Force sensor updates in the physics loop so the 100 Hz IMU value is
        # sampled and held independently of the 50 Hz policy observation call.
        self.scene.lazy_sensor_update = False
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.height_scanner_small is not None:
            self.scene.height_scanner_small.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
