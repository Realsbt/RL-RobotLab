import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import robot_lab.tasks.togo_lfs_quiet.mdp as mdp
from robot_lab.assets.togo_lfs import TOGO_LFS_JOINT_NAMES
from robot_lab.tasks.dmbot.env_cfg import CurriculumCfg as DMBotCurriculumCfg
from robot_lab.tasks.dmbot.env_cfg import EventCfg as DMBotEventCfg
from robot_lab.tasks.togo_lfs.env_cfg import (
    BASE_HEIGHT_TARGET,
    BASE_LINK_NAME,
    FOOT_LINK_NAME,
    ToGoLFsEnvCfg,
    ToGoLFsSceneCfg,
)


JOINT_NAMES = TOGO_LFS_JOINT_NAMES
QUIET_FACTOR = 1.4
VELOCITY_REWARD_SCALE = 1.0 - 0.2 * QUIET_FACTOR


@configclass
class ToGoLFsQuietSceneCfg(ToGoLFsSceneCfg):
    """Flat hard-floor scene used for the first MUTE reproduction stage."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=0.9,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.45, 0.45, 0.45),
            roughness=0.8,
        ),
        debug_vis=False,
    )


@configclass
class ToGoLFsQuietCommandsCfg:
    """Low-speed commands without the base task's high-speed curriculum."""

    base_velocity = mdp.MuteVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.15,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.MuteVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.4, 0.4),
        ),
    )


@configclass
class ToGoLFsQuietRewardsCfg:
    """MUTE reward table with an experimental fixed beta of 1.4."""

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0 * VELOCITY_REWARD_SCALE,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5 * VELOCITY_REWARD_SCALE,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    mute_drop_foot_velocity = RewTerm(
        func=mdp.mute_drop_foot_velocity,
        weight=-0.05 * QUIET_FACTOR,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME),
            "swing_duration": 0.35,
            "stance_duration": 0.35,
        },
    )
    mute_raise_foot_velocity = RewTerm(
        func=mdp.mute_raise_foot_velocity,
        weight=0.01 * QUIET_FACTOR,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_LINK_NAME),
            "swing_duration": 0.35,
            "stance_duration": 0.35,
        },
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_power = RewTerm(
        func=mdp.joint_power,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    joint_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hipp|.*_knee"),
            "threshold": 5.0,
        },
    )
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "target_height": BASE_HEIGHT_TARGET,
            "sensor_cfg": SceneEntityCfg("height_scanner_small"),
        },
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)


@configclass
class ToGoLFsQuietEventCfg(DMBotEventCfg):
    """Narrow first-stage randomization around hard indoor flooring."""

    randomize_push_robot = None
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.7, 1.1),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )


@configclass
class ToGoLFsQuietCurriculumCfg(DMBotCurriculumCfg):
    """Disable terrain and inherited reward curricula for the MUTE baseline."""

    terrain_levels = None
    base_linear_velocity = None
    base_height_l2 = None


@configclass
class ToGoLFsQuietEnvCfg(ToGoLFsEnvCfg):
    """First-stage fixed-beta MUTE reproduction for ToGo_LFs."""

    scene: ToGoLFsQuietSceneCfg = ToGoLFsQuietSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: ToGoLFsQuietCommandsCfg = ToGoLFsQuietCommandsCfg()
    rewards: ToGoLFsQuietRewardsCfg = ToGoLFsQuietRewardsCfg()
    events: ToGoLFsQuietEventCfg = ToGoLFsQuietEventCfg()
    curriculum: ToGoLFsQuietCurriculumCfg = ToGoLFsQuietCurriculumCfg()
