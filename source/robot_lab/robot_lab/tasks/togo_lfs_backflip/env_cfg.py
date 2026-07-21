"""Staged PPO environments for a single ToGo_LFs backflip."""

from __future__ import annotations

import math

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.togo_lfs_backflip.mdp as mdp
from robot_lab.assets.togo_lfs import TOGO_LFS_CFG, TOGO_LFS_JOINT_NAMES
from robot_lab.tasks.togo_lfs.env_cfg import (
    BASE_LINK_NAME,
    FOOT_LINK_NAME,
    ToGoLFsEnvCfg,
    ToGoLFsSceneCfg,
)


JOINT_NAMES = TOGO_LFS_JOINT_NAMES
EPISODE_LENGTH_S = 1.8
GENESIS_EPISODE_LENGTH_S = 2.0
GENESIS_ROTATION_START_S = 0.50
GENESIS_ROTATION_END_S = 1.00
INITIAL_BASE_HEIGHT = 0.30
TAKEOFF_IMPULSE_START_S = 0.20
TAKEOFF_IMPULSE_END_S = 0.36
ROTATION_START_S = 0.30
APEX_TARGET_S = 0.56
ROTATION_END_S = 0.82
MINIMUM_LANDING_ROTATION = 1.50 * math.pi

# Keep the measured torque-speed envelope in every stage, but remove latency
# and response filtering while the skill is first discovered. A separate robust
# scene restores them after a complete landing policy exists.
TOGO_LFS_BACKFLIP_CFG = TOGO_LFS_CFG.copy()
TOGO_LFS_BACKFLIP_CFG.spawn.articulation_props.enabled_self_collisions = False
for actuator_cfg in TOGO_LFS_BACKFLIP_CFG.actuators.values():
    actuator_cfg.min_delay = 0
    actuator_cfg.max_delay = 0
    actuator_cfg.filter_tau = 0.0
    actuator_cfg.filter_tau_range = (0.0, 0.0)

TOGO_LFS_BACKFLIP_ROBUST_CFG = TOGO_LFS_CFG.copy()
TOGO_LFS_BACKFLIP_ROBUST_CFG.spawn.articulation_props.enabled_self_collisions = False

# Genesis-backflip executes the previous policy action, which is a fixed 20 ms
# delay at its 50 Hz control rate. Preserve the ToGo torque-speed model while
# matching that timing in a separate baseline asset. Self-collision remains
# disabled because the current ToGo collision meshes are not yet trustworthy.
TOGO_LFS_BACKFLIP_GENESIS_CFG = TOGO_LFS_CFG.copy()
TOGO_LFS_BACKFLIP_GENESIS_CFG.spawn.articulation_props.enabled_self_collisions = False
for actuator_cfg in TOGO_LFS_BACKFLIP_GENESIS_CFG.actuators.values():
    actuator_cfg.min_delay = 4
    actuator_cfg.max_delay = 4
    actuator_cfg.filter_tau = 0.0
    actuator_cfg.filter_tau_range = (0.0, 0.0)


@configclass
class ToGoLFsBackflipSceneCfg(ToGoLFsSceneCfg):
    """Flat scene with only the sensors needed by the acrobatic task."""

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
            diffuse_color=(0.40, 0.40, 0.42),
            roughness=0.8,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = TOGO_LFS_BACKFLIP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    height_scanner = None
    height_scanner_small = None


@configclass
class ToGoLFsBackflipRobustSceneCfg(ToGoLFsBackflipSceneCfg):
    """Final-stage scene with measured actuator delay and response filtering."""

    robot: ArticulationCfg = TOGO_LFS_BACKFLIP_ROBUST_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


@configclass
class ToGoLFsBackflipGenesisSceneCfg(ToGoLFsBackflipSceneCfg):
    """ToGo scene with Genesis' one-policy-step action latency."""

    robot: ArticulationCfg = TOGO_LFS_BACKFLIP_GENESIS_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


@configclass
class ToGoLFsBackflipActionsCfg:
    """Joint-position residual actions clamped to 95% of URDF limits."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale={".*_hipr": 0.5, ".*_hipp": 0.5, ".*_knee": 0.5},
        use_default_offset=True,
        clip={
            ".*_hipr": (-0.4145, 0.4145),
            ".*_hipp": (-1.8239, 1.8239),
            "J(fl|rl)3_knee": (0.0, 1.6581),
            "J(fr|rr)3_knee": (-1.6581, 0.0),
        },
        preserve_order=True,
    )


@configclass
class ToGoLFsBackflipTerminalFrontActionsCfg:
    """Ordinary PPO actions with a phase-gated terminal front stance blend."""

    joint_pos = mdp.TerminalFrontStanceActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale={".*_hipr": 0.5, ".*_hipp": 0.5, ".*_knee": 0.5},
        use_default_offset=True,
        clip={
            ".*_hipr": (-0.4145, 0.4145),
            ".*_hipp": (-1.8239, 1.8239),
            "J(fl|rl)3_knee": (0.0, 1.6581),
            "J(fr|rr)3_knee": (-1.6581, 0.0),
        },
        preserve_order=True,
        command_name="backflip_phase",
        minimum_rotation=1.52 * math.pi,
        full_rotation=1.75 * math.pi,
        front_target_offsets=(0.0, -0.45, -0.40, 0.0, 0.45, 0.40),
        post_touchdown_all_target_offsets=(0.0,) * 12,
        post_touchdown_blend_duration_s=0.30,
    )


@configclass
class ToGoLFsBackflipCommandsCfg:
    """Deterministic phase clock plus simulator-only episode state."""

    backflip_phase = mdp.BackflipPhaseCommandCfg(
        episode_length_s=EPISODE_LENGTH_S,
        initial_base_height=INITIAL_BASE_HEIGHT,
        backward_pitch_sign=-1.0,
    )


@configclass
class ToGoLFsBackflipGenesisCommandsCfg:
    """Genesis-compatible two-second clock with a half-cycle phase encoding."""

    backflip_phase = mdp.BackflipPhaseCommandCfg(
        episode_length_s=GENESIS_EPISODE_LENGTH_S,
        phase_cycles=0.5,
        initial_base_height=INITIAL_BASE_HEIGHT,
        backward_pitch_sign=-1.0,
    )


@configclass
class ToGoLFsBackflipGenesisLandingCommandsCfg(
    ToGoLFsBackflipGenesisCommandsCfg
):
    """Two-second clock with a practical takeoff-angle gate for landing work."""

    backflip_phase = mdp.BackflipPhaseCommandCfg(
        episode_length_s=GENESIS_EPISODE_LENGTH_S,
        phase_cycles=0.5,
        initial_base_height=INITIAL_BASE_HEIGHT,
        backward_pitch_sign=-1.0,
        maximum_rotation_at_takeoff=0.55 * math.pi,
        minimum_stable_landing_time_s=0.20,
    )


@configclass
class ToGoLFsBackflipObservationsCfg:
    """60-D deployable actor input and asymmetric privileged critic input."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.10, n_max=0.10),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                )
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                )
            },
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.05,
        )
        action_pair = ObsTerm(func=mdp.action_pair)
        phase = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "backflip_phase"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                )
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=JOINT_NAMES, preserve_order=True
                )
            },
            scale=0.05,
        )
        action_pair = ObsTerm(func=mdp.action_pair)
        phase = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "backflip_phase"},
        )
        foot_contacts = ObsTerm(
            func=mdp.foot_contacts,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=FOOT_LINK_NAME
                ),
                "force_threshold": 2.0,
            },
        )
        backflip_state = ObsTerm(
            func=mdp.privileged_backflip_state,
            params={"command_name": "backflip_phase"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ToGoLFsBackflipEventCfg:
    """Deterministic resets for skill acquisition."""

    reset_base = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_robot_joints = EventTerm(
        func=base_mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES),
        },
    )


@configclass
class ToGoLFsBackflipRobustEventCfg(ToGoLFsBackflipEventCfg):
    """Narrow randomization introduced only after consistent landing."""

    randomize_rigid_body_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.6, 1.1),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 32,
            "make_consistent": True,
        },
    )
    randomize_base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_com = EventTerm(
        func=base_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "com_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
            },
        },
    )
    randomize_actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class ToGoLFsBackflipGenesisEventCfg(ToGoLFsBackflipEventCfg):
    """Domain randomization ranges published by Genesis-backflip."""

    randomize_rigid_body_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    randomize_base_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_com = EventTerm(
        func=base_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_LINK_NAME),
            "com_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
            },
        },
    )
    randomize_actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class ToGoLFsBackflipGenesisRewardsCfg:
    """Minimal end-to-end rewards reimplemented from Genesis-backflip."""

    pitch_velocity = RewTerm(
        func=mdp.genesis_backward_pitch_velocity,
        weight=5.0,
        params={
            "start_s": GENESIS_ROTATION_START_S,
            "end_s": GENESIS_ROTATION_END_S,
            "maximum_rate": 7.2,
            "backward_pitch_sign": -1.0,
        },
    )
    yaw_velocity = RewTerm(func=mdp.genesis_yaw_rate_l1, weight=-1.0)
    vertical_velocity = RewTerm(
        func=mdp.genesis_vertical_velocity,
        weight=20.0,
        params={"start_s": 0.50, "end_s": 0.75, "maximum_speed": 3.0},
    )
    orientation_control = RewTerm(
        func=mdp.genesis_orientation_error_l2,
        weight=-1.0,
        params={
            "rotation_start_s": GENESIS_ROTATION_START_S,
            "rotation_end_s": GENESIS_ROTATION_END_S,
            "backward_pitch_sign": -1.0,
        },
    )
    feet_height_before_rotation = RewTerm(
        func=mdp.genesis_feet_height_before_rotation,
        weight=-30.0,
        params={
            "end_s": GENESIS_ROTATION_START_S,
            "ground_clearance": 0.02,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=FOOT_LINK_NAME, preserve_order=True
            ),
        },
    )
    base_height = RewTerm(
        func=mdp.genesis_base_height_l2,
        weight=-10.0,
        params={
            "target_height": INITIAL_BASE_HEIGHT,
            "early_end_s": 0.40,
            "late_start_s": 1.40,
        },
    )
    mirrored_action = RewTerm(func=mdp.mirrored_action_l2, weight=-0.1)
    gravity_y = RewTerm(func=mdp.genesis_gravity_y_l2, weight=-10.0)
    feet_distance = RewTerm(
        func=mdp.genesis_feet_lateral_distance_l2,
        weight=-1.0,
        params={
            "stance_width": 0.0,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=FOOT_LINK_NAME, preserve_order=True
            ),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)


@configclass
class ToGoLFsBackflipGenesisStrictRewardsCfg(ToGoLFsBackflipGenesisRewardsCfg):
    """Nominal Genesis acquisition with all rotation credit ending at touchdown."""

    pitch_velocity = RewTerm(
        func=mdp.genesis_backward_pitch_velocity_before_touchdown,
        weight=10.0,
        params={
            "command_name": "backflip_phase",
            "start_s": 0.35,
            "end_s": 1.10,
            "maximum_rate": 10.0,
            "backward_pitch_sign": -1.0,
        },
    )
    orientation_control = RewTerm(
        func=mdp.genesis_orientation_error_l2,
        weight=-10.0,
        params={
            "rotation_start_s": GENESIS_ROTATION_START_S,
            "rotation_end_s": GENESIS_ROTATION_END_S,
            "backward_pitch_sign": -1.0,
        },
    )
    gravity_y = RewTerm(func=mdp.genesis_gravity_y_l2, weight=-30.0)
    off_axis_rotation = RewTerm(
        func=mdp.off_axis_angular_velocity_l2,
        weight=-0.20,
    )
    vertical_velocity = RewTerm(
        func=mdp.upward_velocity_window,
        weight=20.0,
        params={"start_s": 0.35, "end_s": 0.70, "maximum_speed": 3.0},
    )
    jump_height = RewTerm(
        func=mdp.jump_height,
        weight=5.0,
        params={
            "initial_height": INITIAL_BASE_HEIGHT,
            "maximum_height_gain": 0.60,
        },
    )
    takeoff = RewTerm(
        func=mdp.takeoff_success,
        weight=2.0,
        params={"command_name": "backflip_phase"},
    )
    launch_pitch_momentum = RewTerm(
        func=mdp.takeoff_backward_pitch_quality,
        weight=100.0,
        params={
            "command_name": "backflip_phase",
            "target_rate": 12.0,
            "reward_end_s": 0.60,
        },
    )
    takeoff_rotation_excess = RewTerm(
        func=mdp.takeoff_rotation_excess_l2,
        weight=-200.0,
        params={"command_name": "backflip_phase"},
    )
    supported_rotation = RewTerm(
        func=mdp.supported_rotation_l2,
        weight=-100.0,
        params={"command_name": "backflip_phase"},
    )
    rotation_progress = RewTerm(
        func=mdp.capped_backward_rotation_progress,
        weight=20.0,
        params={
            "command_name": "backflip_phase",
            "maximum_rewarded_rotation": 2.0 * math.pi,
        },
    )
    touchdown_rotation_quality = RewTerm(
        func=mdp.first_touchdown_rotation_quality,
        weight=1000.0,
        params={
            "command_name": "backflip_phase",
            "target_rotation": 2.0 * math.pi,
            "takeoff_excess_scale": 0.25,
        },
    )
    touchdown_nonfoot = RewTerm(
        func=mdp.first_touchdown_nonfoot_contact,
        weight=-200.0,
        params={"command_name": "backflip_phase"},
    )


@configclass
class ToGoLFsBackflipGenesisLandingRewardsCfg(
    ToGoLFsBackflipGenesisStrictRewardsCfg
):
    """R8: retain the discovered revolution and learn a clean stable landing."""

    # Pay launch momentum only through early flight.  The stronger weight
    # compensates for the extra inertia introduced by deploying the legs,
    # without rewarding residual pitch rate all the way to touchdown.
    pitch_velocity = None
    vertical_velocity = RewTerm(
        func=mdp.upward_velocity_window,
        weight=50.0,
        params={"start_s": 0.35, "end_s": 0.70, "maximum_speed": 3.0},
    )
    jump_height = RewTerm(
        func=mdp.jump_height,
        weight=20.0,
        params={
            "initial_height": INITIAL_BASE_HEIGHT,
            "maximum_height_gain": 0.60,
        },
    )
    launch_pitch_momentum = RewTerm(
        func=mdp.takeoff_backward_pitch_quality,
        weight=500.0,
        params={
            "command_name": "backflip_phase",
            "target_rate": 16.0,
            "reward_end_s": 0.55,
        },
    )
    launch_pitch_impulse = RewTerm(
        func=mdp.launch_backward_pitch_quality,
        weight=100.0,
        params={
            "command_name": "backflip_phase",
            "start_s": 0.20,
            "end_s": 0.36,
            "target_rate": 16.0,
            "minimum_upward_speed": 0.20,
            "full_upward_speed": 1.50,
        },
    )
    takeoff_rotation_excess = None
    supported_rotation = None

    rotation_progress = RewTerm(
        func=mdp.capped_backward_rotation_progress,
        weight=50.0,
        params={
            "command_name": "backflip_phase",
            "maximum_rewarded_rotation": 2.0 * math.pi,
        },
    )
    touchdown_rotation_quality = RewTerm(
        func=mdp.first_touchdown_rotation_quality,
        weight=5000.0,
        params={
            "command_name": "backflip_phase",
            "target_rotation": 2.0 * math.pi,
            "takeoff_excess_scale": 0.25,
        },
    )
    touchdown_nonfoot = RewTerm(
        func=mdp.first_touchdown_nonfoot_contact,
        weight=-3000.0,
        params={"command_name": "backflip_phase"},
    )
    touchdown_feet = RewTerm(
        func=mdp.first_touchdown_foot_contact_quality,
        weight=5000.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.50 * math.pi,
            "target_rotation": 2.0 * math.pi,
            "rotation_error_scale": 0.50,
        },
    )
    landing_approach = RewTerm(
        func=mdp.landing_approach,
        weight=100.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.40 * math.pi,
            "target_rotation": 2.0 * math.pi,
            "orientation_error_scale": 0.75,
            "rotation_error_scale": 3.00,
            "angular_velocity_scale": 12.0,
        },
    )
    landing_leg_extension = RewTerm(
        func=mdp.landing_leg_extension,
        weight=1000.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.25 * math.pi,
            "full_rotation": 1.75 * math.pi,
            "minimum_foot_drop": -0.40,
            "full_foot_drop": 0.20,
            "orientation_error_scale": 2.00,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=FOOT_LINK_NAME, preserve_order=True
            ),
        },
    )
    landing_foot_fore_aft_stance = RewTerm(
        func=mdp.landing_foot_fore_aft_stance_l2,
        weight=-1000.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.70 * math.pi,
            "full_rotation": 1.85 * math.pi,
            "front_target_x": 0.15,
            "rear_target_x": -0.19,
        },
    )
    landing_joint_pose = RewTerm(
        func=mdp.landing_joint_pose_l2,
        weight=-1000.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 0.95 * math.pi,
            "full_rotation": 1.30 * math.pi,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=JOINT_NAMES, preserve_order=True
            ),
        },
    )
    landing_foot_clearance = RewTerm(
        func=mdp.landing_foot_origin_clearance,
        weight=1000.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.25 * math.pi,
            "full_rotation": 1.50 * math.pi,
            "foot_rank": 2,
            "minimum_margin": -0.04,
            "full_margin": 0.08,
        },
    )
    landing_all_foot_clearance = RewTerm(
        func=mdp.landing_foot_origin_clearance,
        weight=3000.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.10 * math.pi,
            "full_rotation": 1.45 * math.pi,
            "foot_rank": 4,
            "minimum_margin": -0.60,
            "full_margin": 0.05,
        },
    )
    landing_braking = RewTerm(
        func=mdp.landing_angular_speed_excess_l2,
        weight=-20.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.65 * math.pi,
            "full_rotation": 1.85 * math.pi,
            "maximum_angular_speed": 4.0,
        },
    )
    landing_contact = RewTerm(
        func=mdp.landing_contact_ratio,
        weight=100.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.50 * math.pi,
        },
    )
    landing_clean_support = RewTerm(
        func=mdp.post_touchdown_clean_support,
        weight=5000.0,
        params={"command_name": "backflip_phase"},
    )
    landing_post_foot_clearance = RewTerm(
        func=mdp.post_touchdown_foot_origin_clearance,
        weight=3000.0,
        params={
            "command_name": "backflip_phase",
            "foot_rank": 4,
            "minimum_margin": -0.60,
            "full_margin": 0.05,
        },
    )
    landing_default_action = RewTerm(
        func=mdp.landing_default_action_l2,
        weight=-10.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.20 * math.pi,
            "full_rotation": 1.50 * math.pi,
        },
    )
    landing_retracted_leg_action = RewTerm(
        func=mdp.landing_retracted_leg_action_l2,
        weight=-100.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 0.95 * math.pi,
            "full_rotation": 1.20 * math.pi,
            "minimum_margin": -0.40,
            "full_margin": 0.05,
        },
    )
    landing_front_leg_default_action = RewTerm(
        func=mdp.landing_front_leg_default_action_l2,
        weight=-500.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.70 * math.pi,
            "full_rotation": 1.85 * math.pi,
        },
    )
    landing_post_motion = RewTerm(
        func=mdp.post_touchdown_motion_excess_l2,
        weight=-25.0,
        params={
            "command_name": "backflip_phase",
            "maximum_linear_speed": 0.75,
            "maximum_angular_speed": 2.0,
        },
    )
    landing_post_upright = RewTerm(
        func=mdp.post_touchdown_upright_error_l2,
        weight=-500.0,
        params={"command_name": "backflip_phase"},
    )
    landing_post_joint_pose = RewTerm(
        func=mdp.post_touchdown_joint_pose_l2,
        weight=-100.0,
        params={
            "command_name": "backflip_phase",
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=JOINT_NAMES, preserve_order=True
            ),
        },
    )
    landing_post_nonfoot = RewTerm(
        func=mdp.post_touchdown_nonfoot_contact,
        weight=-200.0,
        params={"command_name": "backflip_phase"},
    )
    landing_stability = RewTerm(
        func=mdp.landing_stability,
        weight=300.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.50 * math.pi,
            "linear_velocity_scale": 1.0,
            "angular_velocity_scale": 4.0,
        },
    )
    rotation_accuracy = RewTerm(
        func=mdp.landing_rotation_accuracy,
        weight=80.0,
        params={"command_name": "backflip_phase", "error_scale": 0.50},
    )
    completed_backflip = RewTerm(
        func=mdp.completed_backflip,
        weight=500.0,
        params={"command_name": "backflip_phase"},
    )


@configclass
class ToGoLFsBackflipGenesisLandingEnergyRewardsCfg(
    ToGoLFsBackflipGenesisLandingRewardsCfg
):
    """Acquire extra launch energy before enabling the front-leg deployment."""

    landing_foot_fore_aft_stance = None
    landing_front_leg_default_action = None


@configclass
class ToGoLFsBackflipGenesisLandingAssistedRewardsCfg(
    ToGoLFsBackflipGenesisLandingRewardsCfg
):
    """Train launch and absorption around the isolated terminal front-leg layer."""

    # The deterministic action term owns terminal front-foot placement.  These
    # actor-space objectives previously changed the shared launch policy before
    # producing usable landing motion, so they must not compete with it here.
    landing_foot_fore_aft_stance = None
    landing_front_leg_default_action = None
    landing_default_action = None

    # Keep the rear-leg deployment curriculum while excluding the front joints
    # whose airborne targets intentionally differ from the nominal stand pose.
    landing_joint_pose = RewTerm(
        func=mdp.landing_joint_pose_l2,
        weight=-1000.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 0.95 * math.pi,
            "full_rotation": 1.30 * math.pi,
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=JOINT_NAMES[6:], preserve_order=True
            ),
        },
    )


@configclass
class ToGoLFsBackflipJumpRewardsCfg:
    """Stage 1: acquire a symmetric, high-energy vertical jump."""

    jump_height = RewTerm(
        func=mdp.jump_height,
        weight=12.0,
        params={
            "initial_height": INITIAL_BASE_HEIGHT,
            "maximum_height_gain": 0.60,
        },
    )
    upward_velocity = RewTerm(
        func=mdp.upward_velocity_window,
        weight=3.0,
        params={"start_s": 0.25, "end_s": 0.70, "maximum_speed": 4.0},
    )
    upright = RewTerm(
        func=mdp.upright_outside_rotation_window,
        weight=1.5,
        params={"rotation_start_s": 10.0, "rotation_end_s": 11.0},
    )
    takeoff = RewTerm(
        func=mdp.takeoff_success,
        weight=2.0,
        params={"command_name": "backflip_phase"},
    )
    mirrored_action = RewTerm(func=mdp.mirrored_action_l2, weight=-0.05)
    lateral_drift = RewTerm(func=mdp.lateral_drift_l2, weight=-0.25)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES)},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_hipp|.*_knee"
            ),
            "threshold": 5.0,
        },
    )


@configclass
class ToGoLFsBackflipRotateRewardsCfg(ToGoLFsBackflipJumpRewardsCfg):
    """Stage 2: retain the jump and acquire one backward rotation."""

    jump_height = RewTerm(
        func=mdp.jump_height,
        weight=12.0,
        params={
            "initial_height": INITIAL_BASE_HEIGHT,
            "maximum_height_gain": 0.60,
        },
    )
    upright = RewTerm(
        func=mdp.upright_outside_rotation_window,
        weight=1.0,
        params={
            "rotation_start_s": ROTATION_START_S,
            "rotation_end_s": ROTATION_END_S,
        },
    )
    pitch_rate = None
    centroidal_pitch_rate = RewTerm(
        func=mdp.centroidal_pitch_velocity,
        weight=30.0,
        params={
            "command_name": "backflip_phase",
            "start_s": ROTATION_START_S,
            "end_s": ROTATION_END_S,
            "maximum_rate": 10.0,
            "negative_rate_floor": -0.1,
            "minimum_peak_height": 0.45,
            "full_peak_height": 0.65,
            "minimum_current_height": 0.25,
            "full_current_height": 0.55,
        },
    )
    launch_centroidal_pitch_rate = RewTerm(
        func=mdp.centroidal_pitch_velocity,
        weight=80.0,
        params={
            "command_name": "backflip_phase",
            "start_s": TAKEOFF_IMPULSE_START_S,
            "end_s": TAKEOFF_IMPULSE_END_S,
            "maximum_rate": 15.0,
            "negative_rate_floor": -0.1,
            "minimum_peak_height": 0.25,
            "full_peak_height": 0.40,
            "minimum_current_height": 0.25,
            "full_current_height": 0.40,
            "require_takeoff": False,
            "require_support": True,
            "minimum_upward_speed": 0.10,
            "full_upward_speed": 0.80,
        },
    )
    takeoff_pitch_momentum = RewTerm(
        func=mdp.takeoff_backward_pitch_quality,
        weight=40.0,
        params={
            "command_name": "backflip_phase",
            "target_rate": 4.0,
            "reward_end_s": APEX_TARGET_S,
        },
    )
    launch_pitch_momentum = RewTerm(
        func=mdp.launch_backward_pitch_quality,
        weight=100.0,
        params={
            "command_name": "backflip_phase",
            "start_s": TAKEOFF_IMPULSE_START_S,
            "end_s": TAKEOFF_IMPULSE_END_S,
            "target_rate": 4.0,
            "minimum_upward_speed": 0.10,
            "full_upward_speed": 0.80,
        },
    )
    apex_half_rotation = RewTerm(
        func=mdp.apex_rotation_quality,
        weight=150.0,
        params={
            "command_name": "backflip_phase",
            "target_rotation": math.pi,
            "reward_end_s": ROTATION_END_S,
        },
    )
    early_rotation_progress = RewTerm(
        func=mdp.safe_backward_rotation_progress,
        weight=30.0,
        params={
            "command_name": "backflip_phase",
            "start_s": ROTATION_START_S,
            "end_s": APEX_TARGET_S,
            "minimum_peak_height": 0.45,
            "full_peak_height": 0.65,
            "minimum_current_height": 0.25,
            "full_current_height": 0.55,
            "maximum_rewarded_rotation": math.pi,
        },
    )
    rotation_progress = RewTerm(
        func=mdp.safe_backward_rotation_progress,
        weight=12.0,
        params={
            "command_name": "backflip_phase",
            "start_s": ROTATION_START_S,
            "end_s": ROTATION_END_S,
            "minimum_peak_height": 0.45,
            "full_peak_height": 0.65,
            "minimum_current_height": 0.25,
            "full_current_height": 0.55,
            "maximum_rewarded_rotation": 2.0 * math.pi,
        },
    )
    rotation_completion = RewTerm(
        func=mdp.rotation_completion_bonus,
        weight=80.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.75 * math.pi,
        },
    )
    rotation_milestone_1_25 = RewTerm(
        func=mdp.rotation_completion_bonus,
        weight=15.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.25 * math.pi,
        },
    )
    rotation_milestone_1_10 = RewTerm(
        func=mdp.rotation_completion_bonus,
        weight=8.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.10 * math.pi,
        },
    )
    rotation_milestone_1_15 = RewTerm(
        func=mdp.rotation_completion_bonus,
        weight=12.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.15 * math.pi,
        },
    )
    rotation_milestone_1_50 = RewTerm(
        func=mdp.rotation_completion_bonus,
        weight=20.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.50 * math.pi,
        },
    )
    rotation_milestone_1_60 = RewTerm(
        func=mdp.rotation_completion_bonus,
        weight=24.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.60 * math.pi,
        },
    )
    rotation_milestone_1_65 = RewTerm(
        func=mdp.rotation_completion_bonus,
        weight=30.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.65 * math.pi,
        },
    )
    airborne_clearance = RewTerm(
        func=mdp.airborne_height_clearance,
        weight=2.0,
        params={
            "command_name": "backflip_phase",
            "minimum_height": 0.25,
            "full_reward_height": 0.55,
        },
    )
    orientation_tracking = RewTerm(
        func=mdp.scheduled_orientation_tracking,
        weight=8.0,
        params={
            "command_name": "backflip_phase",
            "rotation_start_s": ROTATION_START_S,
            "rotation_end_s": ROTATION_END_S,
            "error_scale": 0.25,
        },
    )
    unwrapped_rotation_tracking = RewTerm(
        func=mdp.unwrapped_rotation_tracking,
        weight=40.0,
        params={
            "command_name": "backflip_phase",
            "rotation_start_s": ROTATION_START_S,
            "rotation_end_s": ROTATION_END_S,
            "error_scale": 4.0,
        },
    )
    wrong_direction = RewTerm(
        func=mdp.wrong_pitch_direction,
        weight=-0.5,
        params={"command_name": "backflip_phase", "maximum_rate": 15.0},
    )
    off_axis_rotation = RewTerm(
        func=mdp.off_axis_angular_velocity_l2,
        weight=-0.05,
    )
    premature_pitch_rate = RewTerm(
        func=mdp.premature_pitch_rate_l2,
        weight=-0.02,
        params={"rotation_start_s": TAKEOFF_IMPULSE_START_S},
    )
    termination_penalty = RewTerm(
        func=mdp.failure_before_rotation,
        weight=-5000.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.75 * math.pi,
            "term_keys": ["torso_contact", "base_too_low", "incomplete_touchdown"],
        },
    )


@configclass
class ToGoLFsBackflipEarlyRotateRewardsCfg(ToGoLFsBackflipJumpRewardsCfg):
    """Stage 2A: move backward rotation into takeoff and the rising flight."""

    upright = RewTerm(
        func=mdp.upright_outside_rotation_window,
        weight=1.0,
        params={
            "rotation_start_s": TAKEOFF_IMPULSE_START_S,
            "rotation_end_s": ROTATION_END_S,
        },
    )
    centroidal_pitch_rate = RewTerm(
        func=mdp.centroidal_pitch_velocity,
        weight=150.0,
        params={
            "command_name": "backflip_phase",
            "start_s": ROTATION_START_S,
            "end_s": APEX_TARGET_S,
            "maximum_rate": 15.0,
            "negative_rate_floor": -0.1,
            "minimum_peak_height": 0.45,
            "full_peak_height": 0.65,
            "minimum_current_height": 0.25,
            "full_current_height": 0.55,
        },
    )
    launch_centroidal_pitch_rate = RewTerm(
        func=mdp.centroidal_pitch_velocity,
        weight=200.0,
        params={
            "command_name": "backflip_phase",
            "start_s": TAKEOFF_IMPULSE_START_S,
            "end_s": TAKEOFF_IMPULSE_END_S,
            "maximum_rate": 15.0,
            "negative_rate_floor": -0.1,
            "minimum_peak_height": 0.25,
            "full_peak_height": 0.40,
            "minimum_current_height": 0.25,
            "full_current_height": 0.40,
            "require_takeoff": False,
            "require_support": True,
            "minimum_upward_speed": 0.10,
            "full_upward_speed": 0.80,
        },
    )
    takeoff_pitch_momentum = RewTerm(
        func=mdp.takeoff_backward_pitch_quality,
        weight=200.0,
        params={
            "command_name": "backflip_phase",
            "target_rate": 8.0,
            "reward_end_s": APEX_TARGET_S,
        },
    )
    launch_pitch_momentum = RewTerm(
        func=mdp.launch_backward_pitch_quality,
        weight=50.0,
        params={
            "command_name": "backflip_phase",
            "start_s": TAKEOFF_IMPULSE_START_S,
            "end_s": TAKEOFF_IMPULSE_END_S,
            "target_rate": 8.0,
            "minimum_upward_speed": 0.10,
            "full_upward_speed": 0.80,
        },
    )
    early_rotation_progress = RewTerm(
        func=mdp.safe_backward_rotation_progress,
        weight=200.0,
        params={
            "command_name": "backflip_phase",
            "start_s": ROTATION_START_S,
            "end_s": APEX_TARGET_S,
            "minimum_peak_height": 0.45,
            "full_peak_height": 0.65,
            "minimum_current_height": 0.25,
            "full_current_height": 0.55,
            "maximum_rewarded_rotation": math.pi,
        },
    )
    apex_half_rotation = RewTerm(
        func=mdp.apex_rotation_quality,
        weight=400.0,
        params={
            "command_name": "backflip_phase",
            "target_rotation": math.pi,
            "reward_end_s": ROTATION_END_S,
        },
    )
    early_milestone_half = RewTerm(
        func=mdp.rotation_before_deadline_bonus,
        weight=200.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 0.50 * math.pi,
            "deadline_s": APEX_TARGET_S,
        },
    )
    early_milestone_three_quarters = RewTerm(
        func=mdp.rotation_before_deadline_bonus,
        weight=400.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 0.75 * math.pi,
            "deadline_s": APEX_TARGET_S,
        },
    )
    early_milestone_near_half_turn = RewTerm(
        func=mdp.rotation_before_deadline_bonus,
        weight=800.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 0.95 * math.pi,
            "deadline_s": APEX_TARGET_S,
        },
    )
    rotation_retention = RewTerm(
        func=mdp.rotation_completion_bonus,
        weight=80.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.75 * math.pi,
        },
    )
    wrong_direction = RewTerm(
        func=mdp.wrong_pitch_direction,
        weight=-0.5,
        params={"command_name": "backflip_phase", "maximum_rate": 15.0},
    )
    off_axis_rotation = RewTerm(
        func=mdp.off_axis_angular_velocity_l2,
        weight=-0.05,
    )
    premature_pitch_rate = RewTerm(
        func=mdp.premature_pitch_rate_l2,
        weight=-0.02,
        params={"rotation_start_s": TAKEOFF_IMPULSE_START_S},
    )
    termination_penalty = RewTerm(
        func=mdp.failure_before_rotation,
        weight=-5000.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.75 * math.pi,
            "term_keys": ["torso_contact", "base_too_low", "incomplete_touchdown"],
        },
    )


@configclass
class ToGoLFsBackflipLandRewardsCfg(ToGoLFsBackflipRotateRewardsCfg):
    """Stage 3: complete one rotation, unfold, and settle on the feet.

    Landing is event-driven: there is deliberately no target height, apex
    angle, or prescribed airborne orientation schedule in this stage.
    """

    # Retain enough launch energy to leave the ground, but let PPO choose the
    # height. This is a smooth quality reward, not a takeoff/rotation gate.
    jump_height = RewTerm(
        func=mdp.jump_height,
        weight=3.0,
        params={
            "initial_height": INITIAL_BASE_HEIGHT,
            "maximum_height_gain": 0.60,
        },
    )
    upward_velocity = RewTerm(
        func=mdp.upward_velocity_window,
        weight=1.0,
        params={"start_s": 0.20, "end_s": 0.55, "maximum_speed": 4.0},
    )

    # Stage 2 used these terms to move rotation into takeoff. Once a complete
    # flip exists, fixed apex/time tracking conflicts with learning when to
    # unfold and absorb impact, so the landing stage removes it.
    centroidal_pitch_rate = None
    apex_half_rotation = None
    early_rotation_progress = None
    airborne_clearance = None
    orientation_tracking = None
    unwrapped_rotation_tracking = None

    # Keep only a weaker reminder to generate backward angular momentum while
    # the feet still have ground support. This does not constrain flight height.
    launch_centroidal_pitch_rate = RewTerm(
        func=mdp.centroidal_pitch_velocity,
        weight=20.0,
        params={
            "command_name": "backflip_phase",
            "start_s": TAKEOFF_IMPULSE_START_S,
            "end_s": TAKEOFF_IMPULSE_END_S,
            "maximum_rate": 15.0,
            "negative_rate_floor": -0.1,
            "minimum_peak_height": 0.25,
            "full_peak_height": 0.40,
            "minimum_current_height": 0.25,
            "full_current_height": 0.40,
            "require_takeoff": False,
            "require_support": True,
            "minimum_upward_speed": 0.10,
            "full_upward_speed": 0.80,
        },
    )
    takeoff_pitch_momentum = RewTerm(
        func=mdp.takeoff_backward_pitch_quality,
        weight=10.0,
        params={
            "command_name": "backflip_phase",
            "target_rate": 4.0,
            "reward_end_s": ROTATION_END_S,
        },
    )
    launch_pitch_momentum = RewTerm(
        func=mdp.launch_backward_pitch_quality,
        weight=30.0,
        params={
            "command_name": "backflip_phase",
            "start_s": TAKEOFF_IMPULSE_START_S,
            "end_s": TAKEOFF_IMPULSE_END_S,
            "target_rate": 4.0,
            "minimum_upward_speed": 0.10,
            "full_upward_speed": 0.80,
        },
    )

    termination_penalty = RewTerm(
        func=base_mdp.is_terminated_term,
        weight=-2000.0,
        params={
            "term_keys": ["torso_contact", "base_too_low", "incomplete_touchdown"]
        },
    )

    rotation_progress = RewTerm(
        func=mdp.capped_backward_rotation_progress,
        weight=12.0,
        params={
            "command_name": "backflip_phase",
            "maximum_rewarded_rotation": 2.0 * math.pi,
        },
    )
    rotation_completion = RewTerm(
        func=mdp.rotation_range_bonus,
        weight=40.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.75 * math.pi,
            "maximum_rotation": 2.25 * math.pi,
        },
    )

    # The source policy already flips. Capped progress plus the one-revolution
    # range replace these discovery-stage milestones and remove overspin gain.
    rotation_milestone_1_10 = None
    rotation_milestone_1_15 = None
    rotation_milestone_1_25 = None
    rotation_milestone_1_50 = None
    rotation_milestone_1_60 = None
    rotation_milestone_1_65 = None

    landing_contact = RewTerm(
        func=mdp.landing_contact_ratio,
        weight=40.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": MINIMUM_LANDING_ROTATION,
        },
    )
    landing_approach = RewTerm(
        func=mdp.landing_approach,
        weight=80.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.40 * math.pi,
            "target_rotation": 2.0 * math.pi,
            "orientation_error_scale": 0.50,
            "rotation_error_scale": 4.00,
            "angular_velocity_scale": 16.0,
        },
    )
    landing_leg_extension = RewTerm(
        func=mdp.landing_leg_extension,
        weight=30.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.40 * math.pi,
            "full_rotation": 1.75 * math.pi,
            "minimum_foot_drop": 0.10,
            "full_foot_drop": 0.25,
            "orientation_error_scale": 0.50,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=FOOT_LINK_NAME, preserve_order=True
            ),
        },
    )
    landing_stability = RewTerm(
        func=mdp.landing_stability,
        weight=120.0,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": MINIMUM_LANDING_ROTATION,
            "linear_velocity_scale": 1.0,
            "angular_velocity_scale": 4.0,
        },
    )
    rotation_accuracy = RewTerm(
        func=mdp.landing_rotation_accuracy,
        weight=30.0,
        params={"command_name": "backflip_phase", "error_scale": 0.5},
    )
    completed_backflip = RewTerm(
        func=mdp.completed_backflip,
        weight=200.0,
        params={"command_name": "backflip_phase"},
    )
    landing_foot_slip = RewTerm(
        func=mdp.landing_foot_slip_l2,
        weight=-0.10,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": MINIMUM_LANDING_ROTATION,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_LINK_NAME),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=FOOT_LINK_NAME
            ),
            "force_threshold": 2.0,
        },
    )


@configclass
class ToGoLFsBackflipTerminationsCfg:
    """Allow arbitrary pitch while terminating hard failures."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    torso_contact = DoneTerm(
        func=mdp.torso_contact_after_warmup,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=BASE_LINK_NAME
            ),
            "warmup_time_s": 0.15,
            "force_threshold": 10.0,
        },
    )
    base_too_low = DoneTerm(
        func=mdp.base_too_low_after_warmup,
        params={"minimum_height": 0.08, "warmup_time_s": 0.15},
    )


@configclass
class ToGoLFsBackflipRotateTerminationsCfg(ToGoLFsBackflipTerminationsCfg):
    """Reject any first touchdown that occurs before the aerial revolution."""

    incomplete_touchdown = DoneTerm(
        func=mdp.touchdown_before_rotation,
        params={
            "command_name": "backflip_phase",
            "minimum_rotation": 1.75 * math.pi,
        },
    )


@configclass
class ToGoLFsBackflipGenesisTerminationsCfg:
    """Match Genesis-backflip by allowing every state until the time limit."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class ToGoLFsBackflipJumpEnvCfg(ToGoLFsEnvCfg):
    """Stage-1 environment for jump acquisition with ordinary PPO."""

    randomize_initial_episode_length: bool = False
    scene: ToGoLFsBackflipSceneCfg = ToGoLFsBackflipSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    observations: ToGoLFsBackflipObservationsCfg = ToGoLFsBackflipObservationsCfg()
    actions: ToGoLFsBackflipActionsCfg = ToGoLFsBackflipActionsCfg()
    commands: ToGoLFsBackflipCommandsCfg = ToGoLFsBackflipCommandsCfg()
    rewards: ToGoLFsBackflipJumpRewardsCfg = ToGoLFsBackflipJumpRewardsCfg()
    terminations: ToGoLFsBackflipTerminationsCfg = ToGoLFsBackflipTerminationsCfg()
    events: ToGoLFsBackflipEventCfg = ToGoLFsBackflipEventCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = EPISODE_LENGTH_S
        self.is_finite_horizon = True
        self.decimation = 4
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.lazy_sensor_update = False
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class ToGoLFsBackflipGenesisEnvCfg(ToGoLFsBackflipJumpEnvCfg):
    """Independent end-to-end baseline derived from Genesis-backflip."""

    scene: ToGoLFsBackflipGenesisSceneCfg = ToGoLFsBackflipGenesisSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    commands: ToGoLFsBackflipGenesisCommandsCfg = (
        ToGoLFsBackflipGenesisCommandsCfg()
    )
    rewards: ToGoLFsBackflipGenesisRewardsCfg = ToGoLFsBackflipGenesisRewardsCfg()
    terminations: ToGoLFsBackflipGenesisTerminationsCfg = (
        ToGoLFsBackflipGenesisTerminationsCfg()
    )
    events: ToGoLFsBackflipGenesisEventCfg = ToGoLFsBackflipGenesisEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = GENESIS_EPISODE_LENGTH_S


@configclass
class ToGoLFsBackflipGenesisStrictEnvCfg(ToGoLFsBackflipGenesisEnvCfg):
    """Nominal LFS acquisition without Genesis randomization or action delay."""

    scene: ToGoLFsBackflipSceneCfg = ToGoLFsBackflipSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    rewards: ToGoLFsBackflipGenesisStrictRewardsCfg = (
        ToGoLFsBackflipGenesisStrictRewardsCfg()
    )
    terminations: ToGoLFsBackflipGenesisTerminationsCfg = (
        ToGoLFsBackflipGenesisTerminationsCfg()
    )
    events: ToGoLFsBackflipEventCfg = ToGoLFsBackflipEventCfg()


@configclass
class ToGoLFsBackflipGenesisLandingEnvCfg(
    ToGoLFsBackflipGenesisStrictEnvCfg
):
    """R8 nominal LFS environment for foot-first stable landing acquisition."""

    commands: ToGoLFsBackflipGenesisLandingCommandsCfg = (
        ToGoLFsBackflipGenesisLandingCommandsCfg()
    )
    rewards: ToGoLFsBackflipGenesisLandingRewardsCfg = (
        ToGoLFsBackflipGenesisLandingRewardsCfg()
    )


@configclass
class ToGoLFsBackflipGenesisLandingEnergyEnvCfg(
    ToGoLFsBackflipGenesisLandingEnvCfg
):
    """Temporary launch-energy curriculum with front-leg targets disabled."""

    rewards: ToGoLFsBackflipGenesisLandingEnergyRewardsCfg = (
        ToGoLFsBackflipGenesisLandingEnergyRewardsCfg()
    )


@configclass
class ToGoLFsBackflipGenesisLandingAssistedEnvCfg(
    ToGoLFsBackflipGenesisLandingEnvCfg
):
    """Landing task with isolated terminal front-leg stance deployment."""

    actions: ToGoLFsBackflipTerminalFrontActionsCfg = (
        ToGoLFsBackflipTerminalFrontActionsCfg()
    )
    rewards: ToGoLFsBackflipGenesisLandingAssistedRewardsCfg = (
        ToGoLFsBackflipGenesisLandingAssistedRewardsCfg()
    )

    def __post_init__(self):
        super().__post_init__()
        # The flip and touchdown still happen within the original 2.0 s
        # command horizon.  Keep one extra second only for verifying that the
        # post-touchdown support controller actually settles and stands.
        self.episode_length_s = 3.0


@configclass
class ToGoLFsBackflipRotateEnvCfg(ToGoLFsBackflipJumpEnvCfg):
    """Stage-2 environment for full backward rotation acquisition."""

    rewards: ToGoLFsBackflipRotateRewardsCfg = ToGoLFsBackflipRotateRewardsCfg()
    terminations: ToGoLFsBackflipRotateTerminationsCfg = (
        ToGoLFsBackflipRotateTerminationsCfg()
    )


@configclass
class ToGoLFsBackflipEarlyRotateEnvCfg(ToGoLFsBackflipJumpEnvCfg):
    """Stage-2A environment for establishing rotation before the apex."""

    rewards: ToGoLFsBackflipEarlyRotateRewardsCfg = (
        ToGoLFsBackflipEarlyRotateRewardsCfg()
    )


@configclass
class ToGoLFsBackflipEnvCfg(ToGoLFsBackflipRotateEnvCfg):
    """Stage-3 environment for stable one-backflip landing."""

    rewards: ToGoLFsBackflipLandRewardsCfg = ToGoLFsBackflipLandRewardsCfg()


@configclass
class ToGoLFsBackflipRobustEnvCfg(ToGoLFsBackflipEnvCfg):
    """Stage-4 environment with measured latency and narrow randomization."""

    scene: ToGoLFsBackflipRobustSceneCfg = ToGoLFsBackflipRobustSceneCfg(
        num_envs=4096, env_spacing=2.5
    )
    events: ToGoLFsBackflipRobustEventCfg = ToGoLFsBackflipRobustEventCfg()
