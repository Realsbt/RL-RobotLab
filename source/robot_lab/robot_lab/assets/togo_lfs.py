import math

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR
from robot_lab.assets.custom_actuator import PhysicalMotorCfg


TOGO_LFS_JOINT_NAMES = [
    "Jfl1_hipr", "Jfl2_hipp", "Jfl3_knee",
    "Jfr1_hipr", "Jfr2_hipp", "Jfr3_knee",
    "Jrl1_hipr", "Jrl2_hipp", "Jrl3_knee",
    "Jrr1_hipr", "Jrr2_hipp", "Jrr3_knee",
]

TOGO_LFS_MAX_JOINT_SPEED_RPM = 268.0
TOGO_LFS_MAX_JOINT_SPEED_RAD_S = TOGO_LFS_MAX_JOINT_SPEED_RPM * 2.0 * math.pi / 60.0
TOGO_LFS_COULOMB_FRICTION = 0.194
TOGO_LFS_VISCOUS_FRICTION = 0.007

TOGO_LFS_CFG = ArticulationCfg(
    prim_path=None,
    spawn=sim_utils.UrdfFileCfg(
        asset_path=(
            f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/xtellar/ToGo_LFs_v0p1_new/"
            "urdf/ToGo_LFs_v0p1_prototype_novlnk.urdf"
        ),
        fix_base=False,
        merge_fixed_joints=False,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.30),
        joint_pos={
            "Jfl1_hipr": 0.0,
            "Jfr1_hipr": 0.0,
            "Jrl1_hipr": 0.0,
            "Jrr1_hipr": 0.0,
            "Jfl2_hipp": -0.6,
            "Jfr2_hipp": 0.6,
            "Jrl2_hipp": -0.6,
            "Jrr2_hipp": 0.6,
            "Jfl3_knee": 1.0,
            "Jfr3_knee": -1.0,
            "Jrl3_knee": 1.0,
            "Jrr3_knee": -1.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hipr": PhysicalMotorCfg(
            joint_names_expr=[".*_hipr"],
            effort_limit=48.0,
            saturation_effort=48.0,
            velocity_limit=TOGO_LFS_MAX_JOINT_SPEED_RAD_S,
            stiffness=40.0,
            damping=2.0,
            armature=6.180843e-3,
            friction=TOGO_LFS_COULOMB_FRICTION,
            dynamic_friction=TOGO_LFS_COULOMB_FRICTION,
            viscous_friction=TOGO_LFS_VISCOUS_FRICTION,
            min_delay=1,
            max_delay=4,
            filter_tau=0.0,
            filter_tau_range=(0.005, 0.015),
            physics_dt=0.005,
        ),
        "hipp": PhysicalMotorCfg(
            joint_names_expr=[".*_hipp"],
            effort_limit=48.0,
            saturation_effort=48.0,
            velocity_limit=TOGO_LFS_MAX_JOINT_SPEED_RAD_S,
            stiffness=40.0,
            damping=2.0,
            armature=6.180843e-3,
            friction=TOGO_LFS_COULOMB_FRICTION,
            dynamic_friction=TOGO_LFS_COULOMB_FRICTION,
            viscous_friction=TOGO_LFS_VISCOUS_FRICTION,
            min_delay=1,
            max_delay=4,
            filter_tau=0.0,
            filter_tau_range=(0.005, 0.015),
            physics_dt=0.005,
        ),
        "knee": PhysicalMotorCfg(
            joint_names_expr=[".*_knee"],
            effort_limit=48.0,
            saturation_effort=48.0,
            velocity_limit=TOGO_LFS_MAX_JOINT_SPEED_RAD_S,
            stiffness=40.0,
            damping=2.0,
            armature=6.180843e-3,
            friction=TOGO_LFS_COULOMB_FRICTION,
            dynamic_friction=TOGO_LFS_COULOMB_FRICTION,
            viscous_friction=TOGO_LFS_VISCOUS_FRICTION,
            min_delay=1,
            max_delay=4,
            filter_tau=0.0,
            filter_tau_range=(0.005, 0.015),
            physics_dt=0.005,
        ),
    },
)
