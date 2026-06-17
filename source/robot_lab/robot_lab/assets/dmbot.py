import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from robot_lab.assets.custom_actuator import PhysicalMotorCfg  
from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

DM_CFG = ArticulationCfg(
    prim_path=None,
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/dm/OpenDog_novlnk_stance/OpenDog_novlnk.usd",
        activate_contact_sensors=True,
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4,
            # fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
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
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
        actuators={
        "hipr": PhysicalMotorCfg(
            joint_names_expr=[".*_hipr"],
            effort_limit=30.0,
            saturation_effort=97.0,
            velocity_limit=6.28,
            stiffness=40.0,
            damping=2.0,
            armature=0.139532544,
            friction=0.1,
            min_delay=2,          # 物理步
            max_delay=6,          # 2-6步 = 10~30ms
            filter_tau=0.0,       # 先关掉，只验证延迟
            physics_dt=0.005,     # 必须 == sim.dt
        ),
        "hipp": PhysicalMotorCfg(
            joint_names_expr=[".*_hipp"],
            effort_limit=30.0,
            saturation_effort=97.0,
            velocity_limit=6.28,
            stiffness=40.0,
            damping=2.0,
            armature=0.139532544,
            friction=0.1,
            min_delay=2,
            max_delay=6,
            filter_tau=0.0,
            physics_dt=0.005,
        ),
        "knee": PhysicalMotorCfg(
            joint_names_expr=[".*_knee"],
            effort_limit=30.0,
            saturation_effort=97.0,
            velocity_limit=6.28,
            stiffness=40.0,
            damping=2.0,
            armature=0.139532544,
            friction=0.1,
            min_delay=2,
            max_delay=6,
            filter_tau=0.0,
            physics_dt=0.005,
        ),
    },
    # actuators={
    #     "Hip": DelayedPDActuatorCfg(
    #         joint_names_expr=[".*_hip[r,p]"],
    #         effort_limit=23.7,
    #         velocity_limit=15.7,
    #         stiffness=25.0,
    #         damping=0.5,
    #         friction=0.1,
    #         armature=0.139532544,
    #         min_delay=0,
    #         max_delay=1,
    #     ),
    #     "Knee": DelayedPDActuatorCfg(
    #         joint_names_expr=[".*_knee"],
    #         effort_limit=23.7,
    #         velocity_limit=15.7,
    #         stiffness=25.0,
    #         damping=0.5,
    #         friction=0.1,
    #         armature=0.139532544,
    #         min_delay=0,
    #         max_delay=1,
    #     ),
    # },
)
