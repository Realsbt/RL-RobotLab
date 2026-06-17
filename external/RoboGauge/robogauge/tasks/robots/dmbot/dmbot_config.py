# -*- coding: utf-8 -*-
'''
@File    : dmbot_config.py
@Desc    : DMBot Robot Configuration
'''
from typing import Literal

from robogauge.tasks.robots import RobotConfig


class DMBotConfig(RobotConfig):
    robot_name = 'dmbot'
    robot_class = 'DMBot'

    class assets:
        robot_xml = "{ROBOGAUGE_ROOT_DIR}/resources/robots/dmbot/dmbot.xml"
        base_body_name = 'L0_torso'
        robot_spawn_height = 0.35  # z [m]
        foot_geom_names = ['FL', 'FR', 'RL', 'RR']

    class control(RobotConfig.control):
        device = 'cpu'
        model_path = "{ROBOGAUGE_ROOT_DIR}/resources/models/dmbot/dmbot.pt"
        control_dt = 0.02  # 50 Hz
        control_type = 'P'  # Position control
        support_goal: Literal['velocity', 'position'] = 'velocity'

        p_gains = [40.0] * 12  # [N*m/rad]
        d_gains = [2.0] * 12  # [N*m*s/rad]

        num_observations = 45
        num_actions = 12

        default_dof_pos = [
            0.0, -0.8, -1.5,
            0.0, 0.8, 1.5,
            0.0, -1.0, -1.5,
            0.0, 1.0, 1.5,
        ]

        mj2model_dof_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        save_additional_output = False

        class scales(RobotConfig.control.scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            action = 0.25
            cmd = [1.0, 1.0, 1.0]

    class commands(RobotConfig.commands):
        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y = [-1.0, 1.0]  # min max [m/s]
        lin_vel_z = None  # min max [m/s]
        ang_vel_roll = None  # min max [rad/s]
        ang_vel_pitch = None  # min max [rad/s]
        ang_vel_yaw = [-2.0, 2.0]  # min max [rad/s]


class DMBotTerrainConfig(DMBotConfig):
    """DMBot Robot Configuration for Terrain Tasks."""

    class commands(DMBotConfig.commands):
        lin_vel_x = [-1.0, 1.0]  # min max [m/s]
        lin_vel_y = [-1.0, 1.0]  # min max [m/s]
        ang_vel_yaw = [-1.5, 1.5]  # min max [rad/s]
