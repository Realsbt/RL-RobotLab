# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""ToGo_LFs task registration."""

import gymnasium as gym

from isaaclab_tasks.utils import import_packages


gym.register(
    id="RobotLab-ToGo-LFs-v0",
    entry_point="robot_lab.tasks.dmbot.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robot_lab.tasks.togo_lfs.env_cfg:ToGoLFsEnvCfg",
        "rsl_rl_cfg_entry_point": "robot_lab.tasks.togo_lfs.rsl_rl_cfg:MoECTSRunnerCfg",
    },
)

_BLACKLIST_PKGS = ["utils"]
import_packages(__name__, _BLACKLIST_PKGS)
