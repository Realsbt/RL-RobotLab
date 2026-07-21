# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Event-based quiet-impact diagnostics for the Xtellar ToGo_LFs robot."""

import gymnasium as gym

from isaaclab_tasks.utils import import_packages


gym.register(
    id="RobotLab-ToGo-LFs-Quiet-Impact-v1",
    entry_point="robot_lab.tasks.dmbot.env.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "robot_lab.tasks.togo_lfs_quiet_impact.env_cfg:ToGoLFsQuietImpactEnvCfg",
        "rsl_rl_cfg_entry_point": "robot_lab.tasks.togo_lfs_quiet_impact.rsl_rl_cfg:MoECTSRunnerCfg",
    },
)

_BLACKLIST_PKGS = ["utils"]
import_packages(__name__, _BLACKLIST_PKGS)
