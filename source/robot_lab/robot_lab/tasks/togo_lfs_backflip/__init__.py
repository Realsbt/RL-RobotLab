# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""Staged ordinary-PPO backflip tasks for the Xtellar ToGo_LFs robot."""

import gymnasium as gym

from isaaclab_tasks.utils import import_packages


_TASKS = (
    (
        "RobotLab-ToGo-LFs-Backflip-Genesis-v0",
        "ToGoLFsBackflipGenesisEnvCfg",
        "GenesisPPORunnerCfg",
    ),
    (
        "RobotLab-ToGo-LFs-Backflip-Genesis-Strict-v0",
        "ToGoLFsBackflipGenesisStrictEnvCfg",
        "GenesisStrictPPORunnerCfg",
    ),
    (
        "RobotLab-ToGo-LFs-Backflip-Genesis-Strict-Consolidate-v0",
        "ToGoLFsBackflipGenesisStrictEnvCfg",
        "GenesisStrictConservativePPORunnerCfg",
    ),
    (
        "RobotLab-ToGo-LFs-Backflip-Genesis-Land-v0",
        "ToGoLFsBackflipGenesisLandingEnvCfg",
        "GenesisLandingPPORunnerCfg",
    ),
    (
        "RobotLab-ToGo-LFs-Backflip-Genesis-Land-Energy-v0",
        "ToGoLFsBackflipGenesisLandingEnergyEnvCfg",
        "GenesisLandingPPORunnerCfg",
    ),
    (
        "RobotLab-ToGo-LFs-Backflip-Genesis-Land-Assisted-v0",
        "ToGoLFsBackflipGenesisLandingAssistedEnvCfg",
        "GenesisLandingPPORunnerCfg",
    ),
    (
        "RobotLab-ToGo-LFs-Backflip-Jump-v0",
        "ToGoLFsBackflipJumpEnvCfg",
        "JumpPPORunnerCfg",
    ),
    (
        "RobotLab-ToGo-LFs-Backflip-Early-Rotate-v0",
        "ToGoLFsBackflipEarlyRotateEnvCfg",
        "EarlyRotatePPORunnerCfg",
    ),
    (
        "RobotLab-ToGo-LFs-Backflip-Rotate-v0",
        "ToGoLFsBackflipRotateEnvCfg",
        "RotatePPORunnerCfg",
    ),
    (
        "RobotLab-ToGo-LFs-Backflip-v0",
        "ToGoLFsBackflipEnvCfg",
        "BackflipPPORunnerCfg",
    ),
    (
        "RobotLab-ToGo-LFs-Backflip-Robust-v0",
        "ToGoLFsBackflipRobustEnvCfg",
        "RobustPPORunnerCfg",
    ),
)

for task_id, env_cfg_name, runner_cfg_name in _TASKS:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"robot_lab.tasks.togo_lfs_backflip.env_cfg:{env_cfg_name}"
            ),
            "rsl_rl_cfg_entry_point": (
                f"robot_lab.tasks.togo_lfs_backflip.rsl_rl_cfg:{runner_cfg_name}"
            ),
        },
    )

_BLACKLIST_PKGS = ["utils"]
import_packages(__name__, _BLACKLIST_PKGS)
