from isaaclab.utils import configclass

from robot_lab.tasks.dmbot.rsl_rl_cfg import (
    MoECTSCatELURunnerCfg as DMBotMoECTSCatELURunnerCfg,
    MoECTSRunnerCfg as DMBotMoECTSRunnerCfg,
    PPORunnerCfg as DMBotPPORunnerCfg,
)


@configclass
class PPORunnerCfg(DMBotPPORunnerCfg):
    experiment_name = "togo_lfs_rough"


@configclass
class MoECTSRunnerCfg(DMBotMoECTSRunnerCfg):
    experiment_name = "togo_lfs_moe_cts"


@configclass
class MoECTSCatELURunnerCfg(DMBotMoECTSCatELURunnerCfg):
    experiment_name = "togo_lfs_moe_cts"
