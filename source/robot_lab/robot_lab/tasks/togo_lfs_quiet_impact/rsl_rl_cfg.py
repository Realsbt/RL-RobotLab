from isaaclab.utils import configclass

from robot_lab.tasks.togo_lfs_quiet.rsl_rl_cfg import (
    MoECTSCatELURunnerCfg as QuietMoECTSCatELURunnerCfg,
    MoECTSRunnerCfg as QuietMoECTSRunnerCfg,
    PPORunnerCfg as QuietPPORunnerCfg,
)


@configclass
class PPORunnerCfg(QuietPPORunnerCfg):
    experiment_name = "togo_lfs_quiet_impact_ppo"


@configclass
class MoECTSRunnerCfg(QuietMoECTSRunnerCfg):
    experiment_name = "togo_lfs_quiet_impact_moe_cts"


@configclass
class MoECTSCatELURunnerCfg(QuietMoECTSCatELURunnerCfg):
    experiment_name = "togo_lfs_quiet_impact_moe_cts"
