from isaaclab.utils import configclass

from robot_lab.tasks.togo_lfs.rsl_rl_cfg import (
    MoECTSCatELURunnerCfg as ToGoLFsMoECTSCatELURunnerCfg,
    MoECTSRunnerCfg as ToGoLFsMoECTSRunnerCfg,
    PPORunnerCfg as ToGoLFsPPORunnerCfg,
)


@configclass
class PPORunnerCfg(ToGoLFsPPORunnerCfg):
    experiment_name = "togo_lfs_quiet_ppo"
    max_iterations = 10000


@configclass
class MoECTSRunnerCfg(ToGoLFsMoECTSRunnerCfg):
    experiment_name = "togo_lfs_quiet_moe_cts"
    max_iterations = 10000


@configclass
class MoECTSCatELURunnerCfg(ToGoLFsMoECTSCatELURunnerCfg):
    experiment_name = "togo_lfs_quiet_moe_cts"
    max_iterations = 10000
