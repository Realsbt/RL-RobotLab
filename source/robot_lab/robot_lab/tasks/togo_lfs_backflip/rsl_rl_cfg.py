"""Ordinary PPO configurations for staged LFS backflip training."""

from isaaclab.utils import configclass

from robot_lab.tasks.dmbot.rsl_rl_cfg import PPORunnerCfg as DMBotPPORunnerCfg


@configclass
class BackflipPPORunnerBaseCfg(DMBotPPORunnerCfg):
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}


@configclass
class GenesisPPORunnerCfg(BackflipPPORunnerBaseCfg):
    """Ordinary-PPO settings used by the public Genesis-backflip baseline."""

    experiment_name = "togo_lfs_backflip_genesis_ppo"
    max_iterations = 1000
    save_interval = 100


@configclass
class GenesisStrictPPORunnerCfg(GenesisPPORunnerCfg):
    """High-exploration ordinary PPO for strict nominal aerial acquisition."""

    experiment_name = "togo_lfs_backflip_genesis_strict_ppo"
    max_iterations = 2000
    save_interval = 50


@configclass
class GenesisStrictConservativePPORunnerCfg(GenesisStrictPPORunnerCfg):
    """Conservative consolidation after high-exploration skill discovery."""

    max_iterations = 3000
    save_interval = 25

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.20
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 5.0e-4
        self.algorithm.num_learning_epochs = 3
        self.algorithm.desired_kl = 0.005


@configclass
class GenesisLandingPPORunnerCfg(GenesisStrictConservativePPORunnerCfg):
    """Ordinary PPO for foot placement and impact absorption."""

    experiment_name = "togo_lfs_backflip_genesis_land_ppo"
    max_iterations = 3000
    save_interval = 25

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.10
        self.algorithm.learning_rate = 5.0e-6
        self.algorithm.entropy_coef = 1.0e-4
        self.algorithm.num_learning_epochs = 2
        self.algorithm.desired_kl = 2.0e-4


@configclass
class JumpPPORunnerCfg(BackflipPPORunnerBaseCfg):
    experiment_name = "togo_lfs_backflip_jump_ppo"
    max_iterations = 5000
    save_interval = 100


@configclass
class RotatePPORunnerCfg(BackflipPPORunnerBaseCfg):
    experiment_name = "togo_lfs_backflip_rotate_ppo"
    max_iterations = 10000
    save_interval = 100

    def __post_init__(self):
        super().__post_init__()
        # Stage 2 is normally initialized from an already functional jump or
        # rotation actor.  Conservative PPO updates prevent the timing-shaping
        # rewards from destroying the acquired full-revolution behavior.
        self.policy.init_noise_std = 0.2
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 5.0e-4
        self.algorithm.num_learning_epochs = 3
        self.algorithm.desired_kl = 0.003


@configclass
class EarlyRotatePPORunnerCfg(RotatePPORunnerCfg):
    experiment_name = "togo_lfs_backflip_early_rotate_ppo"
    max_iterations = 2000
    save_interval = 25

    def __post_init__(self):
        super().__post_init__()
        # This short curriculum stage must leave the already-converged late
        # rotation optimum.  Use ordinary PPO with moderately larger updates;
        # the following full-rotation stage switches back to conservative PPO.
        self.policy.init_noise_std = 0.15
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 5.0e-4
        self.algorithm.num_learning_epochs = 3
        self.algorithm.desired_kl = 0.005


@configclass
class BackflipPPORunnerCfg(BackflipPPORunnerBaseCfg):
    experiment_name = "togo_lfs_backflip_land_ppo"
    max_iterations = 3000
    save_interval = 25

    def __post_init__(self):
        super().__post_init__()
        # Landing starts from a functional rotation actor. Small ordinary-PPO
        # updates and low exploration preserve the flip while foot placement
        # and impact absorption are acquired.
        self.policy.init_noise_std = 0.10
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 2.0e-4
        self.algorithm.num_learning_epochs = 3
        self.algorithm.desired_kl = 0.003


@configclass
class RobustPPORunnerCfg(BackflipPPORunnerBaseCfg):
    experiment_name = "togo_lfs_backflip_robust_ppo"
    max_iterations = 5000
    save_interval = 100
