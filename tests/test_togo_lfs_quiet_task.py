import importlib.util
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "source/robot_lab/robot_lab/tasks/togo_lfs_quiet"


def _load_mute_math():
    module_path = TASK_DIR / "mdp" / "mute_math.py"
    spec = importlib.util.spec_from_file_location("mute_math", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_quiet_task_is_registered_separately_from_base_locomotion():
    init_text = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")

    assert 'id="RobotLab-ToGo-LFs-Quiet-v0"' in init_text
    assert "robot_lab.tasks.togo_lfs_quiet.env_cfg:ToGoLFsQuietEnvCfg" in init_text
    assert "class ToGoLFsQuietEnvCfg(ToGoLFsEnvCfg)" in cfg_text
    assert "TOGO_LFS_CFG" not in cfg_text


def test_quiet_task_uses_fixed_low_speed_commands_without_curriculum():
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")

    assert "lin_vel_x=(-0.5, 0.5)" in cfg_text
    assert "lin_vel_y=(-0.15, 0.15)" in cfg_text
    assert "ang_vel_z=(-0.4, 0.4)" in cfg_text
    assert "smooth_command_range_curriculum" not in cfg_text
    assert "terrain_type=\"plane\"" in cfg_text


def test_training_entry_supports_cross_experiment_weight_initialization():
    train_text = (ROOT / "scripts/rsl_rl/train.py").read_text(encoding="utf-8")

    assert '"--pretrained_checkpoint"' in train_text
    assert "runner.load(pretrained_path, load_optimizer=False)" in train_text
    assert "runner.current_learning_iteration = 0" in train_text
    assert "Use either --resume or --pretrained_checkpoint, not both." in train_text


def test_quiet_task_uses_mute_fixed_beta_reward_weights():
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")

    assert "QUIET_FACTOR = 1.4" in cfg_text
    assert "VELOCITY_REWARD_SCALE = 1.0 - 0.2 * QUIET_FACTOR" in cfg_text
    assert "weight=-0.05 * QUIET_FACTOR" in cfg_text
    assert "weight=0.01 * QUIET_FACTOR" in cfg_text
    assert "weight=-2.5e-7" in cfg_text
    assert "weight=-0.1" in cfg_text
    assert "flat_orientation_l2, weight=-0.2" in cfg_text
    assert "base_linear_velocity = None" in cfg_text


def test_quiet_command_logs_impact_proxies_without_reward_terms():
    command_text = (TASK_DIR / "mdp" / "commands.py").read_text(encoding="utf-8")
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")

    assert "class MuteVelocityCommand(UniformVelocityCommand)" in command_text
    assert 'self.metrics["touchdown_velocity"]' in command_text
    assert 'self.metrics["phase_weighted_drop_velocity"]' in command_text
    assert 'self.metrics["peak_foot_contact_force"]' in command_text
    assert 'self.metrics["contact_foot_slip_speed"]' in command_text
    assert "self.previous_foot_velocity_z" in command_text
    assert "base_velocity = mdp.MuteVelocityCommandCfg(" in cfg_text


def test_quiet_command_supports_opt_in_noise_proxy_visualization():
    command_text = (TASK_DIR / "mdp" / "commands.py").read_text(encoding="utf-8")
    play_text = (ROOT / "scripts/rsl_rl/play.py").read_text(encoding="utf-8")

    assert "self.noise_proxy_speed" in command_text
    assert "weighted_drop_speed = torch.exp(0.5 * phase)" in command_text
    assert 'prim_path="/Visuals/MUTE/noise_proxy"' in command_text
    assert '"quiet": sim_utils.SphereCfg(' in command_text
    assert '"medium": sim_utils.SphereCfg(' in command_text
    assert '"loud": sim_utils.SphereCfg(' in command_text
    assert '"--quiet-noise-vis"' in play_text
    assert "base_velocity_cfg.debug_vis = True" in play_text


def test_contact_phase_interpolates_swing_and_stance():
    mute_math = _load_mute_math()
    air_time = torch.tensor([[0.0, 0.175, 0.35, 0.0]])
    contact_time = torch.tensor([[0.0, 0.0, 0.0, 0.175]])

    phase = mute_math.contact_phase(air_time, contact_time, 0.35, 0.35)

    torch.testing.assert_close(phase, torch.tensor([[0.0, 0.5, 1.0, 0.5]]))


def test_contact_phase_rejects_nonpositive_durations():
    mute_math = _load_mute_math()
    zeros = torch.zeros(1, 4)

    with pytest.raises(ValueError, match="must be positive"):
        mute_math.contact_phase(zeros, zeros, 0.0, 0.35)


def test_phase_weighting_penalizes_late_drop_and_rewards_early_raise():
    mute_math = _load_mute_math()
    foot_velocity_z = torch.tensor([[-1.0, -1.0, 1.0, 1.0]])
    phase = torch.tensor([[0.0, 1.0, 0.0, 1.0]])

    drop, raise_term = mute_math.phase_weighted_vertical_velocity(foot_velocity_z, phase)

    assert drop.item() == pytest.approx(1.0 + math.e)
    assert raise_term.item() == pytest.approx(1.0 + math.exp(-1.0))
