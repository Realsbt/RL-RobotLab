import importlib.util
import math
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "source/robot_lab/robot_lab/tasks/togo_lfs_backflip"
DEPLOY_CONFIG = ROOT / "deploy/deploy_mujoco/configs/togo_lfs_backflip.yaml"


def _load_backflip_math():
    module_path = TASK_DIR / "mdp" / "backflip_math.py"
    spec = importlib.util.spec_from_file_location("backflip_math", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_backflip_deploy_config_uses_repository_relative_assets():
    config_text = DEPLOY_CONFIG.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)

    assert "/home/" not in config_text
    assert config["policy_path"] == (
        "{ROOT_DIR}/deploy/pre_train/togo_lfs/backflip_r10j/policy.pt"
    )
    assert config["locomotion_policy_path"] == (
        "{ROOT_DIR}/deploy/pre_train/togo_lfs/locomotion_latency_50000/policy.pt"
    )
    assert config["xml_path"] == (
        "{ROOT_DIR}/resources/Robots/xtellar/ToGo_LFs_v0p1_new/"
        "simenv/mujoco/empty.xml"
    )


def test_backflip_tasks_are_isolated_and_use_ordinary_ppo():
    init_text = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
    runner_text = (TASK_DIR / "rsl_rl_cfg.py").read_text(encoding="utf-8")

    for task_id in (
        "RobotLab-ToGo-LFs-Backflip-Genesis-v0",
        "RobotLab-ToGo-LFs-Backflip-Genesis-Strict-v0",
        "RobotLab-ToGo-LFs-Backflip-Genesis-Strict-Consolidate-v0",
        "RobotLab-ToGo-LFs-Backflip-Genesis-Land-v0",
        "RobotLab-ToGo-LFs-Backflip-Jump-v0",
        "RobotLab-ToGo-LFs-Backflip-Early-Rotate-v0",
        "RobotLab-ToGo-LFs-Backflip-Rotate-v0",
        "RobotLab-ToGo-LFs-Backflip-v0",
        "RobotLab-ToGo-LFs-Backflip-Robust-v0",
    ):
        assert task_id in init_text
    assert 'entry_point="isaaclab.envs:ManagerBasedRLEnv"' in init_text
    assert "DMBotPPORunnerCfg" in runner_text
    assert "MoECTS" not in runner_text


def test_backflip_stages_keep_one_observation_and_action_contract():
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")

    assert "class ToGoLFsBackflipObservationsCfg:" in cfg_text
    assert cfg_text.count("observations: ToGoLFsBackflipObservationsCfg") == 1
    assert "class ToGoLFsBackflipRotateEnvCfg(ToGoLFsBackflipJumpEnvCfg)" in cfg_text
    assert "class ToGoLFsBackflipEnvCfg(ToGoLFsBackflipRotateEnvCfg)" in cfg_text
    assert "class ToGoLFsBackflipRobustEnvCfg(ToGoLFsBackflipEnvCfg)" in cfg_text
    assert "action_pair = ObsTerm(func=mdp.action_pair)" in cfg_text
    assert 'params={"command_name": "backflip_phase"}' in cfg_text
    assert "scale={\".*_hipr\": 0.5" in cfg_text


def test_skill_acquisition_keeps_torque_curve_but_removes_latency():
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")

    assert "TOGO_LFS_BACKFLIP_CFG = TOGO_LFS_CFG.copy()" in cfg_text
    assert "actuator_cfg.min_delay = 0" in cfg_text
    assert "actuator_cfg.max_delay = 0" in cfg_text
    assert "actuator_cfg.filter_tau_range = (0.0, 0.0)" in cfg_text
    assert "TOGO_LFS_BACKFLIP_ROBUST_CFG = TOGO_LFS_CFG.copy()" in cfg_text
    assert cfg_text.count("enabled_self_collisions = False") == 3


def test_backflip_termination_never_rejects_pitch_rotation():
    termination_text = (TASK_DIR / "mdp" / "terminations.py").read_text(encoding="utf-8")
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")

    assert "torso_contact_after_warmup" in termination_text
    assert "base_too_low_after_warmup" in termination_text
    assert "termination_if_pitch" not in termination_text
    assert "root_ang_vel_w" not in termination_text
    termination_cfg = cfg_text.split("class ToGoLFsBackflipTerminationsCfg:", 1)[1].split(
        "class ToGoLFsBackflipJumpEnvCfg", 1
    )[0]
    assert "termination_if_pitch" not in termination_cfg


def test_phase_features_have_expected_shape_and_endpoints():
    backflip_math = _load_backflip_math()
    features = backflip_math.multiscale_phase_features(
        torch.tensor([0.0, 1.8]), episode_length_s=1.8
    )

    assert features.shape == (2, 6)
    torch.testing.assert_close(features[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
    assert features[1, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert features[1, 1].item() == pytest.approx(1.0, abs=1e-6)


def test_genesis_phase_and_linear_rotation_reference_match_upstream_timing():
    backflip_math = _load_backflip_math()
    features = backflip_math.multiscale_phase_features(
        torch.tensor([0.0, 2.0]),
        episode_length_s=2.0,
        phase_cycles=0.5,
    )

    torch.testing.assert_close(
        features[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    )
    torch.testing.assert_close(
        features[1],
        torch.tensor([0.0, -1.0, 1.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]),
        atol=1e-6,
        rtol=0.0,
    )

    gravity = backflip_math.linear_rotation_projected_gravity(
        torch.tensor([0.50, 0.75, 1.00]),
        rotation_start_s=0.50,
        rotation_end_s=1.00,
        backward_pitch_sign=-1.0,
    )
    torch.testing.assert_close(
        gravity[0], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        gravity[1], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        gravity[2], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6, rtol=0.0
    )


def test_genesis_baseline_is_additive_and_uses_published_core_settings():
    init_text = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")
    reward_text = (TASK_DIR / "mdp" / "rewards.py").read_text(encoding="utf-8")
    command_text = (TASK_DIR / "mdp" / "commands.py").read_text(encoding="utf-8")
    runner_text = (TASK_DIR / "rsl_rl_cfg.py").read_text(encoding="utf-8")

    assert "RobotLab-ToGo-LFs-Backflip-Genesis-v0" in init_text
    assert "class ToGoLFsBackflipGenesisEnvCfg" in cfg_text
    assert "GENESIS_EPISODE_LENGTH_S = 2.0" in cfg_text
    assert "phase_cycles=0.5" in cfg_text
    assert "actuator_cfg.min_delay = 4" in cfg_text
    assert "actuator_cfg.max_delay = 4" in cfg_text
    assert "class ToGoLFsBackflipGenesisRewardsCfg" in cfg_text
    assert "func=mdp.genesis_backward_pitch_velocity" in cfg_text
    assert "func=mdp.genesis_orientation_error_l2" in cfg_text
    assert "func=mdp.genesis_feet_height_before_rotation" in cfg_text
    assert "weight=20.0" in cfg_text
    assert "weight=-30.0" in cfg_text
    assert "def genesis_backward_pitch_velocity(" in reward_text
    assert "def genesis_vertical_velocity(" in reward_text
    assert "def genesis_orientation_error_l2(" in reward_text
    assert "class GenesisPPORunnerCfg" in runner_text
    assert 'experiment_name = "togo_lfs_backflip_genesis_ppo"' in runner_text
    assert "max_iterations = 1000" in runner_text


def test_genesis_strict_nominal_variant_cannot_reward_ground_tumbling():
    init_text = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")
    reward_text = (TASK_DIR / "mdp" / "rewards.py").read_text(encoding="utf-8")
    runner_text = (TASK_DIR / "rsl_rl_cfg.py").read_text(encoding="utf-8")

    assert "RobotLab-ToGo-LFs-Backflip-Genesis-Strict-v0" in init_text
    assert "class ToGoLFsBackflipGenesisStrictEnvCfg" in cfg_text
    assert "scene: ToGoLFsBackflipSceneCfg" in cfg_text
    assert "events: ToGoLFsBackflipEventCfg" in cfg_text
    assert "class ToGoLFsBackflipGenesisStrictRewardsCfg" in cfg_text
    assert "func=mdp.genesis_backward_pitch_velocity_before_touchdown" in cfg_text
    assert "func=mdp.first_touchdown_rotation_quality" in cfg_text
    assert "func=mdp.takeoff_rotation_excess_l2" in cfg_text
    assert "def takeoff_rotation_excess_l2(" in reward_text
    assert "func=mdp.supported_rotation_l2" in cfg_text
    assert "def supported_rotation_l2(" in reward_text
    assert "state.backward_rotation_at_takeoff" in reward_text
    assert "weight=-200.0" in cfg_text
    assert "weight=-200.0" in cfg_text
    assert "& ~state.has_touched_down" in reward_text
    assert "1.5 * env.step_dt" in reward_text
    assert "state.signed_backward_rotation_at_touchdown / target_rotation" in reward_text
    assert "state.backward_rotation_at_takeoff" in reward_text
    assert "state.cfg.maximum_rotation_at_takeoff" in reward_text
    assert '"takeoff_excess_scale": 0.25' in cfg_text
    assert "class GenesisStrictPPORunnerCfg" in runner_text
    assert 'experiment_name = "togo_lfs_backflip_genesis_strict_ppo"' in runner_text


def test_genesis_landing_variant_preserves_clock_and_rewards_clean_feet():
    init_text = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")
    reward_text = (TASK_DIR / "mdp" / "rewards.py").read_text(encoding="utf-8")
    command_text = (TASK_DIR / "mdp" / "commands.py").read_text(encoding="utf-8")
    runner_text = (TASK_DIR / "rsl_rl_cfg.py").read_text(encoding="utf-8")

    assert "RobotLab-ToGo-LFs-Backflip-Genesis-Land-v0" in init_text
    assert "RobotLab-ToGo-LFs-Backflip-Genesis-Land-Energy-v0" in init_text
    assert "RobotLab-ToGo-LFs-Backflip-Genesis-Land-Assisted-v0" in init_text
    assert "class ToGoLFsBackflipGenesisLandingEnergyEnvCfg" in cfg_text
    assert "class ToGoLFsBackflipGenesisLandingEnergyRewardsCfg" in cfg_text
    assert "landing_foot_fore_aft_stance = None" in cfg_text
    assert "landing_front_leg_default_action = None" in cfg_text
    assert "class ToGoLFsBackflipGenesisLandingAssistedEnvCfg" in cfg_text
    action_text = (TASK_DIR / "mdp" / "actions.py").read_text(encoding="utf-8")
    assert "class TerminalFrontStanceAction(" in action_text
    assert "self._processed_actions[:, :6]" in action_text
    assert "front_target_offsets" in action_text
    assert "front_target = self._offset[:, :6]" in action_text
    assert "class ToGoLFsBackflipGenesisLandingCommandsCfg" in cfg_text
    assert "episode_length_s=GENESIS_EPISODE_LENGTH_S" in cfg_text
    assert "phase_cycles=0.5" in cfg_text
    assert "maximum_rotation_at_takeoff=0.55 * math.pi" in cfg_text
    assert "class ToGoLFsBackflipGenesisLandingRewardsCfg" in cfg_text
    assert "pitch_velocity = None" in cfg_text
    assert "func=mdp.first_touchdown_foot_contact_quality" in cfg_text
    assert "weight=-3000.0" in cfg_text
    assert "weight=5000.0" in cfg_text
    assert "func=mdp.landing_leg_extension" in cfg_text
    assert "func=mdp.landing_foot_fore_aft_stance_l2" in cfg_text
    assert "def landing_foot_fore_aft_stance_l2(" in reward_text
    assert '"front_target_x": 0.15' in cfg_text
    assert '"rear_target_x": -0.19' in cfg_text
    assert "func=mdp.landing_joint_pose_l2" in cfg_text
    assert "def landing_joint_pose_l2(" in reward_text
    assert "weight=-1000.0" in cfg_text
    assert "func=mdp.landing_foot_origin_clearance" in cfg_text
    assert "def landing_foot_origin_clearance(" in reward_text
    assert '"foot_rank": 2' in cfg_text
    assert '"full_margin": 0.08' in cfg_text
    assert '"minimum_rotation": 0.95 * math.pi' in cfg_text
    assert '"full_rotation": 1.30 * math.pi' in cfg_text
    assert '"foot_rank": 4' in cfg_text
    assert '"minimum_margin": -0.60' in cfg_text
    assert "func=mdp.landing_angular_speed_excess_l2" in cfg_text
    assert "def landing_angular_speed_excess_l2(" in reward_text
    assert "func=mdp.post_touchdown_clean_support" in cfg_text
    assert "func=mdp.post_touchdown_foot_origin_clearance" in cfg_text
    assert "func=mdp.landing_default_action_l2" in cfg_text
    assert "func=mdp.landing_retracted_leg_action_l2" in cfg_text
    assert "func=mdp.landing_front_leg_default_action_l2" in cfg_text
    assert "func=mdp.post_touchdown_motion_excess_l2" in cfg_text
    assert "func=mdp.post_touchdown_upright_error_l2" in cfg_text
    assert "func=mdp.post_touchdown_joint_pose_l2" in cfg_text
    assert "func=mdp.post_touchdown_nonfoot_contact" in cfg_text
    assert "def post_touchdown_clean_support(" in reward_text
    assert "def post_touchdown_foot_origin_clearance(" in reward_text
    assert "def landing_default_action_l2(" in reward_text
    assert "def landing_retracted_leg_action_l2(" in reward_text
    assert "def landing_front_leg_default_action_l2(" in reward_text
    assert "def post_touchdown_motion_excess_l2(" in reward_text
    assert "def post_touchdown_upright_error_l2(" in reward_text
    assert "def post_touchdown_joint_pose_l2(" in reward_text
    assert "def post_touchdown_nonfoot_contact(" in reward_text
    assert "touchdown_fourth_foot_origin_margin_m" in command_text
    assert "maximum_post_touchdown_contact_count" in command_text
    assert "maximum_clean_support_time_s" in command_text
    assert "touchdown_foot_origin_margin_by_leg_m" in command_text
    assert "touchdown_foot_forward_by_leg_m" in command_text
    assert "touchdown_action_l2_by_leg" in command_text
    assert '"minimum_foot_drop": -0.40' in cfg_text
    assert '"full_foot_drop": 0.20' in cfg_text
    assert "def first_touchdown_foot_contact_quality(" in reward_text
    assert "torch.sigmoid(" in reward_text
    assert 'torch.einsum(\n        "nfi,nij->nfj"' in reward_text
    assert "upright_blend = 0.25 + 0.75 * upright_quality" in reward_text
    assert "clean_foot_contact = ~state.touchdown_nonfoot_contact.bool()" in reward_text
    assert "class ToGoLFsBackflipGenesisLandingEnvCfg" in cfg_text
    assert "class GenesisLandingPPORunnerCfg" in runner_text
    assert 'experiment_name = "togo_lfs_backflip_genesis_land_ppo"' in runner_text
    assert "self.policy.init_noise_std = 0.10" in runner_text


def test_orientation_schedule_is_upright_inverted_upright():
    backflip_math = _load_backflip_math()
    gravity = backflip_math.desired_projected_gravity(
        torch.tensor([0.45, 0.80, 1.15]),
        rotation_start_s=0.45,
        rotation_end_s=1.15,
        backward_pitch_sign=-1.0,
    )

    torch.testing.assert_close(gravity[0], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(gravity[1], torch.tensor([0.0, 0.0, 1.0]), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(gravity[2], torch.tensor([0.0, 0.0, -1.0]), atol=1e-5, rtol=0.0)


def test_rotation_progress_only_rewards_new_maximum():
    backflip_math = _load_backflip_math()
    maximum, increment = backflip_math.positive_rotation_increment(
        torch.tensor([1.5, 0.5, -0.5]),
        torch.tensor([1.0, 1.0, 0.0]),
    )

    torch.testing.assert_close(maximum, torch.tensor([1.5, 1.0, 0.0]))
    torch.testing.assert_close(increment, torch.tensor([0.5, 0.0, 0.0]))


def test_projected_gravity_pitch_winding_is_signed_and_roll_gated():
    backflip_math = _load_backflip_math()
    theta = torch.tensor([0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi, 2.0 * math.pi])
    gravity = torch.stack((-torch.sin(theta), torch.zeros_like(theta), -torch.cos(theta)), dim=-1)
    previous = torch.zeros(1)
    total = torch.zeros(1)
    for sample in gravity:
        wrapped, delta = backflip_math.unwrapped_projected_gravity_pitch_increment(
            sample.unsqueeze(0), previous, backward_pitch_sign=-1.0
        )
        total += delta
        previous = wrapped
    assert total.item() == pytest.approx(2.0 * math.pi, abs=1e-5)

    _, invalid_delta = backflip_math.unwrapped_projected_gravity_pitch_increment(
        torch.tensor([[0.1, 0.99, 0.0]]), torch.zeros(1), minimum_xz_norm=0.5
    )
    assert invalid_delta.item() == 0.0


def test_clearance_scale_gates_unsafe_height_and_saturates():
    backflip_math = _load_backflip_math()
    scale = backflip_math.linear_clearance_scale(
        torch.tensor([0.20, 0.30, 0.40, 0.50, 0.60]),
        minimum_value=0.30,
        full_reward_value=0.50,
    )

    torch.testing.assert_close(scale, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))
    with pytest.raises(ValueError, match="greater than"):
        backflip_math.linear_clearance_scale(torch.tensor([0.4]), 0.5, 0.5)


def test_centroidal_velocity_matches_rigid_multibody_rotation():
    backflip_math = _load_backflip_math()
    masses = torch.ones(1, 2)
    positions = torch.tensor([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    linear_velocities = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]]])
    angular_velocities = torch.tensor([[[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]])
    inertias = torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1)

    velocity = backflip_math.centroidal_angular_velocity(
        masses,
        positions,
        linear_velocities,
        angular_velocities,
        inertias,
    )

    torch.testing.assert_close(velocity, torch.tensor([[0.0, 2.0, 0.0]]), atol=1e-6, rtol=0.0)


def test_rotation_stage_gates_progress_and_penalizes_bad_termination():
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")
    reward_text = (TASK_DIR / "mdp" / "rewards.py").read_text(encoding="utf-8")

    assert "func=mdp.safe_backward_rotation_progress" in cfg_text
    assert "func=mdp.centroidal_pitch_velocity" in cfg_text
    assert "weight=30.0" in cfg_text
    assert '"minimum_peak_height": 0.45' in cfg_text
    assert '"full_peak_height": 0.65' in cfg_text
    assert '"minimum_current_height": 0.25' in cfg_text
    assert "func=mdp.takeoff_backward_pitch_quality" in cfg_text
    assert '"target_rate": 4.0' in cfg_text
    assert "takeoff_pitch_momentum = RewTerm(" in cfg_text
    assert "weight=40.0" in cfg_text
    assert "func=mdp.launch_backward_pitch_quality" in cfg_text
    assert '"minimum_upward_speed": 0.10' in cfg_text
    assert '"full_upward_speed": 0.80' in cfg_text
    assert "weight=100.0" in cfg_text
    assert "func=mdp.apex_rotation_quality" in cfg_text
    assert '"target_rotation": math.pi' in cfg_text
    assert "weight=80.0" in cfg_text
    rotate_cfg = cfg_text.split(
        "class ToGoLFsBackflipRotateRewardsCfg", 1
    )[1].split("class ToGoLFsBackflipEarlyRotateRewardsCfg", 1)[0]
    assert "launch_centroidal_pitch_rate = RewTerm(" in rotate_cfg
    assert '"end_s": TAKEOFF_IMPULSE_END_S' in rotate_cfg
    assert "early_rotation_progress = RewTerm(" in rotate_cfg
    assert "weight=150.0" in rotate_cfg
    assert "func=mdp.failure_before_rotation" in cfg_text
    assert "incomplete = _state(env, command_name).max_backward_rotation" in reward_text
    assert "weight=-5000.0" in cfg_text
    assert "rotation_milestone_1_25" in cfg_text
    assert '"minimum_rotation": 1.25 * math.pi' in cfg_text
    assert "rotation_milestone_1_10" in cfg_text
    assert '"minimum_rotation": 1.10 * math.pi' in cfg_text
    assert "rotation_milestone_1_15" in cfg_text
    assert '"minimum_rotation": 1.15 * math.pi' in cfg_text
    assert "rotation_milestone_1_50" in cfg_text
    assert '"minimum_rotation": 1.50 * math.pi' in cfg_text
    assert "rotation_milestone_1_60" in cfg_text
    assert '"minimum_rotation": 1.60 * math.pi' in cfg_text
    assert "rotation_milestone_1_65" in cfg_text
    assert '"minimum_rotation": 1.65 * math.pi' in cfg_text
    assert "func=mdp.rotation_completion_bonus" in cfg_text
    assert '"minimum_rotation": 1.75 * math.pi' in cfg_text
    assert "weight=80.0" in cfg_text


def test_launch_pitch_reward_requires_support_and_upward_motion():
    reward_text = (TASK_DIR / "mdp" / "rewards.py").read_text(encoding="utf-8")

    launch_reward = reward_text.split("def launch_backward_pitch_quality", 1)[1].split(
        "def apex_rotation_quality", 1
    )[0]
    assert "supported = torch.any(state.contact, dim=-1)" in launch_reward
    assert "asset.data.root_lin_vel_w[:, 2]" in launch_reward
    assert "rate_quality * upward_quality * active" in launch_reward


def test_rotation_window_matches_measured_takeoff_and_touchdown():
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")

    assert "TAKEOFF_IMPULSE_START_S = 0.20" in cfg_text
    assert "ROTATION_START_S = 0.30" in cfg_text
    assert "APEX_TARGET_S = 0.56" in cfg_text
    assert "ROTATION_END_S = 0.82" in cfg_text
    assert 'params={"rotation_start_s": TAKEOFF_IMPULSE_START_S}' in cfg_text


def test_early_rotation_substage_only_rewards_progress_before_apex():
    init_text = (TASK_DIR / "__init__.py").read_text(encoding="utf-8")
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")
    reward_text = (TASK_DIR / "mdp" / "rewards.py").read_text(encoding="utf-8")
    runner_text = (TASK_DIR / "rsl_rl_cfg.py").read_text(encoding="utf-8")

    assert "RobotLab-ToGo-LFs-Backflip-Early-Rotate-v0" in init_text
    early_cfg = cfg_text.split(
        "class ToGoLFsBackflipEarlyRotateRewardsCfg", 1
    )[1].split("class ToGoLFsBackflipLandRewardsCfg", 1)[0]
    assert "early_rotation_progress = RewTerm(" in early_cfg
    assert "launch_centroidal_pitch_rate = RewTerm(" in early_cfg
    assert '"require_support": True' in early_cfg
    assert '"require_takeoff": False' in early_cfg
    assert '"end_s": TAKEOFF_IMPULSE_END_S' in early_cfg
    assert '"end_s": APEX_TARGET_S' in early_cfg
    assert '"maximum_rewarded_rotation": math.pi' in early_cfg
    assert "func=mdp.rotation_before_deadline_bonus" in early_cfg
    assert '"deadline_s": APEX_TARGET_S' in early_cfg
    assert '"minimum_rotation": 0.95 * math.pi' in early_cfg
    assert "weight=800.0" in early_cfg
    assert "rotation_retention = RewTerm(" in early_cfg
    assert '"minimum_rotation": 1.75 * math.pi' in early_cfg
    assert "weight=-5000.0" in early_cfg
    assert "before_deadline = _elapsed_s(env) <= deadline_s" in reward_text
    assert "active &= torch.any(state.contact, dim=-1)" in reward_text
    assert "* current_clearance * upward_quality" in reward_text
    assert "class EarlyRotatePPORunnerCfg(RotatePPORunnerCfg)" in runner_text
    assert 'experiment_name = "togo_lfs_backflip_early_rotate_ppo"' in runner_text
    assert "save_interval = 25" in runner_text
    assert "self.algorithm.learning_rate = 1.0e-4" in runner_text
    assert "self.algorithm.desired_kl = 0.005" in runner_text


def test_rotation_schedule_reaches_half_turn_near_expected_apex():
    backflip_math = _load_backflip_math()
    progress = backflip_math.rotation_schedule(
        torch.tensor([0.30, 0.56, 0.82]),
        rotation_start_s=0.30,
        rotation_end_s=0.82,
    )

    torch.testing.assert_close(progress, torch.tensor([0.0, 0.5, 1.0]))


def test_touchdown_and_credible_landing_are_separate_states():
    command_text = (TASK_DIR / "mdp" / "commands.py").read_text(encoding="utf-8")
    reward_text = (TASK_DIR / "mdp" / "rewards.py").read_text(encoding="utf-8")
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")

    assert "self.has_touched_down" in command_text
    assert "rotation_active = (" in command_text
    assert "& ~self.has_invalid_rotation_axis" in command_text
    assert "self.robot.data.root_ang_vel_b[:, 1]" in command_text
    assert "unwrapped_projected_gravity_pitch_increment" in command_text
    assert "minimum_pitch_xz_norm" in command_text
    assert "maximum_rotation_at_takeoff" in command_text
    assert 'self.metrics["backward_rotation_at_takeoff_rad"]' in command_text
    assert 'self.metrics["aerial_backward_rotation_at_touchdown_rad"]' in command_text
    assert "landed_now = (\n            first_touchdown_now" in command_text
    assert "self.cfg.minimum_landing_rotation" in command_text
    assert 'self.contact_sensor.find_bodies(".*"' in command_text
    assert "any_robot_contact = any_contact | self.nonfoot_contact" in command_text
    assert "self.was_airborne & any_robot_contact" in command_text
    assert "& ~self.nonfoot_contact" in command_text
    assert 'self.metrics["touchdown_success"][:] = self.has_touched_down.float()' in command_text
    assert "func=mdp.unwrapped_rotation_tracking" in cfg_text
    assert "state.signed_backward_rotation - desired_rotation" in reward_text
    assert "func=mdp.centroidal_pitch_velocity" in cfg_text
    assert "self.asset.root_physx_view.get_inertias()" in reward_text
    assert "active * peak_quality * current_clearance" in reward_text


def test_rotate_stages_reject_ground_tumble_as_aerial_rotation():
    command_text = (TASK_DIR / "mdp" / "commands.py").read_text(encoding="utf-8")
    termination_text = (TASK_DIR / "mdp" / "terminations.py").read_text(encoding="utf-8")
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")
    reward_text = (TASK_DIR / "mdp" / "rewards.py").read_text(encoding="utf-8")

    assert "def touchdown_before_rotation(" in termination_text
    assert "state.has_touched_down" in termination_text
    assert "state.max_backward_rotation < minimum_rotation" in termination_text
    assert "class ToGoLFsBackflipRotateTerminationsCfg" in cfg_text
    assert "func=mdp.touchdown_before_rotation" in cfg_text
    assert cfg_text.count('"incomplete_touchdown"') == 3
    assert "~state.has_touched_down" in reward_text
    assert "hard aerial boundary" in command_text


def test_landing_stage_is_event_driven_and_does_not_require_apex_height():
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")
    reward_text = (TASK_DIR / "mdp" / "rewards.py").read_text(encoding="utf-8")
    runner_text = (TASK_DIR / "rsl_rl_cfg.py").read_text(encoding="utf-8")

    land_cfg = cfg_text.split(
        "class ToGoLFsBackflipLandRewardsCfg", 1
    )[1].split("class ToGoLFsBackflipTerminationsCfg", 1)[0]
    assert "apex_half_rotation = None" in land_cfg
    assert "early_rotation_progress = None" in land_cfg
    assert "airborne_clearance = None" in land_cfg
    assert "orientation_tracking = None" in land_cfg
    assert "unwrapped_rotation_tracking = None" in land_cfg
    assert "func=mdp.capped_backward_rotation_progress" in land_cfg
    assert '"maximum_rewarded_rotation": 2.0 * math.pi' in land_cfg
    assert "func=mdp.rotation_range_bonus" in land_cfg
    assert '"maximum_rotation": 2.25 * math.pi' in land_cfg
    assert "def capped_backward_rotation_progress(" in reward_text
    assert "def rotation_range_bonus(" in reward_text
    assert "func=mdp.landing_approach" in land_cfg
    assert "func=mdp.landing_leg_extension" in land_cfg
    assert "def landing_approach(" in reward_text
    assert "def landing_leg_extension(" in reward_text
    assert "torch.square(gravity[:, 2] + 1.0)" in reward_text
    assert "contact_quality = state.contact_count.float() / 4.0" in reward_text
    assert "rotation = state.signed_backward_rotation" in reward_text
    assert "rotation_quality * braking_quality" in reward_text
    assert "class BackflipPPORunnerCfg" in runner_text
    assert "self.policy.init_noise_std = 0.10" in runner_text
    assert "self.algorithm.learning_rate = 5.0e-5" in runner_text


def test_phase_diagnostics_record_takeoff_and_flight_timing():
    command_text = (TASK_DIR / "mdp" / "commands.py").read_text(encoding="utf-8")

    assert 'self.metrics["takeoff_time_s"][:] = self.takeoff_time_s' in command_text
    assert 'self.metrics["apex_time_s"][:] = self.apex_time_s' in command_text
    assert 'self.metrics["backward_rotation_at_apex_rad"]' in command_text
    assert 'self.metrics["stable_landing_time_s"]' in command_text
    assert 'self.metrics["touchdown_mean_foot_drop_body_m"]' in command_text
    assert 'self.metrics["touchdown_mean_foot_drop_world_m"]' in command_text
    assert 'self.metrics["touchdown_joint_pose_error_rad"]' in command_text
    assert 'self.metrics["touchdown_knee_pose_error_rad"]' in command_text
    assert 'self.metrics["touchdown_hip_pitch_pose_error_rad"]' in command_text
    assert 'self.metrics["touchdown_first_foot_origin_margin_m"]' in command_text
    assert 'self.metrics["touchdown_second_foot_origin_margin_m"]' in command_text
    assert "self.cfg.minimum_stable_landing_time_s" in command_text
    assert "new_apex = (" in command_text
    assert 'self.metrics["first_touchdown_time_s"]' in command_text
    assert 'self.metrics["flight_time_s"][:] = self.flight_time_s' in command_text
    assert 'self.metrics["backward_pitch_rate_at_takeoff_rad_s"]' in command_text
    assert "first_touchdown_now = touchdown_now & ~self.has_touched_down" in command_text
    assert 'self.metrics["touchdown_upright_cosine"]' in command_text
    assert 'self.metrics["touchdown_vertical_speed_m_s"]' in command_text
    assert 'self.metrics["touchdown_angular_speed_rad_s"]' in command_text
    assert 'self.metrics["touchdown_nonfoot_contact"]' in command_text
    assert 'self.metrics["signed_backward_rotation_at_touchdown_rad"]' in command_text
    assert 'self.metrics["landing_rotation_success"]' in command_text


def test_training_resume_no_longer_requires_velocity_command():
    train_text = (ROOT / "scripts/rsl_rl/train.py").read_text(encoding="utf-8")

    assert 'getattr(env.unwrapped.cfg.commands, "base_velocity", None)' in train_text
    assert "cfg.commands.base_velocity" not in train_text


def test_reward_transfer_can_reset_critic_and_exploration_noise():
    train_text = (ROOT / "scripts/rsl_rl/train.py").read_text(encoding="utf-8")
    runner_text = (TASK_DIR / "rsl_rl_cfg.py").read_text(encoding="utf-8")

    assert '"--pretrained_actor_only"' in train_text
    assert '"--pretrained_action_std"' in train_text
    assert "Retained freshly initialized critic" in train_text
    assert "self.algorithm.learning_rate = 5.0e-5" in runner_text
    assert "self.algorithm.entropy_coef = 5.0e-4" in runner_text
    assert "self.algorithm.num_learning_epochs = 3" in runner_text
    assert "self.algorithm.desired_kl = 0.003" in runner_text


def test_one_shot_backflip_starts_at_phase_zero_and_requires_real_takeoff():
    cfg_text = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")
    command_text = (TASK_DIR / "mdp" / "commands.py").read_text(encoding="utf-8")
    train_text = (ROOT / "scripts/rsl_rl/train.py").read_text(encoding="utf-8")

    assert "randomize_initial_episode_length: bool = False" in cfg_text
    assert 'getattr(env.unwrapped.cfg, "randomize_initial_episode_length", True)' in train_text
    assert "self.has_support" in command_text
    assert "minimum_takeoff_height_gain" in command_text
    assert "minimum_takeoff_vertical_speed" in command_text
