from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "source/robot_lab/robot_lab/tasks/dmbot"


def _subterrain_entry(active_cfg: str, name: str) -> str:
    marker = f'        "{name}":'
    start = active_cfg.index(marker)
    next_start = active_cfg.find('\n        "', start + len(marker))
    if next_start == -1:
        return active_cfg[start:]
    return active_cfg[start:next_start]


def test_dmbot_task_registration_points_to_dmbot_config():
    init_text = (TASK_DIR / "__init__.py").read_text()

    assert 'id="RobotLab-DMBot-v0"' in init_text
    assert 'entry_point="robot_lab.tasks.dmbot.env.go2_env:Go2Env"' in init_text
    assert "robot_lab.tasks.dmbot.env_cfg:DMBotEnvCfg" in init_text
    assert "robot_lab.tasks.dmbot.rsl_rl_cfg:MoECTSRunnerCfg" in init_text
    assert "RobotLab-Go2-v0" not in init_text


def test_dmbot_env_config_uses_dmbot_asset_and_names():
    env_text = (TASK_DIR / "env_cfg.py").read_text()

    assert "import robot_lab.tasks.dmbot.mdp as mdp" in env_text
    assert "from robot_lab.assets.dmbot import DM_CFG" in env_text
    assert "from robot_lab.tasks.dmbot.mdp.terrains import TERRAIN_CFG" in env_text
    assert "class DMBotSceneCfg" in env_text
    assert "class DMBotEnvCfg" in env_text
    assert 'BASE_LINK_NAME = "L0_torso"' in env_text
    assert 'FOOT_LINK_NAME = ".*_foot"' in env_text
    assert "BASE_HEIGHT_TARGET = 0.35" in env_text
    assert "max_init_terrain_level=5" in env_text
    assert 'prim_path="{ENV_REGEX_NS}/Robot/L0_torso"' in env_text
    assert 'robot: ArticulationCfg = DM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")' in env_text
    assert '"Jfl1_hipr", "Jfl2_hipp", "Jfl3_knee"' in env_text
    assert '".*_hipr": 0.25' in env_text
    assert '".*_hipp": 0.25' in env_text
    assert '".*_knee": 0.25' in env_text
    assert 'joint_names=".*_hipr"' in env_text
    assert 'joint_names=".*_(hipp|knee)"' in env_text
    assert "FL_hip_joint" not in env_text
    assert "GO2_CFG_UNITREE" not in env_text


def test_dmbot_runner_uses_separate_experiment_name():
    rsl_text = (TASK_DIR / "rsl_rl_cfg.py").read_text()

    assert 'experiment_name = "dmbot_moe_cts"' in rsl_text
    assert 'experiment_name = "go2_moe_cts"' not in rsl_text





def test_dmbot_terrain_matches_go2_original_strategy():
    terrain_text = (TASK_DIR / "mdp" / "terrains.py").read_text()
    active_cfg = terrain_text.split("TERRAIN_CFG = Go2TerrainGeneratorCfg(", 1)[1]

    expected_proportions = {
        "wave": "proportion=0.05",
        "slope_up": "proportion=0.10",
        "slope_down": "proportion=0.10",
        "rough_slope": "proportion=0.05",
        "stairs_up": "proportion=0.25",
        "stairs_down": "proportion=0.10",
        "obstacles": "proportion=0.20",
        "stepping_stones": "proportion=0.0",
        "gap": "proportion=0.0",
        "flat": "proportion=0.15",
    }
    for terrain_name, proportion in expected_proportions.items():
        assert proportion in _subterrain_entry(active_cfg, terrain_name)

    assert "slope_range=(0.1, 0.568)" in active_cfg
    assert "step_height_range=(0.05, 0.257)" in active_cfg
    assert "obstacle_height_range=(0.05, 0.275)" in active_cfg
    assert "stone_width_range=(0.075, 1.575)" in active_cfg
    assert "gap_width_range=(0.0, 0.9)" in active_cfg
    assert '"random_rough": terrain_gen.HfRandomUniformTerrainCfg(' not in active_cfg


def test_dmbot_command_curriculum_uses_continuous_quintic_smoothing():
    command_text = (TASK_DIR / "mdp" / "commands.py").read_text()

    assert "Ranges:" in command_text
    assert "lin_vel_x: tuple[float, float] = [-0.5, 0.5]" in command_text
    assert "lin_vel_y: tuple[float, float] = [-0.5, 0.5]" in command_text
    assert "ang_vel_yaw: tuple[float, float] = [-1.0, 1.0]" in command_text
    assert "smooth_command_range_curriculum: dict | None" in command_text
    assert '"nodes": [' in command_text
    assert '"iter": 0' in command_text
    assert '"iter": 20000' in command_text
    assert '"iter": 35000' in command_text
    assert '"iter": 60000' in command_text
    assert '"iter": 100000' in command_text
    assert '"lin_vel_x": [-0.75, 0.75]' in command_text
    assert '"lin_vel_x": [-1.0, 1.0]' in command_text
    assert '"lin_vel_x": [-1.5, 1.5]' in command_text
    assert '"lin_vel_x": [-2.0, 2.0]' in command_text
    assert '"lin_vel_y": [-1.0, 1.0]' in command_text
    assert '"ang_vel_yaw": [-2.0, 2.0]' in command_text
    assert "eased_progress = progress**3 * (progress * (progress * 6.0 - 15.0) + 10.0)" in command_text
    assert "segment={start_node['iter']}-{end_node['iter']}" in command_text
    assert "command_range_curriculum: list[dict] = []" in command_text
    assert (
        "zero_command_curriculum: dict = "
        "{'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.15}"
    ) in command_text
    assert "'slope_up':\n            {'lin_vel_x': [-1.5, 1.5]" in command_text
    assert "'rough_slope':\n            {'lin_vel_x': [-1.5, 1.5]" in command_text
    assert "'stairs_up':\n            {'lin_vel_x': [-1.0, 1.0]" in command_text
    assert "'flat':\n            {'lin_vel_x': [-2.0, 2.0]" in command_text


def test_dmbot_auxiliary_curricula_match_go2_original_strategy():
    env_text = (TASK_DIR / "env_cfg.py").read_text()

    assert "lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)" in env_text
    assert "base_height_l2 = RewTerm(\n        func=mdp.base_height_l2,\n        weight=-1.0" in env_text
    assert "base_linear_velocity = CurrTerm(mdp.gradual_reward_weight_modification" in env_text
    assert "base_height_l2 = CurrTerm(mdp.gradual_reward_weight_modification" in env_text
    assert '"term_name": "lin_vel_z_l2", "initial_weight": -2.0, "final_weight": -0.0' in env_text
    assert '"term_name": "base_height_l2", "initial_weight": -1.0, "final_weight": -10.0' in env_text


def test_dmbot_torso_illegal_contact_threshold_allows_light_brushes():
    env_text = (TASK_DIR / "env_cfg.py").read_text()
    init_text = (TASK_DIR / "mdp" / "__init__.py").read_text()
    termination_text = (TASK_DIR / "mdp" / "terminations.py").read_text()

    assert "from .terminations import *" in init_text
    assert "def illegal_contact_consecutive(" in termination_text
    assert "torch.all(frame_has_contact, dim=1)" in termination_text
    assert "illegal_contact = DoneTerm(" in env_text
    assert "func=mdp.illegal_contact_consecutive" in env_text
    assert '"sensor_cfg": SceneEntityCfg("contact_forces", body_names=BASE_LINK_NAME)' in env_text
    assert '"threshold": 10.0' in env_text
    assert '"consecutive_frames": 3' in env_text


def test_dmbot_zero_command_base_motion_penalty_is_configured():
    env_text = (TASK_DIR / "env_cfg.py").read_text()
    reward_text = (TASK_DIR / "mdp" / "rewards.py").read_text()

    assert "def standstill_base_motion_l2(" in reward_text
    assert "planar_motion = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)" in reward_text
    assert "yaw_motion = yaw_rate_scale * torch.square(asset.data.root_ang_vel_b[:, 2])" in reward_text
    assert "standstill_base_motion_l2 = RewTerm(" in env_text
    assert "func=mdp.standstill_base_motion_l2" in env_text
    assert "weight=-3.0" in env_text
    assert '"yaw_rate_scale": 0.25' in env_text


def test_dmbot_zero_command_foot_slide_penalty_is_configured():
    env_text = (TASK_DIR / "env_cfg.py").read_text()
    reward_text = (TASK_DIR / "mdp" / "rewards.py").read_text()
    command_text = (TASK_DIR / "mdp" / "commands.py").read_text()

    assert "'end_value': 0.15" in command_text
    assert "def standstill_foot_slide(" in reward_text
    assert "asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]" in reward_text
    assert "foot_xy_speed * contacts" in reward_text
    assert "standstill_foot_slide = RewTerm(" in env_text
    assert "func=mdp.standstill_foot_slide" in env_text
    assert "weight=-0.05" in env_text
