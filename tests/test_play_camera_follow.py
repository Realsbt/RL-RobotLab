from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rsl_rl/play.py"


def test_play_script_follows_robot_camera_by_default():
    text = SCRIPT.read_text()

    assert "--no-camera-follow" in text
    assert "def camera_follow" in text
    assert "root_pos_w" in text
    assert "set_camera_view" in text
    assert "args_cli.no_camera_follow" in text


def test_play_script_sets_overview_camera_when_follow_is_disabled():
    text = SCRIPT.read_text()

    assert "def camera_overview" in text
    assert "env_origins" in text
    assert "Camera follow disabled" in text
    assert "camera_overview(env)" in text


def test_play_script_can_force_terrain_level_for_playback():
    text = SCRIPT.read_text()

    assert "--terrain-level" in text
    assert "def force_terrain_level" in text
    assert "terrain_level_spec" in text
    assert "terrain.terrain_levels" in text
    assert "terrain.env_origins" in text
    assert "env.reset()" in text
    assert "env_cfg.curriculum.terrain_levels = None" in text


def test_play_script_can_spread_playback_across_all_terrain_levels():
    text = SCRIPT.read_text()

    assert "--terrain-level all" in text
    assert 'terrain_level_spec == "all"' in text
    assert "torch.arange(terrain.terrain_levels.numel()" in text
    assert "% (max_level + 1)" in text
    assert "Spread playback across terrain levels" in text
