import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
QUIET_TASK_DIR = ROOT / "source/robot_lab/robot_lab/tasks/togo_lfs_quiet"
IMPACT_TASK_DIR = ROOT / "source/robot_lab/robot_lab/tasks/togo_lfs_quiet_impact"


def _load_impact_math():
    module_path = IMPACT_TASK_DIR / "mdp" / "impact_math.py"
    spec = importlib.util.spec_from_file_location("impact_math", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_impact_task_is_registered_without_replacing_quiet_baseline():
    quiet_init = (QUIET_TASK_DIR / "__init__.py").read_text(encoding="utf-8")
    impact_init = (IMPACT_TASK_DIR / "__init__.py").read_text(encoding="utf-8")

    assert 'id="RobotLab-ToGo-LFs-Quiet-v0"' in quiet_init
    assert 'id="RobotLab-ToGo-LFs-Quiet-Impact-v1"' in impact_init
    assert "togo_lfs_quiet_impact.env_cfg:ToGoLFsQuietImpactEnvCfg" in impact_init


def test_impact_task_preserves_policy_and_reward_contracts():
    cfg_text = (IMPACT_TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")

    assert "class ToGoLFsQuietImpactEnvCfg(ToGoLFsQuietEnvCfg)" in cfg_text
    assert "observations:" not in cfg_text
    assert "rewards:" not in cfg_text
    assert "quiet_impact_sensor = mdp.FootImpactSensorCfg(" in cfg_text
    assert "history_length=20" in cfg_text
    assert 'prim_path="{ENV_REGEX_NS}/Robot/.*_foot"' in cfg_text


def test_impact_command_uses_event_windows_and_separate_metrics():
    command_text = (IMPACT_TASK_DIR / "mdp" / "commands.py").read_text(encoding="utf-8")

    assert "class ImpactEventVelocityCommand(MuteVelocityCommand)" in command_text
    assert "contact_hysteresis_step(" in command_text
    assert "contact_on_force: float = 1.0" in command_text
    assert "contact_off_force: float = 0.5" in command_text
    assert "self.impact_window_samples" in command_text
    assert 'self.metrics["impact_event_count"]' in command_text
    assert '"event_preimpact_speed"' in command_text
    assert '"event_peak_force_rise_rate"' in command_text
    assert '"event_force_variation_energy"' in command_text
    assert '"contact_slip_power_proxy"' in command_text
    assert 'prim_path="/Visuals/MUTE/impact_events"' in command_text
    assert '"inactive": sim_utils.SphereCfg(' in command_text


def test_contact_hysteresis_has_distinct_enter_and_exit_thresholds():
    impact_math = _load_impact_math()
    state = torch.tensor([False])
    entered_steps = []
    exited_steps = []

    for force in (0.0, 0.9, 1.0, 0.75, 0.51, 0.5):
        entered, exited, state = impact_math.contact_hysteresis_step(
            torch.tensor([force]), state, contact_on_force=1.0, contact_off_force=0.5
        )
        entered_steps.append(bool(entered.item()))
        exited_steps.append(bool(exited.item()))

    assert entered_steps == [False, False, True, False, False, False]
    assert exited_steps == [False, False, False, False, False, True]


def test_contact_hysteresis_rejects_invalid_thresholds():
    impact_math = _load_impact_math()

    with pytest.raises(ValueError, match="contact_on_force"):
        impact_math.contact_hysteresis_step(
            torch.zeros(1), torch.zeros(1, dtype=torch.bool), 2.0, 2.0
        )


def test_normalized_impact_score_is_one_at_all_reference_values():
    impact_math = _load_impact_math()
    references = (0.2, 500.0, 100000.0, 10.0)

    score = impact_math.normalized_impact_score(
        *(torch.tensor([value]) for value in references),
        references=references,
        weights=(0.4, 0.2, 0.3, 0.1),
    )

    assert score.item() == pytest.approx(1.0)


def test_normalized_impact_score_weights_preimpact_speed_quadratically():
    impact_math = _load_impact_math()

    score = impact_math.normalized_impact_score(
        torch.tensor([0.4]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        references=(0.2, 500.0, 100000.0, 10.0),
        weights=(1.0, 0.0, 0.0, 0.0),
    )

    assert score.item() == pytest.approx(4.0)
