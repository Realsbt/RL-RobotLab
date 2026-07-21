import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "scripts/rsl_rl/quiet_impact_report.py"
BENCHMARK_PATH = ROOT / "scripts/rsl_rl/benchmark_quiet_impact.py"
COMMAND_PATH = (
    ROOT
    / "source/robot_lab/robot_lab/tasks/togo_lfs_quiet_impact/mdp/commands.py"
)


def _load_report_module():
    spec = importlib.util.spec_from_file_location("quiet_impact_report", REPORT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reduction_percent_is_positive_for_lower_candidate_metric():
    report = _load_report_module()

    assert report.reduction_percent(10.0, 7.5) == pytest.approx(25.0)
    assert report.reduction_percent(10.0, 12.0) == pytest.approx(-20.0)
    assert report.reduction_percent(0.0, 0.0) is None


def test_comparison_only_contains_shared_lower_is_better_metrics():
    report = _load_report_module()
    baseline = {"impact_score": 2.0, "actual_forward_speed_m_s": 0.2}
    candidate = {"impact_score": 1.0, "actual_forward_speed_m_s": 0.19}

    comparison = report.compare_results(baseline, candidate)

    assert comparison == {"impact_score": pytest.approx(50.0)}


def test_benchmark_protocol_fixes_confounders_and_writes_json():
    benchmark_text = BENCHMARK_PATH.read_text(encoding="utf-8")

    assert 'default="RobotLab-ToGo-LFs-Quiet-Impact-v1"' in benchmark_text
    assert 'default=[0.2, 0.4]' in benchmark_text
    assert '"randomize_rigid_body_material"' in benchmark_text
    assert "env_cfg.observations.policy.enable_corruption = False" in benchmark_text
    assert "env_cfg.terminations.illegal_contact = None" in benchmark_text
    assert "raw_env.reset_to(initial_state" in benchmark_text
    assert "impact_command.clear_impact_statistics()" in benchmark_text
    assert "actual_forward_speed_m_s" in benchmark_text
    assert "json.dumps(report, indent=2)" in benchmark_text


def test_impact_command_exposes_event_weighted_benchmark_statistics():
    command_text = COMMAND_PATH.read_text(encoding="utf-8")

    assert "def clear_impact_statistics(" in command_text
    assert "def get_impact_statistics(" in command_text
    assert '"preimpact_speed_m_s"' in command_text
    assert '"peak_force_rise_rate_n_s"' in command_text
    assert '"slip_power_proxy_w"' in command_text
