"""Pure report helpers for the quiet-impact benchmark."""

from __future__ import annotations


LOWER_IS_BETTER_METRICS = (
    "preimpact_speed_m_s",
    "peak_force_n",
    "peak_force_rise_rate_n_s",
    "impulse_n_s",
    "force_variation_energy_n2_s",
    "impact_score",
    "peak_impact_score",
    "slip_power_proxy_w",
    "tracking_abs_error_m_s",
)


def reduction_percent(baseline: float, candidate: float) -> float | None:
    """Return positive percentage when the candidate reduces a non-negative metric."""
    if baseline <= 0.0:
        return None
    return 100.0 * (baseline - candidate) / baseline


def compare_results(baseline: dict[str, float], candidate: dict[str, float]) -> dict[str, float | None]:
    """Compute lower-is-better reductions for shared benchmark metrics."""
    return {
        metric: reduction_percent(baseline[metric], candidate[metric])
        for metric in LOWER_IS_BETTER_METRICS
        if metric in baseline and metric in candidate
    }
