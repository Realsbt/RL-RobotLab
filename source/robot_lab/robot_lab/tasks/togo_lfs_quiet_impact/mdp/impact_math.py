"""Pure tensor operations used by the event-based impact tracker."""

from __future__ import annotations

import torch


def contact_hysteresis_step(
    force_norm: torch.Tensor,
    was_in_contact: torch.Tensor,
    contact_on_force: float,
    contact_off_force: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance a force-threshold contact state with separate enter and exit thresholds."""
    if contact_on_force <= contact_off_force or contact_off_force < 0.0:
        raise ValueError("Contact thresholds must satisfy contact_on_force > contact_off_force >= 0.")
    if force_norm.shape != was_in_contact.shape:
        raise ValueError("force_norm and was_in_contact must have the same shape.")

    entered = ~was_in_contact & (force_norm >= contact_on_force)
    exited = was_in_contact & (force_norm <= contact_off_force)
    is_in_contact = torch.where(entered, True, torch.where(exited, False, was_in_contact))
    return entered, exited, is_in_contact


def normalized_impact_score(
    preimpact_speed: torch.Tensor,
    peak_force: torch.Tensor,
    peak_force_rise_rate: torch.Tensor,
    impulse: torch.Tensor,
    references: tuple[float, float, float, float],
    weights: tuple[float, float, float, float],
) -> torch.Tensor:
    """Combine normalized event features into a dimensionless visualization score."""
    if any(reference <= 0.0 for reference in references):
        raise ValueError("Impact score references must be positive.")
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("Impact score weights must be non-negative with a positive sum.")

    weight_sum = sum(weights)
    components = (
        torch.square(preimpact_speed / references[0]),
        torch.square(peak_force / references[1]),
        torch.square(peak_force_rise_rate / references[2]),
        torch.square(impulse / references[3]),
    )
    return sum(weight * component for weight, component in zip(weights, components)) / weight_sum
