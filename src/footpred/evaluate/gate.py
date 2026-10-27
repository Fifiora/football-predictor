"""Champion/challenger gate: a new model is promoted only if it is measurably better."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from footpred.evaluate.metrics import expected_calibration_error, log_loss, rps

ECE_TOLERANCE = 0.005


@dataclass(frozen=True)
class GateResult:
    promote: bool
    reason: str
    champion_log_loss: float
    challenger_log_loss: float
    champion_rps: float
    challenger_rps: float


def should_promote(
    champion_probs: np.ndarray,
    challenger_probs: np.ndarray,
    outcomes: np.ndarray,
    min_matches: int = 200,
) -> GateResult:
    """Promote only with enough matches, strictly lower log loss and RPS, and no calibration loss.

    Calibration may not worsen by more than ``ECE_TOLERANCE`` (expected calibration error).
    """
    n = len(outcomes)
    champ_ll, chall_ll = log_loss(champion_probs, outcomes), log_loss(challenger_probs, outcomes)
    champ_rps, chall_rps = rps(champion_probs, outcomes), rps(challenger_probs, outcomes)
    champ_ece = expected_calibration_error(champion_probs, outcomes)
    chall_ece = expected_calibration_error(challenger_probs, outcomes)

    if n < min_matches:
        promote, reason = False, f"only {n} graded matches; the minimum is {min_matches}"
    elif not chall_ll < champ_ll:
        promote, reason = False, f"challenger log loss {chall_ll:.4f} is not lower than champion {champ_ll:.4f}"
    elif not chall_rps < champ_rps:
        promote, reason = False, f"challenger RPS {chall_rps:.4f} is not lower than champion {champ_rps:.4f}"
    elif chall_ece - champ_ece > ECE_TOLERANCE:
        promote, reason = False, (
            f"calibration worsened: challenger ECE {chall_ece:.4f} vs champion {champ_ece:.4f}"
        )
    else:
        promote, reason = True, "challenger is better on log loss and RPS with calibration not worse"
    return GateResult(promote, reason, champ_ll, chall_ll, champ_rps, chall_rps)
