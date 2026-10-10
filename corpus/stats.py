"""Small statistics for reading the evaluation honestly.

The runs of one question are not independent samples (the same question, the same
documents), so a rate over runs overstates how sure we are. The unit that counts is the
question; with a few dozen questions per profile the interval is wide, and the table says so.

Key exports:
    wilson_lower  -- Lower bound of the Wilson score interval for a proportion.
"""

from math import sqrt


def wilson_lower(successes: int, trials: int, z: float = 1.96) -> float | None:
    """The lower bound of the 95% Wilson score interval for ``successes`` of ``trials``.

    Args:
        successes: How many trials succeeded.
        trials: How many trials there were.
        z: The normal quantile (1.96 for 95%).

    Returns:
        The lower bound in [0, 1], or ``None`` when there are no trials.
    """
    if trials <= 0:
        return None
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = p + z * z / (2 * trials)
    margin = z * sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return max(0.0, (centre - margin) / denominator)
