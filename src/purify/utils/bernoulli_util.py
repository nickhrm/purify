import numpy as np

# Tolerance for floating-point overshoot in computed probabilities.
_EPS = 1e-9


def bernoulli(rng: np.random.Generator, probability: float) -> bool:
    """Returns True with the given probability, drawn from `rng`."""
    if not -_EPS <= probability <= 1.0 + _EPS:
        raise ValueError(f"Probability must be in [0, 1], got {probability}")
    return rng.random() < probability
