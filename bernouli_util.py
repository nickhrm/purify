
import numpy as np


rng = np.random.default_rng()

def bernouli_with_probability(probability: float) -> bool:
        if probability > 1:
            raise Exception("Probability must be smaller than 1")
        success: bool = rng.choice(
            [True, False],
            p=[
                probability,
                1 - probability,
            ],
        )
        return success