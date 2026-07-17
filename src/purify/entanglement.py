import logging

import numpy as np

from purify.constants_tuple import ConstantsTuple
from purify.my_enums import LambdaStrategy
from purify.my_time import Time
from purify.utils.generate_lambdas_util import generate_y_z

logger = logging.getLogger(__name__)


class Entanglement:
    """
    Repräsentiert ein erzeugtes Verschränkungs-Paar (Bell-diagonaler Zustand
    mit Parametern F, λ₁, λ₂, λ₃). Hält eine Referenz auf 'Time', damit
    get_current_fidelity() keine Zeit-Parameter braucht.
    """

    def __init__(
        self,
        time: Time,
        creation_time: float,
        creation_fidelity: float,
        creation_lambda_1: float,
        creation_lambda_2: float,
        creation_lambda_3: float,
        decoherence_time: float,
    ):
        self._time: Time = time
        self.creation_time: float = creation_time
        self.creation_fidelity: float = creation_fidelity
        self.creation_lambda_1: float = creation_lambda_1
        self.creation_lambda_2: float = creation_lambda_2
        self.creation_lambda_3: float = creation_lambda_3
        self.decoherence_time: float = decoherence_time

    def get_current_fidelity(self) -> float:
        return self._decay(self.creation_fidelity)

    def get_current_lambda_1(self) -> float:
        return self._decay(self.creation_lambda_1)

    def get_current_lambda_2(self) -> float:
        return self._decay(self.creation_lambda_2)

    def get_current_lambda_3(self) -> float:
        return self._decay(self.creation_lambda_3)

    def _decay(self, start_val: float) -> float:
        """Depolarisierendes Rauschen: exponentieller Zerfall aller vier
        Bell-Koeffizienten Richtung 1/4 (maximal gemischter Zustand)."""
        time_alive = self._time.get_current_time() - self.creation_time
        return np.exp(-time_alive / self.decoherence_time) * (start_val - 0.25) + 0.25

    # ---------------------------------------------------------
    # PUBLIC FACTORIES
    # ---------------------------------------------------------

    @classmethod
    def from_strategy(
        cls,
        time: Time,
        constants: ConstantsTuple,
        rng: np.random.Generator | None = None,
    ) -> "Entanglement":
        """
        Erstellt eine Entanglement-Instanz basierend auf der in 'constants'
        definierten Strategie. Zufällige Strategien ziehen aus 'rng'.
        """
        rng = rng if rng is not None else np.random.default_rng()
        match constants.lambda_strategy:
            case LambdaStrategy.USE_CONSTANTS:
                return cls._from_constants(time, constants)
            case LambdaStrategy.RANDOM_WITH_LARGEST_LAMBDA:
                return cls._from_random_with_biggest_lambda(time, constants, rng)
            case LambdaStrategy.RANDOM:
                return cls._from_random_fidelity_range(time, constants, rng)
            case LambdaStrategy.FIXED:
                return cls._from_fixed(time, constants.coherence_time, rng)
            case LambdaStrategy.PERMUTED_CONSTANTS:
                return cls._from_permuted_constants(time, constants, rng)
            case _:
                raise ValueError(f"Unknown strategy: {constants.lambda_strategy}")

    @classmethod
    def from_fidelity(cls, time: Time, fidelity: float, decoherence_time: float) -> "Entanglement":
        """Erzeugt Entanglement im Werner-Zustand mit gegebener Fidelity."""
        if not 0.0 <= fidelity <= 1.0:
            raise ValueError(f"Fidelity must be in [0, 1], got {fidelity}")

        lambda_value = (1 - fidelity) / 3

        return cls(
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=fidelity,
            creation_lambda_1=lambda_value,
            creation_lambda_2=lambda_value,
            creation_lambda_3=lambda_value,
            decoherence_time=decoherence_time,
        )

    # ---------------------------------------------------------
    # INTERNAL FACTORIES (one per LambdaStrategy)
    # ---------------------------------------------------------

    @classmethod
    def _from_constants(cls, time: Time, constants: ConstantsTuple) -> "Entanglement":
        """Feste Lambdas aus den Constants; Fidelity ergibt sich als 1 - Σλ."""
        l1, l2, l3 = constants.lambdas

        return cls(
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=1.0 - (l1 + l2 + l3),
            creation_lambda_1=l1,
            creation_lambda_2=l2,
            creation_lambda_3=l3,
            decoherence_time=constants.coherence_time,
        )

    @classmethod
    def _from_permuted_constants(
        cls, time: Time, constants: ConstantsTuple, rng: np.random.Generator
    ) -> "Entanglement":
        """Feste Lambda-Werte aus den Constants, zufällig auf die Positionen
        permutiert (z.B. wird (0.3, 0, 0) zu einer der sechs Permutationen).
        Die Fidelity bleibt 1 - Σλ."""
        l1, l2, l3 = rng.permutation(constants.lambdas)

        return cls(
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=1.0 - (l1 + l2 + l3),
            creation_lambda_1=l1,
            creation_lambda_2=l2,
            creation_lambda_3=l3,
            decoherence_time=constants.coherence_time,
        )

    @classmethod
    def _from_random_with_biggest_lambda(
        cls, time: Time, constants: ConstantsTuple, rng: np.random.Generator
    ) -> "Entanglement":
        """λ₁ fest aus den Constants mit λ₁ > λ₂ + λ₃; λ₂, λ₃ zufällig."""
        creation_fidelity = 0.7
        lambda_1 = constants.lambdas[0]
        (y, z) = generate_y_z(lambda_1, creation_fidelity, rng)

        return cls(
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=creation_fidelity,
            creation_lambda_1=lambda_1,
            creation_lambda_2=y,
            creation_lambda_3=z,
            decoherence_time=constants.coherence_time,
        )

    @classmethod
    def _from_random_fidelity_range(
        cls, time: Time, constants: ConstantsTuple, rng: np.random.Generator
    ) -> "Entanglement":
        """Zufällige Fidelity im Bereich [min_fidelity, max_fidelity];
        die restliche Fehlermasse wird gleichverteilt (uniform auf dem
        Simplex) auf λ₁, λ₂, λ₃ aufgeteilt."""
        min_fidelity = constants.min_fidelity
        max_fidelity = constants.max_fidelity
        if not (0 <= min_fidelity <= max_fidelity <= 1):
            raise ValueError("Ungültige Fidelity-Range. Werte müssen zwischen 0 und 1 liegen.")

        chosen_fidelity = rng.uniform(min_fidelity, max_fidelity)
        remaining_mass = 1.0 - chosen_fidelity
        cuts = np.sort(rng.uniform(0, remaining_mass, 2))

        return cls(
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=chosen_fidelity,
            creation_lambda_1=cuts[0],
            creation_lambda_2=cuts[1] - cuts[0],
            creation_lambda_3=remaining_mass - cuts[1],
            decoherence_time=constants.coherence_time,
        )

    @classmethod
    def _from_fixed(
        cls, time: Time, decoherence_time: float, rng: np.random.Generator
    ) -> "Entanglement":
        """F=0.7, λ₁=0.2 fest; die restliche Masse 0.1 zufällig auf λ₂/λ₃."""
        lambda2 = rng.uniform(0, 0.1)
        lambda3 = 0.1 - lambda2

        return cls(
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=0.7,
            creation_lambda_1=0.2,
            creation_lambda_2=lambda2,
            creation_lambda_3=lambda3,
            decoherence_time=decoherence_time,
        )
