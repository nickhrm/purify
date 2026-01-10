import logging

import numpy as np

from purify.constants_tuple import ConstantsTuple
from purify.my_enums import LambdaSrategy
from purify.my_time import Time
from purify.utils.generate_lambdas_util import generate_y_z

logger = logging.getLogger(__name__)


class Entanglement:
    """
    Repräsentiert ein erzeugtes Verschränkungs-Paar.
    Hält eine Referenz auf 'Time', damit get_current_fidelity() keine Zeit-Parameter
    braucht.
    """

    def __init__(
        self,
        time: Time,
        creation_time: float,
        creation_fidelity: float,
        creation_lambda_1: float,
        creation_lambda_2: float,
        creation_lambda_3: float,
        decoherence_time: float
    ):
        # __init__ bleibt technisch public, wird aber idealerweise nur
        # von den internen Factory-Methoden aufgerufen.
        self._time: Time = time
        self.creationTime: float = creation_time
        self.creationFidelity: float = creation_fidelity
        self.creation_lambda_1: float = creation_lambda_1
        self.creation_lambda_2: float = creation_lambda_2
        self.creation_lambda_3: float = creation_lambda_3
        self.decoherence_time: float = decoherence_time

    def get_current_fidelity(self) -> float:
        return self.__depolarization_noise(self.creationFidelity)

    def get_current_lambda_1(self) -> float:
        return self.__depolarization_noise(self.creation_lambda_1)

    def get_current_lambda_2(self) -> float:
        return self.__depolarization_noise(self.creation_lambda_2)

    def get_current_lambda_3(self) -> float:
        return self.__depolarization_noise(self.creation_lambda_3)

    def __depolarization_noise(self, start_val) -> float:
        # Doppelter Unterstrich (Name Mangling) für starke Kapselung
        current_time = self._time.get_current_time()
        time_alive = current_time - self.creationTime
        return (
            np.exp(-time_alive / self.decoherence_time)
            * (start_val - 0.25)
            + 0.25
        )

    # ---------------------------------------------------------
    # PUBLIC FACTORY
    # ---------------------------------------------------------

    @classmethod
    def from_strategy(cls, time: Time, constants: ConstantsTuple):
        """
        Die einzige öffentliche Factory-Methode. Erstellt eine Entanglement-Instanz
        basierend auf der in 'constants' definierten Strategie.
        """
        match constants.lambda_strategy:
            case LambdaSrategy.USE_CONSTANTS:
                return cls._from_constants(
                    time, constants
                )
            case LambdaSrategy.RANDOM_WITH_LARGEST_LAMBDA:
                return cls._from_random_with_biggest_lambda(
                    time, constants
                )
            case LambdaSrategy.RANDOM:
                return cls._from_random_fidelity_range(
                    time, 0.6, 0.8, constants
                )
            case _:
                # Fallback oder Fehlerbehandlung, falls eine Strategie nicht abgedeckt ist
                raise ValueError(f"Unknown strategy: {constants.lambda_strategy}")

    @classmethod
    def _from_constants(cls, time: Time, constants: ConstantsTuple):
        """
        Interne Methode: Erzeugt Entanglement aus vordefinierten Konstanten.
        """
        initial_fidelity = 1.0 - (constants.lambdas[0] + constants.lambdas[1] + constants.lambdas[2])

        return cls(
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=initial_fidelity,
            creation_lambda_1=constants.lambdas[0],
            creation_lambda_2=constants.lambdas[1],
            creation_lambda_3=constants.lambdas[2],
            decoherence_time=constants.coherence_time
        )

    @classmethod
    def _from_random_with_biggest_lambda(cls, time: Time, constants: ConstantsTuple):
        """
        Interne Methode: Erzeugt Entanglement, wo lambda_1 > lambda_2 + lambda_3.
        """
        creation_fidelity = 0.7
        lambda_1 = constants.lambdas[0]
        (y, z) = generate_y_z(lambda_1, creation_fidelity)

        logger.warning(f"lambda_2: {y} und lambda_3: {z}")

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
    def _from_random_fidelity_range(cls, time: Time, min_fidelity: float, max_fidelity: float, constants: ConstantsTuple):
        """
        Interne Methode: Zufällige Fidelity im Bereich und zufällige Lambdas (Dirichlet).
        """
        if not (0 <= min_fidelity <= max_fidelity <= 1):
            raise ValueError("Ungültige Fidelity-Range. Werte müssen zwischen 0 und 1 liegen.")

        chosen_fidelity = np.random.uniform(min_fidelity, max_fidelity)
        remaining_mass = 1.0 - chosen_fidelity
        cuts = np.sort(np.random.uniform(0, remaining_mass, 2))

        l1 = cuts[0]
        l2 = cuts[1] - cuts[0]
        l3 = remaining_mass - cuts[1]

        return cls(
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=chosen_fidelity,
            creation_lambda_1=l1,
            creation_lambda_2=l2,
            creation_lambda_3=l3,
            decoherence_time=constants.coherence_time
        )

    @classmethod
    def from_fidelity(cls, time: Time, fidelity: float, decoherence_time: float):
        """
        Interne Methode: Erzeugt Entanglement im Werner-Zustand.
        Wird aktuell nicht von from_strategy aufgerufen, aber als intern markiert.
        """
        if fidelity > 1:
            raise Exception("Fidelity must be smaller than 1")

        lambda_values = (1 - fidelity) / 3

        return cls(
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=fidelity,
            creation_lambda_1=lambda_values,
            creation_lambda_2=lambda_values,
            creation_lambda_3=lambda_values,
            decoherence_time=decoherence_time
        )
