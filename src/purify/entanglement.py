import logging

import numpy as np

from purify.my_constants import (
    LAMBDA_1,
    LAMBDA_2,
    LAMBDA_3,
)
from purify.my_time import Time

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
        current_time = self._time.get_current_time()
        time_alive = current_time - self.creationTime
        return (
            np.exp(-time_alive / self.decoherence_time)
            * (start_val - 0.25)
            + 0.25
        )

    @classmethod
    def from_default_lambdas(cls, time: Time, decoherence_time: float):
        """
        Erzeugt eine Entanglement-Instanz mit vordefinierten initialen Zustands-Parametern In general Bell-Diagonal-State
        (Fidelity und Lambdas).
        cls (Class): Die Klasse Entanglement selbst (wird automatisch übergeben).
        """
        initial_fidelity = 1.0 - (LAMBDA_1 + LAMBDA_2 + LAMBDA_3)

        return cls(  # 'cls' steht für die Klasse Entanglement
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=initial_fidelity,
            creation_lambda_1=LAMBDA_1,
            creation_lambda_2=LAMBDA_2,
            creation_lambda_3=LAMBDA_3,
            decoherence_time = decoherence_time
        )

    @classmethod
    def from_fidelity(cls, time: Time, fidelity: float, decoherence_time: float):
        """Erzeugt Entanglement in Werner-State"""
        if fidelity > 1:
            raise Exception("Fidelity must be smaller than 1")

        lambda_values = (1-fidelity)/3

        return cls(  # 'cls' steht für die Klasse Entanglement
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=fidelity,
            creation_lambda_1=lambda_values,
            creation_lambda_2=lambda_values,
            creation_lambda_3=lambda_values,
            decoherence_time = decoherence_time
        )
