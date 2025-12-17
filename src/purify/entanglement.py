import logging

import numpy as np

from purify.my_constants import (
    LAMBDA_1,
    LAMBDA_2,
    LAMBDA_3,
)
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
    def from_random_with_biggest_lambda(cls, time: Time, decoherence_time: float):
        """
        Erzeugt entanglement, für das gilt: x > (y + z). x,y,z sind lambda werte
        """

        creation_fidelity = 0.7
        lambda_1 = LAMBDA_1
        (y,z) = generate_y_z(lambda_1,creation_fidelity)

        logger.warning(f"lambda_2: {y} und lambda_3: {z}")

        return cls(
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=creation_fidelity,
            creation_lambda_1= lambda_1,
            creation_lambda_2=y,
            creation_lambda_3=z,
            decoherence_time=decoherence_time,
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
    
    @classmethod
    def from_random_fidelity_range(cls, time: Time, min_fidelity: float, max_fidelity: float, decoherence_time: float):
        """
        Erzeugt Entanglement mit einer zufälligen Fidelity im Bereich [min_fidelity, max_fidelity].
        Die Lambdas werden zufällig so verteilt, dass die Summe 1.0 ergibt.
        """
        if not (0 <= min_fidelity <= max_fidelity <= 1):
            raise ValueError("Ungültige Fidelity-Range. Werte müssen zwischen 0 und 1 liegen.")

        # 1. Zufällige Fidelity im Intervall wählen
        chosen_fidelity = np.random.uniform(min_fidelity, max_fidelity)

        # 2. Restwert berechnen, der auf die 3 Lambdas verteilt werden muss
        remaining_mass = 1.0 - chosen_fidelity

        # 3. Zufällige Verteilung der restlichen Masse auf 3 Lambdas (Dirichlet-Prinzip)
        # Wir ziehen zwei Zufallszahlen, um die Strecke 'remaining_mass' in 3 Teile zu schneiden
        cuts = np.sort(np.random.uniform(0, remaining_mass, 2))

        l1 = cuts[0]
        l2 = cuts[1] - cuts[0]
        l3 = remaining_mass - cuts[1]

        # logger.info(f"Generated Random Entanglement: F={chosen_fidelity:.4f}, L1={l1:.4f}, L2={l2:.4f}, L3={l3:.4f}")
        # print(f"Entanglement has form ({l1},{l2},{l3})")
        return cls(
            time=time,
            creation_time=time.get_current_time(),
            creation_fidelity=chosen_fidelity,
            creation_lambda_1=l1,
            creation_lambda_2=l2,
            creation_lambda_3=l3,
            decoherence_time=decoherence_time
        )
