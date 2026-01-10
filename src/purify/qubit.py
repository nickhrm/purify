import logging

import numpy as np

from purify.constants_tuple import ConstantsTuple
from purify.my_time import Time

logger = logging.getLogger(__name__)


class Qubit:
    def __init__(self, time: Time, constants: ConstantsTuple) -> None:
        self._time = time
        self.creationTime: float = time.get_current_time()
        self.constants: ConstantsTuple = constants

    def get_current_fidelity(self) -> float:
        current_time = self._time.get_current_time()
        time_alive = (
            current_time - self.creationTime
        ) * self.constants.waiting_time_sensitivity
        current_fidelity = (
            np.exp(-time_alive / self.constants.coherence_time) + 2.0
        ) / 3.0
        return current_fidelity

    def teleportation_fidelity(self, entanglement_fidelity: float) -> float:
        Fe = float(np.clip(entanglement_fidelity, 0.0, 1.0))
        Fq = float(np.clip(self.get_current_fidelity(), 0.0, 1.0))

        # Terme aus der Gleichung
        term1 = ((2.0 * Fe + 1.0) * Fq) / 3.0
        term2 = (2.0 * (1.0 - Fe) * (1.0 - Fq)) / 3.0

        # numerisch stabile Berechnung der Wurzeln
        a = np.sqrt(max(term1, 0.0))
        b = np.sqrt(max(term2, 0.0))

        FT = (a - b) ** 2
        return float(np.clip(FT, 0.0, 1.0))

    def get_waiting_time(self):
        return self._time.get_current_time() - self.creationTime
