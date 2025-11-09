import numpy as np

from purify.my_time import Time


class Qubit:
    def __init__(self, time: Time, decoherence_time: float) -> None:
        self._time = time
        self.creationTime: float = time.get_current_time()
        self.decoherence_time: float = decoherence_time


    def get_current_fidelity(self) -> float:
        current_time = self._time.get_current_time()
        time_alive = current_time - self.creationTime
        return (np.exp(-time_alive / self.decoherence_time) + 2.0) / 3.0

    def teleportation_fidelity(self, entanglement_fidelity: float) -> float:
        Fe = float(np.clip(entanglement_fidelity, 0.0, 1.0))
        Fq = self.get_current_fidelity()

        term1 = ((2.0 * Fe + 1.0) * Fq) / 3.0
        term2 = (2.0 * (Fe - 1.0) * (1.0 - Fq)) / 3.0

        # Radikanden stabilisieren (Rundungsfehler -> kleine negative Zahlen vermeiden)
        a = np.sqrt(max(term1, 0.0))
        b = np.sqrt(max(term2, 0.0))

        FT = (a - b) ** 2
        return float(np.clip(FT, 0.0, 1.0))

    def get_waiting_time(self):
        return self._time.get_current_time() - self.creationTime

