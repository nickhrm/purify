import logging
import numpy as np
from purify.my_constants import ENTANGLEMENT_DECOHERENCE_CONSTANT
from purify.my_time import Time

logger = logging.getLogger(__name__)


class Entanglement:
    """
    Repräsentiert ein erzeugtes Verschränkungs-Paar.
    Hält eine Referenz auf 'Time', damit get_current_fidelity() keine Zeit-Parameter braucht.
    """

    def __init__(
        self,
        time: Time,
        creation_time: float,
        creation_fidelity: float,
        lambda_1: float,
        lambda_2: float,
        lambda_3: float,
    ):
        self._time: float = time
        self.creationTime: float = creation_time
        self.creationFidelity: float = creation_fidelity
        self.lambda_1: float = lambda_1
        self.lambda_2: float = lambda_2
        self.lambda_3: float = lambda_3

    def get_current_fidelity(self) -> float | None:
        if self is None:
            return None
        # aktuelle Simulationszeit direkt über Time beziehen
        current_time = self._time.get_current_time()
        time_alive = current_time - self.creationTime
        return (
            np.exp(-time_alive / ENTANGLEMENT_DECOHERENCE_CONSTANT)
            * (self.creationFidelity - 0.25)
            + 0.25
        )

    def get_current_lambda_1(self):
        return self.lambda_1

    def get_current_lambda_2(self):
        return self.lambda_2
    
    def get_current_lambda_3(self):
        return self.lambda_3
    
