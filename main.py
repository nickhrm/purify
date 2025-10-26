from __future__ import annotations
from enum import Enum
import logging
import numpy as np

from constants import (
    ENTANGLEMENT_DECOHERENCE_CONSTANT,
    ENTANGLEMENT_GENERATION_COUNT,
    ENTANGLEMENT_GENERATION_SCALE,
    ENTANGLEMENT_GENERATION_SUCESS_PROBABILITY,
    ENTANGLEMENT_INITIAL_FIDELITY,
)

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class Event(Enum):
    ENTANGLEMENT_GENERATION = 1
    REQUEST_ARRIVED = 2


class Time:
    def __init__(self) -> None:
        self.entanglement_time: float = 0.0
        self.entanglement_count: int = 0
        self.request_time: float = 0.0
        self.request_count: int = 0

    def get_current_time(self) -> float:
        return max(self.entanglement_time, self.request_time)

    def last_event(self) -> Event:
        if self.entanglement_time > self.request_time:
            return Event.ENTANGLEMENT_GENERATION
        else:
            return Event.REQUEST_ARRIVED

    def update(self, entanglement_dif: float, request_dif: float) -> None:
        new_entanglement_time = self.entanglement_time + entanglement_dif
        new_request_time = self.request_time + request_dif

        if new_entanglement_time < new_request_time:
            self.entanglement_time = new_entanglement_time
            self.entanglement_count += 1
        else:
            self.request_time = new_request_time
            self.request_count += 1


class Entanglement:
    """
    Repräsentiert ein erzeugtes Verschränkungs-Paar.
    Hält eine Referenz auf 'Time', damit get_current_fidelity() keine Zeit-Parameter braucht.
    """

    def __init__(self, time: Time, creation_time: float, creation_fidelity: float) -> None:
        self._time = time
        self.creationTime: float = creation_time
        self.creationFidelity: float = creation_fidelity

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


class Qubit: 
    def __init__(self, time: Time, creation_time: float,) -> None:
        self._time = time
        self.creationTime: float = creation_time

    def get_current_fidelity(self) -> float | None:
        if self is None:
            return None
        current_time = self._time.get_current_time()
        time_alive = current_time - self.creationTime
        return (np.exp(-time_alive / ENTANGLEMENT_DECOHERENCE_CONSTANT) + 2.0) / 3.0

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
            

class Node:
    def __init__(self, time: Time) -> None:
        self.time = time
        self.good_memory: Entanglement | None = None
        self.bad_memory: Entanglement | None = None
        self.queue: Qubit | None = None

    def generate_entanglement(self) -> None:
        entanglement = self._generate_entanglement()
        if entanglement is None:
            return

        # Vergleich über aktuelle Fidelity, ohne Zeit-Parameter herumzureichen
        if (self.good_memory is None or
            self.good_memory.get_current_fidelity() < entanglement.get_current_fidelity()):
            logger.info("Updated good-memory. Old value: " + str(self.good_memory.get_current_fidelity() if self.good_memory is not None else None) + ", new value: " + str(entanglement.get_current_fidelity())) 
            self.good_memory = entanglement
        else:
            if (self.bad_memory is None or
                self.bad_memory.get_current_fidelity() < entanglement.get_current_fidelity()):
                logger.info("Updated bad-memory. Old value: " + str(self.bad_memory.get_current_fidelity() if self.bad_memory is not None else None) + ", new value: " + str(entanglement.get_current_fidelity())) 
                self.bad_memory = entanglement

    def _generate_entanglement(self) -> Entanglement | None:
        generation_successful = rng.choice(
            [True, False],
            p=[
                ENTANGLEMENT_GENERATION_SUCESS_PROBABILITY,
                1 - ENTANGLEMENT_GENERATION_SUCESS_PROBABILITY,
            ],
        )
        if generation_successful:
            creation_time = self.time.get_current_time()
            logger.info("Entanglement Generation Successful")
            return Entanglement(self.time, creation_time, ENTANGLEMENT_INITIAL_FIDELITY)
        else:
            logger.info("Entanglement Generation Failed")
            return None
        
    def serve_request(self):
        if self.good_memory is not None:
            


class Simulation:
    def __init__(self) -> None:
        self.time = Time()
        self.node_a = Node(self.time)

        # Samples (hier aus Exponentialverteilungen)
        self.entanglement_samples = rng.exponential(
            ENTANGLEMENT_GENERATION_SCALE, ENTANGLEMENT_GENERATION_COUNT
        )
        self.request_samples = rng.exponential(
            ENTANGLEMENT_GENERATION_SCALE * 100, ENTANGLEMENT_GENERATION_COUNT * 100
        )

    def step(self) -> bool:
        """Eine Simulationsiteration. Gibt False zurück, wenn Samples verbraucht sind."""
        if (self.time.entanglement_count >= len(self.entanglement_samples) or
            self.time.request_count >= len(self.request_samples)):
            return False

        self.time.update(
            self.entanglement_samples[self.time.entanglement_count],
            self.request_samples[self.time.request_count],
        )

        logger.info(
            "Current Time: %s | Last Event: %s",
            self.time.get_current_time(),
            self.time.last_event().name,
        )

        if self.time.last_event() == Event.ENTANGLEMENT_GENERATION:
            self.node_a.generate_entanglement()

        if self.time.last_event() == Event.REQUEST_ARRIVED:
            self.node_a.serve_request()


        # Logging
        gm = self.node_a.good_memory.get_current_fidelity() if self.node_a.good_memory else None
        bm = self.node_a.bad_memory.get_current_fidelity() if self.node_a.bad_memory else None
        logger.info("Node A: good-memory=%s, bad-memory=%s", gm, bm)

        return True

    def run(self) -> None:
        while self.step():
            pass


def main() -> None:
    logging.basicConfig(
        filename="myapp.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting simulation")

    sim = Simulation()
    sim.run()


if __name__ == "__main__":
    main()
