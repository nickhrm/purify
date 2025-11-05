from __future__ import annotations

import argparse
import logging
from enum import Enum
from math import log
from os import error
from platform import win32_is_iot
import string

from clifford_util import Cliford
import numpy as np

from bernouli_util import bernouli_with_probability
from constants import (
    ENTANGLEMENT_DECOHERENCE_CONSTANT,
    ENTANGLEMENT_GENERATION_COUNT,
    ENTANGLEMENT_GENERATION_SCALE,
    ENTANGLEMENT_GENERATION_SUCESS_PROBABILITY,
    ENTANGLEMENT_INITIAL_FIDELITY,
)
from csv_utils import write_results_csv
from plot_util import create_boxplot

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class Event(Enum):
    ENTANGLEMENT_GENERATION = 1
    REQUEST_ARRIVED = 2


class Strategy(Enum):
    ALWAYS_REPLACE = 1
    ALWAYS_PROT_1 = 2
    ALWAYS_PROT_2 = 3
    ALWAYS_PROT_3 = 4
    ALWAYS_PROT_1_WITH_PROBABILITY = 5
    ALWAYS_PROT_2_WITH_PROBABILITY = 6
    ALWAYS_PROT_3_WITH_PROBABILITY = 7
    CHOOSE_RANDOM_ACTION_UNIFORMLY = 8
    GPS = 9


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

    def __init__(
        self, time: Time, creation_time: float, creation_fidelity: float
    ) -> None:
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
    def __init__(self, time: Time) -> None:
        self._time = time
        self.creationTime: float = time.get_current_time()

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

    def get_waiting_time(self):
        return self._time.get_current_time() - self.creationTime


class Node:
    def __init__(self, time: Time, strategy: Strategy) -> None:
        self.time = time
        self.good_memory: Entanglement | None = None
        self.bad_memory: Entanglement | None = None
        self.queue: Qubit | None = None
        self.strategy: Strategy = strategy

    "is called, when event entanglement_generation happend"

    def handle_entanglement_generated_event(self):
        entanglement = self._generate_entanglement()

        # generation was not successful
        if entanglement is None:
            return

        # good memory was empty. Just place new entanglement in good memory
        if self.good_memory is None:
            self.good_memory = entanglement
            return

        match self.strategy:
            case Strategy.ALWAYS_REPLACE:
                self.strategy_always_replace(entanglement)
            case (
                Strategy.ALWAYS_PROT_1 | Strategy.ALWAYS_PROT_2 | Strategy.ALWAYS_PROT_3
            ):
                self.strategy_always_prot_x(entanglement, self.strategy)

    def strategy_always_prot_x(
        self, new_entanglement: Entanglement, strategy: Strategy
    ):
        f_bd = new_entanglement.get_current_fidelity()
        f = self.good_memory.get_current_fidelity()
        fidelity_after_pumping = 0
        success_probability = 0

        match strategy:
            case Strategy.ALWAYS_PROT_1:
                fidelity_after_pumping = Cliford.prot_1_jump_function(f, f_bd)
                success_probability = Cliford.prot_1_success_probability(f, f_bd)
            case Strategy.ALWAYS_PROT_2:
                fidelity_after_pumping = Cliford.prot_2_jump_function(f, f_bd)
                success_probability = Cliford.prot_2_success_probability(f, f_bd)
            case Strategy.ALWAYS_PROT_3:
                fidelity_after_pumping = Cliford.prot_3_jump_function(f, f_bd)
                success_probability = Cliford.prot_3_success_probability(f, f_bd)

        new_entanglement = Entanglement(
            self.time, self.time.get_current_time(), fidelity_after_pumping
        )

        if bernouli_with_probability(success_probability):
            self.good_memory = new_entanglement
            self.bad_memory = None
            logger.info("purification was successfull")
        else:
            self.good_memory = None
            self.bad_memory = None
            logger.info("Purification failed")

    def strategy_always_replace(self, entanglement) -> None:
        if (
            self.good_memory is None
            or self.good_memory.get_current_fidelity()
            < entanglement.get_current_fidelity()
        ):
            logger.info(
                "Updated good-memory. Old value: "
                + str(
                    self.good_memory.get_current_fidelity()
                    if self.good_memory is not None
                    else None
                )
                + ", new value: "
                + str(entanglement.get_current_fidelity())
            )
            self.good_memory = entanglement
        else:
            if (
                self.bad_memory is None
                or self.bad_memory.get_current_fidelity()
                < entanglement.get_current_fidelity()
            ):
                logger.info(
                    "Updated bad-memory. Old value: "
                    + str(
                        self.bad_memory.get_current_fidelity()
                        if self.bad_memory is not None
                        else None
                    )
                    + ", new value: "
                    + str(entanglement.get_current_fidelity())
                )
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
        self.queue = Qubit(
            self.time,
        )
        if self.good_memory is not None:
            tf = self.queue.teleportation_fidelity(
                self.good_memory.get_current_fidelity()
            )
            logger.info(
                "Served request with fidelity %s and waiting time %s",
                tf,
                self.queue.get_waiting_time(),
            )
            write_results_csv(tf, self.queue.get_waiting_time(), self.strategy.name)
            self.good_memory = None
            if self.bad_memory is not None:
                self.good_memory = self.bad_memory
                self.bad_memory = None
        else:
            logger.info("Serving request failed")


class Simulation:
    def __init__(self, strategy: Strategy) -> None:
        self.time = Time()
        self.node_a = Node(self.time, strategy)
        self.strategy = strategy
        # Samples (hier aus Exponentialverteilungen)
        self.entanglement_samples = rng.exponential(
            ENTANGLEMENT_GENERATION_SCALE, ENTANGLEMENT_GENERATION_COUNT
        )
        self.request_samples = rng.exponential(
            ENTANGLEMENT_GENERATION_SCALE * 100, ENTANGLEMENT_GENERATION_COUNT * 100
        )

    def step(self) -> bool:
        """Eine Simulationsiteration. Gibt False zurück, wenn Samples verbraucht sind."""
        if self.time.entanglement_count >= len(
            self.entanglement_samples
        ) or self.time.request_count >= len(self.request_samples):
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
            self.node_a.handle_entanglement_generated_event()

        if self.time.last_event() == Event.REQUEST_ARRIVED:
            self.node_a.serve_request()

        # Logging
        gm = (
            self.node_a.good_memory.get_current_fidelity()
            if self.node_a.good_memory is not None
            else None
        )
        bm = (
            self.node_a.bad_memory.get_current_fidelity()
            if self.node_a.bad_memory is not None
            else None
        )
        logger.info("Node A: good-memory=%s, bad-memory=%s", gm, bm)

        return True

    def run(self) -> None:
        while self.step():
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=int,
        choices=range(1, len(Strategy) + 1),
        required=True,
    )
    args = parser.parse_args()
    strategy = Strategy(args.strategy)

    logging.basicConfig(
        filename="myapp.log",
        filemode="w",
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    logger.info("Starting simulation")

    sim = Simulation(strategy)
    sim.run()
    create_boxplot()


if __name__ == "__main__":
    main()
