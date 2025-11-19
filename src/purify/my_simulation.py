import logging

import numpy as np

from purify.constants_tuple import ConstantsTuple
from purify.my_constants import (
    DELTA_T,
    ENTANGLEMENT_GENERATION_COUNT,
    QUBIT_ARRIVAL_SCALE,
    QUBIT_ENTANGLEMENT_FACTOR,
)
from purify.my_enums import Event
from purify.my_time import Time
from purify.node import Node

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class Simulation:
    def __init__(
        self,  constants: ConstantsTuple
    ) -> None:
        self.time = Time()
        self.node_a = Node(self.time, constants)
        self.constants = constants


        # Samples (hier Bernouli/geometric)
        self.entanglement_samples = [
            DELTA_T for _ in range(ENTANGLEMENT_GENERATION_COUNT)
        ]

        self.request_samples = rng.gamma(
            shape=2,
            scale=1 / QUBIT_ARRIVAL_SCALE,
            size=round(ENTANGLEMENT_GENERATION_COUNT / QUBIT_ENTANGLEMENT_FACTOR),
        )

    def step(self) -> bool:
        """Eine Simulationsiteration. Gibt False zurück, wenn Samples
        verbraucht sind."""
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
            round(self.time.get_current_time(), 2),
            self.time.last_event().name,
        )

        if self.time.last_event() == Event.ENTANGLEMENT_GENERATION:
            self.node_a.handle_entanglement_generation()
            self.node_a.serve_request()

        if self.time.last_event() == Event.REQUEST_ARRIVAL:
            self.node_a.handle_request_arrival()
            self.node_a.serve_request()

        # Logging
        if self.node_a.good_memory is not None:
            e = self.node_a.good_memory
            logger.info(
                "Good Memory = F:%s, lambda_1: %s, lambda_2: %s ,lambda_3: %s,",
                e.get_current_fidelity(),
                e.get_current_lambda_1(),
                e.get_current_lambda_2(),
                e.get_current_lambda_3(),
            )

        else:
            logger.info("Good Memory = None")

        if self.node_a.bad_memory is not None:
            e = self.node_a.bad_memory
            logger.info(
                "Bad Memory = F:%s, lambda_1: %s, lambda_2: %s ,lambda_3: %s,",
                e.get_current_fidelity(),
                e.get_current_lambda_1(),
                e.get_current_lambda_2(),
                e.get_current_lambda_3(),
            )

        else:
            logger.info("Bad Memory = None")

        return True

    def run(self) -> None:
        while self.step():
            pass
