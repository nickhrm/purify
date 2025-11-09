import logging

import numpy as np

from purify.my_constants import (
    ENTANGLEMENT_GENERATION_COUNT,
    ENTANGLEMENT_GENERATION_SCALE,
)
from purify.my_enums import Event, Strategy
from purify.my_time import Time
from purify.node import Node

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class Simulation:
    def __init__(self, strategy: Strategy, decoherence_time: float) -> None:
        # Inittialize Time
        self.time = Time()

        # create Node
        self.node_a = Node(self.time, strategy, decoherence_time)
        self.strategy = strategy
        self.decoerence_time = decoherence_time
        # Samples (hier aus Exponentialverteilungen)
        self.entanglement_samples = rng.exponential(
            ENTANGLEMENT_GENERATION_SCALE, ENTANGLEMENT_GENERATION_COUNT
        )
        self.request_samples = rng.exponential(
            ENTANGLEMENT_GENERATION_SCALE * 100, ENTANGLEMENT_GENERATION_COUNT * 100
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
