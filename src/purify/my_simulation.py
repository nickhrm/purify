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

    
    def step(self) -> bool:
        """Eine Simulationsiteration. Gibt False zurück, wenn Samples
        verbraucht sind."""

        if not self.time.update():
            return False

        logger.info(
            "Current Time: %s | Last Event: %s",
            round(self.time.get_current_time(), 2),
            self.time.last_event().name,
        )

        if self.time.last_event() == Event.ENTANGLEMENT_GENERATION:
            self.node_a.handle_entanglement_generation_use_constants_action()
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
