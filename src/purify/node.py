import logging
from typing import cast

import numpy as np

from purify.constants_tuple import ConstantsTuple
from purify.entanglement import Entanglement
from purify.my_constants import (
    P_G,
)
from purify.my_enums import Action, LambdaSrategy
from purify.my_time import Time
from purify.qubit import Qubit
from purify.utils.bernouli_util import bernouli_with_probability_is_successfull
from purify.utils.purification_util import Purification

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class Node:
    def __init__(self, time: Time, constants: ConstantsTuple) -> None:
        self.time = time
        self.good_memory: Entanglement | None = None
        self.bad_memory: Entanglement | None = None
        self.queue: Qubit | None = None
        self.constants: ConstantsTuple = constants


    def handle_existing_entanglement(self, entanglement: Entanglement, action: Action) -> None:
        """Verarbeitet ein bereits generiertes Paar basierend auf der Agenten-Action."""
        if entanglement is None:
            return

        # If good memory is empty, just put new entanglement there
        if self.good_memory is None:
            self.good_memory = entanglement

        elif action == Action.REPLACE:
                self.replace(entanglement)

        else:
            self.pump(entanglement, action)


    def replace(self, entanglement) -> None:
        if (
            self.good_memory is None
            or self.good_memory.get_current_fidelity()
            < entanglement.get_current_fidelity()
        ):
            self.good_memory = entanglement
        else:
            if (
                self.bad_memory is None
                or self.bad_memory.get_current_fidelity()
                < entanglement.get_current_fidelity()
            ):
                self.bad_memory = entanglement

    def pump(
        self, new_entanglement:Entanglement, action: Action
    ):
        good_memory = cast(Entanglement, self.good_memory)
        success_probability = Purification.success_probability_from_action(good_memory, new_entanglement, action)
        fidelity_after_pumping = Purification.jump_function_from_action(good_memory,new_entanglement, action)

        if bernouli_with_probability_is_successfull(success_probability):
            self.good_memory = Entanglement.from_fidelity(
                self.time, fidelity_after_pumping, self.constants.decoherence_time
            )
            logger.info("purification was successful")

        else:
            self.good_memory = None
            logger.info("Purification failed")

        self.bad_memory = None

    def generate_entanglement(self) -> Entanglement | None:
        generation_successful = bernouli_with_probability_is_successfull(P_G)
        if generation_successful:
            logger.info("Entanglement Generation Successful")

            match self.constants.lambda_strategy:
                case LambdaSrategy.USE_CONSTANTS:
                    return Entanglement.from_default_lambdas(
                        self.time, self.constants.decoherence_time
                    )
                case LambdaSrategy.RANDOM_WITH_LARGEST_LAMBDA:
                    return Entanglement.from_random_with_biggest_lambda(
                        self.time, self.constants.decoherence_time
                    )
                case LambdaSrategy.RANDOM:
                    return Entanglement.from_random_fidelity_range(
                        self.time, 0.6, 0.8, self.constants.decoherence_time
                    )
        else:
            logger.info("Entanglement Generation Failed")
            return None

    def handle_request_arrival(self):
        # fill queue if empty
        if self.queue is None:
            self.queue = Qubit(self.time, self.constants)
        else:
            # if queue was already full, request is dropped
            logger.info("Serving request failed. queue was already full")

    def serve_request(self) -> tuple[float, float] | None:
        if self.queue is None:
            return None

        if self.good_memory is not None:
            teleportation_fidelity: float = self.queue.teleportation_fidelity(
                self.good_memory.get_current_fidelity()
            )
            logger.info(
                "Served request with fidelity %s and waiting time %s. Initial qubit had fidelity: %s",
                teleportation_fidelity,
                self.queue.get_waiting_time(),
                self.queue.get_current_fidelity(),
            )
            # write_results_csv(
            #     teleportation_fidelity, self.queue.get_waiting_time(), self.constants
            # )

            waiting_time = self.queue.get_waiting_time()

            # remove qubit from queue, because it was served
            self.queue = None
            # discard link in good_memory
            self.good_memory = None
            # put bad_memory entanglement in good_memory
            if self.bad_memory is not None:
                self.good_memory = self.bad_memory
                self.bad_memory = None
            return (teleportation_fidelity, waiting_time)

    def get_good_memory_fidelity(self) -> float:
        if self.good_memory is None:
            return 0.0
        else:
            return self.good_memory.get_current_fidelity()
