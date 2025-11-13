import logging
from math import inf

import numpy as np

from purify.entanglement import Entanglement
from purify.my_constants import (
    P_G,
)
from purify.my_enums import Strategy
from purify.my_time import Time
from purify.qubit import Qubit
from purify.utils.bernouli_util import bernouli_with_probability_is_successfull
from purify.utils.csv_utils import write_results_csv
from purify.utils.purification_util import Purification

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class Node:
    def __init__(self, time: Time, strategy: Strategy, decoherence_time: float) -> None:
        self.time = time
        self.good_memory: Entanglement | None = None
        self.bad_memory: Entanglement | None = None
        self.queue: Qubit | None = None
        self.strategy: Strategy = strategy
        self.decoherence_time: float = decoherence_time

    "is called, when event entanglement_generation happend"

    def handle_entanglement_generation(self) -> None:
        entanglement: Entanglement | None = self.__generate_entanglement()

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
            case Strategy.ALWAYS_PROT_1:
                self.strategy_always_prot_1(entanglement)
            case Strategy.ALWAYS_PROT_2:
                self.strategy_always_prot_2(entanglement)
            case Strategy.ALWAYS_PROT_3:
                self.strategy_always_prot_3(entanglement)
            case Strategy.ALWAYS_PMD:
                self.strategy_always_pmd(entanglement)

    def strategy_always_prot_1(self, new_entanglement: Entanglement):
        if self.good_memory is None:
            raise Exception("Cannot pump without Entanglement")
        self.always_prot_x_helper(
            Purification.prot_1_success_probability(self.good_memory, new_entanglement),
            Purification.prot_1_jump_function(self.good_memory, new_entanglement),
        )

    def strategy_always_prot_2(self, new_entanglement: Entanglement):
        if self.good_memory is None:
            raise Exception("Cannot pump without Entanglement")
        self.always_prot_x_helper(
            Purification.prot_2_success_probability(self.good_memory, new_entanglement),
            Purification.prot_2_jump_function(self.good_memory, new_entanglement),
        )

    def strategy_always_prot_3(self, new_entanglement: Entanglement):
        if self.good_memory is None:
            raise Exception("Cannot pump without Entanglement")
        self.always_prot_x_helper(
            Purification.prot_3_success_probability(self.good_memory, new_entanglement),
            Purification.prot_3_jump_function(self.good_memory, new_entanglement),
        )

    def strategy_always_pmd(self, new_entanglement: Entanglement):
        if self.good_memory is None:
            raise Exception("Cannot pump without Entanglement")
        self.always_prot_x_helper(
            Purification.pmd_success_probability(self.good_memory, new_entanglement),
            Purification.pmd_jump_function(self.good_memory, new_entanglement),
        )

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

    def always_prot_x_helper(
        self, success_probability: float, fidelity_after_pumping: float
    ):
        if bernouli_with_probability_is_successfull(success_probability):
            logger.info(
                "Updated good-memory. Old value: "
                + str(
                    self.good_memory.get_current_fidelity()
                    if self.good_memory is not None
                    else None
                )
                + ", new value: "
                + str(fidelity_after_pumping)
                + " using strategy "
                + self.strategy.name
            )

            self.good_memory = Entanglement.from_fidelity(
                self.time, fidelity_after_pumping, self.decoherence_time
            )
            self.bad_memory = None
            logger.info("purification was successfull")

        else:
            self.good_memory = None
            self.bad_memory = None
            logger.info("Purification failed")

    def __generate_entanglement(self) -> Entanglement | None:
        generation_successful = bernouli_with_probability_is_successfull(P_G)
        if generation_successful:
            logger.info("Entanglement Generation Successful")
            return Entanglement.from_default_lambdas(self.time, self.decoherence_time)
        else:
            logger.info("Entanglement Generation Failed")
            return None

    def handle_request_arrival(self):
        # fill queue if empty
        if self.queue is None:
            self.queue = Qubit(self.time, self.decoherence_time)
        else:
            # if queue was already full, request is dropped
            logger.info("Serving request failed. queue was already full")

    def serve_request(self):
        if self.queue is None:
            return

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
            write_results_csv(
                teleportation_fidelity,
                self.queue.get_waiting_time(),
                self.strategy.name,
                self.decoherence_time,
            )

            # remove qubit from queue, because it was served
            self.queue = None
            # discard link in good_memory
            self.good_memory = None
            # put bad_memory entanglement in good_memory
            if self.bad_memory is not None:
                self.good_memory = self.bad_memory
                self.bad_memory = None
