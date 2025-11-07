import logging

import numpy as np

from purify.entanglement import Entanglement
from purify.my_constants import (
    ENTANGLEMENT_GENERATION_SUCESS_PROBABILITY,
)
from purify.my_enums import Strategy
from purify.my_time import Time
from purify.qubit import Qubit
from purify.utils.bernouli_util import bernouli_with_probability_is_successfull
from purify.utils.clifford_util import Cliford
from purify.utils.csv_utils import write_results_csv

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

    def handle_entanglement_generation(self):
        entanglement = self.__generate_entanglement()

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
        fidelity_after_pumping = 0
        success_probability = 0

        logger.info(
            "pump with 1G-Fidelity: %s and 1B-Fidelity: %s, using strategy %s",
            self.good_memory.get_current_fidelity(),
            new_entanglement.get_current_fidelity(),
            strategy.name,
        )

        match strategy:
            case Strategy.ALWAYS_PROT_1:
                fidelity_after_pumping = Cliford.prot_1_jump_function(
                    self.good_memory, new_entanglement
                )
                success_probability = Cliford.prot_1_success_probability(
                    self.good_memory, new_entanglement
                )
            case Strategy.ALWAYS_PROT_2:
                fidelity_after_pumping = Cliford.prot_2_jump_function(
                    self.good_memory, new_entanglement
                )
                success_probability = Cliford.prot_2_success_probability(
                    self.good_memory, new_entanglement
                )
            case Strategy.ALWAYS_PROT_3:
                fidelity_after_pumping = Cliford.prot_3_jump_function(
                    self.good_memory, new_entanglement
                )
                success_probability = Cliford.prot_3_success_probability(
                    self.good_memory, new_entanglement
                )

        if bernouli_with_probability_is_successfull(success_probability):
            self.good_memory = Entanglement.from_fidelity(
                self.time, fidelity_after_pumping, self.decoherence_time
            )
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

    def __generate_entanglement(self) -> Entanglement | None:
        generation_successful = bernouli_with_probability_is_successfull(
            ENTANGLEMENT_GENERATION_SUCESS_PROBABILITY
        )
        if generation_successful:
            logger.info("Entanglement Generation Successful")
            return Entanglement.from_default_lambdas(self.time, self.decoherence_time)
        else:
            logger.info("Entanglement Generation Failed")
            return None

    def handle_request_arrival(self):
        self.queue = Qubit(self.time, self.decoherence_time)
        if self.good_memory is not None:
            teleportation_fidelity = self.queue.teleportation_fidelity(
                self.good_memory.get_current_fidelity()
            )
            logger.info(
                "Served request with fidelity %s and waiting time %s",
                teleportation_fidelity,
                self.queue.get_waiting_time(),
            )
            write_results_csv(
                teleportation_fidelity,
                self.queue.get_waiting_time(),
                f"{self.strategy.name}_{self.decoherence_time}",
            )

            # discard link in good_memory
            self.good_memory = None
            # put bad_memory entanglement in good_memory
            if self.bad_memory is not None:
                self.good_memory = self.bad_memory
                self.bad_memory = None
        else:
            write_results_csv(
                0,
                self.queue.get_waiting_time(),
                f"{self.strategy.name}_{self.decoherence_time}",
            )
            logger.info("Serving request failed")

        self.queue = None
