from asyncio import constants
import logging

import numpy as np

from purify import ConstantsTuple
from purify.entanglement import Entanglement
from purify.my_constants import (
    P_G,
)
from purify.my_enums import LambdaSrategy, Action
from purify.my_time import Time
from purify.qubit import Qubit
from purify.utils.bernouli_util import bernouli_with_probability_is_successfull
from purify.utils.csv_utils import write_results_csv
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

    "is called, when event entanglement_generation happend"

    def handle_entanglement_generation_use_constants_action(self) -> None:
        self.handle_entanglement_generation(self.constants.strategy)


    def handle_entanglement_generation(self, action: Action) -> None:
        entanglement: Entanglement | None = self.generate_entanglement()

        # generation was not successful
        if entanglement is None:
            return

        # good memory was empty. Just place new entanglement in good memory
        if self.good_memory is None:
            self.good_memory = entanglement
            return

        match action:
            case Action.TRAINING_MODE:
                return
            case Action.REPLACE:
                self.strategy_always_replace(entanglement)
            case Action.PROT_1:
                self.strategy_always_prot_1(entanglement)
            case Action.PROT_2:
                self.strategy_always_prot_2(entanglement)
            case Action.PROT_3:
                self.strategy_always_prot_3(entanglement)
            case Action.PMD:
                self.strategy_always_pmd(entanglement)
            case Action.PROT_1_WITH_PROBABILITY:
                self.strategy_always_prot_1_with_probbility(entanglement)
            case Action.PROT_2_WITH_PROBABILITY:
                self.strategy_always_prot_2_with_probbility(entanglement)
            case Action.PROT_3_WITH_PROBABILITY:
                self.strategy_always_prot_3_with_probbility(entanglement)

    def strategy_always_prot_1_with_probbility(self, new_entanglement: Entanglement):
        if self.good_memory is None:
            raise Exception("Cannot pump without Entanglement")
        self.sometimes_prot_x_helper(
            Purification.prot_1_success_probability(self.good_memory, new_entanglement),
            Purification.prot_1_jump_function(self.good_memory, new_entanglement),
        )

    def strategy_always_prot_2_with_probbility(self, new_entanglement: Entanglement):
        if self.good_memory is None:
            raise Exception("Cannot pump without Entanglement")
        self.sometimes_prot_x_helper(
            Purification.prot_2_success_probability(self.good_memory, new_entanglement),
            Purification.prot_2_jump_function(self.good_memory, new_entanglement),
        )

    def strategy_always_prot_3_with_probbility(self, new_entanglement: Entanglement):
        if self.good_memory is None:
            raise Exception("Cannot pump without Entanglement")
        self.sometimes_prot_x_helper(
            Purification.prot_3_success_probability(self.good_memory, new_entanglement),
            Purification.prot_3_jump_function(self.good_memory, new_entanglement),
        )

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

    def sometimes_prot_x_helper(
        self, success_probability: float, fidelity_after_pumping: float
    ):
        if bernouli_with_probability_is_successfull(self.constants.pumping_probability):
            self.always_prot_x_helper(success_probability, fidelity_after_pumping)

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
                + self.constants.strategy.name
            )

            self.good_memory = Entanglement.from_fidelity(
                self.time, fidelity_after_pumping, self.constants.decoherence_time
            )
            self.bad_memory = None
            logger.info("purification was successfull")

        else:
            self.good_memory = None
            self.bad_memory = None
            logger.info("Purification failed")

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

    # TODO: Update simulation to work with new version of this fun
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
            # TODO: auslagern, sodass serve_request unabhängig ist
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
