import logging

import numpy as np

from constants import ENTANGLEMENT_GENERATION_SUCESS_PROBABILITY, ENTANGLEMENT_INITIAL_FIDELITY
from entanglement import Entanglement
from main import Qubit
from my_enums import Strategy
from my_time import Time
from utils.bernouli_util import bernouli_with_probability
from utils.clifford_util import Cliford
from utils.csv_utils import write_results_csv


logger = logging.getLogger(__name__)
rng = np.random.default_rng()


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

