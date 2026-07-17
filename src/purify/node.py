import logging
from typing import cast

import numpy as np

from purify.constants_tuple import ConstantsTuple
from purify.entanglement import Entanglement
from purify.my_constants import P_G
from purify.my_enums import Action
from purify.my_time import Time
from purify.qubit import Qubit
from purify.utils.bernoulli_util import bernoulli
from purify.utils.purification_util import Purification

logger = logging.getLogger(__name__)


class Node:
    def __init__(
        self,
        time: Time,
        constants: ConstantsTuple,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.time = time
        self.constants: ConstantsTuple = constants
        self.good_memory: Entanglement | None = None
        self.queue: Qubit | None = None
        self._rng = rng if rng is not None else np.random.default_rng()

    def needs_agent_decision(self) -> bool:
        """True when good_memory is occupied and the agent must choose
        between REPLACE and PUMP for the newly arrived entanglement.
        False when good_memory is empty – the entanglement is stored
        automatically without any meaningful choice."""
        return self.good_memory is not None

    def store_first_entanglement(self, entanglement: Entanglement) -> None:
        """Store entanglement when good_memory is empty (no agent decision needed).
        Precondition: needs_agent_decision() is False."""
        assert self.good_memory is None, "store_first_entanglement called with occupied memory"
        self.good_memory = entanglement

    def apply_action(self, entanglement: Entanglement, action: Action) -> None:
        """Apply the agent's chosen action against the existing good_memory.
        Precondition: needs_agent_decision() is True."""
        assert self.good_memory is not None, "apply_action called with empty memory"
        if action == Action.REPLACE:
            self._keep_best(entanglement)
        else:
            self._pump(entanglement, action)

    def _keep_best(self, entanglement: Entanglement) -> None:
        if self.good_memory.get_current_fidelity() < entanglement.get_current_fidelity():
            self.good_memory = entanglement

    def _pump(self, new_entanglement: Entanglement, action: Action) -> None:
        good_memory = cast(Entanglement, self.good_memory)
        success_probability = Purification.success_probability_from_action(
            good_memory, new_entanglement, action
        )
        fidelity_after_pumping = Purification.jump_function_from_action(
            good_memory, new_entanglement, action
        )

        if bernoulli(self._rng, success_probability):
            self.good_memory = Entanglement.from_fidelity(
                self.time, fidelity_after_pumping, self.constants.coherence_time
            )
            logger.info("purification was successful")
        else:
            self.good_memory = None
            logger.info("Purification failed")

    def generate_entanglement(self) -> Entanglement | None:
        if bernoulli(self._rng, P_G):
            logger.info("Entanglement Generation Successful")
            return Entanglement.from_strategy(self.time, self.constants, self._rng)
        else:
            logger.info("Entanglement Generation Failed")
            return None

    def put_request_in_queue(self) -> None:
        # fill queue if empty
        if self.queue is None:
            self.queue = Qubit(self.time, self.constants)
        else:
            # if queue was already full, request is dropped
            logger.info("Serving request failed. queue was already full")

    def serve_request_if_available(self) -> tuple[float, float] | None:
        """Serves the waiting request if both a request and a stored
        entanglement are available. Returns (teleportation_fidelity,
        waiting_time) on success, otherwise None."""
        if self.queue is None or self.good_memory is None:
            return None

        teleportation_fidelity: float = self.queue.teleportation_fidelity(
            self.good_memory.get_current_fidelity()
        )
        waiting_time = self.queue.get_waiting_time()
        logger.info(
            "Served request with fidelity %s and waiting time %s. Initial qubit had fidelity: %s",
            teleportation_fidelity,
            waiting_time,
            self.queue.get_current_fidelity(),
        )

        # remove qubit from queue, because it was served
        self.queue = None
        # discard link in good_memory, because it was used
        self.good_memory = None
        return (teleportation_fidelity, waiting_time)

    def get_good_memory_fidelity(self) -> float:
        if self.good_memory is None:
            return 0.0
        return self.good_memory.get_current_fidelity()
