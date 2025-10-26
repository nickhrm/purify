from enum import Enum
import logging
import numpy as np

from constants import (
    ENTANGLEMENT_GENERATION_COUNT,
    ENTANGLEMENT_GENERATION_SCALE,
    ENTANGLEMENT_GENERATION_SUCESS_PROBABILITY,
)

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


class Event(Enum):
    ENTANGLEMENT_GENERATION = 1
    REQUEST_ARRIVED = 2


class Node:
    good_memory = None
    bad_memory = None

    def generate_entanglement(self):
        entanglement = self._generate_entanglement()

        if entanglement is None:
            return

        # Generation was sucessfull
        if self.good_memory is None or self.good_memory < entanglement:
            self.good_memory = entanglement
        else:
            if self.bad_memory is None or self.bad_memory < entanglement:
                self.bad_memory = entanglement

    def _generate_entanglement(self):
        generation_sucessful = rng.choice(
            [True, False],
            p=[
                ENTANGLEMENT_GENERATION_SUCESS_PROBABILITY,
                1 - ENTANGLEMENT_GENERATION_SUCESS_PROBABILITY,
            ],
        )
        if generation_sucessful:
            logger.info("Entanglement Generation Sucessfull")
            return 1
        else:
            logger.info("Entanglement Generation Failed")
            return None



class Time:
    entanglement_time = 0
    entanglement_count = 0
    request_time = 0
    request_count = 0

    def current_time(self):
        return max(self.entanglement_time, self.request_time)

    def last_event(self) -> Event:
        if self.entanglement_time > self.request_time:
            return Event.ENTANGLEMENT_GENERATION
        else:
            return Event.REQUEST_ARRIVED

    def update(self, entanglement_dif, request_dif):
        new_entanglement_time = self.entanglement_time + entanglement_dif

        new_request_time = self.request_time + request_dif

        if new_entanglement_time < new_request_time:
            self.entanglement_time = new_entanglement_time
            self.entanglement_count += 1
        else:
            self.request_time = new_request_time
            self.request_count += 1


def main():
    logging.basicConfig(filename='myapp.log',filemode='w', level=logging.INFO)
    logger.info("Starting")

    node_a = Node()
    time = Time()

    entanglement_samples = rng.exponential(
        ENTANGLEMENT_GENERATION_SCALE, ENTANGLEMENT_GENERATION_COUNT
    )
    request_samples = rng.exponential(
        ENTANGLEMENT_GENERATION_SCALE * 100, ENTANGLEMENT_GENERATION_COUNT * 100
    )

    logger.info("First entanglement: "+ str(entanglement_samples[0]))
    logger.info("First request: "+ str(request_samples[0]))


    while time.entanglement_count < len(
        entanglement_samples
    ) and time.request_count < len(request_samples):
        time.update(
            entanglement_samples[time.entanglement_count],
            request_samples[time.request_count],
        )

        logger.info("Current Time is: " + str(time.current_time()) + " Last Event was: " + str(time.last_event()))
        if time.last_event() == Event.ENTANGLEMENT_GENERATION:
            node_a.generate_entanglement()

        logger.info("Node A: good-memory = " + str(node_a.good_memory) + ", bad-memory = " + str(node_a.bad_memory))

if __name__ == "__main__":
    main()
