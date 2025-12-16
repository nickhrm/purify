

import numpy as np
from purify.my_constants import DELTA_T, ENTANGLEMENT_GENERATION_COUNT, QUBIT_ARRIVAL_SCALE, QUBIT_ENTANGLEMENT_FACTOR
from purify.my_enums import Event

rng = np.random.default_rng()


class Time:
    def __init__(self) -> None:
        self.entanglement_time: float = 0.0
        self.entanglement_count: int = 0
        self.request_time: float = 0.0
        self.request_count: int = 0

        self.entanglement_samples = [
            DELTA_T for _ in range(ENTANGLEMENT_GENERATION_COUNT)
        ]

        self.request_samples = rng.gamma(
            shape=2,
            scale=1 / QUBIT_ARRIVAL_SCALE,
            size=round(ENTANGLEMENT_GENERATION_COUNT / QUBIT_ENTANGLEMENT_FACTOR),
        )


    def get_current_time(self) -> float:
        return max(self.entanglement_time, self.request_time)

    def last_event(self) -> Event:
        if self.entanglement_time > self.request_time:
            return Event.ENTANGLEMENT_GENERATION
        else:
            return Event.REQUEST_ARRIVAL


    '''Returns true if update was made. Returns false if all samples were already used'''
    def update(self) -> bool:

        if self.entanglement_count >= len(
            self.entanglement_samples
        ) or self.request_count >= len(self.request_samples):
            return False


        new_entanglement_time = self.entanglement_time + self.entanglement_samples[self.entanglement_count]
        new_request_time = self.request_time + self.request_samples[self.request_count]

        if new_entanglement_time < new_request_time:
            self.entanglement_time = new_entanglement_time
            self.entanglement_count += 1
        else:
            self.request_time = new_request_time
            self.request_count += 1
        return True


