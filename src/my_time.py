

from my_enums import Event


class Time:
    def __init__(self) -> None:
        self.entanglement_time: float = 0.0
        self.entanglement_count: int = 0
        self.request_time: float = 0.0
        self.request_count: int = 0

    def get_current_time(self) -> float:
        return max(self.entanglement_time, self.request_time)

    def last_event(self) -> Event:
        if self.entanglement_time > self.request_time:
            return Event.ENTANGLEMENT_GENERATION
        else:
            return Event.REQUEST_ARRIVED

    def update(self, entanglement_dif: float, request_dif: float) -> None:
        new_entanglement_time = self.entanglement_time + entanglement_dif
        new_request_time = self.request_time + request_dif

        if new_entanglement_time < new_request_time:
            self.entanglement_time = new_entanglement_time
            self.entanglement_count += 1
        else:
            self.request_time = new_request_time
            self.request_count += 1


