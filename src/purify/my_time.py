
import numpy as np
from purify.my_constants import DELTA_T, QUBIT_ARRIVAL_SCALE, QUBIT_ENTANGLEMENT_FACTOR
from purify.my_enums import Event

rng = np.random.default_rng()

# How many request samples to pre-generate at a time (lazy chunks).
_REQUEST_CHUNK_SIZE = 64


class Time:
    def __init__(self) -> None:
        self.entanglement_time: float = 0.0
        self.entanglement_count: int = 0
        self.request_time: float = 0.0
        self.request_count: int = 0

        # Entanglement arrivals are deterministic (fixed DELTA_T apart),
        # so no pre-allocation needed – we compute the next time on the fly.

        # Request samples are generated lazily in small chunks.
        self._request_samples: list[float] = []
        self._request_offset: int = 0  # how many of _request_samples were consumed
        self._request_exhausted: bool = False

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ensure_next_request_sample(self) -> bool:
        """Generate the next chunk of request samples if needed.
        Returns False if we decide the simulation should end (practically
        unlimited – mirroring the old ENTANGLEMENT_GENERATION_COUNT limit)."""
        if self._request_offset < len(self._request_samples):
            return True
        if self._request_exhausted:
            return False
        new_samples = rng.gamma(
            shape=2,
            scale=1 / QUBIT_ARRIVAL_SCALE,
            size=_REQUEST_CHUNK_SIZE,
        ).tolist()
        self._request_samples.extend(new_samples)
        return True

    # ── public API ────────────────────────────────────────────────────────────

    def get_current_time(self) -> float:
        return max(self.entanglement_time, self.request_time)

    def last_event(self) -> Event:
        if self.entanglement_time > self.request_time:
            return Event.ENTANGLEMENT_GENERATION
        else:
            return Event.REQUEST_ARRIVAL

    def update(self) -> bool:
        """Advance to the next event. Returns True on success, False when
        there are no more request samples available (episode ends)."""
        has_request = self._ensure_next_request_sample()
        if not has_request:
            return False

        new_entanglement_time = self.entanglement_time + DELTA_T
        new_request_time = (
            self.request_time + self._request_samples[self._request_offset]
        )

        if new_entanglement_time < new_request_time:
            self.entanglement_time = new_entanglement_time
            self.entanglement_count += 1
        else:
            self.request_time = new_request_time
            self.request_count += 1
            self._request_offset += 1

        return True
