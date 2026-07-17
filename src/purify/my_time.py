import numpy as np

from purify.my_constants import DELTA_T, QUBIT_ARRIVAL_SCALE
from purify.my_enums import Event

# How many request samples to pre-generate at a time (lazy chunks).
_REQUEST_CHUNK_SIZE = 64


class Time:
    """Event clock of the simulation.

    Entanglement arrivals are deterministic (fixed DELTA_T apart);
    request arrivals are gamma-distributed and sampled lazily in chunks
    from the provided RNG, so a seeded generator makes runs reproducible.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()

        self.entanglement_time: float = 0.0
        self.entanglement_count: int = 0
        self.request_time: float = 0.0
        self.request_count: int = 0

        self._request_samples: list[float] = []
        self._request_offset: int = 0  # how many of _request_samples were consumed

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ensure_next_request_sample(self) -> None:
        """Generate the next chunk of request samples if needed."""
        if self._request_offset < len(self._request_samples):
            return
        new_samples = self._rng.gamma(
            shape=2,
            scale=1 / QUBIT_ARRIVAL_SCALE,
            size=_REQUEST_CHUNK_SIZE,
        ).tolist()
        self._request_samples.extend(new_samples)

    # ── public API ────────────────────────────────────────────────────────────

    def get_current_time(self) -> float:
        return max(self.entanglement_time, self.request_time)

    def last_event(self) -> Event:
        if self.entanglement_time > self.request_time:
            return Event.ENTANGLEMENT_GENERATION
        else:
            return Event.REQUEST_ARRIVAL

    def update(self) -> None:
        """Advance the clock to the next event."""
        self._ensure_next_request_sample()

        new_entanglement_time = self.entanglement_time + DELTA_T
        new_request_time = self.request_time + self._request_samples[self._request_offset]

        if new_entanglement_time < new_request_time:
            self.entanglement_time = new_entanglement_time
            self.entanglement_count += 1
        else:
            self.request_time = new_request_time
            self.request_count += 1
            self._request_offset += 1
