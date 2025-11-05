from __future__ import annotations

import argparse
import logging
import numpy as np
from constants import ENTANGLEMENT_DECOHERENCE_CONSTANT
from my_enums import Strategy
from my_time import Time
from my_simulation import Simulation
from utils.plot_util import create_boxplot

logger = logging.getLogger(__name__)
rng = np.random.default_rng()




class Qubit:
    def __init__(self, time: Time) -> None:
        self._time = time
        self.creationTime: float = time.get_current_time()

    def get_current_fidelity(self) -> float | None:
        if self is None:
            return None
        current_time = self._time.get_current_time()
        time_alive = current_time - self.creationTime
        return (np.exp(-time_alive / ENTANGLEMENT_DECOHERENCE_CONSTANT) + 2.0) / 3.0

    def teleportation_fidelity(self, entanglement_fidelity: float) -> float:
        Fe = float(np.clip(entanglement_fidelity, 0.0, 1.0))
        Fq = self.get_current_fidelity()

        term1 = ((2.0 * Fe + 1.0) * Fq) / 3.0
        term2 = (2.0 * (Fe - 1.0) * (1.0 - Fq)) / 3.0

        # Radikanden stabilisieren (Rundungsfehler -> kleine negative Zahlen vermeiden)
        a = np.sqrt(max(term1, 0.0))
        b = np.sqrt(max(term2, 0.0))

        FT = (a - b) ** 2
        return float(np.clip(FT, 0.0, 1.0))

    def get_waiting_time(self):
        return self._time.get_current_time() - self.creationTime



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        type=int,
        choices=range(1, len(Strategy) + 1),
        required=True,
    )
    args = parser.parse_args()
    strategy = Strategy(args.strategy)

    logging.basicConfig(
        filename="myapp.log",
        filemode="w",
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    logger.info("Starting simulation")

    sim = Simulation(strategy)
    sim.run()
    create_boxplot()


if __name__ == "__main__":
    main()
