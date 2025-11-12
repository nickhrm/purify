from __future__ import annotations

import logging

import numpy as np

from plot.curve_plot import create_decoherence_plot
from purify.my_enums import Strategy
from purify.my_simulation import Simulation

logger = logging.getLogger(__name__)
rng = np.random.default_rng()


def main() -> None:
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--strategy",
    #     type=int,
    #     choices=range(1, len(Strategy) + 1),
    #     required=True,
    # )
    # args = parser.parse_args()
    # strategy = Strategy(args.strategy)

    logging.basicConfig(
        filename="myapp.log",
        filemode="w",
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    logger.info("Starting simulation")

    strategies = [
        Strategy.ALWAYS_REPLACE,

        Strategy.ALWAYS_PROT_1,
        Strategy.ALWAYS_PROT_2,
        Strategy.ALWAYS_PROT_3,
        # Strategy.ALWAYS_PMD
    ]

    # in mili secs
    decoherence_times = [0.001, 0.005, 0.01, 0.05, 0.1]

    for strategy in strategies:
        for decoherence_time in decoherence_times:
            sim = Simulation(strategy, decoherence_time)
            sim.run()

    create_decoherence_plot()
