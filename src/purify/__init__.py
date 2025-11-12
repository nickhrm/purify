from __future__ import annotations

import logging
import os

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
    decoherence_times = [
        0.0001,
        0.001,
        0.005,
        0.01,
        0.05,
        0.1,
        1,
    ]

    try:
        os.remove("results/ALL_RESULTS.csv")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
    
    for strategy in strategies:
        for decoherence_time in decoherence_times:
            sim = Simulation(strategy, decoherence_time)
            sim.run()

    create_decoherence_plot()
