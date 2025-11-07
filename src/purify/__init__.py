from __future__ import annotations

import argparse
import logging

import numpy as np

from purify.my_enums import Strategy
from purify.my_simulation import Simulation
from plot.plot_util import create_boxplot

logger = logging.getLogger(__name__)
rng = np.random.default_rng()



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


