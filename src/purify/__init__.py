import logging
import os

import numpy as np

from purify.my_constants import DECOHERENCE_TIMES, STRATEGIES
from purify.my_simulation import Simulation
from purify.plot.curve_plot import create_decoherence_plot

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
        # filename="myapp.log",
        # filemode="w",
        level=logging.CRITICAL,
        format="%(levelname)s - %(message)s",
    )
    logger.info("Starting simulation")

    try:
        os.remove("results/ALL_RESULTS.csv")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

    for strategy in STRATEGIES:
        for decoherence_time in DECOHERENCE_TIMES:
            sim = Simulation(strategy, decoherence_time)
            sim.run()

    create_decoherence_plot()
