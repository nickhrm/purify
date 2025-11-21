import logging
from math import log
import os

import numpy as np

from plot.curve_plot import create_decoherence_plot
from purify.constants_tuple import ConstantsTuple
from purify.my_constants import DECOHERENCE_TIMES, PUMPING_PROBABILTIES, STRATEGIES
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
        # filename="myapp.log",
        # filemode="w",
        level=logging.WARNING,
        format="%(levelname)s - %(message)s",
    )
    logger.info("Starting simulation")

    try:
        os.remove("results/ALL_RESULTS.csv")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")


    for strategy in STRATEGIES:
        logger.warning(f"Starting Strategy {strategy}")
        for decoherence_time in DECOHERENCE_TIMES:
            logger.warning(f"Starting decoherence time {decoherence_time}")
            for pumping_probabily in PUMPING_PROBABILTIES:
                logger.warning(f"Starting Pumping Probability {pumping_probabily}")
                constant_tuple = ConstantsTuple(
                strategy=strategy,
                decoherence_time=decoherence_time,
                pumping_probability=pumping_probabily
            )
                sim = Simulation(constant_tuple)
                sim.run()

    create_decoherence_plot()


