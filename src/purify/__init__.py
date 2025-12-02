from purify.utils.path_util import path_from_lambdas
from purify.my_constants import LAMBDA_STRAT
import logging
import math
import os

import numpy as np

from purify.constants_tuple import ConstantsTuple
from purify.my_constants import (
    DECOHERENCE_TIMES,
    ETA,
    LAMBDA_1,
    LAMBDA_2,
    LAMBDA_3,
    LENGTH,
    P_G,
    PUMPING_PROBABILTIES,
    STRATEGIES,
    WAITING_TIME_SENSIVITIES,
)
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
        level=logging.WARNING,
        format="%(levelname)s - %(message)s",
    )
    logger.info("Starting simulation")
    logger.warning(math.exp(-ETA * LENGTH))
    logger.warning(P_G)

    try:
        os.remove(path_from_lambdas())
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")

    for strategy in STRATEGIES:
        logger.warning(f"Starting Strategy {strategy}")
        for decoherence_time in DECOHERENCE_TIMES:
            logger.warning(f"Starting decoherence time {decoherence_time}")
            for pumping_probabily in PUMPING_PROBABILTIES:
                logger.warning(f"Starting Pumping Probability {pumping_probabily}")
                for waiting_time_sensitivity in WAITING_TIME_SENSIVITIES:
                    logger.warning(
                        f"Starting Waiting time sensitivity: {waiting_time_sensitivity}"
                    )
                    for lambda_strat in LAMBDA_STRAT:
                        logger.warning(
                        f"Starting Lambda Strategy: {lambda_strat}"
                    )
                        constant_tuple = ConstantsTuple(
                            strategy=strategy,
                            decoherence_time=decoherence_time,
                            pumping_probability=pumping_probabily,
                            waiting_time_sensitivity=waiting_time_sensitivity,
                            lambda_strategy=lambda_strat,
                        )
                        sim = Simulation(constant_tuple)
                        sim.run()
