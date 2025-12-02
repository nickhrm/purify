from typing import NamedTuple

from purify.my_enums import LambdaSrategy, Strategy


class ConstantsTuple(NamedTuple):
    strategy: Strategy
    decoherence_time: float
    pumping_probability: float
    waiting_time_sensitivity:float
    lambda_strategy: LambdaSrategy

