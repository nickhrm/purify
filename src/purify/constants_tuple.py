from typing import NamedTuple

from purify.my_enums import LambdaSrategy


class ConstantsTuple(NamedTuple):
    coherence_time: float
    pumping_probability: float
    waiting_time_sensitivity:float
    lambda_strategy: LambdaSrategy
    lambdas: tuple[float, float, float]

