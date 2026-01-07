from typing import NamedTuple

from ale_py import Action

from purify.my_enums import LambdaSrategy


class ConstantsTuple(NamedTuple):
    decoherence_time: float
    pumping_probability: float
    waiting_time_sensitivity:float
    lambda_strategy: LambdaSrategy

