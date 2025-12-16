from dataclasses import dataclass
from typing import NamedTuple

from purify.my_enums import LambdaSrategy, Action

class ConstantsTuple(NamedTuple):
    strategy: Action
    decoherence_time: float
    pumping_probability: float
    waiting_time_sensitivity:float
    lambda_strategy: LambdaSrategy

