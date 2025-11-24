from typing import NamedTuple

from purify.my_enums import Strategy


class ConstantsTuple(NamedTuple):
    strategy: Strategy
    decoherence_time: float
    pumping_probability: float
    waiting_time_sensitivity:float

