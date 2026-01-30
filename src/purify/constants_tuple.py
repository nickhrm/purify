from typing import NamedTuple

from purify.my_enums import LambdaSrategy, Action


class ConstantsTuple(NamedTuple):
    coherence_time: float
    pumping_probability: float
    waiting_time_sensitivity:float
    lambda_strategy: LambdaSrategy
    lambdas: tuple[float, float, float]
    actions: tuple[Action, ...]

    def folder_name(self) -> str:
        return f"{len(self.actions)}gps_{str(self.lambdas[0]).replace(".","")}_{str(self.lambdas[1]).replace(".","")}_{str(self.lambdas[2]).replace(".","")}"


    def subfolder_name(self) -> str:
        return f"{str(self.coherence_time).replace('.', '_')}"
