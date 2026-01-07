
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy

ENTANGLEMENT_GENERATION_COUNT = 300000

LENGTH = 20000
C = 2e8

# time between entanglemement interrival in [sec]
DELTA_T = LENGTH/C
# Success probability of entangelement generation
P_G = 0.3047

# Transmittion property of the cable. In db/m
ETA = -0.00015

# P_G = decibel_to_linear(ETA * LENGTH)

QUBIT_ENTANGLEMENT_FACTOR = 100

QUBIT_ARRIVAL_SCALE = P_G/(QUBIT_ENTANGLEMENT_FACTOR * DELTA_T)

AVAILABLE_ACTIONS = [
    Action.REPLACE,
    Action.PROT_1,
    Action.PROT_2,
    Action.PROT_3,
]


LAMBDA_1 = 0.3
LAMBDA_2 = 0.0
LAMBDA_3 = 0.0

