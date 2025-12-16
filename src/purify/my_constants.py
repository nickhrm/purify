
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import LambdaSrategy
from purify.my_enums import Action

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



LEARNING_ENV_CONSTANTS = ConstantsTuple(
            decoherence_time=0.05,
            strategy=Action.TRAINING_MODE,

            # These are not used
            lambda_strategy=LambdaSrategy.USE_CONSTANTS,
            waiting_time_sensitivity=1,
            pumping_probability=1.0,
        )



WAITING_TIME_SENSIVITIES = [
  1
]

LAMBDA_1 = 0.1
LAMBDA_2 = 0.0
LAMBDA_3 = 0.2


LAMBDA_STRAT = [
    # LambdaSrategy.RANDOM_WITH_LARGEST_LAMBDA
    LambdaSrategy.USE_CONSTANTS
]



# Stratgies to include in the simulation
STRATEGIES = [
        Action.REPLACE,
        Action.PROT_1,
        Action.PROT_2,
        Action.PROT_3,
        # Strategy.ALWAYS_PMD,
    ]


# Decoherence times to include in the simulation in secs
DECOHERENCE_TIMES = [
        # 0.00001,
         0.0001,
         0.0003,
         0.0005,
         0.0008,
         0.001,
         0.003,
         0.005,
         0.008,
        0.01,
         0.05,
         0.1,
        # 0.5
        # 1.0,
    ]


PUMPING_PROBABILTIES = [
    # 0.0,
    # 0.2,
    # 0.4,
    # 0.6,
    # 0.8,
    1.0
]

