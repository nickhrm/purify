from ppo.custom_env import TrainingEnv
from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, LambdaSrategy

consts = ConstantsTuple(
    coherence_time=0.01,
    lambda_strategy=LambdaSrategy.USE_CONSTANTS,
    waiting_time_sensitivity=1,
    pumping_probability=1.0,
    lambdas=(0.0, 0.3, 0.0),
    actions=(
        Action.REPLACE,
        Action.PROT_1,
        Action.PROT_2,
        Action.PROT_3,
    ),
)
env = TrainingEnv(consts)
obs, info = env.reset()
print("Initial obs:", obs)
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
print("Successfully ran 100 random steps without crashing.")
