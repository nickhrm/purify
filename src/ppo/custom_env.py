from cv2 import log
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces.box import Box

from purify.constants_tuple import ConstantsTuple
from purify.my_constants import (
    LEARNING_ENV_CONSTANTS,
)
from purify.my_enums import Action, Event, LambdaSrategy, Action
from purify.my_simulation import Simulation
from purify.my_time import Time
from purify.node import Node

# ... (Ihre Imports für Simulation, ConstantsTuple, Node, etc.)


class TrainingEnv(gym.Env):
    def __init__(
        self,
    ):
        super().__init__()
        self.time = Time()

        self.constants = LEARNING_ENV_CONSTANTS

        self.node = Node(self.time, self.constants)

        # State (F_e, is_request_waiting, time_since_last_request)
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, np.inf]),
            shape=(3,),
            dtype=np.float64,
        )
        self.action_space = spaces.Discrete(4)

        self.info = {}

    def reset(self, seed=None, options=None):
        self.time = Time()
        self.node = Node(self.time, self.constants)

        req_wait = 0 if self.node.queue is None else 1

        self.curr_obs = np.array(
            [self.node.get_good_memory_fidelity(), req_wait, 0.0], dtype=np.float32
        )

        self.info = {}

        return self.curr_obs, {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        if not self.time.update():
            truncated = True
        current_event = self.time.last_event()

        if current_event == Event.ENTANGLEMENT_GENERATION:
            self.node.handle_entanglement_generation(Action(action))
        elif current_event == Event.REQUEST_ARRIVAL:
            self.node.handle_request_arrival()

        result = self.node.serve_request()
        if result is not None:
            (teleportation_fidelity, waiting_time) = result
            # print(f"Telepored qubit with f={teleportation_fidelity}")
            terminated = True
            reward = teleportation_fidelity

        req_wait = 0 if self.node.queue is None else 1
        time_since_last_req = self.time.get_current_time() - self.time.request_time

        self.curr_obs = np.array(
            [self.node.get_good_memory_fidelity(), req_wait, time_since_last_req],
            dtype=np.float32,
        )
        self.info = {
            "event" : current_event,
            "reward" : reward
        }

        return self.curr_obs, reward, terminated, truncated, self.info
