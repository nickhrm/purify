import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces.box import Box

from purify.my_constants import AVAILABLE_ACTIONS
from purify.my_enums import Event
from purify.my_time import Time
from purify.node import Node


class TrainingEnv(gym.Env):
    def __init__(self, constants):
        super().__init__()
        self.time = Time()
        self.constants = constants

        self.node = Node(self.time, self.constants)

        # [F_mem, req_wait, time_diff, difficulty_feature, L1_new, L2_new, L3_new]
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(7,),
            dtype=np.float64,
        )

        self.action_space = spaces.Discrete(len(AVAILABLE_ACTIONS))

        # Interne Tracking-Variablen für das Look-Ahead
        self.last_generated_entanglement = None
        self.current_event = None

    def _get_obs(self):
        """Creates the observation state"""
        req_wait = 1.0 if self.node.queue is not None else 0.0
        raw_time = self.time.get_current_time() - self.time.request_time
        decay_factor = np.exp(-raw_time / self.constants.decoherence_time)
        f_mem = self.node.get_good_memory_fidelity()

        # Normalize coherence time to [0,1]
        log_t = np.log10(self.constants.decoherence_time)
        difficulty_feature = (log_t + 4.0) / 3.0
        difficulty_feature = np.clip(difficulty_feature, 0.0, 1.0)

        if self.last_generated_entanglement is not None:
            f_new = self.last_generated_entanglement.get_current_fidelity()
            l1 = self.last_generated_entanglement.get_current_lambda_1()
            l2 = self.last_generated_entanglement.get_current_lambda_2()
            l3 = self.last_generated_entanglement.get_current_lambda_3()
        else:
            l1, l2, l3 =  0.0, 0.0, 0.0

        return np.array(
            [f_mem, req_wait, decay_factor, difficulty_feature, l1, l2, l3],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time = Time()
        self.node = Node(self.time, self.constants)

        self.last_generated_entanglement = None
        self.current_event = None

        self.time.update()
        self.current_event = self.time.last_event()

        if self.current_event == Event.ENTANGLEMENT_GENERATION:
            self.last_generated_entanglement = self.node.generate_entanglement()
        print(self.constants.decoherence_time)
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if self.current_event == Event.ENTANGLEMENT_GENERATION:
            if self.last_generated_entanglement is not None:
                self.node.handle_existing_entanglement(self.last_generated_entanglement,AVAILABLE_ACTIONS[action])
            self.last_generated_entanglement = None

        if not self.time.update():
            truncated = True

        self.current_event = self.time.last_event()

        if self.current_event == Event.REQUEST_ARRIVAL:
            self.node.handle_request_arrival()

        if self.current_event == Event.ENTANGLEMENT_GENERATION:
            self.last_generated_entanglement = self.node.generate_entanglement()

        result = self.node.serve_request()
        if result is not None:
            (teleportation_fidelity, waiting_time) = result
            terminated = True
            reward = teleportation_fidelity

        obs = self._get_obs()
        info = {"event": self.current_event, "reward": reward}

        return obs, reward, terminated, truncated, info
