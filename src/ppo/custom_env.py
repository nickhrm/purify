from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces.box import Box
from pydantic_core.core_schema import time_schema

from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, Event
from purify.my_time import Time
from purify.node import Node


class TrainingEnv(gym.Env):
    def __init__(self, constants:ConstantsTuple):
        super().__init__()
        self.time = Time()
        self.constants = constants

        print("constants")
        print(self.constants)

        self.node = Node(self.time, self.constants)

        # [F_mem, request_is_waiting, time_since_last_request, L1_new, L2_new, L3_new]
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1, 1]),
            shape=(6,),
            dtype=np.float64,
        )

        self.action_space = spaces.Discrete(len(self.constants.actions))

        # Interne Tracking-Variablen für das Look-Ahead
        self.last_generated_entanglement = None
        self.current_event = None

    def _get_obs(self):
        """Creates the observation state"""
        request_is_waiting = 1.0 if self.node.queue is not None else 0.0
        time_since_last_request = min(1,self.time.get_current_time() - self.time.request_time)
        f_mem = self.node.get_good_memory_fidelity()

        if self.last_generated_entanglement is not None:
            l1 = self.last_generated_entanglement.get_current_lambda_1()
            l2 = self.last_generated_entanglement.get_current_lambda_2()
            l3 = self.last_generated_entanglement.get_current_lambda_3()
        else:
            l1, l2, l3 = 0.0, 0.0, 0.0

        # === CHANGE IS HERE ===
        # Return a dictionary {} for info, not a tuple ()
        info_dict = {
            "f_mem": f_mem,
            "request_is_waiting": request_is_waiting,
            "time_since_last_request": time_since_last_request,
            "l1": l1,
            "l2": l2,
            "l3": l3
        }

        return np.array(
            [f_mem, request_is_waiting, time_since_last_request, l1, l2, l3],
            dtype=np.float64,
        ), info_dict

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time = Time()
        self.node = Node(self.time, self.constants)

        self.last_generated_entanglement = None
        self.current_event = None

        # Advance internally until the first ENTANGLEMENT_GENERATION so the
        # agent is only ever queried when there is an entanglement to act on.
        while True:
            if not self.time.update():
                break  # Time exhausted before any entanglement – edge case
            self.current_event = self.time.last_event()
            if self.current_event == Event.REQUEST_ARRIVAL:
                self.node.handle_request_arrival()
            elif self.current_event == Event.ENTANGLEMENT_GENERATION:
                self.last_generated_entanglement = self.node.generate_entanglement()
                break

        obs, info = self._get_obs()
        return obs, info

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        # Invariant: agent is only called when current_event == ENTANGLEMENT_GENERATION,
        # so the chosen action always has a causal effect on the simulation.
        chosen_action: Action = self.constants.actions[action]

        if self.last_generated_entanglement is not None:
            self.node.handle_existing_entanglement(
                self.last_generated_entanglement, chosen_action
            )
        self.last_generated_entanglement = None

        # Try to serve a request immediately after the action
        result = self.node.serve_request()
        if result is not None:
            (teleportation_fidelity, waiting_time) = result
            terminated = True
            reward = teleportation_fidelity

        # Advance internally through non-entanglement events until the next
        # ENTANGLEMENT_GENERATION (or episode end). The agent is NOT consulted
        # for REQUEST_ARRIVAL or other intermediate events.
        if not terminated:
            while True:
                if not self.time.update():
                    truncated = True
                    break

                self.current_event = self.time.last_event()

                if self.current_event == Event.REQUEST_ARRIVAL:
                    self.node.handle_request_arrival()
                    # A new request may immediately be serveable with existing memory
                    result = self.node.serve_request()
                    if result is not None:
                        (teleportation_fidelity, waiting_time) = result
                        terminated = True
                        reward = teleportation_fidelity
                        break

                elif self.current_event == Event.ENTANGLEMENT_GENERATION:
                    self.last_generated_entanglement = self.node.generate_entanglement()
                    break  # Hand control back to the agent

        obs, info = self._get_obs()
        return obs, reward, terminated, truncated, info
