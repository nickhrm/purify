import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces.box import Box

from purify.constants_tuple import ConstantsTuple
from purify.my_enums import Action, Event
from purify.my_time import Time
from purify.node import Node


class TrainingEnv(gym.Env):
    """Gym-Umgebung für das Purification-Problem.

    Eine Episode läuft, bis der erste Request nach der ersten
    Agenten-Entscheidung bedient wird; der Reward ist die
    Teleportations-Fidelity. Der Agent wird nur konsultiert, wenn ein
    neues Entanglement ankommt UND good_memory bereits belegt ist
    (REPLACE vs. PUMP) – alle anderen Events laufen intern ab.

    Reproduzierbarkeit: `reset(seed=...)` seedet den Generator, aus dem
    sämtliche Zufälligkeit (Request-Zeiten, Generierung, Purification,
    Lambda-Verteilung) gezogen wird.
    """

    def __init__(self, constants: ConstantsTuple):
        super().__init__()
        self.constants = constants

        self.time = Time(self.np_random)
        self.node = Node(self.time, self.constants, self.np_random)

        # [F_mem, request_is_waiting, time_since_last_request, L1_new, L2_new, L3_new]
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0, 0]),
            high=np.array([1, 1, 1, 1, 1, 1]),
            shape=(6,),
            dtype=np.float64,
        )

        self.action_space = spaces.Discrete(len(self.constants.actions))

        # Das zuletzt generierte Entanglement, über das der Agent entscheidet
        self.last_generated_entanglement = None

    def _get_obs(self):
        """Creates the observation state"""
        request_is_waiting = 1.0 if self.node.queue is not None else 0.0
        time_since_last_request = min(1, self.time.get_current_time() - self.time.request_time)
        f_mem = self.node.get_good_memory_fidelity()

        if self.last_generated_entanglement is not None:
            l1 = self.last_generated_entanglement.get_current_lambda_1()
            l2 = self.last_generated_entanglement.get_current_lambda_2()
            l3 = self.last_generated_entanglement.get_current_lambda_3()
        else:
            l1, l2, l3 = 0.0, 0.0, 0.0

        info_dict = {
            "f_mem": f_mem,
            "request_is_waiting": request_is_waiting,
            "time_since_last_request": time_since_last_request,
            "l1": l1,
            "l2": l2,
            "l3": l3,
        }

        return np.array(
            [f_mem, request_is_waiting, time_since_last_request, l1, l2, l3],
            dtype=np.float64,
        ), info_dict

    def _advance_to_next_decision(self, in_episode: bool) -> tuple[bool, float]:
        """Simuliert Events, bis der Agent eine echte Entscheidung hat
        (REPLACE vs. PUMP) oder ein Request bedient wird.

        Der Agent wird NICHT konsultiert für REQUEST_ARRIVAL,
        fehlgeschlagene Generierungen oder wenn good_memory leer ist.

        `in_episode=True` (in step): Bedienen eines Requests beendet die
        Episode mit der Teleportations-Fidelity als Reward.
        `in_episode=False` (Warm-up in reset): Requests werden zwar
        bedient, aber ohne Agenten-Beteiligung – die Episode beginnt erst
        bei der ersten echten Entscheidung.

        Returns (terminated, reward).
        """
        while True:
            self.time.update()
            serve_result = None

            if self.time.last_event() == Event.REQUEST_ARRIVAL:
                self.node.put_request_in_queue()
                # Der neue Request ist ggf. sofort mit vorhandenem Memory bedienbar
                serve_result = self.node.serve_request_if_available()

            elif self.time.last_event() == Event.ENTANGLEMENT_GENERATION:
                entanglement = self.node.generate_entanglement()
                if entanglement is None:
                    continue  # Generierung fehlgeschlagen – weiter simulieren
                if self.node.needs_agent_decision():
                    # Echte Wahl: REPLACE vs. PUMP
                    self.last_generated_entanglement = entanglement
                    return False, 0.0
                # good_memory leer – automatisch speichern, keine Wahl nötig
                self.node.store_first_entanglement(entanglement)
                serve_result = self.node.serve_request_if_available()

            if serve_result is not None and in_episode:
                (teleportation_fidelity, _waiting_time) = serve_result
                return True, teleportation_fidelity

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.time = Time(self.np_random)
        self.node = Node(self.time, self.constants, self.np_random)
        self.last_generated_entanglement = None

        self._advance_to_next_decision(in_episode=False)

        obs, info = self._get_obs()
        return obs, info

    def step(self, action):
        # Invariant: agent is only called when needs_agent_decision() is True,
        # i.e. an entanglement arrived AND good_memory is already occupied.
        chosen_action: Action = self.constants.actions[action]
        self.node.apply_action(self.last_generated_entanglement, chosen_action)
        self.last_generated_entanglement = None

        # Try to serve a request immediately after the action
        result = self.node.serve_request_if_available()
        if result is not None:
            terminated, reward = True, result[0]
        else:
            terminated, reward = self._advance_to_next_decision(in_episode=True)

        obs, info = self._get_obs()
        return obs, reward, terminated, False, info
