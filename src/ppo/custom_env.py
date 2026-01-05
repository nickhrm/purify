import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.spaces.box import Box

from purify.my_enums import Action, Event
from purify.my_time import Time
from purify.node import Node


class TrainingEnv(gym.Env):
    def __init__(self, constants):
        super().__init__()
        self.time = Time()
        self.constants = constants
        self.node = Node(self.time, self.constants)

        # 7 Dimensionen im State:
        # [F_mem, req_wait, time_diff, F_new, L1_new, L2_new, L3_new]
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(7,),
            dtype=np.float64,
        )

        # Der Agent kann zwischen 4 Protokollen wählen (z.B. REPLACE, PROT_1, PROT_2, PROT_3)
        self.action_space = spaces.Discrete(2)

        # Interne Tracking-Variablen für das Look-Ahead
        self.last_generated_entanglement = None
        self.current_event = None

    def _get_obs(self):
        """Erstellt den State-Vektor inkl. der Lambdas des wartenden Paares."""
        req_wait = 1.0 if self.node.queue is not None else 0.0
        raw_time = self.time.get_current_time() - self.time.request_time
        decay_factor = np.exp(-raw_time / self.constants.decoherence_time)
        f_mem = self.node.get_good_memory_fidelity()

        # Falls ein neues Paar auf Verarbeitung wartet, dessen Werte in den State packen
        if self.last_generated_entanglement is not None:
            f_new = self.last_generated_entanglement.get_current_fidelity()
            l1 = self.last_generated_entanglement.get_current_lambda_1()
            l2 = self.last_generated_entanglement.get_current_lambda_2()
            l3 = self.last_generated_entanglement.get_current_lambda_3()
        else:
            f_new, l1, l2, l3 = 0.0, 0.0, 0.0, 0.0

        return np.array(
            [f_mem, req_wait, decay_factor, f_new, l1, l2, l3],
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = Time()
        self.node = Node(self.time, self.constants)

        self.last_generated_entanglement = None
        self.current_event = None

        # Ersten Zeitschritt triggern, damit wir nicht mit leerem State starten
        self.time.update()
        self.current_event = self.time.last_event()

        if self.current_event == Event.ENTANGLEMENT_GENERATION:
            self.last_generated_entanglement = self.node.generate_entanglement()

        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        # --- PHASE 1: Aktion ausführen (basierend auf dem State der Vorrunde) ---
        if self.current_event == Event.ENTANGLEMENT_GENERATION:
            # Der Agent nutzt 'action', um das Paar zu verarbeiten, das er im State gesehen hat
            if self.last_generated_entanglement is not None:
                self.node.handle_existing_entanglement(self.last_generated_entanglement, Action(action))
            self.last_generated_entanglement = None # Verarbeitung abgeschlossen

        # --- PHASE 2: Simulation einen Schritt weiterbewegen ---
        if not self.time.update():
            truncated = True

        self.current_event = self.time.last_event()

        # Falls ein Request ankommt, direkt verarbeiten (keine Agenten-Interaktion nötig)
        if self.current_event == Event.REQUEST_ARRIVAL:
            self.node.handle_request_arrival()

        # --- PHASE 3: "Look-Ahead" für den nächsten State ---
        if self.current_event == Event.ENTANGLEMENT_GENERATION:
            # Wir würfeln das neue Paar schon JETZT aus, damit es im Rückgabewert (Obs) steht
            self.last_generated_entanglement = self.node.generate_entanglement()

        # --- PHASE 4: Belohnung berechnen ---
        result = self.node.serve_request()
        if result is not None:
            (teleportation_fidelity, waiting_time) = result
            terminated = True
            reward = teleportation_fidelity
            # write_results_csv(teleportation_fidelity, self.time.get_current_time(), self.constants)

        # --- PHASE 5: Rückgabe ---
        obs = self._get_obs()
        info = {"event": self.current_event, "reward": reward}

        return obs, reward, terminated, truncated, info
