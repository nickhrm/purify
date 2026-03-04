from dataclasses import dataclass

@dataclass
class HyperparamsTuple:
    """
    Definiert die Hyperparameter für das PPO-Training.
    """
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    ent_coef: float
    clip_range: float
    vf_coef: float
    max_grad_norm: float

    def to_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ent_coef": self.ent_coef,
            "clip_range": self.clip_range,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}, "activation_fn": "Tanh"}
        }
