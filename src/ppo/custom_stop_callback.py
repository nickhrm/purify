from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement


class CustomStopCallback(StopTrainingOnNoModelImprovement):
    def __init__(self, max_no_improvement_evals: int, min_evals: int, verbose: int = 1, param_label: str = ""):
        super().__init__(max_no_improvement_evals=max_no_improvement_evals, min_evals=min_evals, verbose=verbose)
        self.param_label = param_label
        self.eval_count = 0

    def _on_step(self) -> bool:
        self.eval_count += 1

        continue_training = super()._on_step()

        remaining_lives = self.max_no_improvement_evals - self.no_improvement_evals

        # Logging
        if self.eval_count >= self.min_evals:
            if remaining_lives > 0:
                if self.no_improvement_evals > 0:
                     print(f"   >>> [{self.param_label}] Status: Keine Verbesserung seit {self.no_improvement_evals} Evals.")
                     print(f"   >>> [{self.param_label}] Abbruch in {remaining_lives} weiteren schlechten Evals.")
                else:
                     print(f"   >>> [{self.param_label}] Status: Neues bestes Modell gefunden! Zähler zurückgesetzt.")
            else:
                print(f"   >>> [{self.param_label}] LIMIT ERREICHT ({self.max_no_improvement_evals} Fehlversuche). Stoppe Training.")
        else:
             print(f"   >>> [{self.param_label}] Warmup: Eval {self.eval_count}/{self.min_evals} (Early Stopping noch inaktiv)")

        return continue_training