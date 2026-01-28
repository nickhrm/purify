class CustomStopCallback(StopTrainingOnNoModelImprovement):
    def __init__(self, max_no_improvement_evals: int, min_evals: int, verbose: int = 1, param_label: str = ""):
        super().__init__(max_no_improvement_evals=max_no_improvement_evals, min_evals=min_evals, verbose=verbose)
        self.param_label = param_label

    def _on_step(self) -> bool:
        # Führe erst die normale Logik der Elternklasse aus (prüft improvement)
        continue_training = super()._on_step()

        # Jetzt fügen wir unser Custom Logging hinzu
        # self.no_improvement_evals ist der interne Zähler der Elternklasse
        remaining_lives = self.max_no_improvement_evals - self.no_improvement_evals
        
        # Nur loggen, wenn wir schon über min_evals sind (sonst ist der Zähler irrelevant)
        if self.n_evals >= self.min_evals:
            if remaining_lives > 0:
                print(f"   >>> [{self.param_label}] Status: Keine Verbesserung seit {self.no_improvement_evals} Evals.")
                print(f"   >>> [{self.param_label}] Abbruch in {remaining_lives} weiteren schlechten Evals.")
            else:
                print(f"   >>> [{self.param_label}] LIMIT ERREICHT. Stoppe Training.")
        else:
             print(f"   >>> [{self.param_label}] Warmup: Eval {self.n_evals}/{self.min_evals} (Early Stopping noch inaktiv)")

        return continue_training