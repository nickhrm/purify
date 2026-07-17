# purify

Implementierung meiner Bachelorarbeit: Ein PPO-Agent lernt, an einem
Quantennetzwerk-Knoten zwischen **REPLACE** (besseres Entanglement behalten)
und **Purification-Protokollen** (PROT_1/2/3, PMD) zu entscheiden, um
Teleportations-Requests mit möglichst hoher Fidelity zu bedienen.

## Setup

Benötigt [uv](https://docs.astral.sh/uv/):

```bash
uv sync                        # Environment + Dependencies
uv run pre-commit install      # Git-Hooks aktivieren (ruff + pytest bei jedem Commit)
```

## Struktur

```
src/purify/     Simulations-Kern (Physik)
  my_time.py         Event-Uhr: deterministische Entanglement-Slots, Gamma-verteilte Requests
  entanglement.py    Bell-diagonaler Zustand (F, λ₁, λ₂, λ₃) + Decay + Erzeugungs-Strategien
  node.py            Speicher-Logik: REPLACE / PUMP / Request bedienen
  qubit.py           Wartender Request-Qubit + Teleportations-Fidelity
  utils/purification_util.py   Jump-Funktionen & Erfolgswahrscheinlichkeiten der Protokolle

src/ppo/        RL-Schicht
  custom_env.py      Gymnasium-Env (seedbar über reset(seed=...))
  case_studies.py    Alle Experiment-Konfigurationen (IDs ↔ results/case_study_{id})
  trainer.py         Training (CLI: train)
  evaluate.py        Evaluation + Generalisierungs-Test (CLI: evaluate)

tests/          pytest-Suite für Physik-Kern und Env
```

## Training

```bash
uv run train --case-study 1 --coherence-times 0.01 0.05 --cores 6 --seed 42
```

Ohne `--coherence-times` werden alle Kohärenzzeiten der Case Study trainiert.
Modelle landen in `results/case_study_{id}/T{coherence_time}/`
(`best_model.zip` vom EvalCallback, `end_model.zip` nach Trainingsende).

## Evaluation

```bash
# Normale Evaluation (gleiche Umgebung wie im Training):
uv run evaluate --case-study 1 --episodes 1200 --seed 42

# Generalisierungs-Test: Modell wurde auf festen Lambdas trainiert
# (z.B. λ=(0.3, 0, 0)), wird aber in einer Umgebung evaluiert, in der
# die Fehlermasse zufällig auf λ₁, λ₂, λ₃ verteilt wird:
uv run evaluate --case-study 1 --random-lambdas --episodes 1200 --seed 42

# Dasselbe mit permutations-symmetrisierter Policy (λ werden kanonisch
# sortiert, die PROT-Aktion zurückpermutiert — kein Retraining nötig):
uv run evaluate --case-study 1 --random-lambdas --symmetrized --episodes 1200 --seed 42
```

Trainings-seitige Alternative: Case Study 21 (`PERMUTED_CONSTANTS`) trainiert
mit λ=(0.3, 0, 0), die pro Entanglement zufällig auf die Positionen permutiert
werden.

Ergebnisse: `results/case_study_{id}/evaluation.csv` bzw.
`evaluation_random_lambdas.csv`, Aktions-Verteilungen in
`action_prob*.csv` im selben Ordner.

## Tests & Linting

```bash
uv run pytest          # Unit-Tests (Physik-Invarianten, Env-Reproduzierbarkeit)
uv run ruff check src/ # Lint
uv run ruff format     # Formatierung
```

Die Pre-commit-Hooks führen ruff und pytest automatisch bei jedem Commit aus.

## Physik-Notizen

- Protokoll k ist **blind** für Fehler auf λ_k: PROT_1 kann einen reinen
  λ₁-Fehler nicht purifizieren (verschlechtert die Fidelity sogar), PROT_2/3
  dagegen schon — siehe `tests/test_purification.py`.
- PMD ist nur für Zustände mit λ₂ = λ₃ = 0 definiert.
- Nach erfolgreicher Purification wird der Speicher-Zustand als
  Werner-Zustand modelliert (Twirling-Annahme).
