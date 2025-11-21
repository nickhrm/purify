from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from purify.my_constants import LAMBDA_1, LAMBDA_2, LAMBDA_3

def create_pumping_probability_plot():
    """
    Erstellt ein Diagramm, das die Fidelity gegen die Pumping Probability aufträgt.
    """
    # Dateiname basierend auf den Konstanten
    file_name = f"results/ALL_RESULTS_{LAMBDA_1}_{LAMBDA_2}_{LAMBDA_3}.csv"
    results_file = Path(file_name)

    if not results_file.exists():
        print(f"Fehler: Die Datei '{results_file}' wurde nicht gefunden.")
        return

    try:
        # Datei einlesen
        df = pd.read_csv(results_file)

        # 1. Daten aggregieren: 
        # Gruppieren nach Strategy und PUMPING PROBABILITY (statt Decoherence Time)
        plot_df = (
            df.groupby(["strategy", "pumping_probability"])["fidelity"]
            .mean()
            .reset_index()
        )

    except Exception as e:
        print(f"Fehler beim Lesen oder Verarbeiten der Datei: {e}")
        return

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # 2. Gruppieren nach Strategie für das Plotten
    # WICHTIG: Vorher nach pumping_probability sortieren, damit die Linie nicht springt
    grouped_data = plot_df.sort_values(by="pumping_probability").groupby("strategy")

    for name, group in grouped_data:
        plt.plot(
            group["pumping_probability"],  # X-Achse: Pumping Prob
            group["fidelity"],             # Y-Achse: Fidelity
            marker="o",
            linestyle="-",
            label=name,
        )

    # Diagramm beschriften
    plt.title(f"Fidelity vs. Pumping Probability ({LAMBDA_1}; {LAMBDA_2}; {LAMBDA_3})")
    plt.xlabel("Pumping Probability ($P_{pump}$)")
    plt.ylabel("Teleportation Fidelity ($F$)")
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Strategie")
    
    # Skalierung: Meistens linear für Wahrscheinlichkeiten. 
    # Falls du sehr kleine Werte hast (z.B. 1e-4), nutze: plt.xscale("log")
    plt.xscale("linear") 
    
    plt.ylim(0.4, 1.0)
    plt.tight_layout()

    # Neuer Dateiname
    output_file = f"fidelity_vs_pumping_{str(LAMBDA_1).replace('.', '')}_{str(LAMBDA_2).replace('.', '')}_{str(LAMBDA_3).replace('.', '')}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot erfolgreich gespeichert als: {output_file}")


if __name__ == "__main__":
    create_pumping_probability_plot()