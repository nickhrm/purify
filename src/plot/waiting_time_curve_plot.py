from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from purify.my_constants import LAMBDA_1, LAMBDA_2, LAMBDA_3 # Annahme: Diese Konstanten sind definiert

def create_waiting_time_sensitivity_plot():
    """
    Erstellt ein Diagramm, das die Fidelity gegen die Waiting Time Sensitivity (κ) aufträgt.
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

        # --- 1. Daten aggregieren ---
        # WICHTIG: Gruppieren nach Strategy und WAITING TIME SENSITIVITY
        plot_df = (
            df.groupby(["strategy", "waiting_time_sensitivity"])["fidelity"]
            .mean()
            .reset_index()
        )

    except Exception as e:
        print(f"Fehler beim Lesen oder Verarbeiten der Datei: {e}")
        return

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # 2. Gruppieren nach Strategie für das Plotten
    # WICHTIG: Nach waiting_time_sensitivity sortieren
    grouped_data = plot_df.sort_values(by="waiting_time_sensitivity").groupby("strategy")

    for name, group in grouped_data:
        plt.plot(
            group["waiting_time_sensitivity"],  # X-Achse: Waiting Time Sensitivity (Kappa)
            group["fidelity"],                  # Y-Achse: Fidelity
            marker="o",
            linestyle="-",
            label=name,
        )

    # Diagramm beschriften
    plt.title(f"Fidelity vs. Waiting Time Sensitivity ($\kappa$) ({LAMBDA_1}; {LAMBDA_2}; {LAMBDA_3})")
    
    # X-Achsen-Beschriftung anpassen
    plt.xlabel("Waiting Time Sensitivity ($\kappa$)")
    plt.ylabel("Teleportation Fidelity ($F$)")
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Strategie")
    
    plt.xscale("linear") 
    plt.ylim(0.4, 1.0) # Der Bereich kann an die Daten von Figur 3b (0.7 bis 0.9) angepasst werden
    plt.tight_layout()

    # Neuer Dateiname, der die Änderung widerspiegelt
    output_file = f"fidelity_vs_kappa_{str(LAMBDA_1).replace('.', '')}_{str(LAMBDA_2).replace('.', '')}_{str(LAMBDA_3).replace('.', '')}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot erfolgreich gespeichert als: {output_file}")


if __name__ == "__main__":
    # Die ursprüngliche Funktion wird durch die neue ersetzt
    # create_pumping_probability_plot() 
    create_waiting_time_sensitivity_plot()