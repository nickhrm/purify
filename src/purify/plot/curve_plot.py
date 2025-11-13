from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from purify.my_constants import LAMBDA_1, LAMBDA_2, LAMBDA_3


def create_decoherence_plot():
    """
    Erstellt ein Diagramm, das die Fidelity gegen die Decoherence Time aufträgt,
    basierend auf einer einzigen, zusammengeführten CSV-Datei.
    """
    results_file = Path("results/ALL_RESULTS.csv")

    if not results_file.exists():
        print(f"Fehler: Die Datei '{results_file}' wurde nicht gefunden.")
        return

    try:
        # Nur einmal die gesamte Datei einlesen
        df = pd.read_csv(results_file)

        # 1. Daten aggregieren: Durchschnittliche Fidelity pro Protokoll und Decoherence Time
        plot_df = (
            df.groupby(["protocol_name", "decoherence_time"])["fidelity"]
            .mean()
            .reset_index()
        )

    except Exception as e:
        print(f"Fehler beim Lesen oder Verarbeiten der Datei: {e}")
        return

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    # 2. Gruppieren nach Protokollname für das Plotten
    grouped_data = plot_df.sort_values(by="decoherence_time").groupby("protocol_name")

    # Sortieren der Protokollnamen (optional, aber hilfreich für konsistente
    # Farben/Legenden)
    for name, group in grouped_data:
        # Erzeuge eine Linie für jedes Protokoll
        plt.plot(
            group["decoherence_time"],
            group["fidelity"],
            marker="o",
            linestyle="-",
            label=name,  # Name ist bereits der korrekte Label-String
        )

    # Diagramm beschriften
    plt.title(
        f"Entanglement({LAMBDA_1}; {LAMBDA_2}; {LAMBDA_3})"
    )
    plt.xlabel("Decoherence Time ($t_c$ in s)")
    plt.ylabel("Teleportation Fidelity ($F$)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Strategie")
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.tight_layout()

    output_file = "fidelity_curve_plot.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot erfolgreich gespeichert als: {output_file}")
