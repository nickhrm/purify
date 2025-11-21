from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from purify.my_constants import LAMBDA_1, LAMBDA_2, LAMBDA_3

def create_waiting_time_plot():
    """
    Erstellt ein Diagramm, das die Waiting Time gegen die Decoherence Time aufträgt.
    """
    # Dateiname dynamisch generieren
    file_name = f"results/ALL_RESULTS_{LAMBDA_1}_{LAMBDA_2}_{LAMBDA_3}.csv"
    results_file = Path(file_name)

    if not results_file.exists():
        print(f"Fehler: Die Datei '{results_file}' wurde nicht gefunden.")
        return

    try:
        df = pd.read_csv(results_file)

        # ---------------------------------------------------------
        # ÄNDERUNG 1: Wir aggregieren jetzt die Spalte "time"   
        # statt "fidelity".
        # ---------------------------------------------------------
        plot_df = (
            df.groupby(["strategy", "decoherence_time"])["time"]
            .mean()
            .reset_index()
        )
        
        # Optional: Werte in der Konsole ausgeben zur Kontrolle
        print("Durchschnittliche Wartezeiten pro Strategie:")
        print(plot_df.groupby("strategy")["time"].mean())

    except Exception as e:
        print(f"Fehler beim Lesen oder Verarbeiten der Datei: {e}")
        return

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    grouped_data = plot_df.sort_values(by="decoherence_time").groupby("strategy")

    for name, group in grouped_data:
        plt.plot(
            group["decoherence_time"],
            group["time"],  # <--- ÄNDERUNG 2: Hier 'time' plotten
            marker="o",
            linestyle="-",
            label=name,
        )

    # --- ÄNDERUNG 3: Beschriftungen und Limits anpassen ---
    plt.title(f"Waiting Time Analysis ({LAMBDA_1}; {LAMBDA_2}; {LAMBDA_3})")
    plt.xlabel("Decoherence Time ($t_c$ in s)")
    plt.ylabel("Average Waiting Time ($t$ in s)") # <--- Y-Achse umbenannt
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Strategie")
    plt.xscale("log")
    
    # WICHTIG: Fidelity war zwischen 0.7 und 1.0. 
    # Zeit kann alles sein (0 bis unendlich). 
    # Daher entfernen wir ylim oder setzen es auf 'bottom=0'.
    plt.ylim(bottom=0) 
    # Falls du eine logarithmische Y-Achse brauchst (wenn Zeiten stark schwanken):
    # plt.yscale("log") 

    plt.tight_layout()

    # Neuer Dateiname für den Plot
    output_file = f"waiting_time_plot_{str(LAMBDA_1).replace('.', '')}_{str(LAMBDA_2).replace('.', '')}_{str(LAMBDA_3).replace('.', '')}.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot erfolgreich gespeichert als: {output_file}")