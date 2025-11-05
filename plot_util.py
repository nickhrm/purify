import pandas as pd
import matplotlib.pyplot as plt
import os


def create_boxplot(): 
    # Ordner mit Ergebnissen
    results_dir = "results"

    # Alle CSV-Dateien im Ordner finden
    files = [f for f in os.listdir(results_dir) if f.endswith(".csv")]

    if not files:
        raise FileNotFoundError(f"Keine CSV-Dateien in '{results_dir}' gefunden!")

    datasets = []
    labels = []

    # Alle Dateien einlesen
    for file in files:
        path = os.path.join(results_dir, file)
        df = pd.read_csv(path)

        if "fidelity" not in df.columns:
            raise ValueError(f"Spalte 'fidelity' fehlt in Datei: {file}")

        datasets.append(df["fidelity"])
        labels.append(os.path.splitext(file)[0])  # Dateiname ohne .csv

    # Boxplot erzeugen
    plt.figure(figsize=(12, 6))
    plt.boxplot(datasets, labels=labels, vert=True)

    plt.title("Fidelity-Verteilung pro Strategie")
    plt.ylabel("Fidelity")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Speichern statt anzeigen
    output_file = "fidelity_boxplot.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)

    print(f"✅ Plot gespeichert als '{output_file}'")
    print(f"📁 Verarbeitete Dateien: {files}")
