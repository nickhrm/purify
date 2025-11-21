import pandas as pd
from pathlib import Path

def calculate_average_waiting_times():
    # Pfad zum Ordner mit den Ergebnissen
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("Ordner 'results' nicht gefunden.")
        return

    summary_data = []

    # Suche alle CSV-Dateien, die mit ALL_RESULTS beginnen
    files = list(results_dir.glob("ALL_RESULTS_*.csv"))
    
    print(f"{len(files)} Dateien gefunden. Verarbeite...\n")

    for file_path in files:
        try:
            # 1. Lambda-Werte aus dem Dateinamen extrahieren
            # Dateiname ist z.B. "ALL_RESULTS_00_02_00.csv" oder "ALL_RESULTS_0.2_0.0_0.0.csv"
            parts = file_path.stem.split('_') 
            
            # Wir erwarten, dass die letzten 3 Teile die Lambda-Werte sind
            # (Index 2, 3, 4 nach 'ALL' und 'RESULTS')
            if len(parts) >= 5:
                l1 = float(parts[2])
                l2 = float(parts[3])
                l3 = float(parts[4])
            else:
                print(f"Warnung: Konnte Lambda-Werte aus '{file_path.name}' nicht lesen.")
                continue

            # 2. Datei einlesen
            df = pd.read_csv(file_path)

            # 3. Durchschnittliche Time pro Strategy berechnen
            # Wir gruppieren nur nach Strategy (über alle Decoherence Times hinweg)
            avg_times = df.groupby("strategy")["time"].mean().reset_index()

            # 4. Lambda-Werte hinzufügen
            avg_times["Lambda_1"] = l1
            avg_times["Lambda_2"] = l2
            avg_times["Lambda_3"] = l3

            summary_data.append(avg_times)

        except Exception as e:
            print(f"Fehler bei Datei {file_path.name}: {e}")

    # Alles zusammenfügen
    if summary_data:
        final_df = pd.concat(summary_data, ignore_index=True)
        
        # Spalten sortieren für bessere Lesbarkeit
        final_df = final_df[["Lambda_1", "Lambda_2", "Lambda_3", "strategy", "time"]]
        
        # Sortieren: Erst nach Lambdas, dann nach Strategie
        final_df = final_df.sort_values(by=["Lambda_1", "Lambda_2", "Lambda_3", "strategy"])
        
        # Ausgabe in der Konsole
        print("-" * 60)
        print("DURCHSCHNITTLICHE WAITING TIME (pro Tuple & Strategy)")
        print("-" * 60)
        print(final_df.to_string(index=False))
        
        # Optional: Als CSV speichern
        output_filename = "waiting_time_summary.csv"
        final_df.to_csv(output_filename, index=False)
        print(f"\nZusammenfassung gespeichert als: {output_filename}")
        
    else:
        print("Keine Daten konnten verarbeitet werden.")

if __name__ == "__main__":
    calculate_average_waiting_times()