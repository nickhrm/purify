import csv
from pathlib import Path
from typing import Union # Für die Typisierung von Zahlen

def write_results_csv(
    fidelity: Union[float, int], 
    time: Union[float, int], 
    protocol_name: str, 
    decoherence_time: Union[float, int]
) -> None:
    """
    Hängt einen Datensatz an die zentrale ALL_RESULTS.csv an.
    Fügt die Spalten für das Protokoll und die Dekohärenzzeit hinzu.
    Schreibt den Header nur einmal.
    """
    # 1. Zentraler Dateiname für alle Ergebnisse
    CENTRAL_FILE_NAME = "ALL_RESULTS.csv"
    path = Path(f"results/{CENTRAL_FILE_NAME}")

    # 2. WICHTIG: Erstelle den Ordner, falls er nicht existiert
    path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Prüfe Existenz/Größe der Datei
    file_exists = path.exists()
    # Stat: Die Datei existiert, und ihre Größe ist 0 (leer)
    file_empty = (path.stat().st_size == 0) if file_exists else True
    
    # Der vollständige Header
    HEADER = ["fidelity", "time", "protocol_name", "decoherence_time"]

    # Append-Modus: "a"
    try:
        with path.open(mode="a", newline="", encoding="utf-8") as f:
            # Hier verwenden wir den DictWriter, da er mit Headern besser umgeht.
            writer = csv.DictWriter(
                f, 
                fieldnames=HEADER, 
                delimiter=",", 
                quotechar='"', 
                quoting=csv.QUOTE_MINIMAL
            )

            # Header nur einmal schreiben
            if file_empty:
                writer.writeheader()

            # Datensatz als Dictionary anhängen (mit den neuen Spalten)
            writer.writerow({
                "fidelity": fidelity, 
                "time": time, 
                "protocol_name": protocol_name,
                "decoherence_time": decoherence_time
            })
    except Exception as e:
        print(f"Fehler beim Schreiben der Ergebnisse: {e}")

# Beispiel für die zukünftige Verwendung in Ihrer Simulationsschleife:
# Angenommen, Sie haben diese Werte in einem Simulationsschritt berechnet
# current_fidelity = 0.82
# current_time = 15
# current_protocol = "Protokoll 1"
# current_deco_time = 0.5
#
# write_results_csv(current_fidelity, current_time, current_protocol, current_deco_time)