import csv
from pathlib import Path

from purify import ConstantsTuple


def write_results_csv(
    fidelity: float,
    time: float,
    my_constants: ConstantsTuple
) -> None:
    """
    Hängt einen Datensatz an die zentrale ALL_RESULTS.csv an.
    Die Spalten werden automatisch aus der ConstantsTuple und den Ergebniswerten
    erstellt.
    """
    # 1. Zentraler Dateiname
    CENTRAL_FILE_NAME = "ALL_RESULTS_TUPLE.csv"
    path = Path(f"results/{CENTRAL_FILE_NAME}")

    # 2. Erstelle den Ordner, falls er nicht existiert
    path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Prüfe Existenz/Größe der Datei, um den Header zu steuern
    file_exists = path.exists()
    file_empty = (path.stat().st_size == 0) if file_exists else True

    # 4. Erstelle den Header automatisch
    # Basisfelder
    RESULT_FIELDS = ["fidelity", "time"]
    # Felder aus dem NamedTuple (z.B. 'strategy', 'decoherence_time')
    CONSTANTS_FIELDS = list(ConstantsTuple._fields)

    # Der vollständige Header
    HEADER = RESULT_FIELDS + CONSTANTS_FIELDS

    # Append-Modus: "a"
    try:
        with path.open(mode="a", newline="", encoding="utf-8") as f:
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

            # 5. Datensatz als Dictionary erstellen
            # Starte mit den Ergebnis-Werten
            row_data = {
                "fidelity": fidelity,
                "time": time,
            }
            # Füge die NamedTuple-Werte als Dictionary hinzu
            # my_constants._asdict() konvertiert das NamedTuple in ein Dictionary
            row_data.update(my_constants._asdict())

            # 6. Datensatz anhängen
            writer.writerow(row_data)

    except Exception as e:
        print(f"Fehler beim Schreiben der Ergebnisse: {e}")

