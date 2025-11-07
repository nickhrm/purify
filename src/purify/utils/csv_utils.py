import csv
from pathlib import Path
# 'string' ist das Modul, 'str' ist der Typ
# import string # <- Wird nicht gebraucht

def write_results_csv(fidelity, time, file_name: str) -> None: # <- Typ ist 'str'
    """
    Hängt einen Datensatz (fidelity, time) an result.csv an.
    Schreibt den Header nur einmal, wenn die Datei neu ist/leer ist.
    """
    # 1. Relativer Pfad: KEIN Schrägstrich am Anfang
    path = Path(f"results/{file_name}.csv")

    # 2. WICHTIG: Erstelle den Ordner, falls er nicht existiert
    #    path.parent ist der "results"-Teil
    path.parent.mkdir(parents=True, exist_ok=True)

    # 3. Prüfe Existenz/Größe der Datei
    file_exists = path.exists()
    file_empty = (path.stat().st_size == 0) if file_exists else True

    # Append-Modus: "a" statt "w"
    with path.open(mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Header nur einmal
        if file_empty:
            writer.writerow(["fidelity", "time"])

        # Datensatz anhängen
        writer.writerow([fidelity, time])