import csv
from pathlib import Path
import string


def write_results_csv(fidelity, time, file_name: string) -> None:
    """
    Hängt einen Datensatz (fidelity, time) an result.csv an.
    Schreibt den Header nur einmal, wenn die Datei neu ist/leer ist.
    """
    path = Path(f"results/{file_name}.csv")
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
