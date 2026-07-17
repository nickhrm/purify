import numpy as np


def generate_y_z(x: float, w: float, rng: np.random.Generator) -> tuple[float, float]:
    """Teilt die Fehlermasse (1 - w - x) zufällig auf y und z auf,
    unter der Bedingung x > y + z."""
    if not (0 < w < 1):
        raise ValueError("w muss zwischen 0 und 1 liegen.")

    target_sum = 1 - w  # Das ist (x + y + z)
    s = target_sum - x  # geforderte Summe für y und z

    # x muss die Bedingung x > (y + z) erfüllen können
    if x <= s:
        raise ValueError(f"Nicht lösbar: x ({x}) ist nicht groß genug. x muss > {s} sein.")
    if s <= 0:
        raise ValueError("Nicht lösbar für positive y, z: x ist zu groß.")

    y = rng.uniform(0.0001, s - 0.0001)  # Kleiner Puffer, um 0 zu vermeiden
    z = s - y
    return y, z
