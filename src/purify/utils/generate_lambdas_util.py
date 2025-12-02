import numpy.random as random


def generate_y_z(x, w):
    # 1. Prüfen, ob w gültig ist
    if not (0 < w < 1):
        raise ValueError("w muss zwischen 0 und 1 liegen.")
    
    target_sum = 1 - w  # Das ist (x + y + z)
    
    # 2. Die geforderte Summe für y und z berechnen
    S = target_sum - x
    
    # 3. Prüfen, ob x die Bedingung x > (y+z) erfüllen kann
    # Das bedeutet: x muss größer sein als S.
    if x <= S:
        raise ValueError(f"Nicht lösbar: x ({x}) ist nicht groß genug. x muss > {S} sein.")
    
    # Optional: Wenn y und z positiv sein müssen, muss S positiv sein.
    # Das bedeutet x muss kleiner als target_sum sein.
    if S <= 0:
        raise ValueError("Nicht lösbar für positive y, z: x ist zu groß.")

    # 4. y und z generieren
    # Wir teilen S einfach zufällig auf y und z auf.
    y = random.uniform(0.0001, S - 0.0001) # Kleiner Puffer, um 0 zu vermeiden
    z = S - y
    
    return y, z