import pandas as pd
import numpy as np



def plot_more(): 
    
    try:
        # Lade die CSV-Datei
        # Passe 'results.csv' an, falls deine Datei anders heißt
        file_path = 'results/ALL_RESULTS.csv'  
        df = pd.read_csv(file_path)
    
        # Definiere die zwei eigenen Aggregationsfunktionen,
        # die wir auf die 'fidelity'-Spalte anwenden wollen.
    
        def avg_no_zeros(x):
            """Berechnet den Durchschnitt einer Serie, ohne 0.0-Werte zu berücksichtigen."""
            # Filtere alle Werte, die nicht 0.0 sind
            non_zeros = x[x != 0.0]
            # Gebe den Durchschnitt zurück, oder np.nan, wenn keine Nicht-Null-Werte vorhanden sind
            return non_zeros.mean() if not non_zeros.empty else np.nan
    
        def count_zeros(x):
            """Zählt die Anzahl der 0.0-Werte in einer Serie."""
            # Summiere alle Vorkommen, bei denen der Wert 0.0 ist
            return (x == 0.0).sum()
    
        # 1. Gruppiere die Daten nach 'protocol_name' (Strategie) und 'decoherence_time'
        grouped_data = df.groupby(['protocol_name', 'decoherence_time'])
    
        # 2. Wende die Aggregationen auf die 'fidelity'-Spalte an
        #    .agg() ermöglicht es uns, mehrere Aggregationen auf einmal durchzuführen
        results = grouped_data['fidelity'].agg(
            average_fidelity_no_zeros=avg_no_zeros,
            zero_fidelity_count=count_zeros
        ).reset_index() # .reset_index() macht die Gruppen-Schlüssel wieder zu Spalten
    
        # 3. Stelle sicher, dass die Zählung als Integer (Ganzzahl) angezeigt wird
        results['zero_fidelity_count'] = results['zero_fidelity_count'].astype(int)
    
        # 4. Zeige die Ergebnisse an
        print("Analyse der Fidelity-Daten:")
        print(results)
    
    except FileNotFoundError:
        print(f"Fehler: Die Datei '{file_path}' wurde nicht gefunden.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")