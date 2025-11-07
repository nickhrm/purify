import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re # Für das Parsen der Dateinamen

def create_decoherence_plot():
    """
    Erstellt ein Diagramm, das die Fidelity gegen die Decoherence Time aufträgt,
    gruppiert nach Protokoll (Strategie).
    """
    results_dir = Path("results")
    
    # Sicherstellen, dass der Ordner existiert
    if not results_dir.exists():
        print(f"Fehler: Der Ordner '{results_dir}' wurde nicht gefunden.")
        return

    # Sammeln aller relevanten Daten in einer Liste von Dictionaries
    all_data = []

    # Regex, um Protokollnummer (P), Decoherence Time (T) und den Dateinamen zu parsen
    # Beispiel: ALWAYS_PROT_1_0.1.csv
    # Gruppen: 1=Protokollnummer, 2=Decoherence Time (optional)
    # Behandelt sowohl Dateien mit als auch ohne Zeit (z.B. ALWAYS_PROT_1.csv)
    filename_pattern = re.compile(r"ALWAYS_PROT_(\d+)_?([\d\.]*)\.csv")
    
    # Iteriere durch alle CSV-Dateien im results-Ordner
    for file_path in results_dir.glob("ALWAYS_PROT_*.csv"):
        match = filename_pattern.match(file_path.name)
        
        if match:
            protocol_num = int(match.group(1))
            deco_time_str = match.group(2)
            
            # WICHTIG: Die Standard-Decoherence Time für Dateien ohne Zahl (z.B. ALWAYS_PROT_1.csv) 
            # wird hier als 1.0 (oder der Standardwert, den Sie verwenden) angenommen.
            if deco_time_str == "":
                deco_time = 1.0 
            else:
                try:
                    deco_time = float(deco_time_str)
                except ValueError:
                    # Überspringen, falls das Parsen fehlschlägt
                    continue 

            try:
                # pandas zum Einlesen der CSV
                df = pd.read_csv(file_path)
                
                # Angenommen, wir betrachten die *letzte* erreichte Fidelity im Simulationslauf
                # (Oder den Durchschnitt, je nach Bedarf. Hier nehmen wir den letzten Wert)
                final_fidelity = df['fidelity'].mean()
                
                all_data.append({
                    'protocol': protocol_num,
                    'decoherence_time': deco_time,
                    'fidelity': final_fidelity
                })
            except Exception as e:
                print(f"Fehler beim Lesen der Datei {file_path.name}: {e}")
                continue
    
    if not all_data:
        print("Keine gültigen Daten gefunden, um das Diagramm zu erstellen.")
        return

    # DataFrame aus den gesammelten Daten erstellen
    plot_df = pd.DataFrame(all_data)
    
    # Gruppieren nach Protokoll und nach Decoherence Time sortieren
    grouped_data = plot_df.sort_values(by='decoherence_time').groupby('protocol')

    # --- Plotting ---
    
    plt.figure(figsize=(10, 6))
    
    for name, group in grouped_data:
        # Erzeuge eine Linie für jedes Protokoll
        plt.plot(
            group['decoherence_time'], 
            group['fidelity'], 
            marker='o', 
            linestyle='-', 
            label=f'Protokoll {name}'
        )

    # Diagramm beschriften
    plt.title('Teleportation fidelity for different for different decoerence times and teleportation protocols')
    plt.xlabel('Decoherence Time ($t_c$ in ms)')
    plt.ylabel('Teleportation Fidelity ($F$)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Strategie')
    plt.xscale('log') # Oft ist eine logarithmische Skala für t_c sinnvoller
    plt.tight_layout()

    output_file = "fidelity_curve_plot.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)


if __name__ == "__main__":
    create_decoherence_plot()