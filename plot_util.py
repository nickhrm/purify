import pandas as pd
import matplotlib.pyplot as plt

# CSV-Datei einlesen
df = pd.read_csv("result_always_replace.csv")   # <-- Pfad zu deiner Datei anpassen

# Kontrollausgabe (optional)

# Boxplot der Fidelity-Werte
plt.boxplot(df["fidelity"], vert=True, patch_artist=True)
plt.title("Verteilung der Fidelity-Werte")
plt.ylabel("Fidelity")
plt.grid(True, linestyle="--", alpha=0.6)

# Statt plt.show() (was in Headless-Umgebungen oft nicht funktioniert):
plt.savefig("fidelity_boxplot.png", dpi=300)
print("✅ Plot gespeichert als 'fidelity_boxplot_always_replace.png'")
