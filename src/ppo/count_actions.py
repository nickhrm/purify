import pandas as pd

df = pd.read_csv("action_prob.csv")

# ---------------------------------------------------------
# VARIANTE A: Detaillierte Liste (Direkt aus CSV)
# ---------------------------------------------------------
# Da die Datei nun bereits aggregiert ist ("count" und "percentage"),
# können wir das DataFrame direkt ausgeben oder leicht sortieren.
print("--- Liste mit Prozenten ---")
print(df[["model", "coherence_time", "action", "count", "percentage"]])


# ---------------------------------------------------------
# VARIANTE B: Übersichtliche Tabelle (Pivotieren der Prozente)
# ---------------------------------------------------------
# Wir nutzen pivot_table statt crosstab, da wir die Werte bereits haben (percentage).
pivot_percent = df.pivot_table(
    index=["model", "coherence_time"],
    columns="action",
    values="percentage",
    fill_value=0,  # Fehlende Actions mit 0% auffüllen
)

print("\n--- Prozent-Tabelle (Pivot) ---")
print(pivot_percent.round(2))
