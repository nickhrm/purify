import pandas as pd
import io

df = pd.read_csv("action_prob.csv")

# ---------------------------------------------------------
# VARIANTE A: Detaillierte Liste (Count + Prozent)
# ---------------------------------------------------------
counts = df.groupby(['model', 'coherence_time', 'action']).size().to_frame('count').reset_index()

# 1. Wir berechnen die Summe pro (Model + Coherence Time) Gruppe
# transform('sum') sorgt dafür, dass die Summe in jeder Zeile der Gruppe steht
group_totals = counts.groupby(['model', 'coherence_time'])['count'].transform('sum')

# 2. Berechnung des Anteils
counts['percentage'] = (counts['count'] / group_totals) * 100

print("--- Liste mit Prozenten ---")
print(counts)


# ---------------------------------------------------------
# VARIANTE B: Übersichtliche Tabelle (Nur Prozente)
# ---------------------------------------------------------
# pd.crosstab ist perfekt hierfür, da es 'normalize' eingebaut hat
pivot_percent = pd.crosstab(
    index=[df['model'], df['coherence_time']], 
    columns=df['action'], 
    normalize='index' # 'index' bedeutet: Zeilensumme = 100% (bzw. 1.0)
) * 100

print("\n--- Prozent-Tabelle (Crosstab) ---")
# Optional: Runden auf 2 Nachkommastellen für schönere Ausgabe
print(pivot_percent.round(2))