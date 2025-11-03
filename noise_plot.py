import numpy as np
import matplotlib.pyplot as plt

#Plots the Protocol 2 and Protocol 3 of the 2-1 bilocal clifford protocols, 
# given that the newly generated link has noise that is in one parameter

# Parameter
k = 1  # beliebig änderbar

# Funktionen definieren
def q(x):
    return k * (1 - 0.8)

def w(x):
    return (1 - k / 2) * (1 - 0.8)

def h(x):
    numerator = (6 * w(x) + 4 * q(x) - 3) * x - q(x)
    denominator = (8 * w(x) - 2) * x - 2 * w(x) - 1
    return numerator / denominator

def r(x):
    numerator = (7 * w(x) + 3 * q(x) - 3) * x - w(x)
    denominator = (4 * w(x) + 4 * q(x) - 2) * x - w(x) - 1
    return numerator / denominator

# Definitionsbereich
x = np.linspace(0, 1.5, 400)

# Funktionen auswerten
h_vals = h(x)
r_vals = r(x)
    
# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, h_vals, label="h(x)", color='green')
plt.plot(x, r_vals, label="r(x)", color='blue')
plt.title(f"Funktionen h(x) und r(x) für k = {k}")
plt.xlabel("x")
plt.ylabel("Funktionswert")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("noiseplot.png")
