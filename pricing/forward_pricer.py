from math import *
import numpy as np
import matplotlib.pyplot as plt


def forward_price(S0, r, T):
    """
    Calcule le prix théorique d'un forward.

    Paramètres
    ----------
    S0 : prix spot aujourd’hui
    r : taux sans risque
    T : maturité

    Retour
    ------
    Prix forward F0
    """
    return S0 * exp(r * T)


def plot_payoff(K, S_min, S_max, n):
    """
    Trace le payoff d’un forward long.

    Paramètres
    ----------
    K : prix fixé dans le contrat
    S_min, S_max : plage des prix à maturité
    n : nombre de points
    """
    S_T = np.linspace(S_min, S_max, n)

    # Payoff forward long
    payoff = S_T - K

    plt.figure(figsize=(7, 4))
    plt.plot(S_T, payoff, label="Payoff long forward")

    plt.axhline(0, color="black", lw=1)
    plt.axvline(K, color="gray", ls="--", label="Strike")

    plt.title("Payoff Forward Contract")
    plt.xlabel("Prix à maturité (S_T)")
    plt.ylabel("Profit / Perte")

    plt.legend()
    plt.tight_layout()
    plt.show()


# Paramètres
S_min = 50
S_max = 150
n = 200

S0, r, T, K = 100, 0.03, 1.0, 100

# Calcul du prix forward
F0 = forward_price(S0, r, T)

print(f"Prix théorique du forward : {F0:.2f}")

# Affichage du payoff
plot_payoff(K, S_min, S_max, n)