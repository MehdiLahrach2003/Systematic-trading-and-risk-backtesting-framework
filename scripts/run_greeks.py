# scripts/run_greeks.py

"""
Visualisation des Greeks Black–Scholes en fonction du prix du sous-jacent S.

Objectif :
→ comprendre comment le prix et les sensibilités changent quand S bouge
→ base du trading options + hedging
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Permet d'importer les fonctions Black-Scholes du projet
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pricing.black_scholes import (
    bs_call_price,
    bs_call_delta,
    bs_call_gamma,
    bs_call_vega,
    bs_call_theta,
)


def main():
    # --------------------------------------------------------
    # 1) Paramètres du modèle
    # --------------------------------------------------------
    S0 = 100.0     # spot de référence
    K = 100.0      # strike (ATM)
    r = 0.01       # taux sans risque
    sigma = 0.20   # volatilité
    T = 1.0        # maturité

    # --------------------------------------------------------
    # 2) Grille de prix du sous-jacent
    # --------------------------------------------------------
    # On fait varier S pour voir comment tout change
    spots = np.linspace(50, 150, 200)

    # --------------------------------------------------------
    # 3) Calcul du prix et des Greeks
    # --------------------------------------------------------
    call_prices = bs_call_price(spots, K, r, sigma, T)

    # Sensibilités :
    delta = bs_call_delta(spots, K, r, sigma, T)
    gamma = bs_call_gamma(spots, K, r, sigma, T)
    vega = bs_call_vega(spots, K, r, sigma, T)
    theta = bs_call_theta(spots, K, r, sigma, T)

    # --------------------------------------------------------
    # 4) Plot
    # --------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # ===== 1) Prix du call =====
    ax = axes[0]
    ax.plot(spots, call_prices, label="Call price")
    ax.axvline(K, color="grey", ls="--", label="Strike K")

    ax.set_ylabel("Price")
    ax.set_title("Black–Scholes call price and Greeks vs spot")

    ax.legend()
    ax.grid(alpha=0.3)

    # ===== 2) Delta =====
    ax = axes[1]
    ax.plot(spots, delta)
    ax.axvline(K, color="grey", ls="--")

    ax.set_ylabel("Delta")
    ax.grid(alpha=0.3)

    # ===== 3) Gamma =====
    ax = axes[2]
    ax.plot(spots, gamma)
    ax.axvline(K, color="grey", ls="--")

    ax.set_ylabel("Gamma")
    ax.grid(alpha=0.3)

    # ===== 4) Vega + Theta =====
    ax = axes[3]
    ax.plot(spots, vega, label="Vega")

    # Theta sur un axe secondaire (sinon échelles incompatibles)
    ax2 = ax.twinx()
    ax2.plot(spots, theta, color="tab:red", alpha=0.7, label="Theta")

    ax.set_xlabel("Spot S")
    ax.set_ylabel("Vega")
    ax2.set_ylabel("Theta")

    ax.grid(alpha=0.3)

    # Fusion des légendes
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()