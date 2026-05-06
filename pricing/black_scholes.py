# pricing/black_scholes.py
# ------------------------------------------------------------
# Ce fichier implémente le modèle de Black-Scholes
# pour pricer des options européennes.
#
# Il calcule :
# - le prix d'un call
# - le prix d'un put
# - quelques Greeks de base :
#   delta, gamma, vega, theta
#
# Ce fichier est fondamental car beaucoup d'autres modules
# vont s'appuyer dessus (vol implicite, delta hedging, etc.).
# ------------------------------------------------------------

from __future__ import annotations

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ---------------------------------------------------------------------
# 1) Quantités intermédiaires d1 et d2
# ---------------------------------------------------------------------
def _d1_d2(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule les quantités d1 et d2 du modèle de Black-Scholes.

    Paramètres
    ----------
    S : prix spot du sous-jacent
    K : strike
    r : taux sans risque
    sigma : volatilité annualisée
    T : maturité (en années)

    Retour
    ------
    d1, d2
    """

    if sigma <= 0.0 or T <= 0.0:
        raise ValueError("sigma and T must be strictly positive.")

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)

    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("S and K must be strictly positive.")

    sqrtT = np.sqrt(T)

    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    return d1, d2


# ---------------------------------------------------------------------
# 2) Prix du call
# ---------------------------------------------------------------------
def bs_call_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Prix Black-Scholes d'un call européen.
    """

    # Cas limite : volatilité nulle ou maturité nulle
    # On retombe sur la valeur intrinsèque
    if sigma <= 0.0 or T <= 0.0:
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        return np.maximum(S_arr - K_arr, 0.0)

    d1, d2 = _d1_d2(S, K, r, sigma, T)

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ---------------------------------------------------------------------
# 3) Prix du put
# ---------------------------------------------------------------------
def bs_put_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Prix Black-Scholes d'un put européen.
    """

    if sigma <= 0.0 or T <= 0.0:
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        return np.maximum(K_arr - S_arr, 0.0)

    d1, d2 = _d1_d2(S, K, r, sigma, T)

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------
# 4) Greeks
# ---------------------------------------------------------------------
def bs_call_delta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Delta d'un call européen.

    Le delta mesure la sensibilité du prix de l'option
    au prix du sous-jacent.
    """
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return norm.cdf(d1)


def bs_call_gamma(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Gamma d'un call (et aussi d'un put).

    Le gamma mesure la sensibilité du delta.
    """
    d1, _ = _d1_d2(S, K, r, sigma, T)
    S = np.asarray(S, dtype=float)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_call_vega(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
    as_bp: bool = False,
) -> np.ndarray:
    """
    Vega d'un call (et aussi d'un put).

    Le vega mesure la sensibilité du prix
    à la volatilité.
    """
    d1, _ = _d1_d2(S, K, r, sigma, T)
    S = np.asarray(S, dtype=float)

    vega = S * norm.pdf(d1) * np.sqrt(T)

    # Option : vega par basis point de volatilité
    if as_bp:
        vega = vega / 10_000.0

    return vega


def bs_call_theta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Theta d'un call européen.

    Le theta mesure la sensibilité au temps.
    Ici il est exprimé par année.
    """
    d1, d2 = _d1_d2(S, K, r, sigma, T)

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)

    term1 = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)

    return term1 + term2


# ---------------------------------------------------------------------
# 5) Démo graphique
# ---------------------------------------------------------------------
def main() -> None:
    """
    Petite démo visuelle :
    on trace le prix du call et du put en fonction du spot.
    """
    K = 100.0
    r = 0.02
    sigma = 0.2
    T = 1.0

    S_grid = np.linspace(50, 150, 200)

    calls = bs_call_price(S_grid, K, r, sigma, T)
    puts = bs_put_price(S_grid, K, r, sigma, T)

    plt.figure(figsize=(8, 5))
    plt.plot(S_grid, calls, label="Call price")
    plt.plot(S_grid, puts, label="Put price")
    plt.axvline(K, color="grey", ls="--", lw=1, label="Strike K")

    plt.title("Black-Scholes : prix du call et du put")
    plt.xlabel("Spot S")
    plt.ylabel("Prix de l'option")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()