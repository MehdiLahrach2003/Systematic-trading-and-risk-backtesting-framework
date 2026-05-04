# pricing/implied_vol.py
# ------------------------------------------------------------
# Ce fichier permet de calculer la volatilité implicite
# d'un call européen dans le modèle de Black-Scholes.
#
# Idée :
# on observe le prix de marché d'une option,
# puis on cherche quelle volatilité sigma il faut mettre
# dans Black-Scholes pour retrouver ce prix.
# ------------------------------------------------------------

from __future__ import annotations

import os
import sys
import numpy as np

# Permet d'importer le projet quand on lance ce fichier directement
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import du pricing Black-Scholes du call
from pricing.black_scholes import bs_call_price


# --------------------------------------------------------------
# Volatilité implicite par recherche de racine
# --------------------------------------------------------------
def implied_vol_bs(
    price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Calcule la volatilité implicite d'un call Black-Scholes.

    Paramètres
    ----------
    price : float
        Prix observé de l'option sur le marché.
    S : float
        Prix spot du sous-jacent.
    K : float
        Strike.
    r : float
        Taux sans risque.
    T : float
        Maturité en années.
    tol : float
        Tolérance numérique.
    max_iter : int
        Nombre maximal d'itérations.

    Retour
    ------
    float
        Volatilité implicite annualisée.
    """

    # Fonction objectif :
    # on veut trouver sigma tel que :
    # prix_BS(sigma) - prix_marché = 0
    def f(sig):
        return bs_call_price(S, K, r, sig, T) - price

    # Bornes de recherche pour sigma
    a, b = 1e-6, 5.0

    fa, fb = f(a), f(b)

    # Si le prix de marché est hors de la plage possible du modèle,
    # on ne peut pas trouver de volatilité implicite
    if fa * fb > 0:
        return np.nan

    # Boucle de recherche de racine
    for _ in range(max_iter):
        # Milieu de l'intervalle
        m = 0.5 * (a + b)
        fm = f(m)

        # Si on est suffisamment proche de 0, on s'arrête
        if abs(fm) < tol:
            return m

        # On garde le côté où le signe change
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    # Si pas de convergence
    return np.nan


# --------------------------------------------------------------
# Petit test si on exécute le fichier directement
# --------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Paramètres
    S0 = 100
    K = 100
    r = 0.00
    T = 1.0
    true_sigma = 0.20

    # On calcule un prix de call avec Black-Scholes
    price = bs_call_price(S0, K, r, true_sigma, T)

    # Puis on essaie de retrouver sigma à partir de ce prix
    iv = implied_vol_bs(price, S0, K, r, T) # type: ignore

    print("True σ:", true_sigma)
    print("Implied σ:", iv)

    # Petit graphe de vérification
    plt.figure(figsize=(5, 4))
    plt.title("Test de volatilité implicite")
    plt.axhline(true_sigma, color="orange", label="σ vraie")
    plt.scatter([0], [iv], color="blue", label="σ implicite")
    plt.legend()
    plt.tight_layout()
    plt.show()