# backtesting/risk_measures.py
# ------------------------------------------------------------
# Ce fichier contient des mesures de risque classiques
# appliquées à une série de rendements de stratégie.
#
# Il calcule :
# - la VaR historique
# - la VaR gaussienne
# - la CVaR (ou Expected Shortfall)
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# VaR historique
# ------------------------------------------------------------
def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calcule la Value-at-Risk historique au niveau alpha.

    Idée :
    on regarde directement la distribution empirique des rendements passés.

    Exemple :
    alpha = 0.05 correspond à la queue gauche à 5%.
    """

    # On enlève les valeurs manquantes
    r = returns.dropna()
    if len(r) == 0:
        return np.nan

    # La VaR est définie ici comme un nombre positif représentant une perte
    # On prend donc l'opposé du quantile gauche
    return float(-np.quantile(r, alpha))


# ------------------------------------------------------------
# VaR gaussienne (paramétrique)
# ------------------------------------------------------------
def var_gaussian(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calcule la VaR gaussienne.

    Hypothèse :
    les rendements sont iid et suivent une loi normale.

    Formule :
    VaR = -(mu - z * sigma)
    où z est le quantile de la loi normale standard.
    """

    r = returns.dropna()
    if len(r) == 0:
        return np.nan

    # Moyenne et écart-type empiriques
    mu = r.mean()
    sigma = r.std()

    # Approximation numérique du quantile gaussien
    z = abs(np.quantile(np.random.standard_normal(1_000_000), alpha))

    return float(-(mu - z * sigma))


# ------------------------------------------------------------
# CVaR historique / Expected Shortfall
# ------------------------------------------------------------
def cvar_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calcule la CVaR historique.

    Idée :
    on prend la moyenne des pertes situées au-delà du seuil de VaR.

    Donc :
    - la VaR donne un seuil
    - la CVaR donne la perte moyenne dans les pires cas
    """

    r = returns.dropna()
    if len(r) == 0:
        return np.nan

    # Seuil de la queue gauche
    threshold = np.quantile(r, alpha)

    # Rendements situés en dessous de ce seuil
    tail = r[r < threshold]

    if len(tail) == 0:
        return np.nan

    # On renvoie un nombre positif représentant une perte
    return float(-tail.mean())


# ------------------------------------------------------------
# Fonction pratique : rapport de risque complet
# ------------------------------------------------------------
def compute_risk_measures(returns: pd.Series, alpha: float = 0.05) -> dict:
    """
    Calcule plusieurs mesures de risque à la fois.

    Sortie :
    - VaR historique
    - VaR gaussienne
    - CVaR historique
    """
    return {
        "VaR_hist": var_historical(returns, alpha),
        "VaR_gauss": var_gaussian(returns, alpha),
        "CVaR": cvar_historical(returns, alpha),
    }