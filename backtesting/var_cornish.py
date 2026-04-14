# backtesting/var_cornish.py
# ------------------------------------------------------------
# Ce fichier calcule une VaR de Cornish-Fisher.
#
# Idée :
# la VaR gaussienne suppose que les rendements suivent une loi normale.
# Mais en finance, ce n'est souvent pas vrai :
# - asymétrie possible
# - queues épaisses
#
# La correction de Cornish-Fisher ajuste donc le quantile gaussien
# à l'aide de la skewness et de la kurtosis.
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
import math


def _compute_moments(returns: pd.Series) -> tuple[float, float, float]:
    """
    Calcule :
    - la moyenne
    - la skewness
    - la kurtosis excédentaire
    """

    r = returns.dropna().astype(float)
    if len(r) == 0:
        return np.nan, np.nan, np.nan

    mu = r.mean()
    sigma = r.std()

    # Si la volatilité est nulle ou s'il n'y a pas assez de données,
    # on ne peut pas calculer proprement les moments supérieurs
    if sigma == 0 or len(r) < 4:
        return mu, 0.0, 0.0

    # Standardisation des rendements
    z = (r - mu) / sigma

    # Skewness empirique
    skew = float((z**3).mean())

    # Kurtosis excédentaire empirique
    kurt_excess = float((z**4).mean() - 3.0)

    return mu, skew, kurt_excess


def cornish_fisher_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Calcule la VaR corrigée par Cornish-Fisher.

    Paramètres
    ----------
    returns : pd.Series
        Rendements simples de la stratégie.
    alpha : float
        Niveau de queue (ex : 0.05).

    Retour
    ------
    float
        VaR Cornish-Fisher (positive = perte)
    """

    r = returns.dropna().astype(float)
    if len(r) == 0:
        return np.nan

    mu = r.mean()
    sigma = r.std()

    # Quantile gaussien standard
    z = float(np.quantile(np.random.standard_normal(500_000), alpha))

    # Moments supérieurs
    _, skew, kurt_excess = _compute_moments(r)

    # Quantile corrigé de Cornish-Fisher
    z_cf = (
        z
        + (1/6) * (z*z - 1) * skew
        + (1/24) * (z**3 - 3*z) * kurt_excess
        - (1/36) * (2*z**3 - 5*z) * skew * skew
    )

    # VaR = opposé du quantile corrigé
    return float(-(mu + sigma * z_cf))


def compute_cornish_report(returns: pd.Series, alpha: float = 0.05) -> dict:
    """
    Compare :
    - la VaR gaussienne
    - la VaR Cornish-Fisher

    et renvoie aussi l'ajustement relatif en pourcentage.
    """

    r = returns.dropna()
    if len(r) == 0:
        return {}

    # VaR gaussienne
    mu = r.mean()
    sigma = r.std()
    z = float(np.quantile(np.random.standard_normal(500_000), alpha))
    var_gauss = -(mu + sigma * z)

    # VaR Cornish-Fisher
    var_cf = cornish_fisher_var(r, alpha)

    return {
        "VaR_gaussian": var_gauss,
        "VaR_cornish": var_cf,
        "CF_adjustment_%": 100 * (var_cf - var_gauss) / abs(var_gauss) if var_gauss != 0 else np.nan,
    }