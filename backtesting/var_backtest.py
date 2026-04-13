# backtesting/var_backtest.py
# ------------------------------------------------------------
# Ce fichier sert à tester la qualité d'une VaR.
#
# Il contient :
# - le comptage des exceptions de VaR
# - le test de Kupiec
#
# Idée :
# si une VaR à 5% est correcte, alors environ 5% des jours
# doivent dépasser cette VaR.
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd
import math


# ------------------------------------------------------------
# 1) Comptage des exceptions de VaR
# ------------------------------------------------------------
def count_exceptions(returns: pd.Series, var_series: pd.Series) -> int:
    """
    Compte le nombre de violations de VaR.

    Une violation a lieu si :
        rendement < -VaR

    Paramètres
    ----------
    returns : pd.Series
        Rendements journaliers de la stratégie.
    var_series : pd.Series
        Série temporelle de VaR, exprimée comme une perte positive.

    Retour
    ------
    int
        Nombre d'exceptions observées.
    """

    # On enlève les NaN des rendements
    r = returns.dropna()

    # On aligne la série de VaR sur les mêmes dates
    v = var_series.reindex(r.index).dropna()

    # Si les tailles ne sont pas exactement les mêmes,
    # on garde la partie commune finale
    if len(v) != len(r):
        min_len = min(len(r), len(v))
        r = r.iloc[-min_len:]
        v = v.iloc[-min_len:]

    # Exception = rendement plus mauvais que -VaR
    breaches = r < -v

    return int(breaches.sum())


# ------------------------------------------------------------
# 2) Test de Kupiec
# ------------------------------------------------------------
def kupiec_test(returns: pd.Series, var_series: pd.Series, alpha: float = 0.05) -> dict:
    """
    Test de Kupiec (1995), dit de couverture inconditionnelle.

    Hypothèse nulle :
        la vraie probabilité d'exception est égale à alpha.

    Paramètres
    ----------
    returns : pd.Series
        Rendements de la stratégie.
    var_series : pd.Series
        Série de VaR.
    alpha : float
        Niveau théorique de la VaR, par exemple 0.05.

    Retour
    ------
    dict
        Contient :
        - N : nombre d'exceptions
        - T : nombre total d'observations
        - p_hat : fréquence observée des exceptions
        - LRuc : statistique du test
        - p_value : p-value du test
    """

    r = returns.dropna()
    v = var_series.reindex(r.index).dropna()

    # Alignement défensif
    if len(v) != len(r):
        min_len = min(len(r), len(v))
        r = r.iloc[-min_len:]
        v = v.iloc[-min_len:]

    # Détection des exceptions
    breaches = r < -v
    N = int(breaches.sum())   # nombre d'exceptions
    T = len(r)                # nombre total d'observations

    if T == 0:
        return {}

    # Fréquence observée des exceptions
    p_hat = N / T

    # Cas extrêmes pour éviter log(0)
    if p_hat in (0.0, 1.0):
        return {
            "N": N,
            "T": T,
            "p_hat": p_hat,
            "LRuc": float("inf"),
            "p_value": 0.0,
        }

    # Statistique du test de Kupiec
    num = ((1 - alpha) ** (T - N)) * (alpha ** N)
    den = ((1 - p_hat) ** (T - N)) * (p_hat ** N)
    LRuc = -2 * math.log(num / den)

    # Sous H0, LRuc suit approximativement une loi du chi2 à 1 ddl
    p_value = 1 - chi2_cdf(LRuc, df=1)

    return {
        "N": N,
        "T": T,
        "p_hat": p_hat,
        "LRuc": LRuc,
        "p_value": p_value,
    }


# ------------------------------------------------------------
# 3) Fonction technique : CDF du chi2
# ------------------------------------------------------------
def chi2_cdf(x: float, df: int = 1) -> float:
    """
    Fonction de répartition de la loi du chi2.

    Ici, seule la version à 1 degré de liberté est implémentée,
    car c'est le cas utile pour le test de Kupiec.
    """
    if df != 1:
        raise NotImplementedError("Only df=1 implemented for Kupiec test.")

    return math.erf(math.sqrt(x / 2))