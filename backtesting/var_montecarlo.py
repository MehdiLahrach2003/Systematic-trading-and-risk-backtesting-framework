# backtesting/var_montecarlo.py
# ------------------------------------------------------------
# Ce fichier calcule la VaR et la CVaR par simulation Monte Carlo.
#
# Deux approches :
# - Monte Carlo paramétrique (hypothèse gaussienne)
# - Monte Carlo bootstrap (rééchantillonnage historique)
# ------------------------------------------------------------

from __future__ import annotations
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1) Monte Carlo paramétrique
# ------------------------------------------------------------
def mc_var_parametric(
    returns: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 20_000
) -> tuple[float, float]:
    """
    Calcule la VaR et la CVaR par simulation Monte Carlo paramétrique.

    Hypothèse :
    les rendements suivent une loi normale N(mu, sigma^2).

    Paramètres
    ----------
    returns : pd.Series
        Rendements historiques.
    alpha : float
        Niveau de queue (ex : 0.05).
    n_sims : int
        Nombre de simulations Monte Carlo.

    Retour
    ------
    (VaR, CVaR)
    """

    r = returns.dropna().astype(float)
    if len(r) == 0:
        return np.nan, np.nan

    # Estimation des paramètres de la loi normale
    mu = r.mean()
    sigma = r.std()

    # Simulation de nombreux rendements gaussiens
    sims = np.random.normal(mu, sigma, size=n_sims)
    sims = np.sort(sims)

    # VaR = opposé du quantile alpha
    var_mc = -np.percentile(sims, alpha * 100)

    # CVaR = moyenne des pires alpha% rendements
    cutoff = int(alpha * n_sims)
    cvar_mc = -sims[:cutoff].mean()

    return float(var_mc), float(cvar_mc)


# ------------------------------------------------------------
# 2) Monte Carlo bootstrap
# ------------------------------------------------------------
def mc_var_bootstrap(
    returns: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 20_000
) -> tuple[float, float]:
    """
    Calcule la VaR et la CVaR par bootstrap Monte Carlo.

    Idée :
    on tire au hasard, avec remise, parmi les rendements historiques.

    Paramètres
    ----------
    returns : pd.Series
        Rendements historiques.
    alpha : float
        Niveau de queue (ex : 0.05).
    n_sims : int
        Nombre de simulations.

    Retour
    ------
    (VaR, CVaR)
    """

    r = returns.dropna().astype(float)
    if len(r) == 0:
        return np.nan, np.nan

    # Rééchantillonnage avec remise
    sims = np.random.choice(r, size=n_sims, replace=True)
    sims = np.sort(sims)

    # VaR = opposé du quantile alpha
    var_mc = -np.percentile(sims, alpha * 100)

    # CVaR = moyenne des pires alpha% scénarios
    cutoff = int(alpha * n_sims)
    cvar_mc = -sims[:cutoff].mean()

    return float(var_mc), float(cvar_mc)