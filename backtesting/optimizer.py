# backtesting/optimizer.py
# ------------------------------------------------------------
# Ce fichier contient des outils d'optimisation de portefeuille
# de type Markowitz.
#
# Il permet de :
# - calculer les rendements moyens et covariances à partir
#   de plusieurs stratégies déjà backtestées
# - calculer les statistiques d'un portefeuille
# - trouver le portefeuille de variance minimale
# - trouver le portefeuille de Sharpe maximal
# - construire une frontière efficiente
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .engine import BacktestResult


@dataclass
class PortfolioStats:
    """
    Contient les statistiques d'un portefeuille.
    """
    weights: np.ndarray   # poids du portefeuille
    ret_ann: float        # rendement annuel
    vol_ann: float        # volatilité annuelle
    sharpe: float         # ratio de Sharpe


# ---------------------------------------------------------------------
# 1) Calcul des rendements moyens et covariance à partir des backtests
# ---------------------------------------------------------------------
def compute_mu_cov_from_results(
    results: Dict[str, BacktestResult],
    freq_per_year: int = 252,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calcule le vecteur de rendements moyens annualisés
    et la matrice de covariance annualisée à partir
    de plusieurs BacktestResult.
    """

    if not results:
        raise ValueError("`results` dict is empty.")

    # On rassemble les rendements journaliers des différentes stratégies
    ret_dict = {name: res.returns for name, res in results.items()}
    df_ret = pd.DataFrame(ret_dict).dropna(how="all")

    # Moyenne journalière et covariance journalière
    mu_daily = df_ret.mean()
    cov_daily = df_ret.cov()

    # Annualisation
    mu_ann = mu_daily * float(freq_per_year)
    cov_ann = cov_daily * float(freq_per_year)

    return mu_ann, cov_ann


# ---------------------------------------------------------------------
# 2) Statistiques d'un portefeuille pour des poids donnés
# ---------------------------------------------------------------------
def portfolio_stats(
    weights: np.ndarray,
    mu_ann: pd.Series,
    cov_ann: pd.DataFrame,
    rf: float = 0.0,
) -> PortfolioStats:
    """
    Calcule le rendement annuel, la volatilité annuelle
    et le Sharpe d'un portefeuille.
    """

    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be a 1D array.")

    # On aligne la covariance avec l'ordre des actifs
    assets = list(mu_ann.index)
    cov = cov_ann.loc[assets, assets].values

    # Normalisation défensive : somme des poids = 1
    if w.sum() != 0.0:
        w = w / w.sum()
    else:
        w = np.ones_like(w) / len(w)

    # Rendement moyen du portefeuille
    ret_ann = float(np.dot(w, mu_ann.values)) # type: ignore

    # Variance puis volatilité
    var_ann = float(w @ cov @ w)
    vol_ann = math.sqrt(var_ann) if var_ann > 0.0 else 0.0

    # Sharpe
    if vol_ann > 0.0:
        sharpe = (ret_ann - rf) / vol_ann
    else:
        sharpe = 0.0

    return PortfolioStats(weights=w, ret_ann=ret_ann, vol_ann=vol_ann, sharpe=sharpe)


# ---------------------------------------------------------------------
# 3) Contraintes standard : long-only et somme des poids = 1
# ---------------------------------------------------------------------
def _long_only_constraints(n_assets: int):
    """
    Construit les contraintes :
    - somme des poids = 1
    - 0 <= poids_i <= 1
    """
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_assets
    return cons, bounds


# ---------------------------------------------------------------------
# 4) Portefeuille de variance minimale
# ---------------------------------------------------------------------
def solve_min_var(mu_ann: pd.Series, cov_ann: pd.DataFrame) -> PortfolioStats:
    """
    Cherche le portefeuille long-only de variance minimale.
    """

    n = len(mu_ann)
    cov = cov_ann.values
    cons, bounds = _long_only_constraints(n)

    # Fonction objectif : variance
    def obj(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    # Point de départ : poids égaux
    x0 = np.ones(n) / n

    # Optimisation
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=[cons])

    if not res.success:
        raise RuntimeError(f"Min-variance optimisation failed: {res.message}")

    return portfolio_stats(res.x, mu_ann, cov_ann)


# ---------------------------------------------------------------------
# 5) Portefeuille de Sharpe maximal
# ---------------------------------------------------------------------
def solve_max_sharpe(
    mu_ann: pd.Series,
    cov_ann: pd.DataFrame,
    rf: float = 0.0,
) -> PortfolioStats:
    """
    Cherche le portefeuille long-only qui maximise le ratio de Sharpe.
    """

    n = len(mu_ann)
    cov = cov_ann.values
    cons, bounds = _long_only_constraints(n)

    mu_vec = mu_ann.values

    # On minimise l'opposé du Sharpe
    def obj(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        ret = float(np.dot(w, mu_vec)) # type: ignore
        var = float(w @ cov @ w)
        vol = math.sqrt(var) if var > 0.0 else 0.0

        if vol == 0.0:
            return 1e6

        sharpe = (ret - rf) / vol
        return -sharpe

    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=[cons])

    if not res.success:
        raise RuntimeError(f"Max-Sharpe optimisation failed: {res.message}")

    return portfolio_stats(res.x, mu_ann, cov_ann, rf=rf)


# ---------------------------------------------------------------------
# 6) Frontière efficiente
# ---------------------------------------------------------------------
def efficient_frontier(
    mu_ann: pd.Series,
    cov_ann: pd.DataFrame,
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construit une frontière efficiente long-only
    en balayant plusieurs rendements cibles.
    """

    n = len(mu_ann)
    assets = list(mu_ann.index)
    mu_vec = mu_ann.values
    cov = cov_ann.loc[assets, assets].values

    # Grille de rendements cibles
    mu_min = float(mu_vec.min()) # type: ignore
    mu_max = float(mu_vec.max()) # type: ignore
    target_grid = np.linspace(mu_min, mu_max, n_points)

    cons_sum, bounds = _long_only_constraints(n)

    vols = []
    rets = []
    weights_grid = []

    for target in target_grid:

        # Contrainte de rendement cible
        def cons_ret_fun(w: np.ndarray) -> float:
            return float(np.dot(w, mu_vec) - target) # type: ignore

        constraints = [
            cons_sum,
            {"type": "eq", "fun": cons_ret_fun},
        ]

        # Fonction objectif : variance
        def obj(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        x0 = np.ones(n) / n
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        if res.success:
            stats = portfolio_stats(res.x, mu_ann, cov_ann)
            vols.append(stats.vol_ann)
            rets.append(stats.ret_ann)
            weights_grid.append(stats.weights)

    if not vols:
        raise RuntimeError("Efficient frontier optimisation failed for all target returns.")

    vols = np.asarray(vols)
    rets = np.asarray(rets)
    weights_grid = np.asarray(weights_grid)

    # Tri par volatilité pour tracer une frontière propre
    order = np.argsort(vols)
    return vols[order], rets[order], weights_grid[order]