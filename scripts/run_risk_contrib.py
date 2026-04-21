# scripts/run_risk_contrib.py
# ------------------------------------------------------------
# Ce script construit plusieurs portefeuilles de stratégies SMA
# sur plusieurs actifs, puis calcule leurs contributions au risque.
#
# Portefeuilles étudiés :
# - Equal-weight
# - Inverse-volatility
# - Minimum-variance
# - Maximum-Sharpe
#
# Objectif :
# comprendre non seulement les poids, mais surtout
# quelle part du risque total vient de chaque actif.
# ------------------------------------------------------------

import os
import sys
import math

import numpy as np
import pandas as pd

# Permet d'importer les modules du projet quand on lance ce script directement
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_multi_assets
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.optimizer import (
    compute_mu_cov_from_results,
    solve_min_var,
    solve_max_sharpe,
)
from backtesting.risk_contrib import risk_contributions


# ------------------------------------------------------------
# 1) Construire les stratégies SMA sur chaque actif
# ------------------------------------------------------------
def build_sma_strategies(df_prices: pd.DataFrame) -> dict:
    """
    Pour chaque actif, construit une stratégie SMA 20/100
    et exécute le backtest correspondant.

    Retour :
    dict[str, BacktestResult]
    """
    results = {}

    for symbol in df_prices.columns:
        price = df_prices[symbol].dropna()

        # Signal SMA
        pos = sma_crossover_positions(price, short=20, long=100)

        # Backtest de la stratégie sur cet actif
        res = run_backtest(price, pos, cost_bps=1.0, initial_capital=1.0)

        results[symbol] = res

    return results


# ------------------------------------------------------------
# 2) Construire les poids inverse-volatilité
# ------------------------------------------------------------
def inverse_vol_weights_from_results(results: dict, asset_order: list[str]) -> np.ndarray:
    """
    Calcule des poids inverse-volatilité à partir des rendements
    des stratégies.
    """
    sigmas = []

    for name in asset_order:
        r = results[name].returns
        sigma = float(r.std() * math.sqrt(252.0))
        sigmas.append(sigma)

    sigmas = np.asarray(sigmas, dtype=float)

    # Poids proportionnels à 1 / volatilité
    inv = np.where(sigmas > 0.0, 1.0 / sigmas, 0.0)

    total = inv.sum()
    if total == 0.0:
        return np.ones_like(inv) / len(inv)

    return inv / total


# ------------------------------------------------------------
# 3) Affichage propre des contributions au risque
# ------------------------------------------------------------
def print_risk_table(title: str, rc_result, mu_ann: pd.Series):
    """
    Affiche proprement les poids et contributions au risque
    d'un portefeuille donné.
    """
    print(f"\n===== {title} =====")

    df = pd.DataFrame({
        "Weight": rc_result.weights,
        "MRC": rc_result.mrc,   # contribution marginale au risque
        "CRC": rc_result.crc,   # contribution composante au risque
        "Mu_ann": mu_ann.reindex(rc_result.weights.index),
    })

    # Contribution au risque exprimée en pourcentage
    if rc_result.vol_port > 0.0:
        df["CRC_%"] = 100.0 * df["CRC"] / rc_result.vol_port
    else:
        df["CRC_%"] = 0.0

    print(f"Portfolio volatility (ann.): {rc_result.vol_port:.4f}\n")
    print(df.round(4))


# ------------------------------------------------------------
# 4) Exécution principale
# ------------------------------------------------------------
def main():
    # Univers d'actifs
    symbols = ["MSFT", "BTCUSD", "SP500", "AAPL"]

    # Chargement des prix multi-actifs
    df_prices = load_multi_assets(symbols)

    # Construction des stratégies SMA et backtests
    results = build_sma_strategies(df_prices)

    # Rendements annuels moyens et covariance annuelle
    mu_ann, cov_ann = compute_mu_cov_from_results(results)

    assets = list(mu_ann.index)
    n = len(assets)

    # ------------------------------------------------------------
    # Portefeuille 1 : poids égaux
    # ------------------------------------------------------------
    w_eq = np.ones(n) / n

    # ------------------------------------------------------------
    # Portefeuille 2 : inverse-volatilité
    # ------------------------------------------------------------
    w_inv = inverse_vol_weights_from_results(results, assets)

    # ------------------------------------------------------------
    # Portefeuille 3 : variance minimale
    # ------------------------------------------------------------
    stats_minvar = solve_min_var(mu_ann, cov_ann)
    w_minvar = stats_minvar.weights

    # ------------------------------------------------------------
    # Portefeuille 4 : Sharpe maximal
    # ------------------------------------------------------------
    stats_maxsharpe = solve_max_sharpe(mu_ann, cov_ann, rf=0.0)
    w_maxsharpe = stats_maxsharpe.weights

    # ------------------------------------------------------------
    # Calcul des contributions au risque
    # ------------------------------------------------------------
    rc_eq = risk_contributions(pd.Series(w_eq, index=assets), cov_ann)
    rc_inv = risk_contributions(pd.Series(w_inv, index=assets), cov_ann)
    rc_minvar = risk_contributions(pd.Series(w_minvar, index=assets), cov_ann)
    rc_maxsharpe = risk_contributions(pd.Series(w_maxsharpe, index=assets), cov_ann)

    # ------------------------------------------------------------
    # Affichage des résultats
    # ------------------------------------------------------------
    print_risk_table("Equal-weight portfolio", rc_eq, mu_ann)
    print_risk_table("Inverse-vol portfolio", rc_inv, mu_ann)
    print_risk_table("Min-variance portfolio", rc_minvar, mu_ann)
    print_risk_table("Max-Sharpe portfolio", rc_maxsharpe, mu_ann)


if __name__ == "__main__":
    main()