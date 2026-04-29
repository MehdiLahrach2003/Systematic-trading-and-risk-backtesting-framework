# scripts/run_multiasset_frontier.py
# ------------------------------------------------------------
# Ce script construit une frontière efficiente multi-actifs
# à partir de stratégies SMA 20/100 appliquées sur plusieurs actifs.
#
# Objectif :
# comparer différentes allocations de portefeuille
# dans l'espace (volatilité, rendement).
# ------------------------------------------------------------

import os
import sys
import math

import matplotlib.pyplot as plt
import numpy as np

# Permet d'importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_multi_assets
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.optimizer import (
    compute_mu_cov_from_results,
    portfolio_stats,
    solve_min_var,
    solve_max_sharpe,
    efficient_frontier,
)


# ------------------------------------------------------------
# 1) Construire les stratégies SMA sur chaque actif
# ------------------------------------------------------------
def build_sma_strategies(df_prices):
    """
    Pour chaque colonne de df_prices :
    - construit une stratégie SMA 20/100
    - lance le backtest
    - renvoie un dictionnaire de BacktestResult
    """
    results = {}

    for symbol in df_prices.columns:
        price = df_prices[symbol].dropna()

        # Signal SMA
        pos = sma_crossover_positions(price, short=20, long=100)

        # Backtest de la stratégie
        res = run_backtest(price, pos, cost_bps=1.0, initial_capital=1.0)

        results[symbol] = res

    return results


# ------------------------------------------------------------
# 2) Construire des poids inverse-volatilité
# ------------------------------------------------------------
def compute_inverse_vol_weights_from_results(results, assets):
    """
    Calcule des poids inverse-volatilité
    à partir de la volatilité réalisée des rendements de stratégie.
    """
    sigmas = []

    for name in assets:
        res = results[name]
        sigma = float(res.returns.std() * math.sqrt(252.0))
        sigmas.append(sigma)

    sigmas = np.asarray(sigmas, dtype=float)

    # Poids proportionnels à 1 / volatilité
    inv = np.where(sigmas > 0.0, 1.0 / sigmas, 0.0)

    total = inv.sum()
    if total == 0.0:
        return np.ones_like(inv) / len(inv)

    return inv / total


# ------------------------------------------------------------
# 3) Tracé de la frontière efficiente
# ------------------------------------------------------------
def plot_frontier(
    results,
    mu_ann,
    cov_ann,
    stats_eq,
    stats_inv,
    stats_minvar,
    stats_maxsharpe,
    vols_ef,
    rets_ef,
):
    """
    Trace :
    - les stratégies individuelles
    - les portefeuilles spéciaux
    - la frontière efficiente
    """
    plt.figure(figsize=(10, 6))

    # ------------------------------------------------------------
    # Stratégies individuelles
    # ------------------------------------------------------------
    for name, res in results.items():
        r_daily = res.returns
        mu = float(r_daily.mean() * 252.0)
        vol = float(r_daily.std() * math.sqrt(252.0))

        plt.scatter(vol, mu, label=name, s=50)

    # ------------------------------------------------------------
    # Portefeuilles particuliers
    # ------------------------------------------------------------
    plt.scatter(
        stats_eq.vol_ann,
        stats_eq.ret_ann,
        marker="*", s=120, label="Equal-weight",
    )

    plt.scatter(
        stats_inv.vol_ann,
        stats_inv.ret_ann,
        marker="*", s=120, label="Inverse-vol",
    )

    plt.scatter(
        stats_minvar.vol_ann,
        stats_minvar.ret_ann,
        marker="D", s=80, label="Min-Var",
    )

    plt.scatter(
        stats_maxsharpe.vol_ann,
        stats_maxsharpe.ret_ann,
        marker="D", s=80, label="Max-Sharpe",
    )

    # ------------------------------------------------------------
    # Frontière efficiente
    # ------------------------------------------------------------
    plt.plot(vols_ef, rets_ef, lw=2.0, label="Efficient frontier")

    plt.xlabel("Annualised Volatility")
    plt.ylabel("Annualised Return")
    plt.title("Efficient frontier – SMA 20/100 multi-asset strategies")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "multiasset_efficient_frontier.png",
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Efficient frontier plot saved → {out_png}")

    plt.show()


# ------------------------------------------------------------
# 4) Script principal
# ------------------------------------------------------------
def main():
    # Univers d'actifs
    symbols = ["MSFT", "BTCUSD", "SP500", "AAPL"]

    # Chargement des prix multi-actifs
    df_prices = load_multi_assets(symbols)

    # Construction des stratégies SMA
    results = build_sma_strategies(df_prices)

    # Rendements annuels moyens et covariance en espace stratégies
    mu_ann, cov_ann = compute_mu_cov_from_results(results)

    assets = list(mu_ann.index)
    n = len(assets)

    # ------------------------------------------------------------
    # Portefeuille équipondéré
    # ------------------------------------------------------------
    w_eq = np.ones(n) / n
    stats_eq = portfolio_stats(w_eq, mu_ann, cov_ann)

    # ------------------------------------------------------------
    # Portefeuille inverse-volatilité
    # ------------------------------------------------------------
    w_inv = compute_inverse_vol_weights_from_results(results, assets)
    stats_inv = portfolio_stats(w_inv, mu_ann, cov_ann)

    # ------------------------------------------------------------
    # Portefeuille variance minimale
    # ------------------------------------------------------------
    stats_minvar = solve_min_var(mu_ann, cov_ann)

    # ------------------------------------------------------------
    # Portefeuille Sharpe maximal
    # ------------------------------------------------------------
    stats_maxsharpe = solve_max_sharpe(mu_ann, cov_ann, rf=0.0)

    # ------------------------------------------------------------
    # Frontière efficiente
    # ------------------------------------------------------------
    vols_ef, rets_ef, _ = efficient_frontier(mu_ann, cov_ann, n_points=40)

    # ------------------------------------------------------------
    # Tracé final
    # ------------------------------------------------------------
    plot_frontier(
        results,
        mu_ann,
        cov_ann,
        stats_eq,
        stats_inv,
        stats_minvar,
        stats_maxsharpe,
        vols_ef,
        rets_ef,
    )


if __name__ == "__main__":
    main()