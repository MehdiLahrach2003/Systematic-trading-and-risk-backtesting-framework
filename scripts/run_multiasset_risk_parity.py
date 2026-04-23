# scripts/run_multiasset_risk_parity.py

"""
Portefeuille multi-actifs de type "risk parity" simplifié
à partir de stratégies SMA 20/100.

Idée :
- on applique la stratégie SMA 20/100 sur plusieurs actifs
- on backteste chaque stratégie séparément
- on mesure la volatilité de chaque stratégie
- on attribue à chaque stratégie un poids proportionnel à 1 / volatilité
- on combine ensuite les stratégies dans un portefeuille

Attention :
ici, "risk parity" est pris dans un sens simple :
    poids ∝ 1 / volatilité

Ce n'est pas le risk parity complet avec prise en compte explicite
des corrélations, mais c'est déjà une approximation très utilisée.
"""

import os
import sys
import math

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# Permet d'importer les modules du projet quand on lance ce script directement
# ------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_multi_assets
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.portfolio import combine_backtests


# ============================================================
# 1) Construire les stratégies SMA sur chaque actif
# ============================================================
def build_sma_strategies(df_prices):
    """
    Pour chaque actif :
    - construit les positions SMA 20/100
    - lance le backtest
    - stocke le résultat

    Paramètre
    ---------
    df_prices : DataFrame
        colonnes = actifs, valeurs = prix

    Retour
    ------
    dict[str, BacktestResult]
    """
    results = {}

    for symbol in df_prices.columns:
        price = df_prices[symbol].dropna()

        # Signal SMA 20/100
        pos = sma_crossover_positions(price, short=20, long=100)

        # Backtest de la stratégie sur cet actif
        res = run_backtest(price, pos, cost_bps=1.0, initial_capital=1.0)

        results[symbol] = res

    return results


# ============================================================
# 2) Calcul des poids inverse-volatilité
# ============================================================
def compute_inverse_vol_weights(results):
    """
    Calcule des poids proportionnels à 1 / volatilité.

    Logique :
    - une stratégie très volatile reçoit moins de poids
    - une stratégie plus stable reçoit plus de poids

    Paramètre
    ---------
    results : dict[str, BacktestResult]

    Retour
    ------
    weights : dict[str, float]
        poids normalisés (somme = 1)
    vols : dict[str, float]
        volatilités annualisées de chaque stratégie
    """
    vols = {}
    inv_vol = {}

    for name, res in results.items():
        # Volatilité annualisée à partir des rendements journaliers
        sigma = float(res.returns.std() * math.sqrt(252.0))
        vols[name] = sigma

        # Si la volatilité est positive, on prend son inverse
        if sigma > 0.0:
            inv_vol[name] = 1.0 / sigma
        else:
            inv_vol[name] = 0.0

    # Normalisation pour que la somme des poids fasse 1
    total = sum(inv_vol.values())

    if total == 0.0:
        # Cas extrême : toutes les volatilités sont nulles
        n = len(inv_vol)
        weights = {k: 1.0 / n for k in inv_vol.keys()}
    else:
        weights = {k: v / total for k, v in inv_vol.items()}

    return weights, vols


# ============================================================
# 3) Tracé des courbes d'equity
# ============================================================
def plot_multiasset_equity(results, portfolio_res, title_suffix="(inverse-vol weights)"):
    """
    Trace :
    - les courbes d'equity des stratégies individuelles
    - la courbe d'equity du portefeuille inverse-vol
    """
    plt.figure(figsize=(12, 6))

    # --------------------------------------------------------
    # Stratégies individuelles
    # --------------------------------------------------------
    for name, res in results.items():
        eq = res.equity / res.equity.iloc[0]
        plt.plot(eq.index, eq, label=name)

    # --------------------------------------------------------
    # Portefeuille global
    # --------------------------------------------------------
    port_eq = portfolio_res.equity / portfolio_res.equity.iloc[0]
    plt.plot(port_eq.index, port_eq, label="Portfolio inv-vol", linewidth=2.0)

    plt.axhline(1.0, color="black", lw=0.8, ls="--")
    plt.title(f"Stratégies SMA multi-actifs vs portefeuille {title_suffix}")
    plt.ylabel("Equity (rebasée à 1.0)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "multiasset_risk_parity_equity.png",
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Risk-parity equity plot saved → {out_png}")

    plt.show()


# ============================================================
# 4) Affichage des poids et métriques
# ============================================================
def print_metrics_table(results, portfolio_res, weights, vols):
    """
    Affiche :
    - les poids du portefeuille inverse-vol
    - la volatilité de chaque stratégie
    - les métriques des stratégies individuelles
    - les métriques du portefeuille
    """
    print("\n===== Poids inverse-volatilité (espace stratégies) =====")
    for name in results.keys():
        w = weights[name]
        v = vols[name]
        print(f"{name:10s}  weight = {w:6.3f}   vol_ann = {v:6.3f}")

    print("\n===== Métriques des stratégies =====")
    for name, res in results.items():
        print(f"\n--- {name} ---")
        for k, v in res.metrics.items():
            print(f"{k:25s}: {v:.4f}")
        print(f"{'Final equity':25s}: {res.equity.iloc[-1]:.4f}")

    print("\n===== Portefeuille inverse-vol =====")
    for k, v in portfolio_res.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"{'Final equity':25s}: {portfolio_res.equity.iloc[-1]:.4f}")


# ============================================================
# 5) Main
# ============================================================
def main():
    # --------------------------------------------------------
    # 1) Choix de l'univers d'actifs
    # --------------------------------------------------------
    symbols = ["MSFT", "BTCUSD", "SP500", "AAPL"]

    # --------------------------------------------------------
    # 2) Chargement des prix multi-actifs
    # --------------------------------------------------------
    df_prices = load_multi_assets(symbols)

    # --------------------------------------------------------
    # 3) Construction des stratégies SMA sur chaque actif
    # --------------------------------------------------------
    results = build_sma_strategies(df_prices)

    # --------------------------------------------------------
    # 4) Calcul des poids inverse-volatilité
    # --------------------------------------------------------
    weights, vols = compute_inverse_vol_weights(results)

    # --------------------------------------------------------
    # 5) Construction du portefeuille global
    # --------------------------------------------------------
    portfolio_res = combine_backtests(results, weights=weights)

    # --------------------------------------------------------
    # 6) Affichage des résultats
    # --------------------------------------------------------
    print_metrics_table(results, portfolio_res, weights, vols)

    # --------------------------------------------------------
    # 7) Tracé des courbes
    # --------------------------------------------------------
    plot_multiasset_equity(results, portfolio_res)


if __name__ == "__main__":
    main()