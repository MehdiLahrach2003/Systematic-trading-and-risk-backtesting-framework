# scripts/run_multiasset_minvar.py

"""
Portefeuille multi-actifs de variance minimale à partir de stratégies SMA 20/100.

Idée :
- on prend plusieurs actifs
- on applique une stratégie SMA 20/100 sur chacun
- on backteste chaque stratégie séparément
- on récupère les rendements de stratégie
- on calcule la matrice de covariance de ces rendements
- on construit le portefeuille de variance minimale :
      w proportionnel à Sigma inverse fois le vecteur de 1

On compare ensuite :
- les stratégies individuelles
- un portefeuille équipondéré
- un portefeuille de variance minimale
"""

import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

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
# 2) Calcul des poids de variance minimale
# ============================================================
def compute_minvar_weights(results):
    """
    Calcule les poids du portefeuille de variance minimale
    à partir des rendements journaliers des stratégies.

    Formule théorique non contrainte :
        w* proportionnel à Sigma inverse fois 1

    Les poids sont ensuite normalisés pour sommer à 1.

    Paramètre
    ---------
    results : dict[str, BacktestResult]

    Retour
    ------
    weights : dict[str, float]
        poids min-var
    cov : DataFrame
        matrice de covariance des rendements de stratégie
    """
    import pandas as pd

    # --------------------------------------------------------
    # Construire un DataFrame de rendements de stratégie
    # chaque colonne = une stratégie / un actif
    # --------------------------------------------------------
    ret_df = pd.DataFrame(
        {name: res.returns.astype(float) for name, res in results.items()}
    ).dropna(how="all")

    # --------------------------------------------------------
    # Matrice de covariance Sigma
    # --------------------------------------------------------
    cov = ret_df.cov()

    # Vecteur de 1
    names = list(ret_df.columns)
    ones = np.ones(len(names))

    # --------------------------------------------------------
    # Inversion pseudo-inverse de Sigma
    # (plus robuste si la matrice est mal conditionnée)
    # --------------------------------------------------------
    sigma_inv = np.linalg.pinv(cov.values)

    # --------------------------------------------------------
    # Poids min-var non normalisés
    # --------------------------------------------------------
    raw_w = sigma_inv @ ones

    # --------------------------------------------------------
    # Normalisation pour que la somme des poids fasse 1
    # --------------------------------------------------------
    w = raw_w / raw_w.sum()

    weights = {name: float(w[i]) for i, name in enumerate(names)}
    return weights, cov


# ============================================================
# 3) Tracé des courbes d'equity
# ============================================================
def plot_multiasset_minvar(results, ew_res, minvar_res):
    """
    Trace :
    - les stratégies individuelles
    - le portefeuille équipondéré
    - le portefeuille de variance minimale
    """
    plt.figure(figsize=(12, 6))

    # --------------------------------------------------------
    # Stratégies individuelles
    # --------------------------------------------------------
    for name, res in results.items():
        eq = res.equity / res.equity.iloc[0]
        plt.plot(eq.index, eq, label=name)

    # --------------------------------------------------------
    # Portefeuille équipondéré
    # --------------------------------------------------------
    ew_eq = ew_res.equity / ew_res.equity.iloc[0]
    plt.plot(ew_eq.index, ew_eq, label="Portfolio EW", linewidth=2.0, ls="--")

    # --------------------------------------------------------
    # Portefeuille de variance minimale
    # --------------------------------------------------------
    mv_eq = minvar_res.equity / minvar_res.equity.iloc[0]
    plt.plot(mv_eq.index, mv_eq, label="Portfolio min-var", linewidth=2.0)

    plt.axhline(1.0, color="black", lw=0.8, ls="--")
    plt.title("Stratégies SMA multi-actifs vs portefeuilles (EW vs min-var)")
    plt.ylabel("Equity (rebasée à 1.0)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "multiasset_minvar_equity.png",
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Min-variance equity plot saved → {out_png}")

    plt.show()


# ============================================================
# 4) Affichage des poids, métriques et covariance
# ============================================================
def print_weights_and_metrics(results, ew_res, minvar_res, minvar_w, cov):
    """
    Affiche :
    - les poids du portefeuille min-var
    - les métriques du portefeuille EW et min-var
    - la matrice de covariance
    """
    import pandas as pd

    print("\n===== Poids du portefeuille de variance minimale =====")
    for name, w in minvar_w.items():
        print(f"{name:10s}  w_minvar = {w:7.3f}")

    # --------------------------------------------------------
    # Fonction utilitaire : volatilité annualisée
    # --------------------------------------------------------
    def ann_vol(res):
        return float(res.returns.std() * math.sqrt(252.0))

    print("\n===== Comparaison des portefeuilles =====")
    print(f"{'Portfolio':20s} {'CumReturn':>10s} {'AnnVol':>10s} {'Sharpe?':>10s}")

    for label, res in [
        ("Equal-weight", ew_res),
        ("Min-variance", minvar_res),
    ]:
        eq = res.equity
        cum_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        vol = ann_vol(res)
        sharpe = res.metrics.get("Sharpe Ratio", float("nan"))

        print(f"{label:20s} {cum_ret:10.4f} {vol:10.4f} {sharpe:10.4f}")

    print("\n===== Matrice de covariance des rendements de stratégie =====")
    cov_round = cov.round(4)

    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(cov_round)


# ============================================================
# 5) Main
# ============================================================
def main():
    # --------------------------------------------------------
    # 1) Univers d'actifs
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
    # 4) Portefeuille équipondéré comme baseline
    # --------------------------------------------------------
    n = len(results)
    ew_weights = {name: 1.0 / n for name in results.keys()}
    ew_res = combine_backtests(results, weights=ew_weights)

    # --------------------------------------------------------
    # 5) Calcul des poids min-var
    # --------------------------------------------------------
    minvar_w, cov = compute_minvar_weights(results)

    # --------------------------------------------------------
    # 6) Construction du portefeuille min-var
    # --------------------------------------------------------
    minvar_res = combine_backtests(results, weights=minvar_w)

    # --------------------------------------------------------
    # 7) Affichage des résultats
    # --------------------------------------------------------
    print_weights_and_metrics(results, ew_res, minvar_res, minvar_w, cov)

    # --------------------------------------------------------
    # 8) Tracé des courbes
    # --------------------------------------------------------
    plot_multiasset_minvar(results, ew_res, minvar_res)


if __name__ == "__main__":
    main()