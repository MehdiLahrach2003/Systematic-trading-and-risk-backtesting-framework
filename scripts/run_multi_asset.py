# scripts/run_multi_asset.py
# ------------------------------------------------------------
# Ce script réalise un backtest multi-actifs :
# - on charge plusieurs actifs
# - on applique la stratégie SMA 20/100 sur chacun
# - on backteste chaque actif séparément
# - on construit un portefeuille équipondéré
# - on compare les courbes d'equity
# ------------------------------------------------------------

from __future__ import annotations
import os
import sys
import matplotlib.pyplot as plt

# Permet d'importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.load_multi import load_multi_assets
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest, combine_backtests


# ------------------------------------------------------------
# 1) Tracé des courbes d'equity
# ------------------------------------------------------------
def plot_equities(res_dict):
    """
    Trace les courbes d'equity des différents actifs
    ainsi que celle du portefeuille.
    """
    plt.figure(figsize=(12, 6))

    for ticker, res in res_dict.items():
        eq = res.equity / res.equity.iloc[0]
        plt.plot(eq.index, eq, label=ticker)

    plt.axhline(1.0, color="black", ls="--", lw=0.8)
    plt.title("Multi-asset SMA 20/100 strategies vs portfolio")
    plt.ylabel("Equity (rebased to 1.0)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "multi_asset_equity.png"
    )
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Equity figure saved → {out_png}")

    plt.show()


# ------------------------------------------------------------
# 2) Affichage propre des métriques
# ------------------------------------------------------------
def print_metrics(name, res):
    """
    Affiche les métriques d'un backtest.
    """
    print(f"\n===== {name} =====")
    for k, v in res.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"{'Final equity':25s}: {res.equity.iloc[-1]:.4f}")


# ------------------------------------------------------------
# 3) Exécution principale
# ------------------------------------------------------------
def main():
    # ------------------------------------------------------------
    # 1) Chargement des actifs
    # ------------------------------------------------------------
    assets = load_multi_assets()

    print("\nLoaded assets:")
    for t in assets:
        print(" -", t)

    # ------------------------------------------------------------
    # 2) Backtest de la stratégie SMA sur chaque actif
    # ------------------------------------------------------------
    results = {}

    for ticker, price in assets.items():
        # Signal SMA 20/100
        pos = sma_crossover_positions(price, short=20, long=100)

        # Backtest individuel
        res = run_backtest(price, pos, cost_bps=1.0, initial_capital=1.0)

        results[ticker] = res

    # ------------------------------------------------------------
    # 3) Construction du portefeuille équipondéré
    # ------------------------------------------------------------
    weights = {t: 1.0 / len(results) for t in results}

    portfolio_res = combine_backtests(results, weights=weights)

    # On ajoute le portefeuille à la liste des résultats
    results["Portfolio EW"] = portfolio_res

    # ------------------------------------------------------------
    # 4) Affichage des métriques
    # ------------------------------------------------------------
    for name, res in results.items():
        print_metrics(name, res)

    # ------------------------------------------------------------
    # 5) Tracé des courbes
    # ------------------------------------------------------------
    plot_equities(results)


if __name__ == "__main__":
    main()