# scripts/make_report.py

"""
Ce script lance un backtest SMA 20/100 et génère un mini rapport de performance.

Le rapport contient :
- les métriques principales
- un tear sheet :
    * equity curve vs buy & hold
    * rolling Sharpe
    * drawdown
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Permet d'importer les modules du projet quand on lance ce script directement
# ------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest


# ============================================================
# 1) Calcul du drawdown
# ============================================================
def compute_drawdown(equity: pd.Series) -> pd.Series:
    """
    Calcule la série de drawdown à partir d'une courbe d'equity.

    Formule :
        drawdown_t = equity_t / max_{s <= t}(equity_s) - 1

    Donc :
    - 0 = nouveau plus haut
    - négatif = on est sous le plus haut historique
    """
    eq = equity.astype(float)
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    dd.name = "drawdown"
    return dd


# ============================================================
# 2) Rolling Sharpe
# ============================================================
def rolling_sharpe(returns: pd.Series, window: int = 126, ann_factor: int = 252) -> pd.Series:
    """
    Calcule le Sharpe ratio glissant sur une fenêtre mobile.

    Hypothèse :
    - rendements journaliers
    - Sharpe = moyenne / écart-type * racine(252)
    """
    r = returns.astype(float)

    # Moyenne glissante
    roll_mean = r.rolling(window).mean()

    # Écart-type glissant
    roll_std = r.rolling(window).std(ddof=1)

    # Sharpe glissant
    sharpe = roll_mean / roll_std * np.sqrt(ann_factor)
    sharpe.name = f"rolling_sharpe_{window}"

    return sharpe


# ============================================================
# 3) Construction du tear sheet
# ============================================================
def plot_tear_sheet(
    price: pd.Series,
    equity: pd.Series,
    strategy_returns: pd.Series,
    out_png: str | None = None,
) -> None:
    """
    Trace un tear sheet avec :
    - equity de la stratégie vs buy and hold
    - rolling Sharpe
    - drawdown
    """

    # --------------------------------------------------------
    # Equity de la stratégie rebasée à 1
    # --------------------------------------------------------
    eq = equity.astype(float)
    eq_rebased = eq / eq.iloc[0]

    # --------------------------------------------------------
    # Benchmark buy and hold rebasé à 1
    # --------------------------------------------------------
    bh = price.astype(float) / price.iloc[0]
    bh.name = "buy_and_hold"

    # --------------------------------------------------------
    # Mesures de risque / performance
    # --------------------------------------------------------
    dd = compute_drawdown(eq)
    roll_sh = rolling_sharpe(strategy_returns, window=126)

    # --------------------------------------------------------
    # Figure en 3 panneaux
    # --------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    # ===== 1) Equity vs benchmark =====
    ax1.plot(eq_rebased.index, eq_rebased, label="Strategy", lw=1.5)
    ax1.plot(bh.index, bh, label="Buy & Hold", lw=1.2)
    ax1.set_ylabel("Index (× départ)")
    ax1.set_title("Equity (rebasée) vs Benchmark")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ===== 2) Rolling Sharpe =====
    ax2.plot(roll_sh.index, roll_sh, lw=1.2)
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.set_ylabel("Sharpe")
    ax2.set_title("Rolling Sharpe (126 jours)")
    ax2.grid(alpha=0.3)

    # ===== 3) Drawdown =====
    ax3.fill_between(dd.index, dd, 0.0, color="steelblue", alpha=0.4)
    ax3.set_ylabel("Drawdown")
    ax3.set_title("Drawdown")
    ax3.grid(alpha=0.3)

    plt.tight_layout()

    # Sauvegarde éventuelle
    if out_png is not None:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"[OK] Tear sheet saved → {out_png}")

    plt.show()


# ============================================================
# 4) Main
# ============================================================
def main():
    # --------------------------------------------------------
    # 1) Chargement des prix
    # --------------------------------------------------------
    df = load_prices()
    price = df["price"]

    # --------------------------------------------------------
    # 2) Construction de la stratégie SMA 20/100
    # --------------------------------------------------------
    positions = sma_crossover_positions(price, short=20, long=100)

    # --------------------------------------------------------
    # 3) Backtest
    # --------------------------------------------------------
    result = run_backtest(
        df,
        positions,
        cost_bps=1.0,
        initial_capital=1.0,
    )

    # --------------------------------------------------------
    # 4) Affichage des métriques principales
    # --------------------------------------------------------
    print("\n===== Summary metrics =====")
    for k, v in result.metrics.items():
        print(f"{k:25s}: {v: .4f}")

    print(f"{'Final equity':25s}: {result.equity.iloc[-1]: .4f}")

    # --------------------------------------------------------
    # 5) Génération du tear sheet
    # --------------------------------------------------------
    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "tear_sheet.png"
    )

    plot_tear_sheet(
        price=price,
        equity=result.equity,
        strategy_returns=result.returns,
        out_png=out_png,
    )


if __name__ == "__main__":
    main()