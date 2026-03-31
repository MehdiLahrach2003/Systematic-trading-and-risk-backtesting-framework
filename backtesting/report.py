# backtesting/report.py
# ------------------------------------------------------------
# Ce fichier contient des outils de reporting pour les backtests.
#
# Il sert à :
# - calculer le drawdown
# - calculer un Sharpe glissant
# - construire un tear sheet
# - exporter les trades en CSV
#
# Attention :
# ce fichier redéfinit un BacktestResult qui n'est pas exactement
# le même que celui de engine.py. Il y a donc une petite incohérence
# de structure dans le projet.
# ------------------------------------------------------------

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class BacktestResult:
    """
    Conteneur de reporting pour les résultats du backtest.

    Attention :
    ce format n'est pas exactement le même que celui de engine.py.
    """
    equity: pd.Series
    returns_under: pd.Series        # rendements du sous-jacent
    strategy_ret: pd.Series         # rendements de la stratégie
    costs: pd.Series                # coûts de transaction
    positions: pd.Series            # positions
    trades: Optional[pd.DataFrame]  # journal de trades
    metrics: Dict[str, Any]         # métriques résumées


# ------------------------------------------------------------
# 1) Calcul du drawdown
# ------------------------------------------------------------
def compute_drawdown(equity: pd.Series) -> pd.Series:
    """
    Calcule la série de drawdown à partir de la courbe d'equity.

    Formule :
        drawdown = equity / max_passé - 1
    """
    eq = equity.astype(float)
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    dd.name = "drawdown"
    return dd


# ------------------------------------------------------------
# 2) Sharpe glissant
# ------------------------------------------------------------
def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    """
    Calcule le ratio de Sharpe sur fenêtre glissante.

    Ici, on suppose des rendements journaliers.
    """
    r = returns.astype(float)
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std(ddof=1)

    sharpe = roll_mean / roll_std * np.sqrt(252.0)
    sharpe.name = f"rolling_sharpe_{window}"
    return sharpe


def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 126,
    ann_factor: int = 252,
) -> pd.Series:
    """
    Version compatible avec d'anciens scripts.

    Fait la même chose que rolling_sharpe, mais avec un facteur
    d'annualisation explicite.
    """
    r = returns.astype(float)
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std(ddof=1)

    sharpe = roll_mean / roll_std * np.sqrt(float(ann_factor))
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sharpe.name = f"rolling_sharpe_{window}"
    return sharpe


# ------------------------------------------------------------
# 3) Tear sheet
# ------------------------------------------------------------
def make_tear_sheet(
    result: BacktestResult,
    price: pd.Series,
    out_png: Optional[str] = None,
) -> None:
    """
    Construit un tear sheet simple avec :
    - equity de la stratégie vs buy-and-hold
    - rolling Sharpe
    - drawdown
    """
    eq = result.equity.astype(float)

    # Rebase la stratégie à 1 pour comparer facilement
    eq_rebased = eq / eq.iloc[0]

    # Benchmark buy-and-hold, lui aussi rebasé à 1
    bh = price.astype(float) / price.iloc[0]
    bh.name = "buy_and_hold"

    # Objets de risque / performance
    dd = compute_drawdown(eq)
    roll_sh = rolling_sharpe(result.strategy_ret, window=126)

    # Figure avec 3 panneaux
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    # 1) Equity de la stratégie vs benchmark
    ax1.plot(eq_rebased.index, eq_rebased, label="Strategy", lw=1.5)
    ax1.plot(bh.index, bh, label="Buy & Hold", lw=1.2)
    ax1.set_ylabel("Index (× start)")
    ax1.set_title("Equity (rebased) vs Benchmark")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2) Sharpe glissant
    ax2.plot(roll_sh.index, roll_sh, lw=1.2)
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.set_ylabel("Sharpe")
    ax2.set_title("Rolling Sharpe (126d)")
    ax2.grid(alpha=0.3)

    # 3) Drawdown
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


# ------------------------------------------------------------
# 4) Export du journal de trades
# ------------------------------------------------------------
def export_trades_to_csv(result: BacktestResult, out_csv: str) -> None:
    """
    Exporte le journal des trades vers un CSV.

    Si aucun trade n'est présent, la fonction ne plante pas.
    """
    trades = result.trades

    if trades is None or len(trades) == 0:
        print("[WARN] No trades found in BacktestResult.trades, nothing to export.")
        return

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    trades.to_csv(out_csv, index=True)
    print(f"[OK] Trades exported → {out_csv}")