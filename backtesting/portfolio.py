# backtesting/portfolio.py
# ------------------------------------------------------------
# Ce fichier sert à combiner plusieurs stratégies déjà backtestées
# en un seul portefeuille.
#
# On travaille ici sur les objets BacktestResult produits par engine.py.
#
# Idée :
# si on a plusieurs stratégies avec leurs rendements respectifs,
# alors on peut construire un portefeuille pondéré :
#
#   rendement_portefeuille(t) = somme_i poids_i × rendement_i(t)
#
# Hypothèse :
# toutes les stratégies sont exprimées sur la même base de capital
# et sont rééquilibrées chaque jour vers leurs poids cibles.
# ------------------------------------------------------------

from __future__ import annotations

from typing import Mapping
import math

import numpy as np
import pandas as pd

from .engine import BacktestResult, _annualize_sharpe


def combine_backtests(
    results: Mapping[str, BacktestResult],
    weights: Mapping[str, float] | None = None,
) -> BacktestResult:
    """
    Combine plusieurs objets BacktestResult en un seul portefeuille.

    Paramètres
    ----------
    results : dictionnaire nom -> BacktestResult
        Résultat du backtest de chaque stratégie.
    weights : dictionnaire nom -> poids, optionnel
        Poids du portefeuille.
        Si None, on utilise des poids égaux.

    Retour
    ------
    BacktestResult
        Résultat global du portefeuille.
    """

    # ------------------------------------------------------------
    # 1) Vérification : il faut au moins une stratégie
    # ------------------------------------------------------------
    if len(results) == 0:
        raise ValueError("`results` must contain at least one strategy.")

    # ------------------------------------------------------------
    # 2) Construire un DataFrame des rendements journaliers
    # ------------------------------------------------------------
    # Chaque colonne correspond à une stratégie
    ret_df = pd.DataFrame({name: res.returns for name, res in results.items()})
    ret_df = ret_df.fillna(0.0)

    # ------------------------------------------------------------
    # 3) Définition des poids
    # ------------------------------------------------------------
    if weights is None:
        # Si aucun poids n'est fourni, on met des poids égaux
        n = ret_df.shape[1]
        w = pd.Series(1.0 / n, index=ret_df.columns, dtype=float)
    else:
        # Sinon, on lit les poids donnés
        w = pd.Series(weights, dtype=float)

        # On aligne les poids sur les colonnes du DataFrame
        # Les stratégies sans poids explicite reçoivent 0
        w = w.reindex(ret_df.columns).fillna(0.0)

        # On interdit le cas où tous les poids sont nuls
        if w.sum() == 0.0:
            raise ValueError("All portfolio weights are zero.")

        # On normalise pour que la somme des poids fasse 1
        w = w / w.sum()

    # ------------------------------------------------------------
    # 4) Rendements du portefeuille
    # ------------------------------------------------------------
    # À chaque date :
    # rendement portefeuille = somme des poids × rendements
    port_ret = (ret_df * w).sum(axis=1)
    port_ret.name = "portfolio_returns"

    # ------------------------------------------------------------
    # 5) Courbe d'equity du portefeuille
    # ------------------------------------------------------------
    equity = (1.0 + port_ret).cumprod()
    equity.name = "equity"

    # ------------------------------------------------------------
    # 6) Combinaison des positions
    # ------------------------------------------------------------
    pos_df = pd.DataFrame({name: res.positions for name, res in results.items()}).fillna(0.0)
    portfolio_pos = (pos_df * w).sum(axis=1)
    portfolio_pos.name = "position"

    # ------------------------------------------------------------
    # 7) Combinaison des coûts
    # ------------------------------------------------------------
    # On utilise ici la valeur absolue des poids
    costs_df = pd.DataFrame({name: res.costs for name, res in results.items()}).fillna(0.0)
    portfolio_costs = (costs_df * w.abs()).sum(axis=1)
    portfolio_costs.name = "costs"

    # ------------------------------------------------------------
    # 8) Combinaison des trades
    # ------------------------------------------------------------
    trades_df = pd.DataFrame({name: res.trades for name, res in results.items()}).fillna(0.0)
    portfolio_trades = (trades_df * w.abs()).sum(axis=1)
    portfolio_trades.name = "trades"

    # ------------------------------------------------------------
    # 9) Calcul des métriques globales
    # ------------------------------------------------------------
    if len(equity) > 1:
        cum_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    else:
        cum_ret = 0.0

    if len(port_ret) > 1:
        ann_vol = float(port_ret.std(ddof=1) * math.sqrt(252))
    else:
        ann_vol = 0.0

    sharpe = _annualize_sharpe(port_ret)

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = float(dd.min()) if not dd.empty else 0.0

    total_costs = float(portfolio_costs.sum())

    metrics = {
        "Cumulative Return": cum_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Total Costs": total_costs,
    }

    # ------------------------------------------------------------
    # 10) Retour du résultat global
    # ------------------------------------------------------------
    return BacktestResult(
        equity=equity,
        returns=port_ret,
        costs=portfolio_costs,
        positions=portfolio_pos,
        trades=portfolio_trades,
        metrics=metrics,
    )