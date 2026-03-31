# backtesting/engine.py
# ------------------------------------------------------------
# Moteur simple de backtest long/short sur données journalières.
#
# Ce fichier sert à transformer :
#   - une série de prix
#   - une série de positions
#
# en :
#   - rendements de stratégie
#   - coûts de transaction
#   - equity curve
#   - métriques de performance
# ------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union, Mapping

import math
import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """
    Objet qui contient le résultat complet d'un backtest.

    Tous les objets sont indexés par date.
    """
    equity: pd.Series          # courbe de capital
    returns: pd.Series         # rendements journaliers de la stratégie (nets de coûts)
    costs: pd.Series           # coûts de transaction journaliers
    positions: pd.Series       # positions du trader au cours du temps
    trades: pd.Series          # taille des changements de position
    metrics: Dict[str, float]  # résumé des métriques principales


# -------------------------------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------------------------------


def _to_price_series(x: Union[pd.Series, pd.DataFrame], col: str = "price") -> pd.Series:
    """
    Transforme l'entrée en une série de prix propre.

    Cas possibles :
    - si x est déjà une Series : on la copie
    - si x est un DataFrame : on récupère la colonne 'price'
    - sinon : on essaie de convertir directement en Series
    """
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        if col not in x.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")
        s = x[col].copy()
    else:
        s = pd.Series(x)

    # Conversion en float + tri chronologique
    s = s.astype(float).sort_index()
    s.name = "price"
    return s


def _annualize_sharpe(daily: pd.Series, freq_per_year: int = 252) -> float:
    """
    Calcule le Sharpe annualisé à partir des rendements journaliers.

    Formule :
        moyenne / écart-type × racine(252)
    """
    r = daily.dropna()
    if r.empty or r.std() == 0:
        return 0.0
    return (r.mean() / r.std()) * math.sqrt(freq_per_year)


def _compute_metrics(
    strat_net: pd.Series,
    equity: pd.Series,
    costs: pd.Series,
) -> Dict[str, float]:
    """
    Calcule les métriques principales du backtest.
    """

    # Rendement cumulé final
    if len(equity) > 1:
        cum_ret = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    else:
        cum_ret = 0.0

    # Volatilité annualisée
    if len(strat_net) > 1:
        ann_vol = float(strat_net.std() * math.sqrt(252))
    else:
        ann_vol = 0.0

    # Sharpe
    sharpe = _annualize_sharpe(strat_net)

    # Maximum drawdown calculé à partir de la courbe d'equity
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = float(dd.min()) if not dd.empty else 0.0

    # Coûts totaux
    total_costs = float(costs.sum())

    return {
        "Cumulative Return": cum_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Total Costs": total_costs,
    }


# -------------------------------------------------------------------------
# Backtest d'une seule stratégie
# -------------------------------------------------------------------------


def run_backtest(
    price_like: Union[pd.Series, pd.DataFrame],
    positions_like: Union[pd.Series, pd.DataFrame],
    cost_bps: float = 1.0,
    initial_capital: float = 1.0,
) -> BacktestResult:
    """
    Exécute un backtest vectorisé.

    Entrées :
    - price_like : série de prix
    - positions_like : série de positions
    - cost_bps : coût de transaction en basis points
    - initial_capital : capital initial

    Sortie :
    - un objet BacktestResult
    """

    # On met les prix sous forme standard
    price = _to_price_series(price_like)

    # ------------------------------------------------------------
    # Récupération / standardisation des positions
    # ------------------------------------------------------------
    if isinstance(positions_like, pd.DataFrame):
        if "position" in positions_like.columns:
            pos = positions_like["position"]
        else:
            pos = positions_like.iloc[:, 0]
    else:
        pos = positions_like

    # On aligne les positions sur l'index des prix
    pos = pd.Series(pos, index=price.index).reindex(price.index).fillna(0.0).astype(float)
    pos.name = "position"

    # ------------------------------------------------------------
    # Rendements du sous-jacent
    # ------------------------------------------------------------
    ret_under = price.pct_change().fillna(0.0)

    # ------------------------------------------------------------
    # Rendement brut de la stratégie
    # ------------------------------------------------------------
    # On utilise la position de la veille :
    # si on est long hier et que le prix monte aujourd'hui, on gagne
    # si on est short hier et que le prix baisse aujourd'hui, on gagne
    strat_gross = pos.shift(1).fillna(0.0) * ret_under
    strat_gross.name = "strat_gross"

    # ------------------------------------------------------------
    # Trades et coûts
    # ------------------------------------------------------------
    # Un trade est mesuré ici par la variation absolue de position
    trades = pos.diff().abs().fillna(0.0)
    trades.name = "trades"

    # Coût proportionnel au changement de position
    costs = trades * (cost_bps / 10_000.0)
    costs.name = "costs"

    # ------------------------------------------------------------
    # Rendement net et courbe de capital
    # ------------------------------------------------------------
    strat_net = strat_gross - costs
    strat_net.name = "strat_net"

    equity = (1.0 + strat_net).cumprod() * initial_capital
    equity.name = "equity"

    # ------------------------------------------------------------
    # Métriques
    # ------------------------------------------------------------
    metrics = _compute_metrics(strat_net, equity, costs)

    return BacktestResult(
        equity=equity,
        returns=strat_net,
        costs=costs,
        positions=pos,
        trades=trades,
        metrics=metrics,
    )


# -------------------------------------------------------------------------
# Combinaison de plusieurs backtests en un portefeuille
# -------------------------------------------------------------------------


def combine_backtests(
    results: Mapping[str, BacktestResult],
    weights: Mapping[str, float],
    initial_capital: float = 1.0,
) -> BacktestResult:
    """
    Combine plusieurs backtests individuels en un portefeuille.

    Entrées :
    - results : dictionnaire nom -> résultat de backtest
    - weights : dictionnaire nom -> poids
    - initial_capital : capital initial du portefeuille
    """

    if not results:
        raise ValueError("`results` is empty in combine_backtests.")
    if not weights:
        raise ValueError("`weights` is empty in combine_backtests.")

    # Normalisation des poids pour qu'ils somment à 1
    w_series = pd.Series(weights, dtype=float)
    if w_series.sum() == 0:
        raise ValueError("All portfolio weights are zero.")
    w_series = w_series / w_series.sum()

    # On prend l'index du premier backtest comme calendrier de référence
    first_key = next(iter(results))
    master_index = results[first_key].returns.index

    # Initialisation des séries agrégées
    port_ret = pd.Series(0.0, index=master_index)
    port_costs = pd.Series(0.0, index=master_index)
    port_pos = pd.Series(0.0, index=master_index)
    port_trades = pd.Series(0.0, index=master_index)

    for name, res in results.items():
        if name not in w_series.index:
            continue

        w = float(w_series[name])

        # On réaligne toutes les séries sur le calendrier commun
        r = res.returns.reindex(master_index).fillna(0.0)
        c = res.costs.reindex(master_index).fillna(0.0)
        p = res.positions.reindex(master_index).fillna(0.0)
        t = res.trades.reindex(master_index).fillna(0.0)

        # Agrégation linéaire
        port_ret += w * r
        port_costs += w * c
        port_pos += w * p
        port_trades += w * t

    port_ret.name = "portfolio_returns"
    port_costs.name = "portfolio_costs"
    port_pos.name = "portfolio_position"
    port_trades.name = "portfolio_trades"

    # Reconstruction de la courbe d'equity du portefeuille
    equity = (1.0 + port_ret).cumprod() * initial_capital
    equity.name = "equity"

    # Calcul des métriques du portefeuille
    metrics = _compute_metrics(port_ret, equity, port_costs)

    return BacktestResult(
        equity=equity,
        returns=port_ret,
        costs=port_costs,
        positions=port_pos,
        trades=port_trades,
        metrics=metrics,
    )