# utils/param_search.py

# ------------------------------------------------------------
# Outils de grid-search pour les paramètres de la stratégie SMA crossover.
#
# Objectif :
# tester plusieurs couples (short, long)
# et mesurer leurs performances.
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterable, Tuple

from backtesting.engine import run_backtest
from backtesting.ma_crossover import sma_crossover_positions


def evaluate_sma_grid(
    df: pd.DataFrame,
    shorts: Iterable[int],
    longs: Iterable[int],
    cost_bps: float = 1.0,
    criterion: str = "sharpe",         # "sharpe" ou "cumret"
    initial_capital: float = 1.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recherche exhaustive sur une grille de paramètres SMA.

    On teste tous les couples :
        (short, long)
    avec la contrainte :
        short < long

    Paramètres
    ----------
    df : DataFrame
        doit contenir une colonne 'price'
    shorts : iterable[int]
        liste des fenêtres courtes à tester
    longs : iterable[int]
        liste des fenêtres longues à tester
    cost_bps : float
        coût de transaction en basis points
    criterion : str
        critère pour la matrice pivot :
        - "sharpe"
        - "cumret"
    initial_capital : float
        capital initial du backtest

    Retour
    ------
    results_df : DataFrame
        tableau détaillé avec une ligne par couple (short, long)
    pivot : DataFrame
        matrice de la métrique choisie
        lignes = long
        colonnes = short
    """

    # --------------------------------------------------------
    # 1) Vérification de la présence de la colonne 'price'
    # --------------------------------------------------------
    if "price" not in df.columns:
        raise ValueError("DataFrame must contain 'price'.")

    price = df["price"].astype(float)

    rows = []

    # --------------------------------------------------------
    # 2) Boucle sur toutes les combinaisons de paramètres
    # --------------------------------------------------------
    for s in shorts:
        for l in longs:

            # On ignore les cas absurdes :
            # une moyenne courte doit être plus courte que la longue
            if s >= l:
                continue

            # Construction du signal SMA correspondant
            pos = sma_crossover_positions(price, short=s, long=l)

            # Backtest de cette stratégie
            res = run_backtest(
                price,
                pos,
                cost_bps=cost_bps,
                initial_capital=initial_capital
            )

            # On stocke les métriques importantes
            rows.append(
                {
                    "short": s,
                    "long": l,
                    "sharpe": res.metrics.get("Sharpe Ratio", np.nan),
                    "cumret": res.metrics.get("Cumulative Return", np.nan),
                    "max_dd": res.metrics.get("Max Drawdown", np.nan),
                    "total_costs": res.metrics.get("Total Costs", np.nan),
                }
            )

    # --------------------------------------------------------
    # 3) Conversion en DataFrame
    # --------------------------------------------------------
    results_df = pd.DataFrame(rows)

    # Si aucune combinaison valide n'a été trouvée
    if results_df.empty:
        return results_df, pd.DataFrame()

    # --------------------------------------------------------
    # 4) Choix du critère pour la matrice pivot
    # --------------------------------------------------------
    if criterion not in {"sharpe", "cumret"}:
        criterion = "sharpe"

    # Construction d'une matrice :
    # lignes = fenêtres longues
    # colonnes = fenêtres courtes
    # valeurs = critère choisi
    pivot = results_df.pivot_table(
        index="long",
        columns="short",
        values=criterion,
        aggfunc="mean"
    )

    return results_df, pivot