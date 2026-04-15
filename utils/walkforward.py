# utils/walkforward.py

# ------------------------------------------------------------
# Walk-forward SMA :
# - on calibre les paramètres sur une fenêtre d'entraînement
# - on les teste sur la fenêtre suivante
# - on avance dans le temps et on recommence
#
# Objectif :
# vérifier si les paramètres optimisés restent bons hors échantillon
# ------------------------------------------------------------

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable, Tuple, Dict, List
import math
import numpy as np
import pandas as pd

# Permet d'importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest


# ============================================================
# 1) Résultat d'une fenêtre walk-forward
# ============================================================
@dataclass
class WFWindowResult:
    # Début de la fenêtre de test
    start: pd.Timestamp

    # Fin de la fenêtre de test
    end: pd.Timestamp

    # Meilleure fenêtre courte trouvée sur le train
    best_short: int

    # Meilleure fenêtre longue trouvée sur le train
    best_long: int

    # Sharpe obtenu sur la période d'entraînement
    sharpe_in_sample: float

    # Courbe d'equity obtenue sur la période de test
    equity: pd.Series

    # Métriques du backtest sur la période de test
    metrics: Dict[str, float]


# ============================================================
# 2) Résultat global du walk-forward
# ============================================================
@dataclass
class WalkForwardResult:
    # Courbe d'equity hors échantillon reconstruite en concaténant
    # les différents morceaux de test
    equity_oos: pd.Series

    # Liste détaillée des fenêtres
    windows: List[WFWindowResult]

    # Tableau résumant les paramètres choisis et les performances
    params_table: pd.DataFrame


# ============================================================
# 3) Sharpe annualisé
# ============================================================
def _annualize_sharpe(daily_returns: pd.Series, freq_per_year: int = 252) -> float:
    """
    Calcule le Sharpe annualisé à partir des rendements journaliers.
    """
    r = daily_returns.dropna()

    if r.empty or r.std() == 0:
        return 0.0

    return (r.mean() / r.std()) * math.sqrt(freq_per_year)


# ============================================================
# 4) Recherche des meilleurs paramètres sur la fenêtre train
# ============================================================
def _tune_sma_on_train(
    price_train: pd.Series,
    short_grid: Iterable[int],
    long_grid: Iterable[int],
    cost_bps: float,
) -> Tuple[int, int, float]:
    """
    Cherche le meilleur couple (short, long) sur la période d'entraînement.

    Critère :
    - Sharpe Ratio
    """
    best_short, best_long, best_sharpe = None, None, -np.inf

    for s in short_grid:
        for l in long_grid:
            # Fenêtre courte doit être strictement plus petite
            if s >= l:
                continue

            # Construire les positions SMA sur le train
            pos = sma_crossover_positions(price_train, short=s, long=l)

            # Backtest sur le train
            res = run_backtest(price_train, pos, cost_bps)

            # On récupère le Sharpe
            sharpe = res.metrics.get("Sharpe Ratio", _annualize_sharpe(res.returns))

            # Mise à jour du meilleur couple
            if sharpe > best_sharpe:
                best_short, best_long, best_sharpe = s, l, sharpe

    return int(best_short), int(best_long), float(best_sharpe) # type: ignore


# ============================================================
# 5) Walk-forward principal
# ============================================================
def walkforward_sma(
    df: pd.DataFrame,
    short_grid: Iterable[int] = (5, 10, 20, 30),
    long_grid: Iterable[int] = (50, 100, 150, 200),
    train_months: int = 24,
    test_months: int = 3,
    cost_bps: float = 1.0,
) -> WalkForwardResult:
    """
    Procédure walk-forward pour la stratégie SMA.

    Principe :
    1. on prend une fenêtre train
    2. on optimise les paramètres SMA dessus
    3. on applique ces paramètres sur la fenêtre test suivante
    4. on avance dans le temps
    """

    if "price" not in df.columns:
        raise ValueError("DataFrame must contain a 'price' column.")

    # Tri chronologique
    df = df.sort_index()
    price_all = df["price"].astype(float)

    # Petite fonction utilitaire : ajouter des mois à une date
    def add_months(dt: pd.Timestamp, months: int) -> pd.Timestamp:
        return (dt + pd.DateOffset(months=months)).normalize()

    # Bornes globales de l'échantillon
    start, end = price_all.index.min(), price_all.index.max()

    # Curseur de départ
    cursor = start

    # Stockage des résultats
    windows: List[WFWindowResult] = []
    stitched_parts: List[pd.Series] = []

    # Niveau courant pour raccorder les equities OOS
    level = 1.0

    # --------------------------------------------------------
    # Boucle sur les fenêtres walk-forward
    # --------------------------------------------------------
    while True:
        # Définition des fenêtres train / test
        train_start = cursor
        train_end = add_months(train_start, train_months) - pd.Timedelta(days=1)

        test_start = add_months(train_end, 1)
        test_end = add_months(test_start, test_months) - pd.Timedelta(days=1)

        # Si on dépasse la fin des données, on s'arrête
        if test_start > end or train_end > end:
            break

        # Extraction des slices
        train_slice = price_all.loc[train_start:train_end]
        test_slice = price_all.loc[test_start:test_end]

        # Si les fenêtres sont trop petites, on saute
        if len(train_slice) < 30 or len(test_slice) < 5:
            cursor = add_months(cursor, test_months)
            continue

        # ----------------------------------------------------
        # 1) Calibration sur train
        # ----------------------------------------------------
        best_s, best_l, sharpe_is = _tune_sma_on_train(
            train_slice,
            short_grid,
            long_grid,
            cost_bps
        )

        # ----------------------------------------------------
        # 2) Test hors échantillon
        # ----------------------------------------------------
        pos_test = sma_crossover_positions(test_slice, short=best_s, long=best_l)
        res_test = run_backtest(test_slice, pos_test, cost_bps)

        # Equity sur la fenêtre test
        eq = res_test.equity.copy()

        # ----------------------------------------------------
        # 3) Raccord des segments OOS
        # ----------------------------------------------------
        # Chaque segment est renormalisé pour prolonger le segment précédent
        eq = eq / float(eq.iloc[0]) * level
        level = float(eq.iloc[-1])

        stitched_parts.append(eq)

        # Sauvegarde de la fenêtre
        windows.append(
            WFWindowResult(
                start=test_start,
                end=min(test_end, eq.index.max()),
                best_short=best_s,
                best_long=best_l,
                sharpe_in_sample=sharpe_is,
                equity=eq,
                metrics=res_test.metrics,
            )
        )

        # On avance le curseur de la taille d'une fenêtre test
        cursor = add_months(cursor, test_months)

        # Condition supplémentaire d'arrêt
        if add_months(cursor, train_months + test_months) > end + pd.DateOffset(months=1):
            break

    # --------------------------------------------------------
    # Construction de l'equity globale OOS
    # --------------------------------------------------------
    if stitched_parts:
        equity_oos = pd.concat(stitched_parts).groupby(level=0).last()
    else:
        equity_oos = pd.Series(dtype=float)

    # --------------------------------------------------------
    # Tableau résumé des paramètres et perfs
    # --------------------------------------------------------
    params_table = (
        pd.DataFrame(
            [
                {
                    "start": w.start,
                    "end": w.end,
                    "best_short": w.best_short,
                    "best_long": w.best_long,
                    "train_sharpe": w.sharpe_in_sample,
                    "oos_sharpe": w.metrics.get("Sharpe Ratio", np.nan),
                    "oos_cum_return": w.metrics.get("Cumulative Return", np.nan),
                }
                for w in windows
            ]
        ).set_index("start")
        if windows
        else pd.DataFrame()
    )

    # --------------------------------------------------------
    # Retour final
    # --------------------------------------------------------
    return WalkForwardResult(
        equity_oos=equity_oos,
        windows=windows,
        params_table=params_table,
    )