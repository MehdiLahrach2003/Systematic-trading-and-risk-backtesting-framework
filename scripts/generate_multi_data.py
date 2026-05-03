# scripts/generate_multi_data.py
# ------------------------------------------------------------
# Objectif :
# Générer des séries de prix synthétiques pour plusieurs actifs
# en utilisant un modèle GBM (Geometric Brownian Motion).
#
# Les données sont ensuite sauvegardées en CSV dans data/multi/
# ------------------------------------------------------------

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# ============================================================
# 1) Simulation GBM
# ============================================================
def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    n_days: int,
    start_date: str = "2015-01-01",
) -> pd.DataFrame:
    """
    Simule une série de prix suivant un mouvement brownien géométrique.

    Paramètres
    ----------
    S0 : prix initial
    mu : drift annuel (rendement moyen)
    sigma : volatilité annuelle
    n_days : nombre de jours simulés

    Retour
    -------
    DataFrame avec colonnes : date, price
    """

    dt = 1.0 / 252.0  # pas journalier (année de trading)
    prices = np.empty(n_days)
    prices[0] = S0

    # Génération des dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start + timedelta(days=i) for i in range(n_days)]

    # Générateur aléatoire
    rng = np.random.default_rng(42)

    # Simulation GBM
    for t in range(1, n_days):
        z = rng.standard_normal()

        prices[t] = prices[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    df = pd.DataFrame({
        "date": dates,
        "price": prices
    })

    return df


# ============================================================
# 2) Génération multi-actifs
# ============================================================
def main():
    # Répertoire de sortie
    root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(root, "data", "multi")
    os.makedirs(out_dir, exist_ok=True)

    n_days = 2000  # environ 8 ans

    # Paramètres par actif
    specs = {
        "AAPL":  {"S0": 150.0, "mu": 0.12, "sigma": 0.25},
        "MSFT":  {"S0": 300.0, "mu": 0.10, "sigma": 0.20},
        "SP500": {"S0": 4000.0, "mu": 0.07, "sigma": 0.18},
        "BTCUSD": {"S0": 20000.0, "mu": 0.25, "sigma": 0.80},
    }

    # Boucle sur chaque actif
    for ticker, params in specs.items():
        df = simulate_gbm(
            S0=params["S0"],
            mu=params["mu"],
            sigma=params["sigma"],
            n_days=n_days,
            start_date="2015-01-01",
        )

        # Sauvegarde CSV
        out_path = os.path.join(out_dir, f"{ticker}.csv")
        df.to_csv(out_path, index=False)

        print(f"[OK] Saved {ticker} to {out_path}")

    print("\nDone. You can now run scripts/run_multi_asset.py")


if __name__ == "__main__":
    main()