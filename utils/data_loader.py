# utils/data_loader.py
# -------------------------------------------------------------
# Ce fichier sert à charger des données de prix depuis le dossier /data.
#
# Si les fichiers CSV n'existent pas, il génère des données synthétiques
# pour que le projet puisse quand même fonctionner.
#
# Il gère :
# - le cas d'un seul actif
# - le cas de plusieurs actifs
# -------------------------------------------------------------

import os
from typing import Sequence, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Fonction interne : charger un CSV ou simuler une série de prix
# ---------------------------------------------------------------------
def _load_or_simulate_price(
    csv_path: str,
    start_date: str = "2020-01-01",
    n_days: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Charge un fichier CSV de prix et le transforme en DataFrame standard
    avec une colonne 'price' indexée par les dates.

    Si le fichier n'existe pas, on génère une série synthétique de prix.
    """

    # Si le fichier n'existe pas, on génère des prix artificiels
    if not os.path.exists(csv_path):
        print(f"⚠️  File not found ({os.path.basename(csv_path)}), generating synthetic prices...")

        # Générateur aléatoire avec seed fixe pour reproductibilité
        rng = np.random.default_rng(seed)

        # Paramètres simples de simulation
        S0 = 100.0   # prix initial
        mu = 0.10    # dérive
        sigma = 0.20 # volatilité
        dt = 1.0 / 252.0  # pas journalier

        # Bruit gaussien
        z = rng.normal(size=n_days)

        # Rendements simulés de type GBM
        r = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z

        # Construction de la trajectoire de prix
        s = S0 * np.exp(np.cumsum(r))

        # Index de dates (jours ouvrés)
        idx = pd.bdate_range(start_date, periods=n_days)

        # DataFrame final standardisé
        df = pd.DataFrame({"price": s}, index=idx)
        return df

    # Si le fichier existe, on le lit
    df = pd.read_csv(csv_path)

    # On essaie d'être tolérant sur les noms de colonnes
    cols = {c.lower(): c for c in df.columns}

    # Colonne date :
    # si une colonne "date" existe, on la prend
    # sinon on prend la première colonne
    date_col = cols.get("date", list(df.columns)[0])

    # Colonne prix :
    # on cherche parmi les noms classiques
    price_col = (
        cols.get("close")
        or cols.get("adj close")
        or cols.get("adj_close")
        or cols.get("price")
        or list(df.columns)[-1]
    )

    # Conversion de la date
    df[date_col] = pd.to_datetime(df[date_col])

    # Mise en index temporel et tri chronologique
    df = df.set_index(date_col).sort_index()

    # On renomme la colonne de prix en "price"
    # pour que tout le projet ait toujours le même format
    df = df.rename(columns={price_col: "price"})[["price"]].astype(float)

    return df


# ---------------------------------------------------------------------
# Chargement d'un seul actif
# ---------------------------------------------------------------------
def load_prices(filename: str = "prices.csv") -> pd.DataFrame:
    """
    Charge un fichier de prix unique avec une colonne 'price'.

    Si /data/<filename> n'existe pas, une série synthétique est générée.
    """

    # Chemin du projet
    base_path = os.path.dirname(os.path.dirname(__file__))

    # Chemin complet vers le fichier CSV
    data_path = os.path.join(base_path, "data", filename)

    # Chargement ou simulation
    df = _load_or_simulate_price(
        csv_path=data_path,
        start_date="2020-01-01",
        n_days=1200,
        seed=42,
    )

    return df


# ---------------------------------------------------------------------
# Chargement de plusieurs actifs
# ---------------------------------------------------------------------
def load_multi_assets(symbols: Sequence[str]) -> pd.DataFrame:
    """
    Charge plusieurs actifs depuis le dossier /data et les aligne
    dans un seul DataFrame.

    Chaque actif doit idéalement avoir son propre fichier CSV :
        <SYMBOL>.csv

    Si un fichier manque, des prix synthétiques sont générés.
    """

    # Dossier racine puis dossier data
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_path, "data")

    # Dictionnaire qui contiendra une série par symbole
    series_by_symbol: Dict[str, pd.Series] = {}

    # Seed de base pour que chaque actif simulé soit différent
    base_seed = 1234

    for i, sym in enumerate(symbols):
        # Exemple : data/MSFT.csv
        csv_path = os.path.join(data_dir, f"{sym}.csv")

        # Chargement ou simulation
        df_sym = _load_or_simulate_price(
            csv_path=csv_path,
            start_date="2015-01-01",
            n_days=2000,
            seed=base_seed + i,
        )

        # On garde la colonne "price" et on la renomme avec le nom du symbole
        s = df_sym["price"].rename(sym)
        series_by_symbol[sym] = s

    # Concaténation de toutes les séries dans un seul DataFrame
    df_all = pd.concat(
        [series_by_symbol[sym] for sym in symbols],
        axis=1,
    ).sort_index()

    # On remplit les trous éventuels vers l'avant
    # puis on enlève les NaN restants au début
    df_all = df_all.ffill().dropna()

    return df_all