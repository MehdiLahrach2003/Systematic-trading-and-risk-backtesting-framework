# utils/load_multi.py

"""
Charge plusieurs actifs depuis data/multi/ et renvoie un dictionnaire
ticker -> série de prix.
"""

from __future__ import annotations
import os
import pandas as pd


def load_multi_assets() -> dict[str, pd.Series]:
    """
    Charge tous les fichiers CSV présents dans data/multi/.

    Chaque fichier doit contenir au minimum :
        - une colonne 'date'
        - une colonne 'price'

    Retour :
    -------
    dict[str, pd.Series]
        mapping ticker -> série de prix indexée par date
    """

    # --------------------------------------------------------
    # 1) Construire le chemin vers data/multi/
    # --------------------------------------------------------
    root = os.path.dirname(os.path.dirname(__file__))   # racine du projet
    multi_dir = os.path.join(root, "data", "multi")

    # Vérification que le dossier existe
    if not os.path.isdir(multi_dir):
        raise FileNotFoundError(f"Folder not found: {multi_dir}")

    # Dictionnaire final : ticker -> série de prix
    assets = {}

    # --------------------------------------------------------
    # 2) Parcourir tous les fichiers CSV du dossier
    # --------------------------------------------------------
    for fname in os.listdir(multi_dir):

        # On ignore les fichiers non CSV
        if not fname.endswith(".csv"):
            continue

        # Nom du ticker = nom du fichier sans .csv
        ticker = fname.replace(".csv", "")

        # Chemin complet vers le fichier
        path = os.path.join(multi_dir, fname)

        # ----------------------------------------------------
        # 3) Lecture du CSV
        # ----------------------------------------------------
        df = pd.read_csv(path, parse_dates=["date"])

        # Trier les données dans l'ordre chronologique
        df = df.sort_values("date")

        # Mettre la colonne 'date' en index
        df = df.set_index("date")

        # Vérification : la colonne 'price' doit exister
        if "price" not in df:
            raise ValueError(f"CSV {fname} must contain a 'price' column.")

        # Stocker la série de prix (float)
        assets[ticker] = df["price"].astype(float)

    # --------------------------------------------------------
    # 4) Retour : dict {ticker: price_series}
    # --------------------------------------------------------
    return assets