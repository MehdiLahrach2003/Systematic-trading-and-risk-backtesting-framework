# scripts/tune_sma.py

# ------------------------------------------------------------
# Objectif :
# faire une recherche sur grille des paramètres de la stratégie SMA,
# sauvegarder les résultats dans un CSV,
# afficher les meilleurs paramètres,
# et tracer une heatmap du Sharpe.
# ------------------------------------------------------------

import os
import sys
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Permet d'importer les modules locaux du projet
# quand on lance ce script directement
# ------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from utils.param_search import evaluate_sma_grid


# ------------------------------------------------------------
# Répertoire de sortie pour sauvegarder les résultats
# ------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def main():
    # --------------------------------------------------------
    # 1) Chargement des prix
    # --------------------------------------------------------
    df = load_prices()

    # --------------------------------------------------------
    # 2) Grilles de paramètres à tester
    # --------------------------------------------------------
    # Fenêtres courtes possibles
    shorts = [5, 10, 20, 30, 40]

    # Fenêtres longues possibles
    longs  = [50, 80, 100, 150, 200]

    # --------------------------------------------------------
    # 3) Recherche sur grille
    # --------------------------------------------------------
    # Pour chaque couple (short, long),
    # on backteste la stratégie SMA correspondante
    # puis on évalue la performance selon le critère choisi.
    #
    # Ici :
    # - critère = Sharpe
    # - coûts de transaction = 1 bp
    # - capital initial = 1.0
    # --------------------------------------------------------
    results_df, pivot = evaluate_sma_grid(
        df,
        shorts,
        longs,
        cost_bps=1.0,
        criterion="sharpe",
        initial_capital=1.0,
    )

    # --------------------------------------------------------
    # 4) Sauvegarde des résultats détaillés
    # --------------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "sma_grid_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"[OK] Results saved to: {out_csv}")

    # --------------------------------------------------------
    # 5) Si la matrice pivot n'est pas vide :
    #    - afficher les meilleurs paramètres
    #    - tracer une heatmap
    # --------------------------------------------------------
    if not pivot.empty:
        print("\n=== Best params (by Sharpe) ===")

        # pivot a en général :
        # - index = long windows
        # - colonnes = short windows
        #
        # pivot.stack() transforme la matrice en série,
        # puis idxmax() donne le couple (long, short)
        best_long, best_short = pivot.stack().idxmax() # type: ignore
        best_val = pivot.max().max()

        print(f"short={best_short}, long={best_long}, sharpe={best_val:.4f}")

        # ----------------------------------------------------
        # 6) Construction de la heatmap
        # ----------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))

        # La heatmap affiche la valeur du Sharpe
        # pour chaque couple (short, long)
        im = ax.imshow(pivot.values, aspect="auto", origin="lower")

        # Axe horizontal = fenêtres courtes
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)

        # Axe vertical = fenêtres longues
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)

        ax.set_xlabel("short window")
        ax.set_ylabel("long window")
        ax.set_title("SMA grid — Sharpe")

        # Barre de couleur
        plt.colorbar(im, ax=ax)

        # Sauvegarde de la heatmap
        out_png = os.path.join(RESULTS_DIR, "sma_grid_heatmap.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f"[OK] Heatmap saved to: {out_png}")

        plt.show()

    else:
        print("[WARN] Empty pivot — check grids or data length.")


if __name__ == "__main__":
    main()