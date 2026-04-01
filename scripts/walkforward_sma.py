# scripts/walkforward_sma.py

# ------------------------------------------------------------
# Point d'entrée principal pour lancer un test walk-forward
# de la stratégie SMA.
#
# Ce script :
# - charge les prix
# - lance une procédure walk-forward
# - affiche les paramètres retenus sur chaque fenêtre
# - trace la courbe d'equity hors échantillon (OOS)
# ------------------------------------------------------------

import sys
import os
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Permet d'importer les modules du projet quand on lance ce script directement
# ------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from utils.walkforward import walkforward_sma


def main():
    # --------------------------------------------------------
    # 1) Chargement des prix
    # --------------------------------------------------------
    # Si aucun CSV n'est trouvé, le loader peut générer des données synthétiques
    df = load_prices()   # on suppose une colonne 'price'

    # --------------------------------------------------------
    # 2) Lancement du walk-forward
    # --------------------------------------------------------
    # Idée :
    # - on prend une fenêtre d'entraînement de 24 mois
    # - on cherche les meilleurs paramètres SMA sur cette fenêtre
    # - on les applique ensuite sur 3 mois de test
    # - puis on avance dans le temps et on recommence
    #
    # short_grid :
    #   fenêtres courtes testées
    # long_grid :
    #   fenêtres longues testées
    # train_months :
    #   taille de la fenêtre d'entraînement
    # test_months :
    #   taille de la fenêtre de test hors échantillon
    # cost_bps :
    #   coûts de transaction
    wf = walkforward_sma(
        df,
        short_grid=(5, 10, 20, 30),
        long_grid=(50, 100, 150, 200),
        train_months=24,
        test_months=3,
        cost_bps=1.0,
    )

    # --------------------------------------------------------
    # 3) Affichage du résumé global
    # --------------------------------------------------------
    print("\n===== Walk-Forward summary =====")

    # params_table contient en général :
    # - la période d'entraînement
    # - la période de test
    # - les meilleurs paramètres choisis sur le train
    # - les performances associées
    if not wf.params_table.empty:
        print(wf.params_table.round(4))
    else:
        print("No windows produced (not enough data?).")

    # --------------------------------------------------------
    # 4) Tracé de l'equity hors échantillon
    # --------------------------------------------------------
    # equity_oos = equity construite uniquement sur les périodes de test
    if not wf.equity_oos.empty:
        plt.figure(figsize=(11, 5))

        plt.plot(wf.equity_oos.index, wf.equity_oos, label="OOS Equity", lw=1.6)

        plt.title("Walk-Forward OOS Equity")
        plt.ylabel("Equity (normalized)")
        plt.grid(alpha=0.3)
        plt.legend()

        # Sauvegarde de la figure dans data/
        out_png = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "wf_equity.png"
        )
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)

        print(f"\n[OK] Walk-forward equity plot saved to: {out_png}\n")

        plt.show()
    else:
        print("No equity to plot.")


if __name__ == "__main__":
    main()