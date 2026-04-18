# scripts/run_smile.py
# ------------------------------------------------------------
# Ce script cherche à construire un smile de volatilité implicite
# à partir de prix d'options synthétiques.
#
# Idée :
# - choisir plusieurs strikes
# - générer des prix de calls
# - retrouver la vol implicite associée à chaque prix
# - tracer la vol implicite en fonction du strike
#
# Attention :
# dans la version actuelle, les prix sont générés avec une vol constante,
# donc le smile implicite obtenu devrait être presque plat.
# ------------------------------------------------------------

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Permet d'importer les modules du projet quand on lance ce script directement
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pricing.black_scholes import bs_call_price
from pricing.implied_vol import implied_vol_bs


def build_smile(
    S0: float,
    r: float,
    T: float,
    true_sigma: float,
    n_strikes: int = 17,
):
    """
    Construit une grille de strikes, des prix de calls,
    puis les volatilités implicites associées.
    """

    # Grille de strikes entre 60% et 140% du spot
    K_min, K_max = 0.6 * S0, 1.4 * S0
    strikes = np.linspace(K_min, K_max, n_strikes)

    # Courbe de volatilité "théorique" en forme de smile
    true_vols = 0.20 + 0.3 * ((strikes / S0 - 1.0) ** 2)

    call_prices = np.zeros_like(strikes, dtype=float)
    implied_vols = np.zeros_like(strikes, dtype=float)

    for i, K in enumerate(strikes):
        # ------------------------------------------------------------
        # Prix du call
        # ------------------------------------------------------------
        # ATTENTION :
        # ici le script utilise une volatilité constante true_sigma,
        # et non true_vols[i].
        # Donc les prix sont en réalité ceux d'un modèle BS standard.
        price = bs_call_price(S0, K, r, true_sigma, T)
        call_prices[i] = price

        # Inversion prix -> volatilité implicite
        iv = implied_vol_bs(price, S0, K, r, T) # type: ignore
        implied_vols[i] = iv

    return strikes, true_vols, call_prices, implied_vols


def plot_smile(strikes, true_vols, implied_vols, S0: float):
    """
    Trace :
    - volatilité implicite en fonction du strike
    - volatilité implicite en fonction de la moneyness
    """
    moneyness = strikes / S0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Smile en fonction du strike
    ax = axes[0]
    ax.plot(strikes, implied_vols, "o-", label="Implied vol (from prices)")
    ax.plot(strikes, true_vols, "--", label="True vol (input)")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatility")
    ax.set_title("Volatility smile – σ(K)")
    ax.grid(alpha=0.3)
    ax.legend()

    # Smile en fonction de la moneyness
    ax2 = axes[1]
    ax2.plot(moneyness, implied_vols, "o-", label="Implied vol")
    ax2.axvline(1.0, color="grey", ls="--", lw=0.8)
    ax2.set_xlabel("Moneyness K / S0")
    ax2.set_ylabel("Volatility")
    ax2.set_title("Volatility smile – σ(K/S0)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "vol_smile.png",
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Smile plot saved → {out_png}")

    plt.show()


def main():
    S0 = 100.0
    r = 0.01
    T = 1.0
    true_sigma = 0.20

    strikes, true_vols, call_prices, implied_vols = build_smile(
        S0=S0,
        r=r,
        T=T,
        true_sigma=true_sigma,
        n_strikes=17,
    )

    print("\nStrike   CallPrice   ImpliedVol")
    for K, c, iv in zip(strikes, call_prices, implied_vols):
        print(f"{K:7.2f}   {c:9.4f}   {iv:10.4f}")

    plot_smile(strikes, true_vols, implied_vols, S0=S0)


if __name__ == "__main__":
    main()