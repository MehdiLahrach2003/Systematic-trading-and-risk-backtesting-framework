# scripts/run_delta_hedge.py
# ------------------------------------------------------------
# Ce script simule une couverture en delta d'un call européen
# sur une seule trajectoire de marché.
#
# Objectif :
# visualiser comment évoluent :
# - le spot
# - la valeur du call
# - la valeur du portefeuille de couverture
# ------------------------------------------------------------

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Permet d'importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pricing.black_scholes import bs_call_price, bs_call_delta


# ------------------------------------------------------------
# 1) Simulation d'une trajectoire GBM
# ------------------------------------------------------------
def simulate_gbm_path(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simule une seule trajectoire de GBM.

    Retour :
    - t : grille de temps
    - S : trajectoire du spot
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    S = np.empty(n_steps + 1, dtype=float)
    S[0] = S0

    for i in range(1, n_steps + 1):
        z = rng.standard_normal()

        # Pas exact du GBM
        S[i] = S[i - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        )

    return t, S


# ------------------------------------------------------------
# 2) Couverture en delta sur une trajectoire
# ------------------------------------------------------------
def delta_hedge_path(
    S: np.ndarray,
    t: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> dict:
    """
    Simule le PnL d'une couverture en delta d'un call européen vendu.

    Hypothèse :
    - on vend 1 call au temps 0
    - on reçoit la prime
    - on couvre le risque avec le delta BS
    - on ignore les intérêts sur le cash pour simplifier
    """
    n_steps = len(S) - 1

    option_values = np.zeros_like(S)
    deltas = np.zeros_like(S)
    cash = np.zeros_like(S)
    stock_pos = np.zeros_like(S)
    portfolio_value = np.zeros_like(S)

    # ------------------------------------------------------------
    # t = 0 : on vend le call
    # ------------------------------------------------------------
    tau0 = T - t[0]

    # Prix initial du call
    C0 = bs_call_price(S[0], K, r, sigma, tau0)
    option_values[0] = C0

    # On reçoit la prime en cash
    cash[0] = C0
    stock_pos[0] = 0.0

    # Delta initial du call
    deltas[0] = bs_call_delta(S[0], K, r, sigma, tau0)

    # On achète Delta actions pour couvrir le call vendu
    trade = deltas[0] - stock_pos[0]
    cash[0] -= trade * S[0]
    stock_pos[0] = deltas[0]

    # Valeur du portefeuille
    portfolio_value[0] = cash[0] + stock_pos[0] * S[0]

    # ------------------------------------------------------------
    # Rééquilibrage au cours du temps
    # ------------------------------------------------------------
    for i in range(1, n_steps + 1):
        tau = max(T - t[i], 0.0)

        if tau > 0.0:
            # Avant maturité : prix et delta BS
            option_values[i] = bs_call_price(S[i], K, r, sigma, tau)
            deltas[i] = bs_call_delta(S[i], K, r, sigma, tau)
        else:
            # À maturité : valeur du call = payoff
            option_values[i] = max(S[i] - K, 0.0)
            deltas[i] = 0.0

        # Ajustement de la position en actions
        trade = deltas[i] - stock_pos[i - 1]

        # Mise à jour du cash (sans intérêts)
        cash[i] = cash[i - 1] - trade * S[i]

        # Nouvelle position en actions
        stock_pos[i] = stock_pos[i - 1] + trade

        # Valeur du portefeuille
        portfolio_value[i] = cash[i] + stock_pos[i] * S[i]

    # ------------------------------------------------------------
    # Fin : comparaison avec le payoff
    # ------------------------------------------------------------
    payoff = max(S[-1] - K, 0.0)
    hedge_error = portfolio_value[-1] - payoff

    return {
        "option_values": option_values,
        "deltas": deltas,
        "cash": cash,
        "stock_pos": stock_pos,
        "portfolio_value": portfolio_value,
        "payoff": payoff,
        "hedge_error": hedge_error,
    }


# ------------------------------------------------------------
# 3) Script principal
# ------------------------------------------------------------
def main():
    # Paramètres du modèle
    S0 = 100.0
    K = 100.0
    r = 0.01
    sigma = 0.20
    T = 1.0
    n_steps = 100

    # 1) Simulation d'une trajectoire GBM
    t, S = simulate_gbm_path(S0, r, sigma, T, n_steps, seed=42)

    # 2) Couverture en delta
    res = delta_hedge_path(S, t, K, r, sigma, T)

    option_values = res["option_values"]
    portfolio_value = res["portfolio_value"]
    hedge_error = res["hedge_error"]
    payoff = res["payoff"]

    print("\n===== Delta hedge result (single path) =====")
    print(f"Final spot S_T      : {S[-1]:.4f}")
    print(f"Call payoff         : {payoff:.4f}")
    print(f"Final portfolio     : {portfolio_value[-1]:.4f}")
    print(f"Hedging error       : {hedge_error:.4f}")

    # ------------------------------------------------------------
    # 3) Tracé des résultats
    # ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Spot
    ax = axes[0]
    ax.plot(t, S, label="Spot S_t")
    ax.axhline(K, color="grey", ls="--", lw=1, label="Strike K")
    ax.set_ylabel("Spot")
    ax.set_title("Delta-hedging simulation (single path)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Valeur du call
    ax = axes[1]
    ax.plot(t, option_values, label="Call value")
    ax.set_ylabel("Option price")
    ax.legend()
    ax.grid(alpha=0.3)

    # Portefeuille de couverture
    ax = axes[2]
    ax.plot(t, portfolio_value, label="Hedging portfolio value")
    ax.axhline(payoff, color="red", ls="--", lw=1.2, label="Call payoff at T")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "delta_hedge_single_path.png"
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Delta-hedge plot saved → {out_png}")

    plt.show()


if __name__ == "__main__":
    main()