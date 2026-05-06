# pricing/delta_hedge_backtest.py
# ------------------------------------------------------------
# Ce fichier réalise un backtest de couverture en delta
# d'un call européen.
#
# Idée :
# - on simule plusieurs trajectoires GBM du sous-jacent
# - on couvre dynamiquement le call avec son delta
# - on tient compte des coûts de transaction
# - on regarde le PnL final de la couverture
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.special import erf


# ------------------------------------------------------------
# 1) Briques Black-Scholes
# ------------------------------------------------------------

def _N(x):
    """
    Fonction de répartition de la loi normale standard.
    """
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _d1_d2(S, K, r, sigma, T):
    """
    Calcule d1 et d2 de Black-Scholes.
    Version compatible avec les arrays numpy.
    """
    S = np.asarray(S, dtype=float)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_call_price(S, K, r, sigma, T):
    """
    Prix du call européen dans Black-Scholes.
    """
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * _N(d1) - K * np.exp(-r * T) * _N(d2)


def bs_call_delta(S, K, r, sigma, T):
    """
    Delta du call européen.
    """
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return _N(d1)


# ------------------------------------------------------------
# 2) Simulation GBM
# ------------------------------------------------------------

def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=42):
    """
    Simule n_paths trajectoires GBM avec n_steps pas.

    Retour :
    array de taille (n_paths, n_steps+1)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    # Bruits gaussiens
    Z = rng.standard_normal((n_paths, n_steps))

    # Incréments log-normaux
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Construction des log-prix
    log_S = np.cumsum(increments, axis=1)

    # Ajout du temps initial
    log_S = np.hstack([np.zeros((n_paths, 1)), log_S])

    # Prix simulés
    S = S0 * np.exp(log_S)
    return S


# ------------------------------------------------------------
# 3) Résumé statistique du hedging
# ------------------------------------------------------------

@dataclass
class HedgeStats:
    mean: float
    std: float
    p5: float
    p95: float


# ------------------------------------------------------------
# 4) Backtest de delta hedging
# ------------------------------------------------------------

def delta_hedge_call_paths(S0, K, r, sigma, T,
                           n_steps=252, n_paths=2000,
                           trans_cost_bps=0.0, seed=42):
    """
    Backtest du delta hedging discret sur un call européen.

    Le PnL final est :
        portefeuille répliquant - payoff

    Paramètres
    ----------
    trans_cost_bps : coût de transaction en basis points
    à chaque réajustement.
    """
    dt = T / n_steps

    # Simulation des trajectoires du sous-jacent
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)

    # ------------------------------------------------------------
    # Initialisation du portefeuille de couverture
    # ------------------------------------------------------------
    S_init = paths[:, 0]
    tau_init = T

    # Delta initial
    delta = bs_call_delta(S_init, K, r, sigma, tau_init)

    # Prix initial du call
    price0 = bs_call_price(S_init, K, r, sigma, tau_init)

    # Cash initial du portefeuille
    cash = price0 - delta * S_init

    # Coût de transaction exprimé en taux
    tc_rate = trans_cost_bps / 1e4

    # ------------------------------------------------------------
    # Boucle de rééquilibrage
    # ------------------------------------------------------------
    for t in range(1, n_steps + 1):
        tau = T - t * dt

        # Le cash rapporte le taux sans risque
        cash *= np.exp(r * dt)

        S_t = paths[:, t]

        # Nouveau delta
        if tau > 0:
            new_delta = bs_call_delta(S_t, K, r, sigma, tau)
        else:
            # À maturité : delta ~ 1 si option dans la monnaie, sinon 0
            new_delta = (S_t > K).astype(float)

        # Variation du hedge
        d_delta = new_delta - delta

        # Coûts de transaction
        trade_cost = tc_rate * np.abs(d_delta) * S_t

        # Mise à jour du cash
        cash -= d_delta * S_t
        cash -= trade_cost

        # Mise à jour du delta
        delta = new_delta

    # ------------------------------------------------------------
    # Valeur finale du portefeuille et payoff
    # ------------------------------------------------------------
    S_T = paths[:, -1]

    portfolio_T = delta * S_T + cash
    payoff = np.maximum(S_T - K, 0.0)

    pnl = portfolio_T - payoff

    # ------------------------------------------------------------
    # Statistiques résumées
    # ------------------------------------------------------------
    stats = HedgeStats(
        mean=float(np.mean(pnl)),
        std=float(np.std(pnl, ddof=1)),
        p5=float(np.percentile(pnl, 5)),
        p95=float(np.percentile(pnl, 95)),
    )

    return pnl, stats


# ------------------------------------------------------------
# 5) Démo si exécution directe
# ------------------------------------------------------------

if __name__ == "__main__":
    # Paramètres
    S0 = 100.0
    K = 100.0
    r = 0.02
    sigma = 0.20
    T = 1.0
    n_steps = 252
    n_paths = 5000
    trans_cost_bps = 1.0

    pnl, stats = delta_hedge_call_paths(
        S0, K, r, sigma, T,
        n_steps=n_steps, n_paths=n_paths,
        trans_cost_bps=trans_cost_bps, seed=42
    )

    print(f"\nDelta-hedge call (n_paths={n_paths}, steps={n_steps}, tc={trans_cost_bps} bps)")
    print(f"P&L moyen : {stats.mean: .4f}")
    print(f"Écart-type : {stats.std: .4f}")
    print(f"5e percentile : {stats.p5: .4f}")
    print(f"95e percentile : {stats.p95: .4f}")

    # Histogramme du PnL
    plt.figure(figsize=(7, 4))
    plt.hist(pnl, bins=60)
    plt.axvline(0, linestyle="--")
    plt.title("Delta-hedging P&L distribution (call)")
    plt.xlabel("P&L")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.show()