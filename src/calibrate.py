"""Calibrate candidate risk models to the observed returns.

Calibration = choosing the parameters of a model so that it best matches
the data we have seen. We fit three models of increasing tail-awareness:

  * Normal      -- the textbook default; thin tails, two parameters.
  * Student-t   -- adds a degrees-of-freedom parameter to fatten the tails.
  * EVT (GPD)   -- ignores the body entirely and models ONLY the tail, via
                   the Peaks-Over-Threshold method and the Generalized
                   Pareto Distribution (justified by extreme-value theory).

For each model we also derive the loss-side VaR and Expected Shortfall in
closed form, so we can later compare them against (a) a Monte Carlo estimate
and (b) the empirical reality.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import config as C


def fit_normal(losses):
    mu, sigma = stats.norm.fit(losses)
    return {"mu": mu, "sigma": sigma}


def fit_student_t(losses):
    nu, loc, scale = stats.t.fit(losses)
    return {"nu": nu, "loc": loc, "scale": scale}


def fit_gpd_pot(losses, threshold_q=0.90):
    """Peaks-Over-Threshold: fit a GPD to exceedances above a high quantile."""
    u = np.quantile(losses, threshold_q)
    exceed = losses[losses > u] - u
    xi, _, beta = stats.genpareto.fit(exceed, floc=0.0)
    return {"u": u, "xi": xi, "beta": beta,
            "n": len(losses), "n_exceed": len(exceed),
            "threshold_q": threshold_q}


# --- Closed-form VaR / Expected Shortfall ---------------------------------
def var_es_normal(p, mu, sigma):
    z = stats.norm.ppf(p)
    var = mu + sigma * z
    es = mu + sigma * stats.norm.pdf(z) / (1 - p)
    return var, es


def var_es_student_t(p, nu, loc, scale):
    t_p = stats.t.ppf(p, nu)
    var = loc + scale * t_p
    # ES for the standardised t, scaled and shifted.
    es_std = (stats.t.pdf(t_p, nu) / (1 - p)) * ((nu + t_p**2) / (nu - 1))
    es = loc + scale * es_std
    return var, es


def var_es_gpd(p, u, xi, beta, n, n_exceed, **_):
    """Tail VaR/ES from the POT-GPD fit (McNeil-Frey formulas)."""
    nu_ratio = n / n_exceed
    var = u + (beta / xi) * ((nu_ratio * (1 - p)) ** (-xi) - 1)
    es = var / (1 - xi) + (beta - xi * u) / (1 - xi)
    return var, es


def main():
    df = pd.read_csv(C.RETURNS_CSV)
    # Work in LOSS space: loss = -return. Tail risk lives in the right tail of losses.
    losses = -df["log_return"].to_numpy()

    normal = fit_normal(losses)
    student = fit_student_t(losses)
    gpd = fit_gpd_pot(losses)

    results = {"models": {"normal": normal, "student_t": student, "gpd": gpd},
               "analytic_var_es": {}}

    print("Calibrated parameters")
    print(f"  Normal     : mu={normal['mu']:+.5f}  sigma={normal['sigma']:.5f}")
    print(f"  Student-t  : nu={student['nu']:.2f}  loc={student['loc']:+.5f}  "
          f"scale={student['scale']:.5f}   (true nu = {C.TRUE_NU})")
    print(f"  GPD (POT)  : u={gpd['u']:.4f}  xi={gpd['xi']:+.3f}  beta={gpd['beta']:.5f}  "
          f"(n_exceed={gpd['n_exceed']})")

    print("\nAnalytic VaR / ES (loss space)")
    for p in C.ALPHAS:
        vn, en = var_es_normal(p, **normal)
        vt, et = var_es_student_t(p, **student)
        vg, eg = var_es_gpd(p, **gpd)
        results["analytic_var_es"][f"{p}"] = {
            "normal": {"VaR": vn, "ES": en},
            "student_t": {"VaR": vt, "ES": et},
            "gpd": {"VaR": vg, "ES": eg},
        }
        print(f"  alpha={p:.0%}")
        print(f"     Normal    VaR={vn:.4f}  ES={en:.4f}")
        print(f"     Student-t VaR={vt:.4f}  ES={et:.4f}")
        print(f"     GPD/EVT   VaR={vg:.4f}  ES={eg:.4f}")

    with open(C.RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote calibration results -> {C.RESULTS}")


if __name__ == "__main__":
    main()
