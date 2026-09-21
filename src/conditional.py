"""Conditional VaR: GARCH-filtered EVT (McNeil & Frey, 2000).

The three models in calibrate.py are UNCONDITIONAL: one VaR number for calm
and storm alike. Their breaches therefore cluster in volatile spells, and the
Christoffersen test catches it. The fix splits the problem in two:

  1. GARCH(1,1) describes HOW VOLATILE tomorrow is: sigma_t, forecast from
     information up to yesterday.
  2. EVT describes the SHAPE of the tail of the standardised shocks
     z_t = (r_t - mu) / sigma_t, which should be close to i.i.d. once the
     volatility clustering has been divided out.

Daily VaR is then  VaR_t = -mu + sigma_t * q_z(p),  a fixed tail quantile
scaled by a moving volatility.

GARCH is fitted by Gaussian quasi-maximum likelihood (QMLE): the Normal
likelihood gives consistent GARCH parameters even when the true shocks are
fat-tailed; the fat tail is then left for EVT to model on the residuals.
"""
import json

import numpy as np
import pandas as pd
from scipy import optimize, stats

import config as C
from calibrate import fit_gpd_pot, var_es_gpd

CONDITIONAL_CSV = C.DATA / "conditional_var.csv"


def garch_filter(r, mu, omega, alpha, beta):
    """One-step-ahead conditional variances. sigma2[t] uses returns up to t-1."""
    eps = r - mu
    sigma2 = np.empty_like(r)
    sigma2[0] = eps.var()                      # standard initialisation
    for t in range(1, len(r)):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


def fit_garch_qmle(r):
    """Gaussian QMLE for GARCH(1,1) with constant mean."""
    var0 = r.var()

    def nll(theta):
        mu, omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.9999:
            return 1e10
        s2 = garch_filter(r, mu, omega, alpha, beta)
        return 0.5 * np.sum(np.log(s2) + (r - mu) ** 2 / s2)

    x0 = [r.mean(), var0 * 0.05, 0.05, 0.90]
    res = optimize.minimize(nll, x0, method="Nelder-Mead",
                            options={"xatol": 1e-10, "fatol": 1e-8,
                                     "maxiter": 20_000, "maxfev": 20_000})
    mu, omega, alpha, beta = res.x
    return {"mu": mu, "omega": omega, "alpha": alpha, "beta": beta,
            "converged": bool(res.success)}


def main():
    df = pd.read_csv(C.RETURNS_CSV)
    r = df["log_return"].to_numpy()

    g = fit_garch_qmle(r)
    sigma = np.sqrt(garch_filter(r, g["mu"], g["omega"], g["alpha"], g["beta"]))
    z_loss = -(r - g["mu"]) / sigma            # standardised shocks, loss side

    gpd = fit_gpd_pot(z_loss)                  # EVT on the residuals, not the raw losses
    # Excess kurtosis of the residuals: raw returns carry clustering + fat tails,
    # residuals should carry only the fat tails.
    kurt_raw = float(stats.kurtosis(r))
    kurt_z = float(stats.kurtosis(z_loss))

    with open(C.RESULTS) as f:
        results = json.load(f)
    results["models"]["garch_evt"] = {"garch": g, "gpd_residuals": gpd,
                                      "excess_kurtosis_raw": kurt_raw,
                                      "excess_kurtosis_residuals": kurt_z}

    out = pd.DataFrame({"date": df["date"], "sigma": sigma})
    print(f"GARCH(1,1) QMLE: omega={g['omega']:.2e}  alpha={g['alpha']:.3f}  "
          f"beta={g['beta']:.3f}  (true {C.TRUE_OMEGA:.1e} / {C.TRUE_ALPHA} / "
          f"{C.TRUE_BETA})  converged={g['converged']}")
    print(f"Residual GPD: xi={gpd['xi']:+.3f}  (true t{C.TRUE_NU:.0f} tail -> xi="
          f"{1 / C.TRUE_NU:.3f})")
    print(f"Excess kurtosis: raw returns {kurt_raw:.2f} -> residuals {kurt_z:.2f}")
    for p in C.ALPHAS:
        zq, zes = var_es_gpd(p, **gpd)
        out[f"VaR_{p}"] = -g["mu"] + sigma * zq
        out[f"ES_{p}"] = -g["mu"] + sigma * zes
        results["models"]["garch_evt"][f"z_quantile_{p}"] = {"VaR": zq, "ES": zes}
        print(f"  alpha={p:.0%}: z-quantile {zq:.3f}; daily VaR ranges "
              f"{out[f'VaR_{p}'].min():.4f} .. {out[f'VaR_{p}'].max():.4f}")

    out.to_csv(CONDITIONAL_CSV, index=False)
    with open(C.RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {CONDITIONAL_CSV.name} and updated {C.RESULTS.name}")


if __name__ == "__main__":
    main()
