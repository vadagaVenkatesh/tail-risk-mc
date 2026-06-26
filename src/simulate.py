"""Monte Carlo estimation of VaR and Expected Shortfall.

Why Monte Carlo at all, when we just derived VaR/ES in closed form?
Because in practice the portfolio loss is a complicated function of many
risk factors (options, path-dependence, multiple assets) for which no
closed form exists. Monte Carlo is the general-purpose hammer:

    1. draw a large sample of losses from the calibrated model,
    2. read VaR off the empirical quantile of the sample,
    3. average the losses beyond VaR to get ES.

Here we apply it to the calibrated Student-t model. We can then check the
MC estimate against the closed-form answer -- a sanity check on the SIMULATION,
separate from the question of whether the MODEL itself is right (backtesting).
"""
import json

import numpy as np
from scipy import stats

import config as C


def mc_var_es(sample, p):
    var = np.quantile(sample, p)
    es = sample[sample >= var].mean()
    return var, es


def mc_standard_error(sample, p, n_boot=400, seed=0):
    """Bootstrap standard error of the VaR estimate -> Monte Carlo noise."""
    rng = np.random.default_rng(seed)
    n = len(sample)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[b] = np.quantile(sample[idx], p)
    return boot.std()


def main():
    with open(C.RESULTS) as f:
        results = json.load(f)
    st = results["models"]["student_t"]

    rng = np.random.default_rng(C.SEED + 1)
    # Draw losses from the calibrated Student-t.
    sample = stats.t.rvs(st["nu"], loc=st["loc"], scale=st["scale"],
                         size=C.N_PATHS, random_state=rng)

    results["mc_var_es"] = {}
    print(f"Monte Carlo with N = {C.N_PATHS:,} paths (calibrated Student-t)\n")
    for p in C.ALPHAS:
        var, es = mc_var_es(sample, p)
        se = mc_standard_error(sample, p, seed=C.SEED + 2)
        analytic = results["analytic_var_es"][f"{p}"]["student_t"]
        results["mc_var_es"][f"{p}"] = {"VaR": var, "ES": es, "VaR_se": se}
        print(f"  alpha={p:.0%}")
        print(f"     MC        VaR={var:.4f} (+/-{se:.4f})  ES={es:.4f}")
        print(f"     Analytic  VaR={analytic['VaR']:.4f}            ES={analytic['ES']:.4f}")
        print(f"     |MC-analytic VaR| = {abs(var-analytic['VaR']):.5f}\n")

    with open(C.RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote MC results -> {C.RESULTS}")


if __name__ == "__main__":
    main()
