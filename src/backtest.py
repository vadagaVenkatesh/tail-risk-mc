"""Backtesting: confronting the calibration with reality.

A calibrated model gives a VaR number. Reality gives realised losses.
Backtesting counts how often reality breached the model's VaR and asks:
is that breach rate statistically consistent with the level we claimed?

  * At 99% VaR we EXPECT a breach on ~1% of days.
  * Too many breaches -> the model understates risk (dangerous).
  * Too few breaches  -> the model overstates risk (capital-inefficient).

We use the Kupiec POF (proportion-of-failures) likelihood-ratio test.
This is exactly where the Normal model is exposed: it was calibrated to the
same data, yet its thin tails make it breach far more often than promised.
"""
import json

import numpy as np
import pandas as pd
from scipy import stats

import config as C


def kupiec_pof(n, x, p_expected):
    """Kupiec proportion-of-failures LR test.

    n  : number of observations
    x  : number of VaR breaches
    p_expected : expected breach probability (= 1 - alpha)
    Returns (breach_rate, LR_stat, p_value).
    """
    pi = x / n
    if x == 0:
        lr = -2 * n * np.log(1 - p_expected)
    else:
        lr = -2 * (
            (n - x) * np.log(1 - p_expected) + x * np.log(p_expected)
            - (n - x) * np.log(1 - pi) - x * np.log(pi)
        )
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return pi, lr, p_value


def main():
    df = pd.read_csv(C.RETURNS_CSV)
    losses = -df["log_return"].to_numpy()
    n = len(losses)

    with open(C.RESULTS) as f:
        results = json.load(f)

    results["backtest"] = {}
    print(f"Backtest over {n} realised days "
          f"(static one-shot calibration vs. reality)\n")

    for p in C.ALPHAS:
        expected = 1 - p
        results["backtest"][f"{p}"] = {}
        print(f"alpha={p:.0%}  (expected breach rate = {expected:.1%})")
        for model in ("normal", "student_t", "gpd"):
            var = results["analytic_var_es"][f"{p}"][model]["VaR"]
            breaches = int((losses > var).sum())
            rate, lr, pval = kupiec_pof(n, breaches, expected)
            verdict = "PASS" if pval > 0.05 else "REJECT"
            results["backtest"][f"{p}"][model] = {
                "VaR": var, "breaches": breaches, "rate": rate,
                "expected_breaches": expected * n, "LR": lr,
                "p_value": pval, "verdict": verdict,
            }
            print(f"   {model:10s} VaR={var:.4f}  breaches={breaches:3d} "
                  f"(exp {expected*n:5.1f})  rate={rate:.2%}  "
                  f"Kupiec p={pval:.3f}  -> {verdict}")
        print()

    with open(C.RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote backtest results -> {C.RESULTS}")


if __name__ == "__main__":
    main()
