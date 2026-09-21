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

Kupiec only counts breaches; it is blind to *when* they happen. The
Christoffersen independence test asks whether a breach today makes a breach
tomorrow more likely. Under a GARCH reality, a static VaR is breached in
bursts during high-volatility spells, and this is where every unconditional
model -- even the ones that pass Kupiec -- is exposed.
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


def christoffersen_ind(hits):
    """Christoffersen independence LR test on a 0/1 breach sequence.

    Fits a two-state Markov chain to the hits: pi01 = P(breach | no breach
    yesterday), pi11 = P(breach | breach yesterday). Under independence the
    two are equal. Returns (pi01, pi11, LR_ind, p_value), LR_ind ~ chi2(1).
    """
    prev, curr = hits[:-1], hits[1:]
    n00 = int(((prev == 0) & (curr == 0)).sum())
    n01 = int(((prev == 0) & (curr == 1)).sum())
    n10 = int(((prev == 1) & (curr == 0)).sum())
    n11 = int(((prev == 1) & (curr == 1)).sum())
    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(k, n, q):  # k successes out of n with prob q, 0*log(0) := 0
        return (k * np.log(q) if k else 0.0) + ((n - k) * np.log(1 - q) if n - k else 0.0)

    lr = -2 * (ll(n01 + n11, n00 + n01 + n10 + n11, pi)
               - ll(n01, n00 + n01, pi01) - ll(n11, n10 + n11, pi11))
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return pi01, pi11, lr, p_value


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
            hits = (losses > var).astype(int)
            breaches = int(hits.sum())
            rate, lr, pval = kupiec_pof(n, breaches, expected)
            verdict = "PASS" if pval > 0.05 else "REJECT"
            pi01, pi11, lr_ind, p_ind = christoffersen_ind(hits)
            ind_verdict = "PASS" if p_ind > 0.05 else "REJECT"
            lr_cc = lr + lr_ind
            p_cc = 1 - stats.chi2.cdf(lr_cc, df=2)
            results["backtest"][f"{p}"][model] = {
                "VaR": var, "breaches": breaches, "rate": rate,
                "expected_breaches": expected * n, "LR": lr,
                "p_value": pval, "verdict": verdict,
                "pi01": pi01, "pi11": pi11, "LR_ind": lr_ind,
                "p_ind": p_ind, "ind_verdict": ind_verdict,
                "LR_cc": lr_cc, "p_cc": p_cc,
            }
            print(f"   {model:10s} VaR={var:.4f}  breaches={breaches:3d} "
                  f"(exp {expected*n:5.1f})  rate={rate:.2%}  "
                  f"Kupiec p={pval:.3f} -> {verdict:6s}  "
                  f"P(b|b)={pi11:.1%} vs P(b|no b)={pi01:.1%}  "
                  f"Christoffersen p={p_ind:.3f} -> {ind_verdict}")
        print()

    with open(C.RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote backtest results -> {C.RESULTS}")


if __name__ == "__main__":
    main()
