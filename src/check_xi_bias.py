"""Is the low residual tail index a bug, or a known bias?

conditional.py fits a GPD to the GARCH residuals and gets xi ~ 0.12, where a
Student-t(4) tail has xi = 1/4. Before trusting the model we need to know why.

Three candidate explanations:
  * a bug in the fit,
  * sampling noise (only ~300 exceedances),
  * finite-threshold bias: a t tail is Pareto only ASYMPTOTICALLY, so a GPD
    fitted above a moderate threshold sees a tail that is not yet Pareto and
    under-estimates xi.

The discriminating experiment: run the SAME procedure on a huge, clean t(4)
sample, where noise is negligible and there is no GARCH step to get wrong. If
xi still comes out low at the 90th percentile and climbs toward 0.25 as the
threshold rises, it is threshold bias, not a bug.
"""
import numpy as np
from scipy import stats

import config as C

N = 2_000_000
THRESHOLDS = (0.90, 0.95, 0.99, 0.999)


def main():
    nu = C.TRUE_NU
    z = stats.t.rvs(nu, size=N, random_state=C.SEED) / np.sqrt(nu / (nu - 2))
    print(f"Clean t({nu:.0f}) sample, n={N:,}. True xi = 1/nu = {1 / nu:.3f}")
    for q in THRESHOLDS:
        u = np.quantile(z, q)
        exceed = z[z > u] - u
        xi, _, _ = stats.genpareto.fit(exceed, floc=0.0)
        print(f"  threshold q={q:<6} exceedances={len(exceed):>9,}  xi_hat={xi:.3f}")
    n_exc, xi_fit = 300, 0.12
    print(f"\nSampling s.e. of xi at n_exceed={n_exc}: ~(1+xi)/sqrt(n) = "
          f"{(1 + xi_fit) / np.sqrt(n_exc):.3f}")


if __name__ == "__main__":
    main()
