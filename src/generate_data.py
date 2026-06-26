"""Generate a realistic, calibratable returns dataset.

We simulate daily log-returns from a GARCH(1,1) process with
Student-t innovations. This deliberately reproduces the two stylised
facts that make real-market tail risk hard:

  1. Fat tails        -- extreme moves are far more likely than a Gaussian
                         would predict (Student-t with nu = 4).
  2. Volatility clustering -- big moves cluster in time; calm follows calm,
                         storm follows storm (the GARCH variance recursion).

Because we control the true parameters (see config.py), this dataset is the
"reality" against which every calibrated model is later judged.
"""
import numpy as np
import pandas as pd

import config as C


def simulate_garch_t(n, mu, omega, alpha, beta, nu, seed):
    rng = np.random.default_rng(seed)

    # Standardised Student-t innovations (unit variance) so that omega/alpha/beta
    # alone control the conditional variance.
    z = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2.0))

    r = np.empty(n)
    sigma2 = np.empty(n)
    sigma2[0] = omega / (1.0 - alpha - beta)   # unconditional variance
    r[0] = mu + np.sqrt(sigma2[0]) * z[0]

    for t in range(1, n):
        eps_prev = r[t - 1] - mu
        sigma2[t] = omega + alpha * eps_prev**2 + beta * sigma2[t - 1]
        r[t] = mu + np.sqrt(sigma2[t]) * z[t]

    return r, np.sqrt(sigma2)


def main():
    r, sigma = simulate_garch_t(
        C.N_DAYS, C.TRUE_MU, C.TRUE_OMEGA, C.TRUE_ALPHA, C.TRUE_BETA,
        C.TRUE_NU, C.SEED,
    )
    dates = pd.bdate_range("2014-01-01", periods=C.N_DAYS)
    df = pd.DataFrame({"date": dates, "log_return": r, "true_sigma": sigma})
    df.to_csv(C.RETURNS_CSV, index=False)

    print(f"Wrote {len(df)} daily returns -> {C.RETURNS_CSV}")
    print(f"  mean   : {r.mean():+.5f} (true mu = {C.TRUE_MU:+.5f})")
    print(f"  std    : {r.std():.5f}")
    print(f"  skew   : {pd.Series(r).skew():+.3f}")
    print(f"  kurt   : {pd.Series(r).kurt():+.3f}  (excess; Gaussian = 0)")
    print(f"  min/max: {r.min():+.4f} / {r.max():+.4f}")


if __name__ == "__main__":
    main()
