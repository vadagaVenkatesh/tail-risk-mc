"""Shared configuration: paths, RNG seed, and risk conventions.

Keeping these in one place means every stage of the pipeline
(generate -> calibrate -> simulate -> backtest) agrees on the same
data, the same confidence levels, and the same reproducible seed.
"""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "data" / "results.json"

DATA.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

RETURNS_CSV = DATA / "returns.csv"

# --- Reproducibility -------------------------------------------------------
SEED = 20260626

# --- Risk conventions ------------------------------------------------------
# We measure loss-side tail risk at these confidence levels.
ALPHAS = (0.95, 0.99)          # VaR / ES confidence levels
N_PATHS = 200_000              # Monte Carlo sample size
HORIZON_DAYS = 1               # 1-day risk horizon

# --- "Reality": the true data-generating process ---------------------------
# We KNOW the truth here because we generate it. That is the whole point of
# the experiment: it lets us measure how far a calibrated model lands from
# the process that actually produced the data.
TRUE_NU = 4.0                  # Student-t degrees of freedom (fat tails)
TRUE_MU = 0.0005               # daily drift (~12.6% annualised)
TRUE_OMEGA = 2.0e-6            # GARCH(1,1) constant
TRUE_ALPHA = 0.08              # GARCH ARCH coefficient
TRUE_BETA = 0.90              # GARCH GARCH coefficient  (alpha+beta<1 => stationary)
N_DAYS = 3000                  # ~12 trading years of daily data
