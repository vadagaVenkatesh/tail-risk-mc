# tail-risk-mc

Monte Carlo simulation for **tail risk**, with a hands-on study of the gap
between **calibration** (fitting a model to data you've seen) and **reality**
(the losses the world actually delivers).

The project generates a known fat-tailed, volatility-clustering returns series,
calibrates three risk models to it, estimates Value-at-Risk (VaR) and Expected
Shortfall (ES) by Monte Carlo, and then *backtests* each calibration against the
realised data. The full write-up compiles to a PDF in [`notes/`](notes/).

## Layout

```
tail-risk-mc/
├── README.md
├── requirements.txt
├── src/
│   ├── config.py         # paths, RNG seed, risk levels, the "true" process
│   ├── generate_data.py  # GARCH(1,1)-t data generator  -> data/returns.csv
│   ├── calibrate.py      # MLE fits: Normal, Student-t, EVT/GPD -> results.json
│   ├── simulate.py       # Monte Carlo VaR & ES (+ bootstrap std error)
│   ├── backtest.py       # Kupiec proportion-of-failures test vs reality
│   └── plots.py          # regenerates every figure used in the notes
├── data/                 # generated dataset + results.json (calibration output)
├── figures/              # PDF figures embedded in the LaTeX
└── notes/
    └── tail_risk_notes.tex  # the notes  ->  tail_risk_notes.pdf
```

## Run the pipeline

```bash
pip install -r requirements.txt
cd src
python3 generate_data.py   # the "reality"
python3 calibrate.py       # fit the three models
python3 simulate.py        # Monte Carlo VaR / ES
python3 backtest.py        # calibration vs reality
python3 plots.py           # figures
```

Everything is seeded in `src/config.py`, so results reproduce exactly.

## Build the notes (PDF)

Requires a TeX distribution (MacTeX). With the **LaTeX Workshop** VS Code
extension, just open `notes/tail_risk_notes.tex` and save (build on save), or:

```bash
cd notes
latexmk -pdf tail_risk_notes.tex
```

## The one-line takeaway

Calibration always *succeeds* — it returns the best parameters **within the
family you assumed**. Whether that family can describe reality's tails is a
separate question, answered only by backtesting. Here the Normal model fits the
body of the data yet is statistically **rejected** in the tail, while the
Student-t and EVT models pass. The map is not the territory.

## Result: Kupiec backtest

3,000 realised days, each model's VaR checked against the same series. "Reject" means
the breach count is statistically inconsistent with the promised level (p < 0.05).

| Level | Model | VaR | Breaches (expected) | Breach rate | Kupiec p | Verdict |
|---|---|---:|---:|---:|---:|---|
| 95% | Normal    | 1.71% | 100 (150) | 3.33% | <0.001 | **Reject** (too cautious) |
| 95% | Student-t | 1.41% | 148 (150) | 4.93% | 0.867 | Pass |
| 95% | EVT / GPD | 1.39% | 150 (150) | 5.00% | 1.000 | Pass |
| 99% | Normal    | 2.45% | 42 (30)   | 1.40% | 0.038 | **Reject** (understates risk) |
| 99% | Student-t | 3.23% | 23 (30)   | 0.77% | 0.180 | Pass |
| 99% | EVT / GPD | 2.83% | 29 (30)   | 0.97% | 0.854 | Pass |

The Normal model fails in both directions. Matching the variance of fat-tailed data
forces it to overstate the moderate tail and understate the extreme one. It gets
the *shape* wrong, not just the scale.

**Caveats.** This is an in-sample backtest: every model was fitted to the same 3,000
days it is scored on. That flatters the fitted models, and the GPD's perfect 150/150 at
95% happens almost by construction, because its threshold sits at the 90th
percentile of this sample. Kupiec also counts only *how many* breaches occur, not
*when*. The data comes from a GARCH process, and none of the three unconditional
models tracks volatility clustering, so an independence test (Christoffersen)
is the next check.

Reproduce with `python3 backtest.py`. Full numbers are in `data/results.json`.

## Experiment

Edit the "reality" in `src/config.py` (`TRUE_NU`, `TRUE_BETA`, the confidence
levels) and re-run the pipeline to see how calibration tracks — or fails to
track — a different world.
