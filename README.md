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

## Experiment

Edit the "reality" in `src/config.py` (`TRUE_NU`, `TRUE_BETA`, the confidence
levels) and re-run the pipeline to see how calibration tracks — or fails to
track — a different world.
