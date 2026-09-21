"""Produce the figures used in the LaTeX notes.

Each figure makes one idea visible:
  fig_returns       -- volatility clustering in the raw series.
  fig_density       -- Normal vs Student-t fit; the tail gap on a log scale.
  fig_qq            -- a Normal QQ-plot bends at the tails => non-Gaussian.
  fig_mc_converge   -- Monte Carlo VaR converging as N grows (+/- std-error band).
  fig_breaches      -- where each model's 99% VaR was breached by reality.
  fig_conditional   -- GARCH-EVT 95% VaR moving with volatility vs static Student-t.
  fig_clustering    -- (README, PNG) Student-t 95% breaches: right count, wrong timing.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config as C

plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3})


def _load():
    df = pd.read_csv(C.RETURNS_CSV, parse_dates=["date"])
    with open(C.RESULTS) as f:
        results = json.load(f)
    return df, results


def fig_returns(df):
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
    ax[0].plot(df["date"], df["log_return"], lw=0.5, color="#1f3b73")
    ax[0].set_ylabel("log-return")
    ax[0].set_title("Daily returns: calm and storm cluster in time")
    ax[1].plot(df["date"], df["true_sigma"], lw=0.7, color="#b3331a")
    ax[1].set_ylabel("conditional $\\sigma_t$")
    ax[1].set_xlabel("date")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_returns.pdf")
    plt.close(fig)


def fig_density(df, results):
    losses = -df["log_return"].to_numpy()
    nrm = results["models"]["normal"]
    st = results["models"]["student_t"]
    xs = np.linspace(losses.min(), losses.max(), 600)

    fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.4))
    ax[0].hist(losses, bins=80, density=True, color="#ccd3e0",
               edgecolor="none", label="empirical")
    ax[0].plot(xs, stats.norm.pdf(xs, nrm["mu"], nrm["sigma"]),
               color="#b3331a", lw=1.6, label="Normal fit")
    ax[0].plot(xs, stats.t.pdf(xs, st["nu"], st["loc"], st["scale"]),
               color="#1f3b73", lw=1.6, label="Student-t fit")
    ax[0].set_title("Loss density")
    ax[0].set_xlabel("loss"); ax[0].legend(fontsize=8)

    # Right tail on a log scale -- where the models diverge.
    ax[1].hist(losses, bins=80, density=True, color="#ccd3e0", edgecolor="none")
    ax[1].plot(xs, stats.norm.pdf(xs, nrm["mu"], nrm["sigma"]),
               color="#b3331a", lw=1.6, label="Normal")
    ax[1].plot(xs, stats.t.pdf(xs, st["nu"], st["loc"], st["scale"]),
               color="#1f3b73", lw=1.6, label="Student-t")
    ax[1].set_yscale("log")
    ax[1].set_xlim(np.quantile(losses, 0.95), losses.max() * 1.02)
    ax[1].set_title("Right tail (log scale)")
    ax[1].set_xlabel("loss"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_density.pdf")
    plt.close(fig)


def fig_qq(df):
    losses = -df["log_return"].to_numpy()
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    stats.probplot(losses, dist="norm", plot=ax)
    ax.get_lines()[0].set(marker="o", ms=2.5, color="#1f3b73", alpha=0.6)
    ax.get_lines()[1].set(color="#b3331a", lw=1.5)
    ax.set_title("Normal QQ-plot: tails bend off the line")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_qq.pdf")
    plt.close(fig)


def fig_mc_converge(results):
    st = results["models"]["student_t"]
    p = 0.99
    rng = np.random.default_rng(C.SEED + 7)
    big = stats.t.rvs(st["nu"], loc=st["loc"], scale=st["scale"],
                      size=400_000, random_state=rng)
    analytic = results["analytic_var_es"][f"{p}"]["student_t"]["VaR"]

    ns = np.unique(np.logspace(2.3, 5.6, 40).astype(int))
    est = np.array([np.quantile(big[:n], p) for n in ns])
    # crude +/- band from asymptotic quantile variance
    dens = stats.t.pdf(stats.t.ppf(p, st["nu"]), st["nu"]) / st["scale"]
    se = np.sqrt(p * (1 - p) / ns) / dens

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.fill_between(ns, est - 2 * se, est + 2 * se, color="#1f3b73", alpha=0.15,
                    label="$\\pm 2$ s.e. band")
    ax.plot(ns, est, color="#1f3b73", lw=1.2, label="MC VaR estimate")
    ax.axhline(analytic, color="#b3331a", lw=1.4, ls="--", label="analytic VaR")
    ax.set_xscale("log")
    ax.set_xlabel("number of Monte Carlo paths $N$")
    ax.set_ylabel("99% VaR")
    ax.set_title("Monte Carlo VaR converges as $1/\\sqrt{N}$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_mc_converge.pdf")
    plt.close(fig)


def fig_breaches(df, results):
    losses = -df["log_return"].to_numpy()
    p = 0.99
    var_n = results["analytic_var_es"][f"{p}"]["normal"]["VaR"]
    var_t = results["analytic_var_es"][f"{p}"]["student_t"]["VaR"]
    dates = df["date"]

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.plot(dates, losses, lw=0.4, color="#888", label="realised loss")
    ax.axhline(var_n, color="#b3331a", lw=1.3, label=f"Normal 99% VaR={var_n:.3f}")
    ax.axhline(var_t, color="#1f3b73", lw=1.3, label=f"Student-t 99% VaR={var_t:.3f}")
    br_n = losses > var_n
    ax.scatter(dates[br_n], losses[br_n], s=10, color="#b3331a", zorder=3,
               label=f"Normal breaches ({br_n.sum()})")
    ax.set_title("99% VaR breaches: the thin-tailed model is breached too often")
    ax.set_ylabel("loss"); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_breaches.pdf")
    plt.close(fig)


def fig_clustering(df, results):
    """README figure: the Student-t 95% VaR passes Kupiec, yet its breaches
    arrive in bursts -- the Christoffersen failure, made visible."""
    losses = -df["log_return"].to_numpy()
    bt = results["backtest"]["0.95"]["student_t"]
    var_t = bt["VaR"]
    dates = pd.to_datetime(df["date"])
    br = losses > var_t

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(dates, losses, lw=0.4, color="#888", label="realised loss")
    ax.axhline(var_t, color="#1f3b73", lw=1.2, label=f"Student-t 95% VaR = {var_t:.4f}")
    ax.scatter(dates[br], losses[br], s=9, color="#b3331a", zorder=3,
               label=f"breaches: {br.sum()} (expected 150)")
    p_ind = "<0.001" if bt["p_ind"] < 0.001 else f"={bt['p_ind']:.3f}"
    ax.set_title(f"Student-t 95% VaR: passes Kupiec (p={bt['p_value']:.2f}), "
                 f"fails Christoffersen (p{p_ind})\n"
                 f"P(breach | breach yesterday) = {bt['pi11']:.1%}  vs  "
                 f"P(breach | none yesterday) = {bt['pi01']:.1%}", fontsize=9)
    ax.set_ylabel("daily loss"); ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_clustering.png", dpi=150)
    plt.close(fig)


def fig_conditional(df, results):
    """Conditional vs static 95% VaR: the moving line tracks the storms."""
    losses = -df["log_return"].to_numpy()
    cond = pd.read_csv(C.DATA / "conditional_var.csv")
    dates = pd.to_datetime(df["date"])
    v_c = cond["VaR_0.95"].to_numpy()
    v_t = results["backtest"]["0.95"]["student_t"]["VaR"]
    br = losses > v_c
    bt = results["backtest"]["0.95"]["garch_evt"]

    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.plot(dates, losses, lw=0.4, color="#888", label="realised loss")
    ax.axhline(v_t, color="#1f3b73", lw=1.0, ls="--", label=f"static Student-t VaR = {v_t:.4f}")
    ax.plot(dates, v_c, lw=0.9, color="#2e7d32", label="GARCH-EVT VaR (daily)")
    ax.scatter(dates[br], losses[br], s=6, color="#b3331a", zorder=3,
               label=f"GARCH-EVT breaches: {br.sum()}")
    ax.set_title(f"95% VaR that moves with volatility: breaches no longer cluster "
                 f"(P(b|b) = {bt['pi11']:.1%} vs {bt['pi01']:.1%})", fontsize=9)
    ax.set_ylabel("loss"); ax.legend(fontsize=7, loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(C.FIGURES / "fig_conditional.pdf")
    plt.close(fig)


def main():
    df, results = _load()
    fig_returns(df)
    fig_density(df, results)
    fig_qq(df)
    fig_mc_converge(results)
    fig_breaches(df, results)
    fig_clustering(df, results)
    fig_conditional(df, results)
    print(f"Wrote 5 figures -> {C.FIGURES}")


if __name__ == "__main__":
    main()
