"""Season-to-season development as a longitudinal mixed-effects model.

For each per-90 metric, player-seasons are modelled as

    rate ~ years + position_group + (1 + years | player)

The random intercept captures stable differences between players; the random slope captures
players developing at different speeds. The intraclass correlation (ICC) from the
random-intercept model says how much of a season's number reflects the player rather than
season-to-season noise.
"""
from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from . import config


def _fit(formula: str, data: pd.DataFrame, re_formula: str | None = None):
    """REML fit; tries several optimisers because L-BFGS can hit a singular matrix on these data."""
    model = smf.mixedlm(formula, data, groups=data["player_id"], re_formula=re_formula)
    last_error = None
    for method in ("bfgs", "powell", "nm"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fit = model.fit(reml=True, method=method)
            except (np.linalg.LinAlgError, ValueError) as err:
                last_error = err
                continue
        issues = [str(w.message).split("\n")[0] for w in caught if "converge" in str(w.message).lower()
                  or "singular" in str(w.message).lower() or "boundary" in str(w.message).lower()]
        if fit.converged:
            return fit, issues
    raise RuntimeError(f"Mixed model did not converge: {last_error}")


def fit_metric(panel: pd.DataFrame, metric: str) -> dict:
    col = f"{metric}_p90"
    data = panel[["player_id", "years", "position_group", col]].rename(columns={col: "rate"})
    formula = "rate ~ years + C(position_group)"
    intercept_only, issues_1 = _fit(formula, data)
    slopes, issues_2 = _fit(formula, data, re_formula="~years")
    var_player = float(intercept_only.cov_re.iloc[0, 0])
    var_resid = float(intercept_only.scale)
    ci = intercept_only.conf_int().loc["years"]
    lr = max(2 * (slopes.llf - intercept_only.llf), 0.0)
    # Variance components on the boundary: p-value from a 50:50 mixture of chi2(1) and chi2(2).
    p_slope = 0.5 * stats.chi2.sf(lr, 1) + 0.5 * stats.chi2.sf(lr, 2)
    mean_rate = float(data.rate.mean())
    return {
        "metric": metric, "label": config.RATE_METRICS[metric],
        "player_seasons": len(data), "players": int(data.player_id.nunique()),
        "players_2plus_seasons": int((data.groupby("player_id").size() >= 2).sum()),
        "mean_per90": mean_rate,
        "change_per_year": float(intercept_only.params["years"]),
        "change_per_year_low": float(ci.iloc[0]), "change_per_year_high": float(ci.iloc[1]),
        "change_per_year_pct": float(intercept_only.params["years"] / mean_rate * 100),
        "between_player_sd": var_player ** 0.5, "within_player_sd": var_resid ** 0.5,
        "icc": var_player / (var_player + var_resid),
        "slope_sd": float(slopes.cov_re.loc["years", "years"]) ** 0.5 if "years" in slopes.cov_re.index else np.nan,
        "slope_lr_p_value": float(p_slope),
        "warnings": "; ".join(sorted(set(issues_1 + issues_2))),
    }


def run(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([fit_metric(panel, m) for m in config.RATE_METRICS])


def plot_trajectories(panel: pd.DataFrame, path, metrics=("progressive_passes", "pressures")) -> None:
    seasons_per_player = panel.groupby("player_id").size()
    keep = panel[panel.player_id.isin(seasons_per_player[seasons_per_player >= 3].index)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.3), sharex=True)
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        col = f"{metric}_p90"
        for _, g in keep.groupby("player_id"):
            ax.plot(g.years + 2018, g[col], color="#1c5d99", alpha=0.18, lw=1)
        means = panel.groupby("years")[col].mean()
        ax.plot(means.index + 2018, means.values, color="#e07a1f", lw=2.5, marker="o", label="all players (mean)")
        ax.set_title(f"{config.RATE_METRICS[metric]} per 90")
        ax.set_xticks(sorted(panel.years.unique() + 2018))
        ax.set_xticklabels([config.SEASONS[s] for s in sorted(config.SEASONS, key=config.SEASON_START_YEAR.get)],
                           rotation=20, fontsize=8)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Season-to-season trajectories ({keep.player_id.nunique()} players with 3+ seasons of 900+ minutes)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save(summary: pd.DataFrame, panel: pd.DataFrame) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary.round(4).to_csv(config.REPORTS_DIR / "trajectories_mixed_models.csv", index=False)
    plot_trajectories(panel, config.FIGURES_DIR / "trajectories.png")
