"""Finishing skill as a hierarchical (Gamma-Poisson) model.

Each player's goals are modelled as Poisson(theta * xG), where theta is the player's finishing
multiplier and theta ~ Gamma(alpha, alpha / mu) across players. The league-level parameters
(mu, alpha) are estimated by maximum likelihood on the negative-binomial marginal, and each
player's estimate is the posterior mean: (alpha + goals) / (alpha / mu + xG). Players with little
evidence are pulled towards the league average; players with a lot keep most of their raw ratio.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from . import config


@dataclass
class Prior:
    mu: float      # league-average finishing multiplier
    alpha: float   # concentration: larger alpha means players differ less

    @property
    def sd(self) -> float:
        """Between-player standard deviation of the finishing multiplier."""
        return self.mu / np.sqrt(self.alpha)


def negbin_loglik(goals: np.ndarray, xg: np.ndarray, mu: float, alpha: float) -> float:
    m = mu * xg
    return float(np.sum(special.gammaln(goals + alpha) - special.gammaln(alpha) - special.gammaln(goals + 1)
                        + alpha * np.log(alpha / (alpha + m)) + goals * np.log(m / (alpha + m))))


def fit_prior(goals: np.ndarray, xg: np.ndarray, alpha_max: float = 1e6) -> Prior:
    """Maximum-likelihood estimates of mu and alpha from players' goals and xG.

    The likelihood is often flat in alpha (it can peak at the Poisson limit, alpha -> infinity, meaning
    no detectable difference between players), so alpha is profiled on a log grid up to ``alpha_max``
    instead of trusting a gradient optimiser to stop in the right place.
    """
    goals, xg = np.asarray(goals, float), np.asarray(xg, float)
    keep = xg > 0
    goals, xg = goals[keep], xg[keep]

    def best_mu(alpha: float) -> tuple[float, float]:
        res = optimize.minimize_scalar(lambda log_mu: -negbin_loglik(goals, xg, float(np.exp(log_mu)), alpha),
                                       bounds=(-2.0, 2.0), method="bounded")
        return float(np.exp(res.x)), float(-res.fun)

    fits = [(float(alpha), *best_mu(float(alpha))) for alpha in np.geomspace(0.5, alpha_max, 300)]
    alpha, mu, _ = max(fits, key=lambda f: f[2])
    return Prior(mu=mu, alpha=alpha)


def alpha_lower_bound(goals: np.ndarray, xg: np.ndarray, prior: Prior, chi2: float = 3.841) -> float:
    """Smallest alpha (largest skill spread) still inside the 95% profile-likelihood interval, with mu fixed."""
    goals, xg = np.asarray(goals, float), np.asarray(xg, float)
    keep = xg > 0
    best = negbin_loglik(goals[keep], xg[keep], prior.mu, prior.alpha)
    for alpha in np.geomspace(0.5, prior.alpha, 4000):
        if 2 * (best - negbin_loglik(goals[keep], xg[keep], prior.mu, float(alpha))) <= chi2:
            return float(alpha)
    return prior.alpha


def posterior(goals, xg, prior: Prior, level: float = 0.8) -> pd.DataFrame:
    shape = prior.alpha + np.asarray(goals, float)
    rate = prior.alpha / prior.mu + np.asarray(xg, float)
    tail = (1 - level) / 2
    return pd.DataFrame({
        "shrunk_ratio": shape / rate,
        "low": stats.gamma.ppf(tail, shape, scale=1 / rate),
        "high": stats.gamma.ppf(1 - tail, shape, scale=1 / rate),
    })


def player_totals(shots: pd.DataFrame, names: pd.Series | None = None) -> pd.DataFrame:
    totals = (shots.groupby("player_id")
              .agg(player_name=("player_name", "last"), team=("team", "last"), shots=("is_goal", "size"),
                   goals=("is_goal", "sum"), xg=("xg_model", "sum"))
              .reset_index())
    if names is not None:
        totals["player_name"] = totals.player_id.map(names).fillna(totals.player_name)
    totals["raw_ratio"] = totals.goals / totals.xg
    return totals


def season_validation(shots: pd.DataFrame) -> pd.DataFrame:
    """Predict next-season goals from this season's finishing, three ways, for players who shot in both."""
    seasons = sorted(config.SEASONS, key=config.SEASON_START_YEAR.get)
    rows = []
    for before, after in zip(seasons, seasons[1:]):
        a = player_totals(shots[shots.season_id == before])
        b = player_totals(shots[shots.season_id == after])
        prior = fit_prior(a.goals.to_numpy(), a.xg.to_numpy())
        a = a.assign(shrunk=posterior(a.goals, a.xg, prior).shrunk_ratio.to_numpy())
        both = a.merge(b[["player_id", "goals", "xg"]], on="player_id", suffixes=("", "_next"))
        for method, theta in [("league average", np.full(len(both), prior.mu)),
                              ("raw goals / xG", both.raw_ratio.to_numpy()),
                              ("hierarchical (shrunk)", both.shrunk.to_numpy())]:
            pred = theta * both.xg_next.to_numpy()
            err = both.goals_next.to_numpy() - pred
            rows.append({"from_season": config.SEASONS[before], "to_season": config.SEASONS[after],
                         "players": len(both), "method": method,
                         "mae_goals": float(np.mean(np.abs(err))), "rmse_goals": float(np.sqrt(np.mean(err ** 2)))})
    return pd.DataFrame(rows)


def repeatability(shots: pd.DataFrame, min_shots: int = 20) -> dict:
    """Correlation of raw finishing (goals minus xG per shot) between consecutive seasons."""
    seasons = sorted(config.SEASONS, key=config.SEASON_START_YEAR.get)
    pairs = []
    for before, after in zip(seasons, seasons[1:]):
        a = player_totals(shots[shots.season_id == before])
        b = player_totals(shots[shots.season_id == after])
        both = a.merge(b, on="player_id", suffixes=("_a", "_b"))
        both = both[(both.shots_a >= min_shots) & (both.shots_b >= min_shots)]
        pairs.append(pd.DataFrame({"a": (both.goals_a - both.xg_a) / both.shots_a,
                                   "b": (both.goals_b - both.xg_b) / both.shots_b}))
    pooled = pd.concat(pairs)
    r, p = stats.pearsonr(pooled.a, pooled.b)
    return {"player_season_pairs": int(len(pooled)), "min_shots_each_season": min_shots,
            "pearson_r": float(r), "p_value": float(p)}


@dataclass
class FinishingResult:
    prior: Prior
    wide_prior: Prior   # largest skill spread the data cannot rule out (95% profile bound)
    players: pd.DataFrame
    validation: pd.DataFrame
    repeatability: dict


def run(shots: pd.DataFrame, names: pd.Series | None = None) -> FinishingResult:
    totals = player_totals(shots, names)
    prior = fit_prior(totals.goals.to_numpy(), totals.xg.to_numpy())
    wide = Prior(prior.mu, alpha_lower_bound(totals.goals.to_numpy(), totals.xg.to_numpy(), prior))
    best = posterior(totals.goals, totals.xg, prior)
    generous = posterior(totals.goals, totals.xg, wide).add_suffix("_wide")
    players = pd.concat([totals, best, generous], axis=1)
    return FinishingResult(prior, wide, players, season_validation(shots), repeatability(shots))


def plot_shrinkage(result: FinishingResult, path, min_shots: int = 15) -> None:
    d = result.players[result.players.shots >= min_shots]
    hidden = int((d.raw_ratio > 2.5).sum())
    d = d[d.raw_ratio <= 2.5]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    lim = (0, 2.6)
    ax.plot(lim, lim, color="#999999", ls="--", lw=1, label="raw ratio taken at face value")
    ax.scatter(d.raw_ratio, d.shrunk_ratio_wide, s=d.xg * 6, alpha=0.45, color="#1c5d99", edgecolor="white", lw=0.4,
               label=f"largest plausible skill spread (SD {result.wide_prior.sd:.2f})")
    ax.scatter(d.raw_ratio, d.shrunk_ratio, s=d.xg * 6, alpha=0.35, color="#e07a1f", edgecolor="white", lw=0.4,
               label=f"best estimate (SD {result.prior.sd:.2f})")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    if hidden:
        ax.text(2.55, 0.08, f"{hidden} players with a raw ratio above 2.5 not shown", ha="right", fontsize=7, color="#666666")
    ax.set_xlabel("Raw finishing: goals / xG")
    ax.set_ylabel("Hierarchical estimate (posterior mean)")
    ax.set_title(f"Small samples are pulled to the average\n(players with {min_shots}+ shots; dot size = xG)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save(result: FinishingResult, top: int = 15) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"mu": result.prior.mu, "alpha": result.prior.alpha, "between_player_sd": result.prior.sd,
                   "alpha_95_lower": result.wide_prior.alpha, "between_player_sd_95_upper": result.wide_prior.sd,
                   **{f"repeatability_{k}": v for k, v in result.repeatability.items()}}]).round(4) \
        .to_csv(config.REPORTS_DIR / "finishing_model.csv", index=False)
    result.validation.round(4).to_csv(config.REPORTS_DIR / "finishing_validation.csv", index=False)
    # Highest-volume shooters only: a neutral selection that shows how evidence changes the estimate.
    (result.players.sort_values("xg", ascending=False).head(top)
     [["player_name", "team", "shots", "goals", "xg", "raw_ratio", "shrunk_ratio", "low", "high",
       "shrunk_ratio_wide", "low_wide", "high_wide"]]
     .round(3).to_csv(config.REPORTS_DIR / "finishing_highest_volume_shooters.csv", index=False))
    plot_shrinkage(result, config.FIGURES_DIR / "finishing_shrinkage.png")
