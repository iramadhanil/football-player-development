"""Player profiles for coaches: percentiles within position, with a confidence flag from the reliability study."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def percentiles(panel: pd.DataFrame, season_id: int) -> pd.DataFrame:
    """Percentile (0-100) of each per-90 metric among players in the same position group and season."""
    season = panel[panel.season_id == season_id].copy()
    for metric in config.RATE_METRICS:
        season[f"{metric}_pct"] = season.groupby("position_group")[f"{metric}_p90"].rank(pct=True) * 100
    return season


def flag_confidence(profiles: pd.DataFrame, reliability: pd.DataFrame,
                    target: float = config.RELIABILITY_TARGET) -> pd.DataFrame:
    """Mark metrics where the player's minutes are below the minutes needed for the target reliability."""
    needed = reliability.set_index(["position_group", "metric"])[f"minutes_for_{target}"]
    out = profiles.copy()
    for metric in config.RATE_METRICS:
        req = out.position_group.map(lambda g, m=metric: needed.get((g, m), np.inf))
        out[f"{metric}_reliable"] = out.minutes >= req
    return out


def example_players(profiles: pd.DataFrame, team: str) -> pd.DataFrame:
    """The highest-minute defender, midfielder and forward of one team."""
    squad = profiles[profiles.team == team]
    picks = [squad[squad.position_group == g].sort_values("minutes").tail(1) for g in config.POSITION_GROUPS]
    return pd.concat(picks).reset_index(drop=True)


def plot_profiles(players: pd.DataFrame, path, season_label: str) -> None:
    metrics = list(config.RATE_METRICS)
    fig, axes = plt.subplots(1, len(players), figsize=(5.0 * len(players), 4.8), sharey=True)
    for ax, (_, p) in zip(np.atleast_1d(axes), players.iterrows()):
        values = [p[f"{m}_pct"] for m in metrics]
        reliable = [bool(p[f"{m}_reliable"]) for m in metrics]
        colors = ["#1c5d99" if r else "#b9c7d6" for r in reliable]
        y = np.arange(len(metrics))[::-1]
        ax.barh(y, values, color=colors)
        for yy, v, r in zip(y, values, reliable):
            ax.text(min(v + 2, 92), yy, f"{v:.0f}" + ("" if r else " *"), va="center", fontsize=8)
        ax.axvline(50, color="#999999", ls=":", lw=1)
        ax.set_xlim(0, 100)
        ax.set_yticks(y, [config.RATE_METRICS[m] for m in metrics], fontsize=8)
        ax.set_title(f"{p.player_name} ({p.position_group}, {p.minutes:.0f} min)", fontsize=9)
        ax.set_xlabel("Percentile within position group")
    fig.suptitle(f"{season_label} profiles. * = fewer minutes than needed for reliability {config.RELIABILITY_TARGET} "
                 "(lighter bar)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(panel: pd.DataFrame, reliability: pd.DataFrame, team: str = "Manchester City WFC",
        season_id: int = config.HOLDOUT_SEASON) -> pd.DataFrame:
    profiles = flag_confidence(percentiles(panel, season_id), reliability)
    return example_players(profiles, team)


def save(examples: pd.DataFrame, season_id: int = config.HOLDOUT_SEASON) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["player_name", "team", "position_group", "minutes"]
    for m in config.RATE_METRICS:
        cols += [f"{m}_p90", f"{m}_pct", f"{m}_reliable"]
    examples[cols].round(3).to_csv(config.REPORTS_DIR / "profiles_examples.csv", index=False)
    plot_profiles(examples, config.FIGURES_DIR / "profiles_examples.png", config.SEASONS[season_id])
