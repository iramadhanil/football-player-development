"""How many minutes before a per-90 metric is reliable enough to act on?

For each player-season (900+ minutes) the matches are split at random into two halves and the
metric is computed per 90 in each half. The correlation between halves across players (within a
position group) is the split-half reliability for half a season's minutes. The Spearman-Brown
formula then gives the minutes needed to reach the target reliability. Repeating the random split
200 times removes the luck of a single split.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


def spearman_brown(r: float, k: float) -> float:
    """Reliability of a measure k times as long as one with reliability r."""
    return k * r / (1 + (k - 1) * r)


def length_factor(r: float, target: float) -> float:
    """How many times longer a measure with reliability r must be to reach ``target``."""
    if r <= 0:
        return float("inf")
    return target * (1 - r) / (r * (1 - target))


def split_half(matches: pd.DataFrame, metric: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-90 values in two random halves of each player-season's matches."""
    m = matches.assign(half=0)
    keys = m.groupby(["season_id", "player_id"]).ngroup().to_numpy()
    order = rng.permutation(len(m))
    # Alternate assignment within each player-season after shuffling gives equal-sized halves.
    shuffled = pd.DataFrame({"key": keys[order], "row": order})
    shuffled["half"] = shuffled.groupby("key").cumcount() % 2
    m.iloc[shuffled.row.to_numpy(), m.columns.get_loc("half")] = shuffled.half.to_numpy()
    g = m.groupby(["season_id", "player_id", "half"]).agg(count=(metric, "sum"), minutes=("minutes", "sum")).unstack("half")
    g = g.dropna()
    a = (g[("count", 0)] / g[("minutes", 0)] * 90).to_numpy()
    b = (g[("count", 1)] / g[("minutes", 1)] * 90).to_numpy()
    half_minutes = ((g[("minutes", 0)] + g[("minutes", 1)]) / 2).to_numpy()
    return a, b, half_minutes


def run(players: pd.DataFrame, panel: pd.DataFrame, iterations: int = 200,
        target: float = config.RELIABILITY_TARGET) -> pd.DataFrame:
    rng = np.random.default_rng(config.RANDOM_STATE)
    eligible = panel[["season_id", "player_id", "position_group"]]
    rows = []
    for group in config.POSITION_GROUPS:
        keep = eligible[eligible.position_group == group][["season_id", "player_id"]]
        matches = players.merge(keep, on=["season_id", "player_id"])
        matches = matches[matches.minutes > 0]
        for metric in config.RATE_METRICS:
            zs, halves = [], []
            for _ in range(iterations):
                a, b, half_minutes = split_half(matches, metric, rng)
                zs.append(np.arctanh(np.clip(np.corrcoef(a, b)[0, 1], -0.999, 0.999)))
                halves.append(np.median(half_minutes))
            r = float(np.tanh(np.mean(zs)))  # average correlation on the Fisher-z scale
            half = float(np.median(halves))
            rows.append({"position_group": group, "metric": metric, "label": config.RATE_METRICS[metric],
                         "player_seasons": int(len(keep)), "median_half_minutes": half,
                         "split_half_r": r, "full_season_reliability": spearman_brown(r, 2),
                         f"minutes_for_{target}": length_factor(r, target) * half})
    return pd.DataFrame(rows)


def plot(table: pd.DataFrame, path, target: float = config.RELIABILITY_TARGET) -> None:
    col = f"minutes_for_{target}"
    order = table.groupby("metric")[col].median().sort_values().index
    labels = [config.RATE_METRICS[m] for m in order]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = {"DF": "#1c5d99", "MF": "#e07a1f", "FW": "#3a9a5b"}
    for i, group in enumerate(config.POSITION_GROUPS):
        t = table[table.position_group == group].set_index("metric").loc[order]
        ax.scatter(t[col].clip(upper=4000), np.arange(len(order)) + (i - 1) * 0.18, color=colors[group],
                   label=group, s=36)
    ax.axvline(900, color="#999999", ls=":", lw=1)
    ax.text(915, -0.45, "900 min = 10 full matches", fontsize=7, color="#666666", va="center")
    ax.set_ylim(-0.8, len(order) - 0.4)
    ax.set_yticks(np.arange(len(order)), labels)
    ax.set_xlabel(f"Minutes needed for split-half reliability of {target} (capped at 4,000)")
    ax.set_title("How long before a per-90 number means something?")
    ax.legend(frameon=False, title="Position", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save(table: pd.DataFrame) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(config.REPORTS_DIR / "reliability_minutes.csv", index=False)
    plot(table, config.FIGURES_DIR / "reliability_minutes.png")
