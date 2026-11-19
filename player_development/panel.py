"""Player-season panel: minutes, counts and per-90 rates for outfield players."""
from __future__ import annotations

import pandas as pd

from . import config


def player_seasons(players: pd.DataFrame, min_minutes: float = config.MIN_MINUTES) -> pd.DataFrame:
    """One row per player and season with per-90 rates; outfield players above ``min_minutes`` only."""
    p = players[players.position_group.isin(config.POSITION_GROUPS)]
    counts = list(config.RATE_METRICS)
    agg = p.groupby(["season_id", "player_id"]).agg(
        player_name=("player_name", "last"), minutes=("minutes", "sum"), matches=("match_id", "nunique"),
        **{c: (c, "sum") for c in counts})
    # Team and position: where the player spent most minutes that season.
    team = (p.groupby(["season_id", "player_id", "team"]).minutes.sum().reset_index()
            .sort_values("minutes").drop_duplicates(["season_id", "player_id"], keep="last")
            .set_index(["season_id", "player_id"]).team)
    group = (p.groupby(["season_id", "player_id", "position_group"]).minutes.sum().reset_index()
             .sort_values("minutes").drop_duplicates(["season_id", "player_id"], keep="last")
             .set_index(["season_id", "player_id"]).position_group)
    agg = agg.join(team).join(group).reset_index()
    agg = agg[agg.minutes >= min_minutes].copy()
    for c in counts:
        agg[f"{c}_p90"] = agg[c] / agg.minutes * 90
    agg["season"] = agg.season_id.map(config.SEASONS)
    agg["years"] = agg.season_id.map(config.SEASON_START_YEAR) - min(config.SEASON_START_YEAR.values())
    return agg.sort_values(["player_id", "years"]).reset_index(drop=True)
