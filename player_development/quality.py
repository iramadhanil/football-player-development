"""Data-quality checks on the processed tables. Every check returns a row; failures are reported, not hidden."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def run_checks(matches: pd.DataFrame, shots: pd.DataFrame, players: pd.DataFrame,
               raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or config.RAW_DIR
    rows = []

    def check(name: str, passed: bool, detail: str) -> None:
        rows.append({"check": name, "passed": bool(passed), "detail": detail})

    missing = [m for m in matches.match_id
               if not (raw_dir / "events" / f"{m}.json").exists() or not (raw_dir / "lineups" / f"{m}.json").exists()]
    check("every match has events and lineups", not missing, f"{len(matches)} matches, {len(missing)} missing files")

    score_gap = (matches.home_score != matches.event_home_goals) | (matches.away_score != matches.event_away_goals)
    check("goals in events match final scores", not score_gap.any(),
          f"{int(score_gap.sum())} of {len(matches)} matches differ (own goals included)")

    team = (players.groupby(["match_id", "team_id"]).minutes.sum().reset_index()
            .merge(matches[["match_id", "end_minutes"]], on="match_id"))
    ratio = team.minutes / team.end_minutes
    odd = (ratio > 11.01) | (ratio < 9.5)
    check("team minutes equal 11 players x match length", not odd.any(),
          f"ratio range {ratio.min():.2f}-{ratio.max():.2f}; values below 11 come from red cards")

    check("player minutes within 0-130", players.minutes.between(0, 130).all(),
          f"max {players.minutes.max():.1f} minutes")
    check("one row per player per match", not players.duplicated(["match_id", "player_id"]).any(),
          f"{len(players)} player-match rows")

    on_pitch = shots.x.between(-0.5, 120.5) & shots.y.between(-0.5, 80.5)
    check("shot locations inside the pitch (0.5-yard tolerance)", on_pitch.all(), f"{int((~on_pitch).sum())} outside")
    check("StatsBomb xG between 0 and 1", shots.statsbomb_xg.between(0, 1).all(), f"{len(shots)} shots")
    check("goal flag matches shot outcome", ((shots.outcome == "Goal").astype(int) == shots.is_goal).all(),
          f"{int(shots.is_goal.sum())} goals")
    check("shots have a freeze frame (>= 98%)", shots.has_freeze_frame.mean() >= 0.98,
          f"{shots.has_freeze_frame.mean():.1%} of shots")

    shooters = shots.merge(players[["match_id", "player_id", "minutes"]], on=["match_id", "player_id"], how="left")
    no_minutes = shooters.minutes.isna() | (shooters.minutes <= 0)
    check("every shooter has minutes in that match", not no_minutes.any(), f"{int(no_minutes.sum())} shots without minutes")
    unknown = int((players.position_group == "UNK").sum())
    check("main position known (< 0.5% unknown)", unknown / len(players) < 0.005,
          f"{unknown} player-match rows without a position (late substitutes with no events)")
    lost = int(matches.players_without_minutes.sum())
    check("event players found in minutes table", lost == 0, f"{lost} players with events but no minutes")
    return pd.DataFrame(rows)
