"""Turn raw StatsBomb JSON into three tidy tables: matches, shots and player-match counts."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from . import config
from .geometry import distance_to_goal, in_shot_cone, is_progressive, shot_angle

SET_PIECE_PASSES = {"Corner", "Free Kick", "Goal Kick", "Kick Off", "Throw-in"}
WON = {"Won", "Success", "Success In Play", "Success Out"}
RED_CARDS = {"Red Card", "Second Yellow"}
COUNT_COLUMNS = [
    "passes_attempted", "passes_completed", "progressive_passes", "progressive_carries",
    "dribbles_attempted", "dribbles_completed", "pressures", "ball_recoveries",
    "tackles_won", "interceptions_won", "shots", "np_goals", "np_xg", "xa",
    "penalties_taken", "penalty_goals",
]


def position_group(position: str | None) -> str:
    """Map a StatsBomb position name to GK, DF, MF or FW."""
    if not position:
        return "UNK"
    if position == "Goalkeeper":
        return "GK"
    if "Back" in position:
        return "DF"
    if "Midfield" in position:
        return "MF"
    if "Wing" in position or "Forward" in position or "Striker" in position:
        return "FW"
    return "UNK"


def event_seconds(event: dict) -> int:
    return event["minute"] * 60 + event["second"]


def match_end_seconds(events: list[dict]) -> int:
    """Match clock (seconds) of the last event, used as the end time for players still on the pitch."""
    return max(event_seconds(e) for e in events)


def player_minutes(events: list[dict], lineups: list[dict], end_seconds: int) -> list[dict]:
    """Minutes, main position and starting status for every player who appeared.

    Minutes come from the event stream (Starting XI, substitutions, red cards) because the
    position spells in some StatsBomb lineup files have out-of-order times. The main position is
    the position the player was recorded in most often across their own events.
    """
    nicknames = {p["player_id"]: p.get("player_nickname") or p["player_name"]
                 for team in lineups for p in team["lineup"]}
    on: dict[int, int] = {}
    off: dict[int, int] = {}
    team_of: dict[int, int] = {}
    names: dict[int, str] = {}
    start_position: dict[int, str] = {}
    positions: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for e in events:
        kind = e["type"]["name"]
        if kind == "Starting XI":
            for slot in e["tactics"]["lineup"]:
                pid = slot["player"]["id"]
                on[pid], team_of[pid], names[pid] = 0, e["team"]["id"], slot["player"]["name"]
                start_position[pid] = slot["position"]["name"]
            continue
        player = e.get("player")
        if not player:
            continue
        pid = player["id"]
        if e.get("position"):
            positions[pid][e["position"]["name"]] += 1
        if kind == "Substitution":
            off.setdefault(pid, event_seconds(e))
            new = e["substitution"]["replacement"]
            on.setdefault(new["id"], event_seconds(e))
            team_of[new["id"]], names[new["id"]] = e["team"]["id"], new["name"]
        card = ((e.get("foul_committed") or {}).get("card") or (e.get("bad_behaviour") or {}).get("card") or {})
        if card.get("name") in RED_CARDS:
            off.setdefault(pid, event_seconds(e))
    rows = []
    for pid, start in on.items():
        counted = positions.get(pid)
        main_position = max(counted, key=counted.get) if counted else start_position.get(pid)
        rows.append({
            "team_id": team_of[pid], "player_id": pid,
            "player_name": nicknames.get(pid) or names[pid],
            "minutes": round(max(off.get(pid, end_seconds) - start, 0) / 60, 2),
            "started": start == 0 and pid in start_position,
            "main_position": main_position,
            "position_group": position_group(main_position),
        })
    return rows


def freeze_frame_features(location: list[float], frame: list[dict] | None) -> dict:
    """Defenders and goalkeeper position at the moment of the shot."""
    if not frame:
        return {"has_freeze_frame": False, "n_def_in_cone": None, "n_opp_within_3": None,
                "gk_in_cone": None, "gk_dist_to_goal": None}
    opponents = [p for p in frame if not p.get("teammate")]
    keepers = [p for p in opponents if (p.get("position") or {}).get("name") == "Goalkeeper"]
    outfield = [p for p in opponents if p not in keepers]
    shot = (location[0], location[1])
    gk = keepers[0]["location"] if keepers else None
    return {
        "has_freeze_frame": True,
        "n_def_in_cone": sum(in_shot_cone(shot, p["location"]) for p in outfield),
        "n_opp_within_3": sum(
            ((p["location"][0] - shot[0]) ** 2 + (p["location"][1] - shot[1]) ** 2) ** 0.5 <= 3 for p in opponents
        ),
        "gk_in_cone": bool(in_shot_cone(shot, gk)) if gk else None,
        "gk_dist_to_goal": distance_to_goal(gk[0], gk[1]) if gk else None,
    }


def _key_pass_flags(pass_event: dict | None) -> dict:
    if not pass_event:
        return {"assisted": False, "assist_through_ball": False, "assist_cross": False, "assist_cut_back": False}
    p = pass_event.get("pass", {})
    technique = (p.get("technique") or {}).get("name")
    return {
        "assisted": True,
        "assist_through_ball": bool(p.get("through_ball")) or technique == "Through Ball",
        "assist_cross": bool(p.get("cross")),
        "assist_cut_back": bool(p.get("cut_back")),
    }


def parse_match(match_id: int, season_id: int, raw_dir: Path | None = None) -> dict:
    """Parse one match into shot rows, player-match rows and team goal totals."""
    raw_dir = raw_dir or config.RAW_DIR
    events = json.loads((raw_dir / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    lineups = json.loads((raw_dir / "lineups" / f"{match_id}.json").read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in events}
    end_seconds = match_end_seconds(events)
    counts: dict[int, dict] = defaultdict(lambda: defaultdict(float))
    team_goals: dict[int, int] = defaultdict(int)
    shots = []

    for e in events:
        kind = e["type"]["name"]
        player = (e.get("player") or {}).get("id")
        c = counts[player] if player is not None else None
        if kind == "Own Goal For":
            team_goals[e["team"]["id"]] += 1
        elif kind == "Pass" and c is not None:
            p = e["pass"]
            c["passes_attempted"] += 1
            if "outcome" not in p:
                c["passes_completed"] += 1
                set_piece = (p.get("type") or {}).get("name") in SET_PIECE_PASSES
                if not set_piece and is_progressive(e["location"], p["end_location"]):
                    c["progressive_passes"] += 1
        elif kind == "Carry" and c is not None:
            if is_progressive(e["location"], e["carry"]["end_location"]):
                c["progressive_carries"] += 1
        elif kind == "Dribble" and c is not None:
            c["dribbles_attempted"] += 1
            if (e.get("dribble", {}).get("outcome") or {}).get("name") == "Complete":
                c["dribbles_completed"] += 1
        elif kind == "Pressure" and c is not None:
            c["pressures"] += 1
        elif kind == "Ball Recovery" and c is not None:
            if not (e.get("ball_recovery") or {}).get("recovery_failure"):
                c["ball_recoveries"] += 1
        elif kind == "Duel" and c is not None:
            duel = e.get("duel") or {}
            if (duel.get("type") or {}).get("name") == "Tackle" and (duel.get("outcome") or {}).get("name") in WON:
                c["tackles_won"] += 1
        elif kind == "Interception" and c is not None:
            if ((e.get("interception") or {}).get("outcome") or {}).get("name") in WON:
                c["interceptions_won"] += 1
        elif kind == "Shot" and c is not None:
            s = e["shot"]
            shot_type = s["type"]["name"]
            is_goal = s["outcome"]["name"] == "Goal"
            team_goals[e["team"]["id"]] += int(is_goal)
            xg = float(s.get("statsbomb_xg", 0.0))
            if shot_type == "Penalty":
                c["penalties_taken"] += 1
                c["penalty_goals"] += int(is_goal)
            else:
                c["shots"] += 1
                c["np_goals"] += int(is_goal)
                c["np_xg"] += xg
            key_pass = by_id.get(s.get("key_pass_id")) if s.get("key_pass_id") else None
            if key_pass is not None and shot_type != "Penalty" and key_pass.get("player"):
                counts[key_pass["player"]["id"]]["xa"] += xg
            x, y = e["location"][0], e["location"][1]
            shots.append({
                "shot_id": e["id"], "match_id": match_id, "season_id": season_id,
                "team_id": e["team"]["id"], "player_id": player, "player_name": e["player"]["name"],
                "period": e["period"], "minute": e["minute"], "x": x, "y": y,
                "distance": distance_to_goal(x, y), "angle": shot_angle(x, y),
                "shot_type": shot_type, "body_part": s["body_part"]["name"],
                "technique": (s.get("technique") or {}).get("name", "Normal"),
                "play_pattern": e["play_pattern"]["name"],
                "first_time": bool(s.get("first_time")), "under_pressure": bool(e.get("under_pressure")),
                "one_on_one": bool(s.get("one_on_one")), "open_goal": bool(s.get("open_goal")),
                "follows_dribble": bool(s.get("follows_dribble")),
                **_key_pass_flags(key_pass),
                **freeze_frame_features(e["location"], s.get("freeze_frame")),
                "outcome": s["outcome"]["name"], "is_goal": int(is_goal), "statsbomb_xg": xg,
            })

    player_rows = []
    for row in player_minutes(events, lineups, end_seconds):
        c = counts.get(row["player_id"], {})
        player_rows.append({"match_id": match_id, "season_id": season_id, **row,
                            **{col: float(c.get(col, 0.0)) for col in COUNT_COLUMNS}})
    appeared = {r["player_id"] for r in player_rows}
    # Bench players can be booked ("Bad Behaviour") without playing, so they are not counted as missing.
    on_ball = {e["player"]["id"] for e in events if e.get("player") and e["type"]["name"] != "Bad Behaviour"}
    missing = sorted(p for p in on_ball if p not in appeared)
    return {"shots": shots, "players": player_rows, "team_goals": dict(team_goals),
            "end_seconds": end_seconds, "players_without_minutes": missing}


def load_matches(raw_dir: Path | None = None) -> pd.DataFrame:
    raw_dir = raw_dir or config.RAW_DIR
    rows = []
    for season_id, season in config.SEASONS.items():
        path = raw_dir / "matches" / str(config.COMPETITION_ID) / f"{season_id}.json"
        for m in json.loads(path.read_text(encoding="utf-8")):
            rows.append({
                "match_id": m["match_id"], "season_id": season_id, "season": season,
                "season_year": config.SEASON_START_YEAR[season_id], "match_date": m["match_date"],
                "home_team_id": m["home_team"]["home_team_id"], "away_team_id": m["away_team"]["away_team_id"],
                "home_team": m["home_team"]["home_team_name"], "away_team": m["away_team"]["away_team_name"],
                "home_score": m["home_score"], "away_score": m["away_score"],
                "data_version": (m.get("metadata") or {}).get("data_version"),
            })
    return pd.DataFrame(rows).sort_values(["match_date", "match_id"]).reset_index(drop=True)


def _parse_job(args: tuple[int, int, str]) -> tuple[int, dict]:
    match_id, season_id, raw_dir = args
    return match_id, parse_match(match_id, season_id, Path(raw_dir))


def build(raw_dir: Path | None = None, out_dir: Path | None = None, workers: int | None = None) -> dict[str, pd.DataFrame]:
    """Parse every match and write matches.csv, shots.csv and player_match.csv."""
    raw_dir = raw_dir or config.RAW_DIR
    out_dir = out_dir or config.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = load_matches(raw_dir)
    jobs = [(int(m), int(s), str(raw_dir)) for m, s in zip(matches.match_id, matches.season_id)]
    with ProcessPoolExecutor(max_workers=workers or min(8, os.cpu_count() or 2)) as pool:
        parsed = dict(pool.map(_parse_job, jobs, chunksize=4))

    # Team names differ slightly between files (e.g. "Manchester United" vs "Manchester United W"),
    # so teams are joined on id and named after the match list.
    team_names = dict(zip(matches.home_team_id, matches.home_team)) | dict(zip(matches.away_team_id, matches.away_team))
    shots = pd.DataFrame([s for m in matches.match_id for s in parsed[m]["shots"]])
    players = pd.DataFrame([p for m in matches.match_id for p in parsed[m]["players"]])
    for table in (shots, players):
        table.insert(table.columns.get_loc("team_id") + 1, "team", table["team_id"].map(team_names))
    players["tackles_interceptions"] = players["tackles_won"] + players["interceptions_won"]
    goals = []
    for m in matches.itertuples():
        team_goals = parsed[m.match_id]["team_goals"]
        goals.append({
            "match_id": m.match_id,
            "event_home_goals": team_goals.get(m.home_team_id, 0), "event_away_goals": team_goals.get(m.away_team_id, 0),
            "end_minutes": parsed[m.match_id]["end_seconds"] / 60,
            "players_without_minutes": len(parsed[m.match_id]["players_without_minutes"]),
        })
    matches = matches.merge(pd.DataFrame(goals), on="match_id")

    matches.to_csv(out_dir / "matches.csv", index=False)
    shots.to_csv(out_dir / "shots.csv", index=False)
    players.to_csv(out_dir / "player_match.csv", index=False)
    return {"matches": matches, "shots": shots, "players": players}


def load_processed(out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    out_dir = out_dir or config.PROCESSED_DIR
    return {name: pd.read_csv(out_dir / f"{file}.csv")
            for name, file in (("matches", "matches"), ("shots", "shots"), ("players", "player_match"))}
