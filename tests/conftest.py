"""Synthetic StatsBomb-shaped match used by the tests. No StatsBomb data is stored in this repository."""
import json

import pytest

HOME, AWAY = {"id": 1, "name": "Home FC"}, {"id": 2, "name": "Away FC"}


def player(pid, name):
    return {"id": pid, "name": name}


def event(idx, kind, minute, second=0, team=HOME, who=None, position=None, period=1, **extra):
    e = {"id": f"e{idx}", "index": idx, "period": period, "minute": minute, "second": second,
         "type": {"name": kind}, "team": team, "play_pattern": {"name": "Regular Play"}}
    if who:
        e["player"] = who
    if position:
        e["position"] = {"name": position}
    e.update(extra)
    return e


def synthetic_match():
    p1, p2, p3 = player(11, "Ana Home"), player(12, "Bea Home"), player(13, "Cat Home")
    gk = player(21, "Dee Keeper")
    events = [
        event(1, "Starting XI", 0, team=HOME, tactics={"lineup": [
            {"player": p1, "position": {"name": "Center Forward"}},
            {"player": p2, "position": {"name": "Left Back"}}]}),
        event(2, "Starting XI", 0, team=AWAY, tactics={"lineup": [
            {"player": gk, "position": {"name": "Goalkeeper"}}]}),
        # Progressive open-play pass from Bea, then a shot by Ana assisted by that pass.
        event(3, "Pass", 10, who=p2, position="Left Back", location=[60, 40],
              **{"pass": {"end_location": [100, 40], "shot_assist": True, "through_ball": True}}),
        event(4, "Shot", 10, 5, who=p1, position="Center Forward", location=[108, 40],
              shot={"type": {"name": "Open Play"}, "body_part": {"name": "Right Foot"},
                    "technique": {"name": "Normal"}, "outcome": {"name": "Goal"}, "statsbomb_xg": 0.3,
                    "key_pass_id": "e3", "first_time": True,
                    "freeze_frame": [
                        {"location": [119, 40], "teammate": False, "position": {"name": "Goalkeeper"}},
                        {"location": [114, 40], "teammate": False, "position": {"name": "Center Back"}},
                        {"location": [109, 41], "teammate": False, "position": {"name": "Left Back"}},
                        {"location": [110, 30], "teammate": True, "position": {"name": "Left Wing"}}]}),
        # A long throw-in is a set piece, so it is not counted as a progressive pass.
        event(5, "Pass", 20, who=p2, position="Left Back", location=[50, 0],
              **{"pass": {"end_location": [90, 10], "type": {"name": "Throw-in"}}}),
        event(6, "Pressure", 30, who=p1, position="Center Forward", location=[90, 40]),
        event(7, "Carry", 31, who=p1, position="Center Forward", location=[70, 40],
              carry={"end_location": [95, 40]}),
        event(8, "Dribble", 32, who=p1, position="Center Forward", location=[95, 40],
              dribble={"outcome": {"name": "Complete"}}),
        event(9, "Substitution", 60, team=HOME, who=p1, position="Center Forward",
              substitution={"replacement": p3}),
        event(10, "Penalty", 70, who=p3, position="Right Wing"),  # unknown type, ignored
        event(11, "Shot", 75, who=p3, position="Right Wing", location=[108, 40],
              shot={"type": {"name": "Penalty"}, "body_part": {"name": "Left Foot"},
                    "outcome": {"name": "Saved"}, "statsbomb_xg": 0.78, "freeze_frame": []}),
        event(12, "Foul Committed", 80, who=p2, position="Left Back",
              foul_committed={"card": {"name": "Red Card"}}),
        event(13, "Own Goal For", 88, team=HOME),
        event(14, "Ball Recovery", 94, 59, team=AWAY, who=gk, position="Goalkeeper", location=[5, 40]),
    ]
    lineups = [
        {"team_id": 1, "team_name": "Home FC", "lineup": [
            {"player_id": 11, "player_name": "Ana Home", "player_nickname": "Ana", "positions": [{}]},
            {"player_id": 12, "player_name": "Bea Home", "player_nickname": None, "positions": [{}]},
            {"player_id": 13, "player_name": "Cat Home", "player_nickname": None, "positions": [{}]},
            {"player_id": 14, "player_name": "Unused Sub", "player_nickname": None, "positions": []}]},
        {"team_id": 2, "team_name": "Away FC", "lineup": [
            {"player_id": 21, "player_name": "Dee Keeper", "player_nickname": None, "positions": [{}]}]},
    ]
    return events, lineups


@pytest.fixture
def raw_match(tmp_path):
    events, lineups = synthetic_match()
    (tmp_path / "events").mkdir()
    (tmp_path / "lineups").mkdir()
    (tmp_path / "events" / "1.json").write_text(json.dumps(events))
    (tmp_path / "lineups" / "1.json").write_text(json.dumps(lineups))
    return tmp_path
