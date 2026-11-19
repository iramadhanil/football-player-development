import pytest

from player_development.parse import parse_match, player_minutes, position_group
from tests.conftest import synthetic_match


@pytest.mark.parametrize("position,group", [
    ("Goalkeeper", "GK"), ("Left Back", "DF"), ("Right Wing Back", "DF"), ("Center Back", "DF"),
    ("Center Defensive Midfield", "MF"), ("Left Midfield", "MF"), ("Center Attacking Midfield", "MF"),
    ("Right Wing", "FW"), ("Center Forward", "FW"), ("Secondary Striker", "FW"), (None, "UNK"),
])
def test_position_groups(position, group):
    assert position_group(position) == group


def test_minutes_from_starts_substitutions_and_red_cards():
    events, lineups = synthetic_match()
    rows = {r["player_id"]: r for r in player_minutes(events, lineups, end_seconds=95 * 60)}
    assert rows[11]["minutes"] == pytest.approx(60)   # substituted at 60:00
    assert rows[12]["minutes"] == pytest.approx(80)   # sent off at 80:00
    assert rows[13]["minutes"] == pytest.approx(35)   # on at 60:00, played to the end
    assert rows[11]["started"] and not rows[13]["started"]
    assert rows[11]["player_name"] == "Ana"            # nickname from the lineup file
    assert rows[13]["position_group"] == "FW"         # most frequent position in own events
    assert 14 not in rows                              # unused substitute


def test_parse_match_counts_and_shot_features(raw_match):
    out = parse_match(1, 4, raw_match)
    players = {p["player_id"]: p for p in out["players"]}
    assert players[12]["progressive_passes"] == 1      # the throw-in is excluded
    assert players[12]["passes_attempted"] == 2 and players[12]["passes_completed"] == 2
    assert players[12]["xa"] == pytest.approx(0.3)     # credited for the assisted shot
    assert players[11]["shots"] == 1 and players[11]["np_goals"] == 1
    assert players[11]["progressive_carries"] == 1 and players[11]["dribbles_completed"] == 1
    assert players[13]["shots"] == 0 and players[13]["penalties_taken"] == 1
    assert players[21]["ball_recoveries"] == 1
    assert out["team_goals"] == {1: 2}                 # one shot goal plus one own goal

    shot = next(s for s in out["shots"] if s["shot_type"] == "Open Play")
    assert shot["distance"] == pytest.approx(12.0)
    assert shot["n_def_in_cone"] == 1                  # centre back between shooter and goal
    assert shot["n_opp_within_3"] == 1                 # left back one yard away
    assert shot["gk_in_cone"] is True
    assert shot["gk_dist_to_goal"] == pytest.approx(1.0)
    assert shot["assisted"] and shot["assist_through_ball"]
    assert out["players_without_minutes"] == []
