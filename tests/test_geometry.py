import math

import pytest

from player_development.geometry import distance_to_goal, in_shot_cone, is_progressive, shot_angle


def test_distance_from_penalty_spot():
    assert distance_to_goal(108, 40) == pytest.approx(12.0)


def test_angle_from_penalty_spot():
    # Posts are 4 yards either side of the centre, 12 yards away.
    assert shot_angle(108, 40) == pytest.approx(2 * math.atan(4 / 12))


def test_angle_is_symmetric_and_shrinks_with_distance():
    assert shot_angle(100, 30) == pytest.approx(shot_angle(100, 50))
    assert shot_angle(90, 40) < shot_angle(110, 40)


def test_angle_on_goal_line_between_posts_is_straight():
    assert shot_angle(120, 40) == pytest.approx(math.pi)


def test_shot_cone():
    shot = (108, 40)
    assert in_shot_cone(shot, (114, 40))       # straight between shooter and goal
    assert not in_shot_cone(shot, (114, 20))   # wide of the near post
    assert not in_shot_cone(shot, (100, 40))   # behind the shooter


def test_progressive_requires_share_and_minimum_gain():
    assert is_progressive((60, 40), (90, 40))        # 60 -> 30 yards from goal
    assert not is_progressive((100, 40), (103, 40))  # only 15% closer
    assert not is_progressive((116, 40), (118, 40))  # 50% closer but only 2 yards gained
    assert not is_progressive((90, 40), (80, 40))    # backwards
