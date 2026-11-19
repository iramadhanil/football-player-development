"""Pitch geometry on StatsBomb coordinates (120 x 80 yards, attacking towards x = 120)."""
from __future__ import annotations

import math
from typing import Sequence

from .config import GOAL_X, GOAL_Y, POST_HIGH_Y, POST_LOW_Y

Point = Sequence[float]
LOW_POST = (GOAL_X, POST_LOW_Y)
HIGH_POST = (GOAL_X, POST_HIGH_Y)


def distance_to_goal(x: float, y: float) -> float:
    """Distance in yards from (x, y) to the centre of the goal."""
    return math.hypot(GOAL_X - x, GOAL_Y - y)


def shot_angle(x: float, y: float) -> float:
    """Angle in radians between the lines from (x, y) to each post: how much goal the shooter sees."""
    a_low = math.atan2(POST_LOW_Y - y, GOAL_X - x)
    a_high = math.atan2(POST_HIGH_Y - y, GOAL_X - x)
    angle = abs(a_high - a_low)
    return 2 * math.pi - angle if angle > math.pi else angle


def point_in_triangle(p: Point, a: Point, b: Point, c: Point) -> bool:
    """True if p lies inside or on the edge of triangle abc."""
    def cross(o: Point, u: Point, v: Point) -> float:
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])

    d1, d2, d3 = cross(a, b, p), cross(b, c, p), cross(c, a, p)
    has_neg = d1 < 0 or d2 < 0 or d3 < 0
    has_pos = d1 > 0 or d2 > 0 or d3 > 0
    return not (has_neg and has_pos)


def in_shot_cone(shot: Point, player: Point) -> bool:
    """True if a player stands in the triangle between the shot location and the two posts."""
    return point_in_triangle(player, shot, LOW_POST, HIGH_POST)


def is_progressive(start: Point, end: Point, share: float = 0.25, min_gain: float = 5.0) -> bool:
    """A pass or carry is progressive if it ends at least ``share`` closer to goal and gains ``min_gain`` yards."""
    d_start = distance_to_goal(start[0], start[1])
    d_end = distance_to_goal(end[0], end[1])
    return d_end <= (1 - share) * d_start and (d_start - d_end) >= min_gain
