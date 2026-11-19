"""Project settings: which StatsBomb competition and seasons to use, and where files live."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("PLAYER_DEV_DATA", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# FA Women's Super League: every season currently in StatsBomb open data.
COMPETITION_ID = 37
SEASONS = {4: "2018/2019", 42: "2019/2020", 90: "2020/2021", 281: "2023/2024"}
SEASON_START_YEAR = {4: 2018, 42: 2019, 90: 2020, 281: 2023}

# The xG model is selected on the first three seasons and tested once on the latest one.
TRAIN_SEASONS = (4, 42, 90)
HOLDOUT_SEASON = 281

# StatsBomb pitch: 120 x 80 yards, attacking towards x = 120.
GOAL_X, GOAL_Y = 120.0, 40.0
POST_LOW_Y, POST_HIGH_Y = 36.0, 44.0

MIN_MINUTES = 900          # player-season inclusion threshold (ten full matches)
RELIABILITY_TARGET = 0.7   # split-half reliability we treat as "stable enough to act on"
RANDOM_STATE = 2026

# Per-90 metrics used for trajectories, reliability and profiles (outfield players).
RATE_METRICS = {
    "np_xg": "Non-penalty xG",
    "shots": "Shots",
    "xa": "Expected assists (xA)",
    "progressive_passes": "Progressive passes",
    "progressive_carries": "Progressive carries",
    "dribbles_completed": "Completed dribbles",
    "pressures": "Pressures",
    "ball_recoveries": "Ball recoveries",
    "tackles_interceptions": "Tackles + interceptions won",
}

POSITION_GROUPS = ("DF", "MF", "FW")
