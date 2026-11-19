import numpy as np
import pandas as pd
import pytest

from player_development import finishing, reliability, xg
from player_development.report import md_table


def synthetic_shots(n=600, seed=0):
    rng = np.random.default_rng(seed)
    distance = rng.uniform(3, 35, n)
    p = 1 / (1 + np.exp(-(1.5 - 0.15 * distance)))
    return pd.DataFrame({
        "shot_id": [f"s{i}" for i in range(n)], "match_id": np.arange(n) // 25, "season_id": 4,
        "player_id": rng.integers(1, 30, n), "player_name": "x", "team": "t",
        "distance": distance, "angle": rng.uniform(0.1, 1.2, n),
        "shot_type": rng.choice(["Open Play", "Free Kick", "Penalty"], n, p=[0.9, 0.07, 0.03]),
        "body_part": rng.choice(["Right Foot", "Head", "Other"], n, p=[0.8, 0.18, 0.02]),
        "technique": "Normal", "play_pattern": "Regular Play",
        "first_time": False, "under_pressure": False, "one_on_one": False, "open_goal": False,
        "follows_dribble": False, "assisted": True, "assist_through_ball": False, "assist_cross": False,
        "assist_cut_back": False, "has_freeze_frame": rng.random(n) > 0.05,
        "n_def_in_cone": rng.integers(0, 4, n).astype(float), "n_opp_within_3": rng.integers(0, 3, n).astype(float),
        "gk_in_cone": True, "gk_dist_to_goal": rng.uniform(0.5, 6, n),
        "is_goal": (rng.random(n) < p).astype(int), "statsbomb_xg": p,
    })


def test_prepare_drops_penalties_and_fills_missing_frames():
    shots = synthetic_shots()
    shots.loc[~shots.has_freeze_frame, ["n_def_in_cone", "gk_dist_to_goal"]] = np.nan
    df = xg.prepare(shots)
    assert (df.shot_type != "Penalty").all()
    assert not df[xg.FEATURES].isna().any().any()
    assert df.frame_missing.sum() == (~shots[shots.shot_type != "Penalty"].has_freeze_frame).sum()


def test_out_of_fold_predictions_are_probabilities():
    df = xg.prepare(synthetic_shots())
    oof = xg.out_of_fold(xg.make_models()["logistic_regression"], df, folds=3)
    assert oof.shape == (len(df),) and (oof > 0).all() and (oof < 1).all()


def test_calibration_error_is_zero_for_a_constant_true_rate():
    y = np.array([0, 1] * 50)
    assert xg.expected_calibration_error(y, np.full(100, 0.5)) == pytest.approx(0.0)


def test_posterior_without_evidence_is_the_prior_mean():
    prior = finishing.Prior(mu=1.1, alpha=8.0)
    post = finishing.posterior([0], [0.0], prior)
    assert post.shrunk_ratio.iloc[0] == pytest.approx(1.1)


def test_posterior_with_lots_of_evidence_approaches_the_raw_ratio():
    prior = finishing.Prior(mu=1.0, alpha=8.0)
    post = finishing.posterior([1300], [1000.0], prior)
    assert post.shrunk_ratio.iloc[0] == pytest.approx(1.3, abs=0.01)


def test_fit_prior_recovers_simulated_spread():
    rng = np.random.default_rng(1)
    xg_totals = rng.uniform(5, 40, 3000)
    theta = rng.gamma(shape=6.0, scale=1 / 6.0, size=3000)
    goals = rng.poisson(theta * xg_totals)
    prior = finishing.fit_prior(goals, xg_totals)
    assert prior.mu == pytest.approx(1.0, abs=0.05)
    assert 4.0 < prior.alpha < 9.0


def test_spearman_brown_and_length_factor_are_inverse():
    assert reliability.spearman_brown(0.5, 2) == pytest.approx(2 / 3)
    k = reliability.length_factor(0.4, 0.7)
    assert reliability.spearman_brown(0.4, k) == pytest.approx(0.7)


def test_split_half_gives_perfect_reliability_for_constant_rates():
    rows = []
    for player, rate in [(1, 2.0), (2, 5.0), (3, 9.0)]:
        for match in range(10):
            rows.append({"season_id": 4, "player_id": player, "match_id": match, "minutes": 90.0, "pressures": rate})
    a, b, half = reliability.split_half(pd.DataFrame(rows), "pressures", np.random.default_rng(0))
    assert np.corrcoef(a, b)[0, 1] == pytest.approx(1.0)
    assert np.allclose(half, 450.0)


def test_markdown_table():
    text = md_table(pd.DataFrame({"a": [1.23456], "b": ["x"]}), digits=2)
    assert text.splitlines() == ["| a | b |", "| --- | --- |", "| 1.23 | x |"]
