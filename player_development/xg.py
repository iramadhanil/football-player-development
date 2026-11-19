"""Expected-goals (xG) model for non-penalty shots.

Two scikit-learn models are compared with 5-fold cross-validation grouped by match on the
2018/19-2020/21 seasons. The better one is then tested once on the untouched 2023/24 season and
compared with StatsBomb's own xG. Out-of-fold predictions for every shot feed the finishing model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config

NUMERIC = ["log_distance", "angle", "n_def_in_cone", "n_opp_within_3", "gk_dist_to_goal"]
BINARY = ["is_header", "other_body_part", "first_time", "under_pressure", "one_on_one", "open_goal",
          "follows_dribble", "assisted", "assist_through_ball", "assist_cross", "assist_cut_back",
          "gk_in_cone", "frame_missing"]
CATEGORICAL = ["shot_type", "technique", "play_pattern"]
FEATURES = NUMERIC + BINARY + CATEGORICAL


def prepare(shots: pd.DataFrame) -> pd.DataFrame:
    """Non-penalty shots with model features. Missing freeze frames are imputed and flagged."""
    df = shots[shots.shot_type != "Penalty"].copy()
    df["frame_missing"] = ~df.has_freeze_frame.astype(bool)
    df["log_distance"] = np.log1p(df.distance)
    df["is_header"] = df.body_part == "Head"
    df["other_body_part"] = df.body_part == "Other"
    df["gk_in_cone"] = df.gk_in_cone.fillna(False).astype(bool)
    for col in ["n_def_in_cone", "n_opp_within_3", "gk_dist_to_goal"]:
        df[col] = df[col].astype(float).fillna(df[col].median())
    df["shot_type"] = df.shot_type.where(df.shot_type.isin(["Open Play", "Free Kick"]), "Other set piece")
    for col in BINARY:
        df[col] = df[col].astype(bool).astype(int)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str)
    return df.reset_index(drop=True)


def make_models() -> dict[str, Pipeline]:
    encoder = dict(handle_unknown="infrequent_if_exist", min_frequency=20)
    logistic = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), NUMERIC + BINARY),
            ("cat", OneHotEncoder(**encoder), CATEGORICAL),
        ])),
        ("model", LogisticRegression(C=1.0, max_iter=5000)),
    ])
    boosting = Pipeline([
        ("prep", ColumnTransformer([
            ("num", "passthrough", NUMERIC + BINARY),
            ("cat", OneHotEncoder(sparse_output=False, **encoder), CATEGORICAL),
        ])),
        ("model", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=50,
            l2_regularization=1.0, random_state=config.RANDOM_STATE)),
    ])
    return {"logistic_regression": logistic, "gradient_boosting": boosting}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Shot-weighted gap between predicted and observed goal rates across quantile bins."""
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    gap = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            gap += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(gap)


def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"shots": int(len(y)), "goals": int(y.sum()), "xg_total": float(p.sum()),
            "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "roc_auc": roc_auc_score(y, p), "calibration_error": expected_calibration_error(y, p)}


def out_of_fold(model: Pipeline, df: pd.DataFrame, folds: int = 5) -> np.ndarray:
    """Predictions for each shot from a model that never saw that shot's match."""
    oof = np.zeros(len(df))
    for train, test in GroupKFold(n_splits=folds).split(df, df.is_goal, groups=df.match_id):
        fitted = model.fit(df.iloc[train][FEATURES], df.iloc[train].is_goal)
        oof[test] = fitted.predict_proba(df.iloc[test][FEATURES])[:, 1]
    return oof


@dataclass
class XGResult:
    metrics: pd.DataFrame
    selected: str
    shots: pd.DataFrame
    importance: pd.DataFrame
    coefficients: pd.DataFrame


def run(shots: pd.DataFrame) -> XGResult:
    df = prepare(shots)
    train = df[df.season_id.isin(config.TRAIN_SEASONS)].reset_index(drop=True)
    test = df[df.season_id == config.HOLDOUT_SEASON].reset_index(drop=True)
    rows = []
    cv_preds = {}
    for name, model in make_models().items():
        cv_preds[name] = out_of_fold(model, train)
        rows.append({"model": name, "evaluation": "5-fold CV by match, 2018/19-2020/21",
                     **score(train.is_goal.to_numpy(), cv_preds[name])})
    rows.append({"model": "statsbomb_xg (benchmark)", "evaluation": "5-fold CV by match, 2018/19-2020/21",
                 **score(train.is_goal.to_numpy(), train.statsbomb_xg.to_numpy())})
    selected = min(cv_preds, key=lambda n: log_loss(train.is_goal, np.clip(cv_preds[n], 1e-6, 1 - 1e-6)))

    fitted = {}
    for name, model in make_models().items():
        fitted[name] = model.fit(train[FEATURES], train.is_goal)
        rows.append({"model": name, "evaluation": "held-out season 2023/24",
                     **score(test.is_goal.to_numpy(), fitted[name].predict_proba(test[FEATURES])[:, 1])})
    rows.append({"model": "statsbomb_xg (benchmark)", "evaluation": "held-out season 2023/24",
                 **score(test.is_goal.to_numpy(), test.statsbomb_xg.to_numpy())})
    metrics = pd.DataFrame(rows)

    test = test.assign(xg_holdout=fitted[selected].predict_proba(test[FEATURES])[:, 1])
    imp = permutation_importance(fitted[selected], test[FEATURES], test.is_goal, scoring="neg_log_loss",
                                 n_repeats=10, random_state=config.RANDOM_STATE)
    importance = (pd.DataFrame({"feature": FEATURES, "log_loss_increase": imp.importances_mean,
                                "std": imp.importances_std})
                  .sort_values("log_loss_increase", ascending=False).reset_index(drop=True))

    lr = fitted["logistic_regression"]
    names = lr.named_steps["prep"].get_feature_names_out()
    coefficients = (pd.DataFrame({"term": [n.split("__", 1)[1] for n in names],
                                  "coefficient": lr.named_steps["model"].coef_[0]})
                    .assign(odds_ratio=lambda d: np.exp(d.coefficient))
                    .sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True))

    df = df.assign(xg_model=out_of_fold(make_models()[selected], df))
    df = df.merge(test[["shot_id", "xg_holdout"]], on="shot_id", how="left")
    return XGResult(metrics, selected, df, importance, coefficients)


def plot_calibration(result: XGResult, path) -> None:
    test = result.shots[result.shots.season_id == config.HOLDOUT_SEASON]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 0.8], [0, 0.8], color="#999999", lw=1, ls="--", label="perfect calibration")
    for col, label, color in [("xg_holdout", f"this model ({result.selected.replace('_', ' ')})", "#1c5d99"),
                              ("statsbomb_xg", "StatsBomb xG", "#e07a1f")]:
        obs, pred = calibration_curve(test.is_goal, test[col], n_bins=10, strategy="quantile")
        ax.plot(pred, obs, marker="o", color=color, label=label)
    ax.set_xlabel("Predicted goal probability (decile average)")
    ax.set_ylabel("Observed goal rate")
    ax.set_title("xG calibration on the held-out 2023/24 season")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(result: XGResult, path, top: int = 12) -> None:
    imp = result.importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.barh(imp.feature, imp.log_loss_increase, xerr=imp["std"], color="#1c5d99")
    ax.set_xlabel("Increase in log loss when the feature is shuffled")
    ax.set_title("What drives the xG model\n(permutation importance on the held-out 2023/24 season)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save(result: XGResult) -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result.metrics.round(4).to_csv(config.REPORTS_DIR / "xg_metrics.csv", index=False)
    result.importance.round(5).to_csv(config.REPORTS_DIR / "xg_permutation_importance.csv", index=False)
    result.coefficients.round(4).to_csv(config.REPORTS_DIR / "xg_logistic_coefficients.csv", index=False)
    (config.REPORTS_DIR / "xg_selected_model.json").write_text(json.dumps({"selected": result.selected}, indent=2) + "\n")
    result.shots.to_csv(config.PROCESSED_DIR / "shots_xg.csv", index=False)
    plot_calibration(result, config.FIGURES_DIR / "xg_calibration.png")
    plot_importance(result, config.FIGURES_DIR / "xg_importance.png")
