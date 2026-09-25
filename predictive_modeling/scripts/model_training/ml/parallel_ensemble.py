"""Train parallel machine-learning candidates and select an ensemble."""

from __future__ import annotations

import json
import math
import shutil
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

PREDICTIVE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = PREDICTIVE_ROOT.parent
if str(PREDICTIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(PREDICTIVE_ROOT))

from scripts.config import CONFIG as PROJECT_CONFIG
from scripts.preprocessing.split import SequenceData, split_data_folder


CONFIG: dict[str, Any] = {
    "data_folder": str(PREDICTIVE_ROOT / "data" / "merged_clipped"),
    "output_dir": str(PREDICTIVE_ROOT / "output" / "parallel_ensemble_tapering_to_weldpool"),
    "target_col": "tapering_to_weldpool_dist",
    "history_cols": ["voltage", "current"],
    "context_cols": ["wfs"],
    "diagnostic_context_cols": ["wfs", "ctdw"],
    "frame_order_col": "meas_time",
    "history_lag_points": [0, 5, 10, 25, 50],
    "rolling_summary_windows": [5, 25, 50, 75, 100],
    "training_weight_scheme": "condition_subsequence_balanced",
    "clean_output_dir": True,
    "random_state": 42,
    "n_jobs": 4,
    "selection_metric": "RMSE",
    "split": {
        **PROJECT_CONFIG["preprocessing"]["split"],
        "target_col": "tapering_to_weldpool_dist",
        "history_cols": ["voltage", "current"],
        "context_cols": ["wfs"],
        "offline_metadata_cols": ["wfs", "ctdw"],
        "frame_order_col": "meas_time",
    },
    "candidate_grid": {
        "window_size": [0, 25, 50, 75, 100],
        "feature_recipe": ["full", "online_dynamics"],
        "training_trim_start_rows": [0],
        "final_max_train_rows": [160_000],
        "zero_distance_threshold": [0.05],
        "zero_probability_threshold": [0.45, 0.55],
        "target_transform": ["log1p"],
        "classifier": {
            "max_depth": [24],
            "max_features": ["sqrt"],
            "min_samples_leaf": [4],
            "n_estimators": [160],
        },
        "regressor": {
            "max_depth": [24],
            "max_features": [0.6],
            "min_samples_leaf": [4, 8],
            "n_estimators": [160],
        },
    },
    "candidates": [],
}


def shifted(values: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=float)
    if lag == 0:
        out[:] = values
    elif lag < len(values):
        out[lag:] = values[:-lag]
    return out


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    safe_denominator = np.where(np.abs(denominator) > eps, denominator, np.nan)
    return np.nan_to_num(np.asarray(numerator, dtype=float) / safe_denominator, nan=0.0, posinf=0.0, neginf=0.0)


def transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "log1p":
        return np.log1p(np.maximum(values, 0.0))
    if transform in ("none", ""):
        return values
    raise ValueError(f"Unsupported target transform: {transform}")


def inverse_transform_target(values: np.ndarray, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if transform == "log1p":
        return np.expm1(values)
    if transform in ("none", ""):
        return values
    raise ValueError(f"Unsupported target transform: {transform}")


def values_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def expand_param_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(values_list(grid[key]) for key in keys))]


def candidate_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = [dict(candidate) for candidate in config.get("candidates", [])]
    grid = dict(config.get("candidate_grid", {}))
    if not grid:
        return explicit

    classifier_grid = dict(grid.pop("classifier"))
    regressor_grid = dict(grid.pop("regressor"))
    candidates = explicit
    for base, classifier, regressor in product(
        expand_param_grid(grid),
        expand_param_grid(classifier_grid),
        expand_param_grid(regressor_grid),
    ):
        candidate = {
            "model_kind": "two_stage_extra_trees",
            **base,
            "classifier": classifier,
            "regressor": regressor,
        }
        candidate["name"] = (
            "two_stage_extra_trees"
            f"_w{candidate['window_size']}"
            f"_{candidate['feature_recipe']}"
            f"_trim{candidate['training_trim_start_rows']}"
            f"_rows{candidate['final_max_train_rows']}"
            f"_zd{candidate['zero_distance_threshold']}"
            f"_zp{candidate['zero_probability_threshold']}"
            f"_clfleaf{classifier['min_samples_leaf']}"
            f"_regleaf{regressor['min_samples_leaf']}"
            f"_trees{classifier['n_estimators']}"
        )
        candidates.append(candidate)
    if not candidates:
        raise ValueError("No candidates configured.")
    return candidates


class TwoStageDistanceEstimator(BaseEstimator, RegressorMixin):
    """Estimator that separates zero-distance and positive-distance predictions."""
    def __init__(
        self,
        classifier: Any,
        regressor: Any,
        *,
        zero_distance_threshold: float = 0.05,
        zero_probability_threshold: float = 0.55,
        target_transform: str = "log1p",
    ) -> None:
        self.classifier = classifier
        self.regressor = regressor
        self.zero_distance_threshold = zero_distance_threshold
        self.zero_probability_threshold = zero_probability_threshold
        self.target_transform = target_transform

    def fit(
        self,
        x_values: pd.DataFrame | np.ndarray,
        y_values: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "TwoStageDistanceEstimator":
        y = np.asarray(y_values, dtype=float).reshape(-1)
        weights = np.asarray(sample_weight, dtype=float).reshape(-1) if sample_weight is not None else None
        positive = y > float(self.zero_distance_threshold)
        zero_labels = (~positive).astype(int)
        self.constant_zero_probability_: float | None = None
        if len(np.unique(zero_labels)) < 2:
            self.constant_zero_probability_ = float(zero_labels[0]) if len(zero_labels) else 0.0
        else:
            kwargs = {"sample_weight": weights} if weights is not None else {}
            self.classifier.fit(x_values, zero_labels, **kwargs)
        if positive.any():
            kwargs = {"sample_weight": weights[positive]} if weights is not None else {}
            positive_x = x_values.loc[positive] if isinstance(x_values, pd.DataFrame) else np.asarray(x_values)[positive]
            self.regressor.fit(positive_x, transform_target(y[positive], self.target_transform), **kwargs)
            self.fallback_positive_value_ = float(np.mean(y[positive]))
            self.has_positive_regressor_ = True
        else:
            self.fallback_positive_value_ = float(np.mean(y)) if len(y) else 0.0
            self.has_positive_regressor_ = False
        return self

    def predict(self, x_values: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.constant_zero_probability_ is not None:
            zero_probability = np.full(len(x_values), self.constant_zero_probability_, dtype=float)
        else:
            proba = self.classifier.predict_proba(x_values)
            classes = list(getattr(self.classifier, "classes_", [0, 1]))
            zero_probability = proba[:, classes.index(1) if 1 in classes else len(classes) - 1]
        if self.has_positive_regressor_:
            positive_prediction = inverse_transform_target(self.regressor.predict(x_values), self.target_transform)
        else:
            positive_prediction = np.full(len(zero_probability), self.fallback_positive_value_, dtype=float)
        positive_prediction = np.maximum(positive_prediction, 0.0)
        return np.where(zero_probability >= float(self.zero_probability_threshold), 0.0, positive_prediction)


def build_feature_frame(sequences: list[SequenceData], config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    parts: list[pd.DataFrame] = []
    feature_cols: list[str] | None = None
    history_cols = list(config["history_cols"])
    context_cols = list(config["context_cols"])
    diagnostic_context_cols = list(dict.fromkeys([*context_cols, *list(config.get("diagnostic_context_cols", []))]))
    target_col = str(config["target_col"])
    window_size = int(config["window_size"])
    history_lag_points = [
        int(value) for value in config["history_lag_points"] if int(value) <= window_size
    ]
    rolling_windows = [int(value) for value in config["rolling_summary_windows"] if int(value) <= window_size]

    for sequence in sequences:
        df = sequence.df.reset_index(drop=True)
        features: dict[str, np.ndarray] = {}
        current_cols: list[str] = []
        for column in history_cols:
            values = df[column].to_numpy(dtype=float)
            for lag in history_lag_points:
                name = f"{column}_lag_{lag}"
                features[name] = shifted(values, lag)
                current_cols.append(name)
            series = pd.Series(values, dtype="float64")
            for rolling_window in rolling_windows:
                if rolling_window < 2:
                    continue
                rolling = series.rolling(window=rolling_window, min_periods=rolling_window)
                prefix = f"{column}_roll_{rolling_window}"
                oldest = shifted(values, rolling_window - 1)
                summary = {
                    f"{prefix}_mean": rolling.mean().to_numpy(dtype=float),
                    f"{prefix}_std": rolling.std(ddof=0).to_numpy(dtype=float),
                    f"{prefix}_min": rolling.min().to_numpy(dtype=float),
                    f"{prefix}_max": rolling.max().to_numpy(dtype=float),
                    f"{prefix}_delta": values - oldest,
                    f"{prefix}_slope": (values - oldest) / float(rolling_window - 1),
                }
                features.update(summary)
                current_cols.extend(summary)

        for column in context_cols:
            name = f"{column}_setpoint"
            features[name] = df[column].to_numpy(dtype=float)
            current_cols.append(name)

        current = df["current"].to_numpy(dtype=float)
        voltage = df["voltage"].to_numpy(dtype=float)
        power = current * voltage
        engineered = {
            "arc_active_est_feature": ((current > 5.0) | (voltage > 1.0)).astype(float),
            "current_voltage_product": power,
            "current_to_voltage_ratio": safe_divide(current, voltage),
            "voltage_to_current_ratio": safe_divide(voltage, current),
        }
        if config.get("feature_recipe") == "online_dynamics":
            derived_series = {
                "power": power,
                "current_to_voltage_ratio": engineered["current_to_voltage_ratio"],
                "voltage_to_current_ratio": engineered["voltage_to_current_ratio"],
                "current_diff_1": current - shifted(current, 1),
                "voltage_diff_1": voltage - shifted(voltage, 1),
                "power_diff_1": power - shifted(power, 1),
            }
            for name, values in derived_series.items():
                for lag in history_lag_points:
                    feature_name = f"{name}_dyn_lag_{lag}"
                    features[feature_name] = shifted(values, lag)
                    current_cols.append(feature_name)
                series = pd.Series(values, dtype="float64")
                for rolling_window in rolling_windows:
                    if rolling_window < 2:
                        continue
                    rolling = series.rolling(window=rolling_window, min_periods=rolling_window)
                    oldest = shifted(values, rolling_window - 1)
                    summary = {
                        f"{name}_dyn_roll_{rolling_window}_mean": rolling.mean().to_numpy(dtype=float),
                        f"{name}_dyn_roll_{rolling_window}_std": rolling.std(ddof=0).to_numpy(dtype=float),
                        f"{name}_dyn_roll_{rolling_window}_min": rolling.min().to_numpy(dtype=float),
                        f"{name}_dyn_roll_{rolling_window}_max": rolling.max().to_numpy(dtype=float),
                        f"{name}_dyn_roll_{rolling_window}_delta": values - oldest,
                        f"{name}_dyn_roll_{rolling_window}_slope": (values - oldest) / float(rolling_window - 1),
                    }
                    features.update(summary)
                    current_cols.extend(summary)
        if config.get("feature_recipe") == "full" and "wfs" in df.columns:
            wfs = df["wfs"].to_numpy(dtype=float)
            engineered.update(
                {
                    "current_per_wfs": safe_divide(current, wfs),
                    "voltage_per_wfs": safe_divide(voltage, wfs),
                    "power_per_wfs": safe_divide(power, wfs),
                    "wfs_to_current_ratio": safe_divide(wfs, current),
                }
            )
            for rolling_window in rolling_windows:
                current_roll_mean = features.get(f"current_roll_{rolling_window}_mean")
                if current_roll_mean is not None:
                    engineered[f"wfs_current_roll_{rolling_window}_mean"] = wfs * current_roll_mean
        features.update(engineered)
        current_cols.extend(engineered)

        meta = {
            "file": np.full(len(df), sequence.file_name, dtype=object),
            "source_file": np.full(len(df), sequence.source_file, dtype=object),
            "machine_group": np.full(len(df), sequence.machine_group, dtype=object),
            "row_pos": np.arange(len(df), dtype=int),
            "sample_index": df["sample_index"].to_numpy(dtype=int),
            "y_true": df[target_col].to_numpy(dtype=float),
        }
        for column in ("meas_time", "source_row_pos", "source_subsequence_index"):
            if column in df.columns:
                meta[column] = df[column].to_numpy()
        for column in diagnostic_context_cols:
            if column not in df.columns:
                continue
            meta[column] = df[column].to_numpy(dtype=float)

        available_diagnostic_cols = [column for column in diagnostic_context_cols if column in df.columns]
        if available_diagnostic_cols:
            condition_values = []
            for _, row in df[available_diagnostic_cols].iterrows():
                condition_values.append(
                    "|".join(f"{column}={float(row[column]):.3f}" for column in available_diagnostic_cols)
                )
            meta["condition_group"] = np.asarray(condition_values, dtype=object)
        else:
            meta["condition_group"] = np.full(len(df), "all", dtype=object)

        if "subsequence_id" in df.columns:
            meta["subsequence_id"] = df["subsequence_id"].to_numpy()
        else:
            split_config = dict(config["split"])
            stride = max(1, int(split_config.get("subsequence_stride", split_config.get("subsequence_length", 4000))))
            subsequence_index = meta["sample_index"] // stride
            meta["subsequence_id"] = np.asarray(
                [f"{sequence.file_name}::subseq_{int(index):04d}" for index in subsequence_index],
                dtype=object,
            )
        if "subsequence_row_pos" in df.columns:
            meta["subsequence_row_pos"] = df["subsequence_row_pos"].to_numpy(dtype=int)
        else:
            meta["subsequence_row_pos"] = np.arange(len(df), dtype=int)

        frame = pd.concat([pd.DataFrame(meta), pd.DataFrame(features)], axis=1)
        valid = np.isfinite(frame[["y_true", *current_cols]].to_numpy(dtype=float)).all(axis=1)
        frame = frame.loc[valid].reset_index(drop=True)
        if feature_cols is None:
            feature_cols = current_cols
        elif feature_cols != current_cols:
            raise ValueError("Feature columns differ between sequences.")
        if not frame.empty:
            parts.append(frame)

    if feature_cols is None or not parts:
        raise ValueError("No usable feature rows were produced.")
    return pd.concat(parts, ignore_index=True), feature_cols


def training_sample_weight(frame: pd.DataFrame, scheme: str) -> np.ndarray | None:
    if scheme != "condition_subsequence_balanced" or frame.empty:
        return None
    group_sizes = frame.groupby(["condition_group", "subsequence_id"], sort=False)["y_true"].transform("size")
    weights = 1.0 / np.maximum(group_sizes.to_numpy(dtype=float), 1.0)
    return weights / float(np.mean(weights))


def build_pipeline(candidate: dict[str, Any], random_state: int, n_jobs: int) -> Pipeline:
    classifier_params = {**candidate["classifier"], "n_jobs": n_jobs}
    regressor_params = {**candidate["regressor"], "n_jobs": n_jobs}
    model = TwoStageDistanceEstimator(
        ExtraTreesClassifier(random_state=random_state, **classifier_params),
        ExtraTreesRegressor(random_state=random_state, **regressor_params),
        zero_distance_threshold=float(candidate["zero_distance_threshold"]),
        zero_probability_threshold=float(candidate["zero_probability_threshold"]),
        target_transform=str(candidate["target_transform"]),
    )
    return Pipeline([("model", model)])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if len(y_true) == 0:
        return {"n_samples": 0, "RMSE": math.nan, "MAE": math.nan, "R2": math.nan, "bias": math.nan, "MAPE": math.nan}
    error = y_pred - y_true
    nonzero = np.abs(y_true) > 1.0e-8
    return {
        "n_samples": int(len(y_true)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else math.nan,
        "bias": float(np.mean(error)),
        "MAPE": float(np.mean(np.abs(error[nonzero] / y_true[nonzero])) * 100.0) if nonzero.any() else math.nan,
    }


def compute_zero_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    zero_threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if len(y_true) == 0:
        return {
            "zero_true_rate": math.nan,
            "zero_pred_rate": math.nan,
            "zero_accuracy": math.nan,
            "zero_precision": math.nan,
            "zero_recall": math.nan,
            "zero_f1": math.nan,
        }
    true_zero = y_true <= float(zero_threshold)
    pred_zero = y_pred <= float(zero_threshold)
    true_positive = float(np.sum(true_zero & pred_zero))
    false_positive = float(np.sum(~true_zero & pred_zero))
    false_negative = float(np.sum(true_zero & ~pred_zero))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else math.nan
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else math.nan
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall and not math.isnan(precision + recall) else math.nan
    return {
        "zero_true_rate": float(np.mean(true_zero)),
        "zero_pred_rate": float(np.mean(pred_zero)),
        "zero_accuracy": float(np.mean(true_zero == pred_zero)),
        "zero_precision": float(precision),
        "zero_recall": float(recall),
        "zero_f1": float(f1),
    }


def compute_high_distance_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    high_distance_threshold: float = 3.0,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    mask = finite & (y_true >= float(high_distance_threshold))
    metrics = compute_metrics(y_true[mask], y_pred[mask])
    return {f"high_distance_{key}": value for key, value in metrics.items()}


def candidate_diagnostic_row(
    candidate: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    role: str,
) -> dict[str, Any]:
    zero_threshold = float(candidate["zero_distance_threshold"])
    return {
        "candidate": candidate["name"],
        "role": role,
        **compute_zero_diagnostics(y_true, y_pred, zero_threshold),
        **compute_high_distance_metrics(y_true, y_pred),
    }


def condition_diagnostic_rows(
    candidate: dict[str, Any],
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    role: str,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    pred = pd.Series(np.asarray(y_pred, dtype=float), index=frame.index)
    for condition_group, group in frame.groupby("condition_group", sort=False):
        group_pred = pred.loc[group.index].to_numpy(dtype=float)
        group_true = group["y_true"].to_numpy(dtype=float)
        metrics = compute_metrics(group_true, group_pred)
        diagnostics.append(
            {
                "candidate": candidate["name"],
                "role": role,
                "condition_group": condition_group,
                **metrics,
                **compute_zero_diagnostics(group_true, group_pred, float(candidate["zero_distance_threshold"])),
                **compute_high_distance_metrics(group_true, group_pred),
            }
        )
    return diagnostics


def predict_frame(model: Pipeline, frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = frame[
        [
            "file",
            "source_file",
            "machine_group",
            "row_pos",
            "sample_index",
            "subsequence_id",
            "subsequence_row_pos",
            "condition_group",
            "y_true",
            *[column for column in ("meas_time", "wfs", "ctdw") if column in frame.columns],
        ]
    ].copy()
    out["y_pred"] = model.predict(frame[feature_cols])
    out["error"] = out["y_pred"] - out["y_true"]
    return out


def feature_group(feature: str) -> str:
    if "voltage" in feature and "current" in feature:
        return "voltage_current_interaction"
    if feature.startswith("voltage_") or feature.startswith("voltage"):
        return "voltage"
    if feature.startswith("current_") or feature.startswith("current"):
        return "current"
    if feature.startswith("wfs_") or feature.startswith("wfs"):
        return "wfs"
    if feature.startswith("ctdw_") or feature.startswith("ctdw"):
        return "ctdw"
    if "wfs" in feature:
        return "wfs_interaction"
    return "other"


def save_feature_importances(output_dir: Path, model: Pipeline, feature_cols: list[str]) -> None:
    estimator = model.named_steps["model"]
    classifier_importance = np.asarray(
        getattr(estimator.classifier, "feature_importances_", np.zeros(len(feature_cols))),
        dtype=float,
    )
    regressor_importance = np.asarray(
        getattr(estimator.regressor, "feature_importances_", np.zeros(len(feature_cols))),
        dtype=float,
    )
    classifier_norm = classifier_importance / classifier_importance.sum() if classifier_importance.sum() > 0 else classifier_importance
    regressor_norm = regressor_importance / regressor_importance.sum() if regressor_importance.sum() > 0 else regressor_importance
    combined = (classifier_norm + regressor_norm) / 2.0
    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "group": [feature_group(feature) for feature in feature_cols],
            "classifier_importance": classifier_importance,
            "regressor_importance": regressor_importance,
            "combined_importance": combined,
        }
    ).sort_values("combined_importance", ascending=False, kind="stable")
    importances.to_csv(output_dir / "feature_importances.csv", index=False)
    grouped = (
        importances.groupby("group", as_index=False)[
            ["classifier_importance", "regressor_importance", "combined_importance"]
        ]
        .sum()
        .sort_values("combined_importance", ascending=False, kind="stable")
    )
    grouped.to_csv(output_dir / "feature_importances_grouped.csv", index=False)


def run(config: dict[str, Any]) -> pd.DataFrame:
    output_dir = Path(config["output_dir"])
    if bool(config["clean_output_dir"]) and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = split_data_folder(config["data_folder"], config=config["split"])
    split.manifest.to_csv(output_dir / "split_manifest.csv", index=False)

    candidates = candidate_specs(config)
    candidate_rows = [
        {
            "candidate": candidate["name"],
            "model_kind": candidate["model_kind"],
            "window_size": candidate["window_size"],
            "feature_recipe": candidate["feature_recipe"],
            "training_trim_start_rows": candidate["training_trim_start_rows"],
            "final_max_train_rows": candidate["final_max_train_rows"],
            "zero_distance_threshold": candidate["zero_distance_threshold"],
            "zero_probability_threshold": candidate["zero_probability_threshold"],
            "target_transform": candidate["target_transform"],
            "classifier_json": json.dumps(candidate["classifier"], sort_keys=True),
            "regressor_json": json.dumps(candidate["regressor"], sort_keys=True),
        }
        for candidate in candidates
    ]
    pd.DataFrame(candidate_rows).to_csv(output_dir / "tested_candidates.csv", index=False)

    results: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    candidate_condition_diagnostics: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None
    best_model: Pipeline | None = None
    best_feature_cols: list[str] | None = None
    best_metric = math.inf
    for candidate_index, candidate in enumerate(candidates, start=1):
        start_time = time.perf_counter()
        print(f"[{candidate_index}/{len(candidates)}] Fitting {candidate['name']}", flush=True)
        candidate_config = {**config, **candidate}
        train_frame, feature_cols = build_feature_frame(split.train, candidate_config)
        validation_frame, validation_feature_cols = build_feature_frame(split.validation, candidate_config)
        if feature_cols != validation_feature_cols:
            raise ValueError("Train and validation features differ.")

        trim_start = max(0, int(candidate["training_trim_start_rows"]))
        if trim_start:
            train_frame = train_frame[train_frame["subsequence_row_pos"].astype(int) >= trim_start].reset_index(drop=True)
        if int(candidate.get("final_max_train_rows", 0)) > 0 and len(train_frame) > int(candidate["final_max_train_rows"]):
            train_frame = train_frame.sample(n=int(candidate["final_max_train_rows"]), random_state=int(config["random_state"]))

        weights = training_sample_weight(train_frame, str(config["training_weight_scheme"]))
        model = build_pipeline(candidate, int(config["random_state"]), int(config["n_jobs"]))
        fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
        model.fit(train_frame[feature_cols], train_frame["y_true"].to_numpy(dtype=float), **fit_kwargs)

        train_predictions = model.predict(train_frame[feature_cols])
        validation_predictions = model.predict(validation_frame[feature_cols])
        train_metrics = compute_metrics(train_frame["y_true"].to_numpy(dtype=float), train_predictions)
        validation_metrics = compute_metrics(validation_frame["y_true"].to_numpy(dtype=float), validation_predictions)
        train_diagnostics = candidate_diagnostic_row(
            candidate,
            train_frame["y_true"].to_numpy(dtype=float),
            train_predictions,
            role="train",
        )
        validation_diagnostics = candidate_diagnostic_row(
            candidate,
            validation_frame["y_true"].to_numpy(dtype=float),
            validation_predictions,
            role="validation",
        )
        candidate_diagnostics.extend([train_diagnostics, validation_diagnostics])
        candidate_condition_diagnostics.extend(
            condition_diagnostic_rows(candidate, validation_frame, validation_predictions, role="validation")
        )
        row = {
            "candidate": candidate["name"],
            "model_kind": candidate["model_kind"],
            "window_size": candidate["window_size"],
            "feature_recipe": candidate["feature_recipe"],
            "training_trim_start_rows": candidate["training_trim_start_rows"],
            "final_max_train_rows": candidate["final_max_train_rows"],
            "n_features": len(feature_cols),
            "elapsed_seconds": round(time.perf_counter() - start_time, 3),
            "classifier_json": json.dumps(candidate["classifier"], sort_keys=True),
            "regressor_json": json.dumps(candidate["regressor"], sort_keys=True),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_diagnostics.items() if key not in ("candidate", "role")},
        }
        results.append(row)
        metric = float(validation_metrics[str(config["selection_metric"])])
        print(
            f"[{candidate_index}/{len(candidates)}] "
            f"validation_{config['selection_metric']}={metric:.6f} "
            f"elapsed_seconds={row['elapsed_seconds']}",
            flush=True,
        )
        if metric < best_metric:
            best_metric = metric
            best_candidate = candidate
            best_model = model
            best_feature_cols = feature_cols

    tuning_results = pd.DataFrame(results).sort_values(f"validation_{config['selection_metric']}", kind="stable")
    tuning_results.to_csv(output_dir / "tuning_results.csv", index=False)
    tuning_results.to_csv(output_dir / "candidate_log.csv", index=False)
    pd.DataFrame(candidate_diagnostics).to_csv(output_dir / "candidate_diagnostics.csv", index=False)
    pd.DataFrame(candidate_condition_diagnostics).to_csv(
        output_dir / "candidate_condition_diagnostics.csv",
        index=False,
    )
    if best_candidate is None or best_model is None or best_feature_cols is None:
        raise ValueError("No model was fitted.")
    best_name = str(best_candidate["name"])
    feature_cols = best_feature_cols
    best_config = {**config, **best_candidate}

    frames = {
        "train": build_feature_frame(split.train, best_config)[0],
        "validation": build_feature_frame(split.validation, best_config)[0],
        "test": build_feature_frame(split.test, best_config)[0],
    }
    final_condition_diagnostics: list[dict[str, Any]] = []
    for role, frame in frames.items():
        predictions = predict_frame(best_model, frame, feature_cols)
        metrics = compute_metrics(predictions["y_true"].to_numpy(dtype=float), predictions["y_pred"].to_numpy(dtype=float))
        diagnostics = candidate_diagnostic_row(
            best_candidate,
            predictions["y_true"].to_numpy(dtype=float),
            predictions["y_pred"].to_numpy(dtype=float),
            role=role,
        )
        final_condition_diagnostics.extend(
            condition_diagnostic_rows(best_candidate, frame, predictions["y_pred"].to_numpy(dtype=float), role=role)
        )
        predictions.to_csv(output_dir / f"{role}_predictions.csv", index=False)
        pd.DataFrame([{**{"candidate": best_name}, **metrics, **diagnostics}]).to_csv(
            output_dir / f"{role}_metrics.csv",
            index=False,
        )
    pd.DataFrame(final_condition_diagnostics).to_csv(output_dir / "selected_condition_diagnostics.csv", index=False)
    save_feature_importances(output_dir, best_model, feature_cols)

    resolved = {
        **{key: value for key, value in config.items() if key not in ("candidates", "candidate_grid")},
        "candidate_grid": config.get("candidate_grid", {}),
        "resolved_candidates": candidates,
        "selected_candidate": best_candidate,
        "selected_feature_count": len(feature_cols),
        "split_note": split.note,
    }
    (output_dir / "config_resolved.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    print(f"Selected candidate: {best_name}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Output directory: {output_dir}")
    print(tuning_results.to_string(index=False))
    return tuning_results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit("Configure this script by editing CONFIG; command-line arguments are not supported.")
    run(dict(CONFIG))
