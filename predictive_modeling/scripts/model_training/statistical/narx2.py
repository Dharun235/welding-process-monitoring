"""
Additive-power nonlinear ARX identification using the ARX data and evaluation path.

The model starts with the exact lag vector used by ``arx.py`` and can expand
each regressor independently:

    phi_bounded(t) = [1, z_1, ..., z_p, z_1^2, ..., z_p^2, ...]

where ``z`` is the standardized ARX lag vector. Unlike a complete polynomial
expansion, no cross-products are generated. An exogenous-only nonlinear mode
keeps autoregressive output lags linear while expanding measured-input terms.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictive_modeling.scripts.model_training.statistical import arx


DEFAULT_DATA_FOLDER = arx.DEFAULT_DATA_FOLDER
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "narx2" / "wiretip_to_weldpool"
DEFAULT_TARGET_COL = "wiretip_to_weldpool_dist"
DEFAULT_DEGREE_VALUES = [1, 2]
DEFAULT_ALPHA_VALUES = [1.0e-6]
DEFAULT_ORDER_VALUES = [0, 1, 3, 5]
DEFAULT_MAX_EXPANDED_TERMS = 10_000
SCALE_EPS = 1.0e-12

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundedNARXModel:
    coefficients: np.ndarray
    base_mean: np.ndarray
    base_scale: np.ndarray
    linear_feature_names: list[str]
    degree: int
    nonlinear_scope: str
    higher_power_start: int
    alpha: float
    prediction_bounds: tuple[float, float]
    clip_predictions: bool = False

    @property
    def n_params(self) -> int:
        return int(len(self.coefficients))


def parse_int_list(value: str, label: str, *, minimum: int = 0) -> list[int]:
    try:
        values = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must contain comma-separated integers.") from exc
    if not values or any(item < minimum for item in values):
        raise argparse.ArgumentTypeError(f"{label} must contain values >= {minimum}.")
    return values


def parse_float_list(value: str, label: str, *, minimum: float = 0.0) -> list[float]:
    try:
        values = sorted({float(item.strip()) for item in str(value).split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must contain comma-separated numbers.") from exc
    if not values or any(item < minimum for item in values):
        raise argparse.ArgumentTypeError(f"{label} must contain values >= {minimum}.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit an additive-power NARX model using the same split, live features, "
            "lag construction, and recursive evaluation convention as arx.py."
        )
    )
    parser.add_argument("--data_folder", "--data-folder", dest="data_folder", type=Path, default=DEFAULT_DATA_FOLDER)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target_col",
        "--target-col",
        dest="target_col",
        choices=["tapering_to_weldpool_dist", "wiretip_to_weldpool_dist"],
        default=DEFAULT_TARGET_COL,
    )
    parser.add_argument(
        "--feature_set",
        "--feature-set",
        dest="feature_set",
        choices=["live_engineered", "raw_electrical", "specified_inputs"],
        default="live_engineered",
    )
    parser.add_argument("--summary_window", "--summary-window", dest="summary_window", type=int, default=None)
    parser.add_argument("--machine_type", "--machine-type", dest="machine_type", choices=["we", "fr"], default=None)
    parser.add_argument("--wfs", type=float, default=None)
    parser.add_argument("--ctdw", type=float, default=None)
    parser.add_argument("--na_values", "--na-values", dest="na_values", default="0,1,3,5")
    parser.add_argument("--nb_values", "--nb-values", dest="nb_values", default="1,3,5")
    parser.add_argument("--nk_min", "--nk-min", dest="nk_min", type=int, default=0)
    parser.add_argument("--nk_max", "--nk-max", dest="nk_max", type=int, default=1)
    parser.add_argument("--degree_values", "--degree-values", dest="degree_values", default="1,2")
    parser.add_argument("--alpha_values", "--alpha-values", dest="alpha_values", default="1e-6")
    parser.add_argument(
        "--nonlinear_scope",
        "--nonlinear-scope",
        dest="nonlinear_scope",
        choices=["all", "exogenous_only"],
        default="exogenous_only",
        help="Terms receiving powers above one; exogenous_only keeps autoregressive feedback linear.",
    )
    parser.add_argument(
        "--max_expanded_terms",
        "--max-expanded-terms",
        dest="max_expanded_terms",
        type=int,
        default=DEFAULT_MAX_EXPANDED_TERMS,
        help="Maximum fitted terms including intercept; use 0 to disable the guard.",
    )
    parser.add_argument(
        "--prediction_margin",
        "--prediction-margin",
        dest="prediction_margin",
        type=float,
        default=0.05,
        help="Fraction of training target range allowed outside recursive prediction bounds.",
    )
    clipping_group = parser.add_mutually_exclusive_group()
    clipping_group.add_argument(
        "--enable_prediction_clipping",
        "--enable-prediction-clipping",
        dest="clip_predictions",
        action="store_true",
        help="Clip predictions to bounds derived from the training target (deployment safeguard, not evaluation).",
    )
    clipping_group.add_argument(
        "--disable_prediction_clipping",
        "--disable-prediction-clipping",
        dest="clip_predictions",
        action="store_false",
        help="Legacy alias for the default unclipped prediction behavior.",
    )
    parser.set_defaults(clip_predictions=False)
    parser.add_argument(
        "--selection_mode",
        "--selection-mode",
        dest="selection_mode",
        choices=sorted(arx.SELECTION_MODE_LABELS),
        default="free_run",
    )
    parser.add_argument(
        "--selection_metric",
        "--selection-metric",
        dest="selection_metric",
        choices=sorted(arx.SELECTION_METRIC_LABELS),
        default="rmse",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.na_values = parse_int_list(args.na_values, "--na-values")
    args.nb_values = parse_int_list(args.nb_values, "--nb-values")
    args.degree_values = parse_int_list(args.degree_values, "--degree-values", minimum=1)
    args.alpha_values = parse_float_list(args.alpha_values, "--alpha-values")
    if args.nk_min < 0 or args.nk_max < args.nk_min:
        raise ValueError("--nk-min and --nk-max must define a non-negative increasing range.")
    if args.max_expanded_terms < 0:
        raise ValueError("--max-expanded-terms must be non-negative. Use 0 to disable.")
    if args.prediction_margin < 0.0:
        raise ValueError("--prediction-margin must be non-negative.")
    if args.summary_window is not None and args.summary_window < 2:
        raise ValueError("--summary-window must be >= 2.")
    return args


def expanded_term_count(n_linear_terms: int, degree: int, higher_power_start: int = 0) -> int:
    """Return bounded nonlinear parameter count, including an intercept."""
    if not 0 <= higher_power_start <= n_linear_terms:
        raise ValueError("higher_power_start must index the linear regressor columns.")
    return 1 + n_linear_terms + (degree - 1) * (n_linear_terms - higher_power_start)


def expanded_feature_names(linear_names: list[str], degree: int, higher_power_start: int = 0) -> list[str]:
    if not 0 <= higher_power_start <= len(linear_names):
        raise ValueError("higher_power_start must index the linear regressor names.")
    names = ["intercept", *linear_names]
    for power in range(2, degree + 1):
        names.extend(f"{name}^{power}" for name in linear_names[higher_power_start:])
    return names


def expand_terms(
    phi: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    degree: int,
    higher_power_start: int = 0,
) -> np.ndarray:
    """Construct the additive-power information matrix without interactions."""
    if phi.ndim != 2:
        raise ValueError("The base regressor matrix must be two-dimensional.")
    if not 0 <= higher_power_start <= phi.shape[1]:
        raise ValueError("higher_power_start must index the base regressor columns.")
    z = (np.asarray(phi, dtype=float) - mean) / scale
    parts = [np.ones((len(z), 1), dtype=float), z]
    parts.extend(np.power(z[:, higher_power_start:], power) for power in range(2, degree + 1))
    return np.hstack(parts)


def valid_base_regressors(
    datasets: list[tuple[str, pd.DataFrame]],
    na: int,
    nb: int,
    nk: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    matrices: list[tuple[np.ndarray, np.ndarray]] = []
    live_feature_names: list[str] = []
    for _, df in datasets:
        y = df[arx.TARGET_COL].to_numpy(dtype=float)
        inputs, names = arx.build_live_feature_matrix(df, nb)
        if not live_feature_names:
            live_feature_names = names
        phi, target = arx.build_regressor_matrix(y, inputs, na, nb, nk)
        if len(target) == 0:
            continue
        valid = np.isfinite(phi).all(axis=1) & np.isfinite(target)
        if valid.any():
            matrices.append((np.ascontiguousarray(phi[valid]), np.ascontiguousarray(target[valid])))
    return matrices, arx.arx_coefficient_names(na, nb, nk, live_feature_names)


def fit_bounded_narx(
    datasets: list[tuple[str, pd.DataFrame]],
    na: int,
    nb: int,
    nk: int,
    degree: int,
    alpha: float,
    max_expanded_terms: int,
    prediction_margin: float,
    clip_predictions: bool = False,
    nonlinear_scope: str = "exogenous_only",
) -> tuple[BoundedNARXModel | None, int, str]:
    if nonlinear_scope not in {"all", "exogenous_only"}:
        raise ValueError("nonlinear_scope must be 'all' or 'exogenous_only'.")
    matrices, linear_names = valid_base_regressors(datasets, na, nb, nk)
    if not matrices:
        return None, 0, "no finite training regressors"

    higher_power_start = na if nonlinear_scope == "exogenous_only" else 0
    n_params = expanded_term_count(len(linear_names), degree, higher_power_start)
    if max_expanded_terms and n_params > max_expanded_terms:
        return None, sum(len(target) for _, target in matrices), (
            f"bounded feature budget exceeded ({n_params:,} > {max_expanded_terms:,} terms)"
        )

    n_samples = sum(len(target) for _, target in matrices)
    sums = np.sum([phi.sum(axis=0) for phi, _ in matrices], axis=0)
    square_sums = np.sum([(phi * phi).sum(axis=0) for phi, _ in matrices], axis=0)
    mean = sums / n_samples
    variance = np.maximum(square_sums / n_samples - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale = np.where(scale > SCALE_EPS, scale, 1.0)

    gram = np.zeros((n_params, n_params), dtype=float)
    rhs = np.zeros(n_params, dtype=float)
    for phi, target in matrices:
        expanded = expand_terms(phi, mean, scale, degree, higher_power_start)
        gram += expanded.T @ expanded
        rhs += expanded.T @ target
    regularized = gram.copy()
    regularized.flat[:: n_params + 1] += alpha
    regularized[0, 0] -= alpha
    coefficients = arx.solve_normal_equations(regularized, rhs)
    observed_target = np.concatenate([target for _, target in matrices])
    target_min = float(np.min(observed_target))
    target_max = float(np.max(observed_target))
    margin = max(target_max - target_min, SCALE_EPS) * prediction_margin
    model = BoundedNARXModel(
        coefficients=coefficients,
        base_mean=mean,
        base_scale=scale,
        linear_feature_names=linear_names,
        degree=degree,
        nonlinear_scope=nonlinear_scope,
        higher_power_start=higher_power_start,
        alpha=alpha,
        prediction_bounds=(target_min - margin, target_max + margin),
        clip_predictions=clip_predictions,
    )
    return model, n_samples, "ok"


def apply_prediction_bounds(prediction: np.ndarray | float, model: BoundedNARXModel) -> np.ndarray | float:
    if model.clip_predictions:
        return np.clip(prediction, *model.prediction_bounds)
    return prediction


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_params: int) -> dict[str, float]:
    """Score a full trajectory, treating numerical divergence as model failure."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size and (
        y_true.size != y_pred.size
        or not np.isfinite(y_true).all()
        or not np.isfinite(y_pred).all()
    ):
        return {
            "RMSE": float("inf"),
            "MAE": float("inf"),
            "R2": float("-inf"),
            "fit_pct": float("-inf"),
            "AIC": float("inf"),
            "BIC": float("inf"),
        }
    return arx.compute_metrics(y_true, y_pred, n_params=n_params)


def predict_one_step(
    datasets: list[tuple[str, pd.DataFrame]],
    model: BoundedNARXModel,
    na: int,
    nb: int,
    nk: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    for phi, target in valid_base_regressors(datasets, na, nb, nk)[0]:
        y_true_parts.append(target)
        prediction = expand_terms(
            phi, model.base_mean, model.base_scale, model.degree, model.higher_power_start
        ) @ model.coefficients
        y_pred_parts.append(np.asarray(apply_prediction_bounds(prediction, model), dtype=float))
    if not y_true_parts:
        return np.empty(0), np.empty(0)
    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts)


def build_base_row(y_feedback: np.ndarray, inputs: np.ndarray, t: int, na: int, nb: int, nk: int) -> np.ndarray:
    values = [y_feedback[t - lag] for lag in range(1, na + 1)]
    if arx.FEATURE_SET == "specified_inputs":
        values.extend(inputs[t - nk, :].tolist())
    else:
        for input_index in range(inputs.shape[1]):
            values.extend(inputs[t - nk - offset, input_index] for offset in range(nb))
    return np.asarray(values, dtype=float).reshape(1, -1)


def simulate_free_run_series(
    y: np.ndarray,
    inputs: np.ndarray,
    model: BoundedNARXModel,
    na: int,
    nb: int,
    nk: int,
) -> tuple[int, np.ndarray | None]:
    t_start = arx.compute_t_start(na, nb, nk)
    if t_start >= len(y):
        return t_start, None
    predicted = np.empty(len(y), dtype=float)
    predicted[:t_start] = np.where(np.isfinite(y[:t_start]), y[:t_start], 0.0)
    for t in range(t_start, len(y)):
        phi = build_base_row(predicted, inputs, t, na, nb, nk)
        if not np.isfinite(phi).all():
            predicted[t] = np.nan
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            expanded = expand_terms(phi, model.base_mean, model.base_scale, model.degree, model.higher_power_start)
            predicted[t] = float((expanded @ model.coefficients).item())
        if not np.isfinite(predicted[t]):
            predicted[t:] = np.nan
            break
        predicted[t] = float(apply_prediction_bounds(predicted[t], model))
    return t_start, predicted


def predict_free_run(
    datasets: list[tuple[str, pd.DataFrame]],
    model: BoundedNARXModel,
    na: int,
    nb: int,
    nk: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    for _, df in datasets:
        y = df[arx.TARGET_COL].to_numpy(dtype=float)
        inputs, _ = arx.build_live_feature_matrix(df, nb)
        t_start, predicted = simulate_free_run_series(y, inputs, model, na, nb, nk)
        if predicted is None:
            continue
        observed_phi, target = arx.build_regressor_matrix(y, inputs, na, nb, nk)
        eligible = np.isfinite(observed_phi).all(axis=1) & np.isfinite(target)
        y_true_parts.append(target[eligible])
        y_pred_parts.append(predicted[t_start:][eligible])
    if not y_true_parts:
        return np.empty(0), np.empty(0)
    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts)


def predictions_by_mode(
    datasets: list[tuple[str, pd.DataFrame]],
    model: BoundedNARXModel,
    na: int,
    nb: int,
    nk: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "one_step_ahead":
        return predict_one_step(datasets, model, na, nb, nk)
    return predict_free_run(datasets, model, na, nb, nk)


def build_aligned_prediction_rows(
    datasets: list[tuple[str, pd.DataFrame]],
    model: BoundedNARXModel,
    na: int,
    nb: int,
    nk: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for file_name, df in datasets:
        y = df[arx.TARGET_COL].to_numpy(dtype=float)
        inputs, _ = arx.build_live_feature_matrix(df, nb)
        t_start = arx.compute_t_start(na, nb, nk)
        osa = np.full(len(y), np.nan)
        free_run = np.full(len(y), np.nan)
        phi, target = arx.build_regressor_matrix(y, inputs, na, nb, nk)
        if len(target):
            valid = np.isfinite(phi).all(axis=1) & np.isfinite(target)
            osa_part = np.full(len(target), np.nan)
            osa_part[valid] = apply_prediction_bounds(expand_terms(
                phi[valid], model.base_mean, model.base_scale, model.degree, model.higher_power_start
            ) @ model.coefficients, model)
            osa[t_start:] = osa_part
        _, simulated = simulate_free_run_series(y, inputs, model, na, nb, nk)
        if simulated is not None:
            free_run[t_start:] = simulated[t_start:]
        frame = pd.DataFrame(
            {
                "file": file_name,
                "sample_index": np.arange(len(y)),
                "has_full_history": np.arange(len(y)) >= t_start,
                "y_true": y,
                "y_pred_osa": osa,
                "y_pred_freerun": free_run,
            }
        )
        for column in ("voltage", "current", "delta_voltage", "delta_current", "wfs", "meas_time", "frame_number"):
            if column in df:
                frame[column] = df[column].to_numpy()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def grid_search(
    train_set: list[tuple[str, pd.DataFrame]],
    validation_set: list[tuple[str, pd.DataFrame]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], BoundedNARXModel, dict[str, float], list[dict[str, Any]]]:
    combos = list(
        product(args.na_values, args.nb_values, range(args.nk_min, args.nk_max + 1), args.degree_values, args.alpha_values)
    )
    metric_key = arx.SELECTION_METRIC_LABELS[args.selection_metric]
    best_score = float("inf")
    best_params: dict[str, Any] | None = None
    best_model: BoundedNARXModel | None = None
    best_metrics: dict[str, float] | None = None
    results: list[dict[str, Any]] = []
    logger.info("Grid search: %d bounded NARX combinations", len(combos))

    with arx.ProgressBar(len(combos), "Grid search") as progress:
        for na, nb, nk, degree, alpha in combos:
            label = f"na={na} nb={nb} nk={nk} degree={degree} alpha={alpha:g}"
            model, n_train, status = fit_bounded_narx(
                train_set, na, nb, nk, degree, alpha, args.max_expanded_terms, args.prediction_margin,
                clip_predictions=args.clip_predictions,
                nonlinear_scope=args.nonlinear_scope,
            )
            if model is None:
                results.append({"na": na, "nb": nb, "nk": nk, "degree": degree, "alpha": alpha, "status": status})
                progress.update(message=f"{label} | skipped ({status})")
                continue
            y_true, y_pred = predictions_by_mode(validation_set, model, na, nb, nk, args.selection_mode)
            metrics = compute_metrics(y_true, y_pred, model.n_params)
            score = float(metrics[metric_key])
            row = {
                "na": na, "nb": nb, "nk": nk, "degree": degree, "alpha": alpha,
                "n_base_terms": len(model.linear_feature_names), "n_params": model.n_params,
                "n_train_samples": n_train, "selection_mode": args.selection_mode,
                "selection_metric": args.selection_metric, "selection_score": score,
                "status": "ok" if np.isfinite(score) else f"failed: non-finite {metric_key}",
                **metrics,
            }
            results.append(row)
            if np.isfinite(score) and score < best_score:
                best_score, best_params, best_model, best_metrics = score, {
                    "na": na, "nb": nb, "nk": nk, "degree": degree, "alpha": alpha
                }, model, metrics
                progress.update(message=f"{label} | best {metric_key}={score:.4f}")
            else:
                progress.update(message=f"{label} | {metric_key}={score:.4f}")

    if best_params is None or best_model is None or best_metrics is None:
        raise RuntimeError("Grid search produced no finite bounded NARX model.")
    return best_params, best_model, best_metrics, results


def save_outputs(
    args: argparse.Namespace,
    split: Any,
    params: dict[str, Any],
    model: BoundedNARXModel,
    selection_metrics: dict[str, float],
    train_metrics: dict[str, float],
    validation_osa_metrics: dict[str, float],
    validation_free_run_metrics: dict[str, float],
    test_osa_metrics: dict[str, float],
    test_free_run_metrics: dict[str, float],
    search_results: list[dict[str, Any]],
    train_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    coefficient_names = expanded_feature_names(
        model.linear_feature_names, model.degree, model.higher_power_start
    )
    pd.DataFrame({"term": coefficient_names, "coefficient": model.coefficients}).to_csv(
        args.output_dir / "narx2_coefficients.csv", index=False
    )
    pd.DataFrame(search_results).sort_values("selection_score", na_position="last").to_csv(
        args.output_dir / "narx2_grid_search.csv", index=False
    )
    split.manifest.to_csv(args.output_dir / "narx2_split_manifest.csv", index=False)
    train_predictions.to_csv(args.output_dir / "narx2_train_predictions.csv", index=False)
    validation_predictions.to_csv(args.output_dir / "narx2_validation_predictions.csv", index=False)
    test_predictions.to_csv(args.output_dir / "narx2_test_predictions.csv", index=False)
    payload = {
        "model": "bounded_additive_power_narx",
        "bounded_expansion": "intercept plus independent standardized powers; no interaction terms",
        "nonlinear_scope": model.nonlinear_scope,
        "target_col": args.target_col,
        "feature_set": args.feature_set,
        "best_hyperparameters": params,
        "n_base_terms": len(model.linear_feature_names),
        "n_params": model.n_params,
        "max_expanded_terms": args.max_expanded_terms,
        "prediction_bounds": model.prediction_bounds,
        "prediction_margin": args.prediction_margin,
        "prediction_clipping": model.clip_predictions,
        "linear_feature_names": model.linear_feature_names,
        "expanded_feature_names": coefficient_names,
        "selection_mode": args.selection_mode,
        "selection_metric": args.selection_metric,
        "validation_selection": selection_metrics,
        "train_one_step_ahead": train_metrics,
        "validation_one_step_ahead": validation_osa_metrics,
        "validation_free_run": validation_free_run_metrics,
        "test_one_step_ahead": test_osa_metrics,
        "test_free_run": test_free_run_metrics,
        "split_note": split.note,
    }
    with open(args.output_dir / "narx2_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run(args: argparse.Namespace) -> None:
    arx.TARGET_COL = args.target_col
    arx.FEATURE_SET = args.feature_set
    arx.SUMMARY_WINDOW = args.summary_window
    split = arx.load_dataset_split(
        args.data_folder, machine_type=args.machine_type, wfs=args.wfs, ctdw=args.ctdw
    )
    train_set = arx.split_sequences_to_datasets(split.train)
    validation_set = arx.split_sequences_to_datasets(split.validation)
    test_set = arx.split_sequences_to_datasets(split.test)

    params, model, selection_metrics, search_results = grid_search(train_set, validation_set, args)
    na, nb, nk = params["na"], params["nb"], params["nk"]
    train_y, train_pred = predict_one_step(train_set, model, na, nb, nk)
    val_y, val_osa = predict_one_step(validation_set, model, na, nb, nk)
    val_free_y, val_free = predict_free_run(validation_set, model, na, nb, nk)
    test_y, test_osa = predict_one_step(test_set, model, na, nb, nk)
    test_free_y, test_free = predict_free_run(test_set, model, na, nb, nk)
    train_metrics = compute_metrics(train_y, train_pred, model.n_params)
    validation_osa_metrics = compute_metrics(val_y, val_osa, model.n_params)
    validation_free_metrics = compute_metrics(val_free_y, val_free, model.n_params)
    test_osa_metrics = compute_metrics(test_y, test_osa, model.n_params)
    test_free_metrics = compute_metrics(test_free_y, test_free, model.n_params)
    save_outputs(
        args, split, params, model, selection_metrics, train_metrics,
        validation_osa_metrics, validation_free_metrics, test_osa_metrics, test_free_metrics,
        search_results, build_aligned_prediction_rows(train_set, model, na, nb, nk),
        build_aligned_prediction_rows(validation_set, model, na, nb, nk),
        build_aligned_prediction_rows(test_set, model, na, nb, nk),
    )
    print("NARX2 MODEL IDENTIFICATION - SUMMARY")
    print(f"Best model          : na={na} nb={nb} nk={nk} degree={model.degree} alpha={model.alpha:g}")
    print(f"Bounded parameters  : {model.n_params} ({len(model.linear_feature_names)} base ARX terms)")
    print(f"Nonlinear scope     : {model.nonlinear_scope}")
    print(f"Prediction clipping : {'enabled' if model.clip_predictions else 'disabled'}")
    print(f"Validation free-run : RMSE={validation_free_metrics['RMSE']:.6f}")
    print(f"Test free-run       : RMSE={test_free_metrics['RMSE']:.6f}")
    print(f"Outputs             : {args.output_dir}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
