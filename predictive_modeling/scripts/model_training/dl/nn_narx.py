"""Train neural NARX models for sequence-based process prediction."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictive_modeling.scripts.config import CONFIG as PROJECT_CONFIG
from predictive_modeling.scripts.preprocessing.split import DatasetSplit, SequenceData, split_data_folder


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "merged_clipped"
DEFAULT_MODEL_SAVE_PATH = PROJECT_ROOT / "output" / "nn_narx_tapering_to_weldpool"

TARGET_VARIABLE = "tapering_to_weldpool_dist"
DEFAULT_INPUT_COLUMNS = ["current", "voltage"]
DEFAULT_MACHINE_GROUPS = ("fr", "we")
DEFAULT_ROLLING_WINDOWS = (5, 25)
STATIC_CONTEXT_COLUMNS = ("wfs", "machine_is_fr", "machine_is_we")
RESERVED_LIVE_COLUMNS = {"wfs", "ctdw", "ctwd"}
SAFE_DIVISION_EPS = 1.0e-12
ARC_CURRENT_THRESHOLD = 1.0e-6
ARC_VOLTAGE_THRESHOLD = 1.0e-6


@dataclass(frozen=True)
class RolloutSequence:
    """Represent one sequence used during neural NARX rollout training."""
    file_name: str
    inputs: np.ndarray
    y: np.ndarray
    static_context: np.ndarray
    max_lag: int
    valid_starts: np.ndarray


class NARXMLP(nn.Module):
    """Feed-forward neural NARX model over lagged inputs and outputs."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...],
        dropout: float,
        activation: str = "relu",
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_size, hidden_size))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            if activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "silu":
                layers.append(nn.SiLU())
            else:
                layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous_size = hidden_size
        layers.append(nn.Linear(previous_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).reshape(-1)


def machine_group_from_file(file_name: str) -> str:
    match = re.match(r"([A-Za-z]+)_", file_name)
    return match.group(1).lower() if match else "unknown"


def sequence_machine_group(sequence: SequenceData) -> str:
    if hasattr(sequence, "machine_group"):
        return str(sequence.machine_group)
    return machine_group_from_file(sequence.file_name)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


def input_columns_from_arg(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_INPUT_COLUMNS)
    columns = [column.strip() for column in value.split(",") if column.strip()]
    reserved = [column for column in columns if column.lower() in RESERVED_LIVE_COLUMNS]
    if reserved:
        raise ValueError(
            "ctdw/ctwd and wfs are reserved context columns and must not be passed through --input-cols. "
            "wfs is handled automatically as static context, and ctdw/ctwd is not available live."
        )
    return columns


def rolling_windows_from_arg(value: str | None) -> list[int]:
    if not value:
        return list(DEFAULT_ROLLING_WINDOWS)
    windows = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not windows:
        raise ValueError("--rolling-windows must contain at least one integer window size.")
    if any(window < 2 for window in windows):
        raise ValueError("--rolling-windows values must be >= 2.")
    return windows


def hidden_sizes_from_arg(value: str) -> tuple[int, ...]:
    hidden_sizes = tuple(int(size.strip()) for size in str(value).split(",") if size.strip())
    if not hidden_sizes:
        raise ValueError("--hidden-sizes must contain at least one positive integer.")
    if any(size <= 0 for size in hidden_sizes):
        raise ValueError("--hidden-sizes values must be positive integers.")
    return hidden_sizes


def validate_training_args(args: argparse.Namespace) -> None:
    input_columns_from_arg(args.input_cols)
    rolling_windows_from_arg(args.rolling_windows)
    hidden_sizes_from_arg(str(args.hidden_sizes))

    if int(args.input_lags) < 0:
        raise ValueError("--input-lags must be >= 0.")
    if int(args.output_lags) < 0:
        raise ValueError("--output-lags must be >= 0.")
    if not 0.0 < float(args.train_fraction) < 1.0:
        raise ValueError("--train-fraction must be between 0 and 1.")
    if not 0.0 < float(args.validation_fraction) < 1.0:
        raise ValueError("--validation-fraction must be between 0 and 1.")
    if float(args.train_fraction) + float(args.validation_fraction) >= 1.0:
        raise ValueError("--train-fraction + --validation-fraction must leave a non-empty test split.")
    if not 0.0 <= float(args.dropout) < 1.0:
        raise ValueError("--dropout must be >= 0 and < 1.")
    if float(args.learning_rate) <= 0.0:
        raise ValueError("--learning-rate must be > 0.")
    if float(args.weight_decay) < 0.0:
        raise ValueError("--weight-decay must be >= 0.")
    if float(args.input_noise_std) < 0.0:
        raise ValueError("--input-noise-std must be >= 0.")
    if str(args.loss) == "huber" and float(args.huber_delta) <= 0.0:
        raise ValueError("--huber-delta must be > 0 when --loss huber is used.")
    if str(args.activation) not in {"relu", "gelu", "silu"}:
        raise ValueError("--activation must be one of: relu, gelu, silu.")
    if not 0.0 <= float(args.feedback_prob_final) <= 1.0:
        raise ValueError("--feedback-prob-final must be between 0 and 1.")
    if int(args.feedback_warmup_epochs) < 1:
        raise ValueError("--feedback-warmup-epochs must be >= 1.")
    if int(args.rollout_warmup_epochs) < 0:
        raise ValueError("--rollout-warmup-epochs must be >= 0.")
    if int(args.rollout_train_windows) < 0:
        raise ValueError("--rollout-train-windows must be >= 0.")
    if int(args.rollout_window_size) < 1:
        raise ValueError("--rollout-window-size must be >= 1.")
    if float(args.rollout_loss_weight) < 0.0:
        raise ValueError("--rollout-loss-weight must be >= 0.")
    if float(args.absolute_loss_weight) < 0.0:
        raise ValueError("--absolute-loss-weight must be >= 0.")
    if int(args.epochs) < 1:
        raise ValueError("--epochs must be >= 1.")
    if int(args.batch_size) < 1:
        raise ValueError("--batch-size must be >= 1.")
    if int(args.patience) < 1:
        raise ValueError("--patience must be >= 1.")
    if float(args.min_delta) < 0.0:
        raise ValueError("--min-delta must be >= 0.")
    if int(args.rollout_validation_every) < 1:
        raise ValueError("--rollout-validation-every must be >= 1.")
    if int(args.log_every) < 1:
        raise ValueError("--log-every must be >= 1.")
    if float(args.prediction_max_margin) < 0.0:
        raise ValueError("--prediction-max-margin must be >= 0.")


def machine_groups_from_arg(value: str) -> tuple[str, ...]:
    if value == "both":
        return DEFAULT_MACHINE_GROUPS
    return (value,)


def model_variant_from_args(args: argparse.Namespace) -> str:
    base = "measurement_only" if int(args.output_lags) == 0 else "narx"
    return f"{str(args.target_mode)}_{base}"


def sequence_wfs(sequence: SequenceData) -> float:
    return float(sequence.df["wfs"].iloc[0])


def machine_type_from_arg(value: str) -> str | None:
    return None if value == "both" else value


def build_nn_narx_split_config(
    args: argparse.Namespace,
    input_cols: list[str],
    rolling_windows: list[int],
) -> dict[str, Any]:
    max_model_lag = max(int(args.input_lags), int(args.output_lags), max(rolling_windows))
    split_config = dict(PROJECT_CONFIG["preprocessing"]["split"])
    split_config.update(
        {
            "target_col": str(args.target_col),
            "history_cols": list(input_cols),
            "input_cols": list(input_cols),
            "context_cols": ["wfs"],
            "frame_order_col": str(args.frame_order_col),
            "train_fraction": float(args.train_fraction),
            "validation_fraction": float(args.validation_fraction),
            "random_state": int(args.random_seed),
            "window_size": max_model_lag,
            "window_size_values": [max_model_lag],
            "rolling_summary_windows": list(rolling_windows),
            "output_lag_count": int(args.output_lags),
            "use_past_true_output": int(args.output_lags) > 0,
        }
    )
    return split_config


def build_static_context(sequence: SequenceData) -> tuple[np.ndarray, list[str]]:
    machine_group = sequence_machine_group(sequence)
    wfs_values = sequence.df["wfs"].to_numpy(dtype=float)
    if wfs_values.size == 0 or not np.isfinite(wfs_values[0]):
        raise ValueError(f"{sequence.file_name} does not contain a valid wfs value.")
    machine_is_fr = 1.0 if machine_group == "fr" else 0.0
    machine_is_we = 1.0 if machine_group == "we" else 0.0
    context = np.asarray([float(wfs_values[0]), machine_is_fr, machine_is_we], dtype=float)
    return context, list(STATIC_CONTEXT_COLUMNS)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denominator = np.asarray(denominator, dtype=float)
    safe_denominator = np.where(np.abs(denominator) > SAFE_DIVISION_EPS, denominator, np.nan)
    result = np.asarray(numerator, dtype=float) / safe_denominator
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def add_dynamic_feature(
    feature_values: dict[str, np.ndarray],
    dynamic_cols: list[str],
    name: str,
    values: np.ndarray | pd.Series,
) -> None:
    if name in dynamic_cols:
        return
    feature_values[name] = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    dynamic_cols.append(name)


def build_dynamic_feature_frame(
    df: pd.DataFrame,
    input_cols: list[str],
    rolling_windows: list[int],
    extended_features: bool,
) -> tuple[pd.DataFrame, list[str]]:
    frame = df.copy()
    dynamic_cols: list[str] = []
    feature_values: dict[str, np.ndarray] = {}

    for column in input_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        dynamic_cols.append(column)
        feature_values[column] = frame[column].to_numpy(dtype=float)

    if {"current", "voltage"}.issubset(frame.columns):
        current = frame["current"].to_numpy(dtype=float)
        voltage = frame["voltage"].to_numpy(dtype=float)
        add_dynamic_feature(feature_values, dynamic_cols, "power", current * voltage)
    else:
        raise ValueError("current and voltage columns are required for physics-aware features.")

    current = feature_values["current"]
    voltage = feature_values["voltage"]
    power = feature_values["power"]

    if "wfs" in frame.columns:
        wfs_values = pd.to_numeric(frame["wfs"], errors="coerce")
        add_dynamic_feature(
            feature_values,
            dynamic_cols,
            "specific_power",
            safe_divide(power, wfs_values.to_numpy(dtype=float)),
        )

    add_dynamic_feature(feature_values, dynamic_cols, "voltage_to_current_ratio", safe_divide(voltage, current))
    add_dynamic_feature(feature_values, dynamic_cols, "current_to_voltage_ratio", safe_divide(current, voltage))
    if extended_features:
        add_dynamic_feature(
            feature_values,
            dynamic_cols,
            "arc_active_est",
            ((current > ARC_CURRENT_THRESHOLD) | (voltage > ARC_VOLTAGE_THRESHOLD)).astype(float),
        )

    for column in ("current", "voltage", "power", "voltage_to_current_ratio", "current_to_voltage_ratio"):
        values = pd.Series(feature_values[column], dtype="float64")
        diff_1 = values.diff().fillna(0.0).to_numpy(dtype=float)
        add_dynamic_feature(feature_values, dynamic_cols, f"{column}_diff_1", diff_1)
        if extended_features:
            add_dynamic_feature(
                feature_values,
                dynamic_cols,
                f"{column}_abs_diff_1",
                np.abs(diff_1),
            )
        for window in rolling_windows:
            mean_name = f"{column}_roll_mean_{window}"
            std_name = f"{column}_roll_std_{window}"
            slope_name = f"{column}_roll_slope_{window}"
            rolling = values.rolling(window=window, min_periods=window)
            add_dynamic_feature(feature_values, dynamic_cols, mean_name, rolling.mean().fillna(0.0))
            add_dynamic_feature(feature_values, dynamic_cols, std_name, rolling.std(ddof=0).fillna(0.0))
            add_dynamic_feature(
                feature_values,
                dynamic_cols,
                slope_name,
                ((values - values.shift(window - 1)) / float(window - 1)).fillna(0.0),
            )
            if extended_features:
                add_dynamic_feature(
                    feature_values,
                    dynamic_cols,
                    f"{column}_roll_min_{window}",
                    rolling.min().fillna(0.0),
                )
                add_dynamic_feature(
                    feature_values,
                    dynamic_cols,
                    f"{column}_roll_max_{window}",
                    rolling.max().fillna(0.0),
                )
                add_dynamic_feature(
                    feature_values,
                    dynamic_cols,
                    f"{column}_roll_delta_{window}",
                    (values - values.shift(window - 1)).fillna(0.0),
                )

    engineered = pd.DataFrame(
        {name: feature_values[name] for name in dynamic_cols if name not in input_cols},
        index=frame.index,
    )
    frame = pd.concat([frame, engineered], axis=1)
    frame[dynamic_cols] = frame[dynamic_cols].fillna(0.0)

    return frame, dynamic_cols


def lag_feature_names(dynamic_cols: list[str], input_lags: int, output_lags: int, target_col: str) -> list[str]:
    names: list[str] = []
    for lag in range(input_lags + 1):
        for column in dynamic_cols:
            names.append(f"{column}_lag_{lag}")
    for lag in range(1, output_lags + 1):
        names.append(f"{target_col}_lag_{lag}")
    return names


def make_feature_vector(
    inputs: np.ndarray,
    outputs: np.ndarray,
    row_index: int,
    input_lags: int,
    output_lags: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    for lag in range(input_lags + 1):
        features.append(inputs[row_index - lag])
    output_features = [outputs[row_index - lag] for lag in range(1, output_lags + 1)]
    return np.concatenate([*features, np.asarray(output_features, dtype=float)])


def target_value(y: np.ndarray, row_index: int, target_mode: str) -> float:
    if target_mode == "delta":
        return float(y[row_index] - y[row_index - 1])
    if target_mode == "absolute":
        return float(y[row_index])
    raise ValueError("target_mode must be 'delta' or 'absolute'.")


def finite_target_window(y: np.ndarray, row_index: int, output_lags: int, target_mode: str) -> bool:
    required_indices = [row_index]
    if target_mode == "delta":
        required_indices.append(row_index - 1)
    required_indices.extend(row_index - lag for lag in range(1, output_lags + 1))
    return bool(np.all(np.isfinite(y[required_indices])))


def reconstruct_prediction(
    model_output: float,
    previous_feedback: float,
    target_mode: str,
    prediction_min: float,
    prediction_max: float,
) -> float:
    if target_mode == "delta":
        prediction = previous_feedback + model_output
    elif target_mode == "absolute":
        prediction = model_output
    else:
        raise ValueError("target_mode must be 'delta' or 'absolute'.")
    return float(np.clip(prediction, prediction_min, prediction_max))


def build_training_matrix(
    sequences: list[SequenceData],
    input_cols: list[str],
    target_col: str,
    input_lags: int,
    output_lags: int,
    target_mode: str,
    rolling_windows: list[int],
    extended_features: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str], list[str], int]:
    max_lag = max(input_lags, output_lags, max(rolling_windows), 1 if target_mode == "delta" else 0)
    feature_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    metadata_rows: list[dict[str, Any]] = []
    dynamic_feature_names: list[str] | None = None
    context_names: list[str] | None = None
    dynamic_base_count: int | None = None
    for seq in sequences:
        if len(seq.df) <= max_lag:
            continue
        feature_frame, dynamic_cols = build_dynamic_feature_frame(
            seq.df,
            input_cols,
            rolling_windows,
            extended_features,
        )
        static_context, context_columns = build_static_context(seq)
        inputs = feature_frame[dynamic_cols].to_numpy(dtype=float)
        y = seq.df[target_col].to_numpy(dtype=float)
        if dynamic_feature_names is None:
            dynamic_feature_names = lag_feature_names(dynamic_cols, input_lags, output_lags, target_col)
        if context_names is None:
            context_names = context_columns
        if dynamic_base_count is None:
            dynamic_base_count = len(dynamic_cols)
        for row_index in range(max_lag, len(seq.df)):
            if not finite_target_window(y, row_index, output_lags, target_mode):
                continue
            dynamic_features = make_feature_vector(inputs, y, row_index, input_lags, output_lags)
            feature_row = np.concatenate([dynamic_features, static_context]).astype(float)
            target_row = target_value(y, row_index, target_mode)
            if not np.all(np.isfinite(feature_row)) or not math.isfinite(target_row):
                continue
            feature_rows.append(feature_row)
            target_rows.append(target_row)
            metadata_rows.append(
                {
                    "file": seq.file_name,
                    "machine_group": sequence_machine_group(seq),
                    "split_role": seq.split_role,
                    "wfs": float(static_context[0]),
                    "row_pos": row_index,
                    "sample_index": int(seq.df["sample_index"].iloc[row_index]),
                }
            )
    if not feature_rows:
        raise ValueError("No lagged training rows were produced. Reduce lag counts or provide longer sequences.")
    if dynamic_feature_names is None or context_names is None or dynamic_base_count is None:
        raise ValueError("Could not determine feature names for the training matrix.")
    return (
        np.vstack(feature_rows),
        np.asarray(target_rows, dtype=float),
        pd.DataFrame(metadata_rows),
        dynamic_feature_names,
        context_names,
        dynamic_base_count,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    valid_count = int(finite_mask.sum())
    if valid_count == 0:
        return {
            "MAE": math.nan,
            "RMSE": math.nan,
            "NRMSE": math.nan,
            "R2": math.nan,
            "fit_pct": math.nan,
            "bias": math.nan,
            "valid_label_count": 0,
        }
    truth = y_true[finite_mask]
    pred = y_pred[finite_mask]
    residual = truth - pred
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    target_range = float(np.max(truth) - np.min(truth))
    nrmse = float(rmse / target_range) if target_range > 0.0 else (0.0 if rmse == 0.0 else math.nan)
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(truth - np.mean(truth))))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else math.nan
    denom = float(np.linalg.norm(truth - np.mean(truth)))
    fit_pct = float(100.0 * (1.0 - np.linalg.norm(residual) / denom)) if denom > 0.0 else math.nan
    return {
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "R2": r2,
        "fit_pct": fit_pct,
        "bias": float(np.mean(pred - truth)),
        "valid_label_count": valid_count,
    }


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def format_metric(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(float(value)) else "n/a"


def prediction_bounds(sequences: list[SequenceData], target_col: str, args: argparse.Namespace) -> tuple[float, float]:
    y_values = np.concatenate([seq.df[target_col].to_numpy(dtype=float) for seq in sequences])
    y_values = y_values[np.isfinite(y_values)]
    if y_values.size == 0:
        raise ValueError(f"No finite values found for target column {target_col!r} in the training split.")
    prediction_min = float(args.prediction_min)
    prediction_max = float(np.max(y_values) + float(args.prediction_max_margin))
    if prediction_max <= prediction_min:
        prediction_max = prediction_min + 1.0
    return prediction_min, prediction_max


def inverse_scaled_targets(
    scaled_values: np.ndarray,
    target_scaler: StandardScaler,
) -> np.ndarray:
    return target_scaler.inverse_transform(np.asarray(scaled_values, dtype=float).reshape(-1, 1)).reshape(-1)


def predict_single_model_output(
    model: NARXMLP,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    feature: np.ndarray,
    device: torch.device,
) -> float:
    scale = np.where(scaler_x.scale_ > 0.0, scaler_x.scale_, 1.0)
    x_scaled = ((feature - scaler_x.mean_) / scale).astype(np.float32)
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_scaled).to(device).reshape(1, -1)
        pred_scaled = model(x_tensor).detach().cpu().numpy()
    return float(inverse_scaled_targets(pred_scaled, scaler_y)[0])


def one_step_absolute_predictions(
    model: NARXMLP,
    x_values: np.ndarray,
    metadata: pd.DataFrame,
    sequences: list[SequenceData],
    target_col: str,
    target_mode: str,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    device: torch.device,
    prediction_min: float,
    prediction_max: float,
) -> dict[str, np.ndarray]:
    model_outputs = predict_scaled_batch(model, scaler_x, scaler_y, x_values, device)
    sequence_truth = {seq.file_name: seq.df[target_col].to_numpy(dtype=float) for seq in sequences}
    predictions = {
        seq.file_name: np.full(len(seq.df), np.nan, dtype=float)
        for seq in sequences
    }
    for pred_output, row in zip(model_outputs, metadata.itertuples(index=False)):
        file_name = str(row.file)
        row_pos = int(row.row_pos)
        y = sequence_truth[file_name]
        predictions[file_name][row_pos] = reconstruct_prediction(
            float(pred_output),
            float(y[row_pos - 1]),
            target_mode,
            prediction_min,
            prediction_max,
        )
    return predictions


def scheduled_feedback_features(
    x_base: np.ndarray,
    metadata: pd.DataFrame,
    feedback_predictions: dict[str, np.ndarray],
    dynamic_feature_count: int,
    input_lags: int,
    output_lags: int,
    feedback_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if feedback_probability <= 0.0 or output_lags <= 0:
        return x_base

    x_augmented = x_base.copy()
    output_start = dynamic_feature_count * (input_lags + 1)
    row_files = metadata["file"].astype(str).to_numpy()
    row_positions = metadata["row_pos"].to_numpy(dtype=int)
    for lag in range(1, output_lags + 1):
        column_index = output_start + lag - 1
        replace_mask = rng.random(len(x_augmented)) < feedback_probability
        if not np.any(replace_mask):
            continue
        for row_index in np.flatnonzero(replace_mask):
            file_name = row_files[row_index]
            lagged_pos = row_positions[row_index] - lag
            if lagged_pos < 0:
                continue
            replacement = feedback_predictions[file_name][lagged_pos]
            if np.isfinite(replacement):
                x_augmented[row_index, column_index] = replacement
    return x_augmented


def make_loader(
    x_values: np.ndarray,
    y_values: np.ndarray,
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    args: argparse.Namespace,
    seed: int,
) -> DataLoader:
    x_scaled = input_scaler.transform(x_values).astype(np.float32)
    input_noise_std = float(getattr(args, "input_noise_std", 0.0))
    if input_noise_std > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, input_noise_std, size=x_scaled.shape).astype(np.float32)
        x_scaled = x_scaled + noise
    y_scaled = target_scaler.transform(y_values.reshape(-1, 1)).reshape(-1).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(x_scaled), torch.from_numpy(y_scaled))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        generator=generator,
    )


def feedback_probability_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    if float(args.feedback_prob_final) <= 0.0:
        return 0.0
    warmup_epochs = max(1, int(args.feedback_warmup_epochs))
    progress = min(1.0, max(0.0, (epoch - 1) / float(warmup_epochs - 1 if warmup_epochs > 1 else 1)))
    return float(np.clip(float(args.feedback_prob_final) * progress, 0.0, 1.0))


def rollout_weight_for_epoch(args: argparse.Namespace, epoch: int) -> float:
    base_weight = float(args.rollout_loss_weight)
    if base_weight <= 0.0 or int(args.rollout_train_windows) <= 0:
        return 0.0
    warmup_epochs = int(args.rollout_warmup_epochs)
    if warmup_epochs <= 0:
        return base_weight
    progress = min(1.0, max(0.0, epoch / float(warmup_epochs)))
    return base_weight * progress


def build_rollout_sequences(
    sequences: list[SequenceData],
    input_cols: list[str],
    target_col: str,
    input_lags: int,
    output_lags: int,
    rolling_windows: list[int],
    extended_features: bool,
    target_mode: str,
) -> list[RolloutSequence]:
    max_lag = max(input_lags, output_lags, max(rolling_windows), 1 if target_mode == "delta" else 0)
    required_history = max(output_lags, 1 if target_mode == "delta" else 0)
    rollout_sequences: list[RolloutSequence] = []
    for seq in sequences:
        if len(seq.df) <= max_lag + 1:
            continue
        feature_frame, dynamic_cols = build_dynamic_feature_frame(
            seq.df,
            input_cols,
            rolling_windows,
            extended_features,
        )
        static_context, _ = build_static_context(seq)
        y = seq.df[target_col].to_numpy(dtype=np.float32)
        valid_starts = np.asarray(
            [
                start
                for start in range(max_lag, len(y))
                if np.all(np.isfinite(y[start - required_history : start + 1]))
            ],
            dtype=int,
        )
        if valid_starts.size == 0:
            continue
        rollout_sequences.append(
            RolloutSequence(
                file_name=seq.file_name,
                inputs=feature_frame[dynamic_cols].to_numpy(dtype=np.float32),
                y=y,
                static_context=static_context.astype(np.float32),
                max_lag=max_lag,
                valid_starts=valid_starts,
            )
        )
    return rollout_sequences


def validate_finite_matrix(name: str, x_values: np.ndarray, y_values: np.ndarray) -> None:
    if x_values.size == 0 or y_values.size == 0:
        raise ValueError(f"{name} matrix is empty after filtering non-finite target rows.")
    if not np.all(np.isfinite(x_values)):
        bad_count = int((~np.isfinite(x_values)).sum())
        raise ValueError(f"{name} feature matrix contains {bad_count} non-finite values.")
    if not np.all(np.isfinite(y_values)):
        bad_count = int((~np.isfinite(y_values)).sum())
        raise ValueError(f"{name} target vector contains {bad_count} non-finite values.")


def sequence_feature_tensor(
    inputs: torch.Tensor,
    feedback: torch.Tensor,
    static_context: torch.Tensor,
    row_index: int,
    input_lags: int,
    output_lags: int,
) -> torch.Tensor:
    feature_parts = [inputs[row_index - lag] for lag in range(input_lags + 1)]
    if output_lags > 0:
        feature_parts.append(torch.stack([feedback[row_index - lag] for lag in range(1, output_lags + 1)]))
    feature_parts.append(static_context)
    return torch.cat(feature_parts)


def scaled_output_to_absolute_prediction(
    pred_scaled: torch.Tensor,
    previous_feedback: torch.Tensor,
    target_mode: str,
    target_mean: torch.Tensor,
    target_scale: torch.Tensor,
    prediction_min: float,
    prediction_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    model_output = pred_scaled * target_scale + target_mean
    if target_mode == "delta":
        prediction = previous_feedback + model_output
    elif target_mode == "absolute":
        prediction = model_output
    else:
        raise ValueError("target_mode must be 'delta' or 'absolute'.")
    return model_output, torch.clamp(prediction, prediction_min, prediction_max)


def rollout_training_loss(
    model: NARXMLP,
    rollout_sequences: list[RolloutSequence],
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    abs_target_mean: float,
    abs_target_scale: float,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
    prediction_min: float,
    prediction_max: float,
) -> tuple[torch.Tensor | None, float]:
    if not rollout_sequences or int(args.rollout_train_windows) <= 0 or float(args.rollout_loss_weight) <= 0.0:
        return None, math.nan

    input_mean = torch.as_tensor(input_scaler.mean_, dtype=torch.float32, device=device)
    input_scale = torch.as_tensor(
        np.where(input_scaler.scale_ > 0.0, input_scaler.scale_, 1.0),
        dtype=torch.float32,
        device=device,
    )
    target_mean = torch.tensor(float(target_scaler.mean_[0]), dtype=torch.float32, device=device)
    target_scale = torch.tensor(float(target_scaler.scale_[0]), dtype=torch.float32, device=device)
    abs_mean = torch.tensor(abs_target_mean, dtype=torch.float32, device=device)
    abs_scale = torch.tensor(abs_target_scale if abs_target_scale > 0.0 else 1.0, dtype=torch.float32, device=device)
    horizon = int(args.rollout_window_size)
    input_lags = int(args.input_lags)
    output_lags = int(args.output_lags)
    target_mode = str(args.target_mode)
    window_losses: list[torch.Tensor] = []
    absolute_loss_weight = float(args.absolute_loss_weight)

    eligible_sequences = [
        seq for seq in rollout_sequences if len(seq.y) > seq.max_lag + horizon and seq.valid_starts.size > 0
    ]
    if not eligible_sequences:
        return None, math.nan

    for _ in range(int(args.rollout_train_windows)):
        seq = eligible_sequences[int(rng.integers(0, len(eligible_sequences)))]
        max_start = len(seq.y) - horizon
        possible_starts = seq.valid_starts[seq.valid_starts < max_start]
        possible_starts = np.asarray(
            [
                start
                for start in possible_starts
                if np.all(np.isfinite(seq.y[start - 1 : start + horizon]))
            ],
            dtype=int,
        )
        if possible_starts.size == 0:
            continue
        start = int(possible_starts[int(rng.integers(0, len(possible_starts)))])
        inputs = torch.as_tensor(seq.inputs, dtype=torch.float32, device=device)
        y_true = torch.as_tensor(seq.y, dtype=torch.float32, device=device)
        static_context = torch.as_tensor(seq.static_context, dtype=torch.float32, device=device)
        feedback = y_true.clone()
        step_losses: list[torch.Tensor] = []

        for row_index in range(start, start + horizon):
            feature = sequence_feature_tensor(
                inputs,
                feedback,
                static_context,
                row_index,
                input_lags,
                output_lags,
            )
            pred_scaled = model(((feature - input_mean) / input_scale).reshape(1, -1)).reshape(())
            _, prediction = scaled_output_to_absolute_prediction(
                pred_scaled,
                feedback[row_index - 1],
                target_mode,
                target_mean,
                target_scale,
                prediction_min,
                prediction_max,
            )
            if target_mode == "delta":
                target_output = y_true[row_index] - y_true[row_index - 1]
            else:
                target_output = y_true[row_index]
            target_scaled = (target_output - target_mean) / target_scale
            if str(args.loss) == "mse":
                loss = torch.nn.functional.mse_loss(pred_scaled, target_scaled)
            else:
                loss = torch.nn.functional.smooth_l1_loss(
                    pred_scaled,
                    target_scaled,
                    beta=float(args.huber_delta),
                )
            if absolute_loss_weight > 0.0:
                pred_abs_scaled = (prediction - abs_mean) / abs_scale
                true_abs_scaled = (y_true[row_index] - abs_mean) / abs_scale
                if str(args.loss) == "mse":
                    absolute_loss = torch.nn.functional.mse_loss(pred_abs_scaled, true_abs_scaled)
                else:
                    absolute_loss = torch.nn.functional.smooth_l1_loss(
                        pred_abs_scaled,
                        true_abs_scaled,
                        beta=float(args.huber_delta),
                    )
                loss = loss + absolute_loss_weight * absolute_loss
            step_losses.append(loss)
            feedback = feedback.clone()
            feedback[row_index] = prediction

        if step_losses:
            window_losses.append(torch.stack(step_losses).mean())

    if not window_losses:
        return None, math.nan
    loss_tensor = torch.stack(window_losses).mean()
    return loss_tensor, float(loss_tensor.detach().cpu())


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_metadata: pd.DataFrame,
    train_sequences: list[SequenceData],
    x_val: np.ndarray,
    y_val: np.ndarray,
    validation_sequences: list[SequenceData],
    input_cols: list[str],
    dynamic_feature_count: int,
    rolling_windows: list[int],
    target_col: str,
    args: argparse.Namespace,
    device: torch.device,
    prediction_min: float,
    prediction_max: float,
) -> tuple[NARXMLP, StandardScaler, StandardScaler, pd.DataFrame, dict[str, Any]]:
    input_scaler = StandardScaler()
    target_scaler = StandardScaler()
    validate_finite_matrix("Training", x_train, y_train)
    validate_finite_matrix("Validation", x_val, y_val)
    input_scaler.fit(x_train)
    target_scaler.fit(y_train.reshape(-1, 1))
    train_abs_targets = np.concatenate([seq.df[target_col].to_numpy(dtype=float) for seq in train_sequences])
    train_abs_targets = train_abs_targets[np.isfinite(train_abs_targets)]
    if train_abs_targets.size == 0:
        raise ValueError(f"No finite values found for target column {target_col!r} in the training split.")
    abs_target_mean = float(np.mean(train_abs_targets))
    abs_target_scale = float(np.std(train_abs_targets))
    if abs_target_scale <= 0.0:
        abs_target_scale = 1.0
    x_val_scaled = input_scaler.transform(x_val).astype(np.float32)
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).reshape(-1).astype(np.float32)

    hidden_sizes = hidden_sizes_from_arg(str(args.hidden_sizes))
    model = NARXMLP(
        x_train.shape[1],
        hidden_sizes,
        float(args.dropout),
        str(args.activation),
        bool(args.layer_norm),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    if str(args.loss) == "huber":
        criterion = nn.HuberLoss(delta=float(args.huber_delta))
    else:
        criterion = nn.MSELoss()

    x_val_tensor = torch.from_numpy(x_val_scaled).to(device)
    y_val_tensor = torch.from_numpy(y_val_scaled).to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_selection_rmse = math.inf
    best_teacher_forced_loss = math.inf
    best_epoch: int | None = None
    early_stop_reason = "max_epochs"
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(args.random_seed))
    rollout_every = max(1, int(args.rollout_validation_every))
    rollout_sequences = build_rollout_sequences(
        train_sequences,
        input_cols,
        target_col,
        int(args.input_lags),
        int(args.output_lags),
        rolling_windows,
        bool(args.extended_features),
        str(args.target_mode),
    )

    for epoch in range(1, int(args.epochs) + 1):
        feedback_probability = feedback_probability_for_epoch(args, epoch)
        rollout_loss_weight = rollout_weight_for_epoch(args, epoch)
        if feedback_probability > 0.0:
            feedback_predictions = one_step_absolute_predictions(
                model,
                x_train,
                train_metadata,
                train_sequences,
                target_col,
                str(args.target_mode),
                input_scaler,
                target_scaler,
                device,
                prediction_min,
                prediction_max,
            )
            x_epoch = scheduled_feedback_features(
                x_train,
                train_metadata,
                feedback_predictions,
                dynamic_feature_count,
                int(args.input_lags),
                int(args.output_lags),
                feedback_probability,
                rng,
            )
        else:
            x_epoch = x_train
        loader = make_loader(
            x_epoch,
            y_train,
            input_scaler,
            target_scaler,
            args,
            int(args.random_seed) + epoch,
        )
        model.train()
        batch_losses: list[float] = []
        rollout_losses: list[float] = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            teacher_loss = criterion(model(x_batch), y_batch)
            loss = teacher_loss
            if rollout_loss_weight > 0.0:
                rollout_loss, rollout_loss_value = rollout_training_loss(
                    model,
                    rollout_sequences,
                    input_scaler,
                    target_scaler,
                    abs_target_mean,
                    abs_target_scale,
                    args,
                    device,
                    rng,
                    prediction_min,
                    prediction_max,
                )
                if rollout_loss is not None:
                    loss = loss + rollout_loss_weight * rollout_loss
                    rollout_losses.append(rollout_loss_value)
            loss.backward()
            if float(args.clip_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad_norm))
            optimizer.step()
            batch_losses.append(float(teacher_loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_tensor), y_val_tensor).detach().cpu())
        train_loss = float(np.mean(batch_losses)) if batch_losses else math.nan
        train_rollout_loss = float(np.mean(rollout_losses)) if rollout_losses else math.nan
        validation_free_run_rmse = math.nan
        if epoch == 1 or epoch % rollout_every == 0 or epoch == int(args.epochs):
            _, validation_metrics = evaluate_sequences(
                model,
                validation_sequences,
                input_cols,
                target_col,
                int(args.input_lags),
                int(args.output_lags),
                rolling_windows,
                bool(args.extended_features),
                input_scaler,
                target_scaler,
                device,
                str(args.target_mode),
                prediction_min,
                prediction_max,
            )
            validation_free_run_rmse = float(validation_metrics["free_run"]["RMSE"])
        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "generalization_gap_loss": val_loss - train_loss,
                "train_rollout_loss": train_rollout_loss,
                "validation_free_run_RMSE": validation_free_run_rmse,
                "feedback_probability": feedback_probability,
                "rollout_loss_weight": rollout_loss_weight,
            }
        )
        if epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(args.epochs):
            print(
                f"Epoch {epoch:04d}: train_loss={format_metric(train_loss)}, "
                f"rollout_loss={format_metric(train_rollout_loss)}, "
                f"val_loss={format_metric(val_loss)}, val_free_RMSE={format_metric(validation_free_run_rmse)}, "
                f"feedback_prob={feedback_probability:.3f}, rollout_weight={rollout_loss_weight:.3f}",
                flush=True,
            )
        improved = np.isfinite(validation_free_run_rmse) and (
            validation_free_run_rmse < best_selection_rmse - float(args.min_delta)
        )
        if improved:
            best_selection_rmse = validation_free_run_rmse
            best_teacher_forced_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        elif np.isfinite(validation_free_run_rmse):
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(args.patience):
                early_stop_reason = "patience_exhausted"
                print(
                    f"Early stopping at epoch {epoch}; "
                    f"best validation free-run RMSE={best_selection_rmse:.6f}",
                    flush=True,
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    summary = {
        "best_validation_free_run_RMSE": best_selection_rmse,
        "best_validation_loss": best_teacher_forced_loss,
        "best_epoch": best_epoch,
        "early_stop_reason": early_stop_reason,
        "epochs_trained": len(history_rows),
        "hidden_sizes": hidden_sizes,
        "input_size": int(x_train.shape[1]),
        "dropout": float(args.dropout),
        "activation": str(args.activation),
        "layer_norm": bool(args.layer_norm),
        "weight_decay": float(args.weight_decay),
        "input_noise_std": float(args.input_noise_std),
        "loss": str(args.loss),
        "huber_delta": float(args.huber_delta),
        "target_mode": str(args.target_mode),
        "feedback_prob_final": float(args.feedback_prob_final),
        "feedback_warmup_epochs": int(args.feedback_warmup_epochs),
        "rollout_loss_weight": float(args.rollout_loss_weight),
        "rollout_warmup_epochs": int(args.rollout_warmup_epochs),
        "rollout_train_windows": int(args.rollout_train_windows),
        "rollout_window_size": int(args.rollout_window_size),
        "absolute_loss_weight": float(args.absolute_loss_weight),
        "absolute_target_mean": abs_target_mean,
        "absolute_target_scale": abs_target_scale,
        "prediction_min": prediction_min,
        "prediction_max": prediction_max,
    }
    return model, input_scaler, target_scaler, pd.DataFrame(history_rows), summary


def predict_scaled_batch(
    model: NARXMLP,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    if features.size == 0:
        return np.array([], dtype=float)
    model.eval()
    x_scaled = scaler_x.transform(features).astype(np.float32)
    predictions: list[np.ndarray] = []
    batch_size = 8192
    with torch.no_grad():
        for start in range(0, len(x_scaled), batch_size):
            x = torch.from_numpy(x_scaled[start : start + batch_size]).to(device)
            pred_scaled = model(x).detach().cpu().numpy().reshape(-1, 1)
            predictions.append(scaler_y.inverse_transform(pred_scaled).reshape(-1))
    return np.concatenate(predictions) if predictions else np.array([], dtype=float)


def predict_sequence(
    model: NARXMLP,
    seq: SequenceData,
    input_cols: list[str],
    target_col: str,
    input_lags: int,
    output_lags: int,
    rolling_windows: list[int],
    extended_features: bool,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    device: torch.device,
    target_mode: str,
    prediction_min: float,
    prediction_max: float,
) -> pd.DataFrame:
    max_lag = max(input_lags, output_lags, max(rolling_windows), 1 if target_mode == "delta" else 0)
    df = seq.df.reset_index(drop=True)
    feature_frame, dynamic_cols = build_dynamic_feature_frame(
        df,
        input_cols,
        rolling_windows,
        extended_features,
    )
    static_context, _ = build_static_context(seq)
    inputs = feature_frame[dynamic_cols].to_numpy(dtype=float)
    y_true = df[target_col].to_numpy(dtype=float)
    teacher_predictions = np.full(len(df), np.nan, dtype=float)
    free_predictions = np.full(len(df), np.nan, dtype=float)

    if len(df) > max_lag:
        teacher_features = np.vstack(
            [
                make_feature_vector(inputs, y_true, row_index, input_lags, output_lags)
                for row_index in range(max_lag, len(df))
            ]
        )
        teacher_outputs = predict_scaled_batch(
            model,
            scaler_x,
            scaler_y,
            np.hstack([teacher_features, np.repeat(static_context.reshape(1, -1), len(teacher_features), axis=0)]),
            device,
        )
        for offset, row_index in enumerate(range(max_lag, len(df))):
            teacher_predictions[row_index] = reconstruct_prediction(
                float(teacher_outputs[offset]),
                float(y_true[row_index - 1]),
                target_mode,
                prediction_min,
                prediction_max,
            )

        feedback = y_true.copy()
        for row_index in range(max_lag, len(df)):
            dynamic_feature = make_feature_vector(inputs, feedback, row_index, input_lags, output_lags)
            feature = np.concatenate([dynamic_feature, static_context]).astype(float).reshape(1, -1)
            model_output = float(
                predict_single_model_output(
                    model,
                    scaler_x,
                    scaler_y,
                    feature.reshape(-1),
                    device,
                )
            )
            prediction = reconstruct_prediction(
                model_output,
                float(feedback[row_index - 1]),
                target_mode,
                prediction_min,
                prediction_max,
            )
            free_predictions[row_index] = prediction
            feedback[row_index] = prediction

    output = pd.DataFrame(
        {
            "file": seq.file_name,
            "machine_group": sequence_machine_group(seq),
            "split_role": seq.split_role,
            "wfs": float(static_context[0]),
            "row_pos": np.arange(len(df), dtype=int),
            "sample_index": df["sample_index"].to_numpy(dtype=int),
            "is_scored": np.arange(len(df), dtype=int) >= max_lag,
            "y_true": y_true,
            "y_pred_one_step_ahead": teacher_predictions,
            "y_pred_free_run": free_predictions,
        }
    )
    for column in ("meas_time", "frame_number"):
        if column in df.columns:
            output[column] = df[column].to_numpy()
    for column in input_cols:
        output[column] = df[column].to_numpy(dtype=float)
    if "power" in feature_frame.columns:
        output["power"] = feature_frame["power"].to_numpy(dtype=float)
    if "specific_power" in feature_frame.columns:
        output["specific_power"] = feature_frame["specific_power"].to_numpy(dtype=float)
    output["one_step_ahead_error"] = output["y_pred_one_step_ahead"] - output["y_true"]
    output["free_run_error"] = output["y_pred_free_run"] - output["y_true"]
    return output


def evaluate_sequences(
    model: NARXMLP,
    sequences: list[SequenceData],
    input_cols: list[str],
    target_col: str,
    input_lags: int,
    output_lags: int,
    rolling_windows: list[int],
    extended_features: bool,
    scaler_x: StandardScaler,
    scaler_y: StandardScaler,
    device: torch.device,
    target_mode: str,
    prediction_min: float,
    prediction_max: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    predictions = pd.concat(
        [
            predict_sequence(
                model,
                seq,
                input_cols,
                target_col,
                input_lags,
                output_lags,
                rolling_windows,
                extended_features,
                scaler_x,
                scaler_y,
                device,
                target_mode,
                prediction_min,
                prediction_max,
            )
            for seq in sequences
        ],
        ignore_index=True,
    )
    scored = predictions.loc[predictions["is_scored"]].copy()
    metrics = {
        "one_step_ahead": compute_metrics(
            scored["y_true"].to_numpy(dtype=float),
            scored["y_pred_one_step_ahead"].to_numpy(dtype=float),
        ),
        "free_run": compute_metrics(
            scored["y_true"].to_numpy(dtype=float),
            scored["y_pred_free_run"].to_numpy(dtype=float),
        ),
    }
    return predictions, metrics


def metrics_by_file(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scored = predictions.loc[predictions["is_scored"]].copy()
    for mode, column in (
        ("one_step_ahead", "y_pred_one_step_ahead"),
        ("free_run", "y_pred_free_run"),
    ):
        for file_name, group in scored.groupby("file", sort=False):
            row = {
                "mode": mode,
                "file": file_name,
                "machine_group": str(group["machine_group"].iloc[0]),
            }
            row.update(compute_metrics(group["y_true"].to_numpy(dtype=float), group[column].to_numpy(dtype=float)))
            rows.append(row)
    return pd.DataFrame(rows)


def metrics_by_machine(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scored = predictions.loc[predictions["is_scored"]].copy()
    for mode, column in (
        ("one_step_ahead", "y_pred_one_step_ahead"),
        ("free_run", "y_pred_free_run"),
    ):
        for machine_group, group in scored.groupby("machine_group", sort=False):
            row = {
                "mode": mode,
                "machine_group": machine_group,
            }
            row.update(compute_metrics(group["y_true"].to_numpy(dtype=float), group[column].to_numpy(dtype=float)))
            rows.append(row)
    return pd.DataFrame(rows)


def metrics_by_dimensions(predictions: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scored = predictions.loc[predictions["is_scored"]].copy()
    for mode, column in (
        ("one_step_ahead", "y_pred_one_step_ahead"),
        ("free_run", "y_pred_free_run"),
    ):
        for keys, group in scored.groupby(dimensions, sort=False, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"mode": mode}
            row.update({dimension: key for dimension, key in zip(dimensions, keys)})
            y_true = group["y_true"].to_numpy(dtype=float)
            y_pred = group[column].to_numpy(dtype=float)
            row.update(compute_metrics(y_true, y_pred))
            row.update(
                {
                    "y_true_mean": float(np.nanmean(y_true)),
                    "y_pred_mean": float(np.nanmean(y_pred)),
                    "n_rows": int(len(group)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def split_condition_coverage(split: DatasetSplit) -> pd.DataFrame:
    train_conditions = {
        (sequence_machine_group(seq), sequence_wfs(seq))
        for seq in split.train
    }
    rows: list[dict[str, Any]] = []
    for role, role_sequences in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        for seq in role_sequences:
            condition = (sequence_machine_group(seq), sequence_wfs(seq))
            rows.append(
                {
                    "file": seq.file_name,
                    "split_role": role,
                    "machine_group": condition[0],
                    "wfs": condition[1],
                    "condition_seen_in_train": condition in train_conditions,
                    "is_extrapolation_condition": role != "train" and condition not in train_conditions,
                    "n_rows": int(len(seq.df)),
                }
            )
    return pd.DataFrame(rows)


def save_model_artifact(
    output_dir: Path,
    model: NARXMLP,
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    args: argparse.Namespace,
    feature_names: list[str],
    training_summary: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "NARXMLP",
            "feature_names": feature_names,
            "input_scaler": input_scaler,
            "target_scaler": target_scaler,
            "args": vars(args),
            "training_summary": training_summary,
        },
        output_dir / "nn_narx_model.pt",
    )


def save_outputs(
    output_dir: Path,
    model: NARXMLP,
    input_scaler: StandardScaler,
    target_scaler: StandardScaler,
    args: argparse.Namespace,
    feature_names: list[str],
    training_summary: dict[str, Any],
    history: pd.DataFrame,
    split: DatasetSplit,
    train_predictions: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    train_metrics: dict[str, dict[str, float | int]],
    validation_metrics: dict[str, dict[str, float | int]],
    test_metrics: dict[str, dict[str, float | int]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model_artifact(output_dir, model, input_scaler, target_scaler, args, feature_names, training_summary)
    history.to_csv(output_dir / "nn_narx_training_history.csv", index=False)
    split.manifest.to_csv(output_dir / "nn_narx_split_manifest.csv", index=False)
    train_predictions.to_csv(output_dir / "nn_narx_train_predictions.csv", index=False)
    validation_predictions.to_csv(output_dir / "nn_narx_validation_predictions.csv", index=False)
    test_predictions.to_csv(output_dir / "nn_narx_test_predictions.csv", index=False)
    metrics_by_file(train_predictions).to_csv(output_dir / "nn_narx_train_per_file_metrics.csv", index=False)
    metrics_by_file(validation_predictions).to_csv(output_dir / "nn_narx_validation_per_file_metrics.csv", index=False)
    metrics_by_file(test_predictions).to_csv(output_dir / "nn_narx_test_per_file_metrics.csv", index=False)
    metrics_by_machine(train_predictions).to_csv(output_dir / "nn_narx_train_per_machine_metrics.csv", index=False)
    metrics_by_machine(validation_predictions).to_csv(output_dir / "nn_narx_validation_per_machine_metrics.csv", index=False)
    metrics_by_machine(test_predictions).to_csv(output_dir / "nn_narx_test_per_machine_metrics.csv", index=False)
    split_condition_coverage(split).to_csv(output_dir / "nn_narx_condition_coverage.csv", index=False)
    for prefix, prediction_df in (
        ("train", train_predictions),
        ("validation", validation_predictions),
        ("test", test_predictions),
    ):
        metrics_by_dimensions(prediction_df, ["wfs"]).to_csv(
            output_dir / f"nn_narx_{prefix}_per_wfs_metrics.csv",
            index=False,
        )
        metrics_by_dimensions(prediction_df, ["machine_group", "wfs"]).to_csv(
            output_dir / f"nn_narx_{prefix}_per_machine_wfs_metrics.csv",
            index=False,
        )

    metrics = {
        "model": "NARXMLP",
        "model_variant": model_variant_from_args(args),
        "data_folder": str(args.data_dir),
        "machines": args.machines,
        "target_col": args.target_col,
        "input_cols": input_columns_from_arg(args.input_cols),
        "rolling_windows": rolling_windows_from_arg(args.rolling_windows),
        "extended_features": bool(args.extended_features),
        "static_context_cols": list(STATIC_CONTEXT_COLUMNS),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "input_lags": int(args.input_lags),
        "output_lags": int(args.output_lags),
        "target_mode": str(args.target_mode),
        "feedback_prob_final": float(args.feedback_prob_final),
        "feedback_warmup_epochs": int(args.feedback_warmup_epochs),
        "rollout_loss_weight": float(args.rollout_loss_weight),
        "rollout_warmup_epochs": int(args.rollout_warmup_epochs),
        "rollout_train_windows": int(args.rollout_train_windows),
        "rollout_window_size": int(args.rollout_window_size),
        "absolute_loss_weight": float(args.absolute_loss_weight),
        "input_noise_std": float(args.input_noise_std),
        "loss": str(args.loss),
        "huber_delta": float(args.huber_delta),
        "rollout_validation_every": int(args.rollout_validation_every),
        "prediction_min": float(args.prediction_min),
        "prediction_max_margin": float(args.prediction_max_margin),
        "split_note": split.note,
        "training_summary": training_summary,
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    with open(output_dir / "nn_narx_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(json_ready(metrics), handle, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a neural NARX model and save metrics, split manifest, and predictions.",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_SAVE_PATH)
    parser.add_argument("--target-col", type=str, default=TARGET_VARIABLE)
    parser.add_argument(
        "--input-cols",
        type=str,
        default=",".join(DEFAULT_INPUT_COLUMNS),
        help=(
            "Comma-separated live dynamic input columns. "
            "wfs is added automatically as static context and ctdw/ctwd is not allowed."
        ),
    )
    parser.add_argument("--frame-order-col", type=str, default="meas_time")
    parser.add_argument(
        "--machines",
        choices=["both", "fr", "we"],
        default="both",
        help="Machine CSV filename prefixes to include.",
    )
    parser.add_argument(
        "--split-method",
        choices=["condition", "by_run", "chronological"],
        default="condition",
        help="Deprecated; NN NARX now always uses preprocessing.split.split_data_folder.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--input-lags", type=int, default=10)
    parser.add_argument("--output-lags", type=int, default=3)
    parser.add_argument("--target-mode", choices=["delta", "absolute"], default="delta")
    parser.add_argument("--feedback-prob-final", type=float, default=0.3)
    parser.add_argument(
        "--feedback-warmup-epochs",
        type=int,
        default=20,
        help="Epochs used to ramp scheduled-feedback probability to --feedback-prob-final.",
    )
    parser.add_argument(
        "--rollout-loss-weight",
        type=float,
        default=0.0,
        help="Weight for differentiable free-run rollout loss. Set >0 to train directly against rollout drift.",
    )
    parser.add_argument(
        "--rollout-warmup-epochs",
        type=int,
        default=10,
        help="Epochs used to ramp rollout loss weight to --rollout-loss-weight.",
    )
    parser.add_argument(
        "--rollout-train-windows",
        type=int,
        default=0,
        help="Random recursive rollout windows sampled per supervised batch.",
    )
    parser.add_argument(
        "--rollout-window-size",
        type=int,
        default=250,
        help="Number of timesteps in each differentiable rollout training window.",
    )
    parser.add_argument(
        "--absolute-loss-weight",
        type=float,
        default=0.25,
        help="Auxiliary absolute-level loss inside rollout training, useful for delta targets.",
    )
    parser.add_argument("--rollout-validation-every", type=int, default=1)
    parser.add_argument("--extended-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rolling-windows",
        type=str,
        default=",".join(str(value) for value in DEFAULT_ROLLING_WINDOWS),
        help="Comma-separated rolling window sizes for physics-aware summary features.",
    )
    parser.add_argument("--prediction-min", type=float, default=0.0)
    parser.add_argument("--prediction-max-margin", type=float, default=1.0)
    parser.add_argument("--hidden-sizes", type=str, default="64,32")
    parser.add_argument("--activation", choices=["relu", "gelu", "silu"], default="relu")
    parser.add_argument("--layer-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-2)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1.0e-4)
    parser.add_argument("--clip-grad-norm", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    validate_training_args(args)
    set_random_seed(int(args.random_seed))
    device = resolve_device(str(args.device))
    input_cols = input_columns_from_arg(args.input_cols)
    rolling_windows = rolling_windows_from_arg(args.rolling_windows)
    print(f"Using device: {device}", flush=True)
    machine_groups = machine_groups_from_arg(str(args.machines))
    if str(args.split_method) != "condition":
        print(
            "Warning: --split-method is deprecated and ignored; using preprocessing.split.split_data_folder.",
            flush=True,
        )
    split_config = build_nn_narx_split_config(args, input_cols, rolling_windows)
    print(f"Loading split CSV subsequences from {args.data_dir}", flush=True)
    split = split_data_folder(
        args.data_dir,
        machine_type=machine_type_from_arg(str(args.machines)),
        config=split_config,
    )
    print(
        f"Using machine groups: {', '.join(machine_groups)} "
        f"(train={len(split.train)}, validation={len(split.validation)}, test={len(split.test)} sequences)",
        flush=True,
    )
    print(split.note, flush=True)
    coverage = split_condition_coverage(split)
    extrapolated = coverage.loc[coverage["is_extrapolation_condition"]]
    if not extrapolated.empty:
        conditions = (
            extrapolated[["machine_group", "wfs"]]
            .drop_duplicates()
            .sort_values(["machine_group", "wfs"])
        )
        condition_text = ", ".join(
            f"{row.machine_group}:wfs={row.wfs:g}"
            for row in conditions.itertuples(index=False)
        )
        print(f"Warning: validation/test contains conditions absent from train: {condition_text}", flush=True)

    prediction_min, prediction_max = prediction_bounds(split.train, args.target_col, args)
    x_train, y_train, train_metadata, dynamic_feature_names, context_names, dynamic_base_count = build_training_matrix(
        split.train,
        input_cols,
        args.target_col,
        int(args.input_lags),
        int(args.output_lags),
        str(args.target_mode),
        rolling_windows,
        bool(args.extended_features),
    )
    x_val, y_val, _, _, _, _ = build_training_matrix(
        split.validation,
        input_cols,
        args.target_col,
        int(args.input_lags),
        int(args.output_lags),
        str(args.target_mode),
        rolling_windows,
        bool(args.extended_features),
    )
    full_feature_names = [*dynamic_feature_names, *context_names]
    dynamic_feature_count = len(dynamic_feature_names)
    print(
        f"Training rows={len(x_train):,}, validation rows={len(x_val):,}, features={len(full_feature_names)}",
        flush=True,
    )

    model, input_scaler, target_scaler, history, training_summary = train_model(
        x_train,
        y_train,
        train_metadata,
        split.train,
        x_val,
        y_val,
        split.validation,
        input_cols,
        dynamic_base_count,
        rolling_windows,
        args.target_col,
        args,
        device,
        prediction_min,
        prediction_max,
    )

    train_predictions, train_metrics = evaluate_sequences(
        model,
        split.train,
        input_cols,
        args.target_col,
        int(args.input_lags),
        int(args.output_lags),
        rolling_windows,
        bool(args.extended_features),
        input_scaler,
        target_scaler,
        device,
        str(args.target_mode),
        prediction_min,
        prediction_max,
    )
    validation_predictions, validation_metrics = evaluate_sequences(
        model,
        split.validation,
        input_cols,
        args.target_col,
        int(args.input_lags),
        int(args.output_lags),
        rolling_windows,
        bool(args.extended_features),
        input_scaler,
        target_scaler,
        device,
        str(args.target_mode),
        prediction_min,
        prediction_max,
    )
    test_predictions, test_metrics = evaluate_sequences(
        model,
        split.test,
        input_cols,
        args.target_col,
        int(args.input_lags),
        int(args.output_lags),
        rolling_windows,
        bool(args.extended_features),
        input_scaler,
        target_scaler,
        device,
        str(args.target_mode),
        prediction_min,
        prediction_max,
    )

    save_outputs(
        args.output_dir,
        model,
        input_scaler,
        target_scaler,
        args,
        full_feature_names,
        training_summary,
        history,
        split,
        train_predictions,
        validation_predictions,
        test_predictions,
        train_metrics,
        validation_metrics,
        test_metrics,
    )
    print(f"Saved NN NARX outputs to {args.output_dir}", flush=True)
    print("Validation free-run metrics:", json.dumps(json_ready(validation_metrics["free_run"]), indent=2), flush=True)
    print("Test free-run metrics:", json.dumps(json_ready(test_metrics["free_run"]), indent=2), flush=True)


if __name__ == "__main__":
    run(parse_args())
