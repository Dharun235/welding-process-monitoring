"""
ARX model identification for configurable weldpool-distance prediction.

Discrete-time multi-input ARX model with live engineered exogenous features:

    y(t) = a_1*y(t-1) + ... + a_na*y(t-na)
           + b_{1,0}*u1(t-nk) + ... + b_{m,nb-1}*um(t-nk-nb+1)
           + e(t)

where:
    y  = selected weldpool-distance target column
    ui = live measurements and codex-style engineered live features derived
         from voltage, current, delta_voltage, delta_current, and wfs.
    nk=0 includes live inputs at the current sample t.

Fitted by ordinary least squares (numpy.linalg.lstsq).
Model orders are chosen by a grid search over (na, nb, nk), defaulting to
na, nb in {0, 1, 3, 5, 10, 25, 50} and nk in [0, 5], using the shared
preprocessing train/validation/test split, with the selection objective configurable between
one-step-ahead and free-run scoring.
Lags never cross sequence boundaries, preventing temporal leakage.

Usage:
    python arx.py --data_folder /path/to/csvs
    python arx.py --data_folder /path/to/csvs --na_values 0,1,5 --nb_values 0,1,5 --nk_max 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ─── project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).resolve().parents[3]
REPO_ROOT       = PROJECT_ROOT.parent
DEFAULT_DATA_FOLDER = PROJECT_ROOT / "data" / "merged_diff"
DEFAULT_OUTPUT_DIR  = PROJECT_ROOT / "output" / "arx" / "tapering_to_weldpool"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictive_modeling.scripts.config import CONFIG as PROJECT_CONFIG
from predictive_modeling.scripts.preprocessing.split import DatasetSplit, SequenceData, split_data_folder

# ─── column definitions ────────────────────────────────────────────────────────
TARGET_COL = "tapering_to_weldpool_dist"
RAW_INPUT_COLS = ["voltage", "current"]
HISTORY_COLS = ["voltage", "current", "delta_voltage", "delta_current"]
CONTEXT_COLS = ["wfs"]
INPUT_COLS = [*HISTORY_COLS, *CONTEXT_COLS]
FEATURE_SET = "live_engineered"
SUMMARY_WINDOW: int | None = None
DEFAULT_ORDER_VALUES = [0, 1, 3, 5, 10, 25, 50]
ROLLING_SUMMARY_WINDOWS = [5, 10, 25, 50]
ARC_CURRENT_THRESHOLD = 5.0
ARC_VOLTAGE_THRESHOLD = 1.0
SAFE_DIVISION_EPS = 1.0e-6
NORMAL_EQUATION_JITTER = 1.0e-10

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SELECTION_MODE_LABELS = {
    "one_step_ahead": "one-step-ahead",
    "free_run": "free-run",
}
SELECTION_METRIC_LABELS = {
    "rmse": "RMSE",
    "aic": "AIC",
    "bic": "BIC",
}


class ProgressBar:
    """Minimal terminal progress bar that does not require external packages."""

    def __init__(
        self,
        total: int,
        label: str,
        *,
        width: int = 28,
        min_interval: float = 0.1,
    ) -> None:
        self.total = max(int(total), 0)
        self.label = label
        self.width = max(int(width), 10)
        self.min_interval = max(float(min_interval), 0.0)
        self.current = 0
        self._start = time.perf_counter()
        self._last_render = 0.0
        self._last_message = ""
        self._last_line_len = 0
        self._enabled = self.total > 0 and sys.stderr.isatty()

        if self._enabled:
            self._render(force=True)

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def update(self, step: int = 1, message: str = "") -> None:
        """Advance the bar and refresh the terminal output when appropriate."""
        self.current = min(self.current + step, self.total)
        if message:
            self._last_message = message
        self._render()

    def close(self) -> None:
        """Render the final state and move to the next terminal line."""
        if not self._enabled:
            return
        self._render(force=True)
        print(file=sys.stderr, flush=True)

    def _render(self, *, force: bool = False) -> None:
        if not self._enabled:
            return

        now = time.perf_counter()
        if not force and self.current < self.total:
            if (now - self._last_render) < self.min_interval:
                return

        ratio = self.current / self.total if self.total else 1.0
        filled = min(self.width, int(round(self.width * ratio)))
        bar = "#" * filled + "-" * (self.width - filled)

        elapsed = max(now - self._start, 1e-9)
        rate = self.current / elapsed
        line = (
            f"\r{self.label}: [{bar}] {self.current:>4}/{self.total:<4}"
            f" ({ratio * 100:5.1f}%)"
        )
        if 0 < self.current < self.total and rate > 0:
            eta = (self.total - self.current) / rate
            line += f" ETA {eta:6.1f}s"
        if self._last_message:
            line += f" | {self._last_message}"

        padded = line.ljust(self._last_line_len)
        print(padded, end="", file=sys.stderr, flush=True)
        self._last_line_len = len(line)
        self._last_render = now

# ══════════════════════════════════════════════════════════════════════════════
# DATA SPLITTING
# ══════════════════════════════════════════════════════════════════════════════


def build_arx_split_config() -> dict[str, Any]:
    """
    Return the shared preprocessing split config adjusted to ARX inputs.

    The split policy, fractions, subsequence length, gap, and random seed come
    from preprocessing/split.py's project config. The model-specific required
    history columns are restricted to the ARX exogenous inputs so this baseline
    does not depend on unused engineered features.
    """
    split_config = dict(PROJECT_CONFIG["preprocessing"]["split"])
    if FEATURE_SET == "raw_electrical":
        history_cols = RAW_INPUT_COLS
        input_cols = RAW_INPUT_COLS
        context_cols: list[str] = []
    else:
        history_cols = HISTORY_COLS
        input_cols = INPUT_COLS
        context_cols = CONTEXT_COLS
    split_config.update(
        {
            "target_col": TARGET_COL,
            "history_cols": history_cols,
            "input_cols": input_cols,
            "context_cols": context_cols,
            "offline_metadata_cols": ["wfs", "ctdw"],
        }
    )
    return split_config


def load_dataset_split(
    folder: Path,
    *,
    machine_type: str | None = None,
    wfs: float | None = None,
    ctdw: float | None = None,
) -> DatasetSplit:
    """Load train/validation/test subsequences using preprocessing.split."""
    split = split_data_folder(
        folder,
        machine_type=machine_type,
        wfs=wfs,
        ctdw=ctdw,
        config=build_arx_split_config(),
    )
    logger.info(split.note)
    logger.info(
        "Split sequences: train=%d  validation=%d  test=%d",
        len(split.train), len(split.validation), len(split.test),
    )
    return split


def split_sequences_to_datasets(sequences: list[SequenceData]) -> list[tuple[str, pd.DataFrame]]:
    """Convert split.py SequenceData objects to the local ARX dataset format."""
    return [(sequence.file_name, sequence.df.reset_index(drop=True)) for sequence in sequences]

# ══════════════════════════════════════════════════════════════════════════════
# ARX REGRESSOR CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def shifted(values: np.ndarray, lag: int) -> np.ndarray:
    """Return values delayed by *lag* samples, padded with NaN."""
    out = np.full(len(values), np.nan, dtype=float)
    if lag == 0:
        out[:] = values
    elif lag < len(values):
        out[lag:] = values[:-lag]
    return out


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = SAFE_DIVISION_EPS) -> np.ndarray:
    """Elementwise divide with zero/near-zero denominators mapped to 0."""
    denominator = np.asarray(denominator, dtype=float)
    safe_denominator = np.where(np.abs(denominator) > eps, denominator, np.nan)
    result = np.asarray(numerator, dtype=float) / safe_denominator
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def rolling_summary_features(values: np.ndarray, column: str, windows: list[int]) -> dict[str, np.ndarray]:
    """Codex-style rolling summaries for a live input column."""
    series = pd.Series(values, dtype="float64")
    features: dict[str, np.ndarray] = {}
    for window in windows:
        if window < 2:
            continue
        rolling = series.rolling(window=window, min_periods=window)
        prefix = f"{column}_roll_{window}"
        features[f"{prefix}_mean"] = rolling.mean().to_numpy(dtype=float)
        features[f"{prefix}_std"] = rolling.std(ddof=0).to_numpy(dtype=float)
        features[f"{prefix}_min"] = rolling.min().to_numpy(dtype=float)
        features[f"{prefix}_max"] = rolling.max().to_numpy(dtype=float)
        oldest = shifted(values, window - 1)
        delta = values - oldest
        features[f"{prefix}_delta"] = delta
        features[f"{prefix}_slope"] = delta / float(window - 1)
    return features


def rolling_pair_features(
    first: np.ndarray,
    second: np.ndarray,
    *,
    first_name: str,
    second_name: str,
    windows: list[int],
) -> dict[str, np.ndarray]:
    """Codex-style rolling pair summaries for current and voltage."""
    first_series = pd.Series(first, dtype="float64")
    second_series = pd.Series(second, dtype="float64")
    product_series = first_series * second_series
    features: dict[str, np.ndarray] = {}
    for window in windows:
        if window < 2:
            continue
        prefix = f"{first_name}_{second_name}_roll_{window}"
        first_mean = first_series.rolling(window=window, min_periods=window).mean()
        second_mean = second_series.rolling(window=window, min_periods=window).mean()
        product_mean = product_series.rolling(window=window, min_periods=window).mean()
        first_std = first_series.rolling(window=window, min_periods=window).std(ddof=0)
        second_std = second_series.rolling(window=window, min_periods=window).std(ddof=0)
        covariance = product_mean - (first_mean * second_mean)
        correlation = safe_divide(
            covariance.to_numpy(dtype=float),
            (first_std * second_std).to_numpy(dtype=float),
        )
        features[f"{prefix}_cov"] = covariance.to_numpy(dtype=float)
        features[f"{prefix}_corr"] = correlation
        features[f"{prefix}_mean_product_minus_product_mean"] = (
            (first_mean * second_mean) - product_mean
        ).to_numpy(dtype=float)
    return features


def rolling_windows_for_nb(nb: int) -> list[int]:
    """Only include rolling windows that fit inside the tested ARX input history."""
    return [window for window in ROLLING_SUMMARY_WINDOWS if window <= max(nb, 0)]


def specified_summary_window(nb: int) -> int:
    """Return the rolling-statistics window used by the specified-input model."""
    return int(SUMMARY_WINDOW if SUMMARY_WINDOW is not None else nb)


def build_live_feature_matrix(df: pd.DataFrame, nb: int) -> tuple[np.ndarray, list[str]]:
    """
    Build live exogenous ARX inputs for one sequence.

    The returned columns are measurements or summaries available at timestep t.
    ``build_regressor_matrix`` applies the ARX ``nb`` and ``nk`` history/delay
    to these live columns, so lags still never cross sequence boundaries.
    """
    if FEATURE_SET == "raw_electrical":
        missing = [column for column in RAW_INPUT_COLS if column not in df.columns]
        if missing:
            raise ValueError(f"ARX input data is missing required columns: {missing}")
        if nb <= 0:
            return np.empty((len(df), 0), dtype=float), []
        return (
            df[RAW_INPUT_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
            list(RAW_INPUT_COLS),
        )

    missing = [column for column in INPUT_COLS if column not in df.columns]
    if missing:
        raise ValueError(f"ARX input data is missing required live columns: {missing}")

    if FEATURE_SET == "specified_inputs":
        if nb <= 0:
            return np.empty((len(df), 0), dtype=float), []
        summary_window = specified_summary_window(nb)
        if summary_window < 2:
            raise ValueError("--summary_window must be at least 2 for specified_inputs.")

        current = pd.to_numeric(df["current"], errors="coerce").to_numpy(dtype=float)
        voltage = pd.to_numeric(df["voltage"], errors="coerce").to_numpy(dtype=float)
        delta_current = pd.to_numeric(df["delta_current"], errors="coerce").to_numpy(dtype=float)
        delta_voltage = pd.to_numeric(df["delta_voltage"], errors="coerce").to_numpy(dtype=float)
        power = current * voltage
        features: dict[str, np.ndarray] = {}

        for signal_name, values in (
            ("current", current),
            ("voltage", voltage),
            ("delta_current", delta_current),
            ("delta_voltage", delta_voltage),
            ("power", power),
        ):
            for lag_offset in range(nb):
                features[f"{signal_name}_lag_{lag_offset}"] = shifted(values, lag_offset)

        current_stats = rolling_summary_features(current, "current", [summary_window])
        voltage_stats = rolling_summary_features(voltage, "voltage", [summary_window])
        for suffix in ("mean", "std", "slope", "min", "max"):
            features[f"current_roll_{summary_window}_{suffix}"] = current_stats[
                f"current_roll_{summary_window}_{suffix}"
            ]
            features[f"voltage_roll_{summary_window}_{suffix}"] = voltage_stats[
                f"voltage_roll_{summary_window}_{suffix}"
            ]
        features["wfs"] = pd.to_numeric(df["wfs"], errors="coerce").to_numpy(dtype=float)
        feature_names = list(features)
        return pd.DataFrame(features).to_numpy(dtype=float), feature_names

    features: dict[str, np.ndarray] = {}
    feature_names: list[str] = []
    summary_windows = rolling_windows_for_nb(nb)

    for column in HISTORY_COLS:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        features[column] = values
        feature_names.append(column)
        for feature_name, feature_values in rolling_summary_features(values, column, summary_windows).items():
            features[feature_name] = feature_values
            feature_names.append(feature_name)

    for column in CONTEXT_COLS:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        feature_name = f"{column}_setpoint"
        features[feature_name] = values
        feature_names.append(feature_name)

    current = pd.to_numeric(df["current"], errors="coerce").to_numpy(dtype=float)
    voltage = pd.to_numeric(df["voltage"], errors="coerce").to_numpy(dtype=float)
    wfs = pd.to_numeric(df["wfs"], errors="coerce").to_numpy(dtype=float)
    power = current * voltage

    engineered = {
        "arc_active_est_feature": (
            (current > ARC_CURRENT_THRESHOLD) | (voltage > ARC_VOLTAGE_THRESHOLD)
        ).astype(float),
        "current_voltage_product": power,
        "current_to_voltage_ratio": safe_divide(current, voltage),
        "voltage_to_current_ratio": safe_divide(voltage, current),
        "power_proxy": power,
        "resistance_proxy": safe_divide(voltage, current),
        "conductance_proxy": safe_divide(current, voltage),
        "current_abs_delta_1": np.abs(current - shifted(current, 1)),
        "voltage_abs_delta_1": np.abs(voltage - shifted(voltage, 1)),
        "power_abs_delta_1": np.abs(power - shifted(power, 1)),
        "current_per_wfs": safe_divide(current, wfs),
        "voltage_per_wfs": safe_divide(voltage, wfs),
        "power_per_wfs": safe_divide(power, wfs),
        "wfs_to_current_ratio": safe_divide(wfs, current),
    }
    for feature_name, feature_values in engineered.items():
        features[feature_name] = feature_values
        feature_names.append(feature_name)

    for feature_name, feature_values in rolling_pair_features(
        current,
        voltage,
        first_name="current",
        second_name="voltage",
        windows=summary_windows,
    ).items():
        features[feature_name] = feature_values
        feature_names.append(feature_name)

    for window in summary_windows:
        current_roll_mean = features.get(f"current_roll_{window}_mean")
        if current_roll_mean is not None:
            feature_name = f"wfs_current_roll_{window}_mean"
            features[feature_name] = wfs * current_roll_mean
            feature_names.append(feature_name)

    if nb <= 0:
        return np.empty((len(df), 0), dtype=float), []

    return pd.DataFrame(features, columns=feature_names).to_numpy(dtype=float), feature_names


def arx_coefficient_names(na: int, nb: int, nk: int, feature_names: list[str]) -> list[str]:
    """Return coefficient names in the same order as the ARX regressor matrix."""
    coef_names: list[str] = [f"a{k + 1}" for k in range(na)]
    if FEATURE_SET == "specified_inputs":
        coef_names.extend(f"b_{feature_name}_delay{nk}" for feature_name in feature_names)
        return coef_names
    for feature_name in feature_names:
        for lag_offset in range(nb):
            coef_names.append(f"b_{feature_name}_delay{nk + lag_offset}")
    return coef_names


def compute_t_start(na: int, nb: int, nk: int) -> int:
    """Return the first sample index that has enough lag history for ARX."""
    if FEATURE_SET == "specified_inputs":
        summary_start = specified_summary_window(nb) - 1 if nb > 0 else 0
        input_start = nk + nb - 1 if nb > 0 else 0
        return max(na, input_start, nk + summary_start)
    input_start = nk + nb - 1 if nb > 0 else 0
    summary_start = max((window - 1 for window in rolling_windows_for_nb(nb)), default=0)
    return max(na, input_start, summary_start)


def build_regressor_matrix(
    y: np.ndarray,
    U: np.ndarray,
    na: int,
    nb: int,
    nk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the ARX regressor matrix Phi and target vector Y for one time series.

    Each row of Phi is:
        phi(t) = [y(t-1), ..., y(t-na),
                  u1(t-nk), ..., u1(t-nk-nb+1),
                  u2(t-nk), ..., u2(t-nk-nb+1)]

    Parameters
    ----------
    y    : (N,)      target signal, one file
    U    : (N, nu)   input signals; columns ordered as INPUT_COLS
    na   : int       number of AR lags (y(t-1) … y(t-na))
    nb   : int       number of input lags per channel
    nk   : int       input delay (nk=0 → u(t) is included; nk=1 → u(t-1) is first)

    Returns
    -------
    Phi : (M, na + nu*nb)  regressor matrix  (M = N - t_start valid samples)
    Y   : (M,)             corresponding targets
    """
    N, nu = len(y), U.shape[1]
    t_start = compute_t_start(na, nb, nk)

    input_parameter_count = nu if FEATURE_SET == "specified_inputs" else nu * nb
    if t_start >= N:
        return np.empty((0, na + input_parameter_count), dtype=float), np.empty(0, dtype=float)

    M = N - t_start

    # Vectorised construction: each lag is a contiguous slice of the full array.
    Phi = np.empty((M, na + input_parameter_count), dtype=float)
    col = 0

    # AR lags: y(t-1), y(t-2), ..., y(t-na)
    for lag in range(1, na + 1):
        s = t_start - lag          # index of y(t-lag) at first valid t
        Phi[:, col] = y[s: s + M]
        col += 1

    if FEATURE_SET == "specified_inputs":
        Phi[:, col:] = U[t_start - nk: t_start - nk + M, :]
    else:
        # Input lags for each channel: u(t-nk), u(t-nk-1), ..., u(t-nk-nb+1)
        for u_idx in range(nu):
            for lag_offset in range(nb):
                s = t_start - nk - lag_offset
                Phi[:, col] = U[s: s + M, u_idx]
                col += 1

    return Phi, y[t_start:]


def stack_regressors(
    datasets: list[tuple[str, pd.DataFrame]],
    na: int,
    nb: int,
    nk: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build regressors independently per file and vertically concatenate them.
    Lags never cross file boundaries.
    """
    Phis, Ys = [], []
    feature_names: list[str] = []
    for _, df in datasets:
        y = df[TARGET_COL].to_numpy(dtype=float)
        U, current_feature_names = build_live_feature_matrix(df, nb)
        if not feature_names:
            feature_names = current_feature_names
        Phi, Y = build_regressor_matrix(y, U, na, nb, nk)
        if len(Y) > 0:
            valid = np.isfinite(Phi).all(axis=1) & np.isfinite(Y)
            Phi, Y = Phi[valid], Y[valid]
        if len(Y) > 0:
            Phis.append(Phi)
            Ys.append(Y)

    if not Phis:
        return (
            np.empty((0, len(arx_coefficient_names(na, nb, nk, feature_names))), dtype=float),
            np.empty(0, dtype=float),
            feature_names,
        )

    return np.vstack(Phis), np.concatenate(Ys), feature_names


# ══════════════════════════════════════════════════════════════════════════════
# FIT AND PREDICT
# ══════════════════════════════════════════════════════════════════════════════

def fit_arx(Phi: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fit ARX model via ordinary least squares. Returns coefficient vector theta."""
    theta, _, _, _ = np.linalg.lstsq(Phi, Y, rcond=None)
    return theta


def solve_normal_equations(gram: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve the accumulated least-squares normal equations."""
    if gram.size == 0:
        return np.empty(0, dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", scipy.linalg.LinAlgWarning)
            return scipy.linalg.solve(
                gram,
                rhs,
                assume_a="sym",
                check_finite=False,
            )
    except (scipy.linalg.LinAlgError, scipy.linalg.LinAlgWarning):
        gram_regularized = gram.copy()
        diagonal_jitter = NORMAL_EQUATION_JITTER * max(
            float(np.trace(gram_regularized)) / max(gram_regularized.shape[0], 1),
            1.0,
        )
        gram_regularized.flat[:: gram_regularized.shape[0] + 1] += diagonal_jitter
        return scipy.linalg.solve(
            gram_regularized,
            rhs,
            assume_a="pos",
            check_finite=False,
        )


def fit_arx_from_datasets(
    datasets: list[tuple[str, pd.DataFrame]],
    na: int,
    nb: int,
    nk: int,
) -> tuple[np.ndarray, list[str], int]:
    """
    Fit ARX by accumulating X.T @ X and X.T @ y per subsequence.

    This avoids materializing the full training regressor matrix, which becomes
    multi-GB for high nb values and can be killed by the OS.
    """
    feature_names: list[str] = []
    gram: np.ndarray | None = None
    rhs: np.ndarray | None = None
    n_samples = 0

    for _, df in datasets:
        y = df[TARGET_COL].to_numpy(dtype=float)
        U, current_feature_names = build_live_feature_matrix(df, nb)
        if not feature_names:
            feature_names = current_feature_names
        Phi, Y = build_regressor_matrix(y, U, na, nb, nk)
        if len(Y) == 0:
            continue

        valid = np.isfinite(Phi).all(axis=1) & np.isfinite(Y)
        if not valid.any():
            continue

        Phi_valid = np.ascontiguousarray(Phi[valid], dtype=float)
        Y_valid = np.ascontiguousarray(Y[valid], dtype=float)
        if gram is None or rhs is None:
            n_params = Phi_valid.shape[1]
            gram = np.zeros((n_params, n_params), dtype=float)
            rhs = np.zeros(n_params, dtype=float)

        gram += Phi_valid.T @ Phi_valid
        rhs += Phi_valid.T @ Y_valid
        n_samples += int(Y_valid.size)

    if gram is None or rhs is None:
        n_params = len(arx_coefficient_names(na, nb, nk, feature_names))
        return np.empty(n_params, dtype=float), feature_names, 0

    return solve_normal_equations(gram, rhs), feature_names, n_samples


def simulate_free_run_series(
    y: np.ndarray,
    U: np.ndarray,
    theta: np.ndarray,
    na: int,
    nb: int,
    nk: int,
) -> tuple[int, np.ndarray | None]:
    """
    Simulate one file on its native timestep grid.

    Returns (t_start, y_sim). The leading rows before t_start are seeded with
    true observations so later AR lags are well-defined. If the series is too
    short for the requested orders, y_sim is None.
    """
    N = len(y)
    nu = U.shape[1]
    t_start = compute_t_start(na, nb, nk)

    if t_start >= N:
        return t_start, None

    seeds = np.where(np.isfinite(y[:t_start]), y[:t_start], 0.0)
    y_sim = np.empty(N, dtype=float)
    y_sim[:t_start] = seeds

    input_parameter_count = nu if FEATURE_SET == "specified_inputs" else nu * nb
    n_params = na + input_parameter_count
    features = np.empty(n_params, dtype=float)

    for t in range(t_start, N):
        col = 0
        for lag in range(1, na + 1):
            features[col] = y_sim[t - lag]
            col += 1
        if FEATURE_SET == "specified_inputs":
            features[col:] = U[t - nk, :]
        else:
            for u_idx in range(nu):
                for lag_offset in range(nb):
                    features[col] = U[t - nk - lag_offset, u_idx]
                    col += 1

        y_sim[t] = float(features @ theta)

    return t_start, y_sim


def predict_one_step(
    datasets: list[tuple[str, pd.DataFrame]],
    theta: np.ndarray,
    na: int,
    nb: int,
    nk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-step-ahead prediction using true past observations (no free-run).
    Returns (y_true, y_pred) arrays across all files in *datasets*.
    """
    y_true_parts, y_pred_parts = [], []
    for _, df in datasets:
        y = df[TARGET_COL].to_numpy(dtype=float)
        U, _ = build_live_feature_matrix(df, nb)
        Phi, Y = build_regressor_matrix(y, U, na, nb, nk)
        if len(Y) > 0:
            valid = np.isfinite(Phi).all(axis=1) & np.isfinite(Y)
            Phi, Y = Phi[valid], Y[valid]
        if len(Y) > 0:
            y_true_parts.append(Y)
            y_pred_parts.append(Phi @ theta)

    if not y_true_parts:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts)


def predict_free_run(
    datasets: list[tuple[str, pd.DataFrame]],
    theta: np.ndarray,
    na: int,
    nb: int,
    nk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Free-run (simulated output) prediction.

    The AR lags are filled with the model's own previous predictions instead of
    the true observed target.  Input lags always use the true measured signals
    (voltage and current), which are assumed to be known/measured in real use.

    The first *t_start* true observations seed the AR history so that the
    warm-up window has valid lag values.  After that every AR lag comes from
    the model's own output — prediction errors accumulate over time, giving a
    realistic picture of how the model would behave when deployed online.

    Returns (y_true, y_pred) arrays across all files in *datasets*.
    """
    y_true_parts, y_pred_parts = [], []

    for _, df in datasets:
        y = df[TARGET_COL].to_numpy(dtype=float)
        U, _ = build_live_feature_matrix(df, nb)
        t_start, y_sim = simulate_free_run_series(y, U, theta, na, nb, nk)
        if y_sim is None:
            continue

        y_true = y[t_start:]
        y_pred = y_sim[t_start:]
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_parts.append(y_true[valid])
        y_pred_parts.append(y_pred[valid])

    if not y_true_parts:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)

    return np.concatenate(y_true_parts), np.concatenate(y_pred_parts)


def build_aligned_prediction_rows(
    datasets: list[tuple[str, pd.DataFrame]],
    theta: np.ndarray,
    na: int,
    nb: int,
    nk: int,
) -> pd.DataFrame:
    """
    Return predictions aligned to every timestep in each CSV.

    The first rows without enough lag history are kept in the export with NaN
    predictions so the saved CSV stays aligned with the original validation
    samples.
    """
    aligned_parts: list[pd.DataFrame] = []
    t_start = compute_t_start(na, nb, nk)

    for file_name, df in datasets:
        y = df[TARGET_COL].to_numpy(dtype=float)
        U, _ = build_live_feature_matrix(df, nb)
        N = len(y)

        y_pred_osa = np.full(N, np.nan, dtype=float)
        y_pred_fr = np.full(N, np.nan, dtype=float)
        has_full_history = np.zeros(N, dtype=bool)

        if t_start < N:
            Phi, Y = build_regressor_matrix(y, U, na, nb, nk)
            row_indices = np.arange(t_start, N, dtype=int)
            valid = np.isfinite(Phi).all(axis=1) & np.isfinite(Y)
            if valid.any():
                osa_values = np.full(len(row_indices), np.nan, dtype=float)
                osa_values[valid] = Phi[valid] @ theta
                y_pred_osa[t_start:] = osa_values

            _, y_sim = simulate_free_run_series(y, U, theta, na, nb, nk)
            if y_sim is not None:
                y_pred_fr[t_start:] = y_sim[t_start:]

            has_full_history[t_start:] = True

        file_rows: dict[str, Any] = {
            "file": file_name,
            "sample_index": np.arange(N, dtype=int),
            "has_full_history": has_full_history,
            "y_true": y,
            "y_pred_osa": y_pred_osa,
            "y_pred_freerun": y_pred_fr,
        }
        for column in INPUT_COLS:
            if column in df.columns:
                file_rows[column] = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if "meas_time" in df.columns:
            file_rows["meas_time"] = df["meas_time"].to_numpy(dtype=float)
        if "frame_number" in df.columns:
            file_rows["frame_number"] = df["frame_number"].to_numpy()

        aligned_parts.append(pd.DataFrame(file_rows))

    if not aligned_parts:
        return pd.DataFrame(
            columns=[
                "file",
                "sample_index",
                "voltage",
                "current",
                "delta_voltage",
                "delta_current",
                "wfs",
                "meas_time",
                "frame_number",
                "has_full_history",
                "y_true",
                "y_pred_osa",
                "y_pred_freerun",
            ]
        )

    aligned_df = pd.concat(aligned_parts, ignore_index=True)
    ordered_columns = [
        "file",
        "sample_index",
        "voltage",
        "current",
        "delta_voltage",
        "delta_current",
        "wfs",
        "meas_time",
        "frame_number",
        "has_full_history",
        "y_true",
        "y_pred_osa",
        "y_pred_freerun",
    ]
    present_columns = [col for col in ordered_columns if col in aligned_df.columns]
    return aligned_df[present_columns]


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_information_criteria(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_params: int,
) -> tuple[float, float]:
    """
    Compute Gaussian AIC and BIC from prediction residuals.

    The parameter count includes the ARX coefficients plus one variance term.
    This uses the full Gaussian deviance so criteria remain comparable even if
    different lag orders yield slightly different numbers of scored samples.
    """
    residual = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    n_obs = residual.size
    if n_obs == 0:
        return float("nan"), float("nan")

    rss = max(float(residual @ residual), np.finfo(float).tiny)
    sigma2 = rss / n_obs
    k = int(n_params) + 1
    neg2_log_likelihood = n_obs * (np.log(2.0 * np.pi * sigma2) + 1.0)
    aic = neg2_log_likelihood + 2.0 * k
    bic = neg2_log_likelihood + k * np.log(n_obs)
    return float(aic), float(bic)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_params: int | None = None,
) -> dict[str, float]:
    """Compute RMSE, MAE, R², fit percentage, and optionally AIC/BIC."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.size == 0 or y_pred.size == 0:
        metrics = {"RMSE": float("nan"), "MAE": float("nan"), "R2": float("nan"), "fit_pct": float("nan")}
        if n_params is not None:
            metrics["AIC"] = float("nan")
            metrics["BIC"] = float("nan")
        return metrics

    rmse    = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae     = float(mean_absolute_error(y_true, y_pred))
    r2      = float(r2_score(y_true, y_pred))

    denom   = float(np.linalg.norm(y_true - np.mean(y_true)))
    fit_pct = (
        100.0 * (1.0 - float(np.linalg.norm(y_true - y_pred)) / denom)
        if denom > 0 else float("nan")
    )
    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2, "fit_pct": fit_pct}
    if n_params is not None:
        aic, bic = compute_information_criteria(y_true, y_pred, n_params)
        metrics["AIC"] = aic
        metrics["BIC"] = bic
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH OVER MODEL ORDERS
# ══════════════════════════════════════════════════════════════════════════════

def predict_datasets_by_mode(
    datasets: list[tuple[str, pd.DataFrame]],
    theta: np.ndarray,
    na: int,
    nb: int,
    nk: int,
    selection_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dataset predictions for the requested model-selection mode."""
    if selection_mode == "one_step_ahead":
        return predict_one_step(datasets, theta, na, nb, nk)
    if selection_mode == "free_run":
        return predict_free_run(datasets, theta, na, nb, nk)
    raise ValueError(f"Unsupported selection mode: {selection_mode}")


def grid_search(
    train_set: list[tuple[str, pd.DataFrame]],
    validation_set: list[tuple[str, pd.DataFrame]],
    na_values: list[int],
    nb_values: list[int],
    nk_values: list[int],
    selection_mode: str,
    selection_metric: str,
) -> tuple[int, int, int, dict[str, float], list[dict[str, Any]]]:
    """
    Exhaustive grid search over (na, nb, nk) using the shared validation split.

    Returns
    -------
    best_na, best_nb, best_nk   : winning model orders
    best_val_metrics             : validation metrics for the winning model
    all_results                  : list of dicts with results for every combo tried
    """
    if selection_mode not in SELECTION_MODE_LABELS:
        raise ValueError(
            f"Unsupported selection mode '{selection_mode}'. "
            f"Choose from: {sorted(SELECTION_MODE_LABELS)}"
        )
    if selection_metric not in SELECTION_METRIC_LABELS:
        raise ValueError(
            f"Unsupported selection metric '{selection_metric}'. "
            f"Choose from: {sorted(SELECTION_METRIC_LABELS)}"
        )

    best_score                   = float("inf")
    best_na = best_nb = best_nk  = -1
    best_val_metrics: dict[str, float] = {}
    all_results: list[dict[str, Any]]  = []
    selection_label = SELECTION_MODE_LABELS[selection_mode]
    selection_metric_label = SELECTION_METRIC_LABELS[selection_metric]

    combos = list(product(na_values, nb_values, nk_values))
    logger.info(
        "Grid search: %d combinations  (na∈%s  nb∈%s  nk∈%s)  selection=%s  criterion=%s",
        len(combos), na_values, nb_values, nk_values, selection_label, selection_metric_label,
    )

    with ProgressBar(len(combos), "Grid search") as progress:
        for na, nb, nk in combos:
            status = f"na={na} nb={nb} nk={nk}"
            theta, feature_names, n_train_samples = fit_arx_from_datasets(train_set, na, nb, nk)
            n_params = len(arx_coefficient_names(na, nb, nk, feature_names))
            if n_train_samples <= n_params:
                progress.update(message=f"{status} | skipped (too little training data)")
                continue

            y_val_true, y_val_pred = predict_datasets_by_mode(
                validation_set, theta, na, nb, nk, selection_mode,
            )
            if len(y_val_true) == 0:
                progress.update(message=f"{status} | skipped (no {selection_label} validation samples)")
                continue

            vm = compute_metrics(y_val_true, y_val_pred, n_params=n_params)
            selection_score = vm[selection_metric_label]
            all_results.append(
                {
                    "na": na,
                    "nb": nb,
                    "nk": nk,
                    "split_role": "validation",
                    "selection_mode": selection_mode,
                    "selection_metric": selection_metric,
                    "selection_score": selection_score,
                    "n_params": n_params,
                    "n_samples": len(y_val_true),
                    **vm,
                }
            )

            if selection_score < best_score:
                best_score       = selection_score
                best_na, best_nb, best_nk = na, nb, nk
                best_val_metrics = vm
                status += f" | best {selection_metric_label}={best_score:.4f}"
            else:
                status += f" | {selection_metric_label}={selection_score:.4f}"

            progress.update(message=status)

    if best_na < 0:
        raise RuntimeError(
            "Grid search produced no valid ARX models. "
            "Check that the configured split has enough samples for the chosen lag ranges."
        )

    logger.info(
        "Best model by validation split (%s, %s): na=%d  nb=%d  nk=%d  |  %s=%.4f  RMSE=%.4f  R²=%.4f  fit%%=%.1f%%",
        selection_label,
        selection_metric_label,
        best_na, best_nb, best_nk,
        selection_metric_label,
        best_val_metrics[selection_metric_label],
        best_val_metrics["RMSE"], best_val_metrics["R2"], best_val_metrics["fit_pct"],
    )
    return best_na, best_nb, best_nk, best_val_metrics, all_results


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_outputs(
    output_dir: Path,
    na: int,
    nb: int,
    nk: int,
    theta: np.ndarray,
    feature_names: list[str],
    train_metrics:              dict[str, float],
    val_osa_metrics:            dict[str, float],
    val_freerun_metrics:        dict[str, float],
    test_osa_metrics:           dict[str, float] | None,
    test_freerun_metrics:       dict[str, float] | None,
    search_results:             list[dict[str, Any]],
    selection_mode:             str,
    selection_metric:           str,
    selection_metrics:          dict[str, float] | None = None,
    split_note:                 str | None = None,
    split_manifest:             pd.DataFrame | None = None,
    train_predictions:          pd.DataFrame | None = None,
    val_predictions:            pd.DataFrame | None = None,
    test_predictions:           pd.DataFrame | None = None,
) -> None:
    """Write coefficients, metrics, grid-search results, split manifest, and predictions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Named coefficient table
    coef_names = arx_coefficient_names(na, nb, nk, feature_names)

    pd.DataFrame({"name": coef_names, "value": theta}).to_csv(
        output_dir / "arx_coefficients.csv", index=False
    )

    # Metrics JSON — one-step-ahead and free-run side by side
    metrics_dict: dict[str, Any] = {
        "target_col": TARGET_COL,
        "model_orders": {"na": na, "nb": nb, "nk": nk},
        "model_inputs": {
            "feature_set": FEATURE_SET,
            "history_cols": RAW_INPUT_COLS if FEATURE_SET == "raw_electrical" else HISTORY_COLS,
            "context_cols": [] if FEATURE_SET == "raw_electrical" else CONTEXT_COLS,
            "sliding_window_size": nb,
            "rolling_summary_windows": (
                []
                if FEATURE_SET == "raw_electrical"
                else [specified_summary_window(nb)]
                if FEATURE_SET == "specified_inputs"
                else rolling_windows_for_nb(nb)
            ),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
        },
        "validation_scheme": "preprocessing_split_random_subsequence_by_video",
        "selection_mode": selection_mode,
        "selection_metric": selection_metric,
        "train_one_step_ahead":                  train_metrics,
        "validation_one_step_ahead":             val_osa_metrics,
        "validation_free_run":                   val_freerun_metrics,
    }
    if test_osa_metrics is not None:
        metrics_dict["test_one_step_ahead"] = test_osa_metrics
    if test_freerun_metrics is not None:
        metrics_dict["test_free_run"] = test_freerun_metrics
    if selection_metrics is not None:
        metrics_dict["validation_selection"] = selection_metrics
    if split_note is not None:
        metrics_dict["split_note"] = split_note
    with open(output_dir / "arx_metrics.json", "w") as fh:
        json.dump(metrics_dict, fh, indent=2)

    # Grid search results sorted by validation RMSE
    pd.DataFrame(search_results).sort_values("selection_score").to_csv(
        output_dir / "arx_grid_search.csv", index=False
    )

    if train_predictions is not None:
        train_predictions.to_csv(output_dir / "arx_train_predictions.csv", index=False)

    if val_predictions is not None:
        val_predictions.to_csv(output_dir / "arx_val_predictions.csv", index=False)

    if test_predictions is not None:
        test_predictions.to_csv(output_dir / "arx_test_predictions.csv", index=False)

    if split_manifest is not None:
        split_manifest.to_csv(output_dir / "arx_split_manifest.csv", index=False)

    logger.info("All outputs saved to: %s", output_dir)


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_order_values(raw: str) -> list[int]:
    """Parse a comma-separated list of non-negative lag orders."""
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Order values must be comma-separated integers."
        ) from exc

    if not values:
        raise argparse.ArgumentTypeError("At least one order value is required.")
    if any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("Order values must be non-negative.")

    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Identify a multi-input ARX model for a configurable weldpool-distance target.\n\n"
            "Examples:\n"
            "  python arx.py --data_folder /path/to/csvs\n"
            "  python arx.py --data_folder /path/to/csvs --na_values 0,1,5 --nb_values 0,1,5 --nk_max 2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data_folder", type=Path, default=DEFAULT_DATA_FOLDER,
        help="Folder containing CSV files (searched recursively).",
    )
    p.add_argument(
        "--target_col",
        choices=["tapering_to_weldpool_dist", "wiretip_to_weldpool_dist"],
        default=TARGET_COL,
        help=f"Prediction target column (default: {TARGET_COL}).",
    )
    p.add_argument(
        "--feature_set",
        choices=["live_engineered", "raw_electrical", "specified_inputs"],
        default=FEATURE_SET,
        help="Input feature family (default: live_engineered).",
    )
    p.add_argument(
        "--summary_window",
        type=int,
        default=None,
        help="Rolling statistics window for specified_inputs (default: use nb).",
    )
    p.add_argument("--machine_type", choices=["we", "fr"], default=None, help="Optional machine type filter passed to preprocessing split.")
    p.add_argument("--wfs", type=float, default=None, help="Optional wire-feed-speed filter passed to preprocessing split.")
    p.add_argument("--ctdw", type=float, default=None, help="Optional contact-tip-to-work-distance filter passed to preprocessing split.")
    p.add_argument(
        "--na_values",
        type=parse_order_values,
        default=DEFAULT_ORDER_VALUES,
        help="Comma-separated AR orders na to test (default: 0,1,3,5,10,25,50).",
    )
    p.add_argument(
        "--nb_values",
        type=parse_order_values,
        default=DEFAULT_ORDER_VALUES,
        help="Comma-separated input lag orders nb to test (default: 0,1,3,5,10,25,50).",
    )
    p.add_argument("--nk_min", type=int, default=0, help="Minimum input delay nk (default: 0).")
    p.add_argument("--nk_max", type=int, default=5, help="Maximum input delay nk (default: 5).")
    p.add_argument(
        "--selection_mode",
        choices=sorted(SELECTION_MODE_LABELS),
        default="free_run",
        help=(
            "Validation objective used during grid search: "
            "'free_run' optimizes recursive simulation, while "
            "'one_step_ahead' keeps the earlier teacher-forced selection."
        ),
    )
    p.add_argument(
        "--selection_metric",
        choices=sorted(SELECTION_METRIC_LABELS),
        default="rmse",
        help=(
            "Criterion minimized during grid search. "
            "'rmse' keeps the current behavior, while 'aic' and 'bic' "
            "apply Gaussian information criteria to the validation residuals."
        ),
    )
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for CSV and JSON outputs.")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run(args: argparse.Namespace) -> None:
    """Full ARX identification pipeline."""
    global FEATURE_SET, SUMMARY_WINDOW, TARGET_COL
    TARGET_COL = args.target_col
    FEATURE_SET = args.feature_set
    SUMMARY_WINDOW = args.summary_window

    # 1. Load the shared preprocessing split ───────────────────────────────────
    split = load_dataset_split(
        args.data_folder,
        machine_type=args.machine_type,
        wfs=args.wfs,
        ctdw=args.ctdw,
    )
    train_set = split_sequences_to_datasets(split.train)
    validation_set = split_sequences_to_datasets(split.validation)
    test_set = split_sequences_to_datasets(split.test)

    # 3. Grid search ───────────────────────────────────────────────────────────
    na_values = args.na_values
    nb_values = args.nb_values
    nk_values = list(range(args.nk_min, args.nk_max + 1))

    na, nb, nk, cv_metrics, search_results = grid_search(
        train_set, validation_set, na_values, nb_values, nk_values,
        args.selection_mode, args.selection_metric,
    )

    theta, feature_names, n_train_samples = fit_arx_from_datasets(train_set, na, nb, nk)
    n_params = len(arx_coefficient_names(na, nb, nk, feature_names))
    if n_train_samples <= n_params:
        raise RuntimeError(
            "The training split does not contain enough samples "
            "to fit the selected ARX model."
        )

    # 4. Evaluate on validation and test splits ────────────────────────────────
    y_train_true, y_train_pred     = predict_one_step(train_set, theta, na, nb, nk)
    y_val_osa_true, y_val_osa_pred = predict_one_step(validation_set, theta, na, nb, nk)
    y_val_fr_true, y_val_fr_pred   = predict_free_run(validation_set, theta, na, nb, nk)
    y_test_osa_true, y_test_osa_pred = predict_one_step(test_set, theta, na, nb, nk)
    y_test_fr_true, y_test_fr_pred = predict_free_run(test_set, theta, na, nb, nk)
    train_prediction_rows          = build_aligned_prediction_rows(train_set, theta, na, nb, nk)
    val_prediction_rows            = build_aligned_prediction_rows(validation_set, theta, na, nb, nk)
    test_prediction_rows           = build_aligned_prediction_rows(test_set, theta, na, nb, nk)

    train_metrics           = compute_metrics(y_train_true,           y_train_pred, n_params=n_params)
    val_osa_metrics         = compute_metrics(y_val_osa_true,         y_val_osa_pred, n_params=n_params)
    val_freerun_metrics     = compute_metrics(y_val_fr_true,          y_val_fr_pred, n_params=n_params)
    test_osa_metrics        = compute_metrics(y_test_osa_true,        y_test_osa_pred, n_params=n_params)
    test_freerun_metrics    = compute_metrics(y_test_fr_true,         y_test_fr_pred, n_params=n_params)

    logger.info(
        "Val free-run:  RMSE=%.4f  R²=%.4f  fit%%=%.1f%%",
        val_freerun_metrics["RMSE"], val_freerun_metrics["R2"],
        val_freerun_metrics["fit_pct"],
    )

    # 5. Save ──────────────────────────────────────────────────────────────────
    save_outputs(
        args.output_dir, na, nb, nk, theta,
        feature_names,
        train_metrics, val_osa_metrics, val_freerun_metrics,
        test_osa_metrics, test_freerun_metrics, search_results,
        args.selection_mode, args.selection_metric,
        cv_metrics, split.note, split.manifest,
        train_prediction_rows,
        val_prediction_rows,
        test_prediction_rows,
    )

    tag = f"ARX(na={na}, nb={nb}, nk={nk})"

    # 6. Console summary ───────────────────────────────────────────────────────
    print()
    print("═" * 64)
    print("  ARX MODEL IDENTIFICATION — SUMMARY")
    print("═" * 64)
    print(f"  Data folder   : {args.data_folder}")
    print(
        "  Split         : "
        f"{len(train_set)} train, {len(validation_set)} validation, {len(test_set)} test subsequence(s)"
    )
    print(f"  Split note    : {split.note}")
    print()
    print(f"  Best model    : {tag}")
    if FEATURE_SET == "specified_inputs":
        print(
            f"  Parameters    : {len(theta)}"
            f"  (na={na}  +  {len(feature_names)} specified inputs  =  {n_params})"
        )
    else:
        print(f"  Parameters    : {len(theta)}"
              f"  (na={na}  +  {len(feature_names)} live features × nb={nb}  =  {n_params})")
    print()
    print(
        "  ── Validation selection  (model selection: "
        f"{SELECTION_MODE_LABELS[args.selection_mode]}, "
        f"criterion: {SELECTION_METRIC_LABELS[args.selection_metric]}) ──"
    )
    for k, v in cv_metrics.items():
        print(f"    {k:<10}: {v:.4f}")
    print()
    print("  ── Training  (one-step-ahead) ───────────────")
    for k, v in train_metrics.items():
        print(f"    {k:<10}: {v:.4f}")
    print()
    print("  ── Validation  (one-step-ahead) ─────────────")
    for k, v in val_osa_metrics.items():
        print(f"    {k:<10}: {v:.4f}")
    print()
    print("  ── Validation  (free-run simulation) ────────")
    for k, v in val_freerun_metrics.items():
        print(f"    {k:<10}: {v:.4f}")
    print()
    print("  ── Test  (one-step-ahead) ───────────────────")
    for k, v in test_osa_metrics.items():
        print(f"    {k:<10}: {v:.4f}")
    print()
    print("  ── Test  (free-run simulation) ───────────────")
    for k, v in test_freerun_metrics.items():
        print(f"    {k:<10}: {v:.4f}")
    print()
    print(f"  Outputs       : {args.output_dir}")
    print("═" * 64)
    print()


def main() -> None:
    """Entry point."""
    run(parse_args())


if __name__ == "__main__":
    main()
