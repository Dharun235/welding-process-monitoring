"""Align process measurements with image features and export training tables."""

import argparse
import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import CONFIG
from scripts.preprocessing.interpolate_anomalies import interpolate_anomalies_df

PREPROCESSING_CONFIG = CONFIG["preprocessing"]
DEFAULT_INPUT_CSV_DIR = PREPROCESSING_CONFIG["input_path"]
DEFAULT_DETECTION_CSV_DIR = PREPROCESSING_CONFIG["detect_path"]
DEFAULT_ANOMALY_CSV_DIR = PREPROCESSING_CONFIG["anomaly_path"]
DEFAULT_OUTPUT_DIR = PREPROCESSING_CONFIG["output_path_org"]

CURRENT_SOURCE_COLUMNS = ["Math/IIR_filter_1_AI_4_Current_Filter_2/1", "Math/IIR_filter_1_AI_7_Current_Filter_2/1"]
CURRENT_TIME_COLUMNS = ["Math/IIR_filter_1_AI_4_Current_Filter_2/0", "Math/IIR_filter_1_AI_7_Current_Filter_2/0"]
VOLTAGE_SOURCE_COLUMNS = ["Math/IIR_filter_1_AI_1_Voltage_Filter_2/1"]
VOLTAGE_TIME_COLUMNS = ["Math/IIR_filter_1_AI_1_Voltage_Filter_2/0"]
FRAME_SOURCE_COLUMNS = ["Video/Video_Camera_0_2/1"]
VIDEO_TIME_COLUMNS = ["Video/Video_Camera_0_2/0"]
TRUE_VALUES = {"true", "1", "yes", "y", "t"}
MASKED_VALUE = np.nan

DEFAULT_MAX_ANOMALY_LENGTH = 10

# Continuous features are interpolated in time from the frame-rate detections
# onto the measurement-rate timeline. Boolean/event-style detections stay
# frame-discrete and are assigned by nearest frame instead of being linearly
# interpolated.
INTERPOLATED_DETECTION_COLUMNS = [
    "plasma_attachment_height",
    "plasma_channel_avg_width",
    "tapering_y",
    "weldpool_y",
    "wiretip_y",
    "tooltip_x1",
    "tooltip_x2",
    "tooltip_y1",
    "tooltip_y2",
    "mm_per_px",
]
FRAME_LEVEL_DETECTION_COLUMNS = [
    "arc_detected",
    "droplet_detected",
    "spatter_detected",
]
FRAME_LEVEL_VALUE_COLUMNS = [
    "droplet_diameter_px",
]
COLUMNS_TO_MASK = [
    "tapering_to_weldpool_dist",
    "wiretip_to_weldpool_dist",
    "tooltip_to_weldpool_dist",
    "plasma_to_weldpool_dist",
    "plasma_channel_avg_width",
    "droplet_diameter_px",
]
FINAL_OUTPUT_COLUMNS = [
    "meas_time",
    "frame_number",
    "current",
    "voltage",
    "wfs",
    "ctdw",
    "anomaly",
    "tapering_to_weldpool_dist",
    "wiretip_to_weldpool_dist",
    "tooltip_to_weldpool_dist",
    "plasma_to_weldpool_dist",
    "plasma_channel_avg_width",
    "arc_detected",
    "droplet_detected",
    "droplet_diameter_mm",
    "spatter_detected",
]
NON_NORMALIZED_OUTPUT_COLUMNS = {
    "meas_time",
    "frame_number",
    "anomaly",
    *FRAME_LEVEL_DETECTION_COLUMNS,
}
VALUE_COLUMNS_TO_NORMALIZE = [
    column_name
    for column_name in FINAL_OUTPUT_COLUMNS
    if column_name not in NON_NORMALIZED_OUTPUT_COLUMNS
]


def pick_column(df: pd.DataFrame, candidates: list[str], label: str, required: bool = True) -> str | None:
    for column_name in candidates:
        if column_name in df.columns:
            return column_name
    if required:
        raise KeyError(f"Could not find a column for {label}. Tried: {candidates}")
    return None


def normalize_stem(stem: str) -> str:
    normalized = stem.lower()
    normalized = re.sub(r"^detections_", "", normalized)
    normalized = re.sub(r"^anomaly_frames_", "", normalized)
    normalized = re.sub(r"^(backlit|laserlit)_", "", normalized)
    normalized = re.sub(r"_cam\d+$", "", normalized)
    normalized = re.sub(r"[_\-]+", "_", normalized)
    return normalized.strip("_")


def extract_experiment_settings(file_path: str | Path) -> tuple[int, int]:
    """
    Infer experiment-level settings from filenames like:
      fr_5m_20mm.csv -> (wfs=5, ctdw=20)
      detections_fr_5m_20mm_1_cam0.csv -> (wfs=5, ctdw=20)
    """
    normalized_stem = normalize_stem(Path(file_path).stem)
    match = re.search(r"\b[^_]+_(\d+)m_(\d+)mm(?:_\d+)?\b", normalized_stem)
    if match is None:
        raise ValueError(
            f"Could not infer WFS/CTDW from filename '{Path(file_path).name}'. "
            "Expected a stem like 'fr_5m_20mm' or 'fr_5m_20mm_1'."
        )

    wfs = int(match.group(1))
    ctdw = int(match.group(2))
    return wfs, ctdw


def parse_bool_like(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in TRUE_VALUES


def interpolate_series_to_times(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """
    Linearly interpolate one numeric frame-level feature onto target_times.

    Choices:
    - Within the observed frame-time range: linear interpolation.
    - Before the first frame or after the last frame: edge hold.
    - If fewer than 2 valid source points exist: constant fill or NaN.
    """
    valid_mask = np.isfinite(source_times) & np.isfinite(source_values)
    if not np.any(valid_mask):
        return np.full(len(target_times), np.nan, dtype=float)

    valid_times = source_times[valid_mask]
    valid_values = source_values[valid_mask]

    # Guard against any repeated timestamps before calling np.interp.
    valid_times, unique_idx = np.unique(valid_times, return_index=True)
    valid_values = valid_values[unique_idx]

    if len(valid_times) == 1:
        return np.full(len(target_times), valid_values[0], dtype=float)

    return np.interp(
        target_times,
        valid_times,
        valid_values,
        left=valid_values[0],
        right=valid_values[-1],
    )


def build_detection_frame_data(
    frame_ts: pd.DataFrame,
    detection_csv_path: str | Path,
    anomaly_csv_path: str | Path | None = None,
    interpolation_columns: list[str] | None = None,
    max_anomaly_length: int = 0,
) -> pd.DataFrame:
    """Load one detection CSV and align it to the full frame timeline."""
    detect_df = pd.read_csv(detection_csv_path)

    if "frame_number" not in detect_df.columns:
        raise KeyError(f"Detection CSV is missing 'frame_number': {detection_csv_path}")

    detect_df["frame_number"] = pd.to_numeric(detect_df["frame_number"], errors="coerce")
    detect_df = detect_df.dropna(subset=["frame_number"]).copy()
    detect_df["frame_number"] = detect_df["frame_number"].astype(int)

    if "arc_detected" not in detect_df.columns:
        detect_df["arc_detected"] = False
    if "plasma_attachment_height" not in detect_df.columns:
        detect_df["plasma_attachment_height"] = np.nan
    if "plasma_channel_avg_width" not in detect_df.columns:
        detect_df["plasma_channel_avg_width"] = np.nan

    droplet_cols = sorted([
        c for c in detect_df.columns
        if c.lower().startswith("droplet_") and "diameter" in c.lower()
    ])
    spatter_cols = sorted([
        c for c in detect_df.columns
        if c.lower().startswith("spatter_") and "diameter" in c.lower()
    ])

    droplet_detected_list, droplet_diameter_list, spatter_detected_list = [], [], []
    for _, row in detect_df.iterrows():
        droplet_values = [row[c] for c in droplet_cols if pd.notna(row[c])]
        droplet_detected = len(droplet_values) > 0
        droplet_detected_list.append(droplet_detected)
        droplet_diameter_list.append(max(droplet_values) if droplet_detected else np.nan)

        spatter_values = [row[c] for c in spatter_cols if pd.notna(row[c])]
        spatter_detected_list.append(len(spatter_values) > 0)

    detect_df["droplet_detected"] = droplet_detected_list
    detect_df["droplet_diameter_px"] = droplet_diameter_list
    detect_df["spatter_detected"] = spatter_detected_list

    selected_columns = [
        "frame_number",
        *FRAME_LEVEL_DETECTION_COLUMNS,
        *FRAME_LEVEL_VALUE_COLUMNS,
        *INTERPOLATED_DETECTION_COLUMNS,
    ]
    for column_name in selected_columns:
        if column_name not in detect_df.columns:
            detect_df[column_name] = np.nan
    detect_df = detect_df[selected_columns].drop_duplicates(subset="frame_number", keep="first")

    frame_detection_df = (
        frame_ts[["frame_number", "frame_time"]]
        .merge(detect_df, on="frame_number", how="left")
        .sort_values("frame_time")
        .reset_index(drop=True)
    )

    if anomaly_csv_path is not None:
        anomaly_df = load_anomaly_flags(anomaly_csv_path)
        frame_detection_df = frame_detection_df.merge(anomaly_df, on="frame_number", how="left")
        frame_detection_df["anomaly"] = frame_detection_df["anomaly"].fillna(0).astype(int)
    else:
        frame_detection_df["anomaly"] = 0

    if interpolation_columns is not None and max_anomaly_length > 0:
        frame_detection_df = interpolate_anomalies_df(
            frame_detection_df,
            interpolation_columns=interpolation_columns,
            max_anomaly_length=max_anomaly_length,
        )

    for column_name in FRAME_LEVEL_DETECTION_COLUMNS:
        frame_detection_df[column_name] = frame_detection_df[column_name].apply(parse_bool_like)

    frame_detection_df["anomaly"] = pd.to_numeric(
        frame_detection_df["anomaly"],
        errors="coerce",
    ).fillna(0).astype(int)

    return frame_detection_df


def build_detection_features_by_time(
    frame_detection_df: pd.DataFrame,
    meas_times: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return:
    - numeric detection features interpolated onto measurement timestamps
    - frame-level detection features to merge by frame_number
    """
    interpolated_columns: dict[str, np.ndarray] = {}
    frame_times = pd.to_numeric(frame_detection_df["frame_time"], errors="coerce").to_numpy(dtype=float)
    for column_name in INTERPOLATED_DETECTION_COLUMNS:
        source_values = pd.to_numeric(frame_detection_df[column_name], errors="coerce").to_numpy(dtype=float)
        interpolated_columns[column_name] = interpolate_series_to_times(frame_times, source_values, meas_times)

    interpolated_df = pd.DataFrame(interpolated_columns)

    frame_level_df = frame_detection_df[
        ["frame_number", "anomaly", *FRAME_LEVEL_DETECTION_COLUMNS, *FRAME_LEVEL_VALUE_COLUMNS]
    ].copy()

    return interpolated_df, frame_level_df


def score_detection_match(input_stem: str, detection_stem: str) -> tuple[int, int, int]:
    normalized_input = normalize_stem(input_stem)
    normalized_detection = normalize_stem(detection_stem)

    if normalized_input == normalized_detection:
        return (3, len(normalized_input), len(normalized_detection))
    if normalized_input in normalized_detection or normalized_detection in normalized_input:
        input_tokens = set(normalized_input.split("_"))
        detection_tokens = set(normalized_detection.split("_"))
        overlap = len(input_tokens & detection_tokens)
        return (2, overlap, -abs(len(normalized_input) - len(normalized_detection)))

    input_tokens = set(normalized_input.split("_"))
    detection_tokens = set(normalized_detection.split("_"))
    overlap = len(input_tokens & detection_tokens)
    return (1 if overlap else 0, overlap, -abs(len(input_tokens) - len(detection_tokens)))


def find_matching_detection_csv(input_csv_path: str | Path, detection_csv_dir: str | Path, required: bool = True) -> Path | None:
    detection_dir = Path(detection_csv_dir)
    if not detection_dir.exists():
        if required:
            raise FileNotFoundError(f"Detection CSV folder not found: {detection_dir}")
        return None

    candidates = sorted(path for path in detection_dir.glob("*.csv") if path.is_file())
    if not candidates:
        if required:
            raise FileNotFoundError(f"No detection CSV files found in {detection_dir}")
        return None

    input_stem = Path(input_csv_path).stem
    ranked = sorted(
        ((score_detection_match(input_stem, c.stem), c) for c in candidates),
        reverse=True,
    )

    best_score, best_candidate = ranked[0]
    if best_score[0] == 0:
        if required:
            raise FileNotFoundError(f"No detection CSV matched source file: {Path(input_csv_path).name}")
        return None
    return best_candidate


def load_anomaly_flags(anomaly_csv_path: str | Path) -> pd.DataFrame:
    """
    Load anomaly CSV and return a DataFrame with (frame_number, anomaly).
    anomaly column: 0 = normal, 1 = anomaly.
    """
    df = pd.read_csv(anomaly_csv_path)
    if "frame_number" not in df.columns:
        raise KeyError(f"Anomaly CSV missing 'frame_number': {anomaly_csv_path}")
    if "anomaly" not in df.columns:
        raise KeyError(f"Anomaly CSV missing 'anomaly': {anomaly_csv_path}")
    df["frame_number"] = pd.to_numeric(df["frame_number"], errors="coerce")
    df = df.dropna(subset=["frame_number"]).copy()
    df["frame_number"] = df["frame_number"].astype(int)
    df["anomaly"] = pd.to_numeric(df["anomaly"], errors="coerce").fillna(0).astype(int)
    return df[["frame_number", "anomaly"]].drop_duplicates("frame_number")


def extract_frame_timestamps(df: pd.DataFrame, frame_col: str, time_col: str) -> pd.DataFrame:
    """
    Return DataFrame with (frame_number, frame_time) for rows where frame_col
    is an integer value. Frame numbers are normalized to start at 1.

    Each H5-derived CSV stores the video channel as a separate time series:
    - time_col (/0): timestamp of each frame capture event
    - frame_col (/1): frame counter (integer) at that event

    Rows where frame_col is NaN belong to other channels and are ignored.
    """
    frame_vals = pd.to_numeric(df[frame_col], errors="coerce")
    time_vals = pd.to_numeric(df[time_col], errors="coerce")

    # Keep only rows where the frame counter is a whole number
    is_frame_event = frame_vals.notna() & (frame_vals % 1 == 0) & time_vals.notna()

    frame_ts = pd.DataFrame({
        "frame_number": frame_vals[is_frame_event].astype(int),
        "frame_time": time_vals[is_frame_event],
    })
    frame_ts = (
        frame_ts
        .drop_duplicates("frame_number")
        .sort_values("frame_number")
        .reset_index(drop=True)
    )

    if frame_ts.empty:
        raise ValueError(f"No integer frame events found in column '{frame_col}'.")

    min_frame = int(frame_ts["frame_number"].min())
    frame_ts["frame_number"] = (frame_ts["frame_number"] - min_frame + 1).astype(int)
    return frame_ts


def compute_frame_boundaries(frame_ts: pd.DataFrame) -> pd.DataFrame:
    """
    For each frame, compute a half-open time window [lower_bound, upper_bound).

    Boundaries are midpoints between consecutive frame timestamps:
      boundary[i] = (frame_time[i] + frame_time[i+1]) / 2

    First frame: lower = -inf.  Last frame: upper = +inf.
    """
    times = frame_ts["frame_time"].values
    frame_numbers = frame_ts["frame_number"].values
    n = len(times)

    midpoints = (times[:-1] + times[1:]) / 2.0

    lower_bounds = np.empty(n)
    upper_bounds = np.empty(n)
    lower_bounds[0] = -np.inf
    lower_bounds[1:] = midpoints
    upper_bounds[-1] = np.inf
    upper_bounds[:-1] = midpoints

    return pd.DataFrame({
        "frame_number": frame_numbers,
        "frame_time": times,
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
    })


def assign_frames_by_time(meas_times: np.ndarray, frame_boundaries: pd.DataFrame) -> np.ndarray:
    """
    For each measurement timestamp find the frame whose window contains it.

    Window for frame x: [lower_bound[x], upper_bound[x])
    Returns -1 for timestamps that fall outside all windows.
    """
    lower = frame_boundaries["lower_bound"].values  # sorted ascending
    upper = frame_boundaries["upper_bound"].values
    frame_nums = frame_boundaries["frame_number"].values

    # Binary search: find insertion point into sorted lower_bounds
    idx = np.searchsorted(lower, meas_times, side="right") - 1
    idx = np.clip(idx, 0, len(frame_nums) - 1)

    in_window = (meas_times >= lower[idx]) & (meas_times < upper[idx])
    return np.where(in_window, frame_nums[idx], -1)


def extract_measurement_rows(
    df: pd.DataFrame,
    voltage_time_col: str,
    voltage_col: str,
    current_time_col: str,
    current_col: str,
) -> pd.DataFrame:
    """Extract one measurement row per DAQ sample from the raw input CSV."""
    voltage_time = pd.to_numeric(df[voltage_time_col], errors="coerce")
    voltage_vals = pd.to_numeric(df[voltage_col], errors="coerce")
    current_time = pd.to_numeric(df[current_time_col], errors="coerce")
    current_vals = pd.to_numeric(df[current_col], errors="coerce")

    meas_mask = voltage_time.notna() | current_time.notna()
    meas_time = voltage_time.where(voltage_time.notna(), current_time)

    return pd.DataFrame({
        "meas_time": meas_time[meas_mask],
        "voltage": voltage_vals[meas_mask],
        "current": current_vals[meas_mask],
    }).reset_index(drop=True)


def assign_measurements_to_frames(measurement_df: pd.DataFrame, frame_boundaries: pd.DataFrame) -> pd.DataFrame:
    """Attach each measurement sample to the frame whose time window contains it."""
    assigned_df = measurement_df.copy()
    assigned_df["frame_number"] = assign_frames_by_time(
        pd.to_numeric(assigned_df["meas_time"], errors="coerce").to_numpy(dtype=float),
        frame_boundaries,
    )
    assigned_df = assigned_df[assigned_df["frame_number"] != -1].reset_index(drop=True)
    return assigned_df


def aggregate_measurements_by_frame(measurement_df: pd.DataFrame) -> pd.DataFrame:
    """Average measurement samples that belong to the same frame."""
    if measurement_df.empty:
        return pd.DataFrame(columns=["frame_number", "meas_time", "voltage", "current"])

    return (
        measurement_df
        .groupby("frame_number", as_index=False)
        .agg({
            "meas_time": "mean",
            "voltage": "mean",
            "current": "mean",
        })
    )


def finalize_output_df(out_df: pd.DataFrame, output_csv_path: str | Path) -> pd.DataFrame:
    """Compute derived features, mask unresolved anomalies, convert to mm, and save."""
    for column_name in FRAME_LEVEL_DETECTION_COLUMNS:
        out_df[column_name] = out_df[column_name].fillna(False).astype(bool)

    out_df["anomaly"] = pd.to_numeric(out_df["anomaly"], errors="coerce").fillna(0).astype(int)
    out_df["tooltip_mid_y"] = (out_df["tooltip_y1"] + out_df["tooltip_y2"]) / 2.0
    out_df["tapering_to_weldpool_dist"] = np.abs(out_df["tapering_y"] - out_df["weldpool_y"])
    out_df["wiretip_to_weldpool_dist"] = np.abs(out_df["wiretip_y"] - out_df["weldpool_y"])
    out_df["tooltip_to_weldpool_dist"] = np.abs(out_df["tooltip_mid_y"] - out_df["weldpool_y"])
    out_df["plasma_to_weldpool_dist"] = np.where(
        out_df["arc_detected"] & out_df["plasma_attachment_height"].notna() & out_df["weldpool_y"].notna(),
        np.abs(out_df["plasma_attachment_height"] - out_df["weldpool_y"]),
        np.nan,
    )
    out_df["plasma_channel_avg_width"] = np.where(
        out_df["arc_detected"],
        out_df["plasma_channel_avg_width"],
        np.nan,
    )

    anomaly_mask = out_df["anomaly"] == 1
    for col in COLUMNS_TO_MASK:
        if col in out_df.columns:
            out_df.loc[anomaly_mask, col] = MASKED_VALUE

    scale = out_df["mm_per_px"]
    for col in [
        "droplet_diameter_px",
        "plasma_channel_avg_width",
        "plasma_to_weldpool_dist",
        "tapering_to_weldpool_dist",
        "wiretip_to_weldpool_dist",
        "tooltip_to_weldpool_dist",
    ]:
        out_df[col] = out_df[col] * scale
    out_df = out_df.rename(columns={"droplet_diameter_px": "droplet_diameter_mm"})

    out_df = out_df[FINAL_OUTPUT_COLUMNS]
    write_output_csv(out_df, output_csv_path)
    return out_df


def write_output_csv(out_df: pd.DataFrame, output_csv_path: str | Path) -> Path:
    """Write one output CSV and create parent directories as needed."""
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv_path, index=False)
    return output_csv_path


def transform_value_columns(
    out_df: pd.DataFrame,
    transform: Callable[[pd.Series, str], pd.Series],
) -> pd.DataFrame:
    """Apply a column-wise transform to the numeric value columns only."""
    transformed_df = out_df.copy()
    for column_name in VALUE_COLUMNS_TO_NORMALIZE:
        if column_name not in transformed_df.columns:
            continue
        values = pd.to_numeric(transformed_df[column_name], errors="coerce")
        transformed_df[column_name] = transform(values, column_name)
    return transformed_df


def clip_output_df(out_df: pd.DataFrame) -> pd.DataFrame:
    """Clip negative value columns to zero while preserving NaNs."""
    return transform_value_columns(
        out_df,
        lambda values, _: values.clip(lower=0),
    )


def initialize_column_maxima() -> dict[str, float]:
    return {column_name: np.nan for column_name in VALUE_COLUMNS_TO_NORMALIZE}


def update_column_maxima(column_maxima: dict[str, float], out_df: pd.DataFrame) -> None:
    """Update dataset-wide maxima using the clipped view of one output DataFrame."""
    clipped_df = clip_output_df(out_df)
    for column_name in VALUE_COLUMNS_TO_NORMALIZE:
        if column_name not in clipped_df.columns:
            continue
        column_max = pd.to_numeric(clipped_df[column_name], errors="coerce").max(skipna=True)
        if pd.isna(column_max):
            continue
        if pd.isna(column_maxima[column_name]) or float(column_max) > float(column_maxima[column_name]):
            column_maxima[column_name] = float(column_max)


def normalize_output_df(
    out_df: pd.DataFrame,
    column_maxima: dict[str, float],
) -> pd.DataFrame:
    """Clip value columns to zero and scale them to [0, 1] using global maxima."""

    def normalize_series(values: pd.Series, column_name: str) -> pd.Series:
        clipped_values = values.clip(lower=0)
        column_max = column_maxima.get(column_name, np.nan)
        if pd.isna(column_max):
            return clipped_values
        if column_max <= 0:
            return clipped_values.where(clipped_values.isna(), 0.0)
        return (clipped_values / column_max).clip(lower=0, upper=1)

    return transform_value_columns(out_df, normalize_series)


def write_output_copy(
    out_df: pd.DataFrame,
    output_csv_path: str | Path,
    output_dir: str | Path | None,
    *,
    transform: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
) -> Path | None:
    """Write a transformed copy of an output CSV to another directory."""
    if output_dir is None:
        return None

    copied_df = out_df if transform is None else transform(out_df)
    copied_output_path = Path(output_dir) / Path(output_csv_path).name
    return write_output_csv(copied_df, copied_output_path)


def write_normalized_output_copies(
    source_csv_paths: list[Path],
    output_csv_dir_clipped_norm: str | Path | None,
    column_maxima: dict[str, float],
) -> list[Path]:
    """Write clipped-and-normalized copies using already-computed global maxima."""
    if output_csv_dir_clipped_norm is None:
        return []

    normalized_paths: list[Path] = []
    for source_csv_path in source_csv_paths:
        normalized_output_path = write_output_copy(
            out_df=pd.read_csv(source_csv_path),
            output_csv_path=source_csv_path,
            output_dir=output_csv_dir_clipped_norm,
            transform=lambda df: normalize_output_df(df, column_maxima),
        )
        if normalized_output_path is not None:
            normalized_paths.append(normalized_output_path)
    return normalized_paths


def build_sampling_csv(
    input_csv_path: str,
    detection_csv_path: str,
    output_csv_path: str | Path,
    output_csv_dir_clipped: str | Path | None = None,
    anomaly_csv_path: str | Path | None = None,
    upsample: bool = True,
    interpolation_columns: list[str] | None = None,
    max_anomaly_length: int = 0,
) -> pd.DataFrame:

    output_csv_path = Path(output_csv_path)
    df = pd.read_csv(input_csv_path)
    wfs, ctdw = extract_experiment_settings(input_csv_path)

    # --- Resolve column names ---
    video_time_col = pick_column(df, VIDEO_TIME_COLUMNS, "video time")
    frame_col = pick_column(df, FRAME_SOURCE_COLUMNS, "frame")
    voltage_time_col = pick_column(df, VOLTAGE_TIME_COLUMNS, "voltage time")
    voltage_col = pick_column(df, VOLTAGE_SOURCE_COLUMNS, "voltage")
    current_time_col = pick_column(df, CURRENT_TIME_COLUMNS, "current time")
    current_col = pick_column(df, CURRENT_SOURCE_COLUMNS, "current")

    # --- Build frame time windows from the video channel ---
    frame_ts = extract_frame_timestamps(df, frame_col, video_time_col)
    frame_boundaries = compute_frame_boundaries(frame_ts)

    measurement_df = extract_measurement_rows(
        df,
        voltage_time_col=voltage_time_col,
        voltage_col=voltage_col,
        current_time_col=current_time_col,
        current_col=current_col,
    )
    measurement_df = assign_measurements_to_frames(measurement_df, frame_boundaries)

    frame_detection_df = build_detection_frame_data(
        frame_ts=frame_ts,
        detection_csv_path=detection_csv_path,
        anomaly_csv_path=anomaly_csv_path,
        interpolation_columns=interpolation_columns,
        max_anomaly_length=max_anomaly_length,
    )

    if upsample:
        out_df = measurement_df.copy()
        out_df["wfs"] = wfs
        out_df["ctdw"] = ctdw

        meas_times = pd.to_numeric(out_df["meas_time"], errors="coerce").to_numpy(dtype=float)
        interpolated_detect_df, frame_level_detect_df = build_detection_features_by_time(
            frame_detection_df=frame_detection_df,
            meas_times=meas_times,
        )
        out_df = pd.concat([out_df.reset_index(drop=True), interpolated_detect_df.reset_index(drop=True)], axis=1)
        out_df = pd.merge(out_df, frame_level_detect_df, on="frame_number", how="left")
    else:
        measurement_agg_df = aggregate_measurements_by_frame(measurement_df)
        out_df = frame_detection_df.merge(measurement_agg_df, on="frame_number", how="left")
        out_df["meas_time"] = out_df["meas_time"].where(out_df["meas_time"].notna(), out_df["frame_time"])
        out_df["wfs"] = wfs
        out_df["ctdw"] = ctdw

    out_df = finalize_output_df(out_df, output_csv_path)
    clipped_output_csv_path = write_output_copy(
        out_df=out_df,
        output_csv_path=output_csv_path,
        output_dir=output_csv_dir_clipped,
        transform=clip_output_df,
    )

    print("Saved:", output_csv_path)
    if clipped_output_csv_path is not None:
        print("Saved clipped:", clipped_output_csv_path)
    print("Source rows:", len(df))
    if upsample:
        print("Measurement rows assigned to frames:", len(measurement_df))
    else:
        print("Frames after measurement downsampling:", len(out_df))
    print("Unique frames covered:", out_df["frame_number"].nunique())

    return out_df


def build_sampling_csvs(input_csv_dir: str,
                        detection_csv_dir: str,
                        anomaly_csv_dir: str,
                        output_csv_dir: str,
                        output_csv_dir_clipped: str | Path | None = None,
                        output_csv_dir_clipped_norm: str | Path | None = None,
                        *,
                        upsample: bool,
                        interpolation_columns: list[str],
                        max_anomaly_length: int,
                        ) -> list[Path]:

    input_dir = Path(input_csv_dir)
    output_dir = Path(output_csv_dir)
    output_dir_clipped = Path(output_csv_dir_clipped) if output_csv_dir_clipped is not None else None
    output_dir_clipped_norm = (
        Path(output_csv_dir_clipped_norm)
        if output_csv_dir_clipped_norm is not None
        else None
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input CSV folder not found: {input_dir}")

    input_csv_files = sorted(p for p in input_dir.glob("*.csv") if p.is_file())
    if not input_csv_files:
        print(f"No input CSV files found in {input_dir}")
        return []

    written_files: list[Path] = []
    normalized_source_paths: list[Path] = []
    column_maxima = initialize_column_maxima()
    skipped_files: list[str] = []

    for input_csv_path in input_csv_files:
        try:
            detection_csv_path = find_matching_detection_csv(input_csv_path, detection_csv_dir)
            anomaly_csv_path = find_matching_detection_csv(input_csv_path, anomaly_csv_dir, required=False)
            output_csv_path = output_dir / f"{input_csv_path.stem}.csv"
            out_df = build_sampling_csv(
                input_csv_path=str(input_csv_path),
                detection_csv_path=str(detection_csv_path),
                output_csv_path=output_csv_path,
                output_csv_dir_clipped=output_dir_clipped,
                anomaly_csv_path=anomaly_csv_path,
                upsample=upsample,
                interpolation_columns=interpolation_columns,
                max_anomaly_length=max_anomaly_length,
            )
            written_files.append(output_csv_path)
            if output_dir_clipped_norm is not None:
                update_column_maxima(column_maxima, out_df)
                normalized_source_paths.append(
                    ((output_dir_clipped if output_dir_clipped is not None else output_dir) / output_csv_path.name)
                )
        except Exception as exc:
            skipped_files.append(f"{input_csv_path.name}: {exc}")
            print(f"Skipping {input_csv_path.name}: {exc}")

    normalized_files = write_normalized_output_copies(
        source_csv_paths=normalized_source_paths,
        output_csv_dir_clipped_norm=output_dir_clipped_norm,
        column_maxima=column_maxima,
    )

    print(f"Processed {len(written_files)} file(s).")
    if output_dir_clipped_norm is not None:
        print(f"Saved normalized {len(normalized_files)} file(s).")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} file(s).")
    return written_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sensor CSV with detection CSV using time-based frame assignment.")
    parser.add_argument("--input-csv-dir", default=DEFAULT_INPUT_CSV_DIR)
    parser.add_argument("--detection-csv-dir", default=DEFAULT_DETECTION_CSV_DIR)
    parser.add_argument("--anomaly-csv-dir", default=DEFAULT_ANOMALY_CSV_DIR)
    parser.add_argument("--output-csv-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv-dir-clipped", default=None)
    parser.add_argument("--output-csv-dir-clipped-norm", default=None)
    parser.add_argument("--upsample", default=True, type=parse_bool_like)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_sampling_csvs(input_csv_dir=args.input_csv_dir,
                        detection_csv_dir=args.detection_csv_dir,
                        anomaly_csv_dir=args.anomaly_csv_dir,
                        output_csv_dir=args.output_csv_dir,
                        output_csv_dir_clipped=args.output_csv_dir_clipped,
                        output_csv_dir_clipped_norm=args.output_csv_dir_clipped_norm,
                        upsample=args.upsample,
                        interpolation_columns=INTERPOLATED_DETECTION_COLUMNS,
                        max_anomaly_length=DEFAULT_MAX_ANOMALY_LENGTH)
