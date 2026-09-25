"""Rule-based anomaly scoring.

This module replaces the earlier model-based approach with deterministic checks
that are easier to tune and explain from generated detection CSV data.
"""

from pathlib import Path
import shutil
import re

import pandas as pd
import numpy as np

from scripts.config import CONFIG, get_current_io_paths, get_current_runtime_settings


def _clear_directory_contents(folder_path: Path):
    """Remove all files/subfolders from a directory if it exists."""
    if not folder_path.exists():
        return
    for child in folder_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _prepare_anomaly_output(anomaly_output_dir: Path, clear_existing: bool = True):
    """Create anomaly output folders and optionally clear previous run artifacts."""
    anomaly_output_dir.mkdir(parents=True, exist_ok=True)
    anomaly_csv_dir = anomaly_output_dir / "csv"
    anomaly_csv_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        for old_csv in anomaly_csv_dir.glob("*.csv"):
            old_csv.unlink()
        # Clean legacy flat CSVs if present.
        for old_csv in anomaly_output_dir.glob("*.csv"):
            old_csv.unlink()
    anomaly_images_dir = anomaly_output_dir / "images"
    anomaly_images_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        _clear_directory_contents(anomaly_images_dir)
    return anomaly_images_dir, anomaly_csv_dir


def _rolling_jump_mask(
    series: pd.Series,
    window: int,
    mean_pct_threshold: float,
    min_periods: int = 5,
) -> pd.Series:
    """
    Flag points whose value deviates from previous-window mean by a percentage.
    """
    s = pd.to_numeric(series, errors="coerce")
    baseline_mean = s.shift(1).rolling(window=window, min_periods=min_periods).mean()
    pct_diff = (s - baseline_mean).abs() / (baseline_mean.abs() + 1e-9)
    return (pct_diff > mean_pct_threshold).fillna(False)


def _isolated_short_lived_events(presence: pd.Series, window: int, max_len: int) -> pd.Series:
    """
    Flag short-lived appearance runs (appears from nowhere, vanishes quickly).
    """
    p = presence.fillna(False).astype(bool).to_numpy()
    n = len(p)
    out = np.zeros(n, dtype=bool)

    i = 0
    while i < n:
        if not p[i]:
            i += 1
            continue

        start = i
        while i < n and p[i]:
            i += 1
        end = i - 1
        run_len = end - start + 1

        if run_len <= max_len:
            left_any = p[max(0, start - window):start].any()
            right_any = p[end + 1:min(n, end + 1 + window)].any()
            if not left_any and not right_any:
                out[start:end + 1] = True

    return pd.Series(out, index=presence.index)


def _make_presence_features(df_num: pd.DataFrame):
    """
    Build per-object presence matrices and aggregate count/area features for droplet/spatter.
    """
    object_measure_cols = {}
    for col in df_num.columns:
        m = re.match(r"^(spatter|droplet)_([0-9]+)_(area|diameter_px)$", col)
        if not m:
            continue
        kind = m.group(1)
        object_id = m.group(2)
        measure = m.group(3)
        key = f"{kind}_{object_id}"
        if key not in object_measure_cols:
            object_measure_cols[key] = col
        elif measure == "area":
            # Prefer area when both are present.
            object_measure_cols[key] = col

    spatter_objs = sorted([k for k in object_measure_cols if k.startswith("spatter_")])
    droplet_objs = sorted([k for k in object_measure_cols if k.startswith("droplet_")])

    spatter_presence_data = {
        obj: (pd.to_numeric(df_num[object_measure_cols[obj]], errors="coerce").fillna(0) > 0)
        for obj in spatter_objs
    }
    droplet_presence_data = {
        obj: (pd.to_numeric(df_num[object_measure_cols[obj]], errors="coerce").fillna(0) > 0)
        for obj in droplet_objs
    }

    spatter_presence = pd.DataFrame(spatter_presence_data, index=df_num.index)
    droplet_presence = pd.DataFrame(droplet_presence_data, index=df_num.index)

    agg = pd.DataFrame(index=df_num.index)

    if len(spatter_presence.columns) > 0:
        agg["active_spatter_count"] = spatter_presence.sum(axis=1)
        spatter_area_cols = [c for c in df_num.columns if re.match(r"^spatter_[0-9]+_area$", c)]
        if spatter_area_cols:
            agg["active_spatter_total_area"] = df_num[spatter_area_cols].fillna(0).sum(axis=1)

    if len(droplet_presence.columns) > 0:
        agg["active_droplet_count"] = droplet_presence.sum(axis=1)
        droplet_area_cols = [c for c in df_num.columns if re.match(r"^droplet_[0-9]+_area$", c)]
        if droplet_area_cols:
            agg["active_droplet_total_area"] = df_num[droplet_area_cols].fillna(0).sum(axis=1)

    return spatter_presence, droplet_presence, agg


def run_rule_based_anomaly(
    detection_csv_path: Path,
    anomaly_output_dir: Path,
    detection_images_dir: Path = None,
    source_images_dir: Path = None,
    image_lookup: dict = None,
    output_csv_name: str = "anomaly_frames.csv",
    clear_existing: bool = True,
    verbose: bool = True,
    contamination: float = 0.03,
    random_state: int = 42,
):
    """Run rule-based anomaly scoring on one detection CSV.

    The function computes per-frame scores and reasons, writes
    `anomaly_frames_<source>.csv`, and optionally copies anomalous images from
    detection debug outputs.
    """
    detection_csv_path = Path(detection_csv_path)
    detection_images_dir = Path(detection_images_dir) if detection_images_dir is not None else None
    source_images_dir = Path(source_images_dir) if source_images_dir is not None else None
    anomaly_output_dir = Path(anomaly_output_dir)
    image_lookup = image_lookup or {}

    if not detection_csv_path.exists():
        raise FileNotFoundError(f"Detection CSV not found: {detection_csv_path}")

    anomaly_images_dir, anomaly_csv_dir = _prepare_anomaly_output(anomaly_output_dir, clear_existing=clear_existing)

    df = pd.read_csv(detection_csv_path)
    if "frame_number" in df.columns:
        df = df.sort_values("frame_number")

    if "image_name" not in df.columns:
        raise ValueError("Detection CSV must contain 'image_name' column.")

    # Keep compatibility with old signature (unused in rule-based approach).
    _ = contamination
    _ = random_state

    # Core stable geometry columns expected in most runs.
    critical_cols = [
        "mm_per_px",
        "weldpool_y",
    ]

    # Event-like columns: treat like spatter/droplet and only flag short-lived,
    # isolated "appears and disappears" behavior.
    transient_event_cols = [
        "arc_detected",
        "plasma_attachment_height",
        "plasma_channel_avg_width",
        "transition_x",
        "transition_y",
    ]

    anomaly_cfg = CONFIG.get("anomaly_detection", {}) or {}
    window_size = int(anomaly_cfg.get("window_size", 10))
    short_life_max_len = int(anomaly_cfg.get("short_life_max_len", 2))
    anomaly_min_score = float(anomaly_cfg.get("min_anomaly_score", 2.0))
    jump_mean_pct_threshold = float(anomaly_cfg.get("jump_mean_pct_threshold", 0.5))
    jump_weight = float(anomaly_cfg.get("jump_weight", 1.0))
    missing_weight = float(anomaly_cfg.get("missing_weight", 1.0))
    transient_weight = float(anomaly_cfg.get("transient_weight", 2.0))

    check_cols = [c for c in df.columns if c not in {"image_name", "frame_number"}]
    df_num = df[check_cols].apply(pd.to_numeric, errors="coerce")

    missing_columns = [c for c in critical_cols if c not in df.columns]
    # Report all-NaN only for stable critical columns; event-like columns
    # (arc/plasma/transition) are allowed to be absent in many frames.
    all_nan_cols = [c for c in critical_cols if c in df_num.columns and df_num[c].isna().all()]

    n = len(df)
    anomaly_score = np.zeros(n, dtype=float)
    reasons = [[] for _ in range(n)]

    def _add_reason(mask: pd.Series, text: str, weight: float):
        if mask is None:
            return
        m = pd.Series(mask, index=df.index).fillna(False).to_numpy(dtype=bool)
        if not m.any():
            return
        anomaly_score[m] += weight
        for idx in np.where(m)[0]:
            reasons[idx].append(text)

    # 1) Missing value anomalies in critical columns (if present and not all-NaN).
    present_critical_cols = [c for c in critical_cols if c in df_num.columns]
    valid_critical_cols = [c for c in present_critical_cols if c not in all_nan_cols]
    for col in valid_critical_cols:
        _add_reason(df_num[col].isna(), f"missing:{col}", weight=missing_weight)

    # 2) Build droplet/spatter presence and aggregate features.
    spatter_presence, droplet_presence, agg_features = _make_presence_features(df_num)

    # 3) Sudden random jumps vs previous window in stable critical columns only.
    jump_feature_df = pd.DataFrame(index=df.index)
    for col in valid_critical_cols:
        jump_feature_df[col] = df_num[col]

    for col in jump_feature_df.columns:
        _add_reason(
            _rolling_jump_mask(
                jump_feature_df[col],
                window=window_size,
                mean_pct_threshold=jump_mean_pct_threshold,
            ),
            f"jump:{col}",
            weight=jump_weight,
        )

    #############

        # TEMPORARY REMOVAL OF DROPLET/SPTTER JUMP CHECKS - NOT USED IN PRIMARY ANALYSIS

    #############

    # 4) Short-lived isolated droplet/spatter events.
    #for obj_col in spatter_presence.columns:
    #    mask = _isolated_short_lived_events(
    #        spatter_presence[obj_col],
    #        window=window_size,
    #        max_len=short_life_max_len,
    #    )
    #    _add_reason(mask, f"transient:{obj_col}", weight=transient_weight)
#
    #for obj_col in droplet_presence.columns:
    #    mask = _isolated_short_lived_events(
    #        droplet_presence[obj_col],
    #        window=window_size,
    #        max_len=short_life_max_len,
    #    )
    #    _add_reason(mask, f"transient:{obj_col}", weight=transient_weight)

    # 5) Short-lived isolated events for arc/plasma/transition columns.
    #for col in transient_event_cols:
    #    if col not in df_num.columns:
    #        continue
#
    #    s = pd.to_numeric(df_num[col], errors="coerce")
    #    if col == "arc_detected":
    #        presence = s.fillna(0) > 0
    #    else:
    #        # For these, "present" means a real non-zero numeric value exists.
    #        presence = s.notna() & (s != 0)
#
    #    mask = _isolated_short_lived_events(
    #        presence,
    #        window=window_size,
    #        max_len=short_life_max_len,
    #    )
    #    _add_reason(mask, f"transient:{col}", weight=transient_weight)

    out = df[["image_name"]].copy()
    if "frame_number" in df.columns:
        out["frame_number"] = df["frame_number"]
    else:
        out["frame_number"] = np.arange(len(df))

    out["anomaly_score"] = anomaly_score
    out["anomaly"] = (out["anomaly_score"] >= anomaly_min_score).astype(int)
    out["anomaly_reason"] = [";".join(sorted(set(r))) if r else "" for r in reasons]

    output_csv_path = anomaly_csv_dir / output_csv_name
    out.to_csv(output_csv_path, index=False)

    anomalies = out[out["anomaly"] == 1].sort_values("frame_number")
    source_folder_name = Path(output_csv_name).stem.replace("anomaly_frames_", "") or "unknown_source"
    anomaly_images_source_dir = anomaly_images_dir / source_folder_name
    anomaly_images_source_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    if detection_images_dir is not None:
        for image_name in anomalies["image_name"].tolist():
            src = None
            candidate = detection_images_dir / str(image_name)
            if candidate.exists():
                src = candidate

            if src is not None:
                dst = anomaly_images_source_dir / str(image_name)
                shutil.copy2(src, dst)
                copied_images += 1

    total_frames = len(out)
    anomaly_count = len(anomalies)
    anomaly_ratio = (anomaly_count / total_frames) if total_frames > 0 else 0.0

    anomalous_frames = []
    for _, row in anomalies.iterrows():
        anomalous_frames.append(
            {
                "image_name": str(row["image_name"]),
                "frame_number": int(row["frame_number"]),
                "anomaly_score": float(row["anomaly_score"]),
            }
        )

    if verbose:
        print(f"Anomaly scoring completed: {anomaly_count}/{total_frames} frames flagged.")
        print(f"Anomaly minimum score threshold: {anomaly_min_score:g}")
        if missing_columns:
            print(f"Missing critical columns: {missing_columns}")
        if all_nan_cols:
            print(f"All-NaN feature columns: {all_nan_cols}")
        if anomaly_ratio > 0.10:
            print("Recheck is needed")
        else:
            print("No recheck needed")
        if anomaly_count > 0 and detection_images_dir is None:
            print("Detected anomaly frames were not copied (detection images directory not provided).")
        if copied_images > 0:
            print(f"Copied anomalous images: {copied_images}")

    return {
        "csv_path": output_csv_path,
        "images_dir": anomaly_images_source_dir,
        "copied_images": copied_images,
        "total_frames": total_frames,
        "anomaly_frames": anomaly_count,
        "anomaly_ratio": anomaly_ratio,
        "missing_columns": missing_columns,
        "dropped_columns": all_nan_cols,
        "anomalous_frames": anomalous_frames,
    }


def run_rule_based_anomaly_for_folder(
    detection_folder: Path,
    anomaly_output_dir: Path,
    detection_images_dir: Path = None,
    source_images_dir: Path = None,
    clear_existing: bool = True,
    verbose: bool = True,
):
    """Run rule-based anomaly scoring for every detection CSV in a folder."""
    detection_folder = Path(detection_folder)
    anomaly_output_dir = Path(anomaly_output_dir)

    detection_csv_folder = detection_folder / "csv"
    if detection_csv_folder.exists():
        csv_files = sorted(detection_csv_folder.glob("*.csv"))
    else:
        csv_files = sorted(detection_folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in detection folder: {detection_folder}")

    results = []
    for idx, csv_path in enumerate(csv_files):
        output_csv_name = csv_path.name.replace("detections_", "anomaly_frames_")
        source_name = csv_path.stem.replace("detections_", "")
        source_detection_images_dir = None
        if detection_images_dir is not None:
            candidate_dir = Path(detection_images_dir) / source_name
            source_detection_images_dir = candidate_dir if candidate_dir.exists() else None

        result = run_rule_based_anomaly(
            detection_csv_path=csv_path,
            anomaly_output_dir=anomaly_output_dir,
            detection_images_dir=source_detection_images_dir,
            source_images_dir=source_images_dir,
            image_lookup=None,
            output_csv_name=output_csv_name,
            clear_existing=(clear_existing and idx == 0),
            verbose=verbose,
        )
        results.append(result)

    return results


if __name__ == "__main__":
    input_root, output_root = get_current_io_paths(CONFIG)
    runtime = get_current_runtime_settings(CONFIG)

    detect_dir = output_root / "detect"
    anomaly_dir = output_root / "anomaly"
    save_anomaly_frames = runtime.get("save_anomaly_frames", False)
    detect_images_dir = detect_dir / "images"

    results = run_rule_based_anomaly_for_folder(
        detection_folder=detect_dir,
        anomaly_output_dir=anomaly_dir,
        detection_images_dir=detect_images_dir if save_anomaly_frames and detect_images_dir.exists() else None,
        source_images_dir=input_root if save_anomaly_frames else None,
        clear_existing=True,
        verbose=True,
    )
    print(f"Processed {len(results)} detection CSV file(s).")