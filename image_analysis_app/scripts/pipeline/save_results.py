"""Utilities to persist frame-level detection outputs.

This module normalizes analysis dictionaries into tabular rows and stores:
- per-source detection CSV files
- optional annotated detection images
"""

from pathlib import Path
import cv2
import csv
import re
import shutil
import numpy as np
from scripts.config import CONFIG, get_current_runtime_settings

def extract_timestamp(name: str):
    """Extract frame number from image filename (e.g., frame_00001_*.jpg)."""
    m = re.search(r"frame_(\d+)", name)
    if m:
        return int(m.group(1))
    numbers = re.findall(r'\d+', name)
    return int(numbers[0]) if numbers else 0

def safe_xy(value):
    """Safely extract (x, y) from a coordinate-like object, else return (nan, nan)."""
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
        x, y = value
        if x is not None and y is not None:
            return x, y
    return np.nan, np.nan

def safe_tooltip_points(value):
    """Safely extract up to two tooltip points from a nested tooltip structure."""
    if not isinstance(value, (list, tuple, np.ndarray)):
        return (np.nan, np.nan), (np.nan, np.nan)

    points = list(value)
    pt1 = safe_xy(points[0]) if len(points) > 0 else (np.nan, np.nan)
    pt2 = safe_xy(points[1]) if len(points) > 1 else (np.nan, np.nan)
    return pt1, pt2

def flatten_results(img_name: str, result: dict):
    """Flatten one frame's analysis dictionary into a CSV-ready row."""

    row = {
        "image_name": img_name,
        "frame_number": extract_timestamp(img_name),

        # Define basemetal and weldpool fields
        "basemetal_y": np.nan,
        "weldpool_x": np.nan,
        "weldpool_y": np.nan,

        # Define mm_per_px field
        "mm_per_px": np.nan,

        # Define wiretip fields
        "wiretip_x": np.nan,
        "wiretip_y": np.nan,

        # Define transition fields
        "transition_x": np.nan,
        "transition_y": np.nan,

        # Define tapering point fields
        "tapering_x": np.nan,
        "tapering_y": np.nan,

        # Define arc detection field
        "arc_detected": np.nan,

        # Define plasma channel fields
        "plasma_attachment_height": np.nan,
        "plasma_channel_avg_width": np.nan,

        # Define tooltip fields
        "tooltip_x1": np.nan,
        "tooltip_x2": np.nan,
        "tooltip_y1": np.nan,
        "tooltip_y2": np.nan,
    }

    # Assign measured values, using np.nan for any missing data
    row["basemetal_y"] = result.get("basemetal_y") if result.get("basemetal_y") is not None else np.nan
    row["mm_per_px"] = result.get("mm_per_px") if result.get("mm_per_px") is not None else np.nan
    row["weldpool_x"], row["weldpool_y"] = safe_xy(result.get("weldpool_location"))
    row["wiretip_x"], row["wiretip_y"] = safe_xy(result.get("wiretip_location"))
    row["transition_x"], row["transition_y"] = safe_xy(result.get("transition_point"))
    row["tapering_x"], row["tapering_y"] = safe_xy(result.get("tapering_point"))
    row["arc_detected"] = result.get("arc_detected") if result.get("arc_detected") is not None else np.nan
    row["plasma_attachment_height"] = result.get("plasma_attachment_height") if result.get("plasma_attachment_height") is not None else np.nan
    row["plasma_channel_avg_width"] = result.get("plasma_channel_avg_width") if result.get("plasma_channel_avg_width") is not None else np.nan
    (row["tooltip_x1"], row["tooltip_y1"]), (row["tooltip_x2"], row["tooltip_y2"]) = safe_tooltip_points(result.get("tooltip"))

    # Add droplets dynamically
    for i, d in enumerate(result.get("droplets", []) or [], 1):
        row[f"droplet_{i}_x"], row[f"droplet_{i}_y"] = safe_xy(d.get("centroid"))
        row[f"droplet_{i}_area"] = d.get("area", np.nan)
        row[f"droplet_{i}_diameter_px"] = d.get("diameter_px", np.nan)

    # Add spatters dynamically
    for i, s in enumerate(result.get("spatters", []) or [], 1):
        row[f"spatter_{i}_x"], row[f"spatter_{i}_y"] = safe_xy(s.get("centroid"))
        row[f"spatter_{i}_area"] = s.get("area", np.nan)
        row[f"spatter_{i}_diameter_px"] = s.get("diameter_px", np.nan)

    return row

def save_results(
    results_list,
    output_dir: Path,
    folder_name: str,
    csv_name: str = "detections.csv",
    append: bool = False,
    clear_existing: bool = False,
):
    """Persist detection outputs for one source.

    Args:
        results_list: sequence of `(image_name, (results_dict, vis_image))`.
        output_dir: root output folder for detect stage.
        folder_name: source identifier used to group debug images.
    """
    images_output_dir = output_dir / "images"
    csv_output_dir = output_dir / "csv"
    source_images_dir = images_output_dir / (str(folder_name) if folder_name else "unknown_source")
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = get_current_runtime_settings(CONFIG)
    save_detect_images_debug = runtime["save_detect_images_debug"]

    if clear_existing:
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        for old_csv in csv_output_dir.glob("*.csv"):
            old_csv.unlink()
        # Clean legacy flat CSVs if present.
        for old_csv in output_dir.glob("*.csv"):
            old_csv.unlink()
        if images_output_dir.exists():
            for child in images_output_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

    if save_detect_images_debug:
        images_output_dir.mkdir(parents=True, exist_ok=True)
        source_images_dir.mkdir(parents=True, exist_ok=True)
    elif images_output_dir.exists():
        shutil.rmtree(images_output_dir)

    collected_results = []
    for img_name, (results_dict, vis_image) in results_list:
        if results_dict is None:
            continue

        # flatten for CSV
        flat = flatten_results(img_name, results_dict)
        collected_results.append(flat)

        # save annotated image
        if vis_image is not None and save_detect_images_debug:
            cv2.imwrite(str(source_images_dir / img_name), vis_image)

    # Save CSV
    if collected_results:
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_output_dir / csv_name

        existing_rows = []
        if append and csv_path.exists():
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

        # Sort by frame number
        collected_results.sort(key=lambda x: x["frame_number"])
        all_rows = [*existing_rows, *collected_results]

        # Preserve the column order established by flatten_results.
        all_fieldnames = []
        for entry in all_rows:
            for key in entry.keys():
                if key not in all_fieldnames:
                    all_fieldnames.append(key)

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames)
            writer.writeheader()
            for entry in all_rows:
                writer.writerow(entry)

        print(f"\nSaved CSV: {csv_path}")
        if save_detect_images_debug:
            print(f"Annotated images saved in: {source_images_dir}")
        else:
            print("Annotated images not saved (debug option disabled)")

        return {
            "csv_path": csv_path,
            "images_dir": source_images_dir if save_detect_images_debug else None,
            "rows": len(all_rows),
        }

    return {
        "csv_path": None,
        "images_dir": source_images_dir if save_detect_images_debug else None,
        "rows": 0,
    }
