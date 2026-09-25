"""Droplet/spatter tracking stage.

Creates tracked CSV outputs, optional debug frames, and trajectory/velocity
summary plots grouped per source.
"""

import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scripts.config import CONFIG, get_current_io_paths, get_current_runtime_settings
from scripts.droplets_spatters.track_management import DropletTrackManager
from scripts.pipeline.analyze import analyze_image


TRACKED_CSV_NAME = "tracked_features.csv"
TRACKED_TRAJECTORY_PLOT_NAME = "trajectory_plot.png"
TRACKED_VELOCITY_PLOT_NAME = "velocity_plot.png"


def clear_directory_contents(folder_path: Path):
    """Remove all files/subfolders from a directory if it exists."""
    if not folder_path.exists():
        return
    for child in folder_path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def prepare_output_folder(output_folder: Path, save_track_images_debug: bool, csv_name: str = TRACKED_CSV_NAME):
    """Prepare and clean tracking output subfolders for a fresh run."""
    output_folder.mkdir(parents=True, exist_ok=True)
    tracked_images_folder = output_folder / "images"
    csv_output_folder = output_folder / "csv"
    trajectory_plots_folder = output_folder / "trajectory_plots"
    velocity_plots_folder = output_folder / "velocity_plots"

    if tracked_images_folder.exists():
        clear_directory_contents(tracked_images_folder)
    if trajectory_plots_folder.exists():
        clear_directory_contents(trajectory_plots_folder)
    if velocity_plots_folder.exists():
        clear_directory_contents(velocity_plots_folder)
    if csv_output_folder.exists():
        clear_directory_contents(csv_output_folder)

    if save_track_images_debug:
        tracked_images_folder.mkdir(parents=True, exist_ok=True)
    elif tracked_images_folder.exists():
        shutil.rmtree(tracked_images_folder)

    trajectory_plots_folder.mkdir(parents=True, exist_ok=True)
    velocity_plots_folder.mkdir(parents=True, exist_ok=True)
    csv_output_folder.mkdir(parents=True, exist_ok=True)

    # Clean legacy flat files in output root.
    for legacy_csv in output_folder.glob("*.csv"):
        legacy_csv.unlink()
    for legacy_plot in output_folder.glob("trajectory_plot*.png"):
        legacy_plot.unlink()
    for legacy_plot in output_folder.glob("velocity_plot*.png"):
        legacy_plot.unlink()

    csv_path = csv_output_folder / csv_name

    return tracked_images_folder, csv_path


def get_image_files(input_folder: Path, start_img: int = None, end_img: int = None):
    """Return sorted image files from a folder with optional frame slicing."""
    start_img = start_img if start_img is not None else 0
    image_files = sorted(
        [file for file in input_folder.iterdir() if file.suffix.lower() in {".jpg", ".png"}]
    )
    end_img = end_img if end_img is not None else len(image_files)
    return image_files[start_img:end_img]


def normalize_detected_features(results_dict):
    """Convert raw detection payload into normalized droplet/spatter feature list."""
    features = []
    for feature_type in ("droplet", "spatter"):
        key = f"{feature_type}s"
        for item in results_dict.get(key, []) or []:
            centroid = item.get("centroid")
            if centroid is None or len(centroid) != 2:
                continue
            features.append(
                {
                    "feature_type": feature_type,
                    "centroid": (int(centroid[0]), int(centroid[1])),
                    "area": item.get("area"),
                    "diameter_px": item.get("diameter_px"),
                }
            )
    return features


def split_features_by_type(features):
    """Split normalized features into droplet and spatter lists."""
    droplets = [f for f in features if f["feature_type"] == "droplet"]
    spatters = [f for f in features if f["feature_type"] == "spatter"]
    return droplets, spatters


def create_track_managers():
    """Create tracker managers with tuned parameters per feature type."""
    droplet_manager = DropletTrackManager(
        max_match_distance=20,
        max_missed_frames=3,
        min_confirmed_hits=2,
        min_track_displacement=6,
        min_vertical_displacement=4,
    )
    spatter_manager = DropletTrackManager(
        max_match_distance=18,
        max_missed_frames=2,
        min_confirmed_hits=2,
        min_track_displacement=3,
        min_vertical_displacement=1,
    )
    return droplet_manager, spatter_manager


def get_public_track_id(track, id_map, next_id):
    """Map internal tracker IDs to compact public IDs for output consistency."""
    internal_id = track.id
    if internal_id not in id_map:
        id_map[internal_id] = next_id
        next_id += 1
    return id_map[internal_id], next_id


def draw_tracks(image, tracks, prefix, color, id_map, next_id):
    """Draw track markers/labels on a visualization image."""
    for track in tracks:
        public_id, next_id = get_public_track_id(track, id_map, next_id)
        observation = track.latest_observation()
        x, y = observation.centroid
        cv2.circle(image, (x, y), 5, color, -1)
        cv2.putText(
            image,
            f"{prefix}{public_id}",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return next_id


def append_rows(rows, image_name, frame_idx, feature_type, tracks, id_map, next_id):
    """Append latest track observations as CSV rows for one frame."""
    for track in tracks:
        public_id, next_id = get_public_track_id(track, id_map, next_id)
        observation = track.latest_observation()
        rows.append(
            {
                "image_name": image_name,
                "frame_idx": frame_idx,
                "feature_type": feature_type,
                "track_id": public_id,
                "x": observation.centroid[0],
                "y": observation.centroid[1],
                "area": observation.area,
                "diameter_px": observation.diameter_px,
            }
        )
    return next_id


def save_tracked_features_csv(rows, csv_path: Path):
    """Write tracked rows and remap IDs to compact per-type identifiers."""
    rows.sort(key=lambda row: (row["frame_idx"], row["feature_type"], row["track_id"]))

    compact_id_map = {"droplet": {}, "spatter": {}}
    next_compact_id = {"droplet": 0, "spatter": 0}

    compact_rows = []
    for row in rows:
        feature_type = row["feature_type"]
        original_id = row["track_id"]
        if original_id not in compact_id_map[feature_type]:
            compact_id_map[feature_type][original_id] = next_compact_id[feature_type]
            next_compact_id[feature_type] += 1

        compact_row = dict(row)
        compact_row["track_id"] = compact_id_map[feature_type][original_id]
        compact_rows.append(compact_row)

    with open(csv_path, "w", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "image_name",
                "frame_idx",
                "feature_type",
                "track_id",
                "x",
                "y",
                "area",
                "diameter_px",
            ],
        )
        writer.writeheader()
        writer.writerows(compact_rows)


def load_existing_tracked_rows(csv_path: Path):
    """Load and type-cast existing tracking rows when appending outputs."""
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = list(reader)
    for row in rows:
        row["frame_idx"] = int(float(row["frame_idx"]))
        row["track_id"] = int(float(row["track_id"]))
        row["x"] = int(float(row["x"]))
        row["y"] = int(float(row["y"]))
        if row.get("area") not in (None, "", "nan"):
            row["area"] = float(row["area"])
        else:
            row["area"] = None
        if row.get("diameter_px") not in (None, "", "nan"):
            row["diameter_px"] = float(row["diameter_px"])
        else:
            row["diameter_px"] = None
    return rows


def save_tracking_plots(
    tracked_csv_path: Path,
    output_folder: Path,
    trajectory_plot_name: str = TRACKED_TRAJECTORY_PLOT_NAME,
    velocity_plot_name: str = TRACKED_VELOCITY_PLOT_NAME,
):
    """Generate tracking summary plots from tracked CSV.

    Trajectory figure is rendered as two time-series subplots:
    - x position vs frame index
    - y position vs frame index

    Both use exact image pixel coordinates from tracking CSV (no relative shift).
    """
    trajectory_plots_folder = output_folder / "trajectory_plots"
    velocity_plots_folder = output_folder / "velocity_plots"
    trajectory_plots_folder.mkdir(parents=True, exist_ok=True)
    velocity_plots_folder.mkdir(parents=True, exist_ok=True)

    trajectory_plot_path = trajectory_plots_folder / trajectory_plot_name
    velocity_plot_path = velocity_plots_folder / velocity_plot_name

    def _save_empty_plots(message: str):
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(trajectory_plot_path, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(velocity_plot_path, dpi=150)
        plt.close(fig)

        print(f"Saved trajectory plot: {trajectory_plot_path}")
        print(f"Saved velocity plot: {velocity_plot_path}")

        return {
            "trajectory_plot": trajectory_plot_path,
            "velocity_plot": velocity_plot_path,
        }

    if not tracked_csv_path.exists():
        return _save_empty_plots("No tracking CSV found")

    df = pd.read_csv(tracked_csv_path)
    if df.empty:
        return _save_empty_plots("No confirmed tracked droplets/spatters")

    required_cols = {"feature_type", "track_id", "frame_idx", "x", "y"}
    if not required_cols.issubset(df.columns):
        return _save_empty_plots("Tracking CSV format not supported")

    # Trajectory plot: time-vs-location comparison in image pixel coordinates.
    fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    grouped = df.groupby(["feature_type", "track_id"], dropna=False)
    cmap = plt.cm.get_cmap("tab20")
    grouped_list = list(grouped)
    for idx, ((feature_type, track_id), group) in enumerate(grouped_list):
        group = group.sort_values("frame_idx")
        if len(group) == 0:
            continue

        frame_idx = group["frame_idx"].to_numpy(dtype=float)
        x = group["x"].to_numpy(dtype=float)
        y = group["y"].to_numpy(dtype=float)

        color = cmap(idx % 20)
        linestyle = "-" if feature_type == "droplet" else "--"
        label = f"{feature_type[0].upper()}{int(track_id)}"

        # Exact x position (px) in image coordinates against frame index.
        ax_x.plot(
            frame_idx,
            x,
            color=color,
            alpha=0.9,
            linewidth=1.3,
            linestyle=linestyle,
            label=label,
        )

        # Exact y position (px) in image coordinates against frame index.
        ax_y.plot(
            frame_idx,
            y,
            color=color,
            alpha=0.9,
            linestyle=linestyle,
            linewidth=1.3,
            label=label,
        )

    ax_x.set_title("Tracked Droplet/Spatter X Position vs Frame")
    ax_x.set_ylabel("x (px)")
    ax_x.grid(alpha=0.2)

    ax_y.set_title("Tracked Droplet/Spatter Y Position vs Frame")
    ax_y.set_xlabel("frame index")
    ax_y.set_ylabel("y (px)")
    ax_y.grid(alpha=0.2)

    handles, labels = ax_x.get_legend_handles_labels()
    if handles:
        ax_x.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)

    handles_y, labels_y = ax_y.get_legend_handles_labels()
    if handles_y:
        ax_y.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)

    fig.tight_layout()
    fig.savefig(trajectory_plot_path, dpi=150)
    plt.close(fig)

    # Velocity plot (all tracks together, each track ID shown)
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, ((feature_type, track_id), group) in enumerate(grouped_list):
        group = group.sort_values("frame_idx")
        if len(group) < 2:
            continue

        x = group["x"].to_numpy(dtype=float)
        y = group["y"].to_numpy(dtype=float)
        frame_idx = group["frame_idx"].to_numpy(dtype=float)

        dx = np.diff(x)
        dy = np.diff(y)
        df_idx = np.diff(frame_idx)
        valid = df_idx > 0
        if not np.any(valid):
            continue

        speed = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2) / df_idx[valid]
        speed_frame = frame_idx[1:][valid]

        color = cmap(idx % 20)
        linestyle = "-" if feature_type == "droplet" else "--"
        label = f"{feature_type[0].upper()}{int(track_id)}"
        ax.plot(
            speed_frame,
            speed,
            color=color,
            linestyle=linestyle,
            alpha=0.9,
            linewidth=1.3,
            label=label,
        )

    ax.set_title("Tracked Droplet/Spatter Velocity")
    ax.set_xlabel("frame index")
    ax.set_ylabel("velocity (px/frame)")
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)

    fig.tight_layout()
    fig.savefig(velocity_plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved trajectory plot: {trajectory_plot_path}")
    print(f"Saved velocity plot: {velocity_plot_path}")

    return {
        "trajectory_plot": trajectory_plot_path,
        "velocity_plot": velocity_plot_path,
    }


def _run_tracking_from_frames(
    frames_data,
    output_folder: Path,
    append: bool = False,
    clear_existing: bool = True,
    csv_name: str = TRACKED_CSV_NAME,
    trajectory_plot_name: str = TRACKED_TRAJECTORY_PLOT_NAME,
    velocity_plot_name: str = TRACKED_VELOCITY_PLOT_NAME,
):
    """Run tracking from precomputed per-frame analysis payloads."""
    runtime = get_current_runtime_settings(CONFIG)
    save_track_images_debug = runtime["save_track_images_debug"]

    if clear_existing:
        tracked_images_folder, csv_path = prepare_output_folder(
            output_folder,
            save_track_images_debug,
            csv_name=csv_name,
        )
    else:
        output_folder.mkdir(parents=True, exist_ok=True)
        tracked_images_folder = output_folder / "images"
        csv_output_folder = output_folder / "csv"
        if save_track_images_debug:
            tracked_images_folder.mkdir(parents=True, exist_ok=True)
        csv_output_folder.mkdir(parents=True, exist_ok=True)
        csv_path = csv_output_folder / csv_name

    source_folder_name = Path(csv_name).stem.replace("tracked_features_", "") or "unknown_source"
    tracked_images_source_folder = tracked_images_folder / source_folder_name
    if save_track_images_debug:
        tracked_images_source_folder.mkdir(parents=True, exist_ok=True)

    print(f"Cleared existing tracking outputs in: {output_folder}")

    droplet_manager, spatter_manager = create_track_managers()
    tracked_rows = []
    saved_frame_count = 0
    droplet_id_map = {}
    spatter_id_map = {}
    next_droplet_id = 0
    next_spatter_id = 0

    for frame_idx, frame_data in enumerate(frames_data):
        file_name = frame_data["image_name"]
        results_dict = frame_data["results_dict"]
        vis_image = frame_data["vis_image"]

        if results_dict is None or vis_image is None:
            continue

        detected_features = normalize_detected_features(results_dict)
        droplets, spatters = split_features_by_type(detected_features)

        confirmed_droplets = droplet_manager.update(droplets, frame_idx=frame_idx)
        confirmed_spatters = spatter_manager.update(spatters, frame_idx=frame_idx)

        if not confirmed_droplets and not confirmed_spatters:
            print(f"Frame {frame_idx}: no confirmed tracked droplets or spatters")
            continue

        next_droplet_id = draw_tracks(
            vis_image,
            confirmed_droplets,
            prefix="D",
            color=(0, 255, 0),
            id_map=droplet_id_map,
            next_id=next_droplet_id,
        )
        next_spatter_id = draw_tracks(
            vis_image,
            confirmed_spatters,
            prefix="S",
            color=(0, 165, 255),
            id_map=spatter_id_map,
            next_id=next_spatter_id,
        )

        droplet_public_ids = [droplet_id_map[t.id] for t in confirmed_droplets]
        spatter_public_ids = [spatter_id_map[t.id] for t in confirmed_spatters]

        print(
            f"Frame {frame_idx}: confirmed droplet IDs = {droplet_public_ids}, "
            f"confirmed spatter IDs = {spatter_public_ids}"
        )

        next_droplet_id = append_rows(
            tracked_rows,
            file_name,
            frame_idx,
            "droplet",
            confirmed_droplets,
            id_map=droplet_id_map,
            next_id=next_droplet_id,
        )
        next_spatter_id = append_rows(
            tracked_rows,
            file_name,
            frame_idx,
            "spatter",
            confirmed_spatters,
            id_map=spatter_id_map,
            next_id=next_spatter_id,
        )

        if save_track_images_debug:
            out_path = tracked_images_source_folder / file_name
            cv2.imwrite(str(out_path), vis_image)
            saved_frame_count += 1

    existing_rows = load_existing_tracked_rows(csv_path) if append else []
    all_rows = [*existing_rows, *tracked_rows]
    save_tracked_features_csv(all_rows, csv_path)
    plot_info = save_tracking_plots(
        csv_path,
        output_folder,
        trajectory_plot_name=trajectory_plot_name,
        velocity_plot_name=velocity_plot_name,
    )

    if save_track_images_debug:
        print(
            f"Tracking completed. Saved {saved_frame_count} frame(s) with confirmed tracked droplets/spatters to: {tracked_images_source_folder}"
        )
    else:
        print("Tracking images not saved (debug option disabled)")
    print(f"Tracked droplet/spatter measurements saved to: {csv_path}")

    return {
        "csv_path": csv_path,
        "images_dir": tracked_images_source_folder if save_track_images_debug else None,
        "rows": len(all_rows),
        "trajectory_plot": plot_info["trajectory_plot"],
        "velocity_plot": plot_info["velocity_plot"],
    }


def track_features_from_results(
    analysis_results,
    output_folder: Path,
    append: bool = False,
    clear_existing: bool = True,
    csv_name: str = TRACKED_CSV_NAME,
    trajectory_plot_name: str = TRACKED_TRAJECTORY_PLOT_NAME,
    velocity_plot_name: str = TRACKED_VELOCITY_PLOT_NAME,
):
    """Track droplets/spatters from precomputed analysis results."""
    frames_data = [
        {
            "image_name": img_name,
            "results_dict": payload[0] if payload is not None else None,
            "vis_image": payload[1] if payload is not None else None,
        }
        for img_name, payload in analysis_results
    ]
    print(f"Found {len(frames_data)} images for tracking")
    return _run_tracking_from_frames(
        frames_data,
        output_folder,
        append=append,
        clear_existing=clear_existing,
        csv_name=csv_name,
        trajectory_plot_name=trajectory_plot_name,
        velocity_plot_name=velocity_plot_name,
    )


def track_features_on_folder(input_folder: Path, output_folder: Path, start_img: int = None, end_img: int = None):
    """Analyze and track all images from a folder in one pass."""
    image_files = get_image_files(input_folder, start_img=start_img, end_img=end_img)
    print(f"Found {len(image_files)} images for tracking")

    frames_data = []
    for file in image_files:
        results_dict, vis_image = analyze_image(str(input_folder), file.name)
        frames_data.append(
            {
                "image_name": file.name,
                "results_dict": results_dict,
                "vis_image": vis_image,
            }
        )

    return _run_tracking_from_frames(frames_data, output_folder)


if __name__ == "__main__":
    input_folder, output_root = get_current_io_paths(CONFIG)
    runtime = get_current_runtime_settings(CONFIG)

    output_folder = output_root / "track"
    track_features_on_folder(
        input_folder,
        output_folder,
        start_img=runtime["start_img"],
        end_img=runtime["end_img"],
    )
