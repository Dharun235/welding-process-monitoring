"""Pipeline orchestrator.

This module discovers input sources (AVI files or frame folders) and executes
the full processing flow per source:

detect -> track -> anomaly
"""

from pathlib import Path
from scripts.pipeline.analyze import analyze_image
from scripts.pipeline.save_results import save_results
from scripts.droplets_spatters.tracker import track_features_from_results
from scripts.anomaly.anomaly_score import run_rule_based_anomaly
from scripts.config import CONFIG, get_current_io_paths, get_current_runtime_settings
from scripts.preprocessing.avi_to_frames import avi_to_frames
import shutil
import io
import contextlib
import sys

def analyze_single_image(folder_path: Path, img_name: str, video_mode: str):
    """Run frame-level analysis for one image file in a folder with mode."""
    return analyze_image(str(folder_path), img_name, video_mode)

def get_image_files(input_folder: Path, start_img: int = None, end_img: int = None):
    """Return sorted image files with optional [start:end] slicing."""
    start_img = start_img if start_img is not None else 0
    image_files = sorted([f for f in input_folder.iterdir() if f.suffix.lower() in {".jpg", ".png"}])
    end_img = end_img if end_img is not None else len(image_files)
    return image_files[start_img:end_img]

def resolve_io_folders_from_config():
    """Resolve active input/output roots from shared config."""
    return get_current_io_paths(CONFIG)


def _run_quiet(func, *args, **kwargs):
    """Run a function while suppressing noisy stdout/stderr; return (result, captured_log)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        result = func(*args, **kwargs)
    return result, buffer.getvalue()


def _render_progress_bar(prefix: str, current: int, total: int, width: int = 28):
    """Render a single-line progress bar in terminal using carriage return."""
    if total <= 0:
        total = 1
    pct = max(0.0, min(1.0, current / total))
    filled = int(width * pct)
    bar = "#" * filled + "-" * (width - filled)
    text = f"\r{prefix} [{bar}] {pct * 100:6.2f}% ({current}/{total})"
    sys.stdout.write(text)
    sys.stdout.flush()


def _finish_progress_bar():
    sys.stdout.write("\n")
    sys.stdout.flush()


def _sanitize_source_name(name: str):
    """Convert source names into filesystem-safe tokens."""
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    text = "".join(keep).strip("_")
    return text or "source"


def _rename_for_source(image_name: str, source_name: str):
    """Append source suffix to image filename to avoid cross-source collisions."""
    p = Path(image_name)
    return f"{p.stem}_{source_name}{p.suffix}"


def _is_frame_folder(folder: Path):
    """Check whether a folder directly contains supported image frames."""
    image_suffixes = {".jpg", ".jpeg", ".png"}
    try:
        return any(child.is_file() and child.suffix.lower() in image_suffixes for child in folder.iterdir())
    except PermissionError:
        return False
    
def _detect_video_mode(path: Path):
    """Infer mode by checking whether any directory in the path is named backlit."""
    directories = path.parents if path.is_file() else (path, *path.parents)
    return "backlit" if any(part.name.lower() == "backlit" for part in directories) else "laserlit"


def collect_sources(input_root: Path, temp_frames_root: Path, start_img: int = None, end_img: int = None):
    """
    Discover sources recursively and determine mode from parent folder:
    - .avi files (converted to temporary frame folders)
    - existing frame folders
    """
    sources = []
    seen_folders = set()

    # Support passing a single .avi file directly as input_root
    if input_root.is_file() and input_root.suffix.lower() == ".avi":
        avi_files = [input_root]
        input_root = input_root.parent
    else:
        avi_files = sorted(input_root.rglob("*.avi"))

    for avi_path in avi_files:
        source_name = _sanitize_source_name(avi_path.stem)
        frames_dir = temp_frames_root / source_name
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        first_saved_index = (start_img + 1) if start_img is not None else 1
        avi_to_frames(
            avi_path,
            frames_dir,
            start_index=first_saved_index,
            start_frame=start_img,
            end_frame=end_img,
        )

        # Mode detection from parent folder name
        mode = _detect_video_mode(avi_path)
        
        sources.append({
            "source_name": source_name,
            "input_folder": frames_dir,
            "is_temp": True,
            "range_pre_applied": True,
            "mode": mode,
        })
        seen_folders.add(frames_dir.resolve())

    for folder in sorted(p for p in input_root.rglob("*") if p.is_dir()):
        if temp_frames_root in folder.parents or folder == temp_frames_root:
            continue
        if _is_frame_folder(folder):
            key = folder.resolve()
            if key in seen_folders:
                continue
            source_name = _sanitize_source_name(folder.name)
            mode = _detect_video_mode(folder)
            sources.append({
                "source_name": source_name,
                "input_folder": folder,
                "is_temp": False,
                "range_pre_applied": False,
                "mode": mode,
            })
            seen_folders.add(key)

    # Include root itself if it is a frame folder
    if _is_frame_folder(input_root):
        key = input_root.resolve()
        if key not in seen_folders:
            mode = _detect_video_mode(input_root)
            sources.append({
                "source_name": _sanitize_source_name(input_root.name),
                "input_folder": input_root,
                "is_temp": False,
                "range_pre_applied": False,
                "mode": mode,
            })

    return sources

def welding_analysis_pipeline(input_folder: Path, output_folder: Path, start_img: int = None, end_img: int = None):
    """Run the end-to-end welding analysis pipeline for all discovered sources.

    Returns stage metadata for the latest detect/track result and aggregated
    anomaly results across sources.
    """
    runtime = get_current_runtime_settings(CONFIG)
    start_img = runtime["start_img"] if start_img is None else start_img
    end_img = runtime["end_img"] if end_img is None else end_img

    input_root = Path(input_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    detect_output = output_folder / "detect"
    track_output = output_folder / "track"
    anomaly_output = output_folder / "anomaly"
    temp_frames_root = output_folder / "_temp_frames"

    sources = collect_sources(input_root, temp_frames_root, start_img=start_img, end_img=end_img)
    if not sources:
        print(f"No AVI files or frame folders found under: {input_root}")
        return {
            "detect": None,
            "track": None,
            "anomaly": None,
        }

    print(f"Found {len(sources)} source(s) under {input_root}")
    print(f"Output root: {output_folder}")

    detect_info = None
    track_info = None
    anomaly_infos = []
    image_lookup = {}

    for source_index, source in enumerate(sources):
        source_name = source["source_name"]
        source_folder = source["input_folder"]
        _render_progress_bar("Pipeline", source_index + 1, len(sources))
        _finish_progress_bar()
        print(f"\nSource: {source_name}")

        if source.get("range_pre_applied", False):
            image_files = get_image_files(source_folder)
        else:
            image_files = get_image_files(source_folder, start_img=start_img, end_img=end_img)
        print(f"Found {len(image_files)} image(s) in {source_folder}")
        if not image_files:
            continue

        print("Detection started")
        analysis_results = []
        total_detect_frames = len(image_files)
        mode = source["mode"] if CONFIG["run"]["video_mode"] is None else CONFIG["run"]["video_mode"]
        for idx, file in enumerate(image_files, start=1):
            (results_dict, vis_image), _ = _run_quiet(analyze_single_image, source_folder, file.name, mode)
            out_name = _rename_for_source(file.name, source_name)
            analysis_results.append((out_name, (results_dict, vis_image)))
            image_lookup[out_name] = file
            _render_progress_bar("Detection", idx, total_detect_frames)
        _finish_progress_bar()

        if runtime["save_csv"]:
            detect_csv_name = f"detections_{source_name}.csv"
            detect_info, _ = _run_quiet(
                save_results,
                analysis_results,
                detect_output,
                source_name,
                csv_name=detect_csv_name,
                append=False,
                clear_existing=False,
            )
            print("Wire detection complete")
            print("Arc detection complete")
            print("Spatter and droplet detection complete")

        track_csv_name = f"tracked_features_{source_name}.csv"
        trajectory_plot_name = f"trajectory_plot_{source_name}.png"
        velocity_plot_name = f"velocity_plot_{source_name}.png"
        print("Tracking started")
        track_info, _ = _run_quiet(
            track_features_from_results,
            analysis_results,
            track_output,
            append=False,
            clear_existing=False,
            csv_name=track_csv_name,
            trajectory_plot_name=trajectory_plot_name,
            velocity_plot_name=velocity_plot_name,
        )
        print("Tracking complete")

        if detect_info and detect_info.get("csv_path"):
            anomaly_csv_name = f"anomaly_frames_{source_name}.csv"
            print("Anomaly scoring started")
            save_anomaly_frames = runtime.get("save_anomaly_frames", False)
            anomaly_info = run_rule_based_anomaly(
                detection_csv_path=detect_info["csv_path"],
                detection_images_dir=detect_info["images_dir"] if save_anomaly_frames else None,
                anomaly_output_dir=anomaly_output,
                source_images_dir=input_root if save_anomaly_frames else None,
                image_lookup=image_lookup if save_anomaly_frames else None,
                output_csv_name=anomaly_csv_name,
                clear_existing=False,
                verbose=False,
            )
            anomaly_info["source_name"] = source_name

            anomaly_infos.append(anomaly_info)
            print("Anomaly scoring complete")

    # Run anomaly score using detection CSV only
    anomaly_info = None
    if anomaly_infos:
        anomaly_info = anomaly_infos
        print("\nAnomaly summary by source:")
        for info in anomaly_infos:
            name = info.get("source_name", "unknown_source")
            flagged = info.get("anomaly_frames", 0)
            total = info.get("total_frames", 0)
            ratio = info.get("anomaly_ratio", 0.0)
            print(f"- {name}: {flagged}/{total} frames flagged ({ratio:.2%})")
    else:
        print("Anomaly scoring skipped: detection CSV was not created")

    if temp_frames_root.exists():
        shutil.rmtree(temp_frames_root)

    return {
        "detect": detect_info,
        "track": track_info,
        "anomaly": anomaly_info,
    }

if __name__ == "__main__":
    input_folder, output_folder = resolve_io_folders_from_config()
    runtime = get_current_runtime_settings(CONFIG)
    welding_analysis_pipeline(
        input_folder,
        output_folder,
        start_img=runtime["start_img"],
        end_img=runtime["end_img"],
    )
    print("\nPipeline execution completed.")
