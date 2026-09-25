"""Central configuration for the welding analysis pipeline.

`CONFIG` is intentionally a plain dictionary to keep edits simple during
experimentation. Runtime helpers below provide validated access patterns.
"""

from pathlib import Path
import numpy as np

CONFIG = {

    # Repository-relative defaults keep the pipeline portable across machines.
    "project_root": str(Path(__file__).resolve().parents[2]),

    # Quick edit guide:
    # 1) Set input_root (folder containing AVI files and/or frame folders)
    # 2) Set output_root
    # 3) Pick frame/save settings in run
    # 3) Leave the rest unless tuning detection behavior

    # ================================
    # INPUT / OUTPUT (EDIT THIS BLOCK)
    # ================================
    "input_root": str(Path(__file__).resolve().parents[2] / "data"),
    "output_root": str(Path(__file__).resolve().parents[2] / "output"),

    # ================================
    # CURRENT RUN SETTINGS (EDIT THIS BLOCK)
    # ================================
    "run": {
        # Use None to process all frames
        "start_img": None,
        "end_img": None,

        # Set to None for automatic detection, set to "backlit" or "laserlit" to override
        "video_mode": None,

        # Output/visualization toggles
        "save_csv": True,
        "save_detect_images_debug": False,
        "save_track_images_debug": False,
        "save_anomaly_frames": False,
    },

    # General parameters
    "wire_width_mm": 1.0,

    "wire_tool_transition": {
        "offset_from_center": 10,
        "start_offset": 50,
        "extend_upward": 1000,
    },

    "wire_searchspace": {
        "top_shift_px": 5,
        "length_scale": 1.2,
        "width_scale": 3,
    },

    "solid_molten_transition": {
        "width_threshold_px": 10.0,
        "min_run": 10,
    },

    "weldpool": {
        "min_edge_width_px": 1,
        "scan_half_width_px": 10,
        "remove_circular_droplets": True,
        "droplet_min_area_px": 1,
        "droplet_circularity_threshold": 0.4,
        "show_verification": False,
    },

    "wire_contour": {
        "min_length": 100,
        "min_width": 2,
        "max_width": 1000,
        "max_angle": 30.0,
        "min_h_to_w": 1,
        "split_threshold": 400, #150
    },

    "anomaly_detection": {
        "min_anomaly_score": 1.0,
        # Percentage jump from prior-window mean. Example: 0.20 = 20%.
        "jump_mean_pct_threshold": 0.20,
        "window_size": 30,
        "short_life_max_len": 2,
        "jump_weight": 1.0,
        "missing_weight": 1.0,
        "transient_weight": 2.0,
    },


    

    # Laserlit parameters
    "laserlit": {
        "laserlit_edge": {
            "canny1": 25,
            "canny2": 90, 
            "morph": True,      
        },

        "laserlit_basemetal": {
            "min_edge_pixels": 2,
            "min_edge_width": 10,
        },

        "laserlit_wire_edges": {
            "hough_threshold": 10,
            "min_line_length": 30, #40
            "max_line_gap": 15, #25
            "rho": 1,
            "theta": np.pi/360,  # 1
            "min_length": 30,
            "angle_tolerance_deg": 3.0,
            "vertical_tolerance_deg": 10.0,
            "min_x_distance": 25,
            "max_x_distance": 50,
        },

        "laserlit_contours": {
            "canny1": 25,
            "canny2": 75,
            "morph": True,
        },

        "laserlit_tapering_point": {
            "search_window_radius": 8,
            "vicinity_radius": 2,
            "max_consecutive_misses": 2,
        },

        "arc_detection": { 
            "plasma_channel_edge": {
                "box_width": 150,
                "search_band_height": 5,
                "track_window": 5,
                "min_gradient_score": 3.0,
                "gradient_std_factor": 0.05,
            },
            "plasma_channel_attachment": {
                "box_width": 80,
                "box_height": 80,
                "blur_kernel": 5,
                "std_threshold_factor": 0.75,
                "percentile_threshold": 85,
            },
        },

    },

    # Backlit parameters
    "backlit": {    
        "backlit_edge": {
            "canny1": 75,
            "canny2": 125,
            "morph": False,       
        },
        "backlit_basemetal": {
            "min_edge_pixels": 2,
            "min_edge_width": 10,
        },

        "backlit_wire_edges": {
            "hough_threshold": 15,
            "min_line_length": 30,
            "max_line_gap": 5,     # 10
            "rho": 1,               # 1
            "theta": 1,  # 1
            "min_length": 30,
            "angle_tolerance_deg": 2.0,
            "vertical_tolerance_deg": 15.0,
            "min_x_distance": 30,
            "max_x_distance": 100,
        },

        "backlit_contours": {
            "canny1": 75,
            "canny2": 125,
            "morph": False,
        },

        "backlit_tapering_point": {
            "search_window_radius": 8,
            "vicinity_radius": 1,       # 3
            "max_consecutive_misses": 2,# 3
        },

        "arc_detection": {
            "plasma_channel_edge": {
                "box_width": 150,
                "search_band_height":8,
                "track_window": 5,
                "min_gradient_score": 1.3, # 1.0
                "gradient_std_factor": 0.10, # 0.10
            },
            "plasma_channel_attachment": {
                "box_width": 80,
                "box_height": 80,
                "blur_kernel": 5,
                "std_threshold_factor": 0.85, # 0.75
                "percentile_threshold": 95,
            },
        },
    },

}


def get_current_io_paths(config: dict = CONFIG):
    """Resolve active input/output root folders from config."""
    return Path(config["input_root"]), Path(config["output_root"])


def get_current_runtime_settings(config: dict = CONFIG):
    """Resolve active runtime parameters from config."""
    current = config.get("run", {}) or {}

    return {
        "start_img": current.get("start_img", None),
        "end_img": current.get("end_img", None),
        "save_csv": current.get("save_csv", True),
        "save_detect_images_debug": current.get(
            "save_detect_images_debug",
            current.get("save_images", False),
        ),
        "save_track_images_debug": current.get(
            "save_track_images_debug",
            current.get("save_images", False),
        ),
        "save_anomaly_frames": current.get(
            "save_anomaly_frames",
            current.get("save_images", False),
        ),
    }


def apply_runtime_overrides(
    config: dict = CONFIG,
    input_root: str | Path | None = None,
    output_root: str | Path | None = None,
    start_img: int | None = None,
    end_img: int | None = None,
    save_csv: bool | None = None,
    save_detect_images_debug: bool | None = None,
    save_track_images_debug: bool | None = None,
    save_anomaly_frames: bool | None = None,
):
    """Apply runtime overrides to config in one place (used by CLI/app entrypoint)."""
    if input_root is not None:
        config["input_root"] = str(input_root)
    if output_root is not None:
        config["output_root"] = str(output_root)

    run_cfg = config.setdefault("run", {})
    if start_img is not None:
        run_cfg["start_img"] = int(start_img)
    if end_img is not None:
        run_cfg["end_img"] = int(end_img)
    if save_csv is not None:
        run_cfg["save_csv"] = bool(save_csv)
    if save_detect_images_debug is not None:
        run_cfg["save_detect_images_debug"] = bool(save_detect_images_debug)
    if save_track_images_debug is not None:
        run_cfg["save_track_images_debug"] = bool(save_track_images_debug)
    if save_anomaly_frames is not None:
        run_cfg["save_anomaly_frames"] = bool(save_anomaly_frames)
