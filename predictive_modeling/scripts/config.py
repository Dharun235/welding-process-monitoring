from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIVE_ROOT = PROJECT_ROOT / "predictive_modeling"


CONFIG = {
    "preprocessing": {
        "input_path": str(PROJECT_ROOT / "output" / "measurements"),
        "detect_path": str(PROJECT_ROOT / "output" / "detect" / "csv"),
        "anomaly_path": str(PROJECT_ROOT / "output" / "anomaly" / "csv"),
        "h5_input_path": str(PROJECT_ROOT / "data" / "backlit" / "to_be_processed"),
        "h5_output_path": str(PROJECT_ROOT / "output" / "measurements"),
        "output_path_org": str(PREDICTIVE_ROOT / "data" / "merged_org"),
        "output_path_clipped": str(PREDICTIVE_ROOT / "data" / "merged_clipped"),
        "output_path_clipped_norm": str(PREDICTIVE_ROOT / "data" / "merged_clipped_norm"),
        "interpolation": {
            "interpolation_columns": [
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
            ],
        },
        "max_anomaly_length": 10,
        "upsample": False,
        "split": {
            "allow_missing_target": True,
            "target_col": "wiretip_to_weldpool_dist",
            "history_cols": ["voltage", "current", "delta_voltage", "delta_current"],
            "context_cols": ["wfs"],
            "offline_metadata_cols": ["wfs", "ctdw"],
            "frame_order_col": "meas_time",
            "subsequence_length": 4_000,
            "subsequence_split_gap_rows": 50,
            "min_subsequence_rows": 1_000,
            "train_fraction": 0.60,
            "validation_fraction": 0.20,
            "split_shuffle_trials": 2_000,
            "random_state": 42,
        },
    },
}
