# Image analysis

Frame-level feature extraction for welding-process monitoring.

## What it does

- reads `.avi`, `.jpg`, `.jpeg`, and `.png` inputs;
- detects wire geometry, weld pool, base metal, arc/plasma features, droplets, and spatters;
- tracks frame-level detections;
- assigns deterministic anomaly scores;
- writes detection, tracking, anomaly, and optional visualization outputs.

## Run

From the repository root:

```bash
cd image_analysis_app
python -m scripts.pipeline.main
```

Configure paths and detector behavior in `scripts/config.py`. `video_mode = None` detects `backlit` or `laserlit` from the input path; set it explicitly to override detection. Debug image outputs are disabled by default to keep runs small.

## Input convention

Place source data below `data/`, or point `input_root` to another directory. Folder names containing `backlit` or `laserlit` select the corresponding detector when automatic mode is enabled.

## Output convention

The pipeline writes below `output_root`:

```text
detect/{csv,images}/
track/{csv,images,trajectory_plots,velocity_plots}/
anomaly/{csv,images}/
```

ESAB source data is intentionally not included in this repository. The full research workflow and access limitations are documented in the [root README](../README.md).

> Note: the supplied archive references `scripts/droplets_spatters`, but that package is not present in the archive or repository history. Restore the original ESAB-authorized package before running the end-to-end pipeline.
