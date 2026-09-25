# Image analysis

Computer-vision stage of the welding-process monitoring thesis. It converts welding videos or frame sequences into geometric, physical, and quality-related time-series features.

## Pipeline

```text
video / frames
      ↓
preprocessing → wire / weld-pool / arc detection
      ↓
droplet & spatter detection → tracking → anomaly scoring
      ↓
CSV features + optional annotated images and plots
```

## Components

| Component | Purpose |
| --- | --- |
| `scripts/preprocessing/` | Frame extraction, glare removal, and image preparation |
| `scripts/wire/` | Wire edges, contour, tip, tapering, and tool-transition features |
| `scripts/weldpool/` | Base-metal and weld-pool localization |
| `scripts/preprocessing/backlit/` | Backlit arc and wire processing |
| `scripts/preprocessing/laserlit/` | Laserlit wire processing |
| `scripts/droplets_spatters/` | Droplet/spatter detection, visualization, and tracking |
| `scripts/anomaly/` | Deterministic anomaly scoring |
| `scripts/pipeline/` | End-to-end orchestration and result export |

## Run

From the repository root:

```bash
cd image_analysis_app
python -m scripts.pipeline.main
```

Set paths and runtime behavior in `scripts/config.py`:

- `input_root` — videos or frame folders;
- `output_root` — generated CSVs, images, and plots;
- `run.video_mode` — `None` for folder-based `backlit`/`laserlit` detection, or an explicit mode;
- frame range and debug-output switches.

## Inputs and outputs

Supported inputs: `.avi`, `.jpg`, `.jpeg`, and `.png`. Place inputs below `data/`, or set `input_root` to another location.

The pipeline writes:

```text
output_root/
├── detect/{csv,images}/
├── track/{csv,images,trajectory_plots,velocity_plots}/
└── anomaly/{csv,images}/
```

Detection CSVs provide frame-level measurements such as wire-tip position, tapering point, weld-pool position, arc/plasma features, droplets, spatters, and anomaly metadata. These outputs feed `predictive_modeling`.

ESAB videos and process data are not included. See the [root README](../README.md) for access restrictions, thesis context, and the final report.
