# Image Analysis App
Frame-level welding feature extraction, tracking, and anomaly scoring.

## Features

- Read `.avi` welding videos and image-frame folders.
- Convert `.avi` videos into temporary frame folders.
- Detect wire geometry from backlit and laserlit images.
- Locate wire edges, wire contour, wire tip, tapering point, and tool transition.
- Locate base metal and weld pool positions.
- Detect arc and plasma-channel features.
- Track frame-level detections across each source.
- Score anomalies from missing values, sudden jumps, and transient events.
- Save CSV files, annotated frames, trajectory plots, and velocity plots.

## Project Structure

```text
image_analysis_app/
+-- README.md
+-- scripts/
    +-- config.py
    +-- anomaly/
    |   +-- anomaly_score.py
    +-- pipeline/
    |   +-- analyze.py
    |   +-- main.py
    |   +-- save_results.py
    +-- preprocessing/
    |   +-- avi_to_frames.py
    |   +-- remove_glare.py
    |   +-- backlit/
    |   |   +-- arc.py
    |   |   +-- wire.py
    |   +-- laserlit/
    |       +-- wire.py
    +-- weldpool/
    |   +-- base_metal.py
    |   +-- weldpool.py
    +-- wire/
        +-- edges.py
        +-- searchspace.py
        +-- solid_molten_transition.py
        +-- tapering_point.py
        +-- wire_contour.py
        +-- wire_tool_transition.py
```

## Requirements

- Python 3.10 or newer.
- OpenCV.
- NumPy.
- pandas.
- colorama.
- Root `requirements.txt`.
- Welding videos in `.avi` format, or frame folders with `.jpg`, `.jpeg`, or `.png` files.
- Input folders named `backlit` or `laserlit` for automatic mode selection.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Edit `image_analysis_app/scripts/config.py`.

Main run settings:

- `input_root`, input video or frame root.
- `output_root`, output folder.
- `run.start_img`, first frame index.
- `run.end_img`, final frame index.
- `run.video_mode`, set `backlit`, `laserlit`, or `None`.
- `run.save_csv`, save detection CSV files.
- `run.save_detect_images_debug`, save annotated detection frames.
- `run.save_track_images_debug`, save annotated tracking frames.
- `run.save_anomaly_frames`, copy frames flagged as anomalous.

Detection settings:

- `wire_width_mm`, known wire width for pixel-to-mm conversion.
- `wire_tool_transition`, tool-transition search settings.
- `wire_searchspace`, search area size and shift.
- `wire_contour`, contour filters.
- `solid_molten_transition`, laserlit transition threshold.
- `weldpool`, weld-pool scan and droplet-removal settings.
- `backlit`, backlit thresholds and geometry settings.
- `laserlit`, laserlit thresholds and geometry settings.

Anomaly settings:

- `min_anomaly_score`, score threshold for anomaly flag.
- `jump_mean_pct_threshold`, jump threshold from prior-window mean.
- `window_size`, previous-frame window size.
- `short_life_max_len`, maximum short event length.
- `jump_weight`, jump score weight.
- `missing_weight`, missing-value score weight.
- `transient_weight`, transient-event score weight.

## Usage

Run the full pipeline from `image_analysis_app`:

```bash
cd image_analysis_app
python -m scripts.pipeline.main
```

Run anomaly scoring on existing detection CSV files:

```bash
cd image_analysis_app
python -m scripts.anomaly.anomaly_score
```

Use `scripts/config.py` for paths and runtime settings before each run.

## Inputs

Supported input sources:

- One `.avi` file.
- A folder with `.avi` files.
- A folder with frame images.
- A recursive input root with several video or frame sources.

Supported frame formats:

- `.jpg`
- `.jpeg`
- `.png`

Mode selection:

- Use a parent folder named `backlit` for backlit processing.
- Use a parent folder named `laserlit` for laserlit processing.
- Set `run.video_mode` to override folder-based mode selection.

## Outputs

The pipeline writes files under `output_root`.

Detection outputs:

- `detect/csv/detections_<source>.csv`
- `detect/images/<source>/`, annotated detection frames.

Tracking outputs:

- `track/csv/tracked_features_<source>.csv`
- `track/images/<source>/`, annotated tracking frames.
- `track/trajectory_plots/trajectory_plot_<source>.png`
- `track/velocity_plots/velocity_plot_<source>.png`

Anomaly outputs:

- `anomaly/csv/anomaly_frames_<source>.csv`
- `anomaly/images/<source>/`, copied anomalous frames.

Common CSV fields:

- `image_name`
- `frame_number`
- `basemetal_y`
- `mm_per_px`
- `wiretip_x`
- `wiretip_y`
- `transition_x`
- `transition_y`
- `tapering_x`
- `tapering_y`
- `weldpool_x`
- `weldpool_y`
- `arc_detected`
- `plasma_attachment_height`
- `plasma_channel_avg_width`
- `tooltip_x1`
- `tooltip_x2`
- `tooltip_y1`
- `tooltip_y2`
- `anomaly_score`
- `anomaly`
- `anomaly_reason`

Dynamic CSV fields:

- `droplet_<n>_x`
- `droplet_<n>_y`
- `droplet_<n>_area`
- `droplet_<n>_diameter_px`
- `spatter_<n>_x`
- `spatter_<n>_y`
- `spatter_<n>_area`
- `spatter_<n>_diameter_px`

## Methodology

1. Scan the input root for videos and frame folders.
2. Convert each `.avi` video to temporary image frames.
3. Select backlit or laserlit processing mode.
4. Preprocess each frame for wire, arc, and plasma detection.
5. Detect wire edges and build a wire search area.
6. Locate wire contour, tool transition, tapering point, wire tip, base metal, and weld pool.
7. Save frame-level detection rows and annotated debug frames.
8. Track detections over time and export trajectory summaries.
9. Score anomaly frames with deterministic rules.
10. Save CSV files and optional debug images for later modeling.

## Examples

Run all frames from configured paths:

```bash
cd image_analysis_app
python -m scripts.pipeline.main
```

Run a frame range:

```python
CONFIG["run"]["start_img"] = 0
CONFIG["run"]["end_img"] = 500
```

Force backlit mode:

```python
CONFIG["run"]["video_mode"] = "backlit"
```

Use folder-based mode detection:

```python
CONFIG["run"]["video_mode"] = None
```

Disable annotated debug frames:

```python
CONFIG["run"]["save_detect_images_debug"] = False
CONFIG["run"]["save_track_images_debug"] = False
CONFIG["run"]["save_anomaly_frames"] = False
```

## Authors

- Dharun
- Samuel

## Acknowledgement

Thanks to the ESAB Process Control R&D team for guidance, domain knowledge, data access, and project support.
