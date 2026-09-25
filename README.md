# Master Thesis ESAB Welding Analysis

Automated welding video analysis and predictive modeling for process monitoring.

## Features

- Extract frames from welding videos.
- Detect wire, weld pool, base metal, arc, plasma channel, droplets, and spatters.
- Track droplets and spatters across frames.
- Score frame-level anomalies with deterministic rules.
- Convert HDF5 process measurements to CSV.
- Merge process measurements with image-analysis features.
- Train ARX, NARX, LSTM, neural NARX, XGBoost, and ensemble models.
- Export metrics, predictions, feature importance files, and plots.

## Project Structure

```text
.
+-- README.md
+-- requirements.txt
+-- data/
|   +-- thesis_experiment_matrix.csv
+-- image_analysis_app/
|   +-- README.md
|   +-- scripts/
|       +-- anomaly/
|       +-- pipeline/
|       +-- preprocessing/
|       +-- weldpool/
|       +-- wire/
+-- output/
|   +-- distribution_analysis/
|   +-- image_analysis/
|   +-- predictive_modelling/
+-- predictive_modeling/
    +-- README.md
    +-- scripts/
        +-- config.py
        +-- model_training/
        +-- preprocessing/
        +-- utils/
```

## Requirements

- Python 3.10 or newer.
- Linux, macOS, or WSL.
- Welding video data in `.avi`, `.jpg`, `.jpeg`, or `.png` format.
- Process measurement data in `.h5` format for predictive modeling.
- Enough disk space for generated frames, CSV files, plots, and model outputs.

Main Python packages:

- numpy
- pandas
- scipy
- scikit-learn
- xgboost
- h5py
- matplotlib
- torch
- tqdm
- opencv-python
- colorama

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the same virtual environment for both modules.

## Configuration

Edit these files before a run:

- `image_analysis_app/scripts/config.py`
  - Set `input_root`.
  - Set `output_root`.
  - Set `run.start_img` and `run.end_img` for frame ranges.
  - Set `run.video_mode` to `backlit`, `laserlit`, or `None` for folder-based detection.
  - Set CSV and debug-image save options.

- `predictive_modeling/scripts/config.py`
  - Set `h5_input_path` for raw process measurements.
  - Set `h5_output_path` for converted measurement CSV files.
  - Set `detect_path` for detection CSV files.
  - Set `anomaly_path` for anomaly CSV files.
  - Set merged output folders.
  - Set split fractions, target column, history columns, and context columns.

## Usage
Below are examples but do refer into each README for more information.

Run image analysis:

```bash
cd image_analysis_app
python -m scripts.pipeline.main
```

Run predictive preprocessing:

```bash
cd predictive_modeling
python scripts/preprocessing/preprocessing.py
```

Run model training examples:

```bash
cd predictive_modeling
python scripts/model_training/statistical/arx.py
python scripts/model_training/statistical/narx2.py
python scripts/model_training/dl/nn_narx.py
python scripts/model_training/ml/parallel_ensemble.py
python scripts/model_training/ml/xgb.py
```

Run analysis utilities:

```bash
cd predictive_modeling
python scripts/utils/correlation.py --input-dir data/merged_clipped --output-dir output/correlation
python scripts/utils/plot.py
```

## Inputs

Image-analysis inputs:

- `.avi` videos.
- Frame folders with `.jpg`, `.jpeg`, or `.png` files.
- Folder names containing `backlit` or `laserlit` for automatic mode selection.

Predictive-modeling inputs:

- HDF5 measurement files in the configured `h5_input_path`.
- Detection CSV files from `output/detect/csv`.
- Anomaly CSV files from `output/anomaly/csv`.
- Merged training CSV files in `predictive_modeling/data/merged_clipped` or `predictive_modeling/data/merged_clipped_norm`.

Common columns used by predictive models:

- `voltage`
- `current`
- `delta_voltage`
- `delta_current`
- `wfs`
- `ctdw`
- `wiretip_to_weldpool_dist`
- `tapering_to_weldpool_dist`

## Outputs

Image-analysis outputs:

- `output/detect/csv`
- `output/detect/images`
- `output/track/csv`
- `output/track/images`
- `output/track/trajectory_plots`
- `output/track/velocity_plots`
- `output/anomaly/csv`
- `output/anomaly/images`

Predictive-modeling outputs:

- `output/measurements`, converted measurement CSV files.
- `predictive_modeling/data/merged_org`, merged raw feature CSV files.
- `predictive_modeling/data/merged_clipped`, clipped feature CSV files.
- `predictive_modeling/data/merged_clipped_norm`, normalized feature CSV files.
- `predictive_modeling/output`, local model outputs from some scripts.
- `output/predictive_modelling`, evaluation plots and model summary files.

Model output files include:

- prediction CSV files.
- metrics JSON or CSV files.
- split manifests.
- grid-search logs.
- feature-importance CSV and PNG files.
- training-history CSV and PNG files.

## Methodology

1. Capture welding videos and process measurements for the same experiments.
2. Extract image features from each frame.
3. Track transient objects across frames.
4. Flag unreliable frames with anomaly rules.
5. Convert HDF5 measurement data to CSV.
6. Align measurement samples with frame-level image features.
7. Repair short anomaly gaps through interpolation.
8. Build merged datasets with raw, clipped, and normalized variants.
9. Split data into train, validation, and test sets by sequence.
10. Train statistical, machine-learning, and deep-learning models.
11. Compare models with RMSE, MAE, R2, fit percentage, and prediction plots.

## Examples

Preprocess data with configured paths:

```bash
cd predictive_modeling
python scripts/preprocessing/preprocessing.py
```

Preprocess with custom merged output folders:

```bash
cd predictive_modeling
python scripts/preprocessing/preprocessing.py \
  --output-path-org data/merged_org \
  --output-path-clipped data/merged_clipped \
  --output-path-clipped-norm data/merged_clipped_norm
```

Train ARX with a custom output folder:

```bash
cd predictive_modeling
python scripts/model_training/statistical/arx.py --output_dir output/arx/wiretip_to_weldpool
```

Create correlation plots:

```bash
cd predictive_modeling
python scripts/utils/correlation.py \
  --input-dir data/merged_clipped \
  --output-dir output/correlation
```

## Authors

- Dharun
- Samuel

## Acknowledgement

Thanks to the ESAB Process Control R&D team for guidance, domain knowledge, data access, and project support.
