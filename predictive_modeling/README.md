# Predictive Modeling

Time-series preprocessing and model training for welding process prediction.

## Features

- Convert HDF5 process measurements to CSV.
- Merge electrical measurements with image-analysis features.
- Interpolate short anomaly gaps in selected feature columns.
- Create raw, clipped, and normalized merged datasets.
- Split time-series data into train, validation, and test sequences.
- Train ARX and NARX system-identification models.
- Train LSTM and neural NARX deep-learning models.
- Train XGBoost and ensemble machine-learning models.
- Export predictions, metrics, feature importances, split manifests, and plots.

## Project Structure

```text
predictive_modeling/
+-- README.md
+-- __init__.py
+-- scripts/
    +-- config.py
    +-- preprocessing/
    |   +-- h5_to_csv.py
    |   +-- interpolate_anomalies.py
    |   +-- merge.py
    |   +-- preprocessing.py
    |   +-- split.py
    +-- model_training/
    |   +-- dl/
    |   |   +-- lstm.py
    |   |   +-- nn_narx.py
    |   +-- ml/
    |   |   +-- parallel_ensemble.py
    |   |   +-- xgb.py
    |   +-- statistical/
    |       +-- arx.py
    |       +-- narx2.py
    +-- utils/
        +-- correlation.py
        +-- csv_len.py
        +-- plot.py
        +-- rrse.py
```

## Requirements

- Python 3.10 or newer.
- HDF5 process measurement files.
- Detection CSV files from the image-analysis pipeline.
- Anomaly CSV files from the image-analysis pipeline.
- A working C or C++ runtime for XGBoost.

Python packages live in the root `requirements.txt`.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run commands from `predictive_modeling` unless a script says otherwise.

```bash
cd predictive_modeling
```

## Configuration

Edit `scripts/config.py`.

Preprocessing paths:

- `h5_input_path`, raw HDF5 files.
- `h5_output_path`, converted measurement CSV files.
- `input_path`, measurement CSV input folder.
- `detect_path`, detection CSV folder.
- `anomaly_path`, anomaly CSV folder.
- `output_path_org`, merged raw CSV output folder.
- `output_path_clipped`, clipped merged CSV output folder.
- `output_path_clipped_norm`, normalized merged CSV output folder.

Interpolation settings:

- `interpolation_columns`, image features repaired across short anomaly gaps.
- `max_anomaly_length`, maximum gap length for interpolation.
- `upsample`, alignment mode between measurements and frames.

Split settings:

- `target_col`, prediction target.
- `history_cols`, time-varying input signals.
- `context_cols`, process context signals.
- `offline_metadata_cols`, condition metadata.
- `subsequence_length`, sequence length used for splitting.
- `train_fraction`, training share.
- `validation_fraction`, validation share.
- `random_state`, repeatable split seed.

## Usage

Run the full preprocessing flow:

```bash
python scripts/preprocessing/preprocessing.py
```

Run preprocessing with custom output folders:

```bash
python scripts/preprocessing/preprocessing.py \
  --output-path-org data/merged_org \
  --output-path-clipped data/merged_clipped \
  --output-path-clipped-norm data/merged_clipped_norm
```

Train ARX:

```bash
python scripts/model_training/statistical/arx.py \
  --output_dir output/arx/wiretip_to_weldpool
```

Train NARX:

```bash
python scripts/model_training/statistical/narx2.py \
  --output-dir output/narx2/wiretip_to_weldpool
```

Train neural NARX:

```bash
python scripts/model_training/dl/nn_narx.py \
  --data-dir data/merged_clipped
```

Train ensemble model:

```bash
python scripts/model_training/ml/parallel_ensemble.py
```

Train XGBoost sweep:

```bash
python scripts/model_training/ml/xgb.py
```

Create correlation plots:

```bash
python scripts/utils/correlation.py \
  --input-dir data/merged_clipped \
  --output-dir output/correlation
```

Create summary plots:

```bash
python scripts/utils/plot.py
```

## Inputs

Raw inputs:

- `.h5` files from welding experiments.
- `detections_<source>.csv` from image analysis.
- `anomaly_frames_<source>.csv` from image analysis.

Merged dataset inputs:

- `data/merged_org/*.csv`
- `data/merged_clipped/*.csv`
- `data/merged_clipped_norm/*.csv`

Common model columns:

- `voltage`
- `current`
- `delta_voltage`
- `delta_current`
- `wfs`
- `ctdw`
- `wiretip_to_weldpool_dist`
- `tapering_to_weldpool_dist`
- `plasma_attachment_height`
- `plasma_channel_avg_width`

## Outputs

Preprocessing outputs:

- Converted measurement CSV files in `output/measurements`.
- Raw merged CSV files in `data/merged_org`.
- Clipped merged CSV files in `data/merged_clipped`.
- Normalized merged CSV files in `data/merged_clipped_norm`.

Model outputs:

- `arx_coefficients.csv`
- `arx_metrics.json`
- `arx_grid_search.csv`
- `arx_train_predictions.csv`
- `arx_val_predictions.csv`
- `arx_test_predictions.csv`
- `arx_split_manifest.csv`
- `narx2_*_predictions.csv`
- `train_predictions.csv`, `validation_predictions.csv`, and `test_predictions.csv` for ensemble runs.
- `feature_importances.csv`
- `feature_importances_grouped.csv`
- `config_resolved.json`
- plot files in PNG format.

Evaluation outputs:

- RMSE.
- MAE.
- R2.
- fit percentage.
- AIC and BIC for ARX model selection.
- prediction plots.
- scatter plots.
- feature-importance plots.

## Methodology

1. Convert HDF5 measurement files to CSV.
2. Read detection and anomaly CSV files from image analysis.
3. Align process measurements with frame-level visual features.
4. Interpolate short anomaly spans in selected image-feature columns.
5. Compute derived geometric features such as distance from wire tip to weld pool.
6. Create raw, clipped, and normalized dataset versions.
7. Split sequences by time order and welding condition.
8. Train statistical, machine-learning, and deep-learning models.
9. Evaluate one-step and free-run predictions where supported.
10. Compare models with metrics, plots, and feature-importance files.

## Examples

Prepare a default dataset:

```bash
python scripts/preprocessing/preprocessing.py
```

Prepare a downsampled frame-level dataset:

```bash
python scripts/preprocessing/preprocessing.py --upsample false
```

Prepare an upsampled measurement-level dataset:

```bash
python scripts/preprocessing/preprocessing.py --upsample true
```

Train ARX for a specific target:

```bash
python scripts/model_training/statistical/arx.py \
  --target_col wiretip_to_weldpool_dist \
  --output_dir output/arx/wiretip_to_weldpool
```

Run NARX with prediction clipping:

```bash
python scripts/model_training/statistical/narx2.py \
  --enable-prediction-clipping \
  --output-dir output/narx2/wiretip_to_weldpool_clipped
```

Build lagged scatter plots:

```bash
python scripts/utils/correlation.py \
  --input-dir data/merged_clipped \
  --output-dir output/correlation
```

For LSTM and XGBOOST, the code is different from NARX, but it has the complete model training to evaluation pipeline. 

## Authors

- Dharun
- Samuel

## Acknowledgement

Thanks to the ESAB Process Control R&D team for guidance, domain knowledge, data access, and project support.
