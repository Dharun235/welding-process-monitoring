# Predictive modeling

Time-series preparation and model training for welding-process monitoring.

## Workflow

1. Convert HDF5 process measurements to CSV.
2. Match measurements with image-analysis and anomaly CSV files.
3. Align features by time and interpolate short anomaly gaps.
4. Export raw, clipped, and normalized datasets.
5. Split sequences without mixing temporal samples across train, validation, and test sets.
6. Train and compare statistical, machine-learning, and deep-learning models.

## Run

From this directory:

```bash
python scripts/preprocessing/preprocessing.py
python scripts/model_training/statistical/arx.py
python scripts/model_training/statistical/narx2.py
python scripts/model_training/dl/nn_narx.py
python scripts/model_training/dl/lstm.py
python scripts/model_training/ml/xgb.py
python scripts/model_training/ml/parallel_ensemble.py
```

Edit `scripts/config.py` for input locations, interpolation columns, prediction target, input features, sequence length, and split proportions. Each training script also exposes model-specific options through `--help` where available.

## Data contract

Expected inputs:

- HDF5 process measurements under the configured `h5_input_path`;
- detection CSVs from `image_analysis_app`;
- anomaly CSVs from `image_analysis_app`.

Important configured signals include voltage, current, wire-feed speed, contact-tip-to-work distance, and image-derived distances such as `wiretip_to_weldpool_dist`.

## Outputs

- `output/measurements/` — converted measurement CSVs;
- `data/merged_org/` — aligned raw data;
- `data/merged_clipped/` — clipped data;
- `data/merged_clipped_norm/` — normalized data;
- `output/predictive_modeling/` — model metrics, predictions, plots, manifests, and feature importance.

See the [root README](../README.md) for the research context, reproducibility boundary, and thesis report.
