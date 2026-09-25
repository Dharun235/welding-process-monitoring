# Predictive modeling

Time-series preparation and model comparison stage of the welding-process monitoring thesis. It combines process measurements with image-derived features and predicts weld-process quantities over time.

## Workflow

```text
HDF5 measurements ──> CSV conversion ──┐
                                       ├──> time-aligned datasets ──> models ──> metrics / plots
image features ────> anomaly repair ──┘
```

1. Convert HDF5 measurements to CSV.
2. Match measurement, detection, and anomaly files by experiment.
3. Align signals by measurement/frame timestamps.
4. Interpolate short unreliable feature gaps.
5. Export raw, clipped, and normalized datasets.
6. Split sequences into train, validation, and test partitions.
7. Train and compare model families.

## Model families

| Directory | Models |
| --- | --- |
| `scripts/model_training/statistical/` | ARX and bounded NARX system-identification models |
| `scripts/model_training/ml/` | XGBoost and parallel ensemble models |
| `scripts/model_training/dl/` | LSTM and neural NARX models |
| `scripts/utils/` | Correlation plots, summaries, and evaluation helpers |

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

Use `--help` for script-specific options. Configure shared paths and experiment settings in `scripts/config.py`:

- measurement and image-feature input directories;
- HDF5 conversion output;
- interpolation columns and maximum anomaly-gap length;
- target, history, context, and metadata columns;
- sequence length, split fractions, and random seed.

## Data contract

Expected inputs:

- HDF5 process measurements;
- detection CSVs from `image_analysis_app`;
- anomaly CSVs from `image_analysis_app`.

Important signals include voltage, current, wire-feed speed, contact-tip-to-work distance, and image-derived distances such as `wiretip_to_weldpool_dist`.

## Outputs

```text
output/measurements/                 # HDF5 → CSV
predictive_modeling/data/merged_org/ # aligned raw data
predictive_modeling/data/merged_clipped/
predictive_modeling/data/merged_clipped_norm/
output/predictive_modeling/          # metrics, predictions, plots, manifests
```

ESAB measurements and video-derived data are restricted and are not included. See the [root README](../README.md) for the thesis report and repository-use terms.
