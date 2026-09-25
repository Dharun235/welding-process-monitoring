# Welding Process Monitoring

Research code from a master’s thesis on automated welding-process monitoring with synchronized video and process measurements.

The project combines computer-vision feature extraction with time-series modeling. It is organized as two connected stages:

1. **Image analysis** extracts geometric, arc, weld-pool, droplet, spatter, and anomaly features from welding videos.
2. **Predictive modeling** aligns those features with electrical/process measurements and evaluates statistical, machine-learning, and deep-learning models.

**Thesis report:** [MasterThesisReport.pdf](https://github.com/Dharun235/dharun.github.io/blob/main/MasterThesisReport.pdf)

## Repository layout

```text
.
├── image_analysis_app/       # Frame-level detection, tracking, and anomaly scoring
│   └── scripts/droplets_spatters/  # Droplet/spatter detection and tracking
├── predictive_modeling/      # Measurement conversion, feature alignment, and models
├── requirements.txt          # Python dependencies
└── README.md
```

ESAB experimental data, production images, and process measurements are intentionally excluded from this repository and are not reproduced in this README. The linked thesis report contains the authorized research context, analysis, and results.

## Research workflow

```text
video / frames ──> image features ──┐
                                    ├──> aligned datasets ──> model training ──> evaluation
process HDF5 ──> measurement CSV ──┘
```

The implemented workflow includes:

- wire, weld-pool, base-metal, arc, plasma-channel, droplet, and spatter feature extraction;
- deterministic anomaly scoring for unreliable frame-level detections;
- HDF5-to-CSV conversion and time-based measurement/frame alignment;
- interpolation, clipping, normalization, and sequence-aware train/validation/test splitting;
- ARX, NARX, LSTM, neural NARX, XGBoost, and ensemble model scripts;
- metrics, predictions, split manifests, plots, and feature-importance exports.

## Requirements

- Python 3.10+
- OpenCV, NumPy, pandas, SciPy, scikit-learn, XGBoost, PyTorch, h5py, matplotlib, tqdm, and colorama
- Source welding videos or frame folders
- HDF5 process measurements for predictive modeling

## Setup

```bash
git clone https://github.com/Dharun235/welding-process-monitoring.git
cd welding-process-monitoring
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration

Defaults are repository-relative. Edit only the relevant configuration file before running a pipeline:

- `image_analysis_app/scripts/config.py` — input/output roots, frame range, acquisition mode, detector thresholds, and debug output.
- `predictive_modeling/scripts/config.py` — measurement/detection paths, interpolation, target/features, and sequence splitting.

Keep raw data outside version control. The default locations are `data/` for inputs and `output/` for generated artifacts.

## Running the workflow

Run image analysis from its module root:

```bash
cd image_analysis_app
python -m scripts.pipeline.main
```

Prepare aligned predictive-modeling datasets:

```bash
cd predictive_modeling
python scripts/preprocessing/preprocessing.py
```

Train representative models:

```bash
python scripts/model_training/statistical/arx.py
python scripts/model_training/statistical/narx2.py
python scripts/model_training/dl/nn_narx.py
python scripts/model_training/dl/lstm.py
python scripts/model_training/ml/xgb.py
python scripts/model_training/ml/parallel_ensemble.py
```

Useful analysis commands:

```bash
python scripts/utils/correlation.py
python scripts/utils/rrse.py path/to/predictions.csv
```

Most scripts expose `--help` for path and model-specific options. Run commands from the module directory shown above so package imports resolve correctly.

## Outputs

Typical outputs are written below `output/` and `predictive_modeling/data/`:

- detection and tracking CSV files;
- anomaly flags and optional annotated frames;
- converted and aligned measurement datasets;
- raw, clipped, and normalized dataset variants;
- model predictions, metrics, split manifests, plots, and feature importances.

## Scope and reproducibility

This repository contains the thesis implementation and configuration, not the experimental data package. Results depend on ESAB source videos, HDF5 measurements, acquisition setup, and tuned configuration values. The thesis report provides the scientific context, methodology, experiments, and conclusions.

## Authors and acknowledgements

- Dharun Kumar
- Samuel

Developed with support from the ESAB Process Control R&D team, whose guidance, domain knowledge, and data access enabled the project.

## Citation

If you use this code, cite the thesis report linked above and reference this repository:

```text
Dharun Kumar and Samuel. Welding Process Monitoring: Master’s Thesis Code.
https://github.com/Dharun235/welding-process-monitoring
```

## Use and access

This repository is published for portfolio and employer review only. It is not an open-source release. All code, documentation, and related materials are proprietary and may not be copied, modified, redistributed, published, or used without written permission from the rights holders. See [LICENSE](LICENSE). ESAB data and company-sensitive materials remain restricted.
