"""Plot prediction files and summarize model performance across windows."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-plot-utils"))

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required to run this plotting script. Install the "
        "predictive_modeling requirements first, for example: "
        "python -m pip install -r predictive_modeling/requirements.txt"
    ) from exc

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

source_data_file_path = PROJECT_ROOT / "data" / "merged_clipped_norm" / "fr_5m_10mm.csv"
source_title = "Source Data: Wiretip to Weldpool Distance, Voltage, and Current"
source_xlabel = "Time Steps"
source_ylabel = "Voltage (V), Current (A)"
source_plot_labels = ['wiretip_to_weldpool_dist', 'voltage', 'current']
source_start_index = 16000
source_end_index = 17000
source_normalize = True

narmax_wiretip_file_path = PROJECT_ROOT / "output" / "narmax" / "wiretip_to_weldpool" / "narmax_validation_predictions.csv"
narmax_wiretip_train_file_path = PROJECT_ROOT / "data" / "fr_5m_20mm.csv"
narmax_title = "NARMAX Model Predictions vs True Values"
narmax_xlabel = "Time Steps"
narmax_ylabel = "Distance (mm)"
narmax_plot_labels = ['wiretip_to_weldpool_dist','prediction']
narmax_start_index = None
narmax_end_index = None
narmax_normalize = False

arx_wiretip_file_path = PROJECT_ROOT / "output" / "arx" / "wiretip_to_weldpool" / "arx_test_predictions.csv"
arx_wiretip_specified_file_path = PROJECT_ROOT / "output" / "arx" / "wiretip_to_weldpool_specified_inputs" / "arx_test_predictions.csv"
arx_tapering_file_path = PROJECT_ROOT / "output" / "arx" / "tapering_to_weldpool" / "arx_val_predictions.csv"
arx_tapering_raw_file_path = PROJECT_ROOT / "output" / "arx" / "tapering_to_weldpool_raw_electrical" / "arx_test_predictions.csv"
arx_tapering_specified_file_path = PROJECT_ROOT / "output" / "arx" / "tapering_to_weldpool_specified_inputs" / "arx_test_predictions.csv"
arx_title = "ARX Model Predictions vs True Values"
arx_xlabel = "Time Steps"
arx_ylabel = "Distance (mm)"
arx_plot_labels = ['y_true','y_pred_freerun']#, 'voltage', 'current']
arx_start_index = None
arx_end_index = None
arx_use_best_window = True
arx_best_window_size = 500
arx_best_window_metric = "pearson"
arx_normalize = False

narx_wiretip_file_path = PROJECT_ROOT / "output" / "narx2" / "tapering_to_weldpool_specified_inputs_arx_settings" / "narx2_test_predictions.csv"
narx_tapering_file_path = PROJECT_ROOT / "output" / "narx2" / "wiretip_to_weldpool_specified_inputs_arx_settings" / "narx2_test_predictions.csv"
narx_title = "NARX Model Predictions vs True Values"
narx_xlabel = "Time Steps"
narx_ylabel = ""
narx_plot_labels = ['y_true','y_pred_freerun']#, 'Voltage', 'Current']
narx_use_best_window = False
narx_best_window_size = 500
narx_start_index = 110000
narx_end_index = narx_start_index + narx_best_window_size
narx_best_window_metric = "pearson"
narx_normalize = False

linar_wiretip_file_path = PROJECT_ROOT / "output" / "linear" / "wiretip_to_weldpool" / "linear_test_predictions_free_run.csv"
linar_title = "Linear Model Predictions vs True Values"
linar_xlabel = "Time Steps"
linar_ylabel = "Distance (mm)"
linar_plot_labels = ['Actual','Predicted']#, 'Voltage', 'Current']
linar_start_index = 1000
linar_end_index = 2000
linar_normalize = False

lstm_wiretip_file_path = PROJECT_ROOT / "output" / "lstm" / "wiretip_to_weldpool" / "lstm_val_predictions.csv"
lstm_training_history_file_path = PROJECT_ROOT / "output" / "lstm" / "wiretip_to_weldpool" / "lstm_training_history.csv"
lstm_title = "LSTM Model Predictions vs True Values"
lstm_xlabel = "Time Steps"
lstm_ylabel = "Distance (mm)"
lstm_plot_labels = ['y_true','y_pred_free_run']#, 'Voltage', 'Current']
lstm_start_index = 10000
lstm_end_index = 15000
lstm_normalize = False

nn_narx_wiretip_file_path = PROJECT_ROOT / "output" / "nn_narx" / "nn_narx_test_predictions.csv"
nn_narx_tapering_file_path = PROJECT_ROOT / "output" / "nn_narx_tapering_to_weldpool" / "nn_narx_test_predictions.csv"
nn_narx_title = "NN NARX Model Predictions vs True Values"
nn_narx_xlabel = "Time Steps"
nn_narx_ylabel = "Distance (mm)"
nn_narx_plot_labels = ['y_true','y_pred_free_run']
nn_narx_start_index = None
nn_narx_end_index = None
nn_narx_normalize = False

measurement_only_file_path = PROJECT_ROOT / "output" / "codex" / "wiretip_to_weldpool_3" / "test_predictions.csv"
measurement_only_title = "Random Forest Predictions vs True Values"
measurement_only_xlabel = "Time Steps"
measurement_only_ylabel = "Distance (mm)"
measurement_only_plot_labels = ['y_true','y_pred']
measurement_only_start_index = None
measurement_only_end_index = None
measurement_only_normalize = False
measurement_only_model_name = "measurement_only_estimator"

parallel_ensemble_output_dir = PROJECT_ROOT / "output" / "parallel_ensemble_new"
parallel_ensemble_tapering_output_dir = PROJECT_ROOT / "output" / "parallel_ensemble_tapering_to_weldpool"
parallel_ensemble_title = "Extra Trees Predictions vs True Values"
parallel_ensemble_xlabel = "Time Steps"
parallel_ensemble_ylabel = "Distance (mm)"
parallel_ensemble_plot_labels = ['y_true','y_pred']
parallel_ensemble_start_index = 5500
parallel_ensemble_end_index = 6000
parallel_ensemble_use_best_window = False
parallel_ensemble_best_window_size = 500
parallel_ensemble_best_window_metric = "rmse"
parallel_ensemble_normalize = False

def find_prediction_pairs(plot_labels):
    true_labels = [label for label in plot_labels if label.lower() in {"actual", "true", "y_true"}]
    if not true_labels:
        return []

    true_label = true_labels[0]
    pred_labels = [
        label
        for label in plot_labels
        if label != true_label and ("pred" in label.lower() or label.lower() == "yhat")
    ]
    return [(true_label, pred_label) for pred_label in pred_labels]

def pearson_correlation(y_true, y_pred):
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    if len(y_true) < 2 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return np.nan, len(y_true)
    return float(np.corrcoef(y_true, y_pred)[0, 1]), len(y_true)

def normalize_window_metric(metric: str):
    metric = str(metric).strip().lower().replace("-", "_")
    aliases = {
        "corr": "pearson",
        "correlation": "pearson",
        "pearson_r": "pearson",
        "mean_absolute_error": "mae",
        "mean_squared_error": "mse",
        "root_mean_squared_error": "rmse",
    }
    return aliases.get(metric, metric)

def prediction_labels(plot_labels):
    if len(plot_labels) == 2:
        return plot_labels

    prediction_pairs = find_prediction_pairs(plot_labels)
    if len(prediction_pairs) != 1:
        raise ValueError(
            "Prediction window selection requires exactly two labels, or labels "
            "containing one true/actual column and one prediction column."
        )
    return prediction_pairs[0]

def display_plot_label(label):
    label_map = {
        "y_true": "True",
        "y_pred": "Estimated",
        "y_pred_free_run": "Estimated",
        "y_pred_freerun": "Estimated",
        "prediction": "Estimated",
        "predicted": "Estimated",
    }
    return label_map.get(str(label).lower(), label)

def elapsed_plot_time(data, start_index, end_index):
    if "meas_time" in data.columns:
        x_values = pd.to_numeric(data["meas_time"], errors="coerce").to_numpy()
        xlabel = "Time (s)"
    elif "time" in data.columns:
        x_values = pd.to_numeric(data["time"], errors="coerce").to_numpy()
        xlabel = "Time"
    elif "sample_index" in data.columns:
        x_values = pd.to_numeric(data["sample_index"], errors="coerce").to_numpy()
        xlabel = "Sample Index"
    else:
        x_values = np.arange(len(data), dtype=float)
        xlabel = "Time Steps"

    x_window = x_values[start_index:end_index].astype(float)
    finite_x = x_window[np.isfinite(x_window)]
    if len(finite_x) > 0:
        x_window = x_window - finite_x[0]
    else:
        x_window = np.arange(len(x_window), dtype=float)
    return x_window, xlabel

def prediction_window_arrays(data, plot_labels, sequence_length: int):
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if len(data) < sequence_length:
        raise ValueError(
            f"Cannot find a {sequence_length}-row window in data with only {len(data)} rows"
        )

    true_label, pred_label = prediction_labels(plot_labels)
    missing_labels = [label for label in (true_label, pred_label) if label not in data.columns]
    if missing_labels:
        available_labels = ", ".join(data.columns)
        raise ValueError(
            f"Missing requested columns {missing_labels}. Available columns are: {available_labels}"
        )

    y_true = pd.to_numeric(data[true_label], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(data[pred_label], errors="coerce").to_numpy()
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return true_label, pred_label, y_true, y_pred, valid_mask

def rolling_sum(values, sequence_length: int):
    return np.convolve(values, np.ones(sequence_length, dtype=float), mode="valid")

def score_prediction_windows(y_true, y_pred, valid_mask, sequence_length: int, metric: str):
    metric = normalize_window_metric(metric)
    valid_counts = rolling_sum(valid_mask.astype(float), sequence_length)
    full_windows = valid_counts == sequence_length
    if not np.any(full_windows):
        raise ValueError(
            f"No contiguous {sequence_length}-row window has finite true/predicted values"
        )

    if metric in {"rmse", "mse", "mae"}:
        errors = y_true - y_pred
        if metric in {"rmse", "mse"}:
            sums = rolling_sum(np.where(valid_mask, errors ** 2, 0.0), sequence_length)
            scores = np.full(len(sums), np.inf, dtype=float)
            scores[full_windows] = sums[full_windows] / sequence_length
            if metric == "rmse":
                scores[full_windows] = np.sqrt(scores[full_windows])
            return scores, False

        sums = rolling_sum(np.where(valid_mask, np.abs(errors), 0.0), sequence_length)
        scores = np.full(len(sums), np.inf, dtype=float)
        scores[full_windows] = sums[full_windows] / sequence_length
        return scores, False

    if metric == "pearson":
        x = np.where(valid_mask, y_true, 0.0)
        y = np.where(valid_mask, y_pred, 0.0)
        sum_x = rolling_sum(x, sequence_length)
        sum_y = rolling_sum(y, sequence_length)
        sum_x2 = rolling_sum(x * x, sequence_length)
        sum_y2 = rolling_sum(y * y, sequence_length)
        sum_xy = rolling_sum(x * y, sequence_length)

        numerator = sequence_length * sum_xy - sum_x * sum_y
        variance_product = (
            (sequence_length * sum_x2 - sum_x ** 2)
            * (sequence_length * sum_y2 - sum_y ** 2)
        )
        denominator = np.sqrt(np.maximum(variance_product, 0.0))
        scores = np.full(len(sum_xy), np.nan, dtype=float)
        finite_scores = full_windows & (denominator > 0)
        scores[finite_scores] = numerator[finite_scores] / denominator[finite_scores]
        if not np.any(np.isfinite(scores)):
            raise ValueError(
                f"No contiguous {sequence_length}-row window has a finite Pearson correlation"
            )
        return scores, True

    raise ValueError("metric must be one of: pearson, rmse, mse, mae")

def find_best_performing_window(data, plot_labels, sequence_length: int, metric: str = "pearson"):
    metric = normalize_window_metric(metric)
    true_label, pred_label, y_true, y_pred, valid_mask = prediction_window_arrays(
        data,
        plot_labels,
        sequence_length,
    )
    scores, higher_is_better = score_prediction_windows(
        y_true,
        y_pred,
        valid_mask,
        sequence_length,
        metric,
    )

    start_index = int(np.nanargmax(scores) if higher_is_better else np.nanargmin(scores))
    end_index = start_index + sequence_length

    y_true_window = y_true[start_index:end_index]
    y_pred_window = y_pred[start_index:end_index]
    errors = y_true_window - y_pred_window
    mae = float(np.mean(np.abs(y_true_window - y_pred_window)))
    correlation, n_samples = pearson_correlation(y_true_window, y_pred_window)
    return {
        "start_index": start_index,
        "end_index": end_index,
        "metric": metric,
        "metric_value": float(scores[start_index]),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": mae,
        "correlation": correlation,
        "n_samples": n_samples,
        "true_label": true_label,
        "pred_label": pred_label,
    }

def find_best_prediction_window(data, plot_labels, window_size: int, metric: str = "rmse"):
    return find_best_performing_window(data, plot_labels, window_size, metric)

def select_plot_window(
    data,
    plot_labels,
    start_index,
    end_index,
    best_window_size,
    title,
    best_window_metric: str = "rmse",
):
    if best_window_size is None:
        return start_index, end_index

    window = find_best_prediction_window(
        data,
        plot_labels,
        int(best_window_size),
        best_window_metric,
    )
    correlation = window["correlation"]
    correlation_text = "n/a" if np.isnan(correlation) else f"{correlation:.3f}"
    print(
        f"{title}: best {best_window_size}-point window by {window['metric']} is "
        f"[{window['start_index']}:{window['end_index']}] "
        f"{window['metric']}={window['metric_value']:.4f}, "
        f"RMSE={window['rmse']:.4f}, MAE={window['mae']:.4f}, "
        f"Pearson r={correlation_text} (n={window['n_samples']})"
    )
    return window["start_index"], window["end_index"]

def load_prediction_pair(
    file_path,
    plot_labels,
    normalize: bool = False,
    row_filter: dict = None,
):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Could not find input CSV: {file_path}")

    data = pd.read_csv(file_path)
    if row_filter:
        for column, value in row_filter.items():
            if column not in data.columns:
                available_labels = ", ".join(data.columns)
                raise ValueError(
                    f"{file_path} is missing filter column '{column}'. "
                    f"Available columns are: {available_labels}"
                )
            data = data[data[column] == value]
        if data.empty:
            filters = ", ".join(f"{column}={value!r}" for column, value in row_filter.items())
            raise ValueError(f"No rows found in {file_path} matching filters: {filters}")

    return load_prediction_pair_from_frame(data, file_path, plot_labels, normalize)

def load_prediction_pair_from_frame(
    data,
    file_path,
    plot_labels,
    normalize: bool = False,
):
    true_label, pred_label = prediction_labels(plot_labels)

    missing_labels = [label for label in (true_label, pred_label) if label not in data.columns]
    if missing_labels:
        available_labels = ", ".join(data.columns)
        raise ValueError(
            f"{file_path} is missing requested columns {missing_labels}. "
            f"Available columns are: {available_labels}"
        )

    y_true = pd.to_numeric(data[true_label], errors="coerce").to_numpy()
    y_pred = pd.to_numeric(data[pred_label], errors="coerce").to_numpy()
    valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    if len(y_true) == 0:
        raise ValueError(f"No finite true/predicted value pairs found in {file_path}")

    if normalize:
        data_min = float(min(np.nanmin(y_true), np.nanmin(y_pred)))
        data_range = float(max(np.nanmax(y_true), np.nanmax(y_pred)) - data_min)
        if data_range == 0:
            y_true = np.zeros_like(y_true, dtype=float)
            y_pred = np.zeros_like(y_pred, dtype=float)
        else:
            y_true = (y_true - data_min) / data_range
            y_pred = (y_pred - data_min) / data_range

    return true_label, pred_label, y_true, y_pred

def axis_limits_for_prediction_pairs(
    file_paths,
    plot_labels,
    normalize: bool = False,
    row_filter: dict = None,
):
    values = []
    for file_path in file_paths:
        _, _, y_true, y_pred = load_prediction_pair(file_path, plot_labels, normalize, row_filter)
        values.extend([y_true, y_pred])

    combined = np.concatenate(values)
    axis_min = float(np.nanmin(combined))
    axis_max = float(np.nanmax(combined))
    padding = max((axis_max - axis_min) * 0.04, 0.05)
    return axis_min - padding, axis_max + padding

def scatter(
    file_path,
    plot_labels,
    title: str,
    xlabel: str = "True Values",
    ylabel: str = "Predicted Values",
    start_index: int = None,
    end_index: int = None,
    normalize: bool = False,
    row_filter: dict = None,
    axis_limits: tuple[float, float] = None,
    best_window_size: int = None,
    best_window_metric: str = "rmse",
):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Could not find input CSV: {file_path}")

    data = pd.read_csv(file_path)
    if row_filter:
        for column, value in row_filter.items():
            if column not in data.columns:
                available_labels = ", ".join(data.columns)
                raise ValueError(
                    f"{file_path} is missing filter column '{column}'. "
                    f"Available columns are: {available_labels}"
                )
            data = data[data[column] == value]
        if data.empty:
            filters = ", ".join(f"{column}={value!r}" for column, value in row_filter.items())
            raise ValueError(f"No rows found in {file_path} matching filters: {filters}")

    start_index, end_index = select_plot_window(
        data,
        plot_labels,
        start_index,
        end_index,
        best_window_size,
        title,
        best_window_metric,
    )

    true_label, pred_label = prediction_labels(plot_labels)
    windowed_data = data.iloc[start_index:end_index]
    _, _, y_true, y_pred = load_prediction_pair_from_frame(
        windowed_data,
        file_path,
        (true_label, pred_label),
        normalize,
    )

    if axis_limits is None:
        axis_min = float(min(np.nanmin(y_true), np.nanmin(y_pred)))
        axis_max = float(max(np.nanmax(y_true), np.nanmax(y_pred)))
        padding = max((axis_max - axis_min) * 0.04, 0.05)
        axis_min -= padding
        axis_max += padding
    else:
        axis_min, axis_max = axis_limits

    correlation, n_samples = pearson_correlation(y_true, y_pred)
    if np.isnan(correlation):
        pearson_text = f"Pearson r({true_label}, {pred_label}) = n/a (n={n_samples})"
    else:
        pearson_text = f"Pearson r({true_label}, {pred_label}) = {correlation:.3f} (n={n_samples})"
    print(pearson_text)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.45, s=12, linewidths=0, color="tab:blue")
    ax.plot([axis_min, axis_max], [axis_min, axis_max], linestyle="--", color="black", linewidth=1, label="1:1")
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{title}")#\n{pearson_text}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    return fig

def plot(
    file_path,
    plot_labels,
    title: str,
    xlabel: str,
    ylabel: str,
    start_index: int = None,
    end_index: int = None,
    normalize: bool = False,
    row_filter: dict = None,
    best_window_size: int = None,
    best_window_metric: str = "rmse",
):
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Could not find input CSV: {file_path}")
    data = pd.read_csv(file_path)
    if row_filter:
        for column, value in row_filter.items():
            if column not in data.columns:
                available_labels = ", ".join(data.columns)
                raise ValueError(
                    f"{file_path} is missing filter column '{column}'. "
                    f"Available columns are: {available_labels}"
                )
            data = data[data[column] == value]
        if data.empty:
            filters = ", ".join(f"{column}={value!r}" for column, value in row_filter.items())
            raise ValueError(f"No rows found in {file_path} matching filters: {filters}")
    missing_labels = [label for label in plot_labels if label not in data.columns]
    if missing_labels:
        available_labels = ", ".join(data.columns)
        raise ValueError(
            f"{file_path} is missing requested columns {missing_labels}. "
            f"Available columns are: {available_labels}"
        )
    start_index, end_index = select_plot_window(
        data,
        plot_labels,
        start_index,
        end_index,
        best_window_size,
        title,
        best_window_metric,
    )
    time, plot_xlabel = elapsed_plot_time(data, start_index, end_index)
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_columns = {}
    for label in plot_labels:
        curr_data = pd.to_numeric(data[label], errors="coerce").to_numpy()
        numeric_columns[label] = curr_data
        if normalize:
            data_min = np.nanmin(curr_data)
            data_range = np.nanmax(curr_data) - data_min
            curr_data = np.zeros_like(curr_data, dtype=float) if data_range == 0 else (curr_data - data_min) / data_range
        ax.plot(
            time,
            curr_data[start_index:end_index],
            label=display_plot_label(label),
        )

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(plot_xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    ax.grid()

    if len(plot_labels) != 2:
         print(f"Skipping Pearson correlation since there are {len(plot_labels)} plot labels (expected 2 for correlation)")
         return fig
    
    pearson_lines = []
    true_label, pred_label = plot_labels
    y_true = numeric_columns[true_label][start_index:end_index]
    y_pred = numeric_columns[pred_label][start_index:end_index]
    correlation, n_samples = pearson_correlation(y_true, y_pred)
    if np.isnan(correlation):
        pearson_text = f"Pearson r({true_label}, {pred_label}) = n/a (n={n_samples})"
    else:
        pearson_text = f"Pearson r({true_label}, {pred_label}) = {correlation:.3f} (n={n_samples})"
    pearson_lines.append(pearson_text)
    print(pearson_text)

    ax.set_title(title)# if not pearson_lines else f"{title}\n" + "; ".join(pearson_lines))

    return fig

def plot_prediction_splits(
    output_dirs,
    prediction_files,
    plot_labels,
    title,
    xlabel,
    ylabel,
    start_index=None,
    end_index=None,
    normalize=False,
    best_window_size=None,
    best_window_metric="rmse",
):
    for output_dir in output_dirs:
        for split_name, prediction_filename in prediction_files.items():
            prediction_file_path = output_dir / prediction_filename
            if not prediction_file_path.exists():
                print(f"Skipping {title} {split_name}: missing {prediction_file_path}")
                continue

            split_title = f"{title} ({split_name.title()})"
            prediction_stem = prediction_file_path.stem
            prediction_fig = plot(
                prediction_file_path,
                plot_labels,
                split_title,
                xlabel,
                ylabel,
                start_index,
                end_index,
                normalize,
                best_window_size=best_window_size,
                best_window_metric=best_window_metric,
            )
            prediction_fig.savefig(
                output_dir / f"{prediction_stem}.png",
                dpi=300,
                bbox_inches='tight',
            )
            plt.close(prediction_fig)

            scatter_fig = scatter(
                prediction_file_path,
                plot_labels,
                split_title,
                "True Distance (mm)",
                "Predicted Distance (mm)",
                start_index,
                end_index,
                normalize,
                best_window_size=best_window_size,
                best_window_metric=best_window_metric,
            )
            scatter_fig.savefig(
                output_dir / f"{prediction_stem}_scatter.png",
                dpi=300,
                bbox_inches='tight',
            )
            plt.close(scatter_fig)

def source():
    print("Plotting source data...")
    source_fig = plot(source_data_file_path, source_plot_labels, source_title, source_xlabel, source_ylabel, source_start_index, source_end_index, source_normalize)
    source_fig.savefig(PROJECT_ROOT / "output" / "source" / "voltage_current.png", dpi=300, bbox_inches='tight')
    plt.close(source_fig)

def narmax():
    print("Plotting NARMAX model predictions...")
    narmax_fig = plot(narmax_wiretip_file_path, narmax_plot_labels, narmax_title, narmax_xlabel, narmax_ylabel, narmax_start_index, narmax_end_index, narmax_normalize)
    narmax_fig.savefig(PROJECT_ROOT / "output" / "narmax" / "wiretip_to_weldpool" / "validation_predictions.png", dpi=300, bbox_inches='tight')
    plt.close(narmax_fig)
    narmax_scatter_fig = scatter(
        narmax_wiretip_file_path,
        narmax_plot_labels,
        narmax_title,
        "True Distance (mm)",
        "Predicted Distance (mm)",
        narmax_start_index,
        narmax_end_index,
    )
    narmax_scatter_fig.savefig(PROJECT_ROOT / "output" / "narmax" / "wiretip_to_weldpool" / "validation_predictions_scatter.png", dpi=300, bbox_inches='tight')
    plt.close(narmax_scatter_fig)

def arx():
    print("Plotting ARX model predictions...")
    plot_prediction_splits(
        (
            arx_wiretip_file_path.parent,
            arx_wiretip_specified_file_path.parent,
            arx_tapering_file_path.parent,
            arx_tapering_raw_file_path.parent,
            arx_tapering_specified_file_path.parent,
        ),
        {
            "train": "arx_train_predictions.csv",
            "validation": "arx_val_predictions.csv",
            "test": "arx_test_predictions.csv",
        },
        arx_plot_labels,
        arx_title,
        arx_xlabel,
        arx_ylabel,
        arx_start_index,
        arx_end_index,
        arx_normalize,
        best_window_size=arx_best_window_size if arx_use_best_window else None,
        best_window_metric=arx_best_window_metric,
    )

def narx():
    print("Plotting NARX model predictions...")
    plot_prediction_splits(
        (narx_wiretip_file_path.parent,),
        {
            "train": "narx2_train_predictions.csv",
            "validation": "narx2_validation_predictions.csv",
            "test": "narx2_test_predictions.csv",
        },
        narx_plot_labels,
        narx_title,
        narx_xlabel,
        narx_ylabel,
        narx_start_index,
        narx_end_index,
        narx_normalize,
        best_window_size=narx_best_window_size if narx_use_best_window else None,
        best_window_metric=narx_best_window_metric,
    )
    plot_prediction_splits(
        (narx_tapering_file_path.parent,),
        {
            "train": "narx2_train_predictions.csv",
            "validation": "narx2_validation_predictions.csv",
            "test": "narx2_test_predictions.csv",
        },
        narx_plot_labels,
        narx_title,
        narx_xlabel,
        narx_ylabel,
        narx_start_index,
        narx_end_index,
        narx_normalize,
        best_window_size=narx_best_window_size if narx_use_best_window else None,
        best_window_metric=narx_best_window_metric,
    )

def linear():
    print("Plotting linear model predictions...")
    linar_fig = plot(linar_wiretip_file_path, linar_plot_labels, linar_title, linar_xlabel, linar_ylabel, linar_start_index, linar_end_index, linar_normalize)
    linar_fig.savefig(PROJECT_ROOT / "output" / "linear" / "wiretip_to_weldpool" / "linear_test_predictions.png", dpi=300, bbox_inches='tight')
    plt.close(linar_fig)
    linar_scatter_fig = scatter(
        linar_wiretip_file_path,
        linar_plot_labels,
        linar_title,
        "True Distance (mm)",
        "Predicted Distance (mm)",
        linar_start_index,
        linar_end_index,
    )
    linar_scatter_fig.savefig(PROJECT_ROOT / "output" / "linear" / "wiretip_to_weldpool" / "linear_test_predictions_scatter.png", dpi=300, bbox_inches='tight')
    plt.close(linar_scatter_fig)

def lstm():
    print("Plotting LSTM model predictions...")
    lstm_fig = plot(lstm_wiretip_file_path, lstm_plot_labels, lstm_title, lstm_xlabel, lstm_ylabel, lstm_start_index, lstm_end_index, lstm_normalize)
    lstm_fig.savefig(PROJECT_ROOT / "output" / "lstm" / "wiretip_to_weldpool" / "lstm_val_predictions.png", dpi=300, bbox_inches='tight')
    plt.close(lstm_fig)
    lstm_scatter_fig = scatter(
        lstm_wiretip_file_path,
        lstm_plot_labels,
        lstm_title,
        "True Distance (mm)",
        "Predicted Distance (mm)",
        lstm_start_index,
        lstm_end_index,
    )
    lstm_scatter_fig.savefig(PROJECT_ROOT / "output" / "lstm" / "wiretip_to_weldpool" / "lstm_val_predictions_scatter.png", dpi=300, bbox_inches='tight')
    plt.close(lstm_scatter_fig)
    lstm_fig = plot(lstm_training_history_file_path, ["train_total_loss_scaled", "validation_free_run_NRMSE"], "LSTM Training History", "Epochs", "Loss", None, None, False)
    lstm_fig.savefig(PROJECT_ROOT / "output" / "lstm" / "wiretip_to_weldpool" / "lstm_training_history.png", dpi=300, bbox_inches='tight')
    plt.close(lstm_fig)

def nn_narx():
    print("Plotting NN NARX model predictions...")
    plot_prediction_splits(
        (nn_narx_wiretip_file_path.parent, nn_narx_tapering_file_path.parent),
        {
            "train": "nn_narx_train_predictions.csv",
            "validation": "nn_narx_validation_predictions.csv",
            "test": "nn_narx_test_predictions.csv",
        },
        nn_narx_plot_labels,
        nn_narx_title,
        nn_narx_xlabel,
        nn_narx_ylabel,
        nn_narx_start_index,
        nn_narx_end_index,
        nn_narx_normalize,
    )

def measurement_only():
    print("Plotting Random Forest predictions...")
    measurement_only_fig = plot(
        measurement_only_file_path,
        measurement_only_plot_labels,
        measurement_only_title,
        measurement_only_xlabel,
        measurement_only_ylabel,
        measurement_only_start_index,
        measurement_only_end_index,
        measurement_only_normalize,
        row_filter={"model_name": measurement_only_model_name},
    )
    measurement_only_fig.savefig(PROJECT_ROOT / "output" / "codex" / "wiretip_to_weldpool" / "random_forest_predictions.png", dpi=300, bbox_inches='tight')
    plt.close(measurement_only_fig)
    measurement_only_scatter_fig = scatter(
        measurement_only_file_path,
        measurement_only_plot_labels,
        measurement_only_title,
        "True Distance (mm)",
        "Predicted Distance (mm)",
        measurement_only_start_index,
        measurement_only_end_index,
        measurement_only_normalize,
        row_filter={"model_name": measurement_only_model_name},
    )
    measurement_only_scatter_fig.savefig(PROJECT_ROOT / "output" / "codex" / "wiretip_to_weldpool" / "random_forest_predictions_scatter.png", dpi=300, bbox_inches='tight')
    plt.close(measurement_only_scatter_fig)

def parallel_ensemble():
    print("Plotting parallel ensemble predictions...")
    split_names = ("train", "validation", "test")
    for path in (parallel_ensemble_output_dir, parallel_ensemble_tapering_output_dir):
        prediction_file_paths = {
            split_name: path / f"{split_name}_predictions.csv"
            for split_name in split_names
        }
        scatter_axis_limits = axis_limits_for_prediction_pairs(
            prediction_file_paths.values(),
            parallel_ensemble_plot_labels,
            parallel_ensemble_normalize,
        )
        for split_name in split_names:
            predictions_file_path = prediction_file_paths[split_name]
            split_title = f"{parallel_ensemble_title} ({path.name} {split_name})"
            window_start_index = parallel_ensemble_start_index
            window_end_index = parallel_ensemble_end_index
            if parallel_ensemble_use_best_window and parallel_ensemble_best_window_size is not None:
                prediction_data = pd.read_csv(predictions_file_path)
                window_start_index, window_end_index = select_plot_window(
                    prediction_data,
                    parallel_ensemble_plot_labels,
                    parallel_ensemble_start_index,
                    parallel_ensemble_end_index,
                    parallel_ensemble_best_window_size,
                    split_title,
                    parallel_ensemble_best_window_metric,
                )

            parallel_ensemble_fig = plot(
                predictions_file_path,
                parallel_ensemble_plot_labels,
                split_title,
                parallel_ensemble_xlabel,
                parallel_ensemble_ylabel,
                window_start_index,
                window_end_index,
                parallel_ensemble_normalize,
            )
            parallel_ensemble_fig.savefig(
                path / f"{split_name}_predictions.png",
                dpi=300,
                bbox_inches='tight',
            )
            plt.close(parallel_ensemble_fig)
            parallel_ensemble_scatter_fig = scatter(
                predictions_file_path,
                parallel_ensemble_plot_labels,
                split_title,
                "True Distance (mm)",
                "Predicted Distance (mm)",
                window_start_index,
                window_end_index,
                parallel_ensemble_normalize,
                axis_limits=scatter_axis_limits,
            )
            parallel_ensemble_scatter_fig.savefig(
                path / f"{split_name}_predictions_scatter.png",
                dpi=300,
                bbox_inches='tight',
            )
            plt.close(parallel_ensemble_scatter_fig)


#### Main execution
if __name__ == "__main__":
    #source()
    #narmax()
    #arx()
    narx()
    #linear()
    #lstm()
    #nn_narx()
    #measurement_only()
    #parallel_ensemble()  
