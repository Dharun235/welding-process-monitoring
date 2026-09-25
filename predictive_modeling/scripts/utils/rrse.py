"""Calculate RRSE and Pearson correlation for a prediction CSV."""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

def load_prediction_values(file_path, pred_col, true_col):
    """Load finite target/prediction pairs from a CSV file."""
    df = pd.read_csv(file_path, usecols=[true_col, pred_col])
    values = df[[true_col, pred_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(values).all(axis=1)
    values = values[valid_mask]
    if len(values) == 0:
        raise ValueError(f"No finite true/predicted value pairs found in {file_path}")
    return values[:, 0], values[:, 1]


def rrse(file_path, pred_col, true_col):
    """Calculate root relative squared error for a prediction CSV."""
    y_true, y_pred = load_prediction_values(file_path, pred_col, true_col)
    squared_error = np.sum((y_true - y_pred) ** 2)
    squared_deviation = np.sum((y_true - np.mean(y_true)) ** 2)
    if squared_deviation == 0:
        raise ValueError("RRSE is undefined when all true values are equal.")
    return float(np.sqrt(squared_error / squared_deviation))


def pearson_correlation(file_path, pred_col, true_col):
    """Calculate Pearson correlation, or NaN for constant inputs."""
    y_true, y_pred = load_prediction_values(file_path, pred_col, true_col)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_csv", type=Path)
    parser.add_argument("--prediction-column", default="y_pred_freerun")
    parser.add_argument("--target-column", default="y_true")
    args = parser.parse_args()
    rrse_value = rrse(args.prediction_csv, args.prediction_column, args.target_column)
    pearson_value = pearson_correlation(args.prediction_csv, args.prediction_column, args.target_column)
    print(f"RRSE: {rrse_value:.4f}")
    if np.isnan(pearson_value):
        print("Pearson r: n/a")
    else:
        print(f"Pearson r: {pearson_value:.4f}")
