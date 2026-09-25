'''
    Calculates the Root Relative Squared Error (RRSE) for a prediction model
'''

import numpy as np
import pandas as pd

#file_path = r"/home/samuel/Master-Thesis-ESAB/predictive_modeling/output/arx/wiretip_to_weldpool/arx_val_predictions.csv"
#file_path = r"/home/samuel/Master-Thesis-ESAB/predictive_modeling/output/narx/narx_val_predictions.csv"
#file_path = r"/home/samuel/Master-Thesis-ESAB/predictive_modeling/output/narx2/narx_validation_predictions.csv"
#file_path = r"/home/samuel/Master-Thesis-ESAB/predictive_modeling/output/linear/wiretip_to_weldpool/linear_test_predictions.csv"
file_path = r"/home/samuel/Master-Thesis-ESAB/predictive_modeling/output/narx2/tapering_to_weldpool_specified_inputs_arx_settings/narx2_test_predictions.csv"
#pred_col = "y_pred_freerun"
#true_col = "y_true"
pred_col = "y_pred_freerun"
true_col = "y_true"


def load_prediction_values(file_path, pred_col, true_col):
    df = pd.read_csv(file_path, usecols=[true_col, pred_col])
    values = df[[true_col, pred_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid_mask = np.isfinite(values).all(axis=1)
    values = values[valid_mask]
    if len(values) == 0:
        raise ValueError(f"No finite true/predicted value pairs found in {file_path}")
    return values[:, 0], values[:, 1]


def rrse(file_path, pred_col, true_col):
    y_true, y_pred = load_prediction_values(file_path, pred_col, true_col)
    squared_error = np.sum((y_true - y_pred) ** 2)
    squared_deviation = np.sum((y_true - np.mean(y_true)) ** 2)
    if squared_deviation == 0:
        raise ValueError("RRSE is undefined when all true values are equal.")
    return float(np.sqrt(squared_error / squared_deviation))


def pearson_correlation(file_path, pred_col, true_col):
    y_true, y_pred = load_prediction_values(file_path, pred_col, true_col)
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


if __name__ == "__main__":
    rrse_value = rrse(file_path, pred_col, true_col)
    pearson_value = pearson_correlation(file_path, pred_col, true_col)
    print(f"RRSE: {rrse_value:.4f}")
    if np.isnan(pearson_value):
        print("Pearson r: n/a")
    else:
        print(f"Pearson r: {pearson_value:.4f}")
