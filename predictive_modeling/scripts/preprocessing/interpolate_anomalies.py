"""Repair short, flagged gaps in selected numeric image features."""

from pathlib import Path

import pandas as pd


def _iter_true_runs(mask: pd.Series):
    """Yield (start_idx, end_idx) for contiguous True runs in a boolean Series."""
    start_idx = None

    for position, is_true in enumerate(mask.tolist()):
        if is_true and start_idx is None:
            start_idx = position
        elif not is_true and start_idx is not None:
            yield start_idx, position - 1
            start_idx = None

    if start_idx is not None:
        yield start_idx, len(mask) - 1


def _interpolate_run_for_column(values: pd.Series, start_idx: int, end_idx: int) -> tuple[pd.Series, bool]:
    """Interpolate one run for one numeric column."""
    prev_idx = start_idx - 1
    next_idx = end_idx + 1

    if prev_idx < 0 or next_idx >= len(values):
        return values, False

    interpolated_values = pd.to_numeric(values, errors="coerce").copy()
    prev_value = interpolated_values.iloc[prev_idx]
    next_value = interpolated_values.iloc[next_idx]

    if pd.isna(prev_value) or pd.isna(next_value):
        return interpolated_values, False

    run_length = end_idx - start_idx + 1
    step = (next_value - prev_value) / (run_length + 1)
    for offset, row_idx in enumerate(range(start_idx, end_idx + 1), start=1):
        interpolated_values.iloc[row_idx] = prev_value + step * offset

    return interpolated_values, True


def interpolate_anomalies_df(df, interpolation_columns, max_anomaly_length):
    """
    Interpolate short anomaly runs in a DataFrame.

    For each column in ``interpolation_columns``, contiguous anomaly segments
    shorter than or equal to ``max_anomaly_length`` are replaced with values
    from a straight line between the closest valid point before the segment and
    the closest valid point after the segment.
    """
    interpolated_df = df.copy()
    max_anomaly_length = int(max_anomaly_length)

    if "anomaly" in interpolated_df.columns:
        anomaly_values = pd.to_numeric(interpolated_df["anomaly"], errors="coerce").fillna(0).astype(int)
        anomaly_mask = anomaly_values.astype(bool)
        repaired_mask = pd.Series(False, index=interpolated_df.index)

        for start_idx, end_idx in _iter_true_runs(anomaly_mask):
            run_length = end_idx - start_idx + 1
            if run_length > max_anomaly_length:
                continue

            repaired_any_column = False
            for column in interpolation_columns:
                if column not in interpolated_df.columns:
                    continue

                interpolated_values, repaired = _interpolate_run_for_column(
                    interpolated_df[column],
                    start_idx,
                    end_idx,
                )
                if repaired:
                    repaired_any_column = True
                else:
                    # Do not keep suspect anomaly values in columns that could
                    # not be repaired for a short run.
                    interpolated_values.iloc[start_idx:end_idx + 1] = float("nan")

                interpolated_df[column] = interpolated_values

            if repaired_any_column:
                repaired_mask.iloc[start_idx:end_idx + 1] = True

        interpolated_df["anomaly"] = anomaly_values
        interpolated_df.loc[repaired_mask, "anomaly"] = 0
        return interpolated_df

    for column in interpolation_columns:
        if column not in interpolated_df.columns:
            continue

        numeric_values = pd.to_numeric(interpolated_df[column], errors="coerce")

        if "anomaly" in interpolated_df.columns:
            anomaly_mask = interpolated_df["anomaly"].fillna(0).astype(bool)
        else:
            anomaly_mask = numeric_values.isna()

        if not anomaly_mask.any():
            continue

        filled_values = numeric_values.copy()

        for start_idx, end_idx in _iter_true_runs(anomaly_mask):
            run_length = end_idx - start_idx + 1
            if run_length > max_anomaly_length:
                continue

            filled_values, _ = _interpolate_run_for_column(filled_values, start_idx, end_idx)

        interpolated_df[column] = filled_values

    return interpolated_df


def interpolate_anomalies(csv_path, interpolation_columns, max_anomaly_length):
    """
    Load a CSV, interpolate short anomaly runs, and return the result.

    ``csv_path`` must point to a CSV file. Longer anomaly segments, segments at
    the start/end of the series, and columns not present in the CSV are left
    unchanged.
    """
    df = pd.read_csv(Path(csv_path))
    return interpolate_anomalies_df(df, interpolation_columns, max_anomaly_length)
