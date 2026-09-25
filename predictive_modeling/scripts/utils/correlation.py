"""Generate exploratory correlation and lagged scatter plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictive_modeling.scripts.config import CONFIG

DEFAULT_INPUT_DIR = Path(CONFIG["preprocessing"]["output_path_clipped"])
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "predictive_modeling" / "correlation"
DEFAULT_RANDOM_STATE = 42
DEFAULT_DPI = 300
DEFAULT_MAX_LAG = 50

PLOT_COLUMNS = {
    "wiretip_to_weldpool_dist": "Wiretip-to-weldpool distance (mm)",
    "tapering_to_weldpool_dist": "Tapering-to-weldpool distance (mm)",
    "voltage": "Voltage (V)",
    "current": "Current (A)",
}

SCATTER_PLOTS = [
    ("wiretip_to_weldpool_dist", "tapering_to_weldpool_dist"),
    ("wiretip_to_weldpool_dist", "voltage"),
    ("wiretip_to_weldpool_dist", "current"),
    ("tapering_to_weldpool_dist", "voltage"),
    ("tapering_to_weldpool_dist", "current"),
]
LAGGABLE_COLUMNS = ("voltage", "current")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create scatterplots for wiretip-to-weldpool distance and "
            "tapering-to-weldpool distance against each other, voltage, and current."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing merged predictive-modeling CSV files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where the plots will be saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of points to render per scatterplot after "
            "dropping NaNs. By default, all available points are plotted."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed used for scatterplot sampling. Default: {DEFAULT_RANDOM_STATE}",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Saved figure resolution. Default: {DEFAULT_DPI}",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=DEFAULT_MAX_LAG,
        help=(
            "Maximum lag to generate for voltage/current overview scatterplots. "
            f"Default: {DEFAULT_MAX_LAG}"
        ),
    )
    return parser.parse_args()


def load_merged_data(input_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in input directory: {input_dir}")

    frames: list[pd.DataFrame] = []
    required_columns = list(PLOT_COLUMNS)
    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path, usecols=required_columns)
        frame["source_file"] = csv_path.name
        frames.append(frame)

    merged_df = pd.concat(frames, ignore_index=True)
    return merged_df


def add_lagged_columns(df: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    if max_lag <= 0:
        return df

    lagged_df = df.copy()
    grouped = lagged_df.groupby("source_file", sort=False)
    for column_name in LAGGABLE_COLUMNS:
        for lag in range(1, max_lag + 1):
            lagged_df[f"{column_name}_lag_{lag}"] = grouped[column_name].shift(lag)
    return lagged_df


def build_plot_labels(max_lag: int) -> dict[str, str]:
    labels = dict(PLOT_COLUMNS)
    for column_name in LAGGABLE_COLUMNS:
        for lag in range(1, max_lag + 1):
            labels[f"{column_name}_lag_{lag}"] = f"{PLOT_COLUMNS[column_name]} (lag {lag})"
    return labels


def build_scatter_plots_for_lag(lag: int) -> list[tuple[str, str]]:
    if lag <= 0:
        return list(SCATTER_PLOTS)

    return [
        ("wiretip_to_weldpool_dist", "tapering_to_weldpool_dist"),
        ("wiretip_to_weldpool_dist", f"voltage_lag_{lag}"),
        ("wiretip_to_weldpool_dist", f"current_lag_{lag}"),
        ("tapering_to_weldpool_dist", f"voltage_lag_{lag}"),
        ("tapering_to_weldpool_dist", f"current_lag_{lag}"),
    ]


def sample_pair_dataframe(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    max_points: int | None,
    random_state: int,
) -> pd.DataFrame:
    pair_df = df[[x_col, y_col]].dropna()
    if max_points is not None and len(pair_df) > max_points:
        pair_df = pair_df.sample(n=max_points, random_state=random_state)
    return pair_df


def compute_pair_correlation(pair_df: pd.DataFrame, x_col: str, y_col: str) -> float:
    if len(pair_df) < 2:
        return float("nan")
    return float(pair_df[x_col].corr(pair_df[y_col]))


def build_output_name(x_col: str, y_col: str) -> str:
    return f"{x_col}_vs_{y_col}.png"


def draw_scatter_plot(
    pair_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    plot_labels: dict[str, str],
    output_path: Path,
    dpi: int,
) -> None:
    correlation = compute_pair_correlation(pair_df, x_col, y_col)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        pair_df[x_col],
        pair_df[y_col],
        s=8,
        alpha=0.18,
        c="#0b6e4f",
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlabel(plot_labels[x_col])
    ax.set_ylabel(plot_labels[y_col])
    ax.set_title(
        f"{plot_labels[y_col]} vs {plot_labels[x_col]}\n"
        f"n={len(pair_df):,}, Pearson r={correlation:.3f}"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_overview_figure(
    df: pd.DataFrame,
    output_dir: Path,
    scatter_plots: list[tuple[str, str]],
    plot_labels: dict[str, str],
    output_name: str,
    title: str,
    max_points: int | None,
    random_state: int,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    flat_axes = axes.flatten()

    for idx, (x_col, y_col) in enumerate(scatter_plots):
        pair_df = sample_pair_dataframe(df, x_col, y_col, max_points, random_state)
        correlation = compute_pair_correlation(pair_df, x_col, y_col)
        ax = flat_axes[idx]
        ax.scatter(
            pair_df[x_col],
            pair_df[y_col],
            s=6,
            alpha=0.15,
            c="#1d4ed8",
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlabel(plot_labels[x_col])
        ax.set_ylabel(plot_labels[y_col])
        ax.set_title(f"r={correlation:.3f}")
        ax.grid(True, alpha=0.25)

    flat_axes[-1].axis("off")
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / output_name, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_scatterplots(
    input_dir: Path,
    output_dir: Path,
    max_lag: int,
    max_points: int | None,
    random_state: int,
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = add_lagged_columns(load_merged_data(input_dir), max_lag)
    plot_labels = build_plot_labels(max_lag)

    saved_paths: list[Path] = []
    for x_col, y_col in SCATTER_PLOTS:
        pair_df = sample_pair_dataframe(df, x_col, y_col, max_points, random_state)
        if pair_df.empty:
            continue

        output_path = output_dir / build_output_name(x_col, y_col)
        draw_scatter_plot(pair_df, x_col, y_col, plot_labels, output_path, dpi)
        saved_paths.append(output_path)

    draw_overview_figure(
        df=df,
        output_dir=output_dir,
        scatter_plots=build_scatter_plots_for_lag(0),
        plot_labels=plot_labels,
        output_name="distance_voltage_current_scatterplots.png",
        title="Scatterplots: wiretip-/tapering-to-weldpool distances vs each other, voltage, and current",
        max_points=max_points,
        random_state=random_state,
        dpi=dpi,
    )
    saved_paths.append(output_dir / "distance_voltage_current_scatterplots.png")

    for lag in range(1, max_lag + 1):
        output_name = f"distance_voltage_current_scatterplots_lag_{lag:02d}.png"
        draw_overview_figure(
            df=df,
            output_dir=output_dir,
            scatter_plots=build_scatter_plots_for_lag(lag),
            plot_labels=plot_labels,
            output_name=output_name,
            title=(
                "Scatterplots: wiretip-/tapering-to-weldpool distances vs each other, "
                f"voltage lag {lag}, and current lag {lag}"
            ),
            max_points=max_points,
            random_state=random_state,
            dpi=dpi,
        )
        saved_paths.append(output_dir / output_name)

    return saved_paths


def main() -> None:
    args = parse_args()
    saved_paths = generate_scatterplots(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_lag=max(0, args.max_lag),
        max_points=max(1, args.max_points) if args.max_points is not None else None,
        random_state=args.random_state,
        dpi=max(1, args.dpi),
    )
    for saved_path in saved_paths:
        print(saved_path)


if __name__ == "__main__":
    main()
