import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import CONFIG
from scripts.preprocessing.merge import build_sampling_csvs
from scripts.preprocessing.h5_to_csv import convert_h5_folder_to_csv


TRUE_VALUES = {"true", "1", "yes", "y", "t"}


def parse_bool_arg(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUE_VALUES


def preprocess_train_data(
    input_path: str = CONFIG["preprocessing"]["input_path"],
    detect_path: str = CONFIG["preprocessing"]["detect_path"],
    anomaly_path: str = CONFIG["preprocessing"]["anomaly_path"],
    h5_input_path: str = CONFIG["preprocessing"]["h5_input_path"],
    h5_output_path: str = CONFIG["preprocessing"]["h5_output_path"],
    output_path_org: str = CONFIG["preprocessing"]["output_path_org"],
    output_path_clipped: str = CONFIG["preprocessing"]["output_path_clipped"],
    output_path_clipped_norm: str = CONFIG["preprocessing"]["output_path_clipped_norm"],
    upsample: bool = CONFIG["preprocessing"]["upsample"],
) -> list[Path]:
    """
    Run anomaly repair and merged CSV export for the full training dataset.

    If ``upsample`` is True, detection features are linearly interpolated onto
    the measurement timeline. If ``upsample`` is False, the measurement signals
    are downsampled to one row per frame by averaging the samples assigned to
    each frame.
    """

    convert_h5_folder_to_csv(
        h5_folder_path=h5_input_path,
        csv_output_folder=h5_output_path,
    )

    interpolation_cfg = CONFIG["preprocessing"]["interpolation"]
    return build_sampling_csvs(
        input_csv_dir=input_path,
        detection_csv_dir=detect_path,
        anomaly_csv_dir=anomaly_path,
        output_csv_dir=output_path_org,
        output_csv_dir_clipped=output_path_clipped,
        output_csv_dir_clipped_norm=output_path_clipped_norm,
        upsample=upsample,
        interpolation_columns=interpolation_cfg["interpolation_columns"],
        max_anomaly_length=CONFIG["preprocessing"]["max_anomaly_length"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run anomaly interpolation plus measurement/detection merge for predictive modeling.",
    )
    parser.add_argument("--input-path", default=CONFIG["preprocessing"]["input_path"])
    parser.add_argument("--detect-path", default=CONFIG["preprocessing"]["detect_path"])
    parser.add_argument("--anomaly-path", default=CONFIG["preprocessing"]["anomaly_path"])
    parser.add_argument("--output-path-org", default=CONFIG["preprocessing"]["output_path_org"])
    parser.add_argument("--output-path-clipped", default=CONFIG["preprocessing"]["output_path_clipped"])
    parser.add_argument("--output-path-clipped-norm", default=CONFIG["preprocessing"]["output_path_clipped_norm"])
    parser.add_argument("--upsample", default=CONFIG["preprocessing"]["upsample"], type=parse_bool_arg)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_train_data(
        input_path=args.input_path,
        detect_path=args.detect_path,
        anomaly_path=args.anomaly_path,
        output_path_org=args.output_path_org,
        output_path_clipped=args.output_path_clipped,
        output_path_clipped_norm=args.output_path_clipped_norm,
        upsample=args.upsample,
    )
