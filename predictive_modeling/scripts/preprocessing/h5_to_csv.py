from pathlib import Path
 
import h5py
import pandas as pd
 
DEFAULT_H5_INPUT_DIR = Path("/mnt/e/thesis/Large_timeseries_and_video/we_5m_20mm.h5")
DEFAULT_CSV_OUTPUT_DIR = Path("/home/samuel/Master-Thesis-ESAB/output/measurements")
 
 
def flatten_h5_to_dict(h5_file: h5py.File) -> dict[str, object]:
    data_dict: dict[str, object] = {}
 
    def extract(name: str, obj: h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
 
        dataset = obj[()]
        if dataset.ndim == 1:
            data_dict[name] = dataset
            return
 
        for index in range(dataset.shape[1]):
            data_dict[f"{name}/{index}"] = dataset[:, index]
 
    h5_file.visititems(extract)
    return data_dict
 
 
def convert_h5_folder_to_csv(
    h5_folder_path: str | Path = DEFAULT_H5_INPUT_DIR,
    csv_output_folder: str | Path = DEFAULT_CSV_OUTPUT_DIR,
) -> list[Path]:
    h5_folder = Path(h5_folder_path)
    csv_output = Path(csv_output_folder)
    csv_output.mkdir(parents=True, exist_ok=True)
 
    if not h5_folder.exists():
        raise FileNotFoundError(f"HDF5 input path not found: {h5_folder}")

    written_files: list[Path] = []
    if h5_folder.is_file():
        h5_files = [h5_folder]
    else:
        h5_files = sorted(h5_folder.rglob("*.h5"))
    if not h5_files:
        print(f"No HDF5 files found in {h5_folder}")
        return written_files
 
    for h5_file_path in h5_files:
        print(f"Processing: {h5_file_path.name}")
        with h5py.File(h5_file_path, "r") as h5_file:
            datasets_dict = flatten_h5_to_dict(h5_file)
            df = pd.DataFrame({column_name: pd.Series(values) for column_name, values in datasets_dict.items()})
 
        csv_path = csv_output / f"{h5_file_path.stem}.csv"
        df.to_csv(csv_path, index=False)
        written_files.append(csv_path)
        print(f"  Saved CSV: {csv_path.name}")
 
    print(f"Converted {len(written_files)} HDF5 file(s) to CSV.")
    return written_files
 
 
if __name__ == "__main__":
    convert_h5_folder_to_csv()