"""Create sequence-aware train, validation, and test dataset partitions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import CONFIG as PROJECT_CONFIG

ROLES = ("train", "validation", "test")
VALID_MACHINE_TYPES = {"we", "fr"}
EXPERIMENT_SETTINGS_PATTERN = re.compile(r"\b[^_]+_(\d+)m_(\d+)mm(?:_\d+)?\b")
DEFAULT_SPLIT_CONFIG = PROJECT_CONFIG["preprocessing"]["split"]

__all__ = ["DatasetSplit", "SequenceData", "split_data_folder"]


@dataclass(frozen=True)
class SequenceData:
    """One source CSV and its assigned split role."""
    file_name: str
    split_role: str
    df: pd.DataFrame

    @property
    def source_file(self) -> str:
        if "source_file" in self.df.columns and len(self.df):
            return str(self.df["source_file"].iloc[0])
        return self.file_name

    @property
    def machine_group(self) -> str:
        match = re.match(r"([A-Za-z]+)_", self.source_file)
        return match.group(1).lower() if match else "unknown"


@dataclass(frozen=True)
class DatasetSplit:
    """Container for split sequences and the assignment manifest."""
    train: list[SequenceData]
    validation: list[SequenceData]
    test: list[SequenceData]
    manifest: pd.DataFrame
    note: str


def split_data_folder(
    data_folder: str | Path,
    *,
    machine_type: str | None = None,
    wfs: int | float | None = None,
    ctdw: int | float | None = None,
    config: dict[str, Any] | None = None,
) -> DatasetSplit:
    """Load CSV sequences and assign them to train, validation, and test sets."""
    split_config = dict(DEFAULT_SPLIT_CONFIG)
    if config:
        split_config.update(config)

    sequences = _load_sequences(
        Path(data_folder),
        split_config,
        machine_type=machine_type,
        wfs=wfs,
        ctdw=ctdw,
    )
    return _split_random_subsequences_by_video(sequences, split_config)


def _natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def _machine_group_from_file(file_name: str) -> str:
    match = re.match(r"([A-Za-z]+)_", file_name)
    return match.group(1).lower() if match else "unknown"


def _offline_metadata_columns(config: dict[str, Any]) -> list[str]:
    columns = [*list(config.get("offline_metadata_cols", [])), *list(config.get("context_cols", []))]
    return list(dict.fromkeys(columns))


def _load_sequences(
    data_folder: Path,
    config: dict[str, Any],
    *,
    machine_type: str | None = None,
    wfs: int | float | None = None,
    ctdw: int | float | None = None,
) -> list[SequenceData]:
    normalized_machine = machine_type.lower() if machine_type is not None else None
    if normalized_machine is not None and normalized_machine not in VALID_MACHINE_TYPES:
        raise ValueError("machine_type must be 'we', 'fr', or None.")

    target_col = str(config["target_col"])
    input_cols = [*list(config.get("history_cols", [])), *list(config.get("context_cols", []))]
    if not input_cols:
        input_cols = list(config.get("input_cols", []))
    input_cols = list(dict.fromkeys(input_cols))
    offline_cols = _offline_metadata_columns(config)
    required_cols = [target_col, *input_cols]
    value_required_cols = input_cols if bool(config.get("allow_missing_target", False)) else required_cols
    sequences: list[SequenceData] = []
    paths = sorted(data_folder.rglob("*.csv"), key=_natural_key)
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {data_folder}")

    for path in paths:
        if normalized_machine is not None and _machine_group_from_file(path.name) != normalized_machine:
            continue

        df = pd.read_csv(path)
        missing = [column for column in required_cols if column not in df.columns]
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {missing}")

        filename_match = EXPERIMENT_SETTINGS_PATTERN.search(path.stem.lower())
        filename_wfs = float(filename_match.group(1)) if filename_match is not None else None
        filename_ctdw = float(filename_match.group(2)) if filename_match is not None else None
        file_settings = {"wfs": filename_wfs, "ctdw": filename_ctdw}
        for column in ("wfs", "ctdw"):
            if column in df.columns:
                values = pd.to_numeric(df[column], errors="coerce").dropna()
                if not values.empty:
                    file_settings[column] = float(values.median())
        if (
            wfs is not None
            and (
                file_settings["wfs"] is None
                or not math.isclose(float(file_settings["wfs"]), float(wfs), rel_tol=0.0, abs_tol=1.0e-9)
            )
        ):
            continue
        if (
            ctdw is not None
            and (
                file_settings["ctdw"] is None
                or not math.isclose(float(file_settings["ctdw"]), float(ctdw), rel_tol=0.0, abs_tol=1.0e-9)
            )
        ):
            continue

        for column in dict.fromkeys([str(config["frame_order_col"]), "meas_time", "time", "timestamp", "frame_number"]):
            if column in df.columns:
                df = df.copy()
                df[column] = pd.to_numeric(df[column], errors="coerce")
                df = df.sort_values(column, kind="stable").reset_index(drop=True)
                break
        else:
            df = df.reset_index(drop=True)

        numeric_cols = list(
            dict.fromkeys(
                [
                    *required_cols,
                    *[column for column in offline_cols if column in df.columns],
                ]
            )
        )
        for column in numeric_cols:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=value_required_cols).reset_index(drop=True)
        if df.empty:
            continue

        df["sample_index"] = np.arange(len(df), dtype=int)
        sequences.append(SequenceData(file_name=path.name, split_role="all", df=df))

    if not sequences:
        filters = f"machine_type={machine_type}, wfs={wfs}, ctdw={ctdw}"
        raise ValueError(f"No usable sequences found in {data_folder} for filters: {filters}")
    return sequences


def _sequence_split_metadata(sequence: SequenceData, config: dict[str, Any]) -> dict[str, Any]:
    target_col = str(config["target_col"])
    context_cols = list(config.get("context_cols", []))
    metadata_cols = _offline_metadata_columns(config)
    df = sequence.df
    row: dict[str, Any] = {
        "file": sequence.file_name,
        "source_file": sequence.source_file,
        "machine_group": sequence.machine_group,
        "n_rows": int(len(df)),
        "target_mean": float(df[target_col].mean()),
        "target_std": float(df[target_col].std(ddof=0)),
    }
    if "source_row_pos" in df.columns:
        row["source_row_start"] = int(df["source_row_pos"].iloc[0])
        row["source_row_end"] = int(df["source_row_pos"].iloc[-1])
    if "source_subsequence_index" in df.columns:
        row["source_subsequence_index"] = int(df["source_subsequence_index"].iloc[0])
    for column in metadata_cols:
        if column in df.columns:
            row[column] = float(pd.to_numeric(df[column], errors="coerce").median())
    labels = []
    for column in context_cols:
        value = row.get(column, np.nan)
        labels.append(f"{column}=missing" if pd.isna(value) else f"{column}={float(value):.3f}")
    row["condition_group"] = "|".join(labels) if labels else "all"
    return row


def _split_assignment_score(
    assignments: dict[str, list[int]],
    metadata: list[dict[str, Any]],
    target_counts: dict[str, int],
) -> float:
    def offline_condition_key(row: dict[str, Any]) -> str:
        labels = []
        for column in ("wfs", "ctdw"):
            value = row.get(column, math.nan)
            labels.append(f"{column}={float(value):.3f}" if pd.notna(value) else f"{column}=missing")
        return "|".join(labels)

    n_rows = np.asarray([row["n_rows"] for row in metadata], dtype=float)
    target_mean = np.asarray([row["target_mean"] for row in metadata], dtype=float)
    finite_target_mean = target_mean[np.isfinite(target_mean)]
    target_mean_fallback = float(np.mean(finite_target_mean)) if finite_target_mean.size else 0.0
    target_mean = np.where(np.isfinite(target_mean), target_mean, target_mean_fallback)
    wfs_values = np.asarray([row.get("wfs", 0.0) for row in metadata], dtype=float)
    ctdw_values = np.asarray([row.get("ctdw", 0.0) for row in metadata], dtype=float)
    machine_groups = np.asarray([str(row["machine_group"]) for row in metadata], dtype=object)
    condition_keys = np.asarray(
        [f"{row['machine_group']}|{offline_condition_key(row)}" for row in metadata],
        dtype=object,
    )

    total_rows = float(n_rows.sum())
    global_target_mean = float(np.average(target_mean, weights=n_rows))
    global_wfs_mean = float(np.mean(wfs_values))
    global_ctdw_mean = float(np.mean(ctdw_values))
    target_scale = float(np.std(target_mean)) or 1.0
    wfs_scale = float(np.std(wfs_values)) or 1.0
    ctdw_scale = float(np.std(ctdw_values)) or 1.0
    machine_values = sorted(set(machine_groups.tolist()))
    condition_values, condition_count_values = np.unique(condition_keys, return_counts=True)

    score = 0.0
    for role, indices in assignments.items():
        role_index = np.asarray(indices, dtype=int)
        expected_row_fraction = target_counts[role] / float(len(metadata))
        actual_row_fraction = float(n_rows[role_index].sum()) / total_rows
        score += 2.0 * abs(actual_row_fraction - expected_row_fraction)

        role_target_mean = float(np.average(target_mean[role_index], weights=n_rows[role_index]))
        score += 1.5 * abs(role_target_mean - global_target_mean) / max(target_scale, 1.0e-9)
        score += abs(float(np.mean(wfs_values[role_index])) - global_wfs_mean) / max(wfs_scale, 1.0e-9)
        score += abs(float(np.mean(ctdw_values[role_index])) - global_ctdw_mean) / max(ctdw_scale, 1.0e-9)

        for machine in machine_values:
            global_fraction = float(np.mean(machine_groups == machine))
            role_fraction = float(np.mean(machine_groups[role_index] == machine)) if len(role_index) else 0.0
            score += 12.0 * abs(role_fraction - global_fraction)
            if global_fraction > 0.0 and role_fraction == 0.0:
                score += 25.0

    for condition_key, count in zip(condition_values, condition_count_values):
        if int(count) < 2:
            continue
        roles_with_condition = 0
        max_in_one_role = 0
        for indices in assignments.values():
            role_index = np.asarray(indices, dtype=int)
            role_count = int(np.sum(condition_keys[role_index] == condition_key))
            roles_with_condition += int(role_count > 0)
            max_in_one_role = max(max_in_one_role, role_count)
        score += 12.0 * max(0, int(count) - roles_with_condition)
        score += 4.0 * max(0, max_in_one_role - 1)

    return float(score)


def _split_random_subsequences_by_video(
    sequences: list[SequenceData],
    config: dict[str, Any],
) -> DatasetSplit:
    length = max(1, int(config.get("subsequence_length", 4_000)))
    min_rows = max(1, int(config.get("min_subsequence_rows", 1_000)))
    configured_gap_rows = int(config.get("subsequence_split_gap_rows", 0))
    if configured_gap_rows >= 0:
        gap_rows = configured_gap_rows
    else:
        window_values = [int(value) for value in config.get("window_size_values", [config["window_size"]])]
        warmup = max([int(config["window_size"]), *window_values])
        summary_windows = [int(value) for value in config.get("rolling_summary_windows", [])]
        if summary_windows:
            warmup = max(warmup, max(summary_windows) - 1)
        if bool(config.get("use_initial_context_features", False)):
            warmup = max(warmup, int(config.get("initial_context_window", 1)) - 1)
        if bool(config.get("use_past_true_output", False)):
            warmup = max(warmup, int(config["output_lag_count"]))
        gap_rows = max(warmup, 0)
    step = length + gap_rows
    n_trials = max(1, int(config.get("split_shuffle_trials", 2_000)))
    rng = np.random.default_rng(int(config["random_state"]))
    train: list[SequenceData] = []
    validation: list[SequenceData] = []
    test: list[SequenceData] = []

    for sequence in sorted(sequences, key=lambda seq: _natural_key(Path(seq.file_name))):
        n_rows = len(sequence.df)
        chunks: list[tuple[int, int, int]] = []
        chunk_index = 0
        for start in range(0, n_rows, step):
            end = min(start + length, n_rows)
            if end - start < min_rows:
                continue
            chunks.append((chunk_index, start, end))
            chunk_index += 1
        if not chunks:
            continue

        n_chunks = len(chunks)
        order = rng.permutation(n_chunks)
        train_fraction = float(config["train_fraction"])
        validation_fraction = float(config["validation_fraction"])
        if n_chunks >= 3:
            if not 0.0 < train_fraction < 1.0:
                raise ValueError("train_fraction must be between 0 and 1.")
            if not 0.0 < validation_fraction < 1.0:
                raise ValueError("validation_fraction must be between 0 and 1.")
            if train_fraction + validation_fraction >= 1.0:
                raise ValueError("train_fraction + validation_fraction must leave a non-empty test split.")
            n_train = max(1, int(round(n_chunks * train_fraction)))
            n_validation = max(1, int(round(n_chunks * validation_fraction)))
            if n_train + n_validation >= n_chunks:
                n_validation = max(1, n_chunks - n_train - 1)
            if n_train + n_validation >= n_chunks:
                n_train = max(1, n_chunks - n_validation - 1)
            counts = {"train": n_train, "validation": n_validation, "test": n_chunks - n_train - n_validation}
        elif n_chunks == 2:
            counts = {"train": 1, "validation": 1, "test": 0}
        else:
            counts = {"train": 1, "validation": 0, "test": 0}

        target_counts = {role: int(counts[role]) for role in ROLES}
        if len(chunks) >= 3 and all(target_counts[role] > 0 for role in ROLES):
            candidate_sequences = []
            source_file = sequence.source_file
            for chunk_index, start, end in chunks:
                df = sequence.df.iloc[start:end].copy().reset_index(drop=True)
                df["source_file"] = source_file
                df["source_row_pos"] = np.arange(start, end, dtype=int)
                df["source_row_start"] = int(start)
                df["source_row_end"] = int(end - 1)
                df["source_subsequence_index"] = int(chunk_index)
                df["subsequence_id"] = f"{source_file}::subseq_{chunk_index:04d}"
                df["subsequence_index"] = int(chunk_index)
                df["subsequence_start"] = int(start)
                df["subsequence_end"] = int(end - 1)
                df["subsequence_row_pos"] = np.arange(len(df), dtype=int)
                df["subsequence_role"] = "candidate"
                candidate_sequences.append(
                    SequenceData(
                        file_name=f"{source_file}::subseq_{chunk_index:04d}",
                        split_role="candidate",
                        df=df,
                    )
                )
            metadata = [_sequence_split_metadata(candidate, config) for candidate in candidate_sequences]
            train_end = target_counts["train"]
            validation_end = train_end + target_counts["validation"]
            best_assignments = {
                "train": order[:train_end].astype(int).tolist(),
                "validation": order[train_end:validation_end].astype(int).tolist(),
                "test": order[validation_end:].astype(int).tolist(),
            }
            best_score = _split_assignment_score(best_assignments, metadata, target_counts)
            for _ in range(n_trials - 1):
                trial_order = rng.permutation(len(chunks))
                trial_assignments = {
                    "train": trial_order[:train_end].astype(int).tolist(),
                    "validation": trial_order[train_end:validation_end].astype(int).tolist(),
                    "test": trial_order[validation_end:].astype(int).tolist(),
                }
                score = _split_assignment_score(trial_assignments, metadata, target_counts)
                if score < best_score:
                    best_score = score
                    best_assignments = trial_assignments
            assignments = {
                role: np.asarray(indices, dtype=int)
                for role, indices in best_assignments.items()
            }
        else:
            assignments = {
                "train": order[: counts["train"]],
                "validation": order[counts["train"] : counts["train"] + counts["validation"]],
                "test": order[
                    counts["train"] + counts["validation"] : counts["train"]
                    + counts["validation"]
                    + counts["test"]
                ],
            }
        for role, positions in assignments.items():
            for position in positions:
                chunk_index, start, end = chunks[int(position)]
                source_file = sequence.source_file
                subsequence_id = f"{source_file}::subseq_{chunk_index:04d}"
                df = sequence.df.iloc[start:end].copy().reset_index(drop=True)
                df["source_file"] = source_file
                df["source_row_pos"] = np.arange(start, end, dtype=int)
                df["source_row_start"] = int(start)
                df["source_row_end"] = int(end - 1)
                df["source_subsequence_index"] = int(chunk_index)
                df["subsequence_id"] = subsequence_id
                df["subsequence_index"] = int(chunk_index)
                df["subsequence_start"] = int(start)
                df["subsequence_end"] = int(end - 1)
                df["subsequence_row_pos"] = np.arange(len(df), dtype=int)
                df["subsequence_role"] = role
                subsequence = SequenceData(file_name=subsequence_id, split_role=role, df=df)
                if role == "train":
                    train.append(subsequence)
                elif role == "validation":
                    validation.append(subsequence)
                elif role == "test":
                    test.append(subsequence)

    if not train or not validation or not test:
        raise ValueError("random_subsequence_by_video split could not allocate non-empty train/validation/test splits.")
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)

    manifest_rows = []
    for role, role_sequences in (("train", train), ("validation", validation), ("test", test)):
        for seq in role_sequences:
            metadata = _sequence_split_metadata(seq, config)
            manifest_rows.append(
                {
                    "file": seq.file_name,
                    "source_file": metadata.get("source_file", seq.file_name),
                    "machine_group": metadata["machine_group"],
                    "split_role": role,
                    "start_sample_index": int(seq.df["sample_index"].iloc[0]),
                    "end_sample_index": int(seq.df["sample_index"].iloc[-1]),
                    "source_row_start": metadata.get("source_row_start", int(seq.df["sample_index"].iloc[0])),
                    "source_row_end": metadata.get("source_row_end", int(seq.df["sample_index"].iloc[-1])),
                    "source_subsequence_index": metadata.get("source_subsequence_index", math.nan),
                    "n_rows": int(len(seq.df)),
                    "wfs": metadata.get("wfs", math.nan),
                    "ctdw": metadata.get("ctdw", math.nan),
                    "condition_group": metadata["condition_group"],
                    "target_mean": metadata["target_mean"],
                    "target_std": metadata["target_std"],
                }
            )
    manifest = pd.DataFrame(manifest_rows)
    note = (
        "Random split by contiguous non-overlapping subsequences within each source video. "
        f"subsequence_length={length}, gap_rows={gap_rows}, min_rows={min_rows}; "
        f"uses {n_trials} shuffle trials per eligible video; eligible videos with at least three "
        "subsequences contribute to train, validation, and test."
    )
    return DatasetSplit(train=train, validation=validation, test=test, manifest=manifest, note=note)
