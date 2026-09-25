from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.config import CONFIG
from scripts.preprocessing.split import SequenceData, split_data_folder


cfg = CONFIG["preprocessing"]["split"]
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_FOLDER = Path(CONFIG["preprocessing"]["output_path_clipped"])
OUT_ROOT = PROJECT_ROOT / "output" / "predictive_modeling" / "xgb"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TARGETS = list(dict.fromkeys(cfg.get("target_cols", [cfg["target_col"]])))
FEATURES = list(
    dict.fromkeys(
        [*list(cfg.get("history_cols", [])), *list(cfg.get("context_cols", [])), *list(cfg.get("offline_metadata_cols", []))]
    )
)

# Ensure basic electrical signals are included as history features (if present in data)
BASE_ELEC = ["voltage", "current", "delta_voltage", "delta_current"]
for col in reversed(BASE_ELEC):
    if col not in FEATURES:
        FEATURES.insert(0, col)

# Aggregated window-level features to compute for each window (not expected as raw cols)
AGG_STATS = ["mean", "std", "min", "max", "slope"]
AGG_FEATURES = [f"{sig}_{stat}" for sig in ["voltage", "current"] for stat in AGG_STATS]
# add power aggregates
AGG_FEATURES += ["power_mean", "power_std"]

# Combined feature list for naming/importance (per-timestep FEATURES + aggregated features repeated per timestep)
ALL_FEATURE_NAMES = [*FEATURES, *AGG_FEATURES]

WINDOW_SIZES = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]

print(f"targets: {TARGETS}")
print(f"features: {FEATURES}")
print(f"window sizes: {WINDOW_SIZES}")


def rrse(y: np.ndarray, yhat: np.ndarray) -> float:
    denom = np.sum((y - np.mean(y)) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sqrt(np.sum((y - yhat) ** 2) / denom))


def score(y: np.ndarray, yhat: np.ndarray) -> dict[str, dict]:
    metrics = {}
    for idx, target in enumerate(TARGETS):
        target_y = y[:, idx]
        target_yhat = yhat[:, idx]
        metrics[target] = {
            "rmse": float(np.sqrt(mean_squared_error(target_y, target_yhat))),
            "mae": float(mean_absolute_error(target_y, target_yhat)),
            "rrse": float(rrse(target_y, target_yhat)),
            "pearson": float(pearsonr(target_y, target_yhat)[0]),
        }
    return metrics


def build_windows(seqs: list[SequenceData], window_size: int) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for seq in seqs:
        df = seq.df.copy()
        missing = [column for column in [*TARGETS, *FEATURES] if column not in df.columns]
        if missing:
            raise ValueError(f"{seq.file_name} is missing required columns: {missing}")

        # compute derived electrical columns if raw signals present
        if "voltage" in df.columns and "current" in df.columns:
            df = df.copy()
            df["voltage"] = pd.to_numeric(df["voltage"], errors="coerce")
            df["current"] = pd.to_numeric(df["current"], errors="coerce")
            df["delta_voltage"] = df["voltage"].diff().fillna(0.0)
            df["delta_current"] = df["current"].diff().fillna(0.0)
            df["power"] = df["voltage"] * df["current"]

        frame = df[[*FEATURES, *TARGETS]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
        if len(frame) <= window_size:
            continue

        x_raw = frame[FEATURES].to_numpy(dtype=np.float32)
        y_raw = frame[TARGETS].to_numpy(dtype=np.float32)

        for idx in range(len(frame) - window_size + 1):
            # per-window aggregated statistics for voltage/current and power
            win_df = frame.iloc[idx : idx + window_size]
            vol = win_df["voltage"].to_numpy(dtype=np.float32)
            cur = win_df["current"].to_numpy(dtype=np.float32)
            power = (vol * cur).astype(np.float32)

            def slope_of(arr: np.ndarray) -> float:
                if len(arr) <= 1:
                    return 0.0
                try:
                    return float(np.polyfit(np.arange(len(arr), dtype=float), arr.astype(float), 1)[0])
                except Exception:
                    return 0.0

            agg_vals = [
                float(np.mean(vol)),
                float(np.std(vol, ddof=0)),
                float(np.min(vol)),
                float(np.max(vol)),
                slope_of(vol),
                float(np.mean(cur)),
                float(np.std(cur, ddof=0)),
                float(np.min(cur)),
                float(np.max(cur)),
                slope_of(cur),
                float(np.mean(power)),
                float(np.std(power, ddof=0)),
            ]

            # Repeat aggregated scalars across the time dimension so model sees them as context
            agg_repeated = np.repeat(np.asarray(agg_vals, dtype=np.float32)[None, :], window_size, axis=0)
            xs.append(np.hstack([x_raw[idx : idx + window_size], agg_repeated]))
            ys.append(np.asarray(y_raw[idx + window_size - 1], dtype=np.float32))

    if not xs:
        raise ValueError(f"All sequences empty for window_size={window_size}")

    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def flatten_windows(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def save_plots(name: str, y: np.ndarray, yhat: np.ndarray, out_dir: Path) -> None:
    for idx, target in enumerate(TARGETS):
        plt.figure(figsize=(10, 4))
        plt.plot(y[:500, idx], label="True")
        plt.plot(yhat[:500, idx], label="Pred")
        plt.title(f"{name} | {target} line")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_{target}_line.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.scatter(y[:, idx], yhat[:, idx], s=4, alpha=0.5)
        min_val = float(min(y[:, idx].min(), yhat[:, idx].min()))
        max_val = float(max(y[:, idx].max(), yhat[:, idx].max()))
        plt.plot([min_val, max_val], [min_val, max_val], "r")
        plt.title(f"{name} | {target} scatter")
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_{target}_scatter.png", dpi=150)
        plt.close()


def feature_importance_from_model(model: MultiOutputRegressor, window_size: int, n_features: int) -> np.ndarray:
    per_target_importance: list[np.ndarray] = []
    for estimator in model.estimators_:
        importances = np.asarray(estimator.feature_importances_, dtype=np.float32)
        if importances.shape[0] != window_size * n_features:
            raise ValueError(
                f"Unexpected feature importance size {importances.shape[0]} for window_size={window_size}, n_features={n_features}"
            )
        per_target_importance.append(importances.reshape(window_size, n_features).mean(axis=0))
    return np.mean(per_target_importance, axis=0)


def train_for_window(window_size: int, split: object, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = build_windows(split.train, window_size)
    x_val, y_val = build_windows(split.validation, window_size)
    x_test, y_test = build_windows(split.test, window_size)

    print(f"\nWindow {window_size}: X_train: {x_train.shape}, X_val: {x_val.shape}, X_test: {x_test.shape}")

    x_train_flat = flatten_windows(x_train)
    x_val_flat = flatten_windows(x_val)
    x_test_flat = flatten_windows(x_test)

    base_model = XGBRegressor(
        n_estimators=1200,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        colsample_bylevel=0.9,
        min_child_weight=1,
        gamma=0.05,
        reg_alpha=0.05,
        reg_lambda=1.5,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
    )
    model = MultiOutputRegressor(base_model)
    model.fit(x_train_flat, y_train)
    print("model trained")

    predictions = {
        "train": (y_train, model.predict(x_train_flat)),
        "validation": (y_val, model.predict(x_val_flat)),
        "test": (y_test, model.predict(x_test_flat)),
    }

    results: dict[str, dict] = {}
    for name, (y_true, y_pred) in predictions.items():
        pd.DataFrame(
            np.hstack([y_true, y_pred]),
            columns=[*[f"y_{target}" for target in TARGETS], *[f"yhat_{target}" for target in TARGETS]],
        ).to_csv(out_dir / f"{name}_preds.csv", index=False)
        results[name] = score(y_true, y_pred)
        save_plots(name, y_true, y_pred, out_dir)

    with open(out_dir / "metrics.json", "w") as handle:
        json.dump(results, handle, indent=2)

    # feature importance expects n_features == number of columns per timestep (including aggregated features)
    n_features_per_timestep = x_train.shape[2]
    feature_importance = feature_importance_from_model(model, window_size=window_size, n_features=n_features_per_timestep)
    # feature names are original FEATURES plus AGG_FEATURES
    feature_names = ALL_FEATURE_NAMES if len(ALL_FEATURE_NAMES) == n_features_per_timestep else [f"f{i}" for i in range(n_features_per_timestep)]
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance,
    }).sort_values("importance", ascending=False)
    feature_importance_df.to_csv(out_dir / "feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
    plt.xlabel("Feature Importance")
    plt.title(f"XGB Feature Importance (Window Size: {window_size})")
    plt.tight_layout()
    plt.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close()

    print(f"  Window {window_size} complete. Test RMSE: {results['test']}")

    return {
        "window_size": window_size,
        "metrics": results,
        "feature_importance": feature_importance_df.to_dict("list"),
    }


def main() -> None:
    thesis_manifest_path = DATA_FOLDER / "split_manifest.csv"

    print(f"Source folder: {DATA_FOLDER}")
    print(f"Manifest path: {thesis_manifest_path}")

    split = split_data_folder(
        DATA_FOLDER,
        manifest_path=thesis_manifest_path,
    )

    print(
        f"Loaded from manifest:"
        f" train={len(split.train)},"
        f" val={len(split.validation)},"
        f" test={len(split.test)}"
    )

    split.manifest.to_csv(
        OUT_ROOT / "split_manifest.csv",
        index=False,
    )

    all_results = []
    rmse_by_window = {}

    for ws in WINDOW_SIZES:
        print(f"\n{'='*60}")
        print(f"Training XGB for window size {ws}")
        print(f"{'='*60}")

        out_dir = OUT_ROOT / f"window_{ws:03d}"
        result = train_for_window(ws, split, out_dir)
        all_results.append(result)

        test_rmses = [result["metrics"]["test"][target]["rmse"] for target in TARGETS]
        rmse_by_window[ws] = float(np.mean(test_rmses))

    window_sizes_list = sorted(rmse_by_window.keys())
    rmse_values = [rmse_by_window[ws] for ws in window_sizes_list]

    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes_list, rmse_values, marker="o", linewidth=2, markersize=8)
    plt.xlabel("Window Size")
    plt.ylabel("Test RMSE (Avg across targets)")
    plt.title("XGB: RMSE vs Window Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "rmse_vs_window.png", dpi=150)
    plt.close()

    all_feature_importance: dict[str, list[float]] = {}
    for result in all_results:
        feat_imp = result["feature_importance"]
        for feat, imp in zip(feat_imp["feature"], feat_imp["importance"]):
            all_feature_importance.setdefault(feat, []).append(float(imp))

    avg_feature_importance = pd.DataFrame(
        {
            "feature": list(all_feature_importance.keys()),
            "avg_importance": [np.mean(all_feature_importance[f]) for f in all_feature_importance.keys()],
        }
    ).sort_values("avg_importance", ascending=False)
    avg_feature_importance.to_csv(OUT_ROOT / "avg_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(avg_feature_importance["feature"], avg_feature_importance["avg_importance"])
    plt.xlabel("Average Feature Importance")
    plt.title("XGB: Average Feature Importance Across All Window Sizes")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "avg_feature_importance.png", dpi=150)
    plt.close()

    summary = {
        "window_sizes": window_sizes_list,
        "rmse_by_window": rmse_by_window,
        "best_window": min(rmse_by_window, key=rmse_by_window.get),
        "best_rmse": min(rmse_by_window.values()),
        "avg_feature_importance": avg_feature_importance.to_dict("list"),
    }
    with open(OUT_ROOT / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\n{'='*60}")
    print("All training complete!")
    print(f"Best window size: {summary['best_window']} with RMSE: {summary['best_rmse']:.4f}")
    print(f"Results saved to: {OUT_ROOT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
