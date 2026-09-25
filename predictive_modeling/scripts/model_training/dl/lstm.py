from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from scripts.config import CONFIG
from scripts.preprocessing.split import SequenceData, split_data_folder


cfg = CONFIG["preprocessing"]["split"]
DATA_FOLDER = Path("/home/dharun/Master-Thesis-ESAB/data/Thesis/training_csvs/diff_downsampled")
OUT_ROOT = Path("/home/dharun/Master-Thesis-ESAB/output/predictive_modelling/results/lstm_outputs_sweep_additional_features_without_ctdw")
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
AGG_FEATURES += ["power_mean", "power_std"]

# Combined feature names for reference (per-timestep features + aggregated names)
ALL_FEATURE_NAMES = [*FEATURES, *AGG_FEATURES]

WINDOW_SIZES = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
BATCH_SIZE = 256
EPOCHS = 25
PATIENCE = 5
LR = 1.0e-3
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 3
LSTM_DROPOUT = 0.3
HEAD_HIDDEN_SIZE = 128
HEAD_MID_SIZE = 64
GRAD_CLIP_NORM = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)
print(f"targets: {TARGETS}")
print(f"features: {FEATURES}")
print(f"window sizes: {WINDOW_SIZES}")


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=LSTM_DROPOUT,
            bidirectional=True,
        )
        lstm_out_size = LSTM_HIDDEN_SIZE * 2
        self.temporal_gate = nn.Sequential(
            nn.Linear(lstm_out_size, lstm_out_size),
            nn.Tanh(),
            nn.Linear(lstm_out_size, 1),
        )
        self.proj = nn.Sequential(
            nn.Linear(lstm_out_size * 2, HEAD_HIDDEN_SIZE),
            nn.LayerNorm(HEAD_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(LSTM_DROPOUT),
            nn.Linear(HEAD_HIDDEN_SIZE, HEAD_MID_SIZE),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(HEAD_MID_SIZE, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        out, _ = self.lstm(x)
        attn_logits = self.temporal_gate(out)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(out * attn_weights, dim=1)
        last_step = out[:, -1, :]
        representation = torch.cat([last_step, pooled], dim=-1)
        return self.head(self.proj(representation))


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
            # compute aggregated window stats and append repeated across timesteps
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

            agg_repeated = np.repeat(np.asarray(agg_vals, dtype=np.float32)[None, :], window_size, axis=0)
            xs.append(np.hstack([x_raw[idx : idx + window_size], agg_repeated]))
            ys.append(np.asarray(y_raw[idx + window_size - 1], dtype=np.float32))

    if not xs:
        raise ValueError(f"All sequences empty for window_size={window_size}")

    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def scale_features(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    n_train, seq_len, n_features = x_train.shape

    x_train_scaled = scaler.fit_transform(x_train.reshape(-1, n_features)).reshape(n_train, seq_len, n_features)
    x_val_scaled = scaler.transform(x_val.reshape(-1, n_features)).reshape(x_val.shape[0], seq_len, n_features)
    x_test_scaled = scaler.transform(x_test.reshape(-1, n_features)).reshape(x_test.shape[0], seq_len, n_features)
    return x_train_scaled, x_val_scaled, x_test_scaled, scaler


def scale_targets(y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train)
    y_val_scaled = scaler.transform(y_val)
    y_test_scaled = scaler.transform(y_test)
    return y_train_scaled, y_val_scaled, y_test_scaled, scaler


def batch_predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), BATCH_SIZE):
            xb = torch.tensor(x[start : start + BATCH_SIZE], dtype=torch.float32, device=device)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


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


class SimpleWrapper:
    """Sklearn-compatible wrapper for permutation importance on sequence models."""

    def __init__(self, model: nn.Module, window_size: int, n_features: int, target_idx: int = 0):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.target_idx = target_idx

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "SimpleWrapper":
        # permutation_importance validates estimators by checking for fit().
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_seq = np.asarray(x, dtype=np.float32).reshape(-1, self.window_size, self.n_features)
        return batch_predict(self.model, x_seq)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return -float(np.sqrt(mean_squared_error(y[:, self.target_idx], pred[:, self.target_idx])))


def compute_feature_importance(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    window_size: int,
    n_features: int,
    target_idx: int = 0,
) -> np.ndarray:
    """Compute permutation importance and aggregate per-feature across timesteps."""
    wrapper = SimpleWrapper(model, window_size=window_size, n_features=n_features, target_idx=target_idx)

    # sklearn permutation_importance expects a 2D matrix.
    x_flat = x_test.reshape(x_test.shape[0], window_size * n_features)
    result = permutation_importance(wrapper, x_flat, y_test, n_repeats=5, random_state=42, n_jobs=1)

    # Aggregate lag-wise importances into one score per original feature.
    importances = result.importances_mean.reshape(window_size, n_features)
    return importances.mean(axis=0)


def train_for_window(window_size: int, split: object, out_dir: Path) -> dict:
    """Train a model for a specific window size."""
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train = build_windows(split.train, window_size)
    x_val, y_val = build_windows(split.validation, window_size)
    x_test, y_test = build_windows(split.test, window_size)
    
    print(f"\nWindow {window_size}: X_train: {x_train.shape}, X_val: {x_val.shape}, X_test: {x_test.shape}")

    x_train, x_val, x_test, x_scaler = scale_features(x_train, x_val, x_test)
    y_train, y_val, y_test, y_scaler = scale_targets(y_train, y_val, y_test)

    with open(out_dir / "x_scaler.pkl", "wb") as handle:
        pickle.dump(x_scaler, handle)
    with open(out_dir / "y_scaler.pkl", "wb") as handle:
        pickle.dump(y_scaler, handle)

    # Use actual feature count from data (includes aggregated features)
    n_features_per_timestep = x_train.shape[2]
    model = LSTMRegressor(input_size=n_features_per_timestep, output_size=len(TARGETS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0e-3)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    best_val_rmse = float("inf")
    best_path = out_dir / "best_model.pt"
    counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            total_loss += float(loss.item())

        val_pred_scaled = batch_predict(model, x_val)
        val_pred = y_scaler.inverse_transform(val_pred_scaled)
        val_true = y_scaler.inverse_transform(y_val)
        val_rrse = float(np.mean([rrse(val_true[:, idx], val_pred[:, idx]) for idx in range(val_true.shape[1])]))
        val_rmse = float(np.mean([np.sqrt(mean_squared_error(val_true[:, idx], val_pred[:, idx])) for idx in range(val_true.shape[1])]))
        scheduler.step(val_rmse)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1}: Train Loss: {total_loss / max(len(train_loader), 1):.6f}, "
                f"Val RMSE: {val_rmse:.6f}, Val RRSE: {val_rrse:.6f}"
            )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            counter += 1

        if counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = {
        "train": (y_train, batch_predict(model, x_train)),
        "validation": (y_val, batch_predict(model, x_val)),
        "test": (y_test, batch_predict(model, x_test)),
    }

    results: dict[str, dict] = {}
    for name, (y_scaled, pred_scaled) in predictions.items():
        y_true = y_scaler.inverse_transform(y_scaled)
        y_pred = y_scaler.inverse_transform(pred_scaled)

        pd.DataFrame(
            np.hstack([y_true, y_pred]),
            columns=[*[f"y_{target}" for target in TARGETS], *[f"yhat_{target}" for target in TARGETS]],
        ).to_csv(out_dir / f"{name}_preds.csv", index=False)
        results[name] = score(y_true, y_pred)
        save_plots(name, y_true, y_pred, out_dir)

    with open(out_dir / "metrics.json", "w") as handle:
        json.dump(results, handle, indent=2)

    # Compute feature importance on scaled test windows.
    feature_importance_list = []
    for target_idx in range(len(TARGETS)):
        importance = compute_feature_importance(
            model,
            x_test,
            y_test,
            window_size=window_size,
            n_features=x_test.shape[2],
            target_idx=target_idx,
        )
        feature_importance_list.append(importance)

    feature_importance_avg = np.mean(feature_importance_list, axis=0)
    feature_names = ALL_FEATURE_NAMES if len(ALL_FEATURE_NAMES) == x_test.shape[2] else [f"f{i}" for i in range(x_test.shape[2])]
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance_avg,
    }).sort_values("importance", ascending=False)
    
    feature_importance_df.to_csv(out_dir / "feature_importance.csv", index=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df["feature"], feature_importance_df["importance"])
    plt.xlabel("Permutation Importance")
    plt.title(f"Feature Importance (Window Size: {window_size})")
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
    thesis_manifest_path = Path(
        "/home/dharun/Master-Thesis-ESAB/data/Thesis/result_csvs/split_manifest.csv"
    )

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
        print(f"Training LSTM for window size {ws}")
        print(f"{'='*60}")
        
        out_dir = OUT_ROOT / f"window_{ws:03d}"
        result = train_for_window(ws, split, out_dir)
        all_results.append(result)
        
        test_rmses = [result["metrics"]["test"][target]["rmse"] for target in TARGETS]
        rmse_by_window[ws] = np.mean(test_rmses)

    # Plot RMSE vs Window Size
    window_sizes_list = sorted(rmse_by_window.keys())
    rmse_values = [rmse_by_window[ws] for ws in window_sizes_list]

    plt.figure(figsize=(10, 6))
    plt.plot(window_sizes_list, rmse_values, marker='o', linewidth=2, markersize=8)
    plt.xlabel("Window Size")
    plt.ylabel("Test RMSE (Avg across targets)")
    plt.title("LSTM: RMSE vs Window Size")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "rmse_vs_window.png", dpi=150)
    plt.close()

    # Aggregate feature importance across window sizes
    all_feature_importance = {}
    for result in all_results:
        ws = result["window_size"]
        feat_imp = result["feature_importance"]
        for feat, imp in zip(feat_imp["feature"], feat_imp["importance"]):
            if feat not in all_feature_importance:
                all_feature_importance[feat] = []
            all_feature_importance[feat].append(imp)

    avg_feature_importance = pd.DataFrame({
        "feature": list(all_feature_importance.keys()),
        "avg_importance": [np.mean(all_feature_importance[f]) for f in all_feature_importance.keys()],
    }).sort_values("avg_importance", ascending=False)

    avg_feature_importance.to_csv(OUT_ROOT / "avg_feature_importance.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(avg_feature_importance["feature"], avg_feature_importance["avg_importance"])
    plt.xlabel("Average Permutation Importance")
    plt.title("LSTM: Average Feature Importance Across All Window Sizes")
    plt.tight_layout()
    plt.savefig(OUT_ROOT / "avg_feature_importance.png", dpi=150)
    plt.close()

    # Summary JSON
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