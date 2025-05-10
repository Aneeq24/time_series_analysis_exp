#!/usr/bin/env python3

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import keras
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

base_plot_dir = 'model_comparison_plots'

if not os.path.exists(base_plot_dir):
    os.makedirs(base_plot_dir)
logger.info(f"Created directory '{base_plot_dir}' for saving figures")

logger.info("Loading and preprocessing data")
wd = pd.read_csv("preprocessed_data.csv")  # Update with your file path

target_columns = [
    "scd41_co2",
    "scd41_temperature",
    "svm41_nox_index",
    "scd41_humidity",
    "ens160_eco2",
    "ens160_tvoc",
]
required_columns = ["timestamp", "station_id"] + target_columns

for col in required_columns:
    if col not in wd.columns:
        logger.error(f"Missing required column: {col}")
        raise ValueError(f"Missing required column: {col}")

# ------------------- Temporal Split ------------------- #
wd['timestamp'] = pd.to_datetime(wd['timestamp'])
wd = wd.sort_values(by='timestamp').reset_index(drop=True)
split_idx = int(0.8 * len(wd))
train_raw = wd.iloc[:split_idx].copy()
test_raw = wd.iloc[split_idx:].copy()

# ------------------- Missing Value Handling ------------------- #
train_means = {}
for col in target_columns:
    train_means[col] = train_raw[col].mean()
    train_raw[col] = train_raw[col].fillna(train_means[col])
    test_raw[col] = test_raw[col].fillna(train_means[col])

# ------------------- Normalization ------------------- #
scalers = {}
for col in target_columns:
    scaler = MinMaxScaler()
    scaler.fit(train_raw[col].values.reshape(-1, 1))
    scalers[col] = scaler
    train_raw[col] = scaler.transform(train_raw[col].values.reshape(-1, 1))
    test_raw[col] = scaler.transform(test_raw[col].values.reshape(-1, 1))

# ------------------- Create Sequences ------------------- #
seq_length = 200  # Length of encoder input
pred_length = 90  # Length of decoder output

def create_sequences(data, seq_len, pred_len):
    """
    Non-overlapping sequences, each with seq_len inputs and pred_len future outputs.
    """
    X, y = [], []
    num_sequences = (len(data) - seq_len - pred_len) // pred_len
    for i in range(num_sequences):
        start_idx = i * pred_len
        X.append(data[start_idx:start_idx + seq_len])
        y.append(data[start_idx + seq_len:start_idx + seq_len + pred_len])
    return np.array(X), np.array(y)

X_train_full, y_train_full = create_sequences(train_raw[target_columns].values, seq_length, pred_length)
X_test, y_test = create_sequences(test_raw[target_columns].values, seq_length, pred_length)

# Train/Validation split
val_split = int(0.8 * len(X_train_full))
X_train, y_train = X_train_full[:val_split], y_train_full[:val_split]
X_val, y_val = X_train_full[val_split:], y_train_full[val_split:]

# Convert to float32 (for Keras)
X_train_np = X_train.astype('float32')
y_train_np = y_train.astype('float32')
X_val_np   = X_val.astype('float32')
y_val_np   = y_val.astype('float32')
X_test_np  = X_test.astype('float32')
y_test_np  = y_test.astype('float32')

# ------------------- Naive Baseline (For MASE) ------------------- #
logger.info("Computing naive forecasts for MASE calculation")
naive_forecast_train = X_train_np[:, -1, :]  # shape: (batch, n_features)
naive_forecast_train = np.expand_dims(naive_forecast_train, axis=1)  # (batch, 1, n_features)
naive_forecast_train = np.repeat(naive_forecast_train, pred_length, axis=1)  # (batch, pred_length, n_features)
naive_mae_per_target = np.mean(np.abs(y_train_np - naive_forecast_train), axis=(0, 1))
naive_mae_dict = {col: mae for col, mae in zip(target_columns, naive_mae_per_target)}

# ------------------- Multiple Model Configs ------------------- #

model_configs = [   
    {"name": "lstm_1", "hidden_size": 8, "lr": 0.001, "dropout": 0.2},
    {"name": "lstm_2", "hidden_size": 32, "lr": 0.001, "dropout": 0.2},
    {"name": "lstm_3", "hidden_size": 64, "lr": 0.001, "dropout": 0.2},   
    {"name": "lstm_4", "hidden_size": 64, "lr": 0.005, "dropout": 0.2},]

results = {}          # Will store metrics, losses, predictions, etc.
all_metrics_list = [] # For a detailed CSV across all models and targets

# ------------------- LSTM Seq2Seq Model Training Loop ------------------- #
for config in model_configs:
    model_name   = config["name"]
    hidden_size  = config["hidden_size"]
    lr           = config["lr"]
    dropout_rate = config["dropout"]

    logger.info(f"\nTraining {model_name} - hidden_size={hidden_size}, lr={lr}, dropout={dropout_rate}")

    # Build seq2seq LSTM model
    n_features = X_train_np.shape[2]

    model = Sequential()
    # Encoder: compress the 200-step input into single hidden vector
    model.add(
        LSTM(hidden_size, activation='relu',
             input_shape=(seq_length, n_features), return_sequences=False)
    )
    # Repeat vector for 90 steps
    model.add(RepeatVector(pred_length))
    # Decoder: expand back into 90 steps
    model.add(
        LSTM(hidden_size, activation='relu', return_sequences=True)
    )
    # Output layer predicts n_features per time step
    model.add(TimeDistributed(Dense(n_features)))

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mae')

    history = model.fit(
        X_train_np, y_train_np,
        validation_data=(X_val_np, y_val_np),
        epochs=100,
        batch_size=32,
        verbose=0  # set to 1 if you want epoch logs
    )

    # Evaluate on test set
    test_preds_np = model.predict(X_test_np)  # shape: (batch, pred_length, n_features)

    # Compute metrics
    metrics_per_target = []
    for i, col in enumerate(target_columns):
        preds_col  = test_preds_np[:, :, i].flatten()
        actual_col = y_test_np[:, :, i].flatten()

        valid_mask = np.isfinite(preds_col) & np.isfinite(actual_col)
        preds_clean  = preds_col[valid_mask]
        actual_clean = actual_col[valid_mask]

        if len(preds_clean) == 0 or len(actual_clean) == 0:
            metric_dict = {
                "target": col,
                "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan,
                "MAPE": np.nan, "R2": np.nan, "MASE": np.nan,
                "CRPS": np.nan
            }
        else:
            mae = np.mean(np.abs(preds_clean - actual_clean))
            mse = np.mean((preds_clean - actual_clean) ** 2)
            rmse = np.sqrt(mse)
            mask = actual_clean != 0
            if mask.any():
                mape = np.mean(np.abs((actual_clean[mask] - preds_clean[mask]) / actual_clean[mask])) * 100
            else:
                mape = np.nan

            if len(actual_clean) < 2 or np.var(actual_clean) == 0:
                r2 = np.nan
            else:
                r2 = r2_score(actual_clean, preds_clean)

            mase = mae / naive_mae_dict[col]
            crps = mae  # Simplified CRPS

            metric_dict = {
                "target": col,
                "MAE": mae, "MSE": mse, "RMSE": rmse,
                "MAPE": mape, "R2": r2, "MASE": mase,
                "CRPS": crps
            }

        metrics_per_target.append(metric_dict)

    aggregated = {
        "MAE":  np.nanmean([m["MAE"]  for m in metrics_per_target]),
        "MSE":  np.nanmean([m["MSE"]  for m in metrics_per_target]),
        "RMSE": np.nanmean([m["RMSE"] for m in metrics_per_target]),
        "MAPE": np.nanmean([m["MAPE"] for m in metrics_per_target]),
        "R2":   np.nanmean([m["R2"]   for m in metrics_per_target]),
        "MASE": np.nanmean([m["MASE"] for m in metrics_per_target]),
        "CRPS": np.nanmean([m["CRPS"] for m in metrics_per_target]),
    }

    # Store results
    results[model_name] = {
        "config":       config,
        "train_history": history.history['loss'],
        "val_history":   history.history['val_loss'],
        "metrics": {
            "per_target": metrics_per_target,
            "aggregated": aggregated
        },
        "predictions": test_preds_np,  # (batch, pred_length, n_features)
    }

    # Collect for big CSV
    for m in metrics_per_target:
        row = {"model": model_name, **m}
        all_metrics_list.append(row)

    logger.info(f"{model_name} aggregated metrics: {aggregated}")

# ------------------- Save Detailed Metrics CSV ------------------- #
all_metrics_df = pd.DataFrame(all_metrics_list)
all_metrics_df.to_csv(os.path.join(base_plot_dir, "detailed_metrics_all_models.csv"), index=False)

# ------------------- Create Summary Heatmap ------------------- #
summary_rows = []
for model_name, info in results.items():
    row = {
        "Model": model_name,
        **info["metrics"]["aggregated"],
        "Hidden Size": info["config"]["hidden_size"],
        "Learning Rate": info["config"]["lr"],
        "Dropout": info["config"]["dropout"]
    }
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)

plt.figure(figsize=(12, 6))
numeric_cols = ["MAE", "MSE", "RMSE", "MAPE", "R2", "MASE", "CRPS"]
heatmap_df = summary_df.set_index("Model")[numeric_cols]
sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Seq2Seq LSTM Model Performance Comparison")
plt.tight_layout()
plt.savefig(os.path.join(base_plot_dir, "seq2seq_performance_heatmap.png"), dpi=300)
plt.close()

# ------------------- Plot Loss Comparisons ------------------- #
plt.figure(figsize=(10, 5))
for model_name, info in results.items():
    plt.plot(info["train_history"], label=f"{model_name} (Train)")
plt.title("Seq2Seq LSTM Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_plot_dir, "seq2seq_training_loss_comparison.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
for model_name, info in results.items():
    plt.plot(info["val_history"], label=f"{model_name} (Val)")
plt.title("Seq2Seq LSTM Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_plot_dir, "seq2seq_validation_loss_comparison.png"), dpi=300)
plt.close()

# ------------------- Individual Forecast Plots (Grid per Column) ------------------- #
# We'll pick a single example from the test set to visualize (sample_idx=0).
sample_idx = 0
target_metadata = {
    "scd41_co2":         {"unit": "ppm",    "name": "CO₂"},
    "scd41_temperature": {"unit": "°C",     "name": "Temp"},
    "svm41_nox_index":   {"unit": "index",  "name": "NOx"},
    "scd41_humidity":    {"unit": "% RH",   "name": "Humidity"},
    "ens160_eco2":       {"unit": "ppm",    "name": "eCO₂"},
    "ens160_tvoc":       {"unit": "ppb",    "name": "TVOC"},
}

def inverse_transform(col, values):
    """
    Inverse the MinMax scaling for plotting.
    """
    return scalers[col].inverse_transform(values.reshape(-1, 1)).flatten()

# For each target column: create ONE figure with subplots for each model config
n_configs = len(results)
n_cols = 3  # e.g., 3 columns
n_rows = int(np.ceil(n_configs / n_cols))

for col in target_columns:
    col_idx = target_columns.index(col)
    # Prepare the actual (true) series
    actual_values_scaled = y_test_np[sample_idx, :, col_idx]
    actual_values = inverse_transform(col, actual_values_scaled)
    time_axis = np.arange(pred_length)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=False)
    # If there's only 1 row/col, axes might not be 2D
    axes = axes.flatten() if n_configs > 1 else [axes]

    fig.suptitle(f"{target_metadata[col]['name']} - 90-step Forecast\n({target_metadata[col]['unit']})",
                 fontsize=16, y=1.03)

    for i, (model_name, info) in enumerate(results.items()):
        ax = axes[i]
        # Model config
        preds_values_scaled = info["predictions"][sample_idx, :, col_idx]
        preds_values = inverse_transform(col, preds_values_scaled)

        ax.plot(time_axis, actual_values, label='Actual', marker='o', color='black')
        ax.plot(time_axis, preds_values, label='Predicted', marker='x', color='orange')

        # Subplot title with model config
        c = info["config"]
        ax.set_title(f"{model_name}\nHS={c['hidden_size']}, LR={c['lr']}, Dropout={c['dropout']}")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel(f"{target_metadata[col]['unit']}")

        # Optionally, put a legend on each subplot or only for the top-left
        if i == 0:
            ax.legend()

    # Turn off extra blank subplots if the model count doesn't fill the grid
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    save_path = os.path.join(base_plot_dir, f"grid_forecast_{col}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved grid forecast for {col} to {save_path}")

# ------------------- Save Final Summary CSV ------------------- #
summary_csv_path = os.path.join(base_plot_dir, "seq2seq_performance_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
logger.info(f"Saved aggregated performance summary to {summary_csv_path}")

logger.info("\nAll models trained and per-column forecasts plotted.")
logger.info("Performance Summary (Aggregated):")
print(summary_df.to_markdown(index=False, floatfmt=".4f"))