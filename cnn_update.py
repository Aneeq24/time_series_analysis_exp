#!/usr/bin/env python3

import os
import logging
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import keras
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Flatten, Dense, Reshape

# ----------------------------------------------------------------------------
# Logging Setup
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Create Plot Output Directory
# ----------------------------------------------------------------------------
base_plot_dir = 'model_comparison_plots_cnn'
if not os.path.exists(base_plot_dir):
    os.makedirs(base_plot_dir)
logger.info(f"Created directory '{base_plot_dir}' for saving figures")

# ----------------------------------------------------------------------------
# Load and Preprocess Data
# ----------------------------------------------------------------------------
logger.info("Loading and preprocessing data")
wd = pd.read_csv("preprocessed_data.csv")  # Replace with your file path

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

# Temporal split
wd['timestamp'] = pd.to_datetime(wd['timestamp'])
wd.sort_values(by='timestamp', inplace=True)
wd.reset_index(drop=True, inplace=True)

split_idx = int(0.8 * len(wd))
train_raw = wd.iloc[:split_idx].copy()
test_raw  = wd.iloc[split_idx:].copy()

# Handle missing values using training means
train_means = {}
for col in target_columns:
    train_means[col] = train_raw[col].mean()
    train_raw[col].fillna(train_means[col], inplace=True)
    test_raw[col].fillna(train_means[col], inplace=True)

# Fit MinMaxScaler on training, then transform train and test
scalers = {}
for col in target_columns:
    sc = MinMaxScaler()
    sc.fit(train_raw[col].values.reshape(-1, 1))
    scalers[col] = sc
    train_raw[col] = sc.transform(train_raw[col].values.reshape(-1, 1))
    test_raw[col]  = sc.transform(test_raw[col].values.reshape(-1, 1))

# ----------------------------------------------------------------------------
# Create Sequences
# ----------------------------------------------------------------------------
seq_length = 200  # Input window size
pred_length = 90  # Forecast horizon

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    num_sequences = (len(data) - seq_len - pred_len) // pred_len
    for i in range(num_sequences):
        start_idx = i * pred_len
        X.append(data[start_idx:start_idx + seq_len])
        y.append(data[start_idx + seq_len:start_idx + seq_len + pred_len])
    return np.array(X), np.array(y)

X_train_full, y_train_full = create_sequences(train_raw[target_columns].values, seq_length, pred_length)
X_test, y_test = create_sequences(test_raw[target_columns].values, seq_length, pred_length)

# Split train into train/validation
val_split = int(0.8 * len(X_train_full))
X_train, y_train = X_train_full[:val_split], y_train_full[:val_split]
X_val,   y_val   = X_train_full[val_split:], y_train_full[val_split:]

X_train_np = X_train.astype('float32')
y_train_np = y_train.astype('float32')
X_val_np   = X_val.astype('float32')
y_val_np   = y_val.astype('float32')
X_test_np  = X_test.astype('float32')
y_test_np  = y_test.astype('float32')

# ----------------------------------------------------------------------------
# Compute Naive Baseline (for MASE)
# ----------------------------------------------------------------------------
logger.info("Computing naive forecasts for MASE")
naive_forecast_train = X_train_np[:, -1, :]  # shape: (batch, n_features)
naive_forecast_train = np.expand_dims(naive_forecast_train, axis=1)
naive_forecast_train = np.repeat(naive_forecast_train, pred_length, axis=1)
naive_mae_per_target = np.mean(np.abs(y_train_np - naive_forecast_train), axis=(0, 1))
naive_mae_dict = {col: val for col, val in zip(target_columns, naive_mae_per_target)}

# ----------------------------------------------------------------------------
# CNN Model Configurations
# ----------------------------------------------------------------------------
model_configs = [
    {"name": "cnn_1", "filters": 16, "lr": 0.001, "dropout": 0.2},
    {"name": "cnn_2", "filters": 32, "lr": 0.001, "dropout": 0.2},
    {"name": "cnn_3", "filters": 64, "lr": 0.001, "dropout": 0.2},
    {"name": "cnn_4", "filters": 32, "lr": 0.005, "dropout": 0.2},
]

results = {}
all_metrics_list = []

# ----------------------------------------------------------------------------
# Train Each CNN Model
# ----------------------------------------------------------------------------
for config in model_configs:
    mname = config["name"]
    num_filters = config["filters"]
    lr = config["lr"]
    dr = config["dropout"]

    logger.info(f"\nTraining {mname}, filters={num_filters}, lr={lr}, dropout={dr}")

    n_features = X_train_np.shape[2]

    # Simple CNN approach
    model = Sequential()
    model.add(Conv1D(filters=num_filters, kernel_size=3, activation='relu',
                     padding='causal', input_shape=(seq_length, n_features)))
    model.add(Conv1D(filters=num_filters, kernel_size=3, activation='relu', padding='causal'))
    model.add(Dropout(dr))
    model.add(Flatten())
    model.add(Dense(pred_length * n_features))
    model.add(Reshape((pred_length, n_features)))

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mae')
    model.summary(print_fn=lambda x: logger.info(x))

    history = model.fit(
        X_train_np, y_train_np,
        validation_data=(X_val_np, y_val_np),
        epochs=100,
        batch_size=32,
        verbose=0
    )

    # Evaluate on test data
    test_preds_np = model.predict(X_test_np)  # shape: (batch, pred_length, n_features)

    # Calculate metrics
    metrics_per_target = []
    for i, col in enumerate(target_columns):
        preds_col = test_preds_np[:, :, i].flatten()
        actual_col = y_test_np[:, :, i].flatten()

        valid_mask = np.isfinite(preds_col) & np.isfinite(actual_col)
        preds_clean = preds_col[valid_mask]
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
            mse = np.mean((preds_clean - actual_clean)**2)
            rmse = np.sqrt(mse)
            mask = (actual_clean != 0)
            if mask.any():
                mape = np.mean(np.abs((actual_clean[mask] - preds_clean[mask]) / actual_clean[mask])) * 100
            else:
                mape = np.nan
            if len(actual_clean) >= 2 and np.var(actual_clean) != 0:
                r2 = r2_score(actual_clean, preds_clean)
            else:
                r2 = np.nan
            mase = mae / naive_mae_dict[col]
            crps = mae  # simplified

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

    results[mname] = {
        "config": config,
        "train_history": history.history['loss'],
        "val_history": history.history['val_loss'],
        "metrics": {
            "per_target": metrics_per_target,
            "aggregated": aggregated
        },
        "predictions": test_preds_np,
    }

    for met in metrics_per_target:
        row = {"model": mname, **met}
        all_metrics_list.append(row)

    logger.info(f"{mname} aggregated metrics: {aggregated}")

# ----------------------------------------------------------------------------
# Save Detailed Metrics
# ----------------------------------------------------------------------------
detailed_df = pd.DataFrame(all_metrics_list)
detailed_df.to_csv(os.path.join(base_plot_dir, "detailed_metrics_all_cnn_models.csv"), index=False)

# ----------------------------------------------------------------------------
# Build a Summary Heatmap
# ----------------------------------------------------------------------------
summary_rows = []
for model_name, info in results.items():
    row = {
        "Model": model_name,
        **info["metrics"]["aggregated"],
        "Filters": info["config"]["filters"],
        "LearningRate": info["config"]["lr"],
        "Dropout": info["config"]["dropout"]
    }
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)

plt.figure(figsize=(10, 5))
numeric_cols = ["MAE", "MSE", "RMSE", "MAPE", "R2", "MASE", "CRPS"]
heatmap_df = summary_df.set_index("Model")[numeric_cols]
sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("CNN Model Performance Comparison")
plt.tight_layout()
plt.savefig(os.path.join(base_plot_dir, "cnn_model_performance_heatmap.png"), dpi=300)
plt.close()

# ----------------------------------------------------------------------------
# Training/Validation Loss Comparison
# ----------------------------------------------------------------------------
plt.figure(figsize=(10, 5))
for model_name, info in results.items():
    plt.plot(info["train_history"], label=f"{model_name} (Train)")
plt.title("CNN Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_plot_dir, "cnn_training_loss_comparison.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
for model_name, info in results.items():
    plt.plot(info["val_history"], label=f"{model_name} (Val)")
plt.title("CNN Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("MAE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_plot_dir, "cnn_validation_loss_comparison.png"), dpi=300)
plt.close()

# ----------------------------------------------------------------------------
# Forecast Plots for Each Column
# ----------------------------------------------------------------------------
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
    return scalers[col].inverse_transform(values.reshape(-1, 1)).flatten()

# Optionally, compute real min/max from the entire dataset (pre-scaling):
real_minmax = {}
for col in target_columns:
    col_vals = wd[col].dropna().values
    real_minmax[col] = (col_vals.min(), col_vals.max())

num_models = len(results)
grid_cols = 2
grid_rows = math.ceil(num_models / grid_cols)

for col in target_columns:
    col_idx = target_columns.index(col)

    # Real min / max for y-axis
    col_min, col_max = real_minmax[col]
    margin = 0.1 * (col_max - col_min)
    y_min, y_max = col_min - margin, col_max + margin

    actual_col_scaled = y_test_np[sample_idx, :, col_idx]
    actual_col_inv = inverse_transform(col, actual_col_scaled)
    time_axis = np.arange(pred_length)

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(7 * grid_cols, 4 * grid_rows), sharex=True)
    if grid_rows * grid_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    fig.suptitle(f"{target_metadata[col]['name']} Forecast Comparison ({target_metadata[col]['unit']})",
                 fontsize=14, y=1.03)

    for i, (model_name, info) in enumerate(results.items()):
        ax = axes[i]
        preds_col_scaled = info["predictions"][sample_idx, :, col_idx]
        preds_col_inv = inverse_transform(col, preds_col_scaled)

        ax.plot(time_axis, actual_col_inv, label='Actual', color='black', marker='o', linewidth=1)
        ax.plot(time_axis, preds_col_inv,  label=model_name, color='orange', marker='x', linestyle='--', linewidth=1)

        ax.set_ylim([y_min, y_max])
        cfg = info["config"]
        ax.set_title(f"{model_name}\nFilters={cfg['filters']}, LR={cfg['lr']}, Dropout={cfg['dropout']}")
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel(target_metadata[col]['unit'])
        ax.grid(True)
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outname = f"cnn_compare_preds_grid_{col}.png"
    plt.savefig(os.path.join(base_plot_dir, outname), dpi=300)
    plt.close()
    logger.info(f"Saved multi-subplot forecast comparison for '{col}' to '{outname}'")

logger.info("\nAll CNN models trained and compared successfully.")
logger.info("Performance Summary (aggregated):")
print(summary_df.to_markdown(index=False, floatfmt='.4f'))

summary_csv_path = os.path.join(base_plot_dir, "cnn_performance_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
logger.info(f"Saved summary of CNN performance to {summary_csv_path}")