import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import logging
import os
import seaborn as sns
from sklearn.metrics import r2_score
from torch.cuda.amp import GradScaler, autocast

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create directories for saving plots
base_plot_dir = 'model_comparison_plots_transformer'
if not os.path.exists(base_plot_dir):
    os.makedirs(base_plot_dir)
logger.info(f"Created directory '{base_plot_dir}' for saving figures")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Using {device} device')

# Load and preprocess data
logger.info("Loading and preprocessing data")
wd = pd.read_csv("preprocessed_data.csv")  # Update with your file path

target_columns = [
    "scd41_co2", "scd41_temperature", "svm41_nox_index",
    "scd41_humidity", "ens160_eco2", "ens160_tvoc",
]
required_columns = ["timestamp", "station_id"] + target_columns

# Validate columns
for col in required_columns:
    if col not in wd.columns:
        logger.error(f"Missing required column: {col}")
        raise ValueError(f"Missing required column: {col}")

# Temporal split first to prevent leakage
wd['timestamp'] = pd.to_datetime(wd['timestamp'])
wd = wd.sort_values(by='timestamp').reset_index(drop=True)
split_idx = int(0.8 * len(wd))
train_raw = wd.iloc[:split_idx]
test_raw = wd.iloc[split_idx:]

# Handle missing values using training data only
train_means = {}
for col in target_columns:
    train_means[col] = train_raw[col].mean()
    train_raw[col] = train_raw[col].fillna(train_means[col])
    test_raw[col] = test_raw[col].fillna(train_means[col])  # Use train mean for test

# Normalize using training data statistics
scalers = {}
for col in target_columns:
    scaler = MinMaxScaler()
    scaler.fit(train_raw[col].values.reshape(-1, 1))
    scalers[col] = scaler
    train_raw[col] = scaler.transform(train_raw[col].values.reshape(-1, 1))
    test_raw[col] = scaler.transform(test_raw[col].values.reshape(-1, 1))

# Sequence parameters
seq_length = 200
pred_length = 90

def create_sequences(data, seq_length, pred_length):
    """Create non-overlapping sequences from preprocessed data"""
    X, y = [], []
    num_sequences = (len(data) - seq_length - pred_length) // pred_length
    for i in range(num_sequences):
        start_idx = i * pred_length
        X.append(data[start_idx:start_idx+seq_length])
        y.append(data[start_idx+seq_length:start_idx+seq_length+pred_length])
    return np.array(X), np.array(y)

# Create isolated sequences
X_train_full, y_train_full = create_sequences(train_raw[target_columns].values, seq_length, pred_length)
X_test, y_test = create_sequences(test_raw[target_columns].values, seq_length, pred_length)

# Split training data into train and validation
val_split = int(0.8 * len(X_train_full))
X_train, y_train = X_train_full[:val_split], y_train_full[:val_split]
X_val, y_val = X_train_full[val_split:], y_train_full[val_split:]

# Convert to tensors
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
X_val = torch.FloatTensor(X_val).to(device)
y_val = torch.FloatTensor(y_val).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# Compute naive forecasts for MASE calculation
logger.info("Computing naive forecasts for MASE calculation")
naive_forecast_train = X_train[:, -1, :].unsqueeze(1).repeat(1, pred_length, 1)
naive_mae_per_target = torch.mean(torch.abs(y_train - naive_forecast_train), dim=(0, 1)).cpu().numpy()
naive_mae_dict = {col: mae for col, mae in zip(target_columns, naive_mae_per_target)}

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# Transformer Model
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, num_targets, dropout_rate=0.2):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(d_model * seq_length, num_targets * pred_length)

    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        x = self.input_fc(x)  # (batch_size, seq_length, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_length, d_model)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # (batch_size, seq_length * d_model)
        x = self.fc(x)  # (batch_size, num_targets * pred_length)
        x = x.view(x.size(0), pred_length, -1)  # (batch_size, pred_length, num_targets)
        return x

# Model configurations
model_configs = [
    {"name": "model_1", "d_model": 32, "num_heads": 2, "num_layers": 2, "lr": 0.005, "dropout": 0.2},
    {"name": "model_2", "d_model": 64, "num_heads": 4, "num_layers": 2, "lr": 0.005, "dropout": 0.2},
    {"name": "model_3", "d_model": 32, "num_heads": 2, "num_layers": 2, "lr": 0.001, "dropout": 0.2},
    {"name": "model_4", "d_model": 64, "num_heads": 4, "num_layers": 2, "lr": 0.001, "dropout": 0.2},
    {"name": "model_5", "d_model": 32, "num_heads": 2, "num_layers": 2, "lr": 0.05, "dropout": 0.2},
    {"name": "model_6", "d_model": 64, "num_heads": 4, "num_layers": 2, "lr": 0.05, "dropout": 0.2},
    {"name": "model_7", "d_model": 64, "num_heads": 8, "num_layers": 2, "lr": 0.005, "dropout": 0.2},
    {"name": "model_8", "d_model": 64, "num_heads": 8, "num_layers": 2, "lr": 0.001, "dropout": 0.2},
    {"name": "model_9", "d_model": 128, "num_heads": 8, "num_layers": 2, "lr": 0.005, "dropout": 0.2},
]

# Training metadata
target_metadata = {
    "scd41_co2": {"unit": "ppm", "name": "CO₂"},
    "scd41_temperature": {"unit": "°C", "name": "Temp"},
    "svm41_nox_index": {"unit": "index", "name": "NOx"},
    "scd41_humidity": {"unit": "% RH", "name": "Humidity"},
    "ens160_eco2": {"unit": "ppm", "name": "eCO₂"},
    "ens160_tvoc": {"unit": "ppb", "name": "TVOC"},
}

# Training and evaluation
all_metrics = []
results = {}
scaler = GradScaler() if device.type == 'cuda' else None

for config in model_configs:
    model_name = config["name"]
    logger.info(f"\n{'='*50}")
    logger.info(f"Training {model_name} with d_model={config['d_model']}, num_heads={config['num_heads']}, num_layers={config['num_layers']}, lr={config['lr']}, dropout={config['dropout']}")
    
    # Model initialization
    model = SimpleTransformer(
        input_dim=len(target_columns),
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        num_targets=len(target_columns),
        dropout_rate=config["dropout"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    # Training loop with early stopping
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 50
    epochs_no_improve = 0
    
    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        
        if device.type == 'cuda' and scaler:
            with autocast():
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        
        train_losses.append(loss.item())
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
        val_losses.append(val_loss.item())
        
        scheduler.step(val_loss)
        
        if (epoch+1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/500 - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Evaluation and metrics calculation
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test)
        test_loss = criterion(test_preds, y_test)
    
    test_preds_np = test_preds.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    metrics_per_target = []
    for col_idx, col in enumerate(target_columns):
        preds_col = test_preds_np[:, :, col_idx].flatten()
        actual_col = y_test_np[:, :, col_idx].flatten()
        
        # Data cleaning
        valid_mask = np.isfinite(preds_col) & np.isfinite(actual_col)
        preds_clean = preds_col[valid_mask]
        actual_clean = actual_col[valid_mask]
        
        if len(preds_clean) == 0 or len(actual_clean) == 0:
            metrics_per_target.append({
                "target": col, "MAE": np.nan, "MSE": np.nan, "RMSE": np.nan,
                "MAPE": np.nan, "R2": np.nan, "MASE": np.nan, "CRPS": np.nan
            })
            continue
            
        # Calculate metrics
        mae = np.mean(np.abs(preds_clean - actual_clean))
        mse = np.mean((preds_clean - actual_clean)**2)
        rmse = np.sqrt(mse)
        
        # MAPE calculation
        mask = actual_clean != 0
        if not mask.any():
            mape = np.nan
        else:
            mape = np.mean(np.abs((actual_clean[mask] - preds_clean[mask])/actual_clean[mask]))*100
        
        # R² calculation
        if len(actual_clean) < 2 or np.var(actual_clean) == 0:
            r2 = np.nan
        else:
            r2 = r2_score(actual_clean, preds_clean)
        
        # Other metrics
        mase = mae / naive_mae_dict[col]
        crps = mae  # Simplified CRPS
        
        metrics_per_target.append({
            "target": col, "MAE": mae, "MSE": mse, "RMSE": rmse,
            "MAPE": mape, "R2": r2, "MASE": mase, "CRPS": crps
        })
    
    # Store results
    results[model_name] = {
        "config": config,
        "train_loss": train_losses[-1] if train_losses else np.nan,
        "val_loss": val_losses[-1] if val_losses else np.nan,
        "test_loss": test_loss.item(),
        "predictions": test_preds.cpu().numpy(),
        "train_history": train_losses,
        "val_history": val_losses,
        "metrics": {
            "per_target": metrics_per_target,
            "aggregated": {
                "MAE": np.nanmean([m["MAE"] for m in metrics_per_target]),
                "MSE": np.nanmean([m["MSE"] for m in metrics_per_target]),
                "RMSE": np.nanmean([m["RMSE"] for m in metrics_per_target]),
                "MAPE": np.nanmean([m["MAPE"] for m in metrics_per_target]),
                "R2": np.nanmean([m["R2"] for m in metrics_per_target]),
                "MASE": np.nanmean([m["MASE"] for m in metrics_per_target]),
                "CRPS": np.nanmean([m["CRPS"] for m in metrics_per_target])
            }
        }
    }
    
    # Collect for detailed CSV
    all_metrics.extend([{**m, "model": config["name"]} for m in metrics_per_target])

# Save metrics
detailed_metrics_df = pd.DataFrame(all_metrics)
detailed_metrics_df.to_csv(os.path.join(base_plot_dir, "detailed_metrics.csv"), index=False)

# Create comparison heatmap
summary_df = pd.DataFrame([
    {
        "Model": name,
        **res["metrics"]["aggregated"],
        "d_model": res["config"]["d_model"],
        "Num Heads": res["config"]["num_heads"],
        "Num Layers": res["config"]["num_layers"],
        "Learning Rate": res["config"]["lr"],
        "Dropout": res["config"]["dropout"]
    }
    for name, res in results.items()
])

# Plotting parameters
plt.figure(figsize=(14, 8))
heatmap_df = summary_df.set_index("Model")[["MAE", "MSE", "RMSE", "MAPE", "R2", "MASE", "CRPS"]]
ax = sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="YlGnBu")

# Highlight best performers
for metric in heatmap_df.columns:
    if metric == "R2":
        best_val = heatmap_df[metric].max()
    else:
        best_val = heatmap_df[metric].min()
    
    best_models = heatmap_df[heatmap_df[metric] == best_val].index
    for best_model in best_models:
        ax.text(heatmap_df.columns.get_loc(metric)+0.5, 
                heatmap_df.index.get_loc(best_model)+0.5, 
                "*", ha='center', va='center', color='red')

plt.title("Model Performance Comparison\n(* indicates best performance per metric)")
plt.tight_layout()
plt.savefig(os.path.join(base_plot_dir, "metric_comparison_matrix.png"), dpi=300)
plt.close()

# Plotting functions
def inverse_transform(col, values):
    return scalers[col].inverse_transform(values.reshape(-1, 1)).flatten()

# Generate comparison plots for 90 steps
sample_idx = 0  # First sample in test set
for col in target_columns:
    col_idx = target_columns.index(col)
    
    plt.figure(figsize=(18, 12))
    plt.suptitle(f"90-Step Prediction Comparison: {target_metadata[col]['name']} ({target_metadata[col]['unit']})", y=1.02)
    
    for i, (model_name, result) in enumerate(results.items()):
        plt.subplot(3, 3, i+1)
        
        # Get predictions and actual values
        preds = inverse_transform(col, result["predictions"][sample_idx, :, col_idx])
        actual = inverse_transform(col, y_test[sample_idx, :, col_idx].cpu().numpy())
        
        plt.plot(actual, label='Actual', marker='o', markersize=3)
        plt.plot(preds, label='Predicted', marker='x', markersize=3)
        plt.title(f"{model_name}\nd_model={result['config']['d_model']}, Heads={result['config']['num_heads']}, LR={result['config']['lr']}")
        plt.xlabel("Time Step (0-89)")
        plt.ylabel(target_metadata[col]['unit'])
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_dir, f"90step_comparison_{col}.png"), dpi=300)
    plt.close()
    logger.info(f"Saved 90-step comparison plot for {col} to 'model_comparison_plots_transformer/90step_comparison_{col}.png'")

# Actual vs Predicted comparison plots (all models in one plot)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink']  # One color per model
for col in target_columns:
    col_idx = target_columns.index(col)
    
    plt.figure(figsize=(12, 6))
    actual_values = inverse_transform(col, y_test[sample_idx, :, col_idx].cpu().numpy())
    plt.plot(actual_values, label='Actual', color='black', marker='o', linestyle='-', linewidth=2)
    
    for i, (model_name, result) in enumerate(results.items()):
        predicted_values = inverse_transform(col, result["predictions"][sample_idx, :, col_idx])
        plt.plot(predicted_values, label=f'{model_name} (d_model={result["config"]["d_model"]}, Heads={result["config"]["num_heads"]}, LR={result["config"]["lr"]})',
                 color=colors[i], marker='x', linestyle='--', alpha=0.7)
    
    plt.xlabel('Time Step')
    plt.ylabel(f"{target_metadata[col]['name']} ({target_metadata[col]['unit']})")
    plt.title(f'Actual vs Predicted Comparison: {target_metadata[col]["name"]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_dir, f"actual_vs_predicted_comparison_{col}.png"), dpi=300)
    plt.close()
    logger.info(f"Saved actual vs predicted comparison plot for {col} to 'model_comparison_plots_transformer/actual_vs_predicted_comparison_{col}.png'")

# Forecasting comparison plots (all models in one plot)
for col in target_columns:
    col_idx = target_columns.index(col)
    
    plt.figure(figsize=(12, 6))
    
    # Calculate the start index for the test sample
    start_idx = split_idx + sample_idx * pred_length  # Non-overlapping sequences
    logger.info(f"Extracting historical data for {col}, start_idx={start_idx}, end_idx={start_idx + seq_length}, len(wd)={len(wd)}")
    
    if start_idx < 0 or start_idx + seq_length > len(wd):
        logger.error(f"Invalid index range for historical data: start_idx={start_idx}, end_idx={start_idx + seq_length}, len(wd)={len(wd)}")
        raise ValueError("Invalid index range for historical data")
    
    historical_data = wd[col].values[start_idx:start_idx + seq_length]
    historical_data = inverse_transform(col, historical_data)
    
    # Actual and predicted future values
    actual_future = inverse_transform(col, y_test[sample_idx, :, col_idx].cpu().numpy())
    
    time_steps = np.arange(len(historical_data) + len(actual_future))
    historical_time = time_steps[:len(historical_data)]
    future_time = time_steps[len(historical_data):]
    
    plt.plot(historical_time, historical_data, label='Historical', color='blue', linewidth=2)
    plt.plot(future_time, actual_future, label='Actual Future', color='green', marker='o', linestyle='-', linewidth=2)
    
    for i, (model_name, result) in enumerate(results.items()):
        predicted_future = inverse_transform(col, result["predictions"][sample_idx, :, col_idx])
        plt.plot(future_time, predicted_future,
                 label=f'{model_name} (d_model={result["config"]["d_model"]}, Heads={result["config"]["num_heads"]}, LR={result["config"]["lr"]})',
                 color=colors[i], marker='x', linestyle='--', alpha=0.7)
    
    plt.axvline(x=len(historical_data), color='gray', linestyle='--', label='Forecast Start')
    plt.xlabel('Time Step')
    plt.ylabel(f"{target_metadata[col]['name']} ({target_metadata[col]['unit']})")
    plt.title(f'Historical Data and Forecast Comparison: {target_metadata[col]["name"]}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_dir, f"forecasting_comparison_{col}.png"), dpi=300)
    plt.close()
    logger.info(f"Saved forecasting comparison plot for {col} to 'model_comparison_plots_transformer/forecasting_comparison_{col}.png'")

# Training loss comparison
plt.figure(figsize=(12, 6))
for model_name, result in results.items():
    plt.plot(result["train_history"], 
             label=f"{model_name} (d_model={result['config']['d_model']}, Heads={result['config']['num_heads']}, LR={result['config']['lr']})")
plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_plot_dir, "training_loss_comparison.png"), dpi=300)
plt.close()

# Validation loss comparison
plt.figure(figsize=(12, 6))
for model_name, result in results.items():
    plt.plot(result["val_history"], 
             label=f"{model_name} (d_model={result['config']['d_model']}, Heads={result['config']['num_heads']}, LR={result['config']['lr']})")
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_plot_dir, "validation_loss_comparison.png"), dpi=300)
plt.close()

# Performance summary
summary_df.to_csv(os.path.join(base_plot_dir, "performance_summary.csv"), index=False)

logger.info("\nTraining completed successfully")
print("\nPerformance Summary:")
print(summary_df.to_markdown(index=False, floatfmt=".4f"))