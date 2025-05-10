import os
import warnings
import logging
import uuid
from datetime import datetime
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, QuantileLoss
from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('tft_forecasting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create plots directory
PLOT_DIR = 'plots'
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)
    logger.info(f"Created directory: {PLOT_DIR}")

T_start = time.time()

# Device setup
device = torch.device('cpu')
logger.info(f'Using {device} device')

# Load data
try:
    wd = pd.read_csv("preprocessed_data.csv")
    logger.info("Successfully loaded preprocessed_data.csv")
except Exception as e:
    logger.error(f"Failed to load preprocessed_data.csv: {str(e)}")
    raise

# Define required columns
required_columns = [
    "timestamp", "station_id", "scd41_co2", "scd41_temperature", "scd41_humidity",
    "ens160_eco2", "ens160_tvoc", "ens160_aqi", "svm41_temperature", "svm41_humidity",
    "svm41_nox_index", "svm41_voc_index", "sfa30_temperature", "sfa30_humidity", "sfa30_hco",
    "bme688_temperature", "bme688_humidity", "bme688_pressure", "bme688_gas_resistance",
    "scd41_co2_diff", "scd41_temperature_diff", "scd41_humidity_diff",
    "ens160_eco2_diff", "ens160_tvoc_diff", "ens160_aqi_diff",
    "svm41_temperature_diff", "svm41_humidity_diff", "svm41_nox_index_diff", "svm41_voc_index_diff",
    "sfa30_temperature_diff", "sfa30_humidity_diff", "sfa30_hco_diff",
    "hour_sin", "hour_cos", "week_sin", "week_cos",
    "scd41_co2_lag24", "scd41_co2_lag168", "scd41_temperature_lag24", "scd41_temperature_lag168",
    "scd41_humidity_lag24", "scd41_humidity_lag168", "ens160_eco2_lag24", "ens160_eco2_lag168",
    "ens160_tvoc_lag24", "ens160_tvoc_lag168", "ens160_aqi_lag24", "ens160_aqi_lag168",
    "svm41_temperature_lag24", "svm41_temperature_lag168", "svm41_humidity_lag24", "svm41_humidity_lag168",
    "svm41_nox_index_lag24", "svm41_nox_index_lag168", "svm41_voc_index_lag24", "svm41_voc_index_lag168",
    "sfa30_temperature_lag24", "sfa30_temperature_lag168", "sfa30_humidity_lag24", "sfa30_humidity_lag168",
    "sfa30_hco_lag24", "sfa30_hco_lag168"
]
for col in required_columns:
    if col not in wd.columns:
        logger.error(f"Missing required column: {col}")
        raise ValueError(f"Missing required column: {col}")

# Preprocessing
wd['timestamp'] = pd.to_datetime(wd['timestamp'])
wd = wd.sort_values(by='timestamp').reset_index(drop=True)
wd['timeindex'] = np.arange(len(wd))
wd['station_id'] = wd['station_id'].astype(str)

# Derive month and hour for TimeSeriesDataSet
wd['month'] = wd['timestamp'].dt.month.astype(str)
wd['hour'] = wd['timestamp'].dt.hour.astype(str)
logger.info("Derived month and hour from timestamp")

# Impute missing values for numeric columns
numeric_cols = [
    "scd41_co2", "scd41_temperature", "scd41_humidity",
    "ens160_eco2", "ens160_tvoc", "ens160_aqi",
    "svm41_temperature", "svm41_humidity", "svm41_nox_index", "svm41_voc_index",
    "sfa30_temperature", "sfa30_humidity", "sfa30_hco",
    "bme688_temperature", "bme688_humidity", "bme688_pressure", "bme688_gas_resistance",
    "scd41_co2_diff", "scd41_temperature_diff", "scd41_humidity_diff",
    "ens160_eco2_diff", "ens160_tvoc_diff", "ens160_aqi_diff",
    "svm41_temperature_diff", "svm41_humidity_diff", "svm41_nox_index_diff", "svm41_voc_index_diff",
    "sfa30_temperature_diff", "sfa30_humidity_diff", "sfa30_hco_diff",
    "hour_sin", "hour_cos", "week_sin", "week_cos",
    "scd41_co2_lag24", "scd41_co2_lag168", "scd41_temperature_lag24", "scd41_temperature_lag168",
    "scd41_humidity_lag24", "scd41_humidity_lag168", "ens160_eco2_lag24", "ens160_eco2_lag168",
    "ens160_tvoc_lag24", "ens160_tvoc_lag168", "ens160_aqi_lag24", "ens160_aqi_lag168",
    "svm41_temperature_lag24", "svm41_temperature_lag168", "svm41_humidity_lag24", "svm41_humidity_lag168",
    "svm41_nox_index_lag24", "svm41_nox_index_lag168", "svm41_voc_index_lag24", "svm41_voc_index_lag168",
    "sfa30_temperature_lag24", "sfa30_temperature_lag168", "sfa30_humidity_lag24", "sfa30_humidity_lag168",
    "sfa30_hco_lag24", "sfa30_hco_lag168"
]
for col in numeric_cols:
    wd[col] = wd[col].fillna(wd[col].mean())
logger.info("Imputed missing values with column means")

# Drop unnecessary columns
wd = wd.drop(columns=['station_data_id', 'station_name', 'scd41_id', 'ens160_id', 'sim41_id', 'bme688_id', 'sfa30_id'], errors='ignore')
logger.info("Dropped unnecessary columns")

# Normalize numeric columns
scaler = MinMaxScaler()
wd[numeric_cols] = scaler.fit_transform(wd[numeric_cols])
logger.info("Normalized numeric columns using MinMaxScaler")

# Split data
split_index = int(0.8 * len(wd))
wd1 = wd[:split_index]
wd_11 = wd[split_index:]
logger.info(f"Split data: Training shape={wd1.shape}, Test shape={wd_11.shape}")

# Additional normalization for target variable
mm = MinMaxScaler()
f = mm.fit(wd1['scd41_co2'].values.reshape(-1, 1))
wd1['scd41_co2'] = f.transform(wd1['scd41_co2'].to_numpy().reshape(-1, 1))
f11 = mm.fit(wd_11['scd41_co2'].values.reshape(-1, 1))
wd_11['scd41_co2'] = f11.transform(wd_11['scd41_co2'].to_numpy().reshape(-1, 1))
logger.info("Normalized target variable (scd41_co2) for train and test sets")

# Create TimeSeriesDataSet
max_prediction_length = 10
max_encoder_length = 200
training_cutoff = wd1["timeindex"].max() - max_prediction_length
logger.info(f"Training cutoff: {training_cutoff}")

time_varying_known_reals = ["timeindex", "hour_sin", "hour_cos", "week_sin", "week_cos"]
time_varying_unknown_reals = [col for col in numeric_cols if col != "scd41_co2"]

training = TimeSeriesDataSet(
    wd1[lambda x: x.timeindex <= training_cutoff],
    time_idx="timeindex",
    target="scd41_co2",
    group_ids=["station_id"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length // 2,
    max_prediction_length=max_prediction_length,
    static_categoricals=["station_id"],
    static_reals=[],
    time_varying_known_categoricals=["month", "hour"],
    time_varying_known_reals=time_varying_known_reals + ["timeindex"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=time_varying_unknown_reals + ["scd41_temperature", "scd41_humidity"],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=False,
    target_normalizer=TorchNormalizer(method="standard", transformation="softplus"),
    categorical_encoders={"station_id": NaNLabelEncoder(add_nan=True)},
    randomize_length=None,
    predict_mode=False
)
logger.info("Created training TimeSeriesDataSet")

validation = TimeSeriesDataSet.from_dataset(training, wd1, predict=True, stop_randomization=True)
logger.info("Created validation TimeSeriesDataSet")

# Create dataloaders
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
logger.info("Created train and validation DataLoaders")

# Baseline model
logger.info("Calculating baseline model predictions")
baseline_predictions = Baseline().predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
baseline_mae = MAE()(baseline_predictions.output, baseline_predictions.y[0])
logger.info(f"Baseline MAE: {baseline_mae:.4f} (normalized units)")

# Train the model
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
logger_tb = TensorBoardLogger("lightning_logs")

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=50,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger_tb,
)

tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.0005,
    hidden_size=64,
    attention_head_size=16,
    dropout=0.1,
    hidden_continuous_size=32,
    loss=QuantileLoss(),
    log_interval=10,
    optimizer="Adam",
    reduce_on_plateau_patience=4,
)
logger.info(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# Fit the model
logger.info("Starting model training")
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
logger.info("Training complete")

# Load best model
logger.info("Loading best model from checkpoint")
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
logger.info(f"Loaded best model from {best_model_path}")

# Evaluate on validation set
logger.info("Calculating validation set predictions")
predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
val_mae = MAE()(predictions.output, predictions.y[0])
logger.info(f"Validation MAE: {val_mae:.4f} (normalized units)")

# Plot predictions
logger.info("Generating and saving validation prediction plots")
raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))
for idx in range(1):
    station_id = raw_predictions.x['decoder_cat'][idx][0].item()  # Assuming station_id is the first categorical
    plt.figure(figsize=(10, 6))
    best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
    plt.title(f"CO2 Prediction (ppm) for Station {station_id}, Index {idx}")
    plt.ylabel("CO2 Concentration (ppm, normalized)")
    plt.xlabel("Time Step")
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"prediction_validation_station_{station_id}_idx_{idx}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved validation prediction plot: {plot_path}")

# Variable importance
logger.info("Generating and saving validation variable importance plot")
interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
plt.figure(figsize=(12, 8))
best_tft.plot_interpretation(interpretation)
plt.title("Variable Importance for CO2 Prediction (ppm)")
plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, "interpretation_validation.png")
plt.savefig(plot_path)
plt.close()
logger.info(f"Saved validation interpretation plot: {plot_path}")

# Test dataset
test = TimeSeriesDataSet(
    wd_11[lambda x: x.timeindex > training_cutoff/2],
    time_idx="timeindex",
    target="scd41_co2",
    group_ids=["station_id"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=["station_id"],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=time_varying_known_reals,
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=time_varying_unknown_reals,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True
)
logger.info("Created test TimeSeriesDataSet")

test_dataloader = test.to_dataloader(train=False, batch_size=(batch_size * 10), num_workers=0)
logger.info("Created test DataLoader")

# Test predictions
logger.info("Generating and saving test prediction plots")
raw_predictions_test = best_tft.predict(test_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))
for idx in range(5):
    station_id = raw_predictions_test.x['decoder_cat'][idx][0].item()
    plt.figure(figsize=(10, 6))
    best_tft.plot_prediction(raw_predictions_test.x, raw_predictions_test.output, idx=idx, add_loss_to_title=True)
    plt.title(f"CO2 Prediction (ppm) for Station {station_id}, Index {idx}")
    plt.ylabel("CO2 Concentration (ppm, normalized)")
    plt.xlabel("Time Step")
    plt.tight_layout()
    plot_path = os.path.join(PLOT_DIR, f"prediction_test_station_{station_id}_idx_{idx}.png")
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved test prediction plot: {plot_path}")

# Test interpretation
logger.info("Generating and saving test variable importance plot")
interpretation = best_tft.interpret_output(raw_predictions_test.output, reduction="sum")
plt.figure(figsize=(12, 8))
best_tft.plot_interpretation(interpretation)
plt.title("Variable Importance for CO2 Prediction (ppm)")
plt.tight_layout()
plot_path = os.path.join(PLOT_DIR, "interpretation_test.png")
plt.savefig(plot_path)
plt.close()
logger.info(f"Saved test interpretation plot: {plot_path}")

# Additional test metrics
test1 = TimeSeriesDataSet.from_dataset(test, wd_11, predict=True, stop_randomization=True)
test_dataloader1 = test1.to_dataloader(train=False, batch_size=(batch_size * 10), num_workers=0)
logger.info("Created additional test DataLoader for metrics")
predictions1 = best_tft.predict(test_dataloader1, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
mse = torch.nn.functional.mse_loss(predictions1.output, predictions1.y[0])
mae = torch.nn.functional.l1_loss(predictions1.output, predictions1.y[0])
logger.info(f"Test MSE: {mse:.4f} (normalized units)")
logger.info(f"Test MAE: {mae:.4f} (normalized units)")

T_end = time.time()
T_total = T_end - T_start
logger.info(f"Total execution time: {T_total:.2f} seconds")