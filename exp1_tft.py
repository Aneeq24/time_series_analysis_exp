#!/usr/bin/env python
# coding: utf-8

import os
import warnings
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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import time

T_start = time.time()

# Device setup
device = torch.device('cpu')
print(f'***** using {device} device')

# Load data
wd = pd.read_csv("preprocessed_data.csv")  # Update with your file path

# Define required columns (adjusted for your dataset)
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
        raise ValueError(f"Missing required column: {col}")

# Preprocessing
# Fix: Parse timestamp as datetime string instead of Unix timestamp
wd['timestamp'] = pd.to_datetime(wd['timestamp'])  # Removed unit='s'
wd = wd.sort_values(by='timestamp').reset_index(drop=True)
wd['timeindex'] = np.arange(len(wd))
wd['station_id'] = wd['station_id'].astype(str)

# Impute missing values for all numeric columns
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

# Drop unnecessary columns
wd = wd.drop(columns=['station_data_id', 'station_name', 'scd41_id', 'ens160_id', 'sim41_id', 'bme688_id', 'sfa30_id'], errors='ignore')

# Normalize numeric columns
scaler = MinMaxScaler()
wd[numeric_cols] = scaler.fit_transform(wd[numeric_cols])

# Split data
split_index = int(0.8 * len(wd))  # 80% train, 20% test
wd1 = wd[:split_index]
wd_11 = wd[split_index:]

print("***** wd1.shape=", wd1.shape)
print("***** wd_11.shape=", wd_11.shape)

# Additional normalization for target variable
mm = MinMaxScaler()
f = mm.fit(wd1['scd41_co2'].values.reshape(-1, 1))
wd1['scd41_co2'] = f.transform(wd1['scd41_co2'].to_numpy().reshape(-1, 1))
f11 = mm.fit(wd_11['scd41_co2'].values.reshape(-1, 1))
wd_11['scd41_co2'] = f11.transform(wd_11['scd41_co2'].to_numpy().reshape(-1, 1))

# Create TimeSeriesDataSet
max_prediction_length = 10
max_encoder_length = 200
training_cutoff = wd1["timeindex"].max() - max_prediction_length
print("***** training_cutoff=", training_cutoff)

# Define time-varying known reals (cyclical features)
time_varying_known_reals = ["timeindex", "hour_sin", "hour_cos", "week_sin", "week_cos"]

# Define time-varying unknown reals (all other numeric columns except the target)
time_varying_unknown_reals = [col for col in numeric_cols if col != "scd41_co2"]

training = TimeSeriesDataSet(
    wd1[lambda x: x.timeindex <= training_cutoff],
    time_idx="timeindex",
    target="scd41_co2",
    group_ids=["station_id"],
    # Encoder length - keep these balanced
    min_encoder_length=max_encoder_length // 2,  # 50% of max as minimum
    max_encoder_length=max_encoder_length,       # Your defined maximum
    # Prediction length
    min_prediction_length=max_prediction_length // 2,  # More flexible than fixed 1
    max_prediction_length=max_prediction_length,
    # Static features
    static_categoricals=["station_id"],
    static_reals=[],  # Add if available
    # Known time-varying features (future known)
    time_varying_known_categoricals=["month", "hour"],    # Add temporal features
    time_varying_known_reals=time_varying_known_reals + ["timeindex"],  # Add time index
    # Unknown time-varying features (only past)
    time_varying_unknown_categoricals=[],   # Add if available
    time_varying_unknown_reals=time_varying_unknown_reals + ["scd41_temperature", "scd41_humidity"],  # Add relevant vars
    # Important options
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=False,  # Better set to False if possible for cleaner data
    # New recommended additions
    target_normalizer=TorchNormalizer(method="standard", transformation="softplus"),  # Better normalization
    categorical_encoders={"station_id": NaNLabelEncoder(add_nan=True)},  # Better handling
    randomize_length=None,  # Or use (0.2, 0.1) for some randomization
    predict_mode=False
)

validation = TimeSeriesDataSet.from_dataset(training, wd1, predict=True, stop_randomization=True)

# Create dataloaders
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)

# Baseline model
print("***** calculate baseline model")
baseline_predictions = Baseline().predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
print("Baseline MAE:", MAE()(baseline_predictions.output, baseline_predictions.y[0]))

# Train the model
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("lightning_logs")

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    limit_train_batches=50,
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
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
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# Fit the model
print("***** trainer.fit")
trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

print("***** Training complete")

# Load best model
print("***** load the best model according to the validation loss")
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# Evaluate on validation set
print("***** calculate mean absolute error on validation set")
predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
print("Validation MAE:", MAE()(predictions.output, predictions.y[0]))

# Plot predictions
raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))
for idx in range(1):
    tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
plt.show()

# Variable importance
interpretation = tft.interpret_output(raw_predictions.output, reduction="sum")
tft.plot_interpretation(interpretation)
plt.show()

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

test_dataloader = test.to_dataloader(train=False, batch_size=(batch_size * 10), num_workers=0)

# Test predictions
raw_predictions_test = best_tft.predict(test_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="cpu"))
for idx in range(5):
    tft.plot_prediction(raw_predictions_test.x, raw_predictions_test.output, idx=idx, add_loss_to_title=True)
plt.show()

# Test interpretation
interpretation = tft.interpret_output(raw_predictions_test.output, reduction="sum")
tft.plot_interpretation(interpretation)
plt.show()

# Additional test metrics
test1 = TimeSeriesDataSet.from_dataset(test, wd_11, predict=True, stop_randomization=True)
test_dataloader1 = test1.to_dataloader(train=False, batch_size=(batch_size * 10), num_workers=0)
predictions1 = tft.predict(test_dataloader1, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
mse = torch.nn.functional.mse_loss(predictions1.output, predictions1.y[0])
mae = torch.nn.functional.l1_loss(predictions1.output, predictions1.y[0])
print(f'***** Test MSE: {mse}')
print(f'***** Test MAE: {mae}')

T_end = time.time()
T_total = T_end - T_start
print(f'***** Total time: {T_total}')