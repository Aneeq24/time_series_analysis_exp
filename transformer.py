#!/usr/bin/env python
# coding: utf-8

import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import TorchNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, QuantileLoss
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
import time

def main():
    T_start = time.time()

    # Device setup - Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'***** Using {device} device')

    # Load data
    wd = pd.read_csv("preprocessed_data.csv")  # Update with your file path

    # Preprocessing steps
    wd['timestamp'] = pd.to_datetime(wd['timestamp'])
    wd = wd.sort_values(by='timestamp').reset_index(drop=True)
    wd['timeindex'] = np.arange(len(wd))
    wd['station_id'] = wd['station_id'].astype(str)

    # Add month and hour columns
    wd['month'] = wd['timestamp'].dt.month.astype(str)
    wd['hour'] = wd['timestamp'].dt.hour.astype(str)

    # Impute missing values
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
        wd[col] = wd[col].interpolate(method='linear', limit_direction='both').fillna(wd[col].mean())

    # Normalize numeric columns
    scaler = MinMaxScaler()
    wd[numeric_cols] = scaler.fit_transform(wd[numeric_cols])

    # Split data
    split_index = int(0.8 * len(wd))
    wd1 = wd[:split_index].copy()
    wd_11 = wd[split_index:].copy()
    del wd  # Free memory

    print(f'***** wd1.shape= {wd1.shape}')
    print(f'***** wd_11.shape= {wd_11.shape}')

    # Normalize target variable
    target_scaler = MinMaxScaler()
    wd1['scd41_co2'] = target_scaler.fit_transform(wd1['scd41_co2'].values.reshape(-1, 1))
    wd_11['scd41_co2'] = target_scaler.transform(wd_11['scd41_co2'].values.reshape(-1, 1))

    # TimeSeriesDataSet Configuration
    max_prediction_length = 10
    max_encoder_length = 200
    training_cutoff = wd1["timeindex"].max() - max_prediction_length
    print(f'***** training_cutoff= {training_cutoff}')

    time_varying_known_reals = ["timeindex", "hour_sin", "hour_cos", "week_sin", "week_cos"]
    time_varying_unknown_reals = [col for col in numeric_cols if col != "scd41_co2"]

    training = TimeSeriesDataSet(
        wd1[wd1["timeindex"] <= training_cutoff],
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
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=time_varying_unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
        target_normalizer=TorchNormalizer(method="standard"),
        categorical_encoders={"station_id": NaNLabelEncoder(add_nan=True)},
    )

    validation = TimeSeriesDataSet.from_dataset(training, wd1, predict=True, stop_randomization=True)

    # Create dataloaders
    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2, persistent_workers=True)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2, persistent_workers=True)

    # Baseline model
    print("***** Calculate baseline model")
    baseline_predictions = Baseline().predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="auto"))
    print("Baseline MAE:", MAE()(baseline_predictions.output, baseline_predictions.y[0]))

    # Model Configuration
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"),
            LearningRateMonitor(),
        ],
        logger=TensorBoardLogger("lightning_logs"),
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.001,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.2,
        hidden_continuous_size=64,
        loss=QuantileLoss(),
        log_interval=10,
        optimizer="Adam",
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # Training
    print("***** trainer.fit")
    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    print("***** Training complete")

    # Load best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Evaluate on validation set
    print("***** Calculate validation MAE")
    predictions = best_tft.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="auto"))
    print("Validation MAE:", MAE()(predictions.output, predictions.y[0]))

    # Plot predictions
    raw_predictions = best_tft.predict(val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="auto"))
    for idx in range(1):
        fig = best_tft.plot_prediction(raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True)
        fig.savefig(f"prediction_{idx}.png")
        plt.close(fig)

    # Variable importance
    interpretation = best_tft.interpret_output(raw_predictions.output, reduction="sum")
    fig = best_tft.plot_interpretation(interpretation)
    fig.savefig("interpretation.png")
    plt.close(fig)

    # Test dataset
    test = TimeSeriesDataSet.from_dataset(training, wd_11, predict=True, stop_randomization=True)
    test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=2, persistent_workers=True)

    # Test predictions
    raw_predictions_test = best_tft.predict(test_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="auto"))
    for idx in range(5):
        fig = best_tft.plot_prediction(raw_predictions_test.x, raw_predictions_test.output, idx=idx, add_loss_to_title=True)
        fig.savefig(f"test_prediction_{idx}.png")
        plt.close(fig)

    # Test interpretation
    interpretation = best_tft.interpret_output(raw_predictions_test.output, reduction="sum")
    fig = best_tft.plot_interpretation(interpretation)
    fig.savefig("test_interpretation.png")
    plt.close(fig)

    # Test metrics
    predictions_test = best_tft.predict(test_dataloader, return_y=True, trainer_kwargs=dict(accelerator="auto"))
    mse = mean_squared_error(predictions_test.y[0].cpu(), predictions_test.output.cpu())
    mae = MAE()(predictions_test.output, predictions_test.y[0])
    print(f'***** Test MSE: {mse}')
    print(f'***** Test MAE: {mae}')

    T_end = time.time()
    print(f'***** Total time: {T_end - T_start}')

    # Clean up memory
    del train_dataloader, val_dataloader, test_dataloader, wd1, wd_11
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

if __name__ == '__main__':
    main()