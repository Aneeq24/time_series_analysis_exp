import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import RobustScaler
import warnings
import logging
import os
import matplotlib.dates as mdates

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# Configuration
FILE_PATH = 'data_202504301228.csv'
STATION_COL = 'station_id'
TIME_COL = 'timestamp'
SENSOR_COLS = [
    'scd41_co2', 'scd41_temperature', 'scd41_humidity',
    'ens160_eco2', 'ens160_tvoc', 'ens160_aqi',
    'svm41_temperature', 'svm41_humidity', 'svm41_nox_index', 'svm41_voc_index',
    'sfa30_temperature', 'sfa30_humidity', 'sfa30_hco'
]

CONVERT_COLS = [
    'scd41_co2', 'scd41_temperature', 'scd41_humidity',
    'svm41_temperature', 'svm41_humidity', 'svm41_nox_index', 'svm41_voc_index',
    'sfa30_temperature', 'sfa30_humidity', 'sfa30_hco'
]

# Units for each sensor column
UNITS = {
    'scd41_co2': 'ppm',
    'scd41_temperature': '°C',
    'scd41_humidity': '%',
    'ens160_eco2': 'ppm',
    'ens160_tvoc': 'ppb',
    'ens160_aqi': 'index',
    'svm41_temperature': '°C',
    'svm41_humidity': '%',
    'svm41_nox_index': 'index',
    'svm41_voc_index': 'index',
    'sfa30_temperature': '°C',
    'sfa30_humidity': '%',
    'sfa30_hco': 'ppb'
}

# Function to filter data by date range, frequency, and optional time range
def filter_data_by_date_and_time(df, start_date=None, end_date=None, year=None, month=None, day_freq=1, start_hour=None, end_hour=None):
    try:
        df_filtered = df.copy()
        
        # If start_date and end_date are provided, use them
        if start_date and end_date:
            try:
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                if start_date > end_date:
                    raise ValueError("Start date must be before end date")
                df_filtered = df_filtered[(df_filtered.index >= start_date) & (df_filtered.index <= end_date)]
                logger.info(f"Filtered data to date range: {start_date} to {end_date}")
            except ValueError as e:
                logger.error(f"Invalid date format: {str(e)}")
                raise
        
        if df_filtered.empty:
            logger.error("No data found for the specified date range or month")
            raise ValueError("No data available for the specified date range or month")
        
        # Apply day frequency filter (e.g., every Nth day)
        if day_freq > 1:
            try:
                # Create a mask for every Nth day
                start_day = df_filtered.index.min().date()
                all_dates = pd.date_range(start=start_day, end=df_filtered.index.max(), freq='D')
                selected_dates = all_dates[::day_freq]
                mask = df_filtered.index.date.isin([d.date() for d in selected_dates])
                df_filtered = df_filtered[mask]
                logger.info(f"Applied day frequency filter: every {day_freq} days")
            except Exception as e:
                logger.warning(f"Error applying day frequency filter: {str(e)}. Ignoring frequency.")
        
        if df_filtered.empty:
            logger.error("No data remains after applying day frequency filter")
            raise ValueError("No data available after day frequency filtering")
        
        # Filter by time range if provided
        if start_hour is not None and end_hour is not None:
            try:
                start_hour = int(start_hour)
                end_hour = int(end_hour)
                if not (0 <= start_hour <= 23 and 0 <= end_hour <= 23):
                    raise ValueError("Hours must be between 0 and 23")
                df_filtered = df_filtered[(df_filtered.index.hour >= start_hour) & (df_filtered.index.hour < end_hour)]
                logger.info(f"Filtered data to hours {start_hour}:00 to {end_hour}:00")
            except ValueError as e:
                logger.error(f"Invalid time range: {str(e)}")
                raise
        
        if df_filtered.empty:
            logger.error("No data remains after applying time filter")
            raise ValueError("No data available after time filtering")
        
        logger.info(f"Filtered data, {len(df_filtered)} rows remaining")
        return df_filtered
    except Exception as e:
        logger.error(f"Error filtering data: {str(e)}")
        raise

# ====================
# 1. Data Loading
# ====================
def load_data(file_path):
    """Load and preprocess raw data"""
    try:
        df = pd.read_csv(file_path)
        logger.info("Dataset loaded successfully")
    except FileNotFoundError:
        logger.error(f"File {file_path} not found")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"File {file_path} is empty")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

    # Verify required columns
    required_cols = [TIME_COL, STATION_COL] + SENSOR_COLS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing columns: {missing_cols}")
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Log sample of timestamp data for debugging
    logger.info(f"Sample timestamp values: {df[TIME_COL].head().tolist()}")

    # Attempt to parse timestamps with coercion
    try:
        # Handle Unix timestamps or string formats
        if pd.api.types.is_integer_dtype(df[TIME_COL]):
            min_val = df[TIME_COL].min()
            if min_val > 1e12:  # Likely milliseconds
                df[TIME_COL] = pd.to_datetime(df[TIME_COL], unit='ms')
            else:  # Likely seconds
                df[TIME_COL] = pd.to_datetime(df[TIME_COL], unit='s')
        else:
            df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors='coerce')
        
        invalid_timestamps = df[TIME_COL].isnull()
        if invalid_timestamps.any():
            logger.warning(f"Found {invalid_timestamps.sum()} invalid timestamp entries")
            logger.debug(f"Invalid timestamp rows (first few): {df[invalid_timestamps][TIME_COL].head().tolist()}")
            df = df[~invalid_timestamps].copy()
            logger.info(f"Dropped {invalid_timestamps.sum()} rows with invalid timestamps")
        
        if df.empty:
            logger.error("All timestamp values are invalid or dataset is empty after cleaning")
            raise ValueError("No valid timestamp data remaining after cleaning")

        df = df.sort_values(by=[TIME_COL, STATION_COL])
        df = df.set_index(TIME_COL)
        logger.info("Successfully set DatetimeIndex")
    except Exception as e:
        logger.error(f"Error setting DatetimeIndex: {str(e)}")
        raise ValueError(f"Failed to parse timestamp column: {str(e)}")

    # Scaling function for SensorData columns
    def scale_sensor_data(df, cols):
        for col in cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 1_000_000
                    if df[col].isnull().sum() > len(df) * 0.5:
                        logger.warning(f"Over 50% of values in {col} could not be converted")
                except Exception as e:
                    logger.error(f"Error scaling column {col}: {str(e)}")
                    df[col] = np.nan
        return df

    # Apply scaling to the sensor columns
    df = scale_sensor_data(df, CONVERT_COLS)
    return df

# ====================
# 2. Preprocessing Pipeline
# ====================
def preprocess_sensor_data(df, sensor_col, window_size=24*7, iqr_multiplier=3):
    """Full preprocessing for a sensor column"""
    try:
        if sensor_col not in df.columns:
            logger.error(f"Sensor column {sensor_col} not found")
            return None, None, None

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"DataFrame for {sensor_col} does not have a DatetimeIndex")
            return None, None, None

        # Store original data
        original = df[sensor_col].copy()

        # A. Missing Value Handling (Forward fill + time interpolation)
        if original.isnull().any():
            df[sensor_col] = df[sensor_col].ffill()
            try:
                df[sensor_col] = df[sensor_col].interpolate(method='time', limit_direction='both')
                logger.info(f"Applied time interpolation to {sensor_col}")
            except Exception as e:
                logger.warning(f"Time interpolation failed for {sensor_col}: {str(e)}. Falling back to linear interpolation")
                df[sensor_col] = df[sensor_col].interpolate(method='linear', limit_direction='both')
        else:
            logger.info(f"No missing values in {sensor_col}")

        # B. Outlier Detection and Correction (Rolling median-based)
        try:
            rolling_median = df[sensor_col].rolling(window=window_size, min_periods=1, center=True).median()
            iqr = df[sensor_col].quantile(0.75) - df[sensor_col].quantile(0.25)
            if pd.isna(iqr) or iqr == 0:
                logger.warning(f"Skipping outlier correction for {sensor_col}: invalid IQR")
            else:
                df[sensor_col] = np.where(
                    abs(df[sensor_col] - rolling_median) > iqr_multiplier * iqr,
                    rolling_median,
                    df[sensor_col]
                )
                logger.info(f"Applied outlier correction to {sensor_col}")
        except Exception as e:
            logger.warning(f"Outlier correction failed for {sensor_col}: {str(e)}")

        # C. Stationarity Enforcement (Differencing)
        df[f'{sensor_col}_diff'] = df[sensor_col].diff().fillna(0)
        logger.info(f"Applied differencing to {sensor_col}")

        return original, df[sensor_col], df[f'{sensor_col}_diff']
    except Exception as e:
        logger.error(f"Error preprocessing {sensor_col}: {str(e)}")
        return None, None, None

# ====================
# 3. Visualization Tools
# ====================
def plot_sensor_transformations(original, cleaned, transformed, title, station_id):
    """Create diagnostic plots for preprocessing with proper labels, units, and time formatting"""
    try:
        # Create plots directory
        os.makedirs('plots_new', exist_ok=True)

        fig, ax = plt.subplots(3, 1, figsize=(12, 8))

        # Get the unit for the sensor
        unit = UNITS.get(title, 'unknown')

        # Original vs Cleaned
        sns.lineplot(x=original.index, y=original, ax=ax[0], label='Original', color='gray', alpha=0.7)
        sns.lineplot(x=cleaned.index, y=cleaned, ax=ax[0], label='Cleaned', color='blue')
        ax[0].set_title(f'{title} - Raw vs Processed Data (Station {station_id})')
        ax[0].set_ylabel(f'{title} ({unit})')
        ax[0].set_xlabel('Time')
        ax[0].legend()

        # STL Decomposition Comparison
        try:
            stl_original = STL(original.dropna(), period=24, robust=True).fit()
            stl_cleaned = STL(cleaned.dropna(), period=24, robust=True).fit()
            ax[1].plot(stl_original.trend, label='Original Trend')
            ax[1].plot(stl_cleaned.trend, label='Cleaned Trend')
            ax[1].set_title(f'{title} - STL Decomposition Trend Comparison (Station {station_id})')
            ax[1].set_ylabel(f'Trend Component ({unit})')
            ax[1].set_xlabel('Time')
            ax[1].legend()
        except Exception as e:
            logger.warning(f"STL decomposition failed for {title}: {str(e)}")
            ax[1].text(0.5, 0.5, 'STL Decomposition Failed', horizontalalignment='center', verticalalignment='center')
            ax[1].set_title(f'{title} - STL Decomposition Trend Comparison (Station {station_id})')
            ax[1].set_ylabel(f'Trend Component ({unit})')
            ax[1].set_xlabel('Time')

        # Stationary Series
        sns.lineplot(x=transformed.index, y=transformed, ax=ax[2], color='green')
        ax[2].set_title(f'{title} - Stationary Series After Differencing (Station {station_id})')
        ax[2].set_ylabel('Differenced Values')
        ax[2].set_xlabel('Time')

        # Format x-axis for all subplots
        for a in ax:
            a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            a.xaxis.set_major_locator(mdates.AutoDateLocator())

        plt.tight_layout()
        plot_path = f'plots_new/{title}_station_{station_id}_preprocessing.png'
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved plot: {plot_path}")
    except Exception as e:
        logger.error(f"Error plotting {title}: {str(e)}")

# ====================
# 4. Feature Engineering
# ====================
def create_time_features(df):
    """Create temporal features with cyclical encoding"""
    try:
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame does not have a DatetimeIndex for time features")
            return df

        df = df.copy()
        df['hour'] = df.index.hour
        df['week'] = df.index.isocalendar().week
        df['day_of_week'] = df.index.dayofweek

        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)

        return df.drop(columns=['hour', 'week', 'day_of_week'])
    except Exception as e:
        logger.error(f"Error creating time features: {str(e)}")
        return df

def create_lag_features(df, sensor_col, lags=[24, 24*7]):
    """Create lag features for specified sensor"""
    try:
        for lag in lags:
            df[f'{sensor_col}_lag{lag}'] = df[sensor_col].shift(lag)
        return df
    except Exception as e:
        logger.error(f"Error creating lag features for {sensor_col}: {str(e)}")
        return df

# ====================
# 5. Main Execution
# ====================
def main():
    try:
        # Load and preprocess data
        raw_df = load_data(FILE_PATH)
        
        # Apply flexible date and time filter
        raw_df = filter_data_by_date_and_time(
            raw_df,
            start_date=None,  # e.g., '2023-01-01'
            end_date=None,    # e.g., '2023-01-15'
            year=2025,        # Fallback to prompt or default
            month=3,       # Fallback to prompt or default
            day_freq=2,       # Every 2 days
            start_hour=8,     # 8:00
            end_hour=17       # 17:00
        )
        
        processed_df = raw_df.copy()

        # Process each sensor column
        for sensor in SENSOR_COLS:
            logger.info(f"Processing {sensor}...")
            original, cleaned, transformed = preprocess_sensor_data(processed_df, sensor)
            if original is None:
                logger.warning(f"Skipping plotting for {sensor} due to preprocessing failure")
                continue

            # Generate plots for each unique station
            for station_id in processed_df[STATION_COL].unique():
                mask = processed_df[STATION_COL] == station_id
                plot_sensor_transformations(
                    original[mask],
                    cleaned[mask],
                    transformed[mask],
                    sensor,
                    station_id
                )

        # Create temporal features
        processed_df = create_time_features(processed_df)

        # Add lag features for all sensor columns
        for sensor in SENSOR_COLS:
            processed_df = create_lag_features(processed_df, sensor)

        # Apply RobustScaler to sensor columns
        try:
            scaler = RobustScaler()
            processed_df[SENSOR_COLS] = scaler.fit_transform(processed_df[SENSOR_COLS].astype(float))
            logger.info("Applied RobustScaler to sensor columns")
        except Exception as e:
            logger.error(f"Error applying RobustScaler: {str(e)}")

        # Save processed data
        try:
            processed_df.reset_index().to_csv('preprocessed_data.csv', index=False)
            logger.info("Saved processed data to 'preprocessed_data.csv'")
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")

        # Generate correlation matrix
        try:
            corr_matrix = processed_df[SENSOR_COLS].corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
            plt.title('Sensor Data Correlation Matrix')
            plt.savefig('plots_new/sensor_correlation_matrix.png')
            plt.close()
            logger.info("Saved correlation matrix plot")
        except Exception as e:
            logger.error(f"Error generating correlation matrix: {str(e)}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == '__main__':
    main()