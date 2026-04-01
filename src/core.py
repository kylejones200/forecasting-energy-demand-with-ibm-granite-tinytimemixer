"""Core functions for energy demand forecasting with Granite TinyTimeMixer."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def prepare_energy_data(df: pd.DataFrame, date_col: str, demand_col: str) -> pd.DataFrame:
    """Prepare energy demand data for forecasting."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.set_index(date_col)
    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features."""
    df = df.copy()
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    return df

def calculate_forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculate forecast error metrics."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }

def plot_energy_forecast(actual: np.ndarray, predicted: np.ndarray, title: str, output_path: Path):
 """Plot energy demand forecast """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(actual, label="Actual Demand", color="#4A90A4", linewidth=1.2)
    ax.plot(predicted, label="Predicted Demand", color="#D4A574", linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy Demand")
    ax.legend(loc='best')
    
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()

