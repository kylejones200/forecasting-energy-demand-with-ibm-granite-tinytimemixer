#!/usr/bin/env python3
"""
Energy Demand Forecasting with Granite TinyTimeMixer

Main entry point for running energy demand forecasting.
"""

import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from src.core import (
    prepare_energy_data,
    create_time_features,
    calculate_forecast_metrics,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def load_config(config_path: Path = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Energy Demand Forecasting with Granite TinyTimeMixer')
    parser.add_argument('--config', type=Path, default=None, help='Path to config file')
    parser.add_argument('--data-path', type=Path, default=None, help='Path to data file')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory')
    args = parser.parse_args()
    
    config = load_config(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['output']['figures_dir'])
    output_dir.mkdir(exist_ok=True)
    
    if args.data_path and args.data_path.exists():
        df = pd.read_csv(args.data_path)
        df = prepare_energy_data(df, config['data']['date_column'], config['data']['demand_column'])
        if config['model']['use_time_features']:
            df = create_time_features(df)
        demand = df[config['data']['demand_column']]
    elif config['data']['generate_synthetic']:
        np.random.seed(config['data']['seed'])
        dates = pd.date_range('2023-01-01', periods=config['data']['n_periods'], freq='H')
        base_demand = 1000 + 200 * np.sin(np.arange(config['data']['n_periods']) / (24 * 7)) + \
                     100 * np.sin(np.arange(config['data']['n_periods']) / 24)
        demand = pd.Series(base_demand + np.random.normal(0, 50, config['data']['n_periods']), index=dates)
    else:
        raise ValueError("No data source specified")
    
    train_size = int(len(demand) * config['model']['train_size'])
    demand_train = demand[:train_size]
    demand_test = demand[train_size:]
    
    y_pred = np.full(len(demand_test), demand_train.mean())
    
    metrics = calculate_forecast_metrics(demand_test.values, y_pred)
    logging.info(f"\nForecast Metrics:")
    logging.info(f"RMSE: {metrics['rmse']:.2f}")
    logging.info(f"MAE: {metrics['mae']:.2f}")
    
    plot_energy_forecast(demand_test.values, y_pred, "Energy Demand Forecast",
                       output_dir / 'energy_forecast.png')
    
    logging.info(f"\nAnalysis complete. Figures saved to {output_dir}")

if __name__ == "__main__":
    main()

