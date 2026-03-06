# optimizer_hourly.py

import os
import json
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
from utils import config
from testing.strategies_hourly import HOURLY_STRATEGIES
from testing.backtester import run_backtest
import warnings

warnings.filterwarnings(
    "ignore",
    message="Setting an item of incompatible dtype is deprecated*",
    category=FutureWarning,
)

PARAMETER_SPACE_HOURLY = {
    "HourlyEMACross": {"fast": [8, 12], "slow": [20, 24]},
    "HourlyRSI": {"length": [7, 14], "upper": [75], "lower": [25]},
    "VolumeAnomaly": {"length": [20], "threshold": [2.5, 3.0]},
    "HourlyBollinger": {"length": [20], "std": [2.0, 2.5]},
    "HourlyStochastic": {"k": [14], "d": [3], "upper": [80], "lower": [20]},
    "HourlyMACD": {"fast": [12], "slow": [26], "signal": [9]},
    "HourlyCMF": {"length": [20, 40]},
    "HourlyROC": {"length": [12, 24]},
    "HourlyADX": {"length": [14, 20], "threshold": [20, 25]},
    "HourlyPSAR": {"af": [0.02, 0.03], "max_af": [0.2, 0.3]},
    "HourlyTEMA": {"length": [8, 10, 12]},
    "HourlyLongEMACross": {"fast": [50], "slow": [200]},
    # --- FIX: Added missing strategies to the optimization space ---
    "RVI": {"length": [14, 20], "signal": [4, 6]},
    "FisherTransform": {"length": [9, 15]},
}

GLOBAL_HOURLY_PARAMS_PATH = os.path.join("models", "optimized_hourly_params.json")


def optimize_hourly_strategies(ticker, mode):
    HOURLY_DATA_DIR = os.path.join(config.DATA_DIR, "hourly")

    # --- GLOBAL MODE ---
    if mode == "global":
        print("--- Starting GLOBAL Hourly Strategy Optimization for All Tickers ---")
        all_ticker_data = {}
        for t in config.TICKERS:
            filepath = os.path.join(HOURLY_DATA_DIR, f"{t}.csv")
            if os.path.exists(filepath):
                all_ticker_data[t] = pd.read_csv(
                    filepath, index_col=0, parse_dates=True
                )

        optimized_params = {}
        for name, strategy_func in tqdm(
            HOURLY_STRATEGIES.items(), desc="Optimizing Hourly Strategies Globally"
        ):
            if name not in PARAMETER_SPACE_HOURLY:
                optimized_params[name] = {}
                continue

            param_grid = PARAMETER_SPACE_HOURLY[name]
            param_names, param_combinations = list(param_grid.keys()), list(
                product(*param_grid.values())
            )
            best_params, best_avg_return = None, -np.inf

            for params in tqdm(param_combinations, desc=f"Tuning {name}", leave=False):
                current_params = dict(zip(param_names, params))
                if (
                    "fast" in current_params
                    and "slow" in current_params
                    and current_params["fast"] >= current_params["slow"]
                ):
                    continue
                if (
                    "tenkan" in current_params
                    and "kijun" in current_params
                    and current_params["tenkan"] >= current_params["kijun"]
                ):
                    continue

                total_return = 0
                for ticker_df in all_ticker_data.values():
                    signals = strategy_func(ticker_df.copy(), **current_params)
                    total_return += run_backtest(ticker_df, signals)

                avg_return = total_return / len(all_ticker_data)

                if avg_return > best_avg_return:
                    best_avg_return, best_params = avg_return, current_params

            optimized_params[name] = best_params
            print(
                f"Best GLOBAL params for {name}: {best_params} -> Avg Return: {best_avg_return:.2%}"
            )

        os.makedirs("models", exist_ok=True)
        with open(GLOBAL_HOURLY_PARAMS_PATH, "w") as f:
            json.dump(optimized_params, f, indent=4)
        print(
            f"\n--- GLOBAL Hourly Optimization Complete ---\nSaved best parameters to {GLOBAL_HOURLY_PARAMS_PATH}"
        )

    # --- LOCAL MODE ---
    elif mode == "local":
        if not ticker:
            print("Error: Local mode requires a --ticker argument.")
            return

        print(
            f"--- Starting LOCAL Hourly Strategy Optimization for {ticker.upper()} ---"
        )
        filepath = os.path.join(HOURLY_DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(filepath):
            print(f"Error: Hourly data for {ticker} not found.")
            return

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        optimized_params = {}

        for name, strategy_func in tqdm(
            HOURLY_STRATEGIES.items(),
            desc=f"Optimizing Hourly Strategies for {ticker.upper()}",
        ):
            if name not in PARAMETER_SPACE_HOURLY:
                optimized_params[name] = {}
                continue

            param_grid = PARAMETER_SPACE_HOURLY[name]
            param_names, param_combinations = list(param_grid.keys()), list(
                product(*param_grid.values())
            )
            best_params, best_performance = None, -np.inf

            for params in tqdm(param_combinations, desc=f"Tuning {name}", leave=False):
                current_params = dict(zip(param_names, params))
                if (
                    "fast" in current_params
                    and "slow" in current_params
                    and current_params["fast"] >= current_params["slow"]
                ):
                    continue
                if (
                    "tenkan" in current_params
                    and "kijun" in current_params
                    and current_params["tenkan"] >= current_params["kijun"]
                ):
                    continue

                signals = strategy_func(df.copy(), **current_params)
                performance = run_backtest(df, signals)

                if performance > best_performance:
                    best_performance, best_params = performance, current_params

            optimized_params[name] = best_params
            print(
                f"Best LOCAL params for {name} on {ticker.upper()}: {best_params} -> Return: {best_performance:.2%}"
            )

        STOCK_DIR = os.path.join(config.STOCKS_DIR, ticker.upper(), "hourly")
        os.makedirs(STOCK_DIR, exist_ok=True)
        params_path = os.path.join(STOCK_DIR, "hourly_params.json")
        with open(params_path, "w") as f:
            json.dump(optimized_params, f, indent=4)
        print(
            f"\n--- LOCAL Hourly Optimization Complete for {ticker.upper()} ---\nSaved best parameters to {params_path}"
        )
