# optimizer_daily.py

import os
import pandas as pd
import numpy as np
import json
from itertools import product
from tqdm import tqdm
from utils import config
from testing.strategies import TOP_STRATEGIES
from testing.backtester import run_backtest
import warnings

# Suppresses the known, benign FutureWarning from pandas_ta
warnings.filterwarnings(
    "ignore",
    message="Setting an item of incompatible dtype is deprecated*",
    category=FutureWarning,
)

PARAMETER_SPACE = {
    "MACD": {"fast": [12, 26], "slow": [26, 52], "signal": [9, 12]},
    "RSI": {"length": [14, 21], "upper": [70, 80], "lower": [20, 30]},
    "BollingerBands": {"length": [20], "std": [2.0, 2.5]},
    "EMACross": {"fast": [20, 50], "slow": [100, 200]},
    "Stochastic": {"k": [14, 21], "d": [3, 5], "upper": [80], "lower": [20]},
    "KeltnerChannels": {"length": [20], "multiplier": [2.0, 2.5]},
    "ROC": {"length": [12, 21]},
    "CMF": {"length": [20, 40]},
    "Ichimoku": {"tenkan": [9, 20], "kijun": [26, 40]},
    "AwesomeOscillator": {"fast": [5, 7], "slow": [34, 40]},
    "ADX": {"length": [14, 20], "threshold": [20, 25]},
    # --- FIX: Added missing strategies to the optimization space ---
    "CoppockCurve": {"length": [10, 14], "fast": [11, 15], "slow": [14, 20]},
    "CMO": {"length": [9, 14], "upper": [50, 60], "lower": [-50, -60]},
    "TRIX": {"length": [20, 30], "signal": [9, 12]},
}


def optimize_daily_strategies(ticker, mode):

    # --- GLOBAL MODE ---
    if mode == "global":
        print("--- Starting GLOBAL Daily Strategy Optimization for All Tickers ---")
        all_ticker_data = {}
        for t in config.TICKERS:
            filepath = os.path.join(config.DATA_DIR, f"{t}.csv")
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                all_ticker_data[t] = df

        optimized_params = {}
        for name, strategy_func in tqdm(
            TOP_STRATEGIES.items(), desc="Optimizing Daily Strategies Globally"
        ):
            if name not in PARAMETER_SPACE:
                optimized_params[name] = {}
                continue

            param_grid = PARAMETER_SPACE[name]
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
        with open(config.GLOBAL_OPTIMIZED_PARAMS_PATH, "w") as f:
            json.dump(optimized_params, f, indent=4)
        print(
            f"\n--- GLOBAL Daily Optimization Complete ---\nSaved best parameters to {config.GLOBAL_OPTIMIZED_PARAMS_PATH}"
        )

    # --- LOCAL MODE ---
    elif mode == "local":
        if not ticker:
            print("Error: Local mode requires a --ticker argument.")
            return

        print(
            f"--- Starting LOCAL Daily Strategy Optimization for {ticker.upper()} ---"
        )
        filepath = os.path.join(config.DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(filepath):
            print(f"Error: Daily data for {ticker} not found.")
            return

        df = pd.read_csv(filepath, index_col=0, parse_dates=True)

        optimized_params = {}
        for name, strategy_func in tqdm(
            TOP_STRATEGIES.items(),
            desc=f"Optimizing Daily Strategies for {ticker.upper()}",
        ):
            if name not in PARAMETER_SPACE:
                optimized_params[name] = {}
                continue

            param_grid = PARAMETER_SPACE[name]
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

        STOCK_DIR = os.path.join(config.STOCKS_DIR, ticker.upper(), "daily")
        os.makedirs(STOCK_DIR, exist_ok=True)
        params_path = os.path.join(STOCK_DIR, "daily_params.json")
        with open(params_path, "w") as f:
            json.dump(optimized_params, f, indent=4)
        print(
            f"\n--- LOCAL Daily Optimization Complete for {ticker.upper()} ---\nSaved best parameters to {params_path}"
        )
