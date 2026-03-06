# threshold_optimizer.py - FIXED: Matches SmartDynamicTrader API

import pandas as pd
import numpy as np
import os
import json
from utils import config
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from tqdm import tqdm
import pandas_ta as ta
from testing.strategies import find_col


def run_threshold_optimizer(ticker):
    print(
        f"--- Starting Daily Threshold Optimization for {ticker.upper()} (Year 2025) ---"
    )

    # Import here to avoid circular import
    from testing.paper_trader import (
        SmartDynamicTrader,
        pre_compute_daily_features,
    )
    import joblib

    # Load model and config
    STOCK_DIR = os.path.join(config.STOCKS_DIR, ticker.upper(), "daily")
    AI_MODEL_PATH = os.path.join(STOCK_DIR, "daily_model.joblib")
    PARAMS_PATH = os.path.join(STOCK_DIR, "daily_params.json")
    CONFIG_PATH = os.path.join(STOCK_DIR, "training_config.json")

    try:
        model = joblib.load(AI_MODEL_PATH)
        with open(PARAMS_PATH, "r") as f:
            optimized_params = json.load(f)
        with open(CONFIG_PATH, "r") as f:
            training_config = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Model/params/config for {ticker.upper()} not found.")
        return

    # Fetch data
    api = tradeapi.REST(
        config.API_KEY, config.SECRET_KEY, base_url="https://paper-api.alpaca.markets"
    )
    end_date = pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(minutes=16)
    hist_start_date = "2024-01-01"

    try:
        all_bars_ticker = api.get_bars(
            ticker.upper(),
            TimeFrame.Day,
            start=hist_start_date,
            end=end_date.isoformat(),
            adjustment="raw",
        ).df.tz_convert("America/New_York")

        all_bars_market = api.get_bars(
            config.MARKET_TICKER,
            TimeFrame.Day,
            start=hist_start_date,
            end=end_date.isoformat(),
            adjustment="raw",
        ).df.tz_convert("America/New_York")

        for df in [all_bars_ticker, all_bars_market]:
            df.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                },
                inplace=True,
            )
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Load custom threshold range from config
    start_thresh, end_thresh, step = 0.55, 0.71, 0.01

    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                training_config_loaded = json.load(f)
                opt_range = training_config_loaded.get("optimization_range")
                if opt_range and all(k in opt_range for k in ["start", "end", "step"]):
                    start_thresh = opt_range["start"]
                    end_thresh = opt_range["end"]
                    step = opt_range["step"]
                    print(
                        f"--- Loaded custom optimization range: {start_thresh*100:.1f}% to {end_thresh*100:.1f}% (step: {step*100:.1f}%) ---"
                    )
        except Exception as e:
            print(
                f"Warning: Could not load optimization range. Using defaults. Error: {e}"
            )

    thresholds_to_test = np.arange(start_thresh, end_thresh, step)

    # Pre-compute features once
    print("[INFO] Pre-computing features for all thresholds...")
    all_features = pre_compute_daily_features(
        all_bars_ticker, all_bars_market, optimized_params, training_config, model
    )

    test_prices = all_bars_ticker[all_bars_ticker.index.year >= 2025].copy()
    test_features = all_features[all_features.index.year >= 2025]

    aligned_index = test_prices.index.intersection(test_features.index)
    if aligned_index.empty:
        print("No aligned data for 2025.")
        return

    test_prices = test_prices.loc[aligned_index]
    test_features = test_features.loc[aligned_index]

    # Add ATR
    atr_result = test_prices.ta.atr(length=14)
    if isinstance(atr_result, pd.DataFrame):
        test_prices["atr_for_stop"] = atr_result[find_col(atr_result, "ATRr_14")]
    else:
        test_prices["atr_for_stop"] = atr_result

    # Predict once
    print("[INFO] Generating predictions...")
    buy_probabilities = model.predict_proba(test_features)[:, 1]

    results = []
    best_result = {"threshold": None, "return": -np.inf, "trades": 0, "sharpe": 0}

    print("\n--- Testing thresholds ---")
    for threshold in tqdm(thresholds_to_test, desc="Optimizing Threshold"):
        # Run simulation with this threshold
        trader = SmartDynamicTrader(
            ticker,
            model,
            optimized_params,
            training_config,
            threshold,
            start_cash=10000,
        )

        for i in range(len(test_prices)):
            date = test_prices.index[i]
            row = test_prices.iloc[i]
            current_price = row["Open"]
            confidence = buy_probabilities[i]
            atr = row.get("atr_for_stop", row["Close"] * 0.02)

            if pd.isna(current_price) or pd.isna(atr):
                continue

            # 1. Update State
            trader.update_state(date, current_price, confidence)

            # 2. Detect Regime
            if i >= 20:
                recent_prices = test_prices.iloc[max(0, i - 20) : i]["Close"]
                recent_conf = buy_probabilities[max(0, i - 20) : i]
                regime = trader.detect_market_regime(recent_prices, recent_conf)
            else:
                regime = "UNKNOWN"

            # 3. Check Exit
            should_exit, exit_reason, exit_pct = trader.should_reduce_or_exit(
                confidence, current_price, atr, regime
            )
            if should_exit:
                trader.execute_trade(
                    date,
                    current_price,
                    confidence,
                    atr,
                    "SELL",
                    exit_reason,
                    exit_pct,
                )

            # 4. Check Entry
            should_enter, entry_reason = trader.should_enter_or_add(
                confidence, current_price, atr, regime
            )
            if should_enter:
                allocation = trader.calculate_dynamic_position_size(
                    confidence, current_price, atr, regime
                )
                if allocation > 0.05:
                    trader.execute_trade(
                        date,
                        current_price,
                        confidence,
                        atr,
                        "BUY",
                        entry_reason,
                        allocation,
                    )

        # Close final position
        if trader.shares_held > 0:
            final_price = test_prices["Close"].iloc[-1]
            trader.execute_trade(
                test_prices.index[-1],
                final_price,
                buy_probabilities[-1],
                test_prices["atr_for_stop"].iloc[-1],
                "SELL",
                "END_OF_PERIOD",
                1.0,
            )

        # Calculate metrics
        final_value = trader.cash
        strategy_return = (final_value / 10000 - 1) * 100
        num_trades = len([t for t in trader.trades if t["action"] == "SELL"])

        sharpe_ratio = 0.0
        if len(trader.trade_returns) > 1:
            returns_series = pd.Series(trader.trade_returns)
            if returns_series.std() > 0:
                sharpe_ratio = (returns_series.mean() / returns_series.std()) * np.sqrt(
                    252
                )

        results.append(
            {
                "threshold": threshold,
                "return": strategy_return,
                "trades": num_trades,
                "sharpe": sharpe_ratio,
            }
        )

        # Update best (prioritize Sharpe ratio, then return)
        if sharpe_ratio > best_result["sharpe"] or (
            sharpe_ratio == best_result["sharpe"]
            and strategy_return > best_result["return"]
        ):
            best_result.update(
                {
                    "threshold": threshold,
                    "return": strategy_return,
                    "trades": num_trades,
                    "sharpe": sharpe_ratio,
                }
            )

    # Print report
    print("\n" + "=" * 60)
    print(f"  Daily Threshold Optimization Report for {ticker.upper()}")
    print("=" * 60)
    print(f"{'Threshold':<12} | {'Return (%)':<12} | {'Trades':<10} | {'Sharpe':<10}")
    print("-" * 60)
    for res in results:
        print(
            f"{res['threshold']:<12.2f} | {res['return']:<12.2f} | {res['trades']:<10} | {res['sharpe']:<10.2f}"
        )
    print("=" * 60)
    if best_result["threshold"] is not None:
        print(
            f"Optimal Threshold: {best_result['threshold']:.2f} -> {best_result['return']:.2f}% Return (Sharpe: {best_result['sharpe']:.2f})"
        )
    else:
        print("WARNING: No profitable threshold found. Defaulting to middle of range.")
        best_result["threshold"] = (start_thresh + end_thresh) / 2

    print("=" * 60)

    # Save optimal threshold
    if best_result["threshold"] is not None:
        settings_path = os.path.join(STOCK_DIR, "daily_settings.json")
        settings = {}
        if os.path.exists(settings_path):
            with open(settings_path, "r") as f:
                settings = json.load(f)
        settings["optimal_threshold"] = round(best_result["threshold"], 2)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=4)
        print(f"\n--- Saved optimal threshold to {settings_path} ---")