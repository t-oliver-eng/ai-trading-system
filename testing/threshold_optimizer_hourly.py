# threshold_optimizer_hourly.py - FIXED: Matches SmartDynamicHourlyTrader API

import pandas as pd
import numpy as np
import os
import json
from utils import config
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from tqdm import tqdm
import pandas_ta as ta
from testing.strategies_hourly import find_col


def run_hourly_threshold_optimizer(ticker):
    print(
        f"--- Starting Hourly Threshold Optimization for {ticker.upper()} (Year 2025) ---"
    )

    # Import here to avoid circular import
    from testing.paper_trader_hourly import (
        SmartDynamicHourlyTrader,
        pre_compute_hourly_features,
    )
    import joblib

    # Load model and config
    STOCK_DIR = os.path.join(config.STOCKS_DIR, ticker.upper(), "hourly")
    MODEL_PATH = os.path.join(STOCK_DIR, "hourly_model.joblib")
    PARAMS_PATH = os.path.join(STOCK_DIR, "hourly_params.json")
    CONFIG_PATH = os.path.join(STOCK_DIR, "training_config.json")

    try:
        model = joblib.load(MODEL_PATH)
        with open(PARAMS_PATH, "r") as f:
            optimized_params = json.load(f)
        with open(CONFIG_PATH, "r") as f:
            training_config = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Model/params/config for {ticker.upper()} not found.")
        return

    # Load threshold range
    start_thresh, end_thresh, step = 0.20, 0.91, 0.05
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                config_loaded = json.load(f)
                opt_range = config_loaded.get("optimization_range")
                if opt_range and all(k in opt_range for k in ["start", "end", "step"]):
                    start_thresh, end_thresh, step = (
                        opt_range["start"],
                        opt_range["end"],
                        opt_range["step"],
                    )
                    print(
                        f"--- Loaded custom range: {start_thresh*100:.1f}% to {end_thresh*100:.1f}% (step: {step*100:.1f}%) ---"
                    )
        except Exception as e:
            print(f"[WARNING] Could not load optimization range: {e}")

    thresholds_to_test = np.arange(start_thresh, end_thresh, step)

    # Fetch data
    api = tradeapi.REST(
        config.API_KEY, config.SECRET_KEY, base_url="https://paper-api.alpaca.markets"
    )
    start_date = "2024-01-01"
    end_date = pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(minutes=16)

    try:
        full_bars_hourly = api.get_bars(
            ticker.upper(),
            TimeFrame.Hour,
            start=start_date,
            end=end_date.isoformat(),
            adjustment="raw",
        ).df.tz_convert("America/New_York")

        full_bars_market_hourly = api.get_bars(
            config.MARKET_TICKER,
            TimeFrame.Hour,
            start=start_date,
            end=end_date.isoformat(),
            adjustment="raw",
        ).df.tz_convert("America/New_York")

        daily_data = api.get_bars(
            ticker.upper(),
            TimeFrame.Day,
            start=start_date,
            end=end_date.isoformat(),
            adjustment="raw",
        ).df.tz_convert("America/New_York")

        for df in [full_bars_hourly, full_bars_market_hourly, daily_data]:
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
        print(f"[ERROR] Failed to fetch data: {e}")
        return

    bars_hourly_2025 = full_bars_hourly[full_bars_hourly.index.year >= 2025]
    bars_market_hourly_2025 = full_bars_market_hourly[
        full_bars_market_hourly.index.year >= 2025
    ]

    if bars_hourly_2025.empty:
        print("[ERROR] No hourly data for 2025.")
        return

    # Pre-compute features
    print("[INFO] Pre-computing features...")
    all_features = pre_compute_hourly_features(
        bars_hourly_2025,
        bars_market_hourly_2025,
        daily_data,
        training_config,
        optimized_params,
        model,
    )

    aligned_prices = bars_hourly_2025.loc[all_features.index].copy()
    if aligned_prices.empty:
        print("[ERROR] No aligned data.")
        return

    # Add ATR
    atr_result = aligned_prices.ta.atr(length=14)
    if isinstance(atr_result, pd.DataFrame):
        aligned_prices["atr_stop_loss"] = atr_result[find_col(atr_result, "ATRr_14")]
    else:
        aligned_prices["atr_stop_loss"] = atr_result

    # Predict once
    print("[INFO] Generating predictions...")
    buy_probabilities = model.predict_proba(all_features)[:, 1]

    results = []
    best_result = {"threshold": None, "return": -np.inf, "trades": 0, "sharpe": 0}

    print("\n--- Testing thresholds ---")
    for threshold in tqdm(thresholds_to_test, desc="Optimizing Threshold"):
        trader = SmartDynamicHourlyTrader(
            ticker,
            model,
            optimized_params,
            training_config,
            threshold,
            start_cash=10000,
        )

        for i in range(len(aligned_prices)):
            date = aligned_prices.index[i]
            row = aligned_prices.iloc[i]
            current_price = row["Open"]
            confidence = buy_probabilities[i]
            atr = row.get("atr_stop_loss", row["Close"] * 0.015)

            if pd.isna(current_price) or pd.isna(atr):
                continue

            # 1. Update State
            trader.update_state(date, current_price, confidence)

            # 2. Check Exit
            should_exit, exit_reason, exit_pct = trader.should_reduce_or_exit(
                confidence, current_price, atr, date
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
                    date,
                )

            # 3. Check Entry
            should_enter, entry_reason = trader.should_enter_or_add(
                confidence, current_price, date
            )
            if should_enter:
                allocation = trader.calculate_dynamic_position_size(
                    confidence, current_price, atr, date
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
                        date,
                    )

        # Close final position
        if trader.shares_held > 0:
            final_price = aligned_prices["Close"].iloc[-1]
            trader.execute_trade(
                aligned_prices.index[-1],
                final_price,
                buy_probabilities[-1],
                aligned_prices["atr_stop_loss"].iloc[-1],
                "SELL",
                "END_OF_PERIOD",
                1.0,
                aligned_prices.index[-1],
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
                    252 * 6.5
                )

        results.append(
            {
                "threshold": threshold,
                "return": strategy_return,
                "trades": num_trades,
                "sharpe": sharpe_ratio,
            }
        )

        # Update best
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
    print(f"  Hourly Threshold Optimization Report for {ticker.upper()}")
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
        print(
            "WARNING: No optimal threshold found. Try adjusting the optimization range."
        )
        if results:
            best_result = max(results, key=lambda x: x["return"])
            print(f"Using best return: {best_result['threshold']:.2f}")

    print("=" * 60)

    # Save optimal threshold
    if best_result["threshold"] is not None:
        settings_path = os.path.join(STOCK_DIR, "hourly_settings.json")
        settings = {}
        if os.path.exists(settings_path):
            with open(settings_path, "r") as f:
                settings = json.load(f)
        settings["optimal_threshold"] = round(best_result["threshold"], 2)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=4)
        print(f"\n--- Saved optimal threshold to {settings_path} ---")