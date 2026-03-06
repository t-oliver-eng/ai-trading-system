# paper_trader_hourly.py - SMART DYNAMIC V3: Confidence-driven intraday trading

import pandas as pd
import numpy as np
import joblib
import json
import os
from utils import config
from testing.strategies_hourly import HOURLY_STRATEGIES, find_col
import pandas_ta as ta
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class SmartDynamicHourlyTrader:
    """Smart hourly trader using confidence as continuous signal."""

    def __init__(
        self,
        ticker,
        model,
        optimized_params,
        training_config,
        confidence_threshold,
        start_cash=10000,
    ):
        self.ticker = ticker
        self.model = model
        self.optimized_params = optimized_params
        self.training_config = training_config
        self.confidence_threshold = confidence_threshold
        self.start_cash = start_cash

        # Portfolio state
        self.cash = start_cash
        self.shares_held = 0
        self.entry_price = 0
        self.entry_confidence = 0
        self.current_confidence = 0

        # Position tracking
        self.highest_price = 0
        self.hours_in_position = 0

        # Tracking
        self.trades = []
        self.portfolio_history = []
        self.confidence_history = []
        self.winning_trades = 0
        self.trade_returns = []

        # Adaptive
        self.confidence_momentum = 0
        self.recent_confidence_values = []

    def is_market_hours(self, timestamp):
        """Check if during market hours."""
        hour = timestamp.hour
        minute = timestamp.minute
        if hour == 9 and minute < 30:
            return False
        if hour >= 16:
            return False
        if hour < 9:
            return False
        return True

    def detect_intraday_regime(self, prices_window, confidence_window):
        """Detect trending vs choppy intraday."""
        if len(prices_window) < 10:
            return "UNKNOWN"

        returns = prices_window.pct_change().dropna()
        trend_strength = abs(returns.mean()) / (returns.std() + 1e-9)

        if trend_strength > 0.10:
            return "TRENDING"
        else:
            return "CHOPPY"

    def calculate_dynamic_position_size(self, confidence, price, atr, timestamp):
        """Dynamic position sizing for hourly."""
        # Confidence strength
        confidence_strength = max(0, confidence - self.confidence_threshold) / (
            1.0 - self.confidence_threshold
        )
        confidence_factor = confidence_strength**0.7

        # Time of day adjustment (less aggressive at open/close)
        hour = timestamp.hour
        if 9 <= hour < 10:
            time_factor = 0.7  # Cautious at open
        elif 15 <= hour < 16:
            time_factor = 0.6  # Very cautious near close
        else:
            time_factor = 1.0

        # Volatility adjustment
        atr_pct = atr / price if price > 0 else 0
        volatility_factor = 1.0 - min(atr_pct / 0.05, 0.4)

        # Momentum bonus
        momentum_bonus = max(0, self.confidence_momentum) * 0.15

        allocation = (
            confidence_factor * time_factor * volatility_factor + momentum_bonus
        )
        return np.clip(allocation, 0.0, 0.90)

    def should_enter_or_add(self, confidence, price, timestamp):
        """Check if should enter or add to position."""
        if not self.is_market_hours(timestamp):
            return False, None

        # New entry
        if self.shares_held == 0:
            return confidence > self.confidence_threshold, "NEW_ENTRY"

        # Add to position (pyramid)
        if self.shares_held > 0:
            confidence_increase = confidence - self.entry_confidence
            in_profit = price > self.entry_price * 1.015  # 1.5% profit

            # Only add if confidence jumped and we're winning
            if (
                confidence_increase > 0.15
                and in_profit
                and self.confidence_momentum > 0
            ):
                return True, "ADD_TO_WINNER"

        return False, None

    def should_reduce_or_exit(self, confidence, price, atr, timestamp):
        """Determine exits."""
        if self.shares_held == 0:
            return False, None, 0

        profit_pct = (price / self.entry_price - 1) if self.entry_price > 0 else 0
        confidence_drop = self.entry_confidence - confidence

        # Exit conditions

        # 1. Confidence collapse
        if confidence < self.confidence_threshold * 0.80:
            return True, "CONFIDENCE_COLLAPSE", 1.0

        # 2. Confidence drop in loss
        if confidence_drop > 0.20 and profit_pct < -0.02:
            return True, "CONFIDENCE_DROP_LOSS", 1.0

        # 3. Dynamic trailing stop (confidence-based width)
        trailing_multiplier = 2.0 - (confidence * 0.3)
        trailing_stop = self.highest_price - (atr * trailing_multiplier)
        if price < trailing_stop:
            return True, "TRAILING_STOP", 1.0

        # 4. Partial profits on confidence fade
        if (
            confidence_drop > 0.15
            and profit_pct > 0.08
            and confidence > self.confidence_threshold
        ):
            return True, "PARTIAL_PROFIT", 0.5

        # 5. Gradual exit as confidence weakens
        if confidence < self.confidence_threshold + 0.08 and profit_pct > 0.03:
            return True, "GRADUAL_EXIT", 0.4

        # 6. Hard stop loss
        if profit_pct < -0.12:
            return True, "MAX_LOSS", 1.0

        # 7. Max hold time (prevent overnight drift)
        if self.hours_in_position >= 48:  # ~6 trading days
            return True, "MAX_HOLD_TIME", 1.0

        return False, None, 0

    def execute_trade(
        self,
        date,
        price,
        confidence,
        atr,
        action,
        reason,
        allocation=None,
        timestamp=None,
    ):
        """Execute trade."""
        if action == "BUY":
            if allocation is None:
                allocation = self.calculate_dynamic_position_size(
                    confidence, price, atr, timestamp or date
                )

            investment = self.cash * allocation
            shares_to_buy = int(investment / price)

            if shares_to_buy > 0 and investment <= self.cash:
                cost = shares_to_buy * price
                self.cash -= cost

                if self.shares_held > 0:
                    # Adding to position
                    total_cost = (self.shares_held * self.entry_price) + cost
                    self.shares_held += shares_to_buy
                    self.entry_price = total_cost / self.shares_held
                else:
                    # New position
                    self.shares_held = shares_to_buy
                    self.entry_price = price
                    self.entry_confidence = confidence
                    self.highest_price = price
                    self.hours_in_position = 0

                self.trades.append(
                    {
                        "date": date,
                        "action": "BUY",
                        "reason": reason,
                        "price": price,
                        "shares": shares_to_buy,
                        "value": cost,
                        "confidence": confidence,
                        "allocation": allocation,
                    }
                )
                return True

        elif action == "SELL":
            if allocation is None or allocation >= 1.0:
                shares_to_sell = self.shares_held
            else:
                shares_to_sell = int(self.shares_held * allocation)

            if shares_to_sell > 0:
                proceeds = shares_to_sell * price
                self.cash += proceeds

                cost_basis = shares_to_sell * self.entry_price
                trade_return = (proceeds / cost_basis - 1) if cost_basis > 0 else 0

                if shares_to_sell == self.shares_held:
                    self.trade_returns.append(trade_return)
                    if trade_return > 0:
                        self.winning_trades += 1

                self.trades.append(
                    {
                        "date": date,
                        "action": "SELL",
                        "reason": reason,
                        "price": price,
                        "shares": shares_to_sell,
                        "value": proceeds,
                        "return": trade_return,
                        "hold_hours": self.hours_in_position,
                        "confidence": confidence,
                    }
                )

                self.shares_held -= shares_to_sell

                if self.shares_held == 0:
                    self.entry_price = 0
                    self.entry_confidence = 0
                    self.hours_in_position = 0
                    self.highest_price = 0

                return True

        return False

    def update_state(self, date, price, confidence):
        """Update state."""
        # Confidence momentum
        self.recent_confidence_values.append(confidence)
        if len(self.recent_confidence_values) > 10:
            self.recent_confidence_values.pop(0)

        if len(self.recent_confidence_values) >= 5:
            self.confidence_momentum = confidence - np.mean(
                self.recent_confidence_values[-5:]
            )

        self.confidence_history.append((date, confidence))
        self.current_confidence = confidence

        # Position tracking
        if self.shares_held > 0:
            self.highest_price = max(self.highest_price, price)
            self.hours_in_position += 1

        # Portfolio value
        portfolio_value = self.cash + (self.shares_held * price)
        self.portfolio_history.append(
            {
                "date": date,
                "value": portfolio_value,
                "cash": self.cash,
                "position_value": self.shares_held * price,
                "shares": self.shares_held,
                "confidence": confidence,
            }
        )

        return portfolio_value


def pre_compute_hourly_features(
    bars_hourly,
    bars_market_hourly,
    daily_data,
    training_config,
    optimized_params,
    model,
):
    """Pre-compute hourly features."""
    print("[INFO] Pre-computing hourly features...")

    # Daily context
    daily_features = pd.DataFrame(index=daily_data.index)
    selected_indicators = training_config.get("indicators", [])
    selected_strategies = training_config.get("strategies", [])

    if "Daily_RSI" in selected_indicators:
        daily_features["Daily_RSI"] = daily_data.ta.rsi(length=14)
    if "Daily_50MA_Ratio" in selected_indicators:
        daily_features["Daily_50MA_Ratio"] = daily_data["Close"] / daily_data.ta.ema(
            length=50
        )
    if "Daily_200MA_Ratio" in selected_indicators:
        daily_features["Daily_200MA_Ratio"] = daily_data["Close"] / daily_data.ta.ema(
            length=200
        )

    # Hourly
    df_hourly = bars_hourly.copy()
    feature_df = pd.DataFrame(index=df_hourly.index)

    if "SPY_RSI_Hourly" in selected_indicators:
        feature_df["SPY_RSI_Hourly"] = bars_market_hourly.ta.rsi(length=14)

    if "atr" in selected_indicators:
        atr_result = df_hourly.ta.atr(length=14)
        if isinstance(atr_result, pd.DataFrame):
            col_name = find_col(atr_result, "ATRr_14")
            if col_name:
                feature_df["atr"] = atr_result[col_name]
        elif isinstance(atr_result, pd.Series):
            feature_df["atr"] = atr_result

    # Strategies
    for name in tqdm(selected_strategies, desc="Calculating Strategies"):
        if name in HOURLY_STRATEGIES:
            feature_df[name] = HOURLY_STRATEGIES[name](
                df_hourly.copy(), **optimized_params.get(name, {})
            )

    # Combine with Daily Context
    all_features = pd.merge_asof(
        feature_df.sort_index(),
        daily_features.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    model_features = None
    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_.tolist()
    else:
        try:
            model_features = model.get_booster().feature_names
        except:
            model_features = None

    if model_features is None:
        return all_features.dropna()

    for col in model_features:
        if col not in all_features.columns:
            all_features[col] = np.nan

    return all_features[model_features].dropna()


def run_hourly_paper_trader(ticker, start_cash=10000):
    print(f"--- Starting SMART DYNAMIC Hourly Trader for {ticker.upper()} ---")

    STOCK_DIR = os.path.join(config.STOCKS_DIR, ticker.upper(), "hourly")
    SETTINGS_PATH = os.path.join(STOCK_DIR, "hourly_settings.json")
    MODEL_PATH = os.path.join(STOCK_DIR, "hourly_model.joblib")
    PARAMS_PATH = os.path.join(STOCK_DIR, "hourly_params.json")
    CONFIG_PATH = os.path.join(STOCK_DIR, "training_config.json")

    CONFIDENCE_THRESHOLD = 0.80
    try:
        with open(SETTINGS_PATH, "r") as f:
            CONFIDENCE_THRESHOLD = json.load(f).get(
                "optimal_threshold", CONFIDENCE_THRESHOLD
            )
        print(
            f"--- Using confidence threshold: {CONFIDENCE_THRESHOLD:.2f} (as guide) ---"
        )
    except FileNotFoundError:
        print(f"Warning: Settings not found. Using default {CONFIDENCE_THRESHOLD}.")

    try:
        model = joblib.load(MODEL_PATH)
        with open(PARAMS_PATH, "r") as f:
            optimized_params = json.load(f)
        with open(CONFIG_PATH, "r") as f:
            training_config = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Model/params/config not found.")
        return

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
        print(f"Error fetching data: {e}")
        return

    bars_hourly_2025 = full_bars_hourly[full_bars_hourly.index.year >= 2025]
    bars_market_hourly_2025 = full_bars_market_hourly[
        full_bars_market_hourly.index.year >= 2025
    ]

    if bars_hourly_2025.empty:
        print("No hourly data for 2025.")
        return

    # Pre-compute
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
        print("No aligned data.")
        return

    # Add ATR
    atr_result = aligned_prices.ta.atr(length=14)
    if isinstance(atr_result, pd.DataFrame):
        aligned_prices["atr"] = atr_result[find_col(atr_result, "ATRr_14")]
    else:
        aligned_prices["atr"] = atr_result

    # Predict
    print("[INFO] Predicting confidence signals...")
    confidences = model.predict_proba(all_features)[:, 1]

    # Initialize trader
    trader = SmartDynamicHourlyTrader(
        ticker,
        model,
        optimized_params,
        training_config,
        CONFIDENCE_THRESHOLD,
        start_cash,
    )

    # Simulate
    print("[INFO] Running SMART DYNAMIC hourly simulation...")
    for i in tqdm(range(len(aligned_prices)), desc="Simulating"):
        date = aligned_prices.index[i]
        row = aligned_prices.iloc[i]
        price = row["Open"]
        confidence = confidences[i]
        atr = row.get("atr", row["Close"] * 0.015)

        if pd.isna(price) or pd.isna(atr):
            continue

        trader.update_state(date, price, confidence)

        # Detect regime
        if i >= 10:
            recent_prices = aligned_prices.iloc[max(0, i - 10) : i]["Close"]
            recent_conf = confidences[max(0, i - 10) : i]
            regime = trader.detect_intraday_regime(recent_prices, recent_conf)
        else:
            regime = "UNKNOWN"

        # Check exits
        should_exit, exit_reason, exit_pct = trader.should_reduce_or_exit(
            confidence, price, atr, date
        )
        if should_exit:
            trader.execute_trade(
                date, price, confidence, atr, "SELL", exit_reason, exit_pct, date
            )

        # Check entries
        should_enter, entry_reason = trader.should_enter_or_add(confidence, price, date)
        if should_enter:
            allocation = trader.calculate_dynamic_position_size(
                confidence, price, atr, date
            )
            if allocation > 0.05:
                trader.execute_trade(
                    date, price, confidence, atr, "BUY", entry_reason, allocation, date
                )

    # Close final
    if trader.shares_held > 0:
        final_price = aligned_prices["Close"].iloc[-1]
        trader.execute_trade(
            aligned_prices.index[-1],
            final_price,
            confidences[-1],
            aligned_prices["atr"].iloc[-1],
            "SELL",
            "END_OF_PERIOD",
            1.0,
            aligned_prices.index[-1],
        )

    # Metrics
    final_value = trader.cash
    strategy_return = (final_value / start_cash - 1) * 100
    num_trades = len([t for t in trader.trades if t["action"] == "SELL"])
    win_rate = trader.winning_trades / num_trades if num_trades > 0 else 0.0

    sharpe_ratio = 0.0
    if len(trader.trade_returns) > 1:
        returns_series = pd.Series(trader.trade_returns)
        if returns_series.std() > 0:
            sharpe_ratio = (returns_series.mean() / returns_series.std()) * np.sqrt(
                252 * 6.5
            )

    buy_and_hold_return = (
        bars_hourly_2025["Close"].iloc[-1] / bars_hourly_2025["Open"].iloc[0] - 1
    ) * 100

    # Max drawdown
    if trader.portfolio_history:
        portfolio_values = [h["value"] for h in trader.portfolio_history]
        peak = portfolio_values[0]
        max_dd = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
    else:
        max_dd = 0

    # Save
    results_data = {
        "total_return_pct": strategy_return,
        "buy_hold_return_pct": buy_and_hold_return,
        "sharpe_ratio": float(sharpe_ratio),
        "win_rate": float(win_rate),
        "total_trades": int(num_trades),
        "max_drawdown_pct": float(max_dd * 100),
        "confidence_threshold": float(CONFIDENCE_THRESHOLD),
        "test_period": "2025",
        "timestamp": pd.Timestamp.now().isoformat(),
        "strategy_type": "SMART_DYNAMIC_HOURLY",
        "avg_hold_hours": (
            np.mean(
                [t.get("hold_hours", 0) for t in trader.trades if t["action"] == "SELL"]
            )
            if num_trades > 0
            else 0
        ),
    }

    results_path = os.path.join(STOCK_DIR, "paper_trade_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"\n--- Saved results to {results_path} ---")

    # Print
    print(
        f"\n{'='*50}\n  SMART DYNAMIC Hourly Results for {ticker.upper()} (2025)\n{'='*50}"
    )
    print(f"  AI Strategy Return:  {strategy_return:.2f}%")
    print(f"  Buy and Hold Return: {buy_and_hold_return:.2f}%")
    print(f"  Sharpe Ratio:        {sharpe_ratio:.2f}")
    print(f"  Win Rate:            {win_rate*100:.1f}%")
    print(f"  Number of Trades:    {num_trades}")
    print(f"  Max Drawdown:        {max_dd*100:.1f}%")
    print(f"  Avg Hold Period:     {results_data['avg_hold_hours']:.1f} hours")
    print("=" * 50)

    # Plot
    if trader.portfolio_history:
        history_df = pd.DataFrame(trader.portfolio_history).set_index("date")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

        ax1.plot(
            history_df["value"],
            label="Smart Dynamic Hourly",
            linewidth=2,
            color="#00e676",
        )
        buy_hold_equity = (
            bars_hourly_2025["Close"] / bars_hourly_2025["Open"].iloc[0]
        ) * start_cash
        ax1.plot(
            buy_hold_equity,
            label="Buy and Hold",
            linestyle="--",
            linewidth=2,
            color="#ff6b6b",
        )
        ax1.set_title(
            f"Smart Dynamic Hourly vs Buy and Hold - {ticker.upper()} (2025)",
            fontsize=14,
        )
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        ax2.plot(
            history_df["confidence"], label="Confidence", linewidth=1.5, color="#4dabf7"
        )
        ax2.axhline(
            y=CONFIDENCE_THRESHOLD,
            color="red",
            linestyle="--",
            label="Threshold",
            linewidth=1,
        )
        ax2.fill_between(
            history_df.index, 0, history_df["confidence"], alpha=0.2, color="#4dabf7"
        )
        ax2.set_ylabel("Confidence", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])

        plt.tight_layout()
        plot_path = os.path.join(STOCK_DIR, "performance_hourly.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\n--- Saved plot to {plot_path} ---")
