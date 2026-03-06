# paper_trader.py - SMART DYNAMIC V3: Confidence as continuous signal, adaptive position management

import pandas as pd
import numpy as np
import joblib
import json
from utils import config
import os
from testing.strategies import TOP_STRATEGIES, find_col
import pandas_ta as ta
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class SmartDynamicTrader:
    """Smart trader that uses confidence as a continuous, evolving signal."""

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
        self.position_value = 0
        self.entry_price = 0
        self.entry_confidence = 0
        self.current_confidence = 0

        # Position management
        self.highest_price = 0
        self.lowest_price_after_entry = float("inf")
        self.days_in_position = 0
        self.consecutive_high_confidence_days = 0

        # Tracking
        self.trades = []
        self.portfolio_history = []
        self.confidence_history = []
        self.winning_trades = 0
        self.trade_returns = []

        # Adaptive parameters
        self.recent_volatility = []  # Track recent market volatility
        self.confidence_momentum = 0  # Track if confidence is rising/falling

    def detect_market_regime(self, prices_window, confidence_window):
        """Detect if we're in trending or choppy market."""
        if len(prices_window) < 20:
            return "UNKNOWN"

        # Calculate directional movement
        returns = prices_window.pct_change().dropna()
        trend_strength = abs(returns.mean()) / (returns.std() + 1e-9)

        # High trend strength = trending, low = choppy
        if trend_strength > 0.15:
            return "TRENDING"
        else:
            return "CHOPPY"

    def calculate_dynamic_position_size(self, confidence, price, atr, regime):
        """Calculate position size using confidence as primary driver."""
        # Base: confidence strength relative to threshold
        confidence_strength = max(0, confidence - self.confidence_threshold) / (
            1.0 - self.confidence_threshold
        )

        # Non-linear scaling: high confidence gets exponentially more allocation
        # This allows the system to "load up" on high-conviction trades
        confidence_factor = confidence_strength**0.6  # Softened exponential

        # Regime adjustment
        if regime == "TRENDING":
            regime_multiplier = 1.2  # Be more aggressive in trends
        elif regime == "CHOPPY":
            regime_multiplier = 0.7  # Be more defensive when choppy
        else:
            regime_multiplier = 1.0

        # Volatility adjustment (use ATR)
        atr_pct = atr / price if price > 0 else 0
        volatility_factor = 1.0 - min(atr_pct / 0.08, 0.4)

        # Confidence momentum: if confidence is rising, be more aggressive
        momentum_bonus = max(0, self.confidence_momentum) * 0.1

        # Calculate final allocation (can go from 0% to 95%)
        allocation = (
            confidence_factor * regime_multiplier * volatility_factor + momentum_bonus
        )
        allocation = np.clip(allocation, 0.0, 0.95)

        return allocation

    def should_enter_or_add(self, confidence, price, atr, regime):
        """Determine if we should enter a new position or add to existing."""
        # Entry: confidence above threshold
        if self.shares_held == 0:
            return confidence > self.confidence_threshold, "NEW_ENTRY"

        # Already in position: check if we should add (pyramid)
        if self.shares_held > 0:
            # Only add if:
            # 1. Confidence is significantly higher than entry
            # 2. We're in profit
            # 3. Confidence is rising
            confidence_increase = confidence - self.entry_confidence
            in_profit = price > self.entry_price * 1.02  # At least 2% profit

            if (
                confidence_increase > 0.10
                and in_profit
                and self.confidence_momentum > 0
            ):
                return True, "ADD_TO_WINNER"

        return False, None

    def should_reduce_or_exit(self, confidence, price, atr, regime):
        """Determine if we should reduce or exit position based on confidence."""
        if self.shares_held == 0:
            return False, None, 0

        # Calculate position metrics
        profit_pct = (price / self.entry_price - 1) if self.entry_price > 0 else 0
        confidence_drop = self.entry_confidence - confidence

        # Exit reasons (in priority order):

        # 1. Confidence drops significantly below threshold
        if confidence < self.confidence_threshold * 0.85:
            return True, "CONFIDENCE_COLLAPSE", 1.0  # Exit 100%

        # 2. Large confidence drop while in loss
        if confidence_drop > 0.15 and profit_pct < -0.03:
            return True, "CONFIDENCE_DROP_IN_LOSS", 1.0

        # 3. Trailing stop (ATR-based, dynamic width based on confidence)
        # Higher confidence = tighter trailing stop
        trailing_multiplier = 3.0 - (confidence * 0.5)  # 2.5x to 3.0x ATR
        trailing_stop = self.highest_price - (atr * trailing_multiplier)
        if price < trailing_stop:
            return True, "TRAILING_STOP", 1.0

        # 4. Take partial profits if confidence weakening but still above threshold
        if (
            confidence_drop > 0.10
            and profit_pct > 0.10
            and confidence > self.confidence_threshold
        ):
            return True, "PARTIAL_PROFIT", 0.5  # Reduce 50%

        # 5. Scale out gradually as confidence fades
        if confidence < self.confidence_threshold + 0.05 and profit_pct > 0.05:
            return True, "GRADUAL_EXIT", 0.3  # Reduce 30%

        # 6. Time-based: held too long with no confidence improvement
        if self.days_in_position > 30 and confidence < self.entry_confidence:
            return True, "TIME_DECAY", 0.5

        # 7. Maximum drawdown from peak (hard stop)
        if profit_pct < -0.15:  # -15% from entry
            return True, "MAX_LOSS", 1.0

        return False, None, 0

    def execute_trade(
        self, date, price, confidence, atr, action, reason, allocation=None
    ):
        """Execute a trade (buy, sell, add, reduce)."""
        if action == "BUY":
            # New position or adding to position
            if allocation is None:
                allocation = self.calculate_dynamic_position_size(
                    confidence, price, atr, "UNKNOWN"
                )

            investment = self.cash * allocation
            shares_to_buy = int(investment / price)

            if shares_to_buy > 0 and investment <= self.cash:
                cost = shares_to_buy * price
                self.cash -= cost

                # Update entry price (weighted average if adding)
                if self.shares_held > 0:
                    total_cost = (self.shares_held * self.entry_price) + cost
                    self.shares_held += shares_to_buy
                    self.entry_price = total_cost / self.shares_held
                else:
                    self.shares_held = shares_to_buy
                    self.entry_price = price
                    self.entry_confidence = confidence
                    self.highest_price = price
                    self.lowest_price_after_entry = price
                    self.days_in_position = 0

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
            # Selling partial or full position
            if allocation is None or allocation >= 1.0:
                # Full exit
                shares_to_sell = self.shares_held
            else:
                # Partial exit
                shares_to_sell = int(self.shares_held * allocation)

            if shares_to_sell > 0:
                proceeds = shares_to_sell * price
                self.cash += proceeds

                # Calculate return on this portion
                cost_basis = shares_to_sell * self.entry_price
                trade_return = (proceeds / cost_basis - 1) if cost_basis > 0 else 0

                if shares_to_sell == self.shares_held:
                    # Full exit
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
                        "hold_days": self.days_in_position,
                        "confidence": confidence,
                    }
                )

                self.shares_held -= shares_to_sell

                # Reset position tracking if fully exited
                if self.shares_held == 0:
                    self.entry_price = 0
                    self.entry_confidence = 0
                    self.days_in_position = 0
                    self.highest_price = 0

                return True

        return False

    def update_state(self, date, price, confidence):
        """Update internal state tracking."""
        # Track confidence momentum
        if len(self.confidence_history) > 5:
            recent_confidences = [c for _, c in self.confidence_history[-5:]]
            self.confidence_momentum = confidence - np.mean(recent_confidences)

        self.confidence_history.append((date, confidence))
        self.current_confidence = confidence

        # Update position tracking
        if self.shares_held > 0:
            self.highest_price = max(self.highest_price, price)
            self.lowest_price_after_entry = min(self.lowest_price_after_entry, price)
            self.days_in_position += 1

        # Track portfolio value
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


def pre_compute_daily_features(
    all_bars_ticker, all_bars_market, optimized_params, training_config, model
):
    """Pre-compute all necessary features."""
    print("[INFO] Pre-computing all daily features...")
    df = all_bars_ticker.copy()
    market_df = all_bars_market.copy()

    selected_indicators = training_config.get("indicators", [])
    selected_strategies = training_config.get("strategies", [])

    # Start with an EMPTY DataFrame to ensure we don't include OHLCV/temporary columns
    feature_df = pd.DataFrame(index=df.index)

    # Market Features
    market_df_features = pd.DataFrame(index=market_df.index)
    if "SPY_RSI" in selected_indicators:
        market_df_features["SPY_RSI"] = market_df.ta.rsi(length=14)
    if "SPY_50MA_Ratio" in selected_indicators:
        market_df_features["SPY_50MA_Ratio"] = market_df["Close"] / market_df.ta.ema(length=50)
    if "SPY_200MA_Ratio" in selected_indicators:
        market_df_features["SPY_200MA_Ratio"] = market_df["Close"] / market_df.ta.ema(length=200)

    # Ticker Indicators (MATCHING ORDER IN trainer.py)
    if "RSI" in selected_indicators:
        feature_df["RSI_14"] = df.ta.rsi(length=14)
    if "ATR" in selected_indicators:
        atr_result = df.ta.atr(length=14)
        if isinstance(atr_result, pd.DataFrame):
            col_name = find_col(atr_result, "ATRr_14")
            if col_name:
                feature_df["ATR_14"] = atr_result[col_name]
        elif isinstance(atr_result, pd.Series):
            feature_df["ATR_14"] = atr_result
    if "ADX" in selected_indicators:
        adx_df = df.ta.adx(length=14)
        if adx_df is not None:
            feature_df["ADX_14"] = adx_df[find_col(adx_df, "ADX_14")]
    if "CMF" in selected_indicators:
        feature_df["CMF_20"] = df.ta.cmf(length=20)
    if "MACD" in selected_indicators:
        macd_df = df.ta.macd(fast=12, slow=26, signal=9)
        if macd_df is not None:
            feature_df["MACD_12_26_9"] = macd_df[find_col(macd_df, "MACD_")]
            feature_df["MACDh_12_26_9"] = macd_df[find_col(macd_df, "MACDh_")]
    if "Stochastic" in selected_indicators:
        stoch_df = df.ta.stoch(k=14, d=3)
        if stoch_df is not None:
            feature_df["STOCHk_14_3_3"] = stoch_df[find_col(stoch_df, "STOCHk_")]
    if "VWAP" in selected_indicators:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            feature_df["VWAP"] = df.ta.vwap()
    if "OBV" in selected_indicators:
        feature_df["OBV"] = df.ta.obv()

    # Strategies
    for name in selected_strategies:
        if name in TOP_STRATEGIES:
            feature_df[name] = TOP_STRATEGIES[name](df.copy(), **optimized_params.get(name, {}))

    # Join with market features at the end (MATCHING trainer.py)
    combined_df = feature_df.join(market_df_features.dropna())

    model_features = None
    if hasattr(model, "feature_names_in_"):
        model_features = model.feature_names_in_.tolist()
    else:
        try:
            model_features = model.get_booster().feature_names
        except:
            model_features = None

    if model_features is None:
        # We don't need a warning here because we built feature_df cleanly to match the trainer
        return combined_df.dropna()

    for col in model_features:
        if col not in combined_df.columns:
            combined_df[col] = np.nan

    return combined_df[model_features].dropna()


def run_paper_trader(ticker, start_cash=10000):
    print(f"--- Starting SMART DYNAMIC Daily Trader for {ticker.upper()} ---")
    STOCK_DIR = os.path.join(config.STOCKS_DIR, ticker.upper(), "daily")
    SETTINGS_PATH = os.path.join(STOCK_DIR, "daily_settings.json")
    AI_MODEL_PATH = os.path.join(STOCK_DIR, "daily_model.joblib")
    PARAMS_PATH = os.path.join(STOCK_DIR, "daily_params.json")
    CONFIG_PATH = os.path.join(STOCK_DIR, "training_config.json")

    CONFIDENCE_THRESHOLD = 0.55
    try:
        with open(SETTINGS_PATH, "r") as f:
            CONFIDENCE_THRESHOLD = json.load(f).get(
                "optimal_threshold", CONFIDENCE_THRESHOLD
            )
        print(
            f"--- Using confidence threshold: {CONFIDENCE_THRESHOLD:.2f} (as guide, not gate) ---"
        )
    except FileNotFoundError:
        print(
            f"Warning: Settings file not found. Using default threshold {CONFIDENCE_THRESHOLD}."
        )

    try:
        model = joblib.load(AI_MODEL_PATH)
        with open(PARAMS_PATH, "r") as f:
            optimized_params = json.load(f)
        with open(CONFIG_PATH, "r") as f:
            training_config = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Model/params/config for {ticker.upper()} not found.")
        return

    api = tradeapi.REST(
        config.API_KEY, config.SECRET_KEY, base_url="https://paper-api.alpaca.markets"
    )
    hist_start_date = "2024-01-01"
    end_date = pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(minutes=16)

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

    if all_bars_ticker[all_bars_ticker.index.year >= 2025].empty:
        print("No data for 2025.")
        return

    # Pre-compute features
    all_features = pre_compute_daily_features(
        all_bars_ticker, all_bars_market, optimized_params, training_config, model
    )

    test_prices = all_bars_ticker[all_bars_ticker.index.year >= 2025].copy()
    test_features = all_features[all_features.index.year >= 2025]

    aligned_index = test_prices.index.intersection(test_features.index)
    if aligned_index.empty:
        print("No aligned data.")
        return

    test_prices = test_prices.loc[aligned_index]
    test_features = test_features.loc[aligned_index]

    # Add ATR
    atr_result = test_prices.ta.atr(length=14)
    if isinstance(atr_result, pd.DataFrame):
        test_prices["atr"] = atr_result[find_col(atr_result, "ATRr_14")]
    else:
        test_prices["atr"] = atr_result

    # Predict all signals
    print("[INFO] Predicting confidence signals...")
    confidences = model.predict_proba(test_features)[:, 1]

    # Initialize smart trader
    trader = SmartDynamicTrader(
        ticker,
        model,
        optimized_params,
        training_config,
        CONFIDENCE_THRESHOLD,
        start_cash,
    )

    # Run smart simulation
    print("[INFO] Running SMART DYNAMIC simulation...")
    for i in tqdm(range(len(test_prices)), desc="Simulating"):
        date = test_prices.index[i]
        row = test_prices.iloc[i]
        price = row["Open"]
        confidence = confidences[i]
        atr = row.get("atr", row["Close"] * 0.02)

        if pd.isna(price) or pd.isna(atr):
            continue

        # Update state
        trader.update_state(date, price, confidence)

        # Detect market regime
        if i >= 20:
            recent_prices = test_prices.iloc[max(0, i - 20) : i]["Close"]
            recent_conf = confidences[max(0, i - 20) : i]
            regime = trader.detect_market_regime(recent_prices, recent_conf)
        else:
            regime = "UNKNOWN"

        # Check if we should exit/reduce first
        should_exit, exit_reason, exit_pct = trader.should_reduce_or_exit(
            confidence, price, atr, regime
        )
        if should_exit:
            trader.execute_trade(
                date, price, confidence, atr, "SELL", exit_reason, exit_pct
            )

        # Check if we should enter/add
        should_enter, entry_reason = trader.should_enter_or_add(
            confidence, price, atr, regime
        )
        if should_enter:
            allocation = trader.calculate_dynamic_position_size(
                confidence, price, atr, regime
            )
            if allocation > 0.05:  # Only trade if allocation is meaningful
                trader.execute_trade(
                    date, price, confidence, atr, "BUY", entry_reason, allocation
                )

    # Close final position
    if trader.shares_held > 0:
        final_price = test_prices["Close"].iloc[-1]
        trader.execute_trade(
            test_prices.index[-1],
            final_price,
            confidences[-1],
            test_prices["atr"].iloc[-1],
            "SELL",
            "END_OF_PERIOD",
            1.0,
        )

    # Calculate metrics
    final_value = trader.cash
    strategy_return = (final_value / start_cash - 1) * 100
    num_trades = len([t for t in trader.trades if t["action"] == "SELL"])
    win_rate = trader.winning_trades / num_trades if num_trades > 0 else 0.0

    sharpe_ratio = 0.0
    if len(trader.trade_returns) > 1:
        returns_series = pd.Series(trader.trade_returns)
        if returns_series.std() > 0:
            sharpe_ratio = (returns_series.mean() / returns_series.std()) * np.sqrt(252)

    test_data = all_bars_ticker[all_bars_ticker.index.year >= 2025]
    buy_and_hold_return = (
        test_data["Close"].iloc[-1] / test_data["Open"].iloc[0] - 1
    ) * 100

    # Calculate max drawdown
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

    # Save results
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
        "strategy_type": "SMART_DYNAMIC",
        "avg_hold_days": (
            np.mean(
                [t.get("hold_days", 0) for t in trader.trades if t["action"] == "SELL"]
            )
            if num_trades > 0
            else 0
        ),
    }

    results_path = os.path.join(STOCK_DIR, "paper_trade_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=4)
    print(f"\n--- Saved results to {results_path} ---")

    # Print summary
    print(
        f"\n{'='*50}\n  SMART DYNAMIC Daily Results for {ticker.upper()} (2025)\n{'='*50}"
    )
    print(f"  AI Strategy Return:  {strategy_return:.2f}%")
    print(f"  Buy and Hold Return: {buy_and_hold_return:.2f}%")
    print(f"  Sharpe Ratio:        {sharpe_ratio:.2f}")
    print(f"  Win Rate:            {win_rate*100:.1f}%")
    print(f"  Number of Trades:    {num_trades}")
    print(f"  Max Drawdown:        {max_dd*100:.1f}%")
    print(f"  Avg Hold Period:     {results_data['avg_hold_days']:.1f} days")
    print("=" * 50)

    # Plot
    if trader.portfolio_history:
        history_df = pd.DataFrame(trader.portfolio_history).set_index("date")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])

        # Portfolio value
        ax1.plot(
            history_df["value"],
            label=f"Smart Dynamic AI ({ticker.upper()})",
            linewidth=2,
            color="#00e676",
        )
        buy_hold_equity = (test_data["Close"] / test_data["Open"].iloc[0]) * start_cash
        ax1.plot(
            buy_hold_equity,
            label=f"Buy and Hold",
            linestyle="--",
            linewidth=2,
            color="#ff6b6b",
        )
        ax1.set_title(
            f"Smart Dynamic AI vs Buy and Hold - {ticker.upper()} (2025)", fontsize=14
        )
        ax1.set_ylabel("Portfolio Value ($)", fontsize=12)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Confidence over time
        ax2.plot(
            history_df["confidence"],
            label="AI Confidence",
            linewidth=1.5,
            color="#4dabf7",
        )
        ax2.axhline(
            y=CONFIDENCE_THRESHOLD,
            color="red",
            linestyle="--",
            label="Threshold (guide)",
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
        plot_path = os.path.join(STOCK_DIR, "performance_daily.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\n--- Saved performance plot to {plot_path} ---")
