#!/usr/bin/env python3
"""
run_v33_backtest.py - V3.3 Forward Test with REAL AI Predictions

Forward tests V3.3 strategy on NEW data the AI hasn't seen during training.
Uses REAL daily and hourly predictions matching predictor.py logic exactly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import json
import os
import joblib
import warnings

# Import backtester
from testing.backtester import RealisticBacktester

# Import config and strategies
from utils import config
from testing.strategies import TOP_STRATEGIES, find_col
from testing.strategies_hourly import HOURLY_STRATEGIES
import pandas_ta as ta

# Import Alpaca API
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def ensure_data_exists(ticker, start_date, end_date):
    """Download data if it doesn't exist for the test period"""
    
    # Check if we have data for this period
    daily_path = os.path.join(config.DATA_DIR, f"{ticker}.csv")
    hourly_path = os.path.join(config.DATA_DIR, "hourly", f"{ticker}.csv")
    
    need_download = False
    
    # Define required range (timezone aware)
    req_start = pd.Timestamp(start_date).tz_localize("America/New_York")
    req_end = pd.Timestamp(end_date).tz_localize("America/New_York")
    
    # Check daily data
    if os.path.exists(daily_path):
        try:
            df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
            # Handle timezone-aware strings by parsing as UTC first
            df.index = pd.to_datetime(df.index, utc=True)
            df.index = df.index.tz_convert("America/New_York")
            
            file_start = df.index[0]
            file_end = df.index[-1]
            
            # FIXED: Check BOTH start and end
            if file_start > req_start + timedelta(days=5): # Allow small buffer
                print(f"ℹ  Daily data starts too late ({file_start.date()}), need from {req_start.date()}")
                need_download = True
            elif file_end < req_end:
                print(f"ℹ  Daily data ends too early ({file_end.date()}), need up to {req_end.date()}")
                need_download = True
        except Exception as e:
            print(f" Error reading daily CSV: {e}")
            need_download = True
    else:
        need_download = True
        print(f"ℹ  No daily data found, will download")
    
    # Check hourly data
    if not need_download and os.path.exists(hourly_path):
        try:
            df = pd.read_csv(hourly_path, index_col=0, parse_dates=True)
            # Handle timezone-aware strings by parsing as UTC first
            df.index = pd.to_datetime(df.index, utc=True)
            df.index = df.index.tz_convert("America/New_York")
            
            file_start = df.index[0]
            file_end = df.index[-1]
            
            # FIXED: Check BOTH start and end
            if file_start > req_start + timedelta(days=5):
                print(f"ℹ  Hourly data starts too late ({file_start.date()}), need from {req_start.date()}")
                need_download = True
            elif file_end < req_end:
                print(f"ℹ  Hourly data ends too early ({file_end.date()}), need up to {req_end.date()}")
                need_download = True
        except Exception:
            need_download = True
    elif not os.path.exists(hourly_path):
        need_download = True
        print(f"ℹ  No hourly data found, will download")
    
    if need_download:
        print(f"\n Downloading fresh data for {ticker}...")
        download_test_data(ticker, start_date, end_date)
    else:
        print(f" Data already exists for test period")


def download_test_data(ticker, start_date, end_date):
    """Download both daily and hourly data for testing period"""
    
    api = tradeapi.REST(
        config.API_KEY,
        config.SECRET_KEY,
        base_url="https://paper-api.alpaca.markets"
    )
    
    # Add buffer for indicators (need history)
    # Important: If requesting 2017, this fetches 2016 for calculation buffer
    buffer_start = (pd.Timestamp(start_date) - timedelta(days=365)).strftime("%Y-%m-%d")
    
    print(f"   Downloading daily data (from {buffer_start})...")
    try:
        # Download daily
        daily_bars = api.get_bars(
            ticker,
            TimeFrame.Day,
            start=buffer_start,
            end=end_date,
            adjustment="raw"
        ).df.tz_convert("America/New_York")
        
        daily_bars.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        
        # Save daily
        os.makedirs(config.DATA_DIR, exist_ok=True)
        daily_path = os.path.join(config.DATA_DIR, f"{ticker}.csv")
        daily_bars.to_csv(daily_path)
        print(f"  ✅ Saved {len(daily_bars)} daily bars to {daily_path}")
        
    except Exception as e:
        print(f"   Error downloading daily data: {e}")
        raise
    
    print(f"   Downloading hourly data...")
    try:
        # Download hourly
        hourly_bars = api.get_bars(
            ticker,
            TimeFrame.Hour,
            start=buffer_start,
            end=end_date,
            adjustment="raw"
        ).df.tz_convert("America/New_York")
        
        hourly_bars.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        
        # Save hourly
        hourly_dir = os.path.join(config.DATA_DIR, "hourly")
        os.makedirs(hourly_dir, exist_ok=True)
        hourly_path = os.path.join(hourly_dir, f"{ticker}.csv")
        hourly_bars.to_csv(hourly_path)
        print(f"   Saved {len(hourly_bars)} hourly bars to {hourly_path}")
        
    except Exception as e:
        print(f"   Error downloading hourly data: {e}")
        raise
    
    # Download SPY for market context
    print(f"  Downloading SPY data...")
    try:
        spy_daily = api.get_bars(
            config.MARKET_TICKER,
            TimeFrame.Day,
            start=buffer_start,
            end=end_date,
            adjustment="raw"
        ).df.tz_convert("America/New_York")
        
        spy_daily.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        
        spy_path = os.path.join(config.DATA_DIR, f"{config.MARKET_TICKER}.csv")
        spy_daily.to_csv(spy_path)
        print(f"  Saved SPY data")
        
        # SPY hourly
        spy_hourly = api.get_bars(
            config.MARKET_TICKER,
            TimeFrame.Hour,
            start=buffer_start,
            end=end_date,
            adjustment="raw"
        ).df.tz_convert("America/New_York")
        
        spy_hourly.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low',
            'close': 'Close', 'volume': 'Volume'
        }, inplace=True)
        
        spy_hourly_path = os.path.join(hourly_dir, f"{config.MARKET_TICKER}.csv")
        spy_hourly.to_csv(spy_hourly_path)
        print(f"  Saved SPY hourly data\n")
        
    except Exception as e:
        print(f"  Warning: Could not download SPY data: {e}")


class RealDailyPredictor:
    """Generates REAL daily AI predictions (matches predictor.py exactly)"""
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        
        # Load model and configs
        STOCK_DIR = os.path.join(config.STOCKS_DIR, self.ticker, "daily")
        
        self.model = joblib.load(os.path.join(STOCK_DIR, "daily_model.joblib"))
        
        with open(os.path.join(STOCK_DIR, "daily_params.json")) as f:
            self.optimized_params = json.load(f)
        
        with open(os.path.join(STOCK_DIR, "training_config.json")) as f:
            self.training_config = json.load(f)
        
        with open(os.path.join(STOCK_DIR, "daily_settings.json")) as f:
            self.threshold = json.load(f).get("optimal_threshold", 0.55)
        
        self.selected_indicators = self.training_config.get("indicators", [])
        self.selected_strategies = self.training_config.get("strategies", [])
        
        print(f"Loaded Daily AI - Threshold: {self.threshold:.2%}")
    
    def generate_features(self, bars_ticker, bars_market, date):
        """Generate features for a specific date (matching predictor.py logic)"""
        
        # Get data up to this date
        ticker_data = bars_ticker[bars_ticker.index <= date].copy()
        market_data = bars_market[bars_market.index <= date].copy()
        
        if len(ticker_data) < 200:  # Need enough data for indicators
            return None
        
        features = {}
        
        # SPY indicators
        if "SPY_RSI" in self.selected_indicators:
            features["SPY_RSI"] = market_data.ta.rsi(length=14).iloc[-1]
        
        if "SPY_50MA_Ratio" in self.selected_indicators:
            features["SPY_50MA_Ratio"] = (
                market_data["Close"].iloc[-1] / market_data.ta.ema(length=50).iloc[-1]
            )
        
        if "SPY_200MA_Ratio" in self.selected_indicators:
            features["SPY_200MA_Ratio"] = (
                market_data["Close"].iloc[-1] / market_data.ta.ema(length=200).iloc[-1]
            )
        
        # Ticker indicators
        if "RSI" in self.selected_indicators:
            features["RSI_14"] = ticker_data.ta.rsi(length=14).iloc[-1]
        
        if "ATR" in self.selected_indicators:
            atr_result = ticker_data.ta.atr(length=14)
            if isinstance(atr_result, pd.DataFrame):
                col_name = find_col(atr_result, "ATRr_14")
                if col_name:
                    features["ATR_14"] = atr_result[col_name].iloc[-1]
            elif isinstance(atr_result, pd.Series):
                features["ATR_14"] = atr_result.iloc[-1]
        
        if "ADX" in self.selected_indicators:
            adx_df = ticker_data.ta.adx(length=14)
            if adx_df is not None and not adx_df.empty:
                features["ADX_14"] = adx_df[find_col(adx_df, "ADX_14")].iloc[-1]
        
        if "CMF" in self.selected_indicators:
            features["CMF_20"] = ticker_data.ta.cmf(length=20).iloc[-1]
        
        if "MACD" in self.selected_indicators:
            macd_df = ticker_data.ta.macd(fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty:
                features["MACD_12_26_9"] = macd_df[find_col(macd_df, "MACD_")].iloc[-1]
                features["MACDh_12_26_9"] = macd_df[find_col(macd_df, "MACDh_")].iloc[-1]
        
        if "Stochastic" in self.selected_indicators:
            stoch_df = ticker_data.ta.stoch(k=14, d=3)
            if stoch_df is not None and not stoch_df.empty:
                features["STOCHk_14_3_3"] = stoch_df[find_col(stoch_df, "STOCHk_")].iloc[-1]
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if "VWAP" in self.selected_indicators:
                vwap = ticker_data.ta.vwap()
                if vwap is not None and len(vwap) > 0:
                    features["VWAP"] = vwap.iloc[-1]
        
        if "OBV" in self.selected_indicators:
            features["OBV"] = ticker_data.ta.obv().iloc[-1]
        
        # Strategy signals
        for name in self.selected_strategies:
            if name in TOP_STRATEGIES:
                params = self.optimized_params.get(name, {})
                signal = TOP_STRATEGIES[name](ticker_data.copy(), **params)
                if signal is not None and len(signal) > 0:
                    features[name] = signal.iloc[-1]
        
        return features
    
    def predict(self, bars_ticker, bars_market, date):
        """Get prediction for a specific date"""
        features = self.generate_features(bars_ticker, bars_market, date)
        
        if features is None:
            return None
        
        # Create DataFrame
        input_df = pd.DataFrame([features])
        
        # Ensure correct feature order
        try:
            input_df = input_df[self.model.get_booster().feature_names]
        except:
            return None
        
        # Fill any NaN with 0
        input_df = input_df.fillna(0)
        
        # Predict
        buy_probability = self.model.predict_proba(input_df)[0][1]
        
        return buy_probability


class RealHourlyPredictor:
    """Generates REAL hourly AI predictions (matches predictor_hourly.py exactly)"""
    
    def __init__(self, ticker):
        self.ticker = ticker.upper()
        
        # Load model and configs
        STOCK_DIR = os.path.join(config.STOCKS_DIR, self.ticker, "hourly")
        
        self.model = joblib.load(os.path.join(STOCK_DIR, "hourly_model.joblib"))
        
        with open(os.path.join(STOCK_DIR, "hourly_params.json")) as f:
            self.optimized_params = json.load(f)
        
        with open(os.path.join(STOCK_DIR, "training_config.json")) as f:
            self.training_config = json.load(f)
        
        with open(os.path.join(STOCK_DIR, "hourly_settings.json")) as f:
            self.threshold = json.load(f).get("optimal_threshold", 0.80)
        
        self.selected_indicators = self.training_config.get("indicators", [])
        self.selected_strategies = self.training_config.get("strategies", [])
        
        print(f"Loaded Hourly AI - Threshold: {self.threshold:.2%}")
    
    def predict(self, hourly_bars, market_hourly_bars, daily_bars, timestamp):
        """Get hourly prediction for specific timestamp"""
        
        # Get data up to this timestamp
        hourly_data = hourly_bars[hourly_bars.index <= timestamp].copy()
        market_hourly_data = market_hourly_bars[market_hourly_bars.index <= timestamp].copy()
        daily_data = daily_bars[daily_bars.index <= timestamp].copy()
        
        if len(hourly_data) < 50:  # Need enough hourly data
            return None
        
        if len(daily_data) < 200:  # Need enough daily data for context
            return None
        
        features = {}
        
        # Daily Context Features
        if "Daily_RSI" in self.selected_indicators:
            features["Daily_RSI"] = daily_data.ta.rsi(length=14).iloc[-1]
        
        if "Daily_50MA_Ratio" in self.selected_indicators:
            features["Daily_50MA_Ratio"] = (
                daily_data["Close"].iloc[-1] / daily_data.ta.ema(length=50).iloc[-1]
            )
        
        if "Daily_200MA_Ratio" in self.selected_indicators:
            features["Daily_200MA_Ratio"] = (
                daily_data["Close"].iloc[-1] / daily_data.ta.ema(length=200).iloc[-1]
            )
        
        # Hourly Market Indicators
        if "SPY_RSI_Hourly" in self.selected_indicators:
            features["SPY_RSI_Hourly"] = market_hourly_data.ta.rsi(length=14).iloc[-1]
        
        # Hourly Ticker Indicators
        if "atr" in self.selected_indicators:
            atr_result = hourly_data.ta.atr(length=14)
            if isinstance(atr_result, pd.DataFrame):
                col_name = find_col(atr_result, "ATRr_14")
                if col_name:
                    features["atr"] = atr_result[col_name].iloc[-1]
            elif isinstance(atr_result, pd.Series):
                features["atr"] = atr_result.iloc[-1]
        
        # Hourly Strategies
        for name in self.selected_strategies:
            if name in HOURLY_STRATEGIES:
                params = self.optimized_params.get(name, {})
                signal = HOURLY_STRATEGIES[name](hourly_data.copy(), **params)
                if signal is not None and len(signal) > 0:
                    features[name] = signal.iloc[-1]
        
        # Create DataFrame
        input_df = pd.DataFrame([features])
        
        # Ensure correct feature order
        try:
            input_df = input_df[self.model.get_booster().feature_names]
        except:
            return None
        
        # Fill any NaN with 0
        input_df = input_df.fillna(0)
        
        # Predict
        buy_probability = self.model.predict_proba(input_df)[0][1]
        
        return buy_probability


class V33ForwardTester:
    """Runs V3.3 forward test with REAL AI predictions"""
    
    def __init__(self, ticker, start_date, end_date, initial_capital=100000):
        self.ticker = ticker.upper()
        self.start_date = pd.to_datetime(start_date).tz_localize("America/New_York")
        self.end_date = pd.to_datetime(end_date).tz_localize("America/New_York")
        self.initial_capital = initial_capital
        
        # V3.3 Parameters (from bot_runner.py)
        self.STOP_LOSS_PCT = 0.05
        self.TAKE_PROFIT_PCT = 0.08
        self.MAX_HOLD_DAYS = 7
        self.EXIT_CONFIDENCE_THRESHOLD = 0.50
        self.MIN_ENTRY_CONFIDENCE = 0.35
        
        self.HOURLY_EXIT_CONFIDENT = 0.40
        self.HOURLY_EXIT_URGENT = 0.20
        self.HOURLY_MIN_PROFIT = 0.005
        self.HOURLY_MIN_HOLD_HOURS = 2
        
        self.HOURLY_ENTRY_MIN = 0.50
        self.COOLDOWN_HOURS = 1
        
        # Initialize backtester
        self.backtester = RealisticBacktester(initial_capital=initial_capital)
        
        # Initialize REAL predictors
        print(f"\n🤖 Initializing AI Models for {ticker}...")
        self.daily_predictor = RealDailyPredictor(ticker)
        self.hourly_predictor = RealHourlyPredictor(ticker)
        
        # Tracking
        self.last_trade_time = None
        self.hourly_history = deque(maxlen=5)
        self.peak_pnl_pct = 0
        
        # Stats
        self.total_predictions = 0
        self.predictions_with_signal = 0
        
    def load_data(self):
        """Load all required data"""
        print(f"\n Loading data...")
        
        # Load daily data
        daily_path = os.path.join(config.DATA_DIR, f"{self.ticker}.csv")
        self.daily_data = pd.read_csv(daily_path, index_col=0, parse_dates=True)
        # Handle timezone-aware strings by parsing as UTC first
        self.daily_data.index = pd.to_datetime(self.daily_data.index, utc=True)
        self.daily_data.index = self.daily_data.index.tz_convert("America/New_York")
        
        # Load market data
        market_path = os.path.join(config.DATA_DIR, f"{config.MARKET_TICKER}.csv")
        self.market_data = pd.read_csv(market_path, index_col=0, parse_dates=True)
        # Handle timezone-aware strings
        self.market_data.index = pd.to_datetime(self.market_data.index, utc=True)
        self.market_data.index = self.market_data.index.tz_convert("America/New_York")
        
        # Load hourly data
        hourly_path = os.path.join(config.DATA_DIR, "hourly", f"{self.ticker}.csv")
        self.hourly_data = pd.read_csv(hourly_path, index_col=0, parse_dates=True)
        # Handle timezone-aware strings
        self.hourly_data.index = pd.to_datetime(self.hourly_data.index, utc=True)
        self.hourly_data.index = self.hourly_data.index.tz_convert("America/New_York")
        
        # Load market hourly
        market_hourly_path = os.path.join(config.DATA_DIR, "hourly", f"{config.MARKET_TICKER}.csv")
        self.market_hourly_data = pd.read_csv(market_hourly_path, index_col=0, parse_dates=True)
        # Handle timezone-aware strings
        self.market_hourly_data.index = pd.to_datetime(self.market_hourly_data.index, utc=True)
        self.market_hourly_data.index = self.market_hourly_data.index.tz_convert("America/New_York")
        
        print(f"  Loaded {len(self.daily_data)} daily bars")
        print(f"  Loaded {len(self.hourly_data)} hourly bars")
        
    def check_hourly_momentum(self):
        """Check if hourly is RISING"""
        if len(self.hourly_history) < 2:
            return True
        
        current = self.hourly_history[-1]
        recent_avg = np.mean(list(self.hourly_history)[:-1])
        
        return current > recent_avg
    
    def check_cooldown(self, current_time):
        """Check cooldown period"""
        if self.last_trade_time is None:
            return True
        
        hours_since = (current_time - self.last_trade_time).total_seconds() / 3600
        return hours_since >= self.COOLDOWN_HOURS
    
    def check_entry_conditions(self, date, daily_conf, hourly_conf):
        """Check V3.3 entry conditions"""
        if self.backtester.shares > 0:
            return False, "ALREADY_IN_POSITION"
        
        if not self.check_cooldown(date):
            return False, "COOLDOWN"
        
        if daily_conf is None or daily_conf < self.MIN_ENTRY_CONFIDENCE:
            return False, f"DAILY_CONF_LOW_{daily_conf:.2%}" if daily_conf else "NO_DAILY_PREDICTION"
        
        if hourly_conf is None or hourly_conf < self.HOURLY_ENTRY_MIN:
            return False, f"HOURLY_CONF_LOW_{hourly_conf:.2%}" if hourly_conf else "NO_HOURLY_PREDICTION"
        
        if not self.check_hourly_momentum():
            return False, "HOURLY_NOT_RISING"
        
        return True, "ALL_CONDITIONS_MET"
    
    def check_exit_conditions(self, date, current_price, daily_conf, hourly_conf):
        """Check V3.3 exit conditions"""
        if self.backtester.shares == 0:
            return False, None
        
        position = self.backtester.get_current_position(current_price)
        pnl_pct = position['unrealized_pnl_pct']
        hold_time = date - position['entry_time']
        hold_hours = hold_time.total_seconds() / 3600
        hold_days = hold_time.days
        
        # Update peak
        if pnl_pct > self.peak_pnl_pct:
            self.peak_pnl_pct = pnl_pct
        
        # 1. Stop Loss
        if pnl_pct <= -self.STOP_LOSS_PCT:
            return True, f"STOP_LOSS_{pnl_pct:.2%}"
        
        # 2. Take Profit
        if pnl_pct >= self.TAKE_PROFIT_PCT:
            return True, f"TAKE_PROFIT_{pnl_pct:.2%}"
        
        # 3. Time Stop
        if hold_days >= self.MAX_HOLD_DAYS:
            return True, f"TIME_STOP_{hold_days}d"
        
        # 4. Daily Confidence Drop
        if daily_conf is not None and daily_conf < self.EXIT_CONFIDENCE_THRESHOLD:
            return True, f"DAILY_CONF_DROP_{daily_conf:.2%}"
        
        # 5. Hourly Urgent Exit
        if hourly_conf is not None and hourly_conf < self.HOURLY_EXIT_URGENT:
            return True, f"HOURLY_URGENT_{hourly_conf:.2%}"
        
        # 6. Hourly Profit Protect
        if (hourly_conf is not None and 
            hourly_conf < self.HOURLY_EXIT_CONFIDENT and
            pnl_pct > self.HOURLY_MIN_PROFIT and
            hold_hours >= self.HOURLY_MIN_HOLD_HOURS):
            return True, f"HOURLY_PROFIT_PROTECT_H{hourly_conf:.1%}_P{pnl_pct:+.2%}"
        
        return False, None
    
    def run(self):
        """Run the forward test"""
        print(f"\n{'='*60}")
        print(f"RUNNING V3.3 FORWARD TEST: {self.ticker}")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"{'='*60}\n")
        
        # Filter to test period
        test_data = self.daily_data[
            (self.daily_data.index >= self.start_date) &
            (self.daily_data.index <= self.end_date)
        ]
        
        if len(test_data) == 0:
            print(f"   No data found in test period!")
            print(f"   Available data: {self.daily_data.index[0].date()} to {self.daily_data.index[-1].date()}")
            return None
        
        print(f"Processing {len(test_data)} trading days...\n")
        
        # Simulate day by day
        for date in test_data.index:
            price = test_data.loc[date, 'Close']
            
            # Get REAL AI predictions
            daily_conf = self.daily_predictor.predict(
                self.daily_data, 
                self.market_data, 
                date
            )
            
            # Get hourly prediction (use same day's close time)
            hourly_conf = self.hourly_predictor.predict(
                self.hourly_data,
                self.market_hourly_data,
                self.daily_data,
                date
            )
            
            self.total_predictions += 1
            
            # Track hourly history
            if hourly_conf is not None:
                self.hourly_history.append(hourly_conf)
            
            # Update equity curve
            self.backtester.update_equity(date, price)
            
            # Check exit conditions FIRST
            should_exit, exit_reason = self.check_exit_conditions(
                date, price, daily_conf, hourly_conf
            )
            
            if should_exit:
                self.backtester.execute_sell(date, price, reason=exit_reason)
                self.last_trade_time = date
                self.peak_pnl_pct = 0
                print(f"{date.date()} SELL @ ${price:.2f} - {exit_reason}")
                continue
            
            # Check entry conditions
            can_enter, entry_reason = self.check_entry_conditions(date, daily_conf, hourly_conf)
            
            if can_enter:
                self.predictions_with_signal += 1
                
                metadata = {
                    'daily_conf': daily_conf,
                    'hourly_conf': hourly_conf,
                    'hourly_rising': self.check_hourly_momentum()
                }
                
                self.backtester.execute_buy(date, price, metadata=metadata)
                self.last_trade_time = date
                print(f"📈 {date.date()} BUY @ ${price:.2f} - Daily:{daily_conf:.1%}, Hourly:{hourly_conf:.1%}")
        
        # Close any remaining position
        if self.backtester.shares > 0:
            final_price = test_data.iloc[-1]['Close']
            final_date = test_data.index[-1]
            self.backtester.execute_sell(
                final_date, 
                final_price, 
                reason="END_OF_TEST"
            )
            print(f"{final_date.date()} SELL @ ${final_price:.2f} - END_OF_TEST")
        
        print(f"\n Forward test complete!")
        print(f"   Total predictions: {self.total_predictions}")
        print(f"   Signals above threshold: {self.predictions_with_signal}")
        
    def save_results(self, output_dir=None):
        """Save forward test results"""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            start_str = self.start_date.strftime("%Y%m%d")
            end_str = self.end_date.strftime("%Y%m%d")
            output_dir = f"data/backtest_results/{self.ticker}_V33_ForwardTest_{start_str}_to_{end_str}_{timestamp}"
        
        metadata = {
            'ticker': self.ticker,
            'start_date': self.start_date.strftime("%Y-%m-%d"),
            'end_date': self.end_date.strftime("%Y-%m-%d"),
            'strategy': 'V3.3_Forward_Test',
            'test_type': 'FORWARD_TEST',
            'daily_threshold': self.daily_predictor.threshold,
            'hourly_threshold': self.hourly_predictor.threshold,
            'total_predictions': self.total_predictions,
            'signals_generated': self.predictions_with_signal,
        }
        
        metrics = self.backtester.save_results(output_dir, metadata=metadata)
        # REMOVED: self.backtester.print_summary() to prevent double/interleaved logging
        
        print(f"\n💾 Results saved to: {output_dir}/\n")
        
        return metrics, output_dir


def run_v33_forward_test(ticker, start_date, end_date, initial_capital=100000):
    """
    Main function to run V3.3 forward test with REAL predictions
    
    Args:
        ticker: Stock ticker (e.g., 'NVDA')
        start_date: Start date for forward test (e.g., '2025-01-01')
        end_date: End date for forward test (e.g., '2025-01-24')
        initial_capital: Starting capital
    
    Returns:
        metrics: Performance metrics dict
        output_dir: Path to saved results
    """
    
    print(f"\n{'='*70}")
    print(f"V3.3 FORWARD TEST - Testing on NEW unseen data")
    print(f"{'='*70}\n")
    
    # Ensure we have data for this period
    ensure_data_exists(ticker, start_date, end_date)
    
    # Create forward tester
    tester = V33ForwardTester(ticker, start_date, end_date, initial_capital)
    
    # Load data
    tester.load_data()
    
    # Run forward test
    tester.run()
    
    # Save results
    metrics, output_dir = tester.save_results()
    
    return metrics, output_dir


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python run_v33_backtest.py TICKER [start_date] [end_date] [capital]")
        print("\nExample: python run_v33_backtest.py NVDA 2025-01-01 2025-01-24 100000")
        sys.exit(1)
    
    ticker = sys.argv[1]
    start = sys.argv[2] if len(sys.argv) > 2 else "2025-01-01"
    end = sys.argv[3] if len(sys.argv) > 3 else "2025-01-24"
    capital = float(sys.argv[4]) if len(sys.argv) > 4 else 100000
    
    run_v33_forward_test(ticker, start, end, capital)