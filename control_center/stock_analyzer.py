"""
Stock Analyzer Module
Handles stock data download, horizon optimization, and training
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config
from utils.data_fetcher import fetch_data
from utils.data_fetcher_hourly import fetch_hourly_data
from control_center.utils import (
    print_success, print_error, print_warning, print_info,
    print_section, print_table, get_timestamp
)


class StockAnalyzer:
    """Handles stock analysis and training"""
    
    # Official Dow Jones 30 stocks (as of 2024)
    # Official Dow Jones 30 stocks (as of 2025)
    DOW_30_TICKERS = [
        "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
        "DIS", "SHW", "GS", "HD", "HON", "IBM", "NVDA", "JNJ",
        "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG",
        "TRV", "UNH", "V", "VZ", "WMT", "AMZN"
    ]
    
    def __init__(self):
        self.data_dir = config.DATA_DIR
        self.hourly_dir = os.path.join(config.DATA_DIR, "hourly")
        self.stocks_dir = config.STOCKS_DIR
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.hourly_dir, exist_ok=True)
        os.makedirs(self.stocks_dir, exist_ok=True)
    
    # ==========================================
    # DATA DOWNLOAD
    # ==========================================
    
    def download_all_stocks(self, force=False):
        """Download all Dow 30 stocks (daily + hourly)"""
        print_section("📥 DOWNLOADING ALL DOW 30 STOCKS")
        
        print(f"\n📋 Official Dow 30 Tickers: {len(self.DOW_30_TICKERS)} stocks")
        print(f"   {', '.join(self.DOW_30_TICKERS[:10])}...")
        
        # Check what's already downloaded
        daily_status = self._check_daily_downloads()
        hourly_status = self._check_hourly_downloads()
        
        if force:
            print("\n⚠️  FORCE MODE ACTIVE: Redownloading all stocks...")
            daily_missing = self.DOW_30_TICKERS
            hourly_missing = self.DOW_30_TICKERS
        else:
            daily_missing = [t for t in self.DOW_30_TICKERS if not daily_status.get(t, False)]
            hourly_missing = [t for t in self.DOW_30_TICKERS if not hourly_status.get(t, False)]
        
        print(f"\n📊 Status:")
        print(f"   Daily data: {len(self.DOW_30_TICKERS) - len(daily_missing)}/30 downloaded")
        print(f"   Hourly data: {len(self.DOW_30_TICKERS) - len(hourly_missing)}/30 downloaded")
        
        # Download missing daily data
        if daily_missing:
            print(f"\n⬇️  Downloading {len(daily_missing)} missing daily datasets...")
            for ticker in tqdm(daily_missing, desc="Daily Data"):
                try:
                    fetch_data(ticker, force=force)
                except Exception as e:
                    print_error(f"Failed to download {ticker} daily: {e}")
            print_success(f"Daily data download complete!")
        else:
            print_success("All daily data already downloaded!")
        
        # Download missing hourly data
        if hourly_missing:
            print(f"\n⬇️  Downloading {len(hourly_missing)} missing hourly datasets...")
            for ticker in tqdm(hourly_missing, desc="Hourly Data"):
                try:
                    fetch_hourly_data(ticker, force=force)
                except Exception as e:
                    print_error(f"Failed to download {ticker} hourly: {e}")
            print_success(f"Hourly data download complete!")
        else:
            print_success("All hourly data already downloaded!")
        
        # Final verification
        final_daily = self._check_daily_downloads()
        final_hourly = self._check_hourly_downloads()
        
        daily_complete = sum(final_daily.values())
        hourly_complete = sum(final_hourly.values())
        
        print(f"\n✅ DOWNLOAD SUMMARY:")
        print(f"   Daily: {daily_complete}/30 complete")
        print(f"   Hourly: {hourly_complete}/30 complete")
        
        if daily_complete == 30 and hourly_complete == 30:
            print_success("All Dow 30 stocks downloaded successfully!")
        else:
            print_warning(f"Some downloads failed. Check errors above.")
    
    def _is_data_valid(self, df, data_type="daily"):
        """Check if data covers the configured date range"""
        if df is None or df.empty:
            return False
            
        try:
            # Parse config dates
            req_start = pd.to_datetime(config.START_DATE).replace(tzinfo=None)
            req_end = pd.to_datetime(config.END_DATE).replace(tzinfo=None)
            
            # Get actual dates (remove TZ for comparison)
            if not isinstance(df.index, pd.DatetimeIndex):
                return False
                
            act_start = df.index.min().replace(tzinfo=None)
            act_end = df.index.max().replace(tzinfo=None)
            
            # Check range (allow 60 days buffer at start, 30 at end)
            has_start = act_start <= (req_start + timedelta(days=60))
            has_end = act_end >= (req_end - timedelta(days=30))
            
            return has_start and has_end
        except:
            return False

    def _check_daily_downloads(self):
        """Check which daily datasets are valid and complete"""
        status = {}
        for ticker in self.DOW_30_TICKERS:
            filepath = os.path.join(self.data_dir, f"{ticker}.csv")
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    status[ticker] = self._is_data_valid(df, "daily")
                except:
                    status[ticker] = False
            else:
                status[ticker] = False
        return status
    
    def _check_hourly_downloads(self):
        """Check which hourly datasets are valid and complete"""
        status = {}
        for ticker in self.DOW_30_TICKERS:
            filepath = os.path.join(self.hourly_dir, f"{ticker}.csv")
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    status[ticker] = self._is_data_valid(df, "hourly")
                except:
                    status[ticker] = False
            else:
                status[ticker] = False
        return status
    
    def show_download_status(self):
        """Show detailed download status"""
        daily_status = self._check_daily_downloads()
        hourly_status = self._check_hourly_downloads()
        
        print("\n📊 DOWNLOAD STATUS:")
        
        headers = ["Ticker", "Daily", "Hourly", "Status"]
        rows = []
        
        for ticker in self.DOW_30_TICKERS:
            daily_ok = "✅" if daily_status.get(ticker, False) else "❌"
            hourly_ok = "✅" if hourly_status.get(ticker, False) else "❌"
            
            if daily_status.get(ticker, False) and hourly_status.get(ticker, False):
                status_text = "Ready"
            elif daily_status.get(ticker, False):
                status_text = "Daily only"
            elif hourly_status.get(ticker, False):
                status_text = "Hourly only"
            else:
                status_text = "Missing"
            
            rows.append([ticker, daily_ok, hourly_ok, status_text])
        
        print_table(headers, rows, column_widths=[10, 10, 10, 15])
        
        # Summary
        daily_count = sum(daily_status.values())
        hourly_count = sum(hourly_status.values())
        complete = sum(1 for t in self.DOW_30_TICKERS 
                      if daily_status.get(t, False) and hourly_status.get(t, False))
        
        print(f"\n📈 Summary:")
        print(f"   Daily data: {daily_count}/30")
        print(f"   Hourly data: {hourly_count}/30")
        print(f"   Complete: {complete}/30")
    
    # ==========================================
    # STRATEGY PARAMETER OPTIMIZATION
    # ==========================================
    
    def optimize_all_strategy_parameters(self):
        """Optimize strategy parameters for all stocks"""
        print_section("⚙️  OPTIMIZING STRATEGY PARAMETERS")
        
        print("\n📋 This will:")
        print("   • Optimize daily strategy parameters")
        print("   • Optimize hourly strategy parameters")
        print("   • Test different indicator configurations")
        print("   • Save results to daily_params.json and hourly_params.json")
        
        # Import optimization functions
        from testing.optimizer_daily import optimize_daily_strategies
        from testing.optimizer_hourly import optimize_hourly_strategies
        
        print(f"\n📊 Optimizing all {len(self.DOW_30_TICKERS)} Dow Jones stocks")
        print(f"⏱️  Estimated time: 2-4 hours")
        
        print("\n" + "="*70)
        
        success_count = 0
        failed_stocks = []
        
        for i, ticker in enumerate(self.DOW_30_TICKERS, 1):
            print(f"\n[{i}/{len(self.DOW_30_TICKERS)}] Optimizing {ticker}...")
            print("-" * 70)
            
            try:
                # Optimize daily strategies
                print(f"🌅 Optimizing Daily Strategies for {ticker}...")
                optimize_daily_strategies(ticker, mode="local")
                
                # Optimize hourly strategies
                print(f"\n⏰ Optimizing Hourly Strategies for {ticker}...")
                optimize_hourly_strategies(ticker, mode="local")
                
                success_count += 1
                print_success(f"Strategy parameters optimized for {ticker}")
                
            except Exception as e:
                print_error(f"Failed to optimize {ticker}: {e}")
                failed_stocks.append(ticker)
                continue
        
        # Summary
        print("\n" + "="*70)
        print_section("📊 STRATEGY OPTIMIZATION SUMMARY")
        
        print(f"\n✅ Successfully optimized: {success_count}/{len(self.DOW_30_TICKERS)} stocks")
        
        if failed_stocks:
            print(f"\n❌ Failed stocks: {', '.join(failed_stocks)}")
        
        print("\n📁 Parameters saved to: data/historical/stocks/[TICKER]/daily|hourly/[model]_params.json")
        print("="*70)
    
    # ==========================================
    # HORIZON SETUP
    # ==========================================
    
    def apply_default_horizons(self):
        """Apply sensible default horizons to all stocks (Quick Setup)"""
        print_section("🚀 APPLYING SENSIBLE DEFAULT HORIZONS")
        
        print("\n📊 Default Configuration:")
        print("   Daily AI:")
        print("     • Horizon: 5 days")
        print("     • Threshold: 2.5%")
        print("     • Rationale: Classic swing trading timeframe")
        
        print("\n   Hourly AI:")
        print("     • Horizon: 8 hours")
        print("     • Threshold: 1.5%")
        print("     • Rationale: Intraday momentum trades")
        
        # Default configurations
        default_daily_config = {
            "look_forward": 5,
            "threshold": 0.025,
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "optimization_range": {
                "start": 0.50,
                "end": 0.85,
                "step": 0.01
            },
            "indicators": [
                "RSI", "MACD", "ATR", "ADX", "Stochastic",
                "OBV", "CMF", "VWAP",
                "SPY_RSI", "SPY_50MA_Ratio", "SPY_200MA_Ratio"
            ],
            "strategies": [
                "MACD", "RSI", "BollingerBands", "EMACross", "Stochastic",
                "VWAP", "OBV", "Donchian", "KeltnerChannels", "ROC",
                "CMF", "Ichimoku", "AwesomeOscillator", "ADX",
                "CoppockCurve", "CMO", "TRIX"
            ]
        }
        
        default_hourly_config = {
            "look_forward": 8,
            "threshold": 0.015,
            "start_date": "2020-01-01",
            "end_date": "2024-12-31",
            "optimization_range": {
                "start": 0.60,
                "end": 0.95,
                "step": 0.01
            },
            "indicators": [
                "atr", "Daily_RSI", "Daily_50MA_Ratio", "Daily_200MA_Ratio",
                "SPY_RSI_Hourly"
            ],
            "strategies": [
                "HourlyEMACross", "HourlyRSI", "VolumeAnomaly", "HourlyBollinger",
                "HourlyStochastic", "HourlyMACD", "HourlyCMF", "HourlyROC",
                "HourlyADX", "HourlyPSAR", "HourlyTEMA", "HourlyLongEMACross",
                "HeikinAshi", "RVI", "FisherTransform"
            ]
        }
        
        # Apply to all Dow 30 stocks
        print(f"\n⚙️  Applying to all {len(self.DOW_30_TICKERS)} Dow Jones stocks...")
        
        success_count = 0
        for ticker in tqdm(self.DOW_30_TICKERS, desc="Configuring stocks"):
            try:
                # Create directories
                daily_dir = os.path.join(self.stocks_dir, ticker, "daily")
                hourly_dir = os.path.join(self.stocks_dir, ticker, "hourly")
                os.makedirs(daily_dir, exist_ok=True)
                os.makedirs(hourly_dir, exist_ok=True)
                
                # Save daily config
                daily_config = default_daily_config.copy()
                daily_config["ticker"] = ticker
                daily_config_path = os.path.join(daily_dir, "training_config.json")
                with open(daily_config_path, 'w') as f:
                    json.dump(daily_config, f, indent=4)
                
                # Save hourly config
                hourly_config = default_hourly_config.copy()
                hourly_config["ticker"] = ticker
                hourly_config_path = os.path.join(hourly_dir, "training_config.json")
                with open(hourly_config_path, 'w') as f:
                    json.dump(hourly_config, f, indent=4)
                
                success_count += 1
                
            except Exception as e:
                print_error(f"Failed to configure {ticker}: {e}")
                continue
        
        print(f"\n✅ Successfully configured {success_count}/{len(self.DOW_30_TICKERS)} stocks!")
        print("\n📁 Configs saved to: data/historical/stocks/[TICKER]/daily|hourly/training_config.json")
        print("\n🎯 Ready for training! Proceed to 'Train All AIs' next.")
    
    def find_optimal_horizons_all(self):
        """Find optimal prediction horizons for all stocks"""
        print_section("🔬 FINDING OPTIMAL PREDICTION HORIZONS")
        
        print("\n⚙️  Configuration:")
        print("   Daily Horizons: [3, 5, 7, 10, 14, 21, 30] days")
        print("   Daily Thresholds: [1.0%, 1.5%, 2.0%, 2.5%, 3.0%, 4.0%, 5.0%, 6.0%]")
        print("   Hourly Horizons: [2, 4, 6, 8, 12, 16, 24, 36, 48] hours")
        print("   Hourly Thresholds: [0.5%, 0.75%, 1.0%, 1.5%, 2.0%, 2.5%, 3.0%, 4.0%]")
        print("   Cross-validation: 5-fold TimeSeriesSplit")
        print("   Models per combo: 100 trees, depth 5")
        print("   Parallel processing: 4 cores per model")
        
        # Import horizon optimizer
        from control_center.horizon_optimizer import HorizonOptimizer
        optimizer = HorizonOptimizer()
        
        # Check which stocks have strategy params optimized
        stocks_to_optimize = []
        for ticker in self.DOW_30_TICKERS:
            daily_params_path = os.path.join(self.stocks_dir, ticker, "daily", "daily_params.json")
            hourly_params_path = os.path.join(self.stocks_dir, ticker, "hourly", "hourly_params.json")
            
            if os.path.exists(daily_params_path) and os.path.exists(hourly_params_path):
                stocks_to_optimize.append(ticker)
        
        if not stocks_to_optimize:
            print_warning("\n⚠️  No stocks have optimized strategy parameters!")
            print_info("You need to run strategy optimization first:")
            print("   python main.py optimize_daily --ticker TICKER --mode local")
            print("   python main.py optimize_hourly --ticker TICKER --mode local")
            print("\nOr skip this and use default parameters (not recommended)")
            
            use_defaults = input("\nUse default parameters? (y/n): ").lower()
            if use_defaults != 'y':
                return
            
            # Use all stocks with empty params
            stocks_to_optimize = self.DOW_30_TICKERS
        
        print(f"\n📊 Will optimize horizons for {len(stocks_to_optimize)} stocks")
        print(f"⏱️  Estimated time: {len(stocks_to_optimize) * 10:.0f}-{len(stocks_to_optimize) * 20:.0f} minutes")
        
        confirm = input("\nProceed? (y/n): ").lower()
        if confirm != 'y':
            print_warning("Optimization cancelled")
            return
        
        # Optimize each stock
        print("\n" + "="*70)
        results_summary = []
        
        for i, ticker in enumerate(stocks_to_optimize, 1):
            print(f"\n[{i}/{len(stocks_to_optimize)}] Processing {ticker}...")
            print("-" * 70)
            
            # Load optimized params (or use empty dict)
            daily_params_path = os.path.join(self.stocks_dir, ticker, "daily", "daily_params.json")
            hourly_params_path = os.path.join(self.stocks_dir, ticker, "hourly", "hourly_params.json")
            
            daily_params = {}
            hourly_params = {}
            
            if os.path.exists(daily_params_path):
                with open(daily_params_path, 'r') as f:
                    daily_params = json.load(f)
            
            if os.path.exists(hourly_params_path):
                with open(hourly_params_path, 'r') as f:
                    hourly_params = json.load(f)
            
            # Optimize daily horizon
            daily_result = optimizer.optimize_daily_horizon(ticker, daily_params)
            
            # Optimize hourly horizon
            hourly_result = optimizer.optimize_hourly_horizon(ticker, hourly_params)
            
            # Save results
            if daily_result or hourly_result:
                optimizer.save_horizon_config(ticker, daily_result, hourly_result)
                
                results_summary.append({
                    'ticker': ticker,
                    'daily_horizon': daily_result['horizon'] if daily_result else None,
                    'daily_threshold': daily_result['threshold'] if daily_result else None,
                    'daily_cv_score': daily_result['cv_score'] if daily_result else None,
                    'hourly_horizon': hourly_result['horizon'] if hourly_result else None,
                    'hourly_threshold': hourly_result['threshold'] if hourly_result else None,
                    'hourly_cv_score': hourly_result['cv_score'] if hourly_result else None
                })
        
        # Print summary
        print("\n" + "="*70)
        print_section("📊 HORIZON OPTIMIZATION SUMMARY")
        
        if results_summary:
            print("\n✅ Successfully optimized horizons:\n")
            
            headers = ["Ticker", "Daily H.", "Daily Th.", "Daily CV", "Hourly H.", "Hourly Th.", "Hourly CV"]
            rows = []
            
            for result in results_summary:
                rows.append([
                    result['ticker'],
                    f"{result['daily_horizon']}d" if result['daily_horizon'] else "N/A",
                    f"{result['daily_threshold']*100:.1f}%" if result['daily_threshold'] else "N/A",
                    f"{result['daily_cv_score']:.3f}" if result['daily_cv_score'] else "N/A",
                    f"{result['hourly_horizon']}h" if result['hourly_horizon'] else "N/A",
                    f"{result['hourly_threshold']*100:.1f}%" if result['hourly_threshold'] else "N/A",
                    f"{result['hourly_cv_score']:.3f}" if result['hourly_cv_score'] else "N/A"
                ])
            
            print_table(headers, rows)
            
            print(f"\n✅ Optimization complete for {len(results_summary)} stocks!")
            print("📁 Configs saved to: data/historical/stocks/[TICKER]/daily|hourly/training_config.json")
        else:
            print_warning("No successful optimizations")
        
        print("\n" + "="*70)
    
    # ==========================================
    # TRAINING
    # ==========================================
    
    def train_all_stocks(self):
        """Train all AI models"""
        print_section("🎓 TRAINING ALL AI MODELS")
        
        print("\n📋 This will:")
        print("   • Train Daily AI models for all stocks")
        print("   • Train Hourly AI models for all stocks")
        print("   • Use optimal horizons from training_config.json")
        print("   • Full hyperparameter tuning with GridSearchCV")
        print("   • Save models to data/historical/stocks/[TICKER]/")
        
        # Check which stocks have training configs
        stocks_ready = []
        for ticker in self.DOW_30_TICKERS:
            daily_config = os.path.join(self.stocks_dir, ticker, "daily", "training_config.json")
            hourly_config = os.path.join(self.stocks_dir, ticker, "hourly", "training_config.json")
            
            if os.path.exists(daily_config) and os.path.exists(hourly_config):
                stocks_ready.append(ticker)
        
        if not stocks_ready:
            print_warning("\n⚠️  No stocks have training configs!")
            print_info("Run 'Find Optimal Horizons' first, or create configs with make_config.py")
            return
        
        print(f"\n📊 Found {len(stocks_ready)} stocks ready for training")
        print(f"⏱️  Estimated time: {len(stocks_ready) * 2:.0f}-{len(stocks_ready) * 4:.0f} minutes")
        
        confirm = input("\nProceed? (y/n): ").lower()
        if confirm != 'y':
            print_warning("Training cancelled")
            return
        
        # Import training functions
        from models.daily.trainer import train_daily_ai_model
        from models.hourly.trainer import train_hourly_ai_model
        
        print("\n" + "="*70)
        
        success_count = 0
        failed_stocks = []
        
        for i, ticker in enumerate(stocks_ready, 1):
            print(f"\n[{i}/{len(stocks_ready)}] Training {ticker}...")
            print("-" * 70)
            
            try:
                # Train daily model
                print(f"🌅 Training Daily AI for {ticker}...")
                train_daily_ai_model(ticker)
                print_success(f"Daily AI trained for {ticker}")
                
                # Train hourly model
                print(f"\n⏰ Training Hourly AI for {ticker}...")
                train_hourly_ai_model(ticker)
                print_success(f"Hourly AI trained for {ticker}")
                
                success_count += 1
                
            except Exception as e:
                print_error(f"Failed to train {ticker}: {e}")
                failed_stocks.append(ticker)
                continue
        
        # Summary
        print("\n" + "="*70)
        print_section("📊 TRAINING SUMMARY")
        
        print(f"\n✅ Successfully trained: {success_count}/{len(stocks_ready)} stocks")
        
        if failed_stocks:
            print(f"\n❌ Failed stocks: {', '.join(failed_stocks)}")
        
        print("\n📁 Models saved to: data/historical/stocks/[TICKER]/daily|hourly/")
        print("="*70)
    
    # ==========================================
    # THRESHOLD OPTIMIZATION
    # ==========================================
    
    def optimize_all_thresholds(self):
        """Optimize confidence thresholds"""
        print_section("🎯 OPTIMIZING CONFIDENCE THRESHOLDS")
        
        print("\n📋 This will:")
        print("   • Test confidence thresholds on 2025 data")
        print("   • Daily: 0.50 to 0.85 (step 0.01)")
        print("   • Hourly: 0.60 to 0.95 (step 0.01)")
        print("   • Maximize Sharpe Ratio")
        print("   • Save optimal thresholds to settings.json")
        
        # Check which stocks have trained models
        stocks_trained = []
        for ticker in self.DOW_30_TICKERS:
            daily_model = os.path.join(self.stocks_dir, ticker, "daily", "daily_model.joblib")
            hourly_model = os.path.join(self.stocks_dir, ticker, "hourly", "hourly_model.joblib")
            
            if os.path.exists(daily_model) and os.path.exists(hourly_model):
                stocks_trained.append(ticker)
        
        if not stocks_trained:
            print_warning("\n⚠️  No trained models found!")
            print_info("Run 'Train All AIs' first")
            return
        
        print(f"\n📊 Found {len(stocks_trained)} stocks with trained models")
        print(f"⏱️  Estimated time: {len(stocks_trained) * 1:.0f}-{len(stocks_trained) * 2:.0f} minutes")
        
        confirm = input("\nProceed? (y/n): ").lower()
        if confirm != 'y':
            print_warning("Threshold optimization cancelled")
            return
        
        # Import threshold optimizers
        from testing.threshold_optimizer import run_threshold_optimizer
        from testing.threshold_optimizer_hourly import run_hourly_threshold_optimizer
        
        print("\n" + "="*70)
        
        success_count = 0
        failed_stocks = []
        
        for i, ticker in enumerate(stocks_trained, 1):
            print(f"\n[{i}/{len(stocks_trained)}] Optimizing {ticker}...")
            print("-" * 70)
            
            try:
                # Optimize daily threshold
                print(f"🌅 Optimizing Daily Threshold for {ticker}...")
                run_threshold_optimizer(ticker)
                
                # Optimize hourly threshold
                print(f"\n⏰ Optimizing Hourly Threshold for {ticker}...")
                run_hourly_threshold_optimizer(ticker)
                
                success_count += 1
                print_success(f"Thresholds optimized for {ticker}")
                
            except Exception as e:
                print_error(f"Failed to optimize {ticker}: {e}")
                failed_stocks.append(ticker)
                continue
        
        # Summary
        print("\n" + "="*70)
        print_section("📊 THRESHOLD OPTIMIZATION SUMMARY")
        
        print(f"\n✅ Successfully optimized: {success_count}/{len(stocks_trained)} stocks")
        
        if failed_stocks:
            print(f"\n❌ Failed stocks: {', '.join(failed_stocks)}")
        
        print("\n📁 Thresholds saved to: data/historical/stocks/[TICKER]/daily|hourly/[model]_settings.json")
        print("="*70)
    
    # ==========================================
    # BACKTESTING
    # ==========================================
    
    def backtest_all_stocks(self):
        """Run V3.3 backtests on all stocks"""
        print_section("📈 BACKTESTING ALL STOCKS")
        
        print("\n📋 This will:")
        print("   • Run V3.3 strategy backtest for each stock")
        print("   • Training period: 2020-2024")
        print("   • Validation period: 2025+")
        print("   • Calculate returns, Sharpe Ratio, max drawdown, win rate")
        print("   • Save results to master_config.json")
        
        # Check which stocks are fully ready
        stocks_ready = []
        for ticker in self.DOW_30_TICKERS:
            daily_model = os.path.join(self.stocks_dir, ticker, "daily", "daily_model.joblib")
            hourly_model = os.path.join(self.stocks_dir, ticker, "hourly", "hourly_model.joblib")
            daily_settings = os.path.join(self.stocks_dir, ticker, "daily", "daily_settings.json")
            hourly_settings = os.path.join(self.stocks_dir, ticker, "hourly", "hourly_settings.json")
            
            if all([
                os.path.exists(daily_model),
                os.path.exists(hourly_model),
                os.path.exists(daily_settings),
                os.path.exists(hourly_settings)
            ]):
                stocks_ready.append(ticker)
        
        if not stocks_ready:
            print_warning("\n⚠️  No stocks ready for backtesting!")
            print_info("Complete the full pipeline first:")
            print("   1. Download data")
            print("   2. Find optimal horizons")
            print("   3. Train models")
            print("   4. Optimize thresholds")
            return
        
        print(f"\n📊 Found {len(stocks_ready)} stocks ready for backtesting")
        print(f"⏱️  Estimated time: {len(stocks_ready) * 0.5:.0f}-{len(stocks_ready) * 1:.0f} minutes")
        
        confirm = input("\nProceed? (y/n): ").lower()
        if confirm != 'y':
            print_warning("Backtesting cancelled")
            return
        
        # Import backtest function
        from testing.run_v33_backtest import run_v33_forward_test
        
        # Load master config
        from control_center.model_manager import ModelManager
        manager = ModelManager()
        
        print("\n" + "="*70)
        
        success_count = 0
        backtest_results = []
        
        for i, ticker in enumerate(stocks_ready, 1):
            print(f"\n[{i}/{len(stocks_ready)}] Backtesting {ticker}...")
            print("-" * 70)
            
            try:
                # Run backtest on validation period (2025+)
                # Note: run_v33_forward_test returns (metrics, output_dir)
                result = run_v33_forward_test(
                    ticker=ticker,
                    start_date="2025-01-01",
                    end_date="2025-12-31",
                    initial_capital=100000
                )
                
                # Unpack tuple if returned
                if isinstance(result, tuple):
                    metrics, output_dir = result
                else:
                    metrics = result
                
                if metrics:
                    # Extract key metrics
                    total_return = metrics.get('total_return_pct', metrics.get('total_return', 0))
                    sharpe_ratio = metrics.get('sharpe_ratio', 0)
                    max_drawdown = metrics.get('max_drawdown_pct', metrics.get('max_drawdown', 0))
                    win_rate = metrics.get('win_rate', 0)
                    num_trades = metrics.get('total_trades', metrics.get('num_trades', 0))
                    
                    print(f"   📊 Results: Return={total_return:.2f}%, Sharpe={sharpe_ratio:.2f}, "
                          f"MDD={max_drawdown:.2f}%, WR={win_rate:.1%}")
                    
                    # Save to master config
                    manager.update_stock_info(ticker, {
                        'trained': True,
                        'backtested': True,
                        'backtest_date': get_timestamp(),
                        'backtest_results': {
                            'return': total_return,
                            'sharpe': sharpe_ratio,
                            'max_drawdown': max_drawdown,
                            'win_rate': win_rate,
                            'num_trades': num_trades
                        }
                    })
                    
                    backtest_results.append({
                        'ticker': ticker,
                        'return': total_return,
                        'sharpe': sharpe_ratio,
                        'max_drawdown': max_drawdown,
                        'win_rate': win_rate
                    })
                    
                    success_count += 1
                else:
                    print_warning(f"No metrics returned for {ticker}")
                
            except Exception as e:
                print_error(f"Failed to backtest {ticker}: {e}")
                continue
        
        # Summary
        print("\n" + "="*70)
        print_section("📊 BACKTESTING SUMMARY")
        
        if backtest_results:
            print(f"\n✅ Successfully backtested: {success_count}/{len(stocks_ready)} stocks\n")
            
            # Sort by Sharpe Ratio
            backtest_results.sort(key=lambda x: x['sharpe'], reverse=True)
            
            headers = ["Rank", "Ticker", "Sharpe", "Return %", "Max DD %", "Win Rate"]
            rows = []
            
            for rank, result in enumerate(backtest_results, 1):
                rows.append([
                    rank,
                    result['ticker'],
                    f"{result['sharpe']:.2f}",
                    f"{result['return']:.2f}",
                    f"{result['max_drawdown']:.2f}",
                    f"{result['win_rate']:.1%}"
                ])
            
            print_table(headers, rows)
            
            if len(backtest_results) >= 10:
                top_10 = [r['ticker'] for r in backtest_results[:10]]
                print(f"\n⭐ TOP 10 STOCKS: {', '.join(top_10)}")
        else:
            print_warning("No successful backtests")
        
        print("\n📁 Results saved to: control_center/master_config.json")
        print("="*70)