"""
Horizon Optimizer
Finds optimal prediction horizons for each stock using cross-validation
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import warnings
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config
from testing.strategies import TOP_STRATEGIES, find_col
import pandas_ta as ta

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


class HorizonOptimizer:
    """Optimizes prediction horizons using cross-validation"""
    
    # EXPANDED horizons to test (in days for daily, hours for hourly)
    DAILY_HORIZONS = [3, 5, 7, 10, 14, 21, 30]  # Added 30 days
    DAILY_THRESHOLDS = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]  # Added 5% and 6%
    
    # MUCH WIDER hourly range - the collapse to 24h/1% suggests we need to go further
    HOURLY_HORIZONS = [2, 4, 6, 8, 12, 16, 24, 36, 48]  # Added 2h, 36h, 48h (2 days)
    HOURLY_THRESHOLDS = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04]  # Added 0.5%, 0.75%
    
    def __init__(self):
        self.data_dir = config.DATA_DIR
        self.hourly_dir = os.path.join(config.DATA_DIR, "hourly")
        self.stocks_dir = config.STOCKS_DIR
    
    # ==========================================
    # DAILY HORIZON OPTIMIZATION
    # ==========================================
    
    def optimize_daily_horizon(self, ticker, optimized_params):
        """Find optimal daily prediction horizon for a stock"""
        print(f"\n🔬 Optimizing Daily Horizon for {ticker}...")
        
        # Load data
        filepath = os.path.join(self.data_dir, f"{ticker}.csv")
        market_filepath = os.path.join(self.data_dir, f"{config.MARKET_TICKER}.csv")
        
        if not os.path.exists(filepath):
            print(f"❌ No data for {ticker}")
            return None
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        market_df = pd.read_csv(market_filepath, index_col=0, parse_dates=True)
        
        # Ensure timezone-aware DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
        elif df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        
        if not isinstance(market_df.index, pd.DatetimeIndex):
            market_df.index = pd.to_datetime(market_df.index, utc=True).tz_convert("America/New_York")
        elif market_df.index.tz is None:
            market_df.index = market_df.index.tz_localize("America/New_York")
        
        # Filter to training period
        start_date = pd.Timestamp("2020-01-01").tz_localize("America/New_York")
        end_date = pd.Timestamp("2024-12-31").tz_localize("America/New_York")
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if len(df) < 500:
            print(f"⚠️  Insufficient data for {ticker} ({len(df)} days)")
            return None
        
        # Generate all combinations
        combinations = list(product(self.DAILY_HORIZONS, self.DAILY_THRESHOLDS))
        
        print(f"   Testing {len(combinations)} combinations with 5-fold CV...")
        
        results = []
        
        for horizon, threshold in tqdm(combinations, desc=f"  {ticker} Daily CV", leave=False):
            try:
                # Generate features and labels
                feature_df, target = self._generate_daily_features(
                    df.copy(), 
                    market_df.copy(), 
                    optimized_params,
                    horizon,
                    threshold
                )
                
                if feature_df is None or target is None:
                    continue
                
                # Cross-validation
                cv_scores = self._cross_validate_model(feature_df, target, n_splits=5)
                
                if cv_scores:
                    avg_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)
                    
                    results.append({
                        'horizon': horizon,
                        'threshold': threshold,
                        'cv_score': avg_score,
                        'cv_std': std_score,
                        'cv_scores': cv_scores
                    })
            
            except Exception as e:
                # Skip failed combinations
                continue
        
        if not results:
            print(f"❌ No valid results for {ticker}")
            return None
        
        # Sort by CV score (descending)
        results.sort(key=lambda x: x['cv_score'], reverse=True)
        
        best = results[0]
        
        print(f"   ✅ Best: {best['horizon']} days @ {best['threshold']*100:.1f}% "
              f"(CV Score: {best['cv_score']:.3f} ± {best['cv_std']:.3f})")
        
        return {
            'horizon': best['horizon'],
            'threshold': best['threshold'],
            'cv_score': best['cv_score'],
            'cv_std': best['cv_std'],
            'all_results': results[:5]  # Save top 5
        }
    
    def _generate_daily_features(self, df, market_df, optimized_params, look_forward, threshold):
        """Generate features for daily prediction"""
        try:
            # Create target variable
            future_price = df['Close'].shift(-look_forward)
            target = ((future_price / df['Close'] - 1) > threshold).astype(int)
            
            # Generate features
            feature_df = pd.DataFrame(index=df.index)
            
            # Technical indicators
            feature_df['RSI_14'] = df.ta.rsi(length=14)
            
            atr_result = df.ta.atr(length=14)
            if isinstance(atr_result, pd.DataFrame):
                col_name = find_col(atr_result, "ATRr_14")
                if col_name:
                    feature_df['ATR_14'] = atr_result[col_name]
            else:
                feature_df['ATR_14'] = atr_result
            
            adx_df = df.ta.adx(length=14)
            if adx_df is not None:
                feature_df['ADX_14'] = adx_df[find_col(adx_df, "ADX_14")]
            
            feature_df['CMF_20'] = df.ta.cmf(length=20)
            
            macd_df = df.ta.macd(fast=12, slow=26, signal=9)
            if macd_df is not None:
                feature_df['MACD_12_26_9'] = macd_df[find_col(macd_df, "MACD_")]
                feature_df['MACDh_12_26_9'] = macd_df[find_col(macd_df, "MACDh_")]
            
            stoch_df = df.ta.stoch(k=14, d=3)
            if stoch_df is not None:
                feature_df['STOCHk_14_3_3'] = stoch_df[find_col(stoch_df, "STOCHk_")]
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                feature_df['VWAP'] = df.ta.vwap()
            
            feature_df['OBV'] = df.ta.obv()
            
            # Market features
            market_features = pd.DataFrame(index=market_df.index)
            market_features['SPY_RSI'] = market_df.ta.rsi(length=14)
            market_features['SPY_50MA_Ratio'] = market_df['Close'] / market_df.ta.ema(length=50)
            market_features['SPY_200MA_Ratio'] = market_df['Close'] / market_df.ta.ema(length=200)
            
            # Strategy signals (use optimized params)
            for name in ['MACD', 'RSI', 'BollingerBands', 'EMACross', 'Stochastic']:
                if name in TOP_STRATEGIES:
                    params = optimized_params.get(name, {})
                    feature_df[name] = TOP_STRATEGIES[name](df.copy(), **params)
            
            # Combine features
            combined = feature_df.join(market_features.dropna())
            combined = combined.join(target.rename('target')).dropna()
            
            if len(combined) < 100 or combined['target'].nunique() < 2:
                return None, None
            
            X = combined.drop(columns=['target'])
            y = combined['target']
            
            return X, y
        
        except Exception as e:
            return None, None
    
    # ==========================================
    # HOURLY HORIZON OPTIMIZATION
    # ==========================================
    
    def optimize_hourly_horizon(self, ticker, optimized_params):
        """Find optimal hourly prediction horizon for a stock"""
        print(f"\n🔬 Optimizing Hourly Horizon for {ticker}...")
        
        # Load data
        hourly_filepath = os.path.join(self.hourly_dir, f"{ticker}.csv")
        daily_filepath = os.path.join(self.data_dir, f"{ticker}.csv")
        market_hourly_filepath = os.path.join(self.hourly_dir, f"{config.MARKET_TICKER}.csv")
        
        if not os.path.exists(hourly_filepath):
            print(f"❌ No hourly data for {ticker}")
            return None
        
        df_hourly = pd.read_csv(hourly_filepath, index_col=0, parse_dates=True)
        df_daily = pd.read_csv(daily_filepath, index_col=0, parse_dates=True)
        df_market_hourly = pd.read_csv(market_hourly_filepath, index_col=0, parse_dates=True)
        
        # Ensure timezone-aware DatetimeIndex for all dataframes
        for df in [df_hourly, df_daily, df_market_hourly]:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
            elif df.index.tz is None:
                df.index = df.index.tz_localize("America/New_York")
        
        # Filter to training period
        start_date = pd.Timestamp("2020-01-01").tz_localize("America/New_York")
        end_date = pd.Timestamp("2024-12-31").tz_localize("America/New_York")
        df_hourly = df_hourly[(df_hourly.index >= start_date) & (df_hourly.index <= end_date)]
        
        if len(df_hourly) < 1000:
            print(f"⚠️  Insufficient hourly data for {ticker} ({len(df_hourly)} hours)")
            return None
        
        # Generate all combinations
        combinations = list(product(self.HOURLY_HORIZONS, self.HOURLY_THRESHOLDS))
        
        print(f"   Testing {len(combinations)} combinations with 5-fold CV...")
        
        results = []
        
        for horizon, threshold in tqdm(combinations, desc=f"  {ticker} Hourly CV", leave=False):
            try:
                # Generate features and labels
                feature_df, target = self._generate_hourly_features(
                    df_hourly.copy(),
                    df_daily.copy(),
                    df_market_hourly.copy(),
                    optimized_params,
                    horizon,
                    threshold
                )
                
                if feature_df is None or target is None:
                    continue
                
                # Cross-validation
                cv_scores = self._cross_validate_model(feature_df, target, n_splits=5)
                
                if cv_scores:
                    avg_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)
                    
                    results.append({
                        'horizon': horizon,
                        'threshold': threshold,
                        'cv_score': avg_score,
                        'cv_std': std_score,
                        'cv_scores': cv_scores
                    })
            
            except Exception as e:
                continue
        
        if not results:
            print(f"❌ No valid results for {ticker}")
            return None
        
        # Sort by CV score
        results.sort(key=lambda x: x['cv_score'], reverse=True)
        
        best = results[0]
        
        print(f"   ✅ Best: {best['horizon']} hours @ {best['threshold']*100:.1f}% "
              f"(CV Score: {best['cv_score']:.3f} ± {best['cv_std']:.3f})")
        
        return {
            'horizon': best['horizon'],
            'threshold': best['threshold'],
            'cv_score': best['cv_score'],
            'cv_std': best['cv_std'],
            'all_results': results[:5]
        }
    
    def _generate_hourly_features(self, df_hourly, df_daily, df_market_hourly, 
                                   optimized_params, look_forward, threshold):
        """Generate features for hourly prediction"""
        try:
            # Create target
            future_price = df_hourly['Close'].shift(-look_forward)
            target = ((future_price / df_hourly['Close'] - 1) > threshold).astype(int)
            
            # Generate features
            feature_df = pd.DataFrame(index=df_hourly.index)
            
            # Hourly indicators
            atr_result = df_hourly.ta.atr(length=14)
            if isinstance(atr_result, pd.DataFrame):
                col_name = find_col(atr_result, "ATRr_14")
                if col_name:
                    feature_df['atr'] = atr_result[col_name]
            else:
                feature_df['atr'] = atr_result
            
            # Market features
            feature_df['SPY_RSI_Hourly'] = df_market_hourly.ta.rsi(length=14)
            
            # Daily context features
            daily_features = pd.DataFrame(index=df_daily.index)
            daily_features['Daily_RSI'] = df_daily.ta.rsi(length=14)
            daily_features['Daily_50MA_Ratio'] = df_daily['Close'] / df_daily.ta.ema(length=50)
            daily_features['Daily_200MA_Ratio'] = df_daily['Close'] / df_daily.ta.ema(length=200)
            
            # Strategy signals (hourly)
            from testing.strategies_hourly import HOURLY_STRATEGIES
            for name in ['HourlyEMACross', 'HourlyRSI', 'VolumeAnomaly']:
                if name in HOURLY_STRATEGIES:
                    params = optimized_params.get(name, {})
                    feature_df[name] = HOURLY_STRATEGIES[name](df_hourly.copy(), **params)
            
            # Merge daily features (forward fill)
            feature_df = pd.merge_asof(
                feature_df.sort_index(),
                daily_features.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'
            )
            
            # Combine with target
            feature_df = feature_df.join(target.rename('target')).dropna()
            
            if len(feature_df) < 200 or feature_df['target'].nunique() < 2:
                return None, None
            
            X = feature_df.drop(columns=['target'])
            y = feature_df['target']
            
            return X, y
        
        except Exception as e:
            return None, None
    
    # ==========================================
    # CROSS-VALIDATION
    # ==========================================
    
    def _cross_validate_model(self, X, y, n_splits=5):
        """Perform time series cross-validation"""
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Check class balance
                if y_train.nunique() < 2 or y_test.nunique() < 2:
                    continue
                
                # Train more robust model (not just quick)
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                
                model = XGBClassifier(
                    n_estimators=100,  # Increased from 50
                    max_depth=5,       # Increased from 3
                    learning_rate=0.05,  # Slower learning for better generalization
                    random_state=42,
                    n_jobs=4,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss',
                    subsample=0.8,     # Add regularization
                    colsample_bytree=0.8
                )
                
                model.fit(X_train, y_train, verbose=False)
                
                # Calculate precision (hit rate) and combine with confidence
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate metrics
                buy_signals = y_pred == 1
                if buy_signals.sum() > 0:
                    hit_rate = y_test[buy_signals].mean()
                    avg_confidence = y_pred_proba[buy_signals].mean()
                    
                    # Score = hit_rate * avg_confidence (rewards both accuracy and confidence)
                    score = hit_rate * avg_confidence
                    scores.append(score)
                else:
                    # No buy signals generated - poor model
                    scores.append(0.0)
            
            return scores if len(scores) >= 3 else None
        
        except Exception as e:
            return None
    
    # ==========================================
    # SAVE RESULTS
    # ==========================================
    
    def save_horizon_config(self, ticker, daily_result, hourly_result):
        """Save optimal horizon configuration"""
        if daily_result is None and hourly_result is None:
            print(f"⚠️  No results to save for {ticker}")
            return
        
        # Create stock directories
        daily_dir = os.path.join(self.stocks_dir, ticker, "daily")
        hourly_dir = os.path.join(self.stocks_dir, ticker, "hourly")
        os.makedirs(daily_dir, exist_ok=True)
        os.makedirs(hourly_dir, exist_ok=True)
        
        # Save daily config
        if daily_result:
            daily_config = {
                "ticker": ticker,
                "look_forward": daily_result['horizon'],
                "threshold": daily_result['threshold'],
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
                ],
                "cv_score": daily_result['cv_score'],
                "cv_std": daily_result['cv_std']
            }
            
            config_path = os.path.join(daily_dir, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump(daily_config, f, indent=4)
        
        # Save hourly config
        if hourly_result:
            hourly_config = {
                "ticker": ticker,
                "look_forward": hourly_result['horizon'],
                "threshold": hourly_result['threshold'],
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
                ],
                "cv_score": hourly_result['cv_score'],
                "cv_std": hourly_result['cv_std']
            }
            
            config_path = os.path.join(hourly_dir, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump(hourly_config, f, indent=4)
        
    # ==========================================
    # BATCH OPERATIONS
    # ==========================================
    
    def use_defaults(self):
        """Use default horizons for all stocks"""
        from control_center.stock_analyzer import StockAnalyzer
        analyzer = StockAnalyzer()
        
        for ticker in tqdm(analyzer.DOW_30_TICKERS, desc="Setting Default Horizons"):
            daily_res = {'horizon': 5, 'threshold': 0.03, 'cv_score': 0.5, 'cv_std': 0.05}
            hourly_res = {'horizon': 8, 'threshold': 0.015, 'cv_score': 0.5, 'cv_std': 0.05}
            self.save_horizon_config(ticker, daily_res, hourly_res)
    
    def optimize_all(self):
        """Optimize horizons for all stocks"""
        from control_center.stock_analyzer import StockAnalyzer
        analyzer = StockAnalyzer()
        
        # Load global optimized params for strategies
        params_path = config.GLOBAL_OPTIMIZED_PARAMS_PATH
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                optimized_params = json.load(f)
        else:
            optimized_params = {}
            
        for ticker in analyzer.DOW_30_TICKERS:
            daily_res = self.optimize_daily_horizon(ticker, optimized_params)
            hourly_res = self.optimize_hourly_horizon(ticker, optimized_params)
            self.save_horizon_config(ticker, daily_res, hourly_res)