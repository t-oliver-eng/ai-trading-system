# make_config.py

import os
import json
import argparse
from utils import config

def create_configs(ticker):
    ticker = ticker.upper()
    print(f"--- Creating ULTIMATE Training Configs for {ticker} ---")

    # ==========================================
    # 1. DAILY CONFIGURATION
    # ==========================================
    daily_dir = os.path.join(config.STOCKS_DIR, ticker, "daily")
    os.makedirs(daily_dir, exist_ok=True)
    
    daily_config = {
        "ticker": ticker,
        # Flattened parameters (No "training_params" folder)
        "look_forward": 5,    
        "threshold": 0.03,   
        "start_date": "2020-01-01", # Added these so the trainer doesn't error on dates
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
    
    with open(os.path.join(daily_dir, "training_config.json"), "w") as f:
        json.dump(daily_config, f, indent=4)
    print(f"✅ Created daily/training_config.json")

    # ==========================================
    # 2. HOURLY CONFIGURATION
    # ==========================================
    hourly_dir = os.path.join(config.STOCKS_DIR, ticker, "hourly")
    os.makedirs(hourly_dir, exist_ok=True)
    
    hourly_config = {
        "ticker": ticker,
        # Flattened parameters
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
            "atr", "Daily_RSI", "Daily_50MA_Ratio", "Daily_200MA_Ratio", "SPY_RSI_Hourly"
        ],
        "strategies": [
            "HourlyEMACross", "HourlyRSI", "VolumeAnomaly", "HourlyBollinger", 
            "HourlyStochastic", "HourlyMACD", "HourlyCMF", "HourlyROC", 
            "HourlyADX", "HourlyPSAR", "HourlyTEMA", "HourlyLongEMACross", 
            "HeikinAshi", "RVI", "FisherTransform"
        ]
    }
    
    with open(os.path.join(hourly_dir, "training_config.json"), "w") as f:
        json.dump(hourly_config, f, indent=4)
    print(f"✅ Created hourly/training_config.json")
    
    print("\n---------------------------------------------------")
    print("DONE. Now your original trainers will find the keys they need.")
    print("---------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, required=True)
    args = parser.parse_args()
    create_configs(args.ticker)