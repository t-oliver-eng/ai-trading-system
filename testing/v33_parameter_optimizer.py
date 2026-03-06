import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import json
import os
import sys
import pickle

os.environ["OMP_NUM_THREADS"] = "1"

from .run_v33_backtest import RealDailyPredictor, RealHourlyPredictor
from utils import config

def run_simulation_logic(params, prices_array, dates_array, prediction_array):
    sl, tp, d_min, h_min, ex_conf, max_hold, cooldown = params
    n_steps = len(prices_array)
    
    cash = 100000.0
    shares = 0
    entry_price = 0.0
    entry_idx = 0
    last_trade_idx = -999
    trades = 0
    peak = 100000.0
    mdd = 0.0
    h_hist = [] 
    
    for i in range(n_steps):
        price = prices_array[i]
        d_conf, h_conf = prediction_array[i]
        
        if h_conf > 0:
            h_hist.append(h_conf)
            if len(h_hist) > 5: h_hist.pop(0)
        
        curr = cash + (shares * price)
        if curr > peak: peak = curr
        dd = (curr - peak) / peak * 100
        if dd < mdd: mdd = dd
        
        if mdd < -20.0: return None

        if shares > 0:
            pnl = (price - entry_price) / entry_price
            days = (dates_array[i] - dates_array[entry_idx]).days
            
            exit_now = False
            if pnl <= -sl: exit_now = True
            elif pnl >= tp: exit_now = True
            elif days >= max_hold: exit_now = True
            elif d_conf > 0 and d_conf < ex_conf: exit_now = True
            
            if exit_now:
                cash += shares * price
                shares = 0
                trades += 1
                last_trade_idx = i
                continue

        if shares == 0:
            if (i - last_trade_idx) < cooldown: continue
            if len(h_hist) < 2: continue
            
            if d_conf >= d_min and h_conf >= h_min:
                prev = sum(h_hist[:-1]) / len(h_hist[:-1])
                if h_conf > prev:
                    shares = int(cash / price)
                    cash -= shares * price
                    entry_price = price
                    entry_idx = i

    if mdd >= -20.0 and trades > 0:
        final = cash + (shares * prices_array[-1])
        ret = (final - 100000.0) / 100000.0 * 100
        score = ret / (abs(mdd) + 1.0)
        
        return {
            'sl': sl, 'tp': tp, 'daily_min': d_min, 
            'hourly_min': h_min, 'exit_conf': ex_conf,
            'max_hold': max_hold, 'cooldown': cooldown,
            'return': ret, 'mdd': mdd, 'trades': trades, 'score': score
        }
    return None

def optimize_v33(ticker, start_date="2025-01-01", end_date="2025-12-31"):
    ticker = ticker.upper()
    
    # Set process to high priority
    try:
        import psutil
        p = psutil.Process(os.getpid())
        if sys.platform == 'win32':
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            p.nice(-10)  # Unix/Linux
        print("✅ Process priority set to HIGH")
    except:
        print("⚠️  Could not set high priority (install psutil: pip install psutil)")
    
    print(f"\n🚀 OPTIMIZATION FOR {ticker}")

    # Load Data
    d_path = os.path.join(config.DATA_DIR, f"{ticker}.csv")
    m_path = os.path.join(config.DATA_DIR, f"{config.MARKET_TICKER}.csv")
    h_path = os.path.join(config.DATA_DIR, "hourly", f"{ticker}.csv")
    mh_path = os.path.join(config.DATA_DIR, "hourly", f"{config.MARKET_TICKER}.csv")

    if not os.path.exists(h_path):
        print("❌ Error: No data found.")
        return

    print("📂 Loading data...")
    d_df = pd.read_csv(d_path, index_col=0, parse_dates=True)
    d_df.index = pd.to_datetime(d_df.index, utc=True).tz_convert("America/New_York")
    m_df = pd.read_csv(m_path, index_col=0, parse_dates=True)
    m_df.index = pd.to_datetime(m_df.index, utc=True).tz_convert("America/New_York")
    h_df = pd.read_csv(h_path, index_col=0, parse_dates=True)
    h_df.index = pd.to_datetime(h_df.index, utc=True).tz_convert("America/New_York")
    mh_df = pd.read_csv(mh_path, index_col=0, parse_dates=True)
    mh_df.index = pd.to_datetime(mh_df.index, utc=True).tz_convert("America/New_York")

    start_dt = pd.Timestamp(start_date).tz_localize("America/New_York")
    end_dt = pd.Timestamp(end_date).tz_localize("America/New_York")
    test_df = h_df[(h_df.index >= start_dt) & (h_df.index <= end_dt)]

    if len(test_df) == 0:
        print("❌ No data in date range.")
        return

    # Predictions with caching
    cache_dir = os.path.join(config.STOCKS_DIR, ticker)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"predictions_cache_{start_date}_{end_date}.pkl")
    
    if os.path.exists(cache_file):
        print(f"📦 Loading cached predictions...")
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            flat_preds = cache_data['predictions']
            prices = cache_data['prices']
            dates = cache_data['dates']
        print(f"✅ Loaded {len(flat_preds)} predictions instantly!")
    else:
        print(f"🧠 Generating {len(test_df)} predictions...")
        print(f"   Loading AI models from disk (one-time cost)...")
        
        # Load models ONCE (not 3865 times!)
        d_pred = RealDailyPredictor(ticker)
        h_pred = RealHourlyPredictor(ticker)
        
        print(f"   Processing {len(test_df)} predictions...")
        
        all_dates = test_df.index.tolist()
        flat_preds = []
        d_cache = {}
        
        # Now predictions use pre-loaded models
        for date in tqdm(all_dates, desc="Predicting", unit="hour"):
            d_key = date.date()
            if d_key not in d_cache:
                try:
                    conf = d_pred.predict(d_df, m_df, date)
                    d_cache[d_key] = conf if conf is not None else 0.0
                except: 
                    d_cache[d_key] = 0.0
            
            d_val = d_cache[d_key]
            try:
                h_val = h_pred.predict(h_df, mh_df, d_df, date)
                h_val = h_val if h_val is not None else 0.0
            except: 
                h_val = 0.0
            
            flat_preds.append((d_val, h_val))
        
        prices = test_df['Close'].values
        dates = test_df.index.to_pydatetime()
        
        # Save cache
        print(f"\n💾 Saving predictions to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'predictions': flat_preds,
                'prices': prices,
                'dates': dates
            }, f)
        print(f"✅ Cache saved! Future runs will be instant.")

    # Grid
    grid = {
    # Test even tighter AND looser stops
    'sl': [0.01, 0.015, 0.02, 0.03],  
    
    # Test higher take profits (let winners run more)
    'tp': [0.02, 0.03, 0.05],
    
    # Test LOWER confidence thresholds (more trades)
    'daily_min': [0.3,0.4,0.5,0.6, 0.65, 0.7, 0.75],
    
    # Test LOWER hourly thresholds
    'hourly_min': [0.3,0.4,0.5,0.6, 0.7, 0.75, 0.85],
    
    # Test lower exit thresholds (stay in longer)
    'exit_conf': [0.4, 0.45, 0.5],
    
    # Test longer holds
    'max_hold': [5, 10, 15],
    
    # Test longer cooldowns (avoid overtrading)
    'cooldown': [0, 1]
}
    combos = list(product(*grid.values()))
    print(f"\n🔥 Testing {len(combos):,} strategies...")
    
    # Simulation with progress bar
    results = []
    for params in tqdm(combos, desc="Simulating", unit="strategy"):
        res = run_simulation_logic(params, prices, dates, flat_preds)
        if res:
            results.append(res)

    print(f"\n✅ Complete! {len(results)} profitable strategies found")

    if not results:
        print("❌ No profitable strategy found.")
        return

    df_res = pd.DataFrame(results).sort_values(by='score', ascending=False)
    
    print("\n" + "="*100)
    print(f"🏆 TOP 20 CONFIGURATIONS FOR {ticker}")
    print("="*100)
    print(df_res.head(20).to_string(index=False))
    print("="*100)

    # Save
    best = df_res.iloc[0].to_dict()
    path = os.path.join(config.STOCKS_DIR, ticker, "v33_optimized_params.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        clean = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in best.items()}
        for k in ['return', 'mdd', 'trades', 'score']:
            if k in clean: del clean[k]
        json.dump(clean, f, indent=4)
    print(f"✅ Saved to: {path}")

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "DIA"
    optimize_v33(ticker)