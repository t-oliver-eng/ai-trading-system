#!/usr/bin/env python3
"""
System Health Check - Comprehensive Analysis
"""

import os
import json
from pathlib import Path
from datetime import datetime

def check_data_coverage():
    """Check what data has been downloaded"""
    print("\n" + "="*70)
    print("📊 DATA COVERAGE")
    print("="*70)
    
    # Stock price CSVs
    csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    print(f"\n✅ Stock price CSV files: {len(csv_files)}")
    
    # Historical data folders
    stocks_dir = Path('data/historical/stocks')
    if stocks_dir.exists():
        stock_folders = [d for d in stocks_dir.iterdir() if d.is_dir()]
        print(f"✅ Stocks with historical data: {len(stock_folders)}")
        print(f"   Tickers: {sorted([s.name for s in stock_folders])}")

def check_model_status():
    """Check which models are trained"""
    print("\n" + "="*70)
    print("🤖 MODEL TRAINING STATUS")
    print("="*70)
    
    stocks_dir = Path('data/historical/stocks')
    
    daily_models = []
    hourly_models = []
    meta_models = []
    
    if stocks_dir.exists():
        for stock_folder in stocks_dir.iterdir():
            if stock_folder.is_dir():
                ticker = stock_folder.name
                
                # Check for daily model
                if (stock_folder / 'daily' / 'daily_model.joblib').exists():
                    daily_models.append(ticker)
                
                # Check for hourly model
                if (stock_folder / 'hourly' / 'hourly_model.joblib').exists():
                    hourly_models.append(ticker)
                
                # Check for meta AI
                if (stock_folder / 'meta' / 'meta_ai_final.zip').exists():
                    meta_models.append(ticker)
    
    print(f"\n✅ Daily AI Models (XGBoost): {len(daily_models)}/34")
    print(f"   Trained: {sorted(daily_models)}")
    
    print(f"\n✅ Hourly AI Models (XGBoost): {len(hourly_models)}/34")
    print(f"   Trained: {sorted(hourly_models)}")
    
    print(f"\n🤖 Meta-AI Models (PPO): {len(meta_models)}/34")
    print(f"   Trained: {sorted(meta_models)}")
    
    # Missing models
    all_stocks = set([s.name for s in stocks_dir.iterdir() if s.is_dir()])
    missing_daily = all_stocks - set(daily_models)
    missing_hourly = all_stocks - set(hourly_models)
    missing_meta = all_stocks - set(meta_models)
    
    if missing_daily:
        print(f"\n⚠️  Missing Daily Models: {sorted(missing_daily)}")
    if missing_hourly:
        print(f"⚠️  Missing Hourly Models: {sorted(missing_hourly)}")
    if missing_meta:
        print(f"⚠️  Missing Meta-AI Models: {sorted(missing_meta)}")

def check_aapl_meta_training():
    """Detailed check of AAPL Meta-AI training"""
    print("\n" + "="*70)
    print("🎯 AAPL META-AI DETAILED STATUS")
    print("="*70)
    
    aapl_meta = Path('data/historical/stocks/AAPL/meta')
    
    if not aapl_meta.exists():
        print("❌ AAPL meta directory not found")
        return
    
    # Check files
    files = {
        'Best Model': aapl_meta / 'best_model.zip',
        'Final Model': aapl_meta / 'meta_ai_final.zip',
        'VecNormalize Stats': aapl_meta / 'vec_normalize.pkl',
        'Forward Test Metrics': aapl_meta / 'forward_test_metrics.json',
        'Forward Test Trades': aapl_meta / 'forward_test_trades.json',
        'Prediction Cache': aapl_meta / 'prediction_cache.npz',
    }
    
    print("\n📁 Key Files:")
    for name, path in files.items():
        if path.exists():
            size_mb = path.stat().st_size / (1024**2)
            print(f"   ✅ {name}: {size_mb:.2f} MB")
        else:
            print(f"   ❌ {name}: Missing")
    
    # Checkpoints
    checkpoints_dir = aapl_meta / 'checkpoints'
    if checkpoints_dir.exists():
        checkpoints = list(checkpoints_dir.glob('*.zip'))
        print(f"\n💾 Training Checkpoints:")
        print(f"   Total saved: {len(checkpoints)}")
        
        if checkpoints:
            # Find latest
            latest = sorted(checkpoints)[-1]
            print(f"   Latest: {latest.name}")
            
            # Calculate total size
            total_size = sum([c.stat().st_size for c in checkpoints])
            print(f"   Total size: {total_size / (1024**3):.2f} GB")
    
    # Prediction caches
    prediction_files = list(aapl_meta.glob('predictions_*.npz'))
    if prediction_files:
        print(f"\n🔮 Prediction Caches: {len(prediction_files)}")
        for pf in sorted(prediction_files):
            print(f"   - {pf.name}")
    
    # Forward test metrics
    metrics_file = aapl_meta / 'forward_test_metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print(f"\n📊 Forward Test Results:")
        print(f"   Period: {metrics.get('test_period', 'N/A')}")
        print(f"   Total Return: {metrics.get('total_return_pct', 0):.2f}%")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
        print(f"   Final Value: ${metrics.get('final_value', 0):,.2f}")
        print(f"   Total Trades: {metrics.get('total_trades', 0)}")

def check_backtest_results():
    """Check backtest results"""
    print("\n" + "="*70)
    print("📈 BACKTEST RESULTS")
    print("="*70)
    
    backtest_dir = Path('data/backtest_results')
    
    if not backtest_dir.exists():
        print("❌ No backtest directory found")
        return
    
    results = [d for d in backtest_dir.iterdir() if d.is_dir()]
    print(f"\n✅ Total backtests: {len(results)}")
    
    # Extract unique tickers
    tickers = set()
    for result in results:
        ticker = result.name.split('_')[0]
        tickers.add(ticker)
    
    print(f"✅ Unique tickers backtested: {len(tickers)}")
    print(f"   Tickers: {sorted(tickers)}")
    
    # Sample a few results
    print(f"\n📊 Sample Backtest Results:")
    for result_dir in sorted(results)[:3]:
        metrics_file = result_dir / 'metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            ticker = metrics.get('ticker', result_dir.name.split('_')[0])
            sharpe = metrics.get('sharpe_ratio', 0)
            total_return = metrics.get('total_return_pct', 0)
            print(f"   {ticker}: Sharpe {sharpe:.2f}, Return {total_return:.2f}%")

def check_system_files():
    """Check system configuration files"""
    print("\n" + "="*70)
    print("⚙️  SYSTEM FILES")
    print("="*70)
    
    files_to_check = {
        'requirements.txt': 'Dependencies list',
        'control_center.py': 'Main control center',
        'models/meta/meta_trainer.py': 'Meta-AI trainer',
        'models/meta/trading_env.py': 'Trading environment',
        'utils/config.py': 'Configuration',
        'data/meta_ai_results.json': 'Meta-AI rankings',
    }
    
    for filepath, description in files_to_check.items():
        path = Path(filepath)
        if path.exists():
            print(f"   ✅ {description}: {filepath}")
        else:
            print(f"   ⚠️  {description}: {filepath} (missing)")

def check_disk_usage():
    """Check disk space usage"""
    print("\n" + "="*70)
    print("💾 DISK USAGE")
    print("="*70)
    
    directories = {
        'data': 'Data directory',
        'data/historical/stocks': 'Stock data',
        'data/backtest_results': 'Backtest results',
        'data/logs': 'Logs',
    }
    
    for dir_path, description in directories.items():
        path = Path(dir_path)
        if path.exists():
            total_size = 0
            for root, dirs, files in os.walk(path):
                total_size += sum([os.path.getsize(os.path.join(root, f)) for f in files])
            
            size_gb = total_size / (1024**3)
            print(f"   {description}: {size_gb:.2f} GB")

def main():
    print("\n" + "="*70)
    print("🔍 TRADING SYSTEM HEALTH CHECK")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    check_data_coverage()
    check_model_status()
    check_aapl_meta_training()
    check_backtest_results()
    check_system_files()
    check_disk_usage()
    
    print("\n" + "="*70)
    print("✅ HEALTH CHECK COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
