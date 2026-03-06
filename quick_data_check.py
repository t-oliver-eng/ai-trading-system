#!/usr/bin/env python3
"""
Quick Data Status Check
Shows which stocks have complete data for training
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import config


def quick_check_stock(ticker):
    """Quick check if stock has data"""
    daily_path = os.path.join(config.DATA_DIR, f"{ticker}.csv")
    hourly_path = os.path.join(config.DATA_DIR, "hourly", f"{ticker}.csv")
    
    result = {
        'ticker': ticker,
        'daily_exists': os.path.exists(daily_path),
        'hourly_exists': os.path.exists(hourly_path),
        'daily_rows': 0,
        'hourly_rows': 0,
        'status': '❌'
    }
    
    # Check daily
    if result['daily_exists']:
        try:
            df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
            result['daily_rows'] = len(df)
            result['daily_start'] = str(df.index.min())[:10]
            result['daily_end'] = str(df.index.max())[:10]
        except:
            pass
    
    # Check hourly
    if result['hourly_exists']:
        try:
            df = pd.read_csv(hourly_path, index_col=0, parse_dates=True)
            result['hourly_rows'] = len(df)
            result['hourly_start'] = str(df.index.min())[:10]
            result['hourly_end'] = str(df.index.max())[:10]
        except:
            pass
    
    # Determine status
    # Rough estimate: 2016-2025 = ~10 years = ~2520 days, ~16380 hours
    if result['daily_rows'] >= 2400 and result['hourly_rows'] >= 15000:
        result['status'] = '✅'
    elif result['daily_rows'] > 0 or result['hourly_rows'] > 0:
        result['status'] = '⚠️'
    
    return result


def main():
    print("\n" + "="*80)
    print("📊 QUICK DATA STATUS CHECK")
    print("="*80)
    
    print(f"\n📅 Looking for data from {config.START_DATE} to {config.END_DATE}")
    
    # Check all stocks
    results = []
    for ticker in config.TICKERS:
        result = quick_check_stock(ticker)
        results.append(result)
    
    # Categorize
    complete = [r for r in results if r['status'] == '✅']
    incomplete = [r for r in results if r['status'] == '⚠️']
    missing = [r for r in results if r['status'] == '❌']
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   ✅ Complete: {len(complete)}")
    print(f"   ⚠️  Incomplete: {len(incomplete)}")
    print(f"   ❌ Missing: {len(missing)}")
    
    # Details
    if complete:
        print(f"\n✅ READY FOR TRAINING ({len(complete)} stocks):")
        for r in complete:
            print(f"   {r['ticker']:6s} - Daily: {r['daily_rows']:5d} rows, Hourly: {r['hourly_rows']:6d} rows")
    
    if incomplete:
        print(f"\n⚠️  INCOMPLETE DATA ({len(incomplete)} stocks):")
        for r in incomplete:
            daily_info = f"{r['daily_rows']:5d} rows" if r['daily_exists'] else "MISSING"
            hourly_info = f"{r['hourly_rows']:6d} rows" if r['hourly_exists'] else "MISSING"
            print(f"   {r['ticker']:6s} - Daily: {daily_info}, Hourly: {hourly_info}")
    
    if missing:
        print(f"\n❌ NO DATA ({len(missing)} stocks):")
        print(f"   {', '.join([r['ticker'] for r in missing])}")
    
    # Recommendations
    print(f"\n" + "="*80)
    if incomplete or missing:
        print("💡 RECOMMENDATION:")
        print(f"   Run: python validate_and_download_data.py")
        print(f"   This will re-download incomplete/missing data")
    else:
        print("✅ ALL STOCKS READY FOR TRAINING!")
        print(f"   You can safely train Meta-AIs for all {len(complete)} stocks")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
