#!/usr/bin/env python3
"""
Data Validator & Re-Downloader
Checks which stocks have complete data for the configured date range
and re-downloads any that are missing or incomplete
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import config
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from tqdm import tqdm


def parse_date(date_str):
    """Parse date string to datetime"""
    return pd.to_datetime(date_str)


def check_data_coverage(df, start_date, end_date, data_type="daily"):
    """
    Check if DataFrame has sufficient coverage for the date range
    
    Returns:
        tuple: (has_coverage: bool, actual_start: str, actual_end: str, missing_days: int)
    """
    if df is None or df.empty:
        return False, None, None, None
    
    # Get actual date range
    actual_start = df.index.min()
    actual_end = df.index.max()
    
    # Parse expected dates
    expected_start = parse_date(start_date)
    expected_end = parse_date(end_date)
    
    # Calculate expected trading days (rough estimate)
    # Daily: ~252 trading days per year
    # Hourly: ~252 * 6.5 hours per year
    years = (expected_end - expected_start).days / 365.25
    
    if data_type == "daily":
        expected_rows = int(years * 252)
        tolerance = 0.95  # Allow 5% missing (holidays, etc.)
    else:  # hourly
        expected_rows = int(years * 252 * 6.5)
        tolerance = 0.90  # Allow 10% missing (extended hours variations)
    
    min_required_rows = int(expected_rows * tolerance)
    
    # Check if we have enough data
    has_coverage = (
        len(df) >= min_required_rows and
        actual_start <= expected_start + timedelta(days=30) and  # Allow 30 day buffer
        actual_end >= expected_end - timedelta(days=30)
    )
    
    missing_pct = ((expected_rows - len(df)) / expected_rows * 100) if expected_rows > 0 else 0
    
    return has_coverage, actual_start, actual_end, missing_pct


def validate_stock_data(ticker):
    """
    Validate that a stock has complete daily and hourly data
    
    Returns:
        dict: Status for daily and hourly data
    """
    results = {
        'ticker': ticker,
        'daily': {'exists': False, 'complete': False, 'details': None},
        'hourly': {'exists': False, 'complete': False, 'details': None}
    }
    
    # Check daily data
    daily_path = os.path.join(config.DATA_DIR, f"{ticker}.csv")
    if os.path.exists(daily_path):
        results['daily']['exists'] = True
        try:
            df = pd.read_csv(daily_path, index_col=0, parse_dates=True)
            has_coverage, start, end, missing_pct = check_data_coverage(
                df, config.START_DATE, config.END_DATE, "daily"
            )
            
            results['daily']['complete'] = has_coverage
            results['daily']['details'] = {
                'rows': len(df),
                'start': str(start)[:10] if start else None,
                'end': str(end)[:10] if end else None,
                'missing_pct': missing_pct
            }
        except Exception as e:
            results['daily']['error'] = str(e)
    
    # Check hourly data
    hourly_path = os.path.join(config.DATA_DIR, "hourly", f"{ticker}.csv")
    if os.path.exists(hourly_path):
        results['hourly']['exists'] = True
        try:
            df = pd.read_csv(hourly_path, index_col=0, parse_dates=True)
            has_coverage, start, end, missing_pct = check_data_coverage(
                df, config.START_DATE, config.END_DATE, "hourly"
            )
            
            results['hourly']['complete'] = has_coverage
            results['hourly']['details'] = {
                'rows': len(df),
                'start': str(start)[:10] if start else None,
                'end': str(end)[:10] if end else None,
                'missing_pct': missing_pct
            }
        except Exception as e:
            results['hourly']['error'] = str(e)
    
    return results


def download_stock_data(ticker, data_type="both", force=False):
    """
    Download data for a specific stock
    
    Args:
        ticker: Stock ticker
        data_type: "daily", "hourly", or "both"
        force: Force re-download even if exists
    """
    api = tradeapi.REST(
        config.API_KEY, 
        config.SECRET_KEY, 
        base_url="https://paper-api.alpaca.markets"
    )
    
    success = {'daily': False, 'hourly': False}
    
    # Download daily data
    if data_type in ["daily", "both"]:
        daily_path = os.path.join(config.DATA_DIR, f"{ticker}.csv")
        
        if force or not os.path.exists(daily_path):
            try:
                print(f"  📥 Downloading daily data for {ticker}...")
                bars = api.get_bars(
                    ticker,
                    TimeFrame.Day,
                    start=config.START_DATE,
                    end=config.END_DATE,
                    adjustment="raw",
                ).df
                
                if not bars.empty:
                    bars = bars.tz_convert("America/New_York")
                    bars.rename(
                        columns={
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                        },
                        inplace=True,
                    )
                    bars.to_csv(daily_path)
                    print(f"    ✅ Daily: {len(bars)} days saved")
                    success['daily'] = True
                else:
                    print(f"    ⚠️  Daily: No data returned")
            except Exception as e:
                print(f"    ❌ Daily: {e}")
    
    # Download hourly data
    if data_type in ["hourly", "both"]:
        hourly_dir = os.path.join(config.DATA_DIR, "hourly")
        os.makedirs(hourly_dir, exist_ok=True)
        hourly_path = os.path.join(hourly_dir, f"{ticker}.csv")
        
        if force or not os.path.exists(hourly_path):
            try:
                print(f"  📥 Downloading hourly data for {ticker}...")
                bars = api.get_bars(
                    ticker,
                    TimeFrame.Hour,
                    start=config.START_DATE,
                    end=config.END_DATE,
                    adjustment="raw",
                ).df
                
                if not bars.empty:
                    bars = bars.tz_convert("America/New_York")
                    bars.rename(
                        columns={
                            "open": "Open",
                            "high": "High",
                            "low": "Low",
                            "close": "Close",
                            "volume": "Volume",
                        },
                        inplace=True,
                    )
                    bars.to_csv(hourly_path)
                    print(f"    ✅ Hourly: {len(bars)} hours saved")
                    success['hourly'] = True
                else:
                    print(f"    ⚠️  Hourly: No data returned")
            except Exception as e:
                print(f"    ❌ Hourly: {e}")
    
    return success


def main():
    """Main validation and download routine"""
    print("\n" + "="*80)
    print("📊 DATA VALIDATION & RE-DOWNLOAD UTILITY")
    print("="*80)
    
    print(f"\n📅 Configured Date Range:")
    print(f"   Start: {config.START_DATE}")
    print(f"   End:   {config.END_DATE}")
    
    # Calculate expected data
    start = parse_date(config.START_DATE)
    end = parse_date(config.END_DATE)
    years = (end - start).days / 365.25
    
    print(f"   Duration: {years:.1f} years")
    print(f"   Expected daily rows: ~{int(years * 252)}")
    print(f"   Expected hourly rows: ~{int(years * 252 * 6.5)}")
    
    # Get tickers to check
    tickers = config.TICKERS
    
    print(f"\n🔍 Validating {len(tickers)} stocks...")
    print("="*80)
    
    # Validate all stocks
    validation_results = {}
    complete = []
    incomplete = []
    missing = []
    
    for ticker in tickers:
        result = validate_stock_data(ticker)
        validation_results[ticker] = result
        
        daily_ok = result['daily']['complete']
        hourly_ok = result['hourly']['complete']
        
        if daily_ok and hourly_ok:
            complete.append(ticker)
        elif result['daily']['exists'] or result['hourly']['exists']:
            incomplete.append(ticker)
        else:
            missing.append(ticker)
    
    # Print summary
    print(f"\n📊 VALIDATION SUMMARY:")
    print("="*80)
    print(f"✅ Complete (both daily & hourly): {len(complete)}")
    print(f"⚠️  Incomplete (missing or partial): {len(incomplete)}")
    print(f"❌ Missing (no data at all): {len(missing)}")
    
    # Show detailed results
    if incomplete or missing:
        print(f"\n⚠️  STOCKS NEEDING ATTENTION:")
        print("="*80)
        
        for ticker in incomplete + missing:
            result = validation_results[ticker]
            print(f"\n{ticker}:")
            
            # Daily status
            if result['daily']['exists']:
                details = result['daily']['details']
                if details:
                    if result['daily']['complete']:
                        print(f"  ✅ Daily: {details['rows']} rows ({details['start']} to {details['end']})")
                    else:
                        print(f"  ⚠️  Daily: {details['rows']} rows ({details['start']} to {details['end']}) - INCOMPLETE")
                        if details.get('missing_pct'):
                            print(f"      Missing ~{details['missing_pct']:.1f}% of expected data")
                else:
                    error_msg = result['daily'].get('error', 'Unknown error')
                    print(f"  ❌ Daily: Error reading data - {error_msg}")
            else:
                print(f"  ❌ Daily: NOT FOUND")
            
            # Hourly status
            if result['hourly']['exists']:
                details = result['hourly']['details']
                if details:
                    if result['hourly']['complete']:
                        print(f"  ✅ Hourly: {details['rows']} rows ({details['start']} to {details['end']})")
                    else:
                        print(f"  ⚠️  Hourly: {details['rows']} rows ({details['start']} to {details['end']}) - INCOMPLETE")
                        if details.get('missing_pct'):
                            print(f"      Missing ~{details['missing_pct']:.1f}% of expected data")
                else:
                    error_msg = result['hourly'].get('error', 'Unknown error')
                    print(f"  ❌ Hourly: Error reading data - {error_msg}")
            else:
                print(f"  ❌ Hourly: NOT FOUND")
    
    if complete:
        print(f"\n✅ COMPLETE STOCKS ({len(complete)}):")
        print(f"   {', '.join(complete)}")
    
    # Ask if user wants to re-download
    if incomplete or missing:
        print(f"\n" + "="*80)
        print("🔄 RE-DOWNLOAD OPTIONS:")
        print("="*80)
        
        stocks_to_fix = incomplete + missing
        print(f"\n{len(stocks_to_fix)} stocks need to be downloaded/updated:")
        print(f"{', '.join(stocks_to_fix)}")
        
        print(f"\nThis will download {years:.1f} years of data for each stock.")
        print(f"Estimated time: ~{len(stocks_to_fix) * 0.5:.1f} minutes")
        
        response = input(f"\nRe-download all incomplete/missing stocks? (y/n): ").strip().lower()
        
        if response == 'y':
            print(f"\n📥 Starting re-download...")
            print("="*80)
            
            success_count = 0
            for ticker in tqdm(stocks_to_fix, desc="Re-downloading"):
                result = download_stock_data(ticker, data_type="both", force=True)
                if result['daily'] and result['hourly']:
                    success_count += 1
            
            print(f"\n✅ Re-download complete!")
            print(f"   Successfully downloaded: {success_count}/{len(stocks_to_fix)}")
            
            # Re-validate
            print(f"\n🔍 Re-validating...")
            still_incomplete = []
            
            for ticker in stocks_to_fix:
                result = validate_stock_data(ticker)
                if not (result['daily']['complete'] and result['hourly']['complete']):
                    still_incomplete.append(ticker)
            
            if still_incomplete:
                print(f"\n⚠️  Still incomplete: {', '.join(still_incomplete)}")
                print(f"   These may not have data available for the full date range")
            else:
                print(f"\n✅ All stocks now have complete data!")
        else:
            print(f"\n⚠️  Re-download cancelled")
    else:
        print(f"\n🎉 All stocks have complete data!")
    
    # Update master config
    print(f"\n📝 Updating system status...")
    update_system_status(validation_results)
    
    print(f"\n" + "="*80)
    print("✅ DATA VALIDATION COMPLETE")
    print("="*80 + "\n")


def update_system_status(validation_results):
    """Update master config with data download status"""
    master_config_path = os.path.join("control_center", "master_config.json")
    
    if not os.path.exists(master_config_path):
        print("⚠️  master_config.json not found - skipping status update")
        return
    
    try:
        import json
        
        with open(master_config_path, 'r') as f:
            config_data = json.load(f)
        
        # Count complete stocks
        complete_count = sum(
            1 for r in validation_results.values()
            if r['daily']['complete'] and r['hourly']['complete']
        )
        
        # Update system info
        config_data['system_info']['stocks_downloaded'] = complete_count
        config_data['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(master_config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        print(f"✅ Updated master_config.json")
        print(f"   Stocks with complete data: {complete_count}")
        
    except Exception as e:
        print(f"⚠️  Could not update master_config.json: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
