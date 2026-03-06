# utils/data_fetcher.py

import os
import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame
from tqdm import tqdm
from utils import config


def fetch_data(specific_ticker=None, force=False):
    """
    Downloads and saves daily resolution data.
    If specific_ticker is provided, downloads only that one.
    Otherwise, downloads all tickers in config.TICKERS.
    force: If True, force full re-download even if file exists.
    """
    api = tradeapi.REST(
        config.API_KEY, config.SECRET_KEY, base_url="https://paper-api.alpaca.markets"
    )

    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Determine which tickers to download
    if specific_ticker:
        tickers_to_fetch = [specific_ticker]
        print(f"--- Starting Daily Data Download for {specific_ticker} ---")
    else:
        tickers_to_fetch = config.TICKERS
        print("--- Starting Daily Data Download (All Config Tickers) ---")

    for ticker in tqdm(tickers_to_fetch, desc="Downloading Daily Data"):
        ticker_filepath = os.path.join(config.DATA_DIR, f"{ticker}.csv")
        
        start_date = config.START_DATE
        mode = "full"
        existing_df = None
        
        # Check if we can do an incremental update
        if not force and os.path.exists(ticker_filepath):
            try:
                existing_df = pd.read_csv(ticker_filepath, index_col=0, parse_dates=True)
                if not existing_df.empty:
                    # Ensure index is DatetimeIndex
                    if not isinstance(existing_df.index, pd.DatetimeIndex):
                        existing_df.index = pd.to_datetime(existing_df.index, utc=True)
                    
                    # Handle index timezone
                    act_start = existing_df.index.min()
                    if hasattr(act_start, 'tzinfo') and act_start.tzinfo is not None:
                        act_start = act_start.tz_convert(None)
                    
                    req_start = pd.to_datetime(config.START_DATE).replace(tzinfo=None)
                    
                    if act_start <= req_start + pd.Timedelta(days=60):
                        # Start is good, check end
                        act_end = existing_df.index.max()
                        start_date = (act_end + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                        mode = "update"
                    else:
                        print(f"  ℹ️ {ticker}: Missing historical data (Starts {act_start.date()}). Downloading full history.")
            except Exception as e:
                print(f"  ⚠️ Error reading existing file for {ticker}: {e}. Redownloading.")

        try:
            # Skip if we are updating and up to date
            if mode == "update":
                req_end = pd.to_datetime(config.END_DATE).replace(tzinfo=None)
                last_date = pd.to_datetime(start_date).replace(tzinfo=None)
                if last_date > req_end:
                    continue
            
            bars = api.get_bars(
                ticker,
                TimeFrame.Day,
                start=start_date,
                end=config.END_DATE,
                adjustment="raw",
            ).df

            # Check if data was returned
            if bars.empty:
                if mode == "full":
                    print(f"⚠ No daily data found for {ticker}")
                continue

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

            if mode == "update" and existing_df is not None:
                # Combine and deduplicate
                combined = pd.concat([existing_df, bars])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index(inplace=True)
                combined.to_csv(ticker_filepath)
                print(f"✓ Updated {ticker} (Added {len(bars)} days)")
            else:
                bars.to_csv(ticker_filepath)
                print(f"✓ Saved {ticker} ({len(bars)} days)")

        except Exception as e:
            print(f"✗ Could not download daily data for {ticker}: {e}")

    print("\n--- Daily Data Download Complete ---")


if __name__ == "__main__":
    fetch_data()