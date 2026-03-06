#!/usr/bin/env python3
"""
Training Monitor - Robust Version
"""

import os
import sys
import time
from pathlib import Path
import argparse

def find_latest_progress_csv(ticker):
    """Find the most recently modified progress.csv anywhere in the logs"""
    
    # Places to look
    search_roots = [
        Path(f"data/logs/meta_ai/{ticker}"),
        Path(f"data/logs/meta_ai"),
        Path(f"logs/meta_ai/{ticker}")
    ]
    
    candidates = []
    
    for root in search_roots:
        if root.exists():
            # Recursively find all progress.csv files
            for path in root.rglob("progress.csv"):
                # Ensure it belongs to the ticker we want (if checking generic folder)
                if ticker in str(path):
                    candidates.append(path)
    
    if not candidates:
        return None
        
    # Sort by modification time, newest first
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]

def parse_progress_file(filepath):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2: return None
        
        header = lines[0].strip().split(',')
        # Get last non-empty line
        last_line = lines[-1].strip().split(',')
        if not last_line or len(last_line) != len(header):
            last_line = lines[-2].strip().split(',')
            
        return dict(zip(header, last_line))
    except:
        return None

def format_num(val):
    try:
        f = float(val)
        if abs(f) > 1000: return f"{f:,.0f}"
        return f"{f:.4f}"
    except: return val

def monitor(ticker, watch=True, interval=5):
    print(f"🔍 Searching for logs for {ticker}...")
    
    csv_path = find_latest_progress_csv(ticker)
    
    if not csv_path:
        print(f"❌ No logs found. Are you sure '{ticker}' is training?")
        print(f"   Checked data/logs/meta_ai/{ticker} and subfolders.")
        return

    print(f"✅ Found log: {csv_path}")
    print(f"{'='*60}")
    
    try:
        while True:
            data = parse_progress_file(csv_path)
            
            if data:
                # Clear screen
                os.system('cls' if os.name == 'nt' else 'clear')
                
                ts = data.get('time/total_timesteps', '0')
                rew = data.get('rollout/ep_rew_mean', '0')
                length = data.get('rollout/ep_len_mean', '0')
                fps = data.get('time/fps', '0')
                
                print(f"📊 MONITORING: {ticker}")
                print(f"📂 Source: {csv_path.name}")
                print(f"{'='*60}")
                print(f"⏱️  Steps:      {format_num(ts)}")
                print(f"💰 Avg Reward: {format_num(rew)}  <-- POSITIVE IS GOOD!")
                print(f"📏 Ep Length:  {format_num(length)}")
                print(f"🚀 Speed:      {format_num(fps)} fps")
                print(f"\n(Ctrl+C to stop monitoring)")
            else:
                print("Waiting for data...")
            
            if not watch: break
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", type=str)
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    monitor(args.ticker.upper(), watch=args.watch)