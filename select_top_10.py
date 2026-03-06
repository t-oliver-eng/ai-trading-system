#!/usr/bin/env python3
"""
Select Top 10 Stocks Helper
This script helps you select your top 10 stocks from v3.3 backtest results
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_master_config():
    """Load master config"""
    config_path = os.path.join("control_center", "master_config.json")
    
    if not os.path.exists(config_path):
        print("❌ Error: master_config.json not found!")
        print(f"   Expected location: {config_path}")
        return None
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def save_master_config(config_data):
    """Save master config"""
    config_path = os.path.join("control_center", "master_config.json")
    
    try:
        config_data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        return True
    except Exception as e:
        print(f"❌ Error saving config: {e}")
        return False

def show_v33_rankings(config_data):
    """Show V3.3 backtest rankings"""
    stocks_with_results = []
    
    for ticker, info in config_data.get("stocks", {}).items():
        if "backtest_results" in info and info["backtest_results"]:
            results = info["backtest_results"]
            stocks_with_results.append({
                "ticker": ticker,
                "sharpe": results.get("sharpe", 0),
                "return": results.get("return", 0),
                "max_drawdown": results.get("max_drawdown", 0),
                "num_trades": results.get("num_trades", 0)
            })
    
    if not stocks_with_results:
        print("\n❌ No backtest results found!")
        print("   Run backtests first using the Control Center.")
        return None
    
    # Sort by Sharpe Ratio
    stocks_with_results.sort(key=lambda x: x["sharpe"], reverse=True)
    
    print("\n" + "="*80)
    print("🏆 V3.3 BACKTEST PERFORMANCE RANKINGS (by Sharpe Ratio)")
    print("="*80)
    print(f"\n{'Rank':<6} {'Ticker':<8} {'Sharpe':<10} {'Return %':<12} {'Max DD %':<12} {'Trades':<8}")
    print("-"*80)
    
    for i, stock in enumerate(stocks_with_results, 1):
        print(f"{i:<6} {stock['ticker']:<8} {stock['sharpe']:<10.2f} {stock['return']:<12.2f} "
              f"{stock['max_drawdown']:<12.2f} {stock['num_trades']:<8}")
    
    print("="*80)
    
    return stocks_with_results

def select_by_sharpe(stocks_with_results):
    """Auto-select top 10 by Sharpe ratio"""
    top_10 = [s["ticker"] for s in stocks_with_results[:10]]
    
    print("\n✅ Auto-selected Top 10 by Sharpe Ratio:")
    for i, ticker in enumerate(top_10, 1):
        stock = next(s for s in stocks_with_results if s["ticker"] == ticker)
        print(f"   {i:2d}. {ticker:6s} - Sharpe: {stock['sharpe']:6.2f}, "
              f"Return: {stock['return']:7.2f}%, Trades: {stock['num_trades']:3d}")
    
    return top_10

def select_manually(stocks_with_results):
    """Manually select stocks"""
    print("\n📝 Manual Stock Selection:")
    print("   Enter 10 ticker symbols separated by commas")
    print("   Example: AAPL,MSFT,GOOGL,JPM,V,CAT,NKE,UNH,GS,AXP")
    
    available_tickers = [s["ticker"] for s in stocks_with_results]
    
    while True:
        user_input = input("\nYour top 10: ").strip()
        
        if user_input.lower() == 'cancel':
            return None
        
        selected = [ticker.strip().upper() for ticker in user_input.split(',')]
        selected = [t for t in selected if t]  # Remove empty strings
        
        if len(selected) != 10:
            print(f"❌ Error: You entered {len(selected)} stocks. Please enter exactly 10.")
            continue
        
        # Validate tickers
        invalid = [t for t in selected if t not in available_tickers]
        
        if invalid:
            print(f"❌ Error: Invalid tickers: {', '.join(invalid)}")
            print(f"   Available: {', '.join(available_tickers)}")
            continue
        
        return selected

def select_balanced(stocks_with_results):
    """Select balanced mix of high Sharpe and high return stocks"""
    # Get top 5 by Sharpe
    by_sharpe = sorted(stocks_with_results, key=lambda x: x["sharpe"], reverse=True)
    top_sharpe = set([s["ticker"] for s in by_sharpe[:5]])
    
    # Get top performers by return that aren't already selected
    by_return = sorted(stocks_with_results, key=lambda x: x["return"], reverse=True)
    top_return = []
    
    for stock in by_return:
        if stock["ticker"] not in top_sharpe:
            top_return.append(stock["ticker"])
            if len(top_return) == 5:
                break
    
    top_10 = list(top_sharpe) + top_return
    
    print("\n✅ Balanced Selection (5 by Sharpe, 5 by Return):")
    print("\n   Top 5 by Sharpe Ratio:")
    for ticker in top_sharpe:
        stock = next(s for s in stocks_with_results if s["ticker"] == ticker)
        print(f"   • {ticker:6s} - Sharpe: {stock['sharpe']:6.2f}, Return: {stock['return']:7.2f}%")
    
    print("\n   Top 5 by Return (not in Sharpe top 5):")
    for ticker in top_return:
        stock = next(s for s in stocks_with_results if s["ticker"] == ticker)
        print(f"   • {ticker:6s} - Sharpe: {stock['sharpe']:6.2f}, Return: {stock['return']:7.2f}%")
    
    return top_10

def main():
    """Main function"""
    print("\n" + "="*80)
    print("🎯 SELECT YOUR TOP 10 STOCKS FOR META-AI TRAINING")
    print("="*80)
    
    # Load master config
    config_data = load_master_config()
    if config_data is None:
        return
    
    # Show V3.3 rankings
    stocks_with_results = show_v33_rankings(config_data)
    if stocks_with_results is None:
        return
    
    # Selection method
    print("\n" + "="*80)
    print("SELECTION METHODS:")
    print("="*80)
    print("\n  [1] Auto-select Top 10 by Sharpe Ratio (Recommended)")
    print("  [2] Balanced Mix (5 by Sharpe, 5 by Return)")
    print("  [3] Manual Selection (Choose your own)")
    print("  [0] Cancel")
    
    while True:
        try:
            choice = int(input("\nSelect method (0-3): ").strip())
            if choice in [0, 1, 2, 3]:
                break
            print("❌ Please enter 0, 1, 2, or 3")
        except ValueError:
            print("❌ Please enter a valid number")
    
    if choice == 0:
        print("\n⚠️  Selection cancelled.")
        return
    
    # Perform selection
    if choice == 1:
        top_10 = select_by_sharpe(stocks_with_results)
        method = "Auto-select by Sharpe Ratio"
    elif choice == 2:
        top_10 = select_balanced(stocks_with_results)
        method = "Balanced Mix (Sharpe + Return)"
    else:  # choice == 3
        top_10 = select_manually(stocks_with_results)
        method = "Manual Selection"
        
        if top_10 is None:
            print("\n⚠️  Selection cancelled.")
            return
    
    # Confirm selection
    print("\n" + "="*80)
    print("CONFIRM YOUR SELECTION:")
    print("="*80)
    print(f"\nMethod: {method}")
    print(f"Selected: {', '.join(top_10)}")
    
    confirm = input("\n✅ Save this selection? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("\n⚠️  Selection not saved.")
        return
    
    # Save to master config
    config_data["top_10"] = top_10
    config_data["top_10_selected_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config_data["selection_method"] = method
    
    if save_master_config(config_data):
        print("\n✅ SUCCESS! Top 10 saved to master_config.json")
        print(f"\n⭐ Your Top 10 Stocks:")
        for i, ticker in enumerate(top_10, 1):
            print(f"   {i:2d}. {ticker}")
        
        print("\n📋 Next Steps:")
        print("   1. Use the Control Center to train Meta-AIs for these stocks")
        print("   2. Option [12]: Train Multiple Meta-AIs (select how many)")
        print("   3. Option [13]: Train All Untrained Meta-AIs")
        print("\n💡 You can start with 2-3 stocks to test, then train the rest.")
    else:
        print("\n❌ Failed to save selection.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
