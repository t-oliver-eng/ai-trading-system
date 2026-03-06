#!/usr/bin/env python3
"""
Meta-AI Training Status Checker
Quick overview of your training progress
"""

import os
import sys
import json
from datetime import datetime

def load_master_config():
    """Load master config"""
    config_path = os.path.join("control_center", "master_config.json")
    
    if not os.path.exists(config_path):
        print("❌ master_config.json not found!")
        return None
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return None

def main():
    """Main function"""
    print("\n" + "="*80)
    print("📊 META-AI TRAINING STATUS")
    print("="*80)
    
    config_data = load_master_config()
    if config_data is None:
        return
    
    # Get top 10
    top_10 = config_data.get("top_10", [])
    
    if not top_10:
        print("\n⚠️  NO TOP 10 STOCKS SELECTED YET")
        print("\n📋 Next Steps:")
        print("   1. Run: python select_top_10.py")
        print("   2. Or use Control Center option [9]")
        return
    
    # Show top 10 info
    print(f"\n⭐ YOUR TOP 10 STOCKS:")
    print(f"   Selected: {config_data.get('top_10_selected_date', 'Unknown')}")
    print(f"   Method: {config_data.get('selection_method', 'Unknown')}")
    print(f"\n   Stocks: {', '.join(top_10)}")
    
    # Get Meta-AI training status
    meta_status = config_data.get("meta_training_status", {})
    
    print(f"\n" + "="*80)
    print("🤖 META-AI TRAINING STATUS:")
    print("="*80)
    
    trained = []
    untrained = []
    
    for ticker in top_10:
        status = meta_status.get(ticker, {})
        if status.get("trained", False):
            trained.append((ticker, status))
        else:
            untrained.append(ticker)
    
    # Show trained
    if trained:
        print(f"\n✅ TRAINED ({len(trained)}/{len(top_10)}):")
        print(f"\n   {'Ticker':<8} {'Sharpe':<10} {'Return %':<12} {'Trades':<10} {'Date Trained':<15}")
        print("   " + "-"*65)
        
        for ticker, status in trained:
            metrics = status.get("metrics", {})
            sharpe = metrics.get("sharpe_ratio", 0)
            return_pct = metrics.get("total_return_pct", 0)
            trades = metrics.get("total_trades", 0)
            date = status.get("last_trained", "Unknown")[:10]
            
            print(f"   {ticker:<8} {sharpe:<10.2f} {return_pct:<12.2f} {trades:<10} {date:<15}")
    else:
        print(f"\n✅ TRAINED (0/{len(top_10)}): None yet")
    
    # Show untrained
    if untrained:
        print(f"\n⏳ UNTRAINED ({len(untrained)}/{len(top_10)}):")
        print(f"   {', '.join(untrained)}")
        
        # Estimate time
        hours_per_stock = 1.5  # With GPU
        total_hours = len(untrained) * hours_per_stock
        
        print(f"\n   ⏱️  Estimated training time: {total_hours:.1f} hours with GPU")
        print(f"   💡 Recommended: Train {min(3, len(untrained))} stocks first to test")
    else:
        print(f"\n🎉 ALL STOCKS TRAINED!")
        print(f"   Ready for Phase 2: Portfolio Manager RL training")
    
    # Show next steps
    print(f"\n" + "="*80)
    print("📋 NEXT STEPS:")
    print("="*80)
    
    if untrained:
        if len(trained) == 0:
            print("\n   🚀 START TRAINING:")
            print("      1. Run: python control_center.py")
            print("      2. Select [12] Train Multiple Meta-AIs")
            print("      3. Start with 2-3 stocks to test the system")
        else:
            print("\n   🔄 CONTINUE TRAINING:")
            print("      1. Run: python control_center.py")
            print("      2. Option [12]: Train specific stocks")
            print("      3. Option [13]: Train all remaining stocks")
    else:
        print("\n   ✨ ALL META-AIs TRAINED! Next:")
        print("      1. Review performance: Control Center option [14]")
        print("      2. Start building your Portfolio Manager RL")
        print("      3. Use Meta-AI outputs + confidence scores")
    
    print("\n" + "="*80)
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
