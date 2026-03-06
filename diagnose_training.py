#!/usr/bin/env python3
"""
Training Diagnostics - Analyze what's wrong with current training

Usage:
    python diagnose_training.py
"""

import os
import sys
from pathlib import Path
import csv

def find_all_progress_files():
    """Find all progress.csv files in the project"""
    progress_files = []
    
    # Search common locations
    search_dirs = [
        Path("data/logs"),
        Path("logs"),
        Path("../data/logs"),
        Path(".")
    ]
    
    for search_dir in search_dirs:
        if search_dir.exists():
            for progress_file in search_dir.rglob("progress.csv"):
                progress_files.append(progress_file)
    
    return progress_files

def analyze_progress_file(filepath):
    """Analyze a progress file and diagnose issues"""
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        if not rows:
            return None
        
        last_row = rows[-1]
        
        # Extract key metrics
        timesteps = int(float(last_row.get('time/total_timesteps', 0)))
        ep_rew_mean = float(last_row.get('rollout/ep_rew_mean', 0))
        ep_len_mean = float(last_row.get('rollout/ep_len_mean', 0))
        value_loss = float(last_row.get('train/value_loss', 0))
        
        # Diagnose issues
        issues = []
        status = "🟢 GOOD"
        
        if ep_rew_mean < -10000:
            issues.append("🔴 CRITICAL: Extremely negative rewards - OLD BROKEN REWARD SYSTEM")
            status = "🔴 CRITICAL"
        elif ep_rew_mean < -1000 and timesteps > 2_000_000:
            issues.append("🔴 BAD: Still very negative rewards after 2M steps")
            status = "🔴 BAD"
        elif ep_rew_mean < 0 and timesteps > 5_000_000:
            issues.append("🟠 POOR: Negative rewards after 5M steps")
            status = "🟠 POOR"
        
        if ep_len_mean < 5000:
            issues.append("⚠️ Episodes ending early - agent dying or giving up")
        
        if value_loss > 100000:
            issues.append("🔴 CRITICAL: Massive value loss - reward scale is broken")
            status = "🔴 CRITICAL"
        elif value_loss > 10000:
            issues.append("⚠️ High value loss - agent is confused")
        
        if timesteps > 0 and ep_rew_mean < -100000:
            issues.append("💀 DIAGNOSIS: Using OLD reward system - MUST UPDATE FILES")
            status = "💀 BROKEN"
        
        return {
            'filepath': filepath,
            'timesteps': timesteps,
            'ep_rew_mean': ep_rew_mean,
            'ep_len_mean': ep_len_mean,
            'value_loss': value_loss,
            'issues': issues,
            'status': status
        }
    
    except Exception as e:
        return None

def main():
    print("\n" + "="*70)
    print("🔍 TRAINING DIAGNOSTICS")
    print("="*70 + "\n")
    
    # Find all progress files
    progress_files = find_all_progress_files()
    
    if not progress_files:
        print("❌ No training progress files found!")
        print("\nSearched in:")
        print("   - data/logs/")
        print("   - logs/")
        print("   - Current directory")
        print("\n💡 Have you started training yet?")
        return
    
    print(f"📂 Found {len(progress_files)} training log(s):\n")
    
    # Analyze each
    for pf in progress_files:
        result = analyze_progress_file(pf)
        
        if not result:
            continue
        
        print(f"📊 {result['status']} - {pf.parent.name}")
        print(f"   File: {pf}")
        print(f"   Timesteps: {result['timesteps']:,}")
        print(f"   Avg Reward: {result['ep_rew_mean']:,.0f}")
        print(f"   Episode Length: {result['ep_len_mean']:,.0f}")
        print(f"   Value Loss: {result['value_loss']:,.0f}")
        
        if result['issues']:
            print(f"\n   ⚠️  ISSUES DETECTED:")
            for issue in result['issues']:
                print(f"      {issue}")
        else:
            print(f"\n   ✅ No major issues detected")
        
        print()
    
    # Overall recommendation
    print("="*70)
    print("💡 RECOMMENDATION:")
    print("="*70 + "\n")
    
    # Check if any critical issues
    critical_found = False
    for pf in progress_files:
        result = analyze_progress_file(pf)
        if result and result['status'] in ["🔴 CRITICAL", "💀 BROKEN"]:
            critical_found = True
            break
    
    if critical_found:
        print("🛑 CRITICAL ISSUES DETECTED!")
        print("\n   Your training is using the OLD BROKEN reward system.")
        print("   This will NEVER produce profitable trades.\n")
        print("   📋 ACTION REQUIRED:")
        print("      1. STOP current training (Ctrl+C)")
        print("      2. Replace trading_env.py with the NEW version")
        print("      3. Replace meta_trainer.py with the NEW version")
        print("      4. Delete old cache: rm data/stocks/*/meta/prediction_cache.npz")
        print("      5. Start fresh: python meta_trainer.py --ticker AAPL\n")
        print("   🎯 With new system, you should see:")
        print("      - ep_rew_mean trending from -1000 → 0 → +2000")
        print("      - value_loss < 10,000")
        print("      - Episodes completing (ep_len_mean ~10,000)\n")
    else:
        print("✅ Training looks healthy!")
        print("\n   Continue monitoring progress.")
        print("   Use: python monitor_training.py TICKER --watch\n")

if __name__ == "__main__":
    main()