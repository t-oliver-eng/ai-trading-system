#!/usr/bin/env python3
"""
PRE-FLIGHT CHECK SCRIPT
Verifies everything is ready before starting the trading bot

Run this BEFORE starting the bot to catch any issues early!
"""

import os
import sys
import importlib.util

print("""
╔════════════════════════════════════════════════════════╗
║         TRADING BOT PRE-FLIGHT CHECK                  ║
║   This will verify everything is ready to run         ║
╚════════════════════════════════════════════════════════╝
""")

errors = []
warnings = []
success = []

# ============================================
# CHECK 1: Python Version
# ============================================
print("\n[1/10] Checking Python version...")
if sys.version_info < (3, 8):
    errors.append("Python 3.8+ required. You have: " + sys.version)
else:
    success.append(f"✓ Python version OK: {sys.version.split()[0]}")

# ============================================
# CHECK 2: Required Python Packages
# ============================================
print("[2/10] Checking Python packages...")
required_packages = [
    'flask',
    'pandas',
    'numpy',
    'sklearn',
    'xgboost',
    'lightgbm',
    'alpaca_trade_api',
    'pandas_ta',
    'joblib',
    'schedule',
    'matplotlib'
]

missing_packages = []
for package in required_packages:
    if package == 'sklearn':
        package_name = 'scikit-learn'
        import_name = 'sklearn'
    elif package == 'alpaca_trade_api':
        package_name = 'alpaca-trade-api'
        import_name = 'alpaca_trade_api'
    else:
        package_name = package
        import_name = package
    
    try:
        __import__(import_name)
        success.append(f"  ✓ {package_name}")
    except ImportError:
        missing_packages.append(package_name)
        errors.append(f"  ✗ Missing package: {package_name}")

if missing_packages:
    print("\n  Missing packages detected. Install with:")
    print(f"  pip3 install --break-system-packages {' '.join(missing_packages)}")

# ============================================
# CHECK 3: Project Files
# ============================================
print("[3/10] Checking project files...")
required_files = [
    'config.py',
    'predictor.py',
    'predictor_hourly.py',
    'strategies.py',
    'strategies_hourly.py',
    'pi_dashboard.py',
    'bot_runner.py',
    'templates/dashboard.html'
]

for file in required_files:
    if os.path.exists(file):
        success.append(f"  ✓ {file}")
    else:
        errors.append(f"  ✗ Missing file: {file}")

# ============================================
# CHECK 4: Configuration File
# ============================================
print("[4/10] Checking configuration...")
try:
    import config
    
    # Check API keys
    if not hasattr(config, 'API_KEY') or not config.API_KEY:
        errors.append("  ✗ API_KEY not set in config.py")
    elif config.API_KEY == "YOUR_API_KEY_HERE":
        errors.append("  ✗ API_KEY not updated in config.py")
    else:
        success.append("  ✓ API_KEY configured")
    
    if not hasattr(config, 'SECRET_KEY') or not config.SECRET_KEY:
        errors.append("  ✗ SECRET_KEY not set in config.py")
    elif config.SECRET_KEY == "YOUR_SECRET_KEY_HERE":
        errors.append("  ✗ SECRET_KEY not updated in config.py")
    else:
        success.append("  ✓ SECRET_KEY configured")
    
    # Check directories
    if hasattr(config, 'STOCKS_DIR'):
        success.append(f"  ✓ STOCKS_DIR: {config.STOCKS_DIR}")
    else:
        warnings.append("  ! STOCKS_DIR not defined, using default 'stocks/'")
        
except ImportError as e:
    errors.append(f"  ✗ Cannot import config.py: {e}")

# ============================================
# CHECK 5: Alpaca API Connection
# ============================================
print("[5/10] Testing Alpaca API connection...")
try:
    import config
    import alpaca_trade_api as tradeapi
    
    api = tradeapi.REST(
        config.API_KEY,
        config.SECRET_KEY,
        base_url="https://paper-api.alpaca.markets"
    )
    
    # Test connection
    account = api.get_account()
    success.append(f"  ✓ Alpaca API connected")
    success.append(f"  ✓ Account status: {account.status}")
    success.append(f"  ✓ Buying power: ${float(account.buying_power):.2f}")
    
    if account.status != 'ACTIVE':
        warnings.append(f"  ! Account status is '{account.status}' (not ACTIVE)")
        
except Exception as e:
    errors.append(f"  ✗ Alpaca API connection failed: {e}")

# ============================================
# CHECK 6: Model Files
# ============================================
print("[6/10] Checking trained models...")
TICKER = "NVDA"  # Default ticker

# Check if STOCKS_DIR exists
try:
    stocks_dir = config.STOCKS_DIR if hasattr(config, 'STOCKS_DIR') else 'stocks'
except:
    stocks_dir = 'stocks'

daily_model_path = os.path.join(stocks_dir, TICKER, 'daily', 'daily_model.joblib')
daily_params_path = os.path.join(stocks_dir, TICKER, 'daily', 'daily_params.json')
daily_settings_path = os.path.join(stocks_dir, TICKER, 'daily', 'daily_settings.json')
daily_config_path = os.path.join(stocks_dir, TICKER, 'daily', 'training_config.json')

hourly_model_path = os.path.join(stocks_dir, TICKER, 'hourly', 'hourly_model.joblib')
hourly_params_path = os.path.join(stocks_dir, TICKER, 'hourly', 'hourly_params.json')
hourly_settings_path = os.path.join(stocks_dir, TICKER, 'hourly', 'hourly_settings.json')
hourly_config_path = os.path.join(stocks_dir, TICKER, 'hourly', 'training_config.json')

model_files = {
    'Daily Model': daily_model_path,
    'Daily Params': daily_params_path,
    'Daily Settings': daily_settings_path,
    'Daily Config': daily_config_path,
    'Hourly Model': hourly_model_path,
    'Hourly Params': hourly_params_path,
    'Hourly Settings': hourly_settings_path,
    'Hourly Config': hourly_config_path
}

missing_models = False
for name, path in model_files.items():
    if os.path.exists(path):
        success.append(f"  ✓ {name}: {path}")
    else:
        errors.append(f"  ✗ Missing: {name} at {path}")
        missing_models = True

if missing_models:
    print("\n  To transfer models from your laptop:")
    print(f"  scp -r stocks/{TICKER}/ pi@YOUR_PI_IP:/home/pi/trading_bot/stocks/")

# ============================================
# CHECK 7: Database
# ============================================
print("[7/10] Checking database...")
try:
    import sqlite3
    from dashboard.app import init_database
    
    init_database()
    
    # Test database connection
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    
    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    required_tables = ['predictions', 'trades', 'portfolio_summary', 'events']
    for table in required_tables:
        if table in tables:
            success.append(f"  ✓ Table '{table}' exists")
        else:
            errors.append(f"  ✗ Table '{table}' missing")
    
    conn.close()
    
except Exception as e:
    errors.append(f"  ✗ Database check failed: {e}")

# ============================================
# CHECK 8: Disk Space
# ============================================
print("[8/10] Checking disk space...")
try:
    import shutil
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2**30)
    
    if free_gb < 1:
        errors.append(f"  ✗ Low disk space: {free_gb}GB free")
    elif free_gb < 5:
        warnings.append(f"  ! Low disk space: {free_gb}GB free (recommend 5GB+)")
    else:
        success.append(f"  ✓ Disk space OK: {free_gb}GB free")
        
except Exception as e:
    warnings.append(f"  ! Could not check disk space: {e}")

# ============================================
# CHECK 9: Network Connectivity
# ============================================
print("[9/10] Checking network connectivity...")
try:
    import urllib.request
    urllib.request.urlopen('https://api.alpaca.markets', timeout=5)
    success.append("  ✓ Internet connection OK")
except Exception as e:
    errors.append(f"  ✗ No internet connection: {e}")

# ============================================
# CHECK 10: Timezone Settings
# ============================================
print("[10/10] Checking timezone settings...")
try:
    from datetime import datetime
    import time
    
    # Get system timezone
    is_dst = time.daylight and time.localtime().tm_isdst > 0
    utc_offset = - (time.altzone if is_dst else time.timezone)
    
    now = datetime.now()
    success.append(f"  ✓ Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    success.append(f"  ✓ UTC offset: {utc_offset / 3600:.1f} hours")
    
    if utc_offset == 0:
        success.append("  ✓ Timezone: GMT (UK winter time)")
    elif utc_offset == 3600:
        success.append("  ✓ Timezone: BST (UK summer time)")
    else:
        warnings.append(f"  ! Timezone may not be set to UK time")
        
except Exception as e:
    warnings.append(f"  ! Could not check timezone: {e}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*60)
print("                    SUMMARY")
print("="*60)

if success:
    print(f"\n✓ PASSED CHECKS ({len(success)}):")
    for item in success:
        print(f"  {item}")

if warnings:
    print(f"\n! WARNINGS ({len(warnings)}):")
    for item in warnings:
        print(f"  {item}")

if errors:
    print(f"\n✗ ERRORS ({len(errors)}):")
    for item in errors:
        print(f"  {item}")

print("\n" + "="*60)

if errors:
    print("\n❌ PRE-FLIGHT CHECK FAILED")
    print("Please fix the errors above before starting the bot.")
    sys.exit(1)
elif warnings:
    print("\n⚠️  PRE-FLIGHT CHECK PASSED WITH WARNINGS")
    print("Bot can run, but please review warnings above.")
    sys.exit(0)
else:
    print("\n✅ PRE-FLIGHT CHECK PASSED")
    print("\nYour bot is ready to run!")
    print("\nNext steps:")
    print("  1. Start dashboard: python3 pi_dashboard.py &")
    print("  2. Start bot: python3 bot_runner.py")
    print("  3. Access dashboard at: http://YOUR_PI_IP:5000")
    sys.exit(0)