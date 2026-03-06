"""
Model Manager Module v3.3
Enhanced Features:
- Meta-AI training status tracking
- Improved performance viewing
- Better top 10 management
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config
from control_center.utils import (
    print_success, print_error, print_warning, print_info,
    print_table, get_date_string, get_timestamp
)


class ModelManager:
    """Manages model tracking and deployment"""
    
    def __init__(self):
        self.control_center_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "control_center"
        )
        self.master_config_path = os.path.join(self.control_center_dir, "master_config.json")
        self.stocks_dir = config.STOCKS_DIR
        
        # Initialize master config if it doesn't exist
        if not os.path.exists(self.master_config_path):
            self._init_master_config()
    
    # ==========================================
    # MASTER CONFIG MANAGEMENT
    # ==========================================
    
    def _init_master_config(self):
        """Initialize master config file"""
        os.makedirs(self.control_center_dir, exist_ok=True)
        
        master_config = {
            "created": get_timestamp(),
            "last_updated": get_timestamp(),
            "version": "3.3",
            "stocks": {},
            "top_10": [],
            "meta_training_status": {},
            "deployment_history": [],
            "system_info": {
                "total_stocks": 30,
                "stocks_downloaded": 0,
                "stocks_trained": 0,
                "last_backtest_date": None
            }
        }
        
        self._save_master_config(master_config)
        print_success(f"Initialized master config at: {self.master_config_path}")
    
    def _load_master_config(self):
        """Load master config"""
        try:
            with open(self.master_config_path, 'r') as f:
                config_data = json.load(f)
            
            # Ensure meta_training_status exists (backward compatibility)
            if "meta_training_status" not in config_data:
                config_data["meta_training_status"] = {}
            
            return config_data
        except Exception as e:
            print_error(f"Failed to load master config: {e}")
            return None
    
    def _save_master_config(self, config_data):
        """Save master config"""
        try:
            config_data["last_updated"] = get_timestamp()
            with open(self.master_config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
        except Exception as e:
            print_error(f"Failed to save master config: {e}")
    
    def update_stock_info(self, ticker, info):
        """Update information for a specific stock"""
        config_data = self._load_master_config()
        if config_data is None:
            return
        
        if ticker not in config_data["stocks"]:
            config_data["stocks"][ticker] = {}
        
        config_data["stocks"][ticker].update(info)
        self._save_master_config(config_data)
    
    # ==========================================
    # SYSTEM STATUS
    # ==========================================
    
    def get_training_status(self):
        """Get training status for all stocks
        
        Returns:
            dict: Dictionary of stock ticker -> training info
        """
        config_data = self._load_master_config()
        if config_data is None:
            return {}
        
        return config_data.get("stocks", {})
    
    def sync_meta_ai_status(self):
        """Scan filesystem and sync Meta-AI training status with master_config
        
        This automatically detects trained Meta-AIs that aren't registered
        in master_config.json and adds them.
        """
        config_data = self._load_master_config()
        if config_data is None:
            return
        
        if "meta_training_status" not in config_data:
            config_data["meta_training_status"] = {}
        
        synced_count = 0
        
        # Scan for trained Meta-AI models
        for ticker in config.TICKERS:
            meta_dir = os.path.join(self.stocks_dir, ticker, "meta")
            
            if os.path.exists(meta_dir):
                # Check for model files
                final_model = os.path.join(meta_dir, "meta_ai_final.zip")
                best_model = os.path.join(meta_dir, "best_model.zip")
                
                if os.path.exists(final_model) or os.path.exists(best_model):
                    # Found a trained model
                    model_file = final_model if os.path.exists(final_model) else best_model
                    
                    # Check if already registered
                    if ticker not in config_data["meta_training_status"] or \
                       not config_data["meta_training_status"].get(ticker, {}).get("trained", False):
                        
                        # Register this model
                        last_modified = os.path.getmtime(model_file)
                        last_trained = datetime.fromtimestamp(last_modified).strftime("%Y-%m-%d %H:%M:%S")
                        
                        config_data["meta_training_status"][ticker] = {
                            "trained": True,
                            "last_trained": last_trained
                        }
                        
                        # Try to load existing metrics if available
                        metrics_file = os.path.join(meta_dir, "forward_test_metrics.json")
                        if os.path.exists(metrics_file):
                            try:
                                with open(metrics_file, 'r') as f:
                                    metrics = json.load(f)
                                    config_data["meta_training_status"][ticker]["metrics"] = {
                                        "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                                        "total_return_pct": metrics.get("total_return_pct", 0),
                                        "max_drawdown_pct": metrics.get("max_drawdown_pct", 0),
                                        "total_trades": metrics.get("total_trades", 0)
                                    }
                            except:
                                pass
                        
                        synced_count += 1
        
        if synced_count > 0:
            self._save_master_config(config_data)
    
    def get_meta_training_status(self):
        """Get Meta-AI training status from master config
        
        Auto-syncs with filesystem before returning status.
        
        Returns:
            dict: Dictionary of ticker -> training status
        """
        # Auto-sync before returning status
        self.sync_meta_ai_status()
        
        config_data = self._load_master_config()
        if config_data is None:
            return {}
        
        return config_data.get("meta_training_status", {})
    
    def get_system_status(self):
        """Get current system status - auto-calculated"""
        config_data = self._load_master_config()
        if config_data is None:
            return {
                "stocks_downloaded": 0,
                "stocks_trained": 0,
                "meta_ais_trained": 0,
                "top_10_selected": False,
                "last_updated": "Never"
            }
        
        # Quick count: Check if any stock data files exist (sampling approach)
        sample_tickers = config.TICKERS[:5]  # Just check first 5 stocks
        daily_dir = config.DATA_DIR
        hourly_dir = os.path.join(config.DATA_DIR, "hourly")
        
        sample_downloaded = sum(
            1 for ticker in sample_tickers
            if os.path.exists(os.path.join(daily_dir, f"{ticker}.csv")) and
               os.path.exists(os.path.join(hourly_dir, f"{ticker}.csv"))
        )
        
        # Estimate total (if most samples exist, assume all exist)
        if sample_downloaded >= 4:
            stocks_downloaded = len(config.TICKERS)
        elif sample_downloaded >= 2:
            stocks_downloaded = int(len(config.TICKERS) * 0.6)
        else:
            stocks_downloaded = sample_downloaded * 6  # Rough extrapolation
        
        # Count trained stocks by checking actual model files
        stocks_trained = 0
        for ticker in config.TICKERS:
            daily_model = os.path.join(self.stocks_dir, ticker, "daily", "daily_model.joblib")
            hourly_model = os.path.join(self.stocks_dir, ticker, "hourly", "hourly_model.joblib")
            if os.path.exists(daily_model) and os.path.exists(hourly_model):
                stocks_trained += 1
        
        # Count trained Meta-AIs from meta_training_status
        meta_training_status = config_data.get("meta_training_status", {})
        meta_ais_trained = sum(
            1 for status in meta_training_status.values()
            if status.get("trained", False)
        )
        
        # Get top 10 info
        top_10_list = config_data.get("top_10", [])
        top_10_selected = len(top_10_list) == 10
        
        return {
            "stocks_downloaded": stocks_downloaded,
            "stocks_trained": stocks_trained,
            "meta_ais_trained": meta_ais_trained,
            "top_10_count": len(top_10_list),
            "top_10_selected": top_10_selected,
            "last_updated": config_data.get("last_updated", "Never")
        }
    
    def _check_stock_trained(self, ticker):
        """Check if a stock has trained models"""
        daily_model = os.path.join(self.stocks_dir, ticker, "daily", "daily_model.joblib")
        hourly_model = os.path.join(self.stocks_dir, ticker, "hourly", "hourly_model.joblib")
        
        return os.path.exists(daily_model) and os.path.exists(hourly_model)
    
    # ==========================================
    # TOP 10 MANAGEMENT
    # ==========================================
    
    def save_top_10(self, top_10_list, method="Manual Selection"):
        """Save top 10 selection to master config
        
        Args:
            top_10_list: List of 10 ticker symbols
            method: Selection method description
        """
        config_data = self._load_master_config()
        if config_data is None:
            config_data = {}
        
        config_data["top_10"] = top_10_list
        config_data["top_10_selected_date"] = get_timestamp()
        config_data["selection_method"] = method
        
        self._save_master_config(config_data)
    
    # ==========================================
    # RANKINGS
    # ==========================================
    
    def show_rankings(self):
        """Show stock performance rankings from V3.3 backtest"""
        config_data = self._load_master_config()
        if config_data is None:
            print_error("Could not load rankings")
            return
        
        stocks_with_results = []
        for ticker, info in config_data.get("stocks", {}).items():
            if "backtest_results" in info and info["backtest_results"]:
                results = info["backtest_results"]
                stocks_with_results.append({
                    "ticker": ticker,
                    "sharpe": results.get("sharpe", 0),
                    "return": results.get("return", 0),
                    "max_drawdown": results.get("max_drawdown", 0),
                    "win_rate": results.get("win_rate", 0)
                })
        
        if not stocks_with_results:
            print_warning("No backtest results available yet.")
            print_info("Run 'Backtest All Stocks' first to generate rankings.")
            return
        
        # Sort by Sharpe Ratio (descending)
        stocks_with_results.sort(key=lambda x: x["sharpe"], reverse=True)
        
        # Add rank
        for i, stock in enumerate(stocks_with_results, 1):
            stock["rank"] = i
        
        print("\n🏆 PERFORMANCE RANKINGS (by Sharpe Ratio):")
        print("="*80)
        
        headers = ["Rank", "Ticker", "Sharpe", "Return %", "Max DD %", "Win Rate"]
        rows = []
        
        for stock in stocks_with_results:
            rows.append([
                stock["rank"],
                stock["ticker"],
                f"{stock['sharpe']:.2f}",
                f"{stock['return']:.2f}",
                f"{stock['max_drawdown']:.2f}",
                f"{stock['win_rate']:.2%}"
            ])
        
        print_table(headers, rows, column_widths=[8, 10, 10, 12, 12, 12])
        
        # Highlight top 10
        if len(stocks_with_results) >= 10:
            print(f"\n⭐ TOP 10 STOCKS:")
            top_10_tickers = [s["ticker"] for s in stocks_with_results[:10]]
            print(f"   {', '.join(top_10_tickers)}")
    
    def show_meta_ai_rankings(self):
        """Show Meta-AI performance rankings from master config"""
        config_data = self._load_master_config()
        if config_data is None:
            print_error("Could not load master config")
            return
        
        meta_status = config_data.get("meta_training_status", {})
        
        if not meta_status:
            print_warning("No Meta-AI training data found yet.")
            print_info("Train Meta-AIs first using the training options.")
            return
        
        # Get all trained Meta-AIs with metrics
        results = []
        for ticker, status in meta_status.items():
            if status.get("trained", False) and "metrics" in status:
                metrics = status["metrics"]
                results.append({
                    'ticker': ticker,
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'total_return_pct': metrics.get('total_return_pct', 0),
                    'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
                    'total_trades': metrics.get('total_trades', 0),
                    'last_trained': status.get('last_trained', 'Unknown')
                })
        
        if not results:
            print_warning("No Meta-AI results available yet")
            print_info("Trained Meta-AIs will appear here once evaluation is complete")
            return
        
        # Sort by Sharpe Ratio
        results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        
        print("\n🏆 META-AI PERFORMANCE RANKINGS (by Sharpe Ratio):")
        print("="*80)
        
        headers = ["Rank", "Ticker", "Return %", "Sharpe", "Max DD %", "Trades", "Last Trained"]
        rows = []
        
        for rank, result in enumerate(results, 1):
            rows.append([
                rank,
                result['ticker'],
                f"{result['total_return_pct']:.2f}",
                f"{result['sharpe_ratio']:.2f}",
                f"{result['max_drawdown_pct']:.2f}",
                result['total_trades'],
                result['last_trained'][:10]  # Just the date
            ])
        
        print_table(headers, rows, column_widths=[6, 8, 10, 8, 10, 8, 12])
        
        # Show summary
        trained_count = len(results)
        top_10 = config_data.get("top_10", [])
        
        print(f"\n📊 Summary:")
        print(f"   Total Meta-AIs Trained: {trained_count}")
        if top_10:
            top_10_trained = sum(1 for ticker in top_10 if ticker in meta_status and meta_status[ticker].get("trained", False))
            print(f"   Top 10 Progress: {top_10_trained}/{len(top_10)} trained")
        
        # Show best performing
        if results:
            best = results[0]
            print(f"\n🏆 Best Performer: {best['ticker']}")
            print(f"   Sharpe Ratio: {best['sharpe_ratio']:.2f}")
            print(f"   Total Return: {best['total_return_pct']:.2f}%")
    
    # ==========================================
    # HISTORY & HEALTH
    # ==========================================
    
    def show_training_history(self):
        """Show training history"""
        config_data = self._load_master_config()
        if config_data is None:
            return
        
        print("\n📜 TRAINING HISTORY:")
        print(f"   System created: {config_data.get('created', 'Unknown')}")
        print(f"   Last updated: {config_data.get('last_updated', 'Unknown')}")
        print(f"   Total stocks tracked: {len(config_data.get('stocks', {}))}")
        
        # Meta-AI training history
        meta_status = config_data.get("meta_training_status", {})
        trained_meta_ais = sum(1 for s in meta_status.values() if s.get("trained", False))
        
        print(f"\n🤖 Meta-AI Status:")
        print(f"   Total Meta-AIs trained: {trained_meta_ais}")
        
        if trained_meta_ais > 0:
            print(f"\n   Recently trained Meta-AIs:")
            # Get last 5 trained
            trained_list = [(ticker, status) for ticker, status in meta_status.items() 
                          if status.get("trained", False)]
            trained_list.sort(key=lambda x: x[1].get("last_trained", ""), reverse=True)
            
            for ticker, status in trained_list[:5]:
                print(f"   • {ticker:6s} - {status.get('last_trained', 'Unknown')}")
        
        if config_data.get("deployment_history"):
            print(f"\n📦 Recent Deployments:")
            for deployment in config_data["deployment_history"][-5:]:
                print(f"   • {deployment}")
        else:
            print(f"\n   No deployments yet")
    
    def run_health_check(self):
        """Run system health check"""
        print("\n⚙️  SYSTEM HEALTH CHECK:")
        print("="*70)
        
        checks_passed = 0
        total_checks = 5
        
        # Check 1: Master config exists
        if os.path.exists(self.master_config_path):
            print_success("Master config exists")
            checks_passed += 1
        else:
            print_error("Master config missing")
        
        # Check 2: Data directories exist
        if os.path.exists(self.stocks_dir):
            print_success("Stocks directory exists")
            checks_passed += 1
        else:
            print_error("Stocks directory missing")
        
        # Check 3: Check API connection
        try:
            import alpaca_trade_api as tradeapi
            api = tradeapi.REST(
                config.API_KEY,
                config.SECRET_KEY,
                base_url="https://paper-api.alpaca.markets"
            )
            account = api.get_account()
            print_success(f"Alpaca API connected (Status: {account.status})")
            checks_passed += 1
        except Exception as e:
            print_error(f"Alpaca API connection failed: {e}")
        
        # Check 4: Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            free_gb = free // (2**30)
            if free_gb > 5:
                print_success(f"Disk space OK ({free_gb}GB free)")
                checks_passed += 1
            else:
                print_warning(f"Low disk space ({free_gb}GB free)")
        except:
            print_warning("Could not check disk space")
        
        # Check 5: Check required packages
        try:
            import xgboost
            import pandas_ta
            import sklearn
            print_success("Required packages installed")
            checks_passed += 1
        except ImportError as e:
            print_error(f"Missing required packages: {e}")
        
        print("\n" + "="*70)
        print(f"\n📊 Health Score: {checks_passed}/{total_checks} checks passed")
        
        if checks_passed == total_checks:
            print_success("System healthy! ✅")
        elif checks_passed >= 3:
            print_warning("System functional with minor issues ⚠️")
        else:
            print_error("System has critical issues ❌")
    
    def clear_all_caches(self):
        """Clear all prediction caches"""
        cache_count = 0
        
        # Find all .pkl cache files
        for root, dirs, files in os.walk(self.stocks_dir):
            for file in files:
                if file.endswith('.pkl') and 'predictions_cache' in file:
                    cache_path = os.path.join(root, file)
                    try:
                        os.remove(cache_path)
                        cache_count += 1
                    except Exception as e:
                        print_error(f"Failed to delete {cache_path}: {e}")
        
        if cache_count > 0:
            print_success(f"Cleared {cache_count} cache files")
        else:
            print_info("No cache files found")