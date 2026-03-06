#!/usr/bin/env python3
"""
CONTROL CENTER v3.3 - AI Trading System Orchestration
Enhanced Features:
- Selective Meta-AI training
- Training status tracking
- Improved stock selection workflow
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_center.stock_analyzer import StockAnalyzer
from control_center.model_manager import ModelManager
from control_center.horizon_optimizer import HorizonOptimizer
from control_center.utils import (
    print_header,
    print_section,
    print_success,
    print_error,
    print_warning,
    print_info,
    get_user_choice,
    clear_screen,
    wait_for_enter,
    confirm_action,
    Colors
)


class ControlCenter:
    """Main Control Center orchestrator"""
    
    def __init__(self):
        self.analyzer = StockAnalyzer()
        self.model_manager = ModelManager()
        self.horizon_optimizer = HorizonOptimizer()
        self.running = True
        self.selected_top_10 = self._load_selected_top_10()
        
    def _load_selected_top_10(self):
        """Load the currently selected top 10 stocks from master config"""
        try:
            with open(os.path.join("control_center", "master_config.json"), 'r') as f:
                config = json.load(f)
                return config.get("top_10", [])
        except:
            return []
    
    def show_menu(self):
        """Display main menu"""
        clear_screen()
        print_header("🎯 AI TRADING SYSTEM - CONTROL CENTER v3.3")
        
        # Show system status
        status = self.model_manager.get_system_status()
        meta_status = self.model_manager.get_meta_training_status()
        
        print("\n📊 SYSTEM STATUS:")
        print(f"  Stocks Downloaded: {status['stocks_downloaded']}/30")
        print(f"  Stocks Trained (XGBoost): {status['stocks_trained']}/30")
        
        # Show Meta-AI training status
        if self.selected_top_10:
            trained_count = sum(1 for ticker in self.selected_top_10 
                              if meta_status.get(ticker, {}).get("trained", False))
            print(f"  Meta-AIs Trained: {trained_count}/{len(self.selected_top_10)} (from selected Top 10)")
            print(f"  Top 10 Selected: {', '.join(self.selected_top_10[:5])}{'...' if len(self.selected_top_10) > 5 else ''}")
        else:
            print(f"  Meta-AIs Trained: 0/0 (No Top 10 selected yet)")
            print(f"  Top 10 Selected: No")
        
        print(f"  Last Updated: {status['last_updated']}")
        
        print("\n" + "="*70)
        print("MAIN MENU:")
        print("="*70)
        print("\n📥 DATA MANAGEMENT:")
        print("  [1] Download All Dow 30 Stocks (Daily + Hourly)")
        print("  [2] View Downloaded Stocks")
        
        print("\n🔬 TRAINING & OPTIMIZATION:")
        print("  [3] Setup Prediction Horizons (Quick or Advanced)")
        print("  [4] Optimize Strategy Parameters (All Stocks)")
        print("  [5] Train All AIs (Daily + Hourly)")
        print("  [6] Optimize Confidence Thresholds (All Stocks)")
        
        print("\n📈 BACKTESTING & ANALYSIS:")
        print("  [7] Backtest All Stocks (V3.3 - Optional)")
        print("  [8] View V3.3 Performance Rankings")
        
        print("\n🎯 TOP 10 SELECTION:")
        print("  [9] Select My Top 10 Stocks (Manual)")
        print("  [10] View Current Top 10 Selection")
        
        print("\n🤖 META-AI TRAINING:")
        print("  [11] Train Single Meta-AI (Test)")
        print("  [12] Train Multiple Meta-AIs (Select from Top 10)")
        print("  [13] Train All Untrained Meta-AIs (from Top 10)")
        print("  [14] Evaluate Trained Meta-AI (Run Forward Test)")
        print("  [15] View Meta-AI Performance & Status")
        
        print("\n🔄 MODEL MANAGEMENT:")
        print("  [16] Compare Models (Old vs New)")
        print("  [17] View Training History")
        
        print("\n🚀 DEPLOYMENT:")
        print("  [18] Deploy to Raspberry Pi")
        
        print("\n⚙️  SYSTEM:")
        print("  [19] System Health Check")
        print("  [20] Clear Cache")
        print("  [0] Exit")
        
        print("\n" + "="*70)
        
    def run(self):
        """Main control loop"""
        while self.running:
            self.show_menu()
            choice = get_user_choice("\nSelect option: ", valid_range=range(0, 21))
            
            if choice == 0:
                self.exit_program()
            elif choice == 1:
                self.download_all_stocks()
            elif choice == 2:
                self.view_downloaded_stocks()
            elif choice == 3:
                self.find_optimal_horizons()
            elif choice == 4:
                self.optimize_strategy_params()
            elif choice == 5:
                self.train_all_ais()
            elif choice == 6:
                self.optimize_confidence_thresholds()
            elif choice == 7:
                self.backtest_all_stocks()
            elif choice == 8:
                self.view_rankings()
            elif choice == 9:
                self.select_my_top_10()
            elif choice == 10:
                self.view_current_top_10()
            elif choice == 11:
                self.train_single_meta_ai()
            elif choice == 12:
                self.train_multiple_meta_ais()
            elif choice == 13:
                self.train_all_untrained_meta_ais()
            elif choice == 14:
                self.evaluate_meta_ai()
            elif choice == 15:
                self.view_meta_ai_performance()
            elif choice == 16:
                self.compare_models()
            elif choice == 17:
                self.view_history()
            elif choice == 18:
                self.deploy_to_pi()
            elif choice == 19:
                self.system_health_check()
            elif choice == 20:
                self.clear_cache()
    
    # ==========================================
    # DATA MANAGEMENT
    # ==========================================
    
    def download_all_stocks(self):
        """Download data for all Dow 30 stocks"""
        clear_screen()
        print_header("📥 DOWNLOAD ALL DOW 30 STOCKS")
        
        print("\n📋 This will download:")
        print("   • Daily data (2010-present)")
        print("   • Hourly data (2016-present)")
        print("   • For all 30 Dow Jones stocks")
        
        if not confirm_action("Proceed with download?"):
            print_warning("Download cancelled.")
            wait_for_enter()
            return
        
        force = confirm_action("Force re-download of ALL stocks (overwrite existing)?")
        
        self.analyzer.download_all_stocks(force=force)
        
        print_success("All stocks downloaded!")
        wait_for_enter()
    
    def view_downloaded_stocks(self):
        """View which stocks have been downloaded"""
        clear_screen()
        print_header("📊 DOWNLOADED STOCKS STATUS")
        
        self.analyzer.show_download_status()
        
        wait_for_enter()
    
    # ==========================================
    # TRAINING & OPTIMIZATION
    # ==========================================
    
    def find_optimal_horizons(self):
        """Find optimal prediction horizons"""
        clear_screen()
        print_header("🔬 SETUP PREDICTION HORIZONS")
        
        print("\n📋 Choose optimization mode:")
        print("  [1] Quick Setup (Use default horizons)")
        print("  [2] Advanced (Optimize for each stock)")
        
        choice = get_user_choice("\nSelect option: ", range(1, 3))
        
        if choice == 1:
            print_info("Using default horizons: Daily=5, Hourly=3")
            self.horizon_optimizer.use_defaults()
        else:
            self.horizon_optimizer.optimize_all()
        
        print_success("Horizons configured!")
        wait_for_enter()
    
    def optimize_strategy_params(self):
        """Optimize strategy parameters for all stocks"""
        clear_screen()
        print_header("⚙️  OPTIMIZE STRATEGY PARAMETERS")
        
        print("\n📋 This will optimize:")
        print("   • Bollinger Bands parameters")
        print("   • MACD parameters")
        print("   • RSI thresholds")
        print("   • For all 30 stocks")
        
        if not confirm_action("Proceed with optimization?"):
            print_warning("Optimization cancelled.")
            wait_for_enter()
            return
        
        print_info("Running parameter optimization...")
        # Call your optimizer here
        
        print_success("Parameters optimized!")
        wait_for_enter()
    
    def train_all_ais(self):
        """Train daily and hourly AIs for all stocks"""
        clear_screen()
        print_header("🎓 TRAIN ALL AIs")
        
        print("\n📋 This will train:")
        print("   • Daily AI (XGBoost)")
        print("   • Hourly AI (XGBoost)")
        print("   • For all 30 Dow stocks")
        print("   • Using optimized horizons and parameters")
        
        if not confirm_action("Proceed with training?"):
            print_warning("Training cancelled.")
            wait_for_enter()
            return
        
        print_info("Training AIs...")
        # Call your training code here
        
        print_success("All AIs trained!")
        wait_for_enter()
    
    def optimize_confidence_thresholds(self):
        """Optimize confidence thresholds"""
        clear_screen()
        print_header("🎯 OPTIMIZE CONFIDENCE THRESHOLDS")
        
        print("\n📋 This will find optimal:")
        print("   • Buy confidence threshold")
        print("   • Sell confidence threshold")
        print("   • For each stock individually")
        
        if not confirm_action("Proceed with optimization?"):
            print_warning("Optimization cancelled.")
            wait_for_enter()
            return
        
        print_info("Optimizing thresholds...")
        # Call your optimizer here
        
        print_success("Thresholds optimized!")
        wait_for_enter()
    
    # ==========================================
    # BACKTESTING & ANALYSIS
    # ==========================================
    
    def backtest_all_stocks(self):
        """Backtest all stocks using V3.3"""
        clear_screen()
        print_header("📈 BACKTEST ALL STOCKS (V3.3)")
        
        print("\n📋 This will run backtests on:")
        print("   • Test Period: 2025-01-01 to 2025-12-31")
        print("   • All 30 Dow Jones stocks")
        print("   • Using trained models")
        
        if not confirm_action("Proceed with backtesting?"):
            print_warning("Backtesting cancelled.")
            wait_for_enter()
            return
        
        # Execute backtesting
        self.analyzer.backtest_all_stocks()
        
        wait_for_enter()
    
    def view_rankings(self):
        """Show stock rankings by Sharpe Ratio"""
        clear_screen()
        print_header("🏆 STOCK PERFORMANCE RANKINGS (V3.3 BACKTEST)")
        
        self.model_manager.show_rankings()
        
        wait_for_enter()
    
    # ==========================================
    # TOP 10 SELECTION
    # ==========================================
    
    def select_my_top_10(self):
        """Manually select top 10 stocks"""
        clear_screen()
        print_header("🎯 SELECT MY TOP 10 STOCKS")
        
        print("\n📋 This allows you to manually select your top 10 stocks for Meta-AI training.")
        print("   You can choose based on:")
        print("   • V3.3 backtest performance")
        print("   • Company fundamentals")
        print("   • Your research and preferences")
        
        # Show available stocks with their v3.3 performance
        print("\n" + "="*70)
        print("AVAILABLE STOCKS (sorted by V3.3 Sharpe Ratio):")
        print("="*70)
        
        self.model_manager.show_rankings()
        
        print("\n" + "="*70)
        print("\n📝 Enter your top 10 stock picks:")
        print("   Format: Comma-separated tickers (e.g., AAPL,MSFT,GOOGL,...)")
        print("   Type 'cancel' to abort")
        
        user_input = input("\nYour top 10: ").strip()
        
        if user_input.lower() == 'cancel':
            print_warning("Selection cancelled.")
            wait_for_enter()
            return
        
        # Parse input
        selected = [ticker.strip().upper() for ticker in user_input.split(',')]
        selected = [t for t in selected if t]  # Remove empty strings
        
        if len(selected) != 10:
            print_error(f"You entered {len(selected)} stocks. Please enter exactly 10.")
            wait_for_enter()
            return
        
        # Validate tickers
        from control_center.stock_analyzer import StockAnalyzer
        analyzer = StockAnalyzer()
        invalid = [t for t in selected if t not in analyzer.DOW_30_TICKERS]
        
        if invalid:
            print_error(f"Invalid tickers: {', '.join(invalid)}")
            print_info(f"Valid tickers: {', '.join(analyzer.DOW_30_TICKERS)}")
            wait_for_enter()
            return
        
        # Confirm selection
        print(f"\n✅ Your selection: {', '.join(selected)}")
        
        if not confirm_action("Save this as your top 10?"):
            print_warning("Selection cancelled.")
            wait_for_enter()
            return
        
        # Save to master config
        self.model_manager.save_top_10(selected, method="Manual Selection")
        self.selected_top_10 = selected
        
        print_success("Top 10 saved successfully!")
        print(f"\n📁 Saved to: control_center/master_config.json")
        print(f"\n⭐ Your Top 10: {', '.join(selected)}")
        
        wait_for_enter()
    
    def view_current_top_10(self):
        """View currently selected top 10"""
        clear_screen()
        print_header("⭐ CURRENT TOP 10 SELECTION")
        
        if not self.selected_top_10:
            print_warning("No top 10 stocks selected yet.")
            print_info("Use option [9] to select your top 10 stocks.")
            wait_for_enter()
            return
        
        # Load config to get selection date and method
        try:
            with open(os.path.join("control_center", "master_config.json"), 'r') as f:
                config = json.load(f)
            
            selection_date = config.get("top_10_selected_date", "Unknown")
            selection_method = config.get("selection_method", "Unknown")
        except:
            selection_date = "Unknown"
            selection_method = "Unknown"
        
        print(f"\n📅 Selection Date: {selection_date}")
        print(f"📋 Selection Method: {selection_method}")
        print(f"\n⭐ Your Top 10 Stocks:")
        
        for i, ticker in enumerate(self.selected_top_10, 1):
            print(f"   {i:2d}. {ticker}")
        
        # Show training status for each
        print(f"\n🤖 Meta-AI Training Status:")
        meta_status = self.model_manager.get_meta_training_status()
        
        for ticker in self.selected_top_10:
            status = meta_status.get(ticker, {})
            if status.get("trained", False):
                metrics = status.get("metrics", {})
                sharpe = metrics.get("sharpe_ratio", 0)
                return_pct = metrics.get("total_return_pct", 0)
                print(f"   ✅ {ticker:6s} - Trained (Sharpe: {sharpe:6.2f}, Return: {return_pct:7.2f}%)")
            else:
                print(f"   ⏳ {ticker:6s} - Not trained yet")
        
        wait_for_enter()
    
    # ==========================================
    # META-AI TRAINING
    # ==========================================
    
    def train_single_meta_ai(self):
        """Train Meta-AI for a single stock (test)"""
        clear_screen()
        print_header("🤖 TRAIN SINGLE META-AI (TEST)")
        
        print("\n📋 This will train a Meta-AI for one stock as a test.")
        print("   Use this to verify everything works before training multiple stocks.")
        
        ticker = input("\nEnter ticker symbol (e.g., AAPL): ").upper().strip()
        
        if not ticker:
            print_warning("No ticker entered.")
            wait_for_enter()
            return
        
        print(f"\n🚀 Training Meta-AI for {ticker}...")
        print("⏱️  Estimated time: 1-2 hours (10M timesteps with GPU)")
        print("   Training Period: 2016-2025 (9 years)")
        print("   Test Period: 2025-2025 (1 year)")
        print("   Meta-AI will learn optimal trading strategy")
        print("   Output: BUY/SELL/HOLD + position size (0-95%)")
        
        if not confirm_action("Proceed with training?"):
            print_warning("Training cancelled")
            wait_for_enter()
            return
        
        # Execute training
        from models.meta.meta_trainer import train_meta_ai
        
        try:
            model = train_meta_ai(ticker, total_timesteps=10_000_000, use_gpu=True)
            
            # Metrics are already printed in train_meta_ai
            print_success(f"Meta-AI training complete for {ticker}!")
            
        except Exception as e:
            print_error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
        
        wait_for_enter()
    
    def train_multiple_meta_ais(self):
        """Train multiple Meta-AIs with selection"""
        clear_screen()
        print_header("🤖 TRAIN MULTIPLE META-AIs")
        
        if not self.selected_top_10:
            print_warning("No top 10 stocks selected yet!")
            print_info("Please select your top 10 stocks first using option [9].")
            wait_for_enter()
            return
        
        # Get untrained stocks
        from models.meta.meta_trainer import get_untrained_stocks
        untrained = get_untrained_stocks(self.selected_top_10)
        
        if not untrained:
            print_success("All selected stocks have been trained!")
            print_info("View results with option [14]")
            wait_for_enter()
            return
        
        print(f"\n📋 Untrained stocks from your Top 10:")
        for i, ticker in enumerate(untrained, 1):
            print(f"   {i:2d}. {ticker}")
        
        print(f"\n💡 You can train some or all of these stocks now.")
        
        # Ask how many to train
        print(f"\nHow many stocks would you like to train?")
        print(f"  Enter a number (1-{len(untrained)}) or 'all'")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'all':
            stocks_to_train = untrained
        else:
            try:
                count = int(choice)
                if count < 1 or count > len(untrained):
                    print_error(f"Please enter a number between 1 and {len(untrained)}")
                    wait_for_enter()
                    return
                
                # Let user select which specific stocks
                print(f"\nWhich {count} stock(s) would you like to train?")
                print(f"Enter numbers separated by commas (e.g., 1,2,3):")
                
                selection = input("\nYour selection: ").strip()
                indices = [int(i.strip())-1 for i in selection.split(',')]
                
                if len(indices) != count:
                    print_error(f"Please select exactly {count} stocks")
                    wait_for_enter()
                    return
                
                stocks_to_train = [untrained[i] for i in indices if 0 <= i < len(untrained)]
                
            except ValueError:
                print_error("Invalid input")
                wait_for_enter()
                return
        
        print(f"\n✅ Will train: {', '.join(stocks_to_train)}")
        print(f"⏱️  Estimated time: {len(stocks_to_train) * 1.5:.1f} hours")
        
        if not confirm_action(f"Proceed with training {len(stocks_to_train)} Meta-AIs?"):
            print_warning("Training cancelled")
            wait_for_enter()
            return
        
        # Execute training
        from models.meta.meta_trainer import train_selected_stocks
        
        try:
            results = train_selected_stocks(stocks_to_train, total_timesteps=10_000_000)
            
            if results:
                print_success(f"Training complete for {len(results)} stocks!")
                print_info("View detailed results with option [14]")
        except Exception as e:
            print_error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
        
        wait_for_enter()
    
    def train_all_untrained_meta_ais(self):
        """Train all untrained Meta-AIs from top 10"""
        clear_screen()
        print_header("🤖 TRAIN ALL UNTRAINED META-AIs")
        
        if not self.selected_top_10:
            print_warning("No top 10 stocks selected yet!")
            print_info("Please select your top 10 stocks first using option [9].")
            wait_for_enter()
            return
        
        # Get untrained stocks
        from models.meta.meta_trainer import get_untrained_stocks
        untrained = get_untrained_stocks(self.selected_top_10)
        
        if not untrained:
            print_success("All selected stocks have been trained!")
            print_info("View results with option [14]")
            wait_for_enter()
            return
        
        print(f"\n📋 This will train Meta-AIs for {len(untrained)} untrained stocks:")
        for i, ticker in enumerate(untrained, 1):
            print(f"   {i:2d}. {ticker}")
        
        print(f"\n⏱️  Estimated time: {len(untrained) * 1.5:.1f} hours")
        print(f"   Training: 10M timesteps per stock")
        print(f"   Uses GPU if available")
        
        if not confirm_action(f"Proceed with training all {len(untrained)} Meta-AIs?"):
            print_warning("Training cancelled")
            wait_for_enter()
            return
        
        # Execute training
        from models.meta.meta_trainer import train_selected_stocks
        
        try:
            results = train_selected_stocks(untrained, total_timesteps=10_000_000)
            
            if results:
                print_success(f"All {len(results)} Meta-AIs trained successfully!")
                print_info("View detailed results with option [14]")
        except Exception as e:
            print_error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
        
        wait_for_enter()
    
    def evaluate_meta_ai(self):
        """Evaluate a trained Meta-AI on forward test period"""
        clear_screen()
        print_header("📊 EVALUATE TRAINED META-AI")
        
        print("\n📋 This will run a forward test (2023-2025) on a trained Meta-AI")
        print("   and update the performance metrics in the system.")
        
        # Sync filesystem to detect all trained models
        print_info("Scanning for trained Meta-AI models...")
        self.model_manager.sync_meta_ai_status()
        
        # Get list of trained Meta-AIs
        meta_status = self.model_manager.get_meta_training_status()
        trained_tickers = [ticker for ticker, status in meta_status.items() 
                          if status.get("trained", False)]
        
        if not trained_tickers:
            print_warning("No trained Meta-AIs found!")
            print_info("Train a Meta-AI first using option [11], [12], or [13]")
            wait_for_enter()
            return
        
        print(f"\n📋 Trained Meta-AIs available:")
        for i, ticker in enumerate(sorted(trained_tickers), 1):
            status = meta_status[ticker]
            last_trained = status.get('last_trained', 'Unknown')
            has_metrics = 'metrics' in status
            print(f"   {i:2d}. {ticker:6s} - Last trained: {last_trained[:10]} {'✅ Has metrics' if has_metrics else '⚠️  No metrics yet'}")
        
        ticker_input = input(f"\nEnter ticker symbol to evaluate (or press Enter to cancel): ").strip().upper()
        
        if not ticker_input:
            print_warning("Evaluation cancelled")
            wait_for_enter()
            return
        
        if ticker_input not in trained_tickers:
            print_error(f"{ticker_input} is not a trained Meta-AI!")
            wait_for_enter()
            return
        
        print(f"\n📈 Running forward test evaluation for {ticker_input}...")
        print(f"   Test Period: 2023-2025 (out-of-sample)")
        print(f"   This will take ~1-2 minutes")
        
        # Run evaluation
        from models.meta.meta_trainer import evaluate_meta_ai
        
        try:
            metrics = evaluate_meta_ai(ticker_input)
            
            if metrics:
                print_success(f"\n✅ Evaluation complete for {ticker_input}!")
                print_info("Metrics saved and status updated")
                print_info("View all rankings with option [15]")
            else:
                print_error("Evaluation failed - no metrics returned")
        except Exception as e:
            print_error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
        wait_for_enter()
    
    def view_meta_ai_performance(self):
        """View Meta-AI performance rankings and status"""
        clear_screen()
        print_header("🏆 META-AI PERFORMANCE & STATUS")
        
        # Sync filesystem with master_config before displaying
        print_info("Checking for unregistered Meta-AI models...")
        self.model_manager.sync_meta_ai_status()
        print("")
        
        self.model_manager.show_meta_ai_rankings()
        
        wait_for_enter()
    
    # ==========================================
    # MODEL MANAGEMENT
    # ==========================================
    
    def compare_models(self):
        """Compare old vs new models"""
        clear_screen()
        print_header("🔄 COMPARE MODELS")
        
        print("\n[Feature coming in Phase 2]")
        wait_for_enter()
    
    def view_history(self):
        """View training history"""
        clear_screen()
        print_header("📜 TRAINING HISTORY")
        
        self.model_manager.show_training_history()
        
        wait_for_enter()
    
    # ==========================================
    # DEPLOYMENT
    # ==========================================
    
    def deploy_to_pi(self):
        """Deploy models to Raspberry Pi"""
        clear_screen()
        print_header("🚀 DEPLOY TO RASPBERRY PI")
        
        print("\n[Feature coming in Phase 2]")
        wait_for_enter()
    
    # ==========================================
    # SYSTEM
    # ==========================================
    
    def system_health_check(self):
        """Run system health check"""
        clear_screen()
        print_header("🏥 SYSTEM HEALTH CHECK")
        
        self.model_manager.run_health_check()
        
        wait_for_enter()
    
    def clear_cache(self):
        """Clear prediction caches"""
        clear_screen()
        print_header("🧹 CLEAR CACHE")
        
        print("\n⚠️  This will delete all prediction caches.")
        print("   Caches will be rebuilt on next training.")
        
        if not confirm_action("Proceed with cache clearing?"):
            print_warning("Cache clear cancelled")
            wait_for_enter()
            return
        
        self.model_manager.clear_all_caches()
        
        print_success("Caches cleared!")
        wait_for_enter()
    
    def exit_program(self):
        """Exit the control center"""
        print_info("\nExiting Control Center. Goodbye! 👋")
        self.running = False
        sys.exit(0)


def main():
    """Main entry point"""
    try:
        control_center = ControlCenter()
        control_center.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()