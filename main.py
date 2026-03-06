# main.py

import argparse


def main():
    parser = argparse.ArgumentParser(description="AI Trading Bot Workflow Manager")
    parser.add_argument(
        "command",
        help="The command to run",
        choices=[
            "download_daily",
            "optimize_daily",
            "train_daily",
            "predict_daily",
            "paper_trade_daily",
            "optimize_threshold_daily",
            "download_hourly",
            "optimize_hourly",
            "train_hourly",
            "predict_hourly",
            "paper_trade_hourly",
            "optimize_threshold_hourly",
            "backtest_v33",
            "optimize_v33",
        ],
    )
    parser.add_argument(
        "--ticker", help="Ticker symbol for per-stock analysis", type=str, default=None
    )
    parser.add_argument(
        "--mode",
        help="Optimization mode ('global' or 'local')",
        type=str,
        default="local",
    )
    parser.add_argument(
        "--start",
        help="Start date for backtest (YYYY-MM-DD)",
        type=str,
        default="2025-01-01",
    )
    parser.add_argument(
        "--end",
        help="End date for backtest (YYYY-MM-DD)",
        type=str,
        default="2025-12-31",
    )
    parser.add_argument(
        "--capital",
        help="Initial capital for backtest",
        type=float,
        default=100000,
    )
    args = parser.parse_args()

    # --- Corrected Argument Validation ---
    is_global_command = (
        args.command in ["optimize_daily", "optimize_hourly"]
    ) and args.mode == "global"

    # Check if a ticker is required
    if (
        not args.ticker
        and not is_global_command
        and args.command not in ["download_daily", "download_hourly"]
    ):
        print(f"Error: The '{args.command}' command requires a --ticker argument.")
        return

    # --- Daily Workflow ---
    if args.command == "download_daily":
        from utils.data_fetcher import fetch_data
        
        # FIXED: Pass the ticker argument
        fetch_data(args.ticker)

    elif args.command == "optimize_daily":
        from testing.optimizer_daily import optimize_daily_strategies
        optimize_daily_strategies(args.ticker, args.mode)
    elif args.command == "train_daily":
        from models.daily.trainer import train_daily_ai_model

        train_daily_ai_model(args.ticker)
    elif args.command == "predict_daily":
        from models.daily.predictor import get_prediction

        get_prediction(args.ticker)
    elif args.command == "paper_trade_daily":
        from testing.paper_trader import run_paper_trader

        run_paper_trader(args.ticker)
    elif args.command == "optimize_threshold_daily":
        from testing.threshold_optimizer import run_threshold_optimizer

        run_threshold_optimizer(args.ticker)

    # --- Hourly Workflow ---
    elif args.command == "download_hourly":
        from utils.data_fetcher_hourly import fetch_hourly_data
        
        # FIXED: Pass the ticker argument
        fetch_hourly_data(args.ticker)

    elif args.command == "optimize_hourly":
        from testing.optimizer_hourly import optimize_hourly_strategies
        optimize_hourly_strategies(args.ticker, args.mode)
    elif args.command == "train_hourly":
        from models.hourly.trainer import train_hourly_ai_model

        train_hourly_ai_model(args.ticker)
    elif args.command == "predict_hourly":
        from models.hourly.predictor import get_hourly_prediction

        get_hourly_prediction(args.ticker)
    elif args.command == "paper_trade_hourly":
        from testing.paper_trader_hourly import run_hourly_paper_trader

        run_hourly_paper_trader(args.ticker)
    elif args.command == "optimize_threshold_hourly":
        from testing.threshold_optimizer_hourly import run_hourly_threshold_optimizer

        run_hourly_threshold_optimizer(args.ticker)
    
    # --- V3.3 Backtest ---
    elif args.command == "backtest_v33":
        # FIXED: Import correct function name 'run_v33_forward_test'
        from testing.run_v33_backtest import run_v33_forward_test
        
        if not args.ticker:
            print("Error: backtest_v33 requires --ticker argument")
            return
        
        print(f"\n🔬 Running V3.3 Backtest on {args.ticker}")
        print(f"   Period: {args.start} to {args.end}")
        print(f"   Initial Capital: ${args.capital:,.2f}\n")
        
        # FIXED: Call correct function name
        run_v33_forward_test(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital
        )
    # --- V3.3 Parameter Optimization ---
    elif args.command == "optimize_v33":
        from testing.v33_parameter_optimizer import optimize_v33
        
        if not args.ticker:
            print("Error: optimize_v33 requires --ticker argument")
            return
            
        optimize_v33(args.ticker, start_date=args.start, end_date=args.end)


if __name__ == "__main__":
    main()