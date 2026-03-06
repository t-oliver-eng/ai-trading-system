# backtester.py - Improved Realistic Backtesting Engine

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os


class RealisticBacktester:
    """
    Realistic backtesting engine with:
    - Order-by-order execution
    - Realistic slippage and commissions
    - Proper position tracking
    - Comprehensive metrics
    - Visualization
    """
    
    def __init__(self, 
                 initial_capital=100000,
                 commission_pct=0.0001,  # 0.01% commission (very low for Alpaca)
                 slippage_pct=0.0005):   # 0.05% slippage
        
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        
        # Portfolio state
        self.cash = initial_capital
        self.shares = 0
        self.position_value = 0
        
        # Position tracking
        self.entry_price = 0
        self.entry_time = None
        self.position_metadata = {}  # Store any custom data about position
        
        # Trade history
        self.trades = []
        self.equity_curve = []
        
    def reset(self):
        """Reset backtester to initial state"""
        self.cash = self.initial_capital
        self.shares = 0
        self.position_value = 0
        self.entry_price = 0
        self.entry_time = None
        self.position_metadata = {}
        self.trades = []
        self.equity_curve = []
        
    def execute_buy(self, timestamp, price, shares=None, metadata=None):
        """
        Execute a buy order
        
        Args:
            timestamp: Trade timestamp
            price: Execution price
            shares: Number of shares (if None, buy with all available cash)
            metadata: Dict of any additional data to store about this trade
        
        Returns:
            bool: True if trade executed, False if insufficient funds
        """
        if shares is None:
            # Calculate max shares we can afford
            shares = int(self.cash / (price * (1 + self.commission_pct + self.slippage_pct)))
        
        # Calculate total cost with slippage and commission
        execution_price = price * (1 + self.slippage_pct)
        cost = shares * execution_price
        commission = cost * self.commission_pct
        total_cost = cost + commission
        
        if total_cost > self.cash:
            return False  # Insufficient funds
        
        # Execute trade
        self.cash -= total_cost
        self.shares += shares
        self.entry_price = execution_price
        self.entry_time = timestamp
        self.position_metadata = metadata or {}
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'action': 'BUY',
            'price': price,
            'execution_price': execution_price,
            'shares': shares,
            'commission': commission,
            'slippage': execution_price - price,
            'cash_after': self.cash,
            'portfolio_value': self.cash + (self.shares * price),
            'metadata': metadata
        })
        
        return True
    
    def execute_sell(self, timestamp, price, shares=None, reason=None):
        """
        Execute a sell order
        
        Args:
            timestamp: Trade timestamp
            price: Execution price
            shares: Number of shares (if None, sell all)
            reason: Exit reason (for tracking)
        
        Returns:
            bool: True if trade executed, False if no shares to sell
        """
        if self.shares == 0:
            return False  # No position to sell
        
        if shares is None:
            shares = self.shares
        
        # Calculate proceeds with slippage and commission
        execution_price = price * (1 - self.slippage_pct)
        proceeds = shares * execution_price
        commission = proceeds * self.commission_pct
        net_proceeds = proceeds - commission
        
        # Calculate P&L
        cost_basis = self.entry_price * shares
        pnl = net_proceeds - cost_basis
        pnl_pct = (execution_price - self.entry_price) / self.entry_price
        
        # Hold time
        hold_time = timestamp - self.entry_time if self.entry_time else timedelta(0)
        
        # Execute trade
        self.cash += net_proceeds
        self.shares -= shares
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'action': 'SELL',
            'price': price,
            'execution_price': execution_price,
            'shares': shares,
            'commission': commission,
            'slippage': price - execution_price,
            'cash_after': self.cash,
            'portfolio_value': self.cash + (self.shares * price),
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'hold_time': hold_time,
            'hold_days': hold_time.days,
            'entry_price': self.entry_price,
            'exit_reason': reason,
            'entry_metadata': self.position_metadata
        })
        
        # Reset position tracking if fully closed
        if self.shares == 0:
            self.entry_price = 0
            self.entry_time = None
            self.position_metadata = {}
        
        return True
    
    def update_equity(self, timestamp, current_price):
        """Update equity curve with current portfolio value"""
        portfolio_value = self.cash + (self.shares * current_price)
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'cash': self.cash,
            'position_value': self.shares * current_price,
            'portfolio_value': portfolio_value,
            'shares': self.shares
        })
    
    def get_current_position(self, current_price):
        """Get current position status"""
        if self.shares == 0:
            return None
        
        current_value = self.shares * current_price
        cost_basis = self.shares * self.entry_price
        unrealized_pnl = current_value - cost_basis
        unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
        
        return {
            'shares': self.shares,
            'entry_price': self.entry_price,
            'current_price': current_price,
            'current_value': current_value,
            'cost_basis': cost_basis,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'entry_time': self.entry_time,
            'metadata': self.position_metadata
        }
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.equity_curve:
            return None
        
        df_equity = pd.DataFrame(self.equity_curve)
        df_equity.set_index('timestamp', inplace=True)
        
        # Basic metrics
        final_value = df_equity['portfolio_value'].iloc[-1]
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # Trade statistics
        completed_trades = [t for t in self.trades if t['action'] == 'SELL']
        num_trades = len(completed_trades)
        
        if num_trades > 0:
            winning_trades = [t for t in completed_trades if t['pnl'] > 0]
            losing_trades = [t for t in completed_trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / num_trades * 100
            
            avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) * 100 if winning_trades else 0
            avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) * 100 if losing_trades else 0
            avg_return = np.mean([t['pnl_pct'] for t in completed_trades]) * 100
            
            best_trade = max([t['pnl_pct'] for t in completed_trades]) * 100
            worst_trade = min([t['pnl_pct'] for t in completed_trades]) * 100
            
            total_commission = sum([t['commission'] for t in self.trades])
            total_slippage_cost = sum([abs(t['slippage']) * t['shares'] for t in self.trades])
            
            avg_hold_days = np.mean([t['hold_days'] for t in completed_trades])
        else:
            win_rate = avg_win = avg_loss = avg_return = 0
            best_trade = worst_trade = 0
            total_commission = total_slippage_cost = 0
            avg_hold_days = 0
        
        # Drawdown analysis
        df_equity['cummax'] = df_equity['portfolio_value'].cummax()
        df_equity['drawdown'] = (df_equity['portfolio_value'] - df_equity['cummax']) / df_equity['cummax'] * 100
        max_drawdown = df_equity['drawdown'].min()
        
        # Sharpe Ratio (annualized)
        returns = df_equity['portfolio_value'].pct_change().dropna()
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Sortino Ratio (uses only downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino_ratio = (returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = 0
        
        # Calmar Ratio (return / max drawdown)
        if max_drawdown < 0:
            calmar_ratio = total_return / abs(max_drawdown)
        else:
            calmar_ratio = 0
        
        metrics = {
            'total_return_pct': total_return,
            'final_portfolio_value': final_value,
            'num_trades': num_trades,
            'win_rate_pct': win_rate,
            'avg_return_pct': avg_return,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'best_trade_pct': best_trade,
            'worst_trade_pct': worst_trade,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'total_commission': total_commission,
            'total_slippage_cost': total_slippage_cost,
            'avg_hold_days': avg_hold_days
        }
        
        return metrics, df_equity
    
    def save_results(self, output_dir, metadata=None):
        """Save backtest results to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metrics
        metrics, df_equity = self.calculate_metrics()
        
        # Add metadata
        if metadata:
            metrics.update({'backtest_metadata': metadata})
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        
        # Process trades for export (extract confidence)
        export_trades = []
        for t in self.trades:
            trade_data = t.copy()
            # Extract confidence from metadata if present
            confidence = None
            if 'metadata' in t and t['metadata'] and 'confidence' in t['metadata']:
                confidence = t['metadata']['confidence']
            elif 'entry_metadata' in t and t['entry_metadata'] and 'confidence' in t['entry_metadata']:
                confidence = t['entry_metadata']['confidence']
            
            if confidence is not None:
                trade_data['confidence'] = confidence
            
            export_trades.append(trade_data)

        # Save trades (using processed list)
        with open(os.path.join(output_dir, 'trades.json'), 'w') as f:
            json.dump({'trades': export_trades}, f, indent=4, default=str)
            
        df_trades = pd.DataFrame(export_trades)
        df_trades.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
        
        # Save equity curve
        df_equity.to_csv(os.path.join(output_dir, 'equity_curve.csv'))
        
        # Generate plots
        self.plot_equity_curve(df_equity, output_dir)
        self.plot_drawdown(df_equity, output_dir)
        if metrics['num_trades'] > 0:
            self.plot_trade_distribution(output_dir)
        
        return metrics
    
    def plot_equity_curve(self, df_equity, output_dir):
        """Plot equity curve with buy/sell markers"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(df_equity.index, df_equity['portfolio_value'], 
                linewidth=2, label='Portfolio Value', color='steelblue')
        ax.axhline(y=self.initial_capital, color='gray', 
                   linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Mark trades
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        
        for trade in buy_trades:
            ax.scatter(trade['timestamp'], trade['portfolio_value'], 
                      color='green', marker='^', s=100, zorder=5, alpha=0.7)
        
        for trade in sell_trades:
            color = 'blue' if trade.get('pnl', 0) > 0 else 'red'
            ax.scatter(trade['timestamp'], trade['portfolio_value'], 
                      color=color, marker='v', s=100, zorder=5, alpha=0.7)
        
        ax.set_title('Backtest Equity Curve', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'equity_curve.png'), dpi=300)
        plt.close()
    
    def plot_drawdown(self, df_equity, output_dir):
        """Plot drawdown chart"""
        fig, ax = plt.subplots(figsize=(14, 5))
        
        ax.fill_between(df_equity.index, 0, df_equity['drawdown'], 
                        color='red', alpha=0.3)
        ax.plot(df_equity.index, df_equity['drawdown'], 
                color='red', linewidth=2)
        
        ax.set_title('Portfolio Drawdown', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'drawdown.png'), dpi=300)
        plt.close()
    
    def plot_trade_distribution(self, output_dir):
        """Plot trade return distribution"""
        completed_trades = [t for t in self.trades if t['action'] == 'SELL']
        returns = [t['pnl_pct'] * 100 for t in completed_trades]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(returns, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break Even')
        ax.axvline(x=np.mean(returns), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
        
        ax.set_title('Trade Return Distribution', fontsize=16, fontweight='bold')
        ax.set_xlabel('Return (%)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trade_distribution.png'), dpi=300)
        plt.close()
    
    def print_summary(self):
        """Print backtest summary to console"""
        metrics, _ = self.calculate_metrics()
        
        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Return:        {metrics['total_return_pct']:+.2f}%")
        print(f"Final Value:         ${metrics['final_portfolio_value']:,.2f}")
        print(f"Number of Trades:    {metrics['num_trades']}")
        print(f"Win Rate:            {metrics['win_rate_pct']:.1f}%")
        print(f"Average Return:      {metrics['avg_return_pct']:+.2f}%")
        print(f"Average Win:         {metrics['avg_win_pct']:+.2f}%")
        print(f"Average Loss:        {metrics['avg_loss_pct']:+.2f}%")
        print(f"Best Trade:          {metrics['best_trade_pct']:+.2f}%")
        print(f"Worst Trade:         {metrics['worst_trade_pct']:+.2f}%")
        print(f"Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio:       {metrics['sortino_ratio']:.2f}")
        print(f"Calmar Ratio:        {metrics['calmar_ratio']:.2f}")
        print(f"Total Commission:    ${metrics['total_commission']:,.2f}")
        print(f"Total Slippage:      ${metrics['total_slippage_cost']:,.2f}")
        print(f"Avg Hold Days:       {metrics['avg_hold_days']:.1f}")
        print(f"{'='*60}\n")

# --- Legacy / Optimizer Backtester ---
def run_backtest(price_data, signals):
    """
    Runs a simple vectorized backtest for the Optimizer.
    
    Args:
        price_data (pd.DataFrame): DataFrame with at least a 'Close' column.
        signals (pd.Series): Series with signals (1 for buy, -1 for sell, 0 for hold).
        
    Returns:
        float: The cumulative return of the strategy.
    """
    # Start with a copy to avoid modifying original data
    data = price_data.copy()
    
    # Ensure signal alignment
    data["signal"] = signals

    # Calculate daily returns of the asset
    data["asset_return"] = data["Close"].pct_change()

    # Calculate strategy returns
    # We shift the signal because we buy/sell on one day and realize the return on the next
    data["strategy_return"] = data["asset_return"] * data["signal"].shift(1)

    # Calculate cumulative returns
    data["cumulative_strategy_return"] = (1 + data["strategy_return"]).cumprod() - 1

    if (
        "cumulative_strategy_return" not in data 
        or data["cumulative_strategy_return"].empty
        or data["cumulative_strategy_return"].isna().all()
    ):
        return 0.0

    # Return the final cumulative return
    final_return = data["cumulative_strategy_return"].iloc[-1]

    return final_return if not np.isnan(final_return) else 0.0