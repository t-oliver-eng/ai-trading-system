"""
Enhanced Logging for Meta-AI Training
Logs detailed data every check for future Meta-AI development
"""

import csv
import os
from datetime import datetime
import pandas as pd

META_LOG_FILE = 'meta_ai_training_data.csv'

def init_meta_logging():
    """Initialize CSV file with headers if it doesn't exist"""
    if not os.path.exists(META_LOG_FILE):
        headers = [
            # Timestamp
            'timestamp',
            'date',
            'time',
            'hour',
            'day_of_week',
            
            # Market Status
            'market_status',  # REGULAR, PRE, AFTER, CLOSED
            
            # Price Data
            'current_price',
            'price_change_1h_pct',
            'price_change_4h_pct',
            'price_change_1d_pct',
            
            # AI Predictions
            'daily_confidence',
            'daily_decision',
            'hourly_confidence',
            'hourly_decision',
            
            # Position Info
            'holding_position',
            'shares',
            'entry_price',
            'current_pnl_pct',
            'current_pnl_dollars',
            'time_in_position_hours',
            'peak_pnl_pct',
            'drawdown_from_peak_pct',
            'entry_confidence',
            'confidence_change',
            
            # Exit Conditions Status
            'would_hit_stop_loss',
            'would_hit_take_profit',
            'would_hit_time_stop',
            'would_hit_conf_threshold',
            
            # Future Labels (filled in later for training)
            'return_next_1h',
            'return_next_4h',
            'return_next_1d',
            'return_next_5d',
            'max_gain_next_1d',
            'max_loss_next_1d',
        ]
        
        with open(META_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"✅ Initialized {META_LOG_FILE}")


def log_meta_data(ticker, current_price, position_state, daily_pred, hourly_pred, market_status):
    """
    Log comprehensive data for Meta-AI training
    
    Args:
        ticker: Stock ticker
        current_price: Current price
        position_state: Dict with position info
        daily_pred: Dict with daily prediction
        hourly_pred: Dict with hourly prediction (can be None)
        market_status: 'REGULAR', 'PRE', 'AFTER', 'CLOSED'
    """
    
    now = datetime.now()
    
    # Calculate price changes (if we have historical data)
    price_1h_pct = 0  # Would need price history to calculate
    price_4h_pct = 0
    price_1d_pct = 0
    
    # Position info
    holding = position_state.get('holding', False)
    shares = position_state.get('shares', 0)
    entry_price = position_state.get('entry_price', 0)
    entry_conf = position_state.get('entry_confidence', 0)
    
    # Calculate P&L if holding
    pnl_pct = 0
    pnl_dollars = 0
    time_held = 0
    peak_pnl = 0
    drawdown = 0
    conf_change = 0
    
    if holding and entry_price > 0:
        pnl_pct = (current_price - entry_price) / entry_price * 100
        pnl_dollars = (current_price - entry_price) * shares
        
        if position_state.get('entry_time'):
            time_held = (now - position_state['entry_time']).total_seconds() / 3600
        
        # Track peak (would need to maintain this in position_state)
        peak_pnl = position_state.get('peak_pnl_pct', pnl_pct)
        if pnl_pct > peak_pnl:
            peak_pnl = pnl_pct
        
        drawdown = peak_pnl - pnl_pct
        conf_change = (daily_pred['confidence'] - entry_conf) * 100 if daily_pred else 0
    
    # Exit condition checks
    would_stop_loss = pnl_pct <= -5.0 if holding else False
    would_take_profit = pnl_pct >= 8.0 if holding else False
    would_time_stop = time_held >= (7 * 24) if holding else False
    would_conf_exit = (daily_pred['confidence'] < 0.50) if (holding and daily_pred) else False
    
    # Prepare row
    row = [
        # Timestamp
        now.isoformat(),
        now.strftime('%Y-%m-%d'),
        now.strftime('%H:%M:%S'),
        now.hour,
        now.strftime('%A'),
        
        # Market Status
        market_status,
        
        # Price Data
        current_price,
        price_1h_pct,
        price_4h_pct,
        price_1d_pct,
        
        # AI Predictions
        daily_pred['confidence'] * 100 if daily_pred else 0,
        daily_pred['decision'] if daily_pred else '',
        hourly_pred['confidence'] * 100 if hourly_pred else 0,
        hourly_pred['decision'] if hourly_pred else '',
        
        # Position Info
        holding,
        shares,
        entry_price,
        round(pnl_pct, 2),
        round(pnl_dollars, 2),
        round(time_held, 1),
        round(peak_pnl, 2),
        round(drawdown, 2),
        entry_conf * 100 if entry_conf else 0,
        round(conf_change, 2),
        
        # Exit Conditions
        would_stop_loss,
        would_take_profit,
        would_time_stop,
        would_conf_exit,
        
        # Future labels (empty for now, filled later)
        '',  # return_next_1h
        '',  # return_next_4h
        '',  # return_next_1d
        '',  # return_next_5d
        '',  # max_gain_next_1d
        '',  # max_loss_next_1d
    ]
    
    # Append to CSV
    try:
        with open(META_LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        print(f"❌ Error logging meta data: {e}")


def get_training_data_summary():
    """Get summary of collected training data"""
    try:
        if not os.path.exists(META_LOG_FILE):
            return "No training data collected yet"
        
        df = pd.read_csv(META_LOG_FILE)
        
        summary = f"""
📊 META-AI TRAINING DATA SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Rows: {len(df)}
Date Range: {df['date'].min()} to {df['date'].max()}
Hours Logged: {len(df) * 0.5:.1f}h (assuming 30-min checks)

Position Data:
  - Checks with position: {df['holding_position'].sum()}
  - Checks without: {(~df['holding_position']).sum()}

Average When Holding:
  - Daily Confidence: {df[df['holding_position']]['daily_confidence'].mean():.1f}%
  - P&L: {df[df['holding_position']]['current_pnl_pct'].mean():+.2f}%
  - Time Held: {df[df['holding_position']]['time_in_position_hours'].mean():.1f}h

File Location: {META_LOG_FILE}
File Size: {os.path.getsize(META_LOG_FILE) / 1024:.1f} KB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return summary
    except Exception as e:
        return f"Error reading training data: {e}"


if __name__ == "__main__":
    # Test
    init_meta_logging()
    print("Meta-AI logging initialized!")
    print(get_training_data_summary())
