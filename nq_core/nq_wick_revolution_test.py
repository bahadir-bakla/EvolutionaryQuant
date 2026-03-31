"""
Test the NQ Wick Reversal strategy with the implemented logic
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nq_core.nq_wick_reversal_final import NQWickReversalStrategy

def fetch_nq_data(symbol="NQ=F", interval="5m", lookback_days=30):
    """Fetch NQ data from Yahoo Finance"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    print(f"Fetching {interval} data for {symbol} ({start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')})...")
    df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df.index.name = 'time'
    if df.index.tz is not None: 
        df.index = df.index.tz_convert(None)
    return df

def add_indicators_for_strategy(df):
    """Add the indicators required by our strategy"""
    df = df.copy()
    
    # Calculate ATR for dynamic stop loss
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = true_range.rolling(14).mean()
    
    # Calculate candle properties
    df['body_size'] = np.abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
    df['is_bullish'] = df['close'] > df['open']
    df['is_bearish'] = df['close'] < df['open']
    
    # Wick strength as ratio of wick to body (avoid division by zero)
    df['upper_wick_ratio'] = np.where(
        df['body_size'] > 0, 
        df['upper_wick'] / df['body_size'], 
        0
    )
    df['lower_wick_ratio'] = np.where(
        df['body_size'] > 0, 
        df['lower_wick'] / df['body_size'], 
        0
    )
    
    return df

def main():
    print("Testing NQ Wick Reversal Strategy")
    print("=" * 50)
    
    # Fetch data
    df_5m = fetch_nq_data("NQ=F", "5m", 30)  # 30 days for testing
    
    if df_5m.empty:
        print("Failed to fetch data")
        return
        
    print(f"Retrieved {len(df_5m)} 5m bars")
    print(f"Date range: {df_5m.index[0]} to {df_5m.index[-1]}")
    
    # Add indicators
    df_ind = add_indicators_for_strategy(df_5m.copy())
    
    # Initialize strategy
    strategy = NQWickReversalStrategy()
    print(f"\nStrategy: {strategy.name}")
    print(f"Parameters: {strategy.get_parameters()}")
    
    # Test signal generation on a few days
    print(f"\n--- SIGNAL GENERATION TEST ---")
    signals_found = 0
    
    # Look for signals in the data
    for i in range(20, min(len(df_ind)-1, 1000)):  # Start after warmup, limit for testing
        signal = strategy.generate_signals(df_ind, i)
        
        if signal.signal_type != 'NEUTRAL':
            print(f"Signal at {df_ind.index[i]}: {signal.signal_type}")
            print(f"  Entry: {signal.price:.2f}")
            print(f"  Stop: {signal.stop_loss:.2f}")
            print(f"  Target: {signal.take_profit:.2f}")
            print(f"  Range: [{strategy.daily_range_low:.2f}, {strategy.daily_range_high:.2f}]")
            print(f"  Metadata: {signal.metadata}")
            signals_found += 1
            
            if signals_found >= 5:  # Limit output
                break
    
    if signals_found == 0:
        print("No signals found in the test period")
    
    # Run a simple backtest
    print(f"\n--- SIMPLE BACKTEST ---")
    balance = 1000.0
    trades = []
    
    i = 0
    while i < len(df_ind):
        signal = strategy.generate_signals(df_ind, i)
        
        if signal.signal_type != 'NEUTRAL':
            # Execute trade
            entry_price = signal.price
            direction = 1 if signal.signal_type == 'LONG' else -1
            
            # Look for exit
            exit_price = None
            exit_reason = ""
            exit_idx = i
            
            # Check subsequent candles for SL/TP hit (max 50 candles lookahead)
            for j in range(i+1, min(i+51, len(df_ind))):
                future_row = df_ind.iloc[j]
                
                if signal.signal_type == 'LONG':
                    if future_row['low'] <= signal.stop_loss:
                        exit_price = signal.stop_loss
                        exit_reason = "STOP_LOSS"
                        exit_idx = j
                        break
                    elif future_row['high'] >= signal.take_profit:
                        exit_price = signal.take_profit
                        exit_reason = "TAKE_PROFIT"
                        exit_idx = j
                        break
                else:  # SHORT
                    if future_row['high'] >= signal.stop_loss:
                        exit_price = signal.stop_loss
                        exit_reason = "STOP_LOSS"
                        exit_idx = j
                        break
                    elif future_row['low'] <= signal.take_profit:
                        exit_price = signal.take_profit
                        exit_reason = "TAKE_PROFIT"
                        exit_idx = j
                        break
            
            # If neither hit, close at last price in lookback period
            if exit_price is None:
                exit_price = df_ind.iloc[min(i+50, len(df_ind)-1)]['close']
                exit_reason = "TIME_EXIT"
                exit_idx = min(i+50, len(df_ind)-1)
            
            # Calculate P&L
            if signal.signal_type == "LONG":
                pnl_points = exit_price - entry_price
            else:
                pnl_points = entry_price - exit_price
                
            # NQ point value is $20
            pnl_usd = pnl_points * 20.0 * signal.position_size
            balance += pnl_usd
            
            # Record trade
            trades.append({
                'entry_time': df_ind.index[i],
                'exit_time': df_ind.index[exit_idx],
                'direction': signal.signal_type,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_points': pnl_points,
                'pnl_usd': pnl_usd,
                'balance': balance,
                'exit_reason': exit_reason,
                'position_size': signal.position_size
            })
            
            # Move index past this trade
            i = exit_idx + 1
        else:
            i += 1
    
    # Calculate results
    print(f"Starting Balance: $1000.00")
    print(f"Ending Balance:   ${balance:.2f}")
    if balance > 0:
        print(f"Net Return:       {((balance - 1000) / 1000) * 100:+.2f}%")
    print(f"Total Trades:     {len(trades)}")
    
    if trades:
        wins = [t for t in trades if t['pnl_usd'] > 0]
        win_rate = len(wins) / len(trades) * 100 if trades else 0
        print(f"Win Rate:         {win_rate:.1f}%")
        
        total_pnl = sum(t['pnl_usd'] for t in trades)
        avg_win = sum(t['pnl_usd'] for t in wins) / len(wins) if wins else 0
        losses = [t for t in trades if t['pnl_usd'] <= 0]
        avg_loss = sum(t['pnl_usd'] for t in losses) / len(losses) if losses else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        print(f"Average Win:      ${avg_win:.2f}")
        print(f"Average Loss:     ${avg_loss:.2f}")
        print(f"Profit Factor:    {profit_factor:.2f}")
        
        # Show recent trades
        print(f"\n--- RECENT TRADES (Last 5) ---")
        for trade in trades[-5:]:
            print(f"[{trade['entry_time']}] {trade['direction']} | "
                  f"Entry: {trade['entry_price']:.2f} | "
                  f"Exit: {trade['exit_price']:.2f} | "
                  f"P&L: ${trade['pnl_usd']:+.2f} | "
                  f"Balance: ${trade['balance']:.2f} | "
                  f"Exit: {trade['exit_reason']}")
    else:
        print("No trades executed during the period.")

if __name__ == "__main__":
    main()