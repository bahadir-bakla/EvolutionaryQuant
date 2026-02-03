# Backtest Demo Script
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import yfinance as yf
from nq_core.brain import QuantBrain
from nq_core.confluence import ConfluenceEngine
from nq_core.order_blocks import detect_order_blocks, get_active_order_blocks
from nq_core.backtest import NQBacktestEngine

def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calc_atr(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def main():
    # Fetch data
    print('Fetching NQ=F data...')
    ticker = yf.Ticker('NQ=F')
    df = ticker.history(period='6mo', interval='1d')
    df.columns = df.columns.str.lower()
    print(f'Fetched {len(df)} bars')
    
    # Calculate indicators
    df['rsi'] = calc_rsi(df['close'])
    df['atr'] = calc_atr(df)
    
    # Generate signals
    print('Generating signals...')
    brain = QuantBrain()
    confluence = ConfluenceEngine(min_confluence=2)  # Lower threshold for more signals
    obs = detect_order_blocks(df)
    
    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        state = brain.update(row['close'], df.index[i])
        active_obs = get_active_order_blocks(obs, i, max_age=50)
        rsi = row['rsi'] if pd.notna(row['rsi']) else None
        atr = row['atr'] if pd.notna(row['atr']) else None
        signal = confluence.evaluate(state, active_obs, rsi=rsi, atr=atr)
        signals.append({
            'signal': signal.direction,
            'stop_loss': signal.stop_loss,
            'tp1': signal.take_profit_1,
            'atr': atr if atr else 0
        })
    
    signals_df = pd.DataFrame(signals, index=df.index)
    
    # Count signals
    signal_counts = signals_df['signal'].value_counts()
    print('Signal distribution:')
    for sig, count in signal_counts.items():
        print(f'  {sig}: {count}')
    
    # Run backtest
    print('\nRunning backtest...')
    engine = NQBacktestEngine(initial_capital=100000, use_kelly=True)
    result = engine.run(df, signals_df)
    
    print(result)
    
    # Save equity curve
    result.equity_curve.to_csv('equity_curve.csv')
    print('\nEquity curve saved to equity_curve.csv')

if __name__ == '__main__':
    main()
