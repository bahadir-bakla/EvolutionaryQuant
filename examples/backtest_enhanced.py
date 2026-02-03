# NQ Quant Bot - Enhanced Backtest with All Indicators
import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import yfinance as yf
from nq_core.brain import QuantBrain
from nq_core.confluence_v2 import EnhancedConfluenceEngine
from nq_core.order_blocks import detect_order_blocks, get_active_order_blocks
from nq_core.indicators import (
    calculate_vwap, calculate_adx, detect_fvg,
    calculate_pivots, PivotType
)
from nq_core.indicators.market_structure import detect_bos_choch
from nq_core.backtest import NQBacktestEngine


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators to DataFrame"""
    df = df.copy()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # VWAP
    df = calculate_vwap(df, reset_daily=False)
    
    # ADX
    df = calculate_adx(df)
    
    # Market Structure
    df = detect_bos_choch(df)
    
    return df


def run_enhanced_backtest(period: str = '1y'):
    """Run backtest with enhanced confluence engine"""
    
    print('='*60)
    print('NQ QUANT BOT - ENHANCED BACKTEST')
    print('='*60)
    
    # Fetch data
    print('\n[1] Fetching NQ=F data...')
    ticker = yf.Ticker('NQ=F')
    df = ticker.history(period=period, interval='1d')
    df.columns = df.columns.str.lower()
    print(f'    Fetched {len(df)} bars')
    
    # Add indicators
    print('[2] Calculating indicators...')
    df = prepare_data(df)
    
    # Detect order blocks and FVGs
    print('[3] Detecting order blocks and FVGs...')
    order_blocks = detect_order_blocks(df)
    fvgs = detect_fvg(df)
    print(f'    Order Blocks: {len(order_blocks)}, FVGs: {len(fvgs)}')
    
    # Generate signals
    print('[4] Generating enhanced signals...')
    brain = QuantBrain(hurst_window=50)
    engine = EnhancedConfluenceEngine(min_factors=3, min_score=2.0)
    
    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Get brain state
        brain_state = brain.update(row['close'], df.index[i])
        
        # Get active order blocks
        active_obs = get_active_order_blocks(order_blocks, i, max_age=30)
        
        # Get pivots from previous day
        if i > 0:
            prev = df.iloc[i-1]
            pivots = calculate_pivots(prev['high'], prev['low'], prev['close'])
        else:
            pivots = None
        
        # Get signal
        signal = engine.evaluate(
            df, i, brain_state, active_obs, pivots, fvgs,
            rsi=row['rsi'] if pd.notna(row['rsi']) else None,
            atr=row['atr'] if pd.notna(row['atr']) else None
        )
        
        signals.append({
            'signal': signal.direction,
            'stop_loss': signal.stop_loss,
            'tp1': signal.take_profit_1,
            'atr': row['atr'] if pd.notna(row['atr']) else 0
        })
    
    signals_df = pd.DataFrame(signals, index=df.index)
    
    # Signal distribution
    signal_counts = signals_df['signal'].value_counts()
    print('\n    Signal distribution:')
    for sig, count in signal_counts.items():
        pct = count / len(signals_df) * 100
        print(f'      {sig}: {count} ({pct:.1f}%)')
    
    # Run backtest
    print('\n[5] Running backtest...')
    backtest = NQBacktestEngine(initial_capital=100000, use_kelly=True)
    result = backtest.run(df, signals_df)
    
    print(result)
    
    # Save results
    result.equity_curve.to_csv('enhanced_equity_curve.csv')
    signals_df.to_csv('enhanced_signals.csv')
    
    print('\nFiles saved:')
    print('  - enhanced_equity_curve.csv')
    print('  - enhanced_signals.csv')
    
    return result


if __name__ == '__main__':
    result = run_enhanced_backtest('1y')
