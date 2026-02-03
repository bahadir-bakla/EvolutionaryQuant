# Final Validation with Optimized Parameters
import sys
sys.path.insert(0, '.')

import pandas as pd
import yfinance as yf
from nq_core.brain import QuantBrain
from nq_core.confluence import ConfluenceEngine
from nq_core.order_blocks import detect_order_blocks, get_active_order_blocks
from nq_core.backtest import NQBacktestEngine
from config.optimized_params import OPTIMIZED_PARAMS

def main():
    print('='*60)
    print('FINAL VALIDATION WITH OPTIMIZED PARAMETERS')
    print('='*60)
    print('\nOptimized Params:')
    for k, v in OPTIMIZED_PARAMS.items():
        print(f'  {k}: {v}')

    # Fetch data
    print('\nFetching NQ=F data (6mo)...')
    ticker = yf.Ticker('NQ=F')
    df = ticker.history(period='6mo', interval='1d')
    df.columns = df.columns.str.lower()
    print(f'Fetched {len(df)} bars')

    # Add indicators
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    # Generate signals with optimized params
    print('\nGenerating signals...')
    brain = QuantBrain(hurst_window=OPTIMIZED_PARAMS['hurst_window'])
    confluence = ConfluenceEngine(min_confluence=OPTIMIZED_PARAMS['min_confluence'])
    obs = detect_order_blocks(df)

    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        state = brain.update(row['close'], df.index[i])
        active_obs = get_active_order_blocks(obs, i, max_age=50)
        rsi = row['rsi'] if pd.notna(row['rsi']) else None
        atr = row['atr'] if pd.notna(row['atr']) else None
        signal = confluence.evaluate(state, active_obs, rsi=rsi, atr=atr)
        
        entry = row['close']
        if atr and signal.direction != 'NEUTRAL':
            if signal.direction == 'LONG':
                stop = entry - (atr * OPTIMIZED_PARAMS['atr_stop_multiplier'])
                tp = entry + (atr * OPTIMIZED_PARAMS['atr_tp_multiplier'])
            else:
                stop = entry + (atr * OPTIMIZED_PARAMS['atr_stop_multiplier'])
                tp = entry - (atr * OPTIMIZED_PARAMS['atr_tp_multiplier'])
        else:
            stop = signal.stop_loss
            tp = signal.take_profit_1
        
        signals.append({
            'signal': signal.direction, 
            'stop_loss': stop, 
            'tp1': tp, 
            'atr': atr if atr else 0
        })

    signals_df = pd.DataFrame(signals, index=df.index)
    
    # Signal counts
    signal_counts = signals_df['signal'].value_counts()
    print('Signal distribution:')
    for sig, count in signal_counts.items():
        print(f'  {sig}: {count}')

    # Run backtest
    print('\nRunning backtest...')
    engine = NQBacktestEngine(initial_capital=100000, use_kelly=True)
    result = engine.run(df, signals_df)
    print(result)
    
    # Comparison
    print('\n' + '='*60)
    print('COMPARISON: Before vs After Optimization')
    print('='*60)
    print('Metric               BEFORE      AFTER')
    print('-'*60)
    print(f'Sharpe Ratio         -2.13       {result.sharpe_ratio:.2f}')
    print(f'Total Return         -2.27%      {result.total_return:.2%}')
    print(f'Max Drawdown         -3.72%      {result.max_drawdown:.2%}')
    print(f'Win Rate             31.58%      {result.win_rate:.2%}')
    print(f'Profit Factor        0.57        {result.profit_factor:.2f}')

if __name__ == '__main__':
    main()
