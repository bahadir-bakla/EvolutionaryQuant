# NQ Quant Bot - Out-of-Sample Testing (2021)
# Testing strategy robustness on historical data

import numpy as np
import pandas as pd
import yfinance as yf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nq_core.optimized_strategy import OptimizedStrategy, add_optimized_indicators
from nq_core.backtest_intraday import AdvancedIntradayBacktest

def run_oos_test(year: int):
    print(f"\n{'='*60}")
    print(f"OUT-OF-SAMPLE TEST: YEAR {year} (Daily Data)")
    print(f"{'='*60}")
    print("(*) Note: 5m data not available for historical years via public API.")
    print("(*) Testing on Daily (1D) timeframe to verify trend logic.\n")
    
    # Fetch Data
    print(f"Fetching NQ=F data for {year}...")
    ticker = yf.Ticker("NQ=F")
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    df = ticker.history(start=start_date, end=end_date, interval="1d")
    df.columns = df.columns.str.lower()
    
    print(f"Bars: {len(df)}")
    
    if len(df) == 0:
        print(f"Error: No data found for {year}.")
        return
    
    # Add indicators
    print("Adding optimized indicators...")
    df = add_optimized_indicators(df)
    
    # Strategy (adjusted for Daily)
    strategy = OptimizedStrategy(
        atr_stop_mult=2.0,  # Wider stop for daily
        atr_tp_mult=4.0,    # Big trend targets
        min_rr=2.0
    )
    
    signals = []
    start_idx = 50
    for i in range(start_idx, len(df)):
        try:
            signal = strategy.evaluate(df, i)
            signals.append({
                'signal': signal.direction,
                'stop_loss': signal.stop_loss,
                'tp1': signal.take_profit_1,
                'tp2': signal.take_profit_2,
                'tp3': signal.take_profit_3,
                'atr': df.iloc[i].get('atr', 0)
            })
        except Exception as e:
            continue
    
    if not signals:
        print("No signals generated.")
        return

    signals_df = pd.DataFrame(signals, index=df.index[start_idx:start_idx+len(signals)])
    test_df = df.iloc[start_idx:start_idx+len(signals)]
    
    # Backtest
    print("Running backtest...")
    backtest = AdvancedIntradayBacktest(
        initial_capital=100000,
        risk_per_trade=0.02,     # 2% risk
        max_daily_trades=1,      # Daily chart
        trailing_stop_atr=2.5,   # Wide trail
        contract_value=20.0      # NQ
    )
    
    results = backtest.run(test_df, signals_df)
    backtest.print_results(results)
    
    return results

if __name__ == "__main__":
    # Run for requested years
    run_oos_test(2021)
    run_oos_test(2023)
