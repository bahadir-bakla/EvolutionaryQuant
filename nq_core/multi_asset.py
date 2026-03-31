# NQ Quant Bot - Multi-Asset Enhanced Strategy
# Testing NVDA, Silver (SI=F), and others with aggressive optimization

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional
import yfinance as yf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nq_core.optimized_strategy import OptimizedStrategy, add_optimized_indicators, OptimizedSignal
from nq_core.backtest_intraday import AdvancedIntradayBacktest


@dataclass
class AssetConfig:
    symbol: str
    interval: str = "5m"
    days: int = 7
    
    # Risk mgmt
    atr_stop: float = 1.0  # Tighter stop for higher R:R
    atr_tp: float = 3.0    # 1:3 Target
    trailing_stop: float = 1.2
    
    # Strategy
    min_vwap_dist: float = 0.1
    min_kalman_conf: float = 0.5


def run_asset_test(config: AssetConfig):
    """Run backtest for a specific asset"""
    print(f"\n{'='*60}")
    print(f"TESTING {config.symbol} ({config.interval})")
    print(f"{'='*60}")
    
    # Fetch data
    print(f"Fetching data...")
    ticker = yf.Ticker(config.symbol)
    
    if config.interval in ["5m", "15m"]:
        period = "60d"
    else:
        period = "1y"
        
    df = ticker.history(period=period, interval=config.interval)
    df.columns = df.columns.str.lower()
    
    # Filter for last N days
    if len(df) > 0:
        cutoff = df.index[-1] - pd.Timedelta(days=config.days)
        df = df[df.index >= cutoff]
    
    print(f"Bars: {len(df)}")
    
    if len(df) < 100:
        print("Not enough data!")
        return None
    
    # Indicators
    print("Adding indicators...")
    df = add_optimized_indicators(df)
    
    # Strategy
    strategy = OptimizedStrategy(
        atr_stop_mult=config.atr_stop,
        atr_tp_mult=config.atr_tp,
        min_rr=3.0  # Aggressive R:R
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
            
    signals_df = pd.DataFrame(signals, index=df.index[start_idx:start_idx+len(signals)])
    test_df = df.iloc[start_idx:start_idx+len(signals)]
    
    # Backtest
    print("Running backtest...")
    backtest = AdvancedIntradayBacktest(
        initial_capital=100000,
        risk_per_trade=0.02,     # 2% risk per trade (aggressive)
        max_daily_trades=10,
        trailing_stop_atr=config.trailing_stop,
        contract_value=1.0 # Stock default
    )
    
    # Adjust contract value for features
    if "=F" in config.symbol:
        if "NQ" in config.symbol: backtest.contract_value = 20.0
        elif "GC" in config.symbol: backtest.contract_value = 100.0
        elif "SI" in config.symbol: backtest.contract_value = 5000.0 # Silver is $5000/point
        elif "CL" in config.symbol: backtest.contract_value = 1000.0 # Oil
    
    results = backtest.run(test_df, signals_df)
    
    # Print summary
    print(f"\nRESULTS for {config.symbol}:")
    print(f"Return: {results['total_return_pct']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Trades: {results['total_trades']}")
    
    return results


if __name__ == "__main__":
    assets = [
        AssetConfig(symbol="NVDA", interval="5m", days=7, atr_stop=1.2), # Wider stop for stocks
        AssetConfig(symbol="SI=F", interval="5m", days=7, atr_stop=0.8, atr_tp=3.5), # Aggressive Silver
        AssetConfig(symbol="NQ=F", interval="5m", days=7),     # Benchmark
    ]
    
    summary = []
    
    for asset in assets:
        res = run_asset_test(asset)
        if res:
            summary.append({
                'Symbol': asset.symbol,
                'Return': res['total_return_pct'],
                'WinRate': res['win_rate'],
                'PF': res['profit_factor'],
                'Trades': res['total_trades']
            })
            
    print("\n" + "="*60)
    print("MULTI-ASSET SUMMARY (7 Days - 5m)")
    print("="*60)
    print(f"{'Symbol':<10} {'Return':<10} {'WinRate':<10} {'PF':<10} {'Trades':<10}")
    print("-" * 50)
    for s in summary:
        print(f"{s['Symbol']:<10} {s['Return']:>8.2f}% {s['WinRate']:>8.1f}% {s['PF']:>8.2f} {s['Trades']:>8}")
