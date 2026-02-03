# NQ Quant Bot - Gold Futures Trading
# Adapted for GC=F (Gold Futures) and XAUUSD

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict
import yfinance as yf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nq_core.kalman_predict import KalmanPredictor, get_kalman_signal
from nq_core.optimized_strategy import add_optimized_indicators, OptimizedSignal
from nq_core.backtest_intraday import AdvancedIntradayBacktest


@dataclass
class GoldConfig:
    """Configuration for Gold futures trading"""
    symbol: str = "GC=F"  # Gold Futures
    contract_value: float = 100.0  # Gold futures: $100 per point
    
    # Gold-specific parameters (different volatility profile)
    atr_stop_mult: float = 1.5  # Wider stops for gold
    atr_tp_mult: float = 2.5    # Good R:R
    
    # Gold responds well to these
    use_vwap: bool = True
    use_kalman: bool = True
    use_momentum: bool = True


class GoldStrategy:
    """
    Optimized strategy for Gold Futures
    
    Gold characteristics:
    - Safe haven asset (inversely correlated to USD sometimes)
    - High liquidity during London/NY overlap
    - Moves with inflation expectations
    - Strong trending behavior
    """
    
    def __init__(self, config: GoldConfig = None):
        self.config = config or GoldConfig()
        
        self.kalman = KalmanPredictor(
            process_noise=0.005,
            measurement_noise=0.05,
            prediction_steps=5
        )
    
    def evaluate(self, df: pd.DataFrame, idx: int) -> OptimizedSignal:
        """Generate gold signal"""
        row = df.iloc[idx]
        price = row['close']
        
        factors = {}
        long_score = 0.0
        short_score = 0.0
        
        # === 1. KALMAN VELOCITY ===
        kalman_pred = self.kalman.update(price)
        kalman_signal, kalman_weight = get_kalman_signal(self.kalman)
        
        if kalman_signal == 'LONG':
            long_score += 3.0 * kalman_weight
            factors['kalman'] = f'LONG (vel={kalman_pred.velocity:.2f})'
        elif kalman_signal == 'SHORT':
            short_score += 3.0 * kalman_weight
            factors['kalman'] = f'SHORT (vel={kalman_pred.velocity:.2f})'
        
        # === 2. VWAP DISTANCE ===
        vwap_dist = row.get('vwap_dist', 0)
        if vwap_dist > 0.2:
            long_score += 2.0
            factors['vwap'] = f'LONG (+{vwap_dist:.2f}%)'
        elif vwap_dist < -0.2:
            short_score += 2.0
            factors['vwap'] = f'SHORT ({vwap_dist:.2f}%)'
        
        # === 3. EMA STACK ===
        ema_stack = row.get('ema_stack', 0)
        if ema_stack == 2:
            long_score += 2.0
            factors['ema'] = 'LONG (8>21>50)'
        elif ema_stack == -2:
            short_score += 2.0
            factors['ema'] = 'SHORT (8<21<50)'
        
        # === 4. RSI EXTREMES ===
        rsi = row.get('rsi', 50)
        if rsi < 30:
            long_score += 1.5
            factors['rsi'] = f'LONG (oversold: {rsi:.0f})'
        elif rsi > 70:
            short_score += 1.5
            factors['rsi'] = f'SHORT (overbought: {rsi:.0f})'
        
        # === 5. MOMENTUM ===
        return_5d = row.get('return_5d', 0)
        if return_5d > 1.0:  # Strong bullish momentum
            long_score += 1.5
            factors['momentum'] = f'LONG (+{return_5d:.2f}%)'
        elif return_5d < -1.0:
            short_score += 1.5
            factors['momentum'] = f'SHORT ({return_5d:.2f}%)'
        
        # === 6. VOLATILITY FILTER ===
        vol_state = row.get('vol_state', 'NORMAL')
        if vol_state == 'HIGH':
            long_score *= 0.8
            short_score *= 0.8
            factors['volatility'] = 'HIGH'
        
        # === 7. VOLUME ===
        vol_ratio = row.get('volume_ratio', 1)
        if vol_ratio > 1.3:
            if long_score > short_score:
                long_score *= 1.2
            else:
                short_score *= 1.2
        
        # === DIRECTION ===
        min_score = 3.5  # Slightly lower for gold
        
        if long_score >= min_score and long_score > short_score:
            direction = 'LONG'
            confidence = min(1.0, long_score / 10)
        elif short_score >= min_score and short_score > long_score:
            direction = 'SHORT'
            confidence = min(1.0, short_score / 10)
        else:
            direction = 'NEUTRAL'
            confidence = 0.0
        
        # === LEVELS ===
        atr = row.get('atr', price * 0.002)
        
        if direction == 'LONG':
            stop = price - (atr * self.config.atr_stop_mult)
            tp1 = price + (atr * self.config.atr_tp_mult)
            tp2 = price + (atr * self.config.atr_tp_mult * 1.5)
            tp3 = price + (atr * self.config.atr_tp_mult * 2)
        elif direction == 'SHORT':
            stop = price + (atr * self.config.atr_stop_mult)
            tp1 = price - (atr * self.config.atr_tp_mult)
            tp2 = price - (atr * self.config.atr_tp_mult * 1.5)
            tp3 = price - (atr * self.config.atr_tp_mult * 2)
        else:
            stop = tp1 = tp2 = tp3 = price
        
        risk = abs(price - stop)
        reward = abs(tp1 - price)
        rr = reward / (risk + 1e-10)
        
        return OptimizedSignal(
            direction=direction,
            confidence=confidence,
            entry=price,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            kalman_velocity=kalman_pred.velocity,
            vwap_dist=vwap_dist,
            ema_trend=int(row.get('ema_trend', 0)),
            volatility_state=vol_state,
            risk_reward=rr,
            factors=factors
        )


def run_gold_backtest(interval: str = "15m", days: int = 30):
    """Run backtest on Gold Futures"""
    
    print(f"\n{'='*60}")
    print(f"GOLD FUTURES STRATEGY BACKTEST ({interval})")
    print(f"{'='*60}")
    
    # Fetch data
    print(f"\n[1] Fetching GC=F (Gold Futures) {interval} data...")
    ticker = yf.Ticker("GC=F")
    
    if interval in ["5m", "15m"]:
        period = f"{min(days, 60)}d"
    else:
        period = f"{days}d"
    
    df = ticker.history(period=period, interval=interval)
    df.columns = df.columns.str.lower()
    print(f"    Bars: {len(df)}")
    
    if len(df) < 100:
        print("Not enough data!")
        return None, None
    
    # Add indicators
    print("[2] Adding optimized indicators...")
    df = add_optimized_indicators(df)
    
    # Generate signals
    print("[3] Generating signals...")
    config = GoldConfig()
    strategy = GoldStrategy(config)
    
    signals = []
    start_idx = 50
    for i in range(start_idx, len(df)):
        signal = strategy.evaluate(df, i)
        signals.append({
            'signal': signal.direction,
            'stop_loss': signal.stop_loss,
            'tp1': signal.take_profit_1,
            'tp2': signal.take_profit_2,
            'tp3': signal.take_profit_3,
            'atr': df.iloc[i].get('atr', 0)
        })
    
    signals_df = pd.DataFrame(signals, index=df.index[start_idx:])
    test_df = df.iloc[start_idx:]
    
    # Stats
    signal_counts = signals_df['signal'].value_counts()
    print("\n    Signal Distribution:")
    for sig, count in signal_counts.items():
        print(f"      {sig}: {count} ({count/len(signals_df)*100:.1f}%)")
    
    # Backtest (Gold contract = $100 per point)
    print("\n[4] Running backtest...")
    backtest = AdvancedIntradayBacktest(
        initial_capital=100000,
        risk_per_trade=0.015,  # 1.5%
        max_daily_trades=10,
        trailing_stop_atr=1.8,
        contract_value=100.0  # Gold futures
    )
    
    results = backtest.run(test_df, signals_df)
    backtest.print_results(results)
    
    # Projection
    trading_days = days
    annual_mult = 252 / trading_days
    projected_annual = results['total_return_pct'] * annual_mult
    
    print(f"\n[*] PROJECTED ANNUAL RETURN: {projected_annual:.1f}%")
    
    return results, backtest


def compare_assets():
    """Compare NQ and Gold performance"""
    from nq_core.optimized_strategy import run_optimized_backtest
    
    print("\n" + "="*70)
    print("ASSET COMPARISON: NQ vs GOLD")
    print("="*70)
    
    # NQ (5m, 7 days)
    print("\n>>> NQ Futures (NQ=F)")
    nq_results, _ = run_optimized_backtest("5m", 7)
    
    # Gold (15m, 30 days - more data available)
    print("\n>>> Gold Futures (GC=F)")
    gold_results, _ = run_gold_backtest("15m", 30)
    
    if gold_results is None:
        print("Could not get Gold data")
        return
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"{'Metric':<20} {'NQ (5m/7d)':<20} {'Gold (15m/30d)':<20}")
    print("-"*60)
    print(f"{'Return':<20} {nq_results['total_return_pct']:.2f}%{'':<15} {gold_results['total_return_pct']:.2f}%")
    print(f"{'Trades':<20} {nq_results['total_trades']:<20} {gold_results['total_trades']:<20}")
    print(f"{'Win Rate':<20} {nq_results['win_rate']:.1f}%{'':<15} {gold_results['win_rate']:.1f}%")
    print(f"{'Profit Factor':<20} {nq_results['profit_factor']:.2f}{'':<17} {gold_results['profit_factor']:.2f}")
    print(f"{'Max Drawdown':<20} {nq_results['max_drawdown_pct']:.2f}%{'':<15} {gold_results['max_drawdown_pct']:.2f}%")


if __name__ == "__main__":
    # Test Gold strategy
    print("Testing Gold Futures Strategy...")
    results, backtest = run_gold_backtest("15m", 30)
    
    print("\n" + "="*60)
    print("Also testing 5m timeframe for Gold...")
    results_5m, _ = run_gold_backtest("5m", 7)
