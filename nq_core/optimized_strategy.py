# NQ Quant Bot - Optimized High-Performance Strategy
# Target: 20-30% return with optimized features

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import yfinance as yf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nq_core.kalman_predict import KalmanPredictor, get_kalman_signal


@dataclass
class OptimizedSignal:
    """Optimized trading signal"""
    direction: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # Key factors used
    kalman_velocity: float
    vwap_dist: float
    ema_trend: int
    volatility_state: str
    
    risk_reward: float
    factors: Dict[str, str]


def add_optimized_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add only the most predictive indicators based on feature analysis"""
    df = df.copy()
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # === TOP FEATURES FROM ANALYSIS ===
    
    # 1. VWAP Distance (most important: 14.57%)
    tp = (high + low + close) / 3
    df['tp_vol'] = tp * volume
    df['vwap'] = df['tp_vol'].cumsum() / volume.cumsum()
    df['vwap_dist'] = (close - df['vwap']) / close * 100
    df = df.drop(columns=['tp_vol'])
    
    # 2. Volatility (12.59%)
    df['volatility'] = close.pct_change().rolling(10).std() * 100
    df['vol_avg'] = df['volatility'].rolling(50).mean()
    df['vol_state'] = np.where(df['volatility'] > df['vol_avg'] * 1.5, 'HIGH',
                              np.where(df['volatility'] < df['vol_avg'] * 0.5, 'LOW', 'NORMAL'))
    
    # 3. ATR Ratio (10.40%)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr'] / close * 100
    
    # 4. EMA 21 Distance (8.96%)
    df['ema_8'] = close.ewm(span=8).mean()
    df['ema_21'] = close.ewm(span=21).mean()
    df['ema_50'] = close.ewm(span=50).mean()
    df['ema_21_dist'] = (close - df['ema_21']) / close * 100
    df['ema_trend'] = np.where(df['ema_8'] > df['ema_21'], 1, -1)
    df['ema_stack'] = np.where(
        (df['ema_8'] > df['ema_21']) & (df['ema_21'] > df['ema_50']), 2,
        np.where((df['ema_8'] < df['ema_21']) & (df['ema_21'] < df['ema_50']), -2, 0)
    )
    
    # 5. RSI (8.81% - it IS useful!)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(9).mean()  # Faster RSI
    loss = (-delta.where(delta < 0, 0)).rolling(9).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 6. Volume Ratio (6.06%)
    df['volume_ratio'] = volume / volume.rolling(20).mean()
    
    # 7. BB Position (6.02%)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_position'] = (close - bb_mid) / (bb_std * 2 + 1e-10)
    
    # 8. Returns (for momentum)
    df['return_3d'] = close.pct_change(3) * 100
    df['return_5d'] = close.pct_change(5) * 100
    
    # 9. Higher Highs (still useful per correlation)
    df['hh'] = (high > high.rolling(5).max().shift(1)).astype(int)
    
    return df


class OptimizedStrategy:
    """
    High-performance strategy using only proven indicators
    
    Key signals:
    1. Kalman velocity + acceleration (trend prediction)
    2. VWAP distance (institutional level)
    3. EMA stack (trend confirmation)
    4. Volatility filter (timing)
    5. RSI extremes (reversals)
    """
    
    def __init__(
        self,
        atr_stop_mult: float = 1.2,
        atr_tp_mult: float = 3.0,  # Higher R:R for more profit
        min_rr: float = 2.0
    ):
        self.atr_stop = atr_stop_mult
        self.atr_tp = atr_tp_mult
        self.min_rr = min_rr
        
        self.kalman = KalmanPredictor(
            process_noise=0.005,
            measurement_noise=0.05,
            prediction_steps=5
        )
    
    def evaluate(self, df: pd.DataFrame, idx: int) -> OptimizedSignal:
        """Generate optimized signal"""
        row = df.iloc[idx]
        price = row['close']
        
        factors = {}
        long_score = 0.0
        short_score = 0.0
        
        # === 1. KALMAN VELOCITY (most important) ===
        kalman_pred = self.kalman.update(price)
        kalman_signal, kalman_weight = get_kalman_signal(self.kalman)
        
        if kalman_signal == 'LONG':
            long_score += 3.0 * kalman_weight
            factors['kalman'] = f'LONG (vel={kalman_pred.velocity:.2f})'
        elif kalman_signal == 'SHORT':
            short_score += 3.0 * kalman_weight
            factors['kalman'] = f'SHORT (vel={kalman_pred.velocity:.2f})'
        
        # === 2. VWAP DISTANCE (14.57% importance) ===
        vwap_dist = row.get('vwap_dist', 0)
        if vwap_dist > 0.3:  # Strong above VWAP
            long_score += 2.5
            factors['vwap'] = f'LONG (+{vwap_dist:.2f}%)'
        elif vwap_dist < -0.3:  # Strong below VWAP
            short_score += 2.5
            factors['vwap'] = f'SHORT ({vwap_dist:.2f}%)'
        elif vwap_dist > 0.1:
            long_score += 1.0
        elif vwap_dist < -0.1:
            short_score += 1.0
        
        # === 3. EMA STACK (trend confirmation) ===
        ema_stack = row.get('ema_stack', 0)
        if ema_stack == 2:  # Perfect bullish stack
            long_score += 2.0
            factors['ema'] = 'LONG (8>21>50)'
        elif ema_stack == -2:  # Perfect bearish stack
            short_score += 2.0
            factors['ema'] = 'SHORT (8<21<50)'
        elif row.get('ema_trend', 0) == 1:
            long_score += 0.5
        elif row.get('ema_trend', 0) == -1:
            short_score += 0.5
        
        # === 4. RSI (8.81% importance - extremes only) ===
        rsi = row.get('rsi', 50)
        if rsi < 25:
            long_score += 1.5  # Oversold reversal
            factors['rsi'] = f'LONG (oversold: {rsi:.0f})'
        elif rsi > 75:
            short_score += 1.5  # Overbought reversal
            factors['rsi'] = f'SHORT (overbought: {rsi:.0f})'
        
        # === 5. VOLATILITY FILTER ===
        vol_state = row.get('vol_state', 'NORMAL')
        if vol_state == 'HIGH':
            # Reduce position in high volatility
            long_score *= 0.7
            short_score *= 0.7
            factors['volatility'] = 'HIGH (reduced size)'
        elif vol_state == 'LOW':
            # Skip in very low volatility
            long_score *= 0.5
            short_score *= 0.5
            factors['volatility'] = 'LOW (skip)'
        
        # === 6. VOLUME CONFIRMATION ===
        vol_ratio = row.get('volume_ratio', 1)
        if vol_ratio > 1.5:
            # High volume confirms signal
            if long_score > short_score:
                long_score *= 1.3
            else:
                short_score *= 1.3
            factors['volume'] = 'HIGH (confirmed)'
        
        # === 7. HIGHER HIGH STRUCTURE ===
        if row.get('hh', 0) == 1 and long_score > short_score:
            long_score += 1.0
            factors['structure'] = 'LONG (HH)'
        
        # === DETERMINE DIRECTION ===
        min_score = 4.0  # Higher threshold for quality
        
        if long_score >= min_score and long_score > short_score:
            direction = 'LONG'
            confidence = min(1.0, long_score / 10)
        elif short_score >= min_score and short_score > long_score:
            direction = 'SHORT'
            confidence = min(1.0, short_score / 10)
        else:
            direction = 'NEUTRAL'
            confidence = 0.0
        
        # === CALCULATE LEVELS ===
        atr = row.get('atr', price * 0.002)
        
        if direction == 'LONG':
            stop = price - (atr * self.atr_stop)
            tp1 = price + (atr * self.atr_tp)
            tp2 = price + (atr * self.atr_tp * 1.5)
            tp3 = price + (atr * self.atr_tp * 2)
        elif direction == 'SHORT':
            stop = price + (atr * self.atr_stop)
            tp1 = price - (atr * self.atr_tp)
            tp2 = price - (atr * self.atr_tp * 1.5)
            tp3 = price - (atr * self.atr_tp * 2)
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


def run_optimized_backtest(interval: str = "5m", days: int = 7):
    """Run backtest with optimized strategy"""
    from nq_core.backtest_intraday import AdvancedIntradayBacktest
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZED STRATEGY BACKTEST ({interval})")
    print(f"{'='*60}")
    
    # Fetch data
    print(f"\n[1] Fetching {interval} data...")
    ticker = yf.Ticker("NQ=F")
    
    if interval in ["5m", "15m"]:
        period = f"{min(days, 60)}d"
    else:
        period = f"{days}d"
    
    df = ticker.history(period=period, interval=interval)
    df.columns = df.columns.str.lower()
    print(f"    Bars: {len(df)}")
    
    # Add indicators
    print("[2] Adding optimized indicators...")
    df = add_optimized_indicators(df)
    
    # Generate signals
    print("[3] Generating signals with Kalman + top features...")
    strategy = OptimizedStrategy(
        atr_stop_mult=1.2,
        atr_tp_mult=3.0,  # High R:R
        min_rr=2.0
    )
    
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
    
    # Backtest
    print("\n[4] Running backtest with trailing stops...")
    backtest = AdvancedIntradayBacktest(
        initial_capital=100000,
        risk_per_trade=0.015,  # 1.5% risk
        max_daily_trades=10,
        trailing_stop_atr=1.5,
    )
    
    results = backtest.run(test_df, signals_df)
    backtest.print_results(results)
    
    # Annualized projection
    trading_days = days
    annual_mult = 252 / trading_days
    projected_annual = results['total_return_pct'] * annual_mult
    
    print(f"\n[*] PROJECTED ANNUAL RETURN: {projected_annual:.1f}%")
    
    return results, backtest


if __name__ == "__main__":
    # Test optimized strategy
    results, backtest = run_optimized_backtest("5m", 7)
