# NQ Quant Bot - Intraday Trading Engine
# Optimized for 5m, 15m, 1h timeframes

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from enum import Enum
import yfinance as yf
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class IntradayMode(Enum):
    SCALP = "SCALP"       # 5m, tight stops, quick profits
    SWING = "SWING"       # 15m-1h, wider stops, bigger targets
    HYBRID = "HYBRID"     # Adaptive based on volatility


@dataclass
class IntradayConfig:
    """Configuration for intraday trading"""
    timeframe: str = "15m"
    
    # Signal thresholds
    min_confluence: int = 2          # Lower for more signals
    min_score: float = 1.5           # Lower threshold
    
    # Risk parameters
    atr_stop_mult: float = 1.5       # Tighter stops for intraday
    atr_tp_mult: float = 2.0         # Quicker TP
    risk_reward_min: float = 1.5     # Minimum R:R
    
    # Session filters
    trade_london: bool = True
    trade_ny: bool = True
    trade_asian: bool = False        # Usually avoid low vol
    trade_overlap: bool = True       # Best time!
    
    # Position sizing
    max_daily_trades: int = 10
    max_concurrent: int = 2
    risk_per_trade: float = 0.01     # 1% risk
    
    # Indicators
    ema_fast: int = 8
    ema_slow: int = 21
    rsi_period: int = 9              # Faster RSI for intraday
    atr_period: int = 10


@dataclass
class IntradaySignal:
    """Intraday trading signal"""
    direction: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    session: str
    is_high_volatility: bool
    vwap_bias: str
    structure_bias: str
    
    risk_reward: float
    expected_duration: int  # bars
    
    factors: Dict[str, str]


def fetch_intraday_data(
    symbol: str = "NQ=F",
    interval: str = "15m",
    days: int = 30
) -> pd.DataFrame:
    """
    Fetch intraday data
    
    Note: yfinance limits:
    - 1m: 7 days max
    - 5m, 15m, 30m: 60 days max
    - 1h: 730 days max
    """
    ticker = yf.Ticker(symbol)
    
    # Adjust period based on interval
    if interval in ["1m", "2m"]:
        period = "7d"
    elif interval in ["5m", "15m", "30m"]:
        period = f"{min(days, 60)}d"
    else:
        period = f"{min(days, 730)}d"
    
    df = ticker.history(period=period, interval=interval)
    df.columns = df.columns.str.lower()
    
    return df


def add_intraday_indicators(
    df: pd.DataFrame,
    config: IntradayConfig = None
) -> pd.DataFrame:
    """Add intraday-specific indicators"""
    if config is None:
        config = IntradayConfig()
    
    df = df.copy()
    
    # === EMAs ===
    df['ema_fast'] = df['close'].ewm(span=config.ema_fast).mean()
    df['ema_slow'] = df['close'].ewm(span=config.ema_slow).mean()
    df['ema_trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
    
    # === RSI (faster) ===
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(config.rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(config.rsi_period).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # === ATR ===
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(config.atr_period).mean()
    
    # === VWAP (reset each day) ===
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['typical_price'] * df['volume']
    
    if isinstance(df.index, pd.DatetimeIndex):
        df['date'] = df.index.date
        
        vwap_list = []
        for _, group in df.groupby('date'):
            cum_tp_vol = group['tp_volume'].cumsum()
            cum_vol = group['volume'].cumsum()
            vwap = cum_tp_vol / (cum_vol + 1e-10)
            vwap_list.extend(vwap.values)
        
        df['vwap'] = vwap_list
        df = df.drop(columns=['date', 'typical_price', 'tp_volume'])
    
    df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap'] * 100
    
    # === Session ===
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
        
        df['session'] = 'OFF'
        df.loc[(hour >= 0) & (hour < 9), 'session'] = 'ASIAN'
        df.loc[(hour >= 8) & (hour < 17), 'session'] = 'LONDON'
        df.loc[(hour >= 13) & (hour < 22), 'session'] = 'NY'
        df.loc[(hour >= 13) & (hour < 17), 'session'] = 'OVERLAP'
    
    # === Momentum ===
    df['momentum_3'] = df['close'].pct_change(3) * 100
    df['momentum_5'] = df['close'].pct_change(5) * 100
    
    # === Volatility relative ===
    df['vol_ratio'] = df['atr'] / df['atr'].rolling(50).mean()
    
    # === Higher highs / Lower lows ===
    df['hh'] = (df['high'] > df['high'].rolling(5).max().shift(1)).astype(int)
    df['ll'] = (df['low'] < df['low'].rolling(5).min().shift(1)).astype(int)
    
    # === Candle patterns ===
    body = df['close'] - df['open']
    df['body_size'] = body / (df['high'] - df['low'] + 1e-10)
    df['is_bullish'] = (body > 0).astype(int)
    
    # === Previous session high/low ===
    if isinstance(df.index, pd.DatetimeIndex):
        df['prev_high'] = df['high'].rolling(24).max().shift(1)  # ~1 day for 1h
        df['prev_low'] = df['low'].rolling(24).min().shift(1)
    
    return df


class IntradayConfluenceEngine:
    """
    Intraday-optimized confluence engine
    
    More aggressive signal generation for day trading
    """
    
    def __init__(self, config: IntradayConfig = None):
        self.config = config or IntradayConfig()
        
    def evaluate(
        self,
        df: pd.DataFrame,
        current_idx: int
    ) -> IntradaySignal:
        """Generate intraday signal"""
        row = df.iloc[current_idx]
        price = row['close']
        
        factors = {}
        long_score = 0.0
        short_score = 0.0
        
        # === 1. EMA TREND (strong filter) ===
        ema_trend = row.get('ema_trend', 0)
        if ema_trend == 1:
            long_score += 2.0
            factors['ema'] = 'LONG (fast > slow)'
        elif ema_trend == -1:
            short_score += 2.0
            factors['ema'] = 'SHORT (fast < slow)'
        
        # === 2. VWAP BIAS ===
        vwap_dist = row.get('vwap_dist', 0)
        if vwap_dist > 0.15:  # Must be clearly above
            long_score += 1.5
            factors['vwap'] = f'LONG (+{vwap_dist:.2f}%)'
        elif vwap_dist < -0.15:  # Must be clearly below
            short_score += 1.5
            factors['vwap'] = f'SHORT ({vwap_dist:.2f}%)'
        
        # === 3. RSI (extreme only) ===
        rsi = row.get('rsi', 50)
        if rsi < 25:  # Strongly oversold
            long_score += 1.5
            factors['rsi'] = f'LONG (oversold: {rsi:.0f})'
        elif rsi > 75:  # Strongly overbought
            short_score += 1.5
            factors['rsi'] = f'SHORT (overbought: {rsi:.0f})'
        elif 40 < rsi < 60:
            # Neutral zone - add small trend bias
            if ema_trend == 1:
                long_score += 0.3
            elif ema_trend == -1:
                short_score += 0.3
        
        # === 4. MOMENTUM CONFIRMATION ===
        mom = row.get('momentum_3', 0)
        if mom > 0.2 and ema_trend == 1:  # Momentum with trend
            long_score += 1.0
            factors['momentum'] = f'LONG (+{mom:.2f}%)'
        elif mom < -0.2 and ema_trend == -1:  # Momentum with trend
            short_score += 1.0
            factors['momentum'] = f'SHORT ({mom:.2f}%)'
        
        # === 5. STRUCTURE (HH/LL) ===
        if row.get('hh', 0) == 1 and ema_trend == 1:
            long_score += 1.0
            factors['structure'] = 'LONG (Higher High)'
        if row.get('ll', 0) == 1 and ema_trend == -1:
            short_score += 1.0
            factors['structure'] = 'SHORT (Lower Low)'
        
        # === 6. CANDLE STRENGTH ===
        body_size = abs(row.get('body_size', 0))
        if row.get('is_bullish', 0) == 1 and body_size > 0.7:
            long_score += 0.5
            factors['candle'] = 'LONG (strong bullish)'
        elif row.get('is_bullish', 0) == 0 and body_size > 0.7:
            short_score += 0.5
            factors['candle'] = 'SHORT (strong bearish)'
        
        # === 7. SESSION FILTER (critical for forex/futures) ===
        session = row.get('session', 'OFF')
        
        # Only trade during good sessions
        if session == 'OVERLAP':
            session_mult = 1.2  # Best time - boost signals
            factors['session'] = 'OVERLAP (prime time)'
        elif session == 'NY':
            session_mult = 1.0
            factors['session'] = 'NY session'
        elif session == 'LONDON':
            session_mult = 0.9
            factors['session'] = 'LONDON session'
        elif session == 'ASIAN':
            session_mult = 0.5  # Heavy penalty
            factors['session'] = 'ASIAN (avoid)'
        else:
            session_mult = 0.3  # Off hours
            factors['session'] = 'OFF HOURS'
        
        long_score *= session_mult
        short_score *= session_mult
        
        # === 8. VOLATILITY FILTER ===
        vol_ratio = row.get('vol_ratio', 1)
        if vol_ratio < 0.5:  # Too quiet
            long_score *= 0.5
            short_score *= 0.5
            factors['volatility'] = 'LOW (reduced size)'
        elif vol_ratio > 2.0:  # Too volatile
            long_score *= 0.7
            short_score *= 0.7
            factors['volatility'] = 'HIGH (caution)'
        
        # === CONFLUENCE CHECK ===
        # Count actual factors (not modifiers)
        factor_count = sum([
            abs(ema_trend) > 0,
            abs(vwap_dist) > 0.15,
            rsi < 25 or rsi > 75,
            abs(mom) > 0.2,
            row.get('hh', 0) == 1 or row.get('ll', 0) == 1,
            body_size > 0.7
        ])
        
        # === DETERMINE DIRECTION ===
        net_score = long_score - short_score
        min_score = self.config.min_score * 1.5  # Higher threshold
        
        # Need minimum factors AND score
        if net_score >= min_score and factor_count >= 3 and long_score > short_score:
            direction = 'LONG'
            confidence = min(1.0, net_score / 6)
        elif net_score <= -min_score and factor_count >= 3 and short_score > long_score:
            direction = 'SHORT'
            confidence = min(1.0, abs(net_score) / 6)
        else:
            direction = 'NEUTRAL'
            confidence = 0.0
        
        # === CALCULATE LEVELS ===
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
        
        # Risk/Reward
        risk = abs(price - stop)
        reward = abs(tp1 - price)
        rr = reward / (risk + 1e-10)
        
        return IntradaySignal(
            direction=direction,
            confidence=confidence,
            entry=price,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            session=session,
            is_high_volatility=vol_ratio > 1.2,
            vwap_bias='BULL' if vwap_dist > 0 else 'BEAR',
            structure_bias='UP' if row.get('hh', 0) else 'DOWN' if row.get('ll', 0) else 'NEUTRAL',
            risk_reward=rr,
            expected_duration=10,
            factors=factors
        )


def run_intraday_backtest(
    interval: str = "15m",
    days: int = 30
):
    """Run intraday backtest"""
    from nq_core.backtest import NQBacktestEngine
    
    print('='*60)
    print(f'NQ QUANT BOT - INTRADAY BACKTEST ({interval})')
    print('='*60)
    
    # Fetch data
    print(f'\n[1] Fetching NQ=F {interval} data ({days} days)...')
    df = fetch_intraday_data("NQ=F", interval, days)
    print(f'    Fetched {len(df)} bars')
    
    if len(df) < 100:
        print("Not enough data. Try a longer period or different interval.")
        return
    
    # Add indicators
    print('[2] Adding intraday indicators...')
    config = IntradayConfig(
        timeframe=interval,
        min_confluence=2,
        min_score=1.5,
        atr_stop_mult=1.5,
        atr_tp_mult=2.0
    )
    df = add_intraday_indicators(df, config)
    
    # Generate signals
    print('[3] Generating signals...')
    engine = IntradayConfluenceEngine(config)
    
    signals = []
    for i in range(50, len(df)):  # Skip first 50 for indicators
        signal = engine.evaluate(df, i)
        signals.append({
            'signal': signal.direction,
            'stop_loss': signal.stop_loss,
            'tp1': signal.take_profit_1,
            'atr': df.iloc[i].get('atr', 0)
        })
    
    signals_df = pd.DataFrame(signals, index=df.index[50:])
    
    # Signal distribution
    signal_counts = signals_df['signal'].value_counts()
    total = len(signals_df)
    
    print('\n    Signal distribution:')
    for sig, count in signal_counts.items():
        pct = count / total * 100
        print(f'      {sig}: {count} ({pct:.1f}%)')
    
    # Calculate signals per day
    if isinstance(df.index, pd.DatetimeIndex):
        trading_days = (df.index[-1] - df.index[0]).days + 1
        trades_per_day = len([s for s in signals if s['signal'] != 'NEUTRAL']) / trading_days
        print(f'\n    Potential trades per day: {trades_per_day:.1f}')
    
    # Run backtest
    print('\n[4] Running backtest...')
    test_df = df.iloc[50:]
    backtest = NQBacktestEngine(initial_capital=100000, use_kelly=True)
    result = backtest.run(test_df, signals_df)
    
    print(result)
    
    return result, signals_df


# === TEST ===
if __name__ == "__main__":
    # Test with 15m data
    result, signals = run_intraday_backtest(interval="15m", days=30)
