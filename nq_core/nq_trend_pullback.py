import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class NQTrendConfig:
    ema_fast: int = 9
    ema_slow: int = 21
    pullback_tolerance: float = 0.5 # Points away from VWAP/EMA to be considered a pullback
    atr_stop_mult: float = 1.5
    atr_tp_mult_1: float = 2.0
    atr_tp_mult_2: float = 4.0

class NQ_Pullback_Strategy:
    """
    Specifically designed for the HMM Trend Regime on Nasdaq.
    waits for structural trend confirmation (EMA stack & VWAP alignment)
    and enters on micro-pullbacks to the fast EMA or VWAP.
    """
    def __init__(self, config: NQTrendConfig = None):
        self.config = config or NQTrendConfig()
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'ema_fast' not in df.columns:
            df['ema_fast'] = df['close'].ewm(span=self.config.ema_fast, adjust=False).mean()
        if 'ema_slow' not in df.columns:
            df['ema_slow'] = df['close'].ewm(span=self.config.ema_slow, adjust=False).mean()
            
        # Calculate VWAP if not exists
        if 'vwap' not in df.columns:
            # Simplistic session VWAP approximation (rolling 288 bars = 24h on 5m)
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['cv'] = df['typical_price'] * df['volume']
            df['vwap'] = df['cv'].rolling(window=288, min_periods=1).sum() / df['volume'].rolling(window=288, min_periods=1).sum()
            
        return df

    def evaluate(self, df: pd.DataFrame, idx: int):
        df = self.add_indicators(df)
        row = df.iloc[idx]
        
        # State
        close = row['close']
        vwap = row.get('vwap', close)
        ema_f = row.get('ema_fast', close)
        ema_s = row.get('ema_slow', close)
        atr = row.get('atr', 20)
        
        class Sig: pass
        signal = Sig()
        signal.direction = 'NEUTRAL'
        signal.stop_loss = 0.0
        signal.take_profit_2 = 0.0
        
        # 1. Structural Trend Filter
        is_bull_trend = (ema_f > ema_s) and (close > vwap)
        is_bear_trend = (ema_f < ema_s) and (close < vwap)
        
        # 2. Pullback Condition
        # If in a Bull Trend, look for a red candle that pulls back near or touches the Fast EMA
        is_red = close < row['open']
        is_green = close > row['open']
        
        # Did the low of this candle touch or come very close to the EMA fast?
        near_ema_f = abs(row['low'] - ema_f) < self.config.pullback_tolerance * atr
        near_ema_bull = row['low'] <= ema_f and close >= ema_f # Touched and rejected
        
        near_ema_bear = row['high'] >= ema_f and close <= ema_f # Touched and rejected
        
        if is_bull_trend and is_red and (near_ema_f or near_ema_bull):
            signal.direction = 'LONG'
            signal.stop_loss = close - (atr * self.config.atr_stop_mult)
            signal.take_profit_2 = close + (atr * self.config.atr_tp_mult_2)
            
        elif is_bear_trend and is_green and near_ema_bear:
            signal.direction = 'SHORT'
            signal.stop_loss = close + (atr * self.config.atr_stop_mult)
            signal.take_profit_2 = close - (atr * self.config.atr_tp_mult_2)
            
        return signal
