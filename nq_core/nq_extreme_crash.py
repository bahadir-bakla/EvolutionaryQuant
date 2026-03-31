import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class NQCrashConfig:
    ema_fast: int = 5
    ema_slow: int = 15
    atr_stop_mult: float = 1.5   # Tight stop, quick cuts
    atr_tp_mult: float = 4.0     # Massive drops 

class NQ_Crash_Short_Strategy:
    """
    Specifically designed for the HMM EXTREME Regime on Nasdaq.
    This regime typically triggers during high-volatility crashes or blowoff tops.
    This strategy assumes Extreme = Crash, so it aggressively looks for Short momentum.
    Uses fast EMA crosses and expansion to snipe intraday plunge legs.
    """
    def __init__(self, config: NQCrashConfig = None):
        self.config = config or NQCrashConfig()
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'crash_ema_fast' not in df.columns:
            df['crash_ema_fast'] = df['close'].ewm(span=self.config.ema_fast, adjust=False).mean()
            df['crash_ema_slow'] = df['close'].ewm(span=self.config.ema_slow, adjust=False).mean()
        return df

    def evaluate(self, df: pd.DataFrame, idx: int):
        df = self.add_indicators(df)
        row = df.iloc[idx]
        close = row['close']
        atr = row.get('atr', 20)
        
        ema_f = row['crash_ema_fast']
        ema_s = row['crash_ema_slow']
        
        prev_f = df['crash_ema_fast'].iloc[idx-1] if idx > 0 else close
        prev_s = df['crash_ema_slow'].iloc[idx-1] if idx > 0 else close
        
        class Sig: pass
        signal = Sig()
        signal.direction = 'NEUTRAL'
        signal.stop_loss = 0.0
        signal.take_profit_2 = 0.0
        
        # AGGRESSIVE SHORT LOGIC
        # 1. EMA Fast crosses deeply below EMA Slow, or is expanding downwards rapidly
        is_bearish_expansion = (ema_f < ema_s) and ((prev_s - prev_f) < (ema_s - ema_f))
        
        # 2. Large red candle signaling panic selling
        is_red = close < row['open']
        body_size = row['open'] - close
        is_large_red = body_size > (atr * 0.8)
        
        # 3. Micro Pullback / Touch of fast EMA
        # Sometimes it's better to just short the breakdown in a crash
        is_breakdown = close < df['low'].iloc[idx-1]
        
        if is_bearish_expansion and is_red and is_breakdown:
            signal.direction = 'SHORT'
            # Tight stop above the crash candle
            signal.stop_loss = row['high'] + (atr * 0.5) 
            signal.take_profit_2 = close - (atr * self.config.atr_tp_mult)
            
        return signal
