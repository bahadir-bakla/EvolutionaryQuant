import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class NQBreakdownConfig:
    lookback_bars: int = 50    # ~4 hours on 5m
    atr_stop_mult: float = 5.0 # Extremely wide stop (survive wicks)
    atr_tp_mult: float = 12.0  # Massive target for the crash

class NQ_Extreme_Breakdown_Strategy:
    """
    Designed for the HMM EXTREME Regime exactly based on the user's insight:
    Nasdaq crashes relentlessy without stopping. Instead of trying to snipe a micro-pullback
    (which gets squeezed), we just short the breakdown of the 4-hour macro support level.
    We use an extremely wide stop loss to survive the volatility bots, and an enormous Take Profit.
    """
    def __init__(self, config: NQBreakdownConfig = None):
        self.config = config or NQBreakdownConfig()
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'macro_low' not in df.columns:
            # Shift by 1 so current bar doesn't use its own low
            df['macro_low'] = df['low'].shift(1).rolling(self.config.lookback_bars).min()
        return df

    def evaluate(self, df: pd.DataFrame, idx: int):
        df = self.add_indicators(df)
        row = df.iloc[idx]
        close = row['close']
        atr = row.get('atr', 20)
        macro_low = row.get('macro_low', 0)
        
        class Sig: pass
        signal = Sig()
        signal.direction = 'NEUTRAL'
        signal.stop_loss = 0.0
        signal.take_profit_2 = 0.0
        
        # We need sufficient data
        if pd.isna(macro_low):
            return signal
            
        # BREAKDOWN LOGIC
        # If the close heavily breaks the 4-hour trailing support
        is_breakdown = close < macro_low
        is_red = close < row['open']
        
        # To avoid entering at the absolute bottom of a green squeeze candle,
        # we only enter if the 5m candle closed bearish
        if is_breakdown and is_red:
            signal.direction = 'SHORT'
            # Wide Stop Loss to survive the V-Shape Bounces
            signal.stop_loss = close + (atr * self.config.atr_stop_mult)
            # Massive Take Profit
            signal.take_profit_2 = close - (atr * self.config.atr_tp_mult)
            
        return signal
