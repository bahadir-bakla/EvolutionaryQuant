import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class NQHMM_ExtremeConfig:
    fisher_len: int = 10
    fisher_trigger: float = 1.0  # SD limit for extreme
    atr_stop_mult: float = 2.5   # Very wide sl for extreme vol
    atr_tp_mult_2: float = 5.0   # Massive target

class NQ_Extreme_Fisher_Strategy:
    """
    Designed for the HMM EXTREME (High Volatility/Crash) Regime on Nasdaq.
    Uses the Ehlers Fisher Transform to normalize prices into a Gaussian distribution.
    When the transform hits extreme standard deviations (> 2.0 or < -2.0) and crosses its trigger,
    it signals a violent reversal or momentum burst typical of V-shape crash environments.
    """
    def __init__(self, config: NQHMM_ExtremeConfig = None):
        self.config = config or NQHMM_ExtremeConfig()
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'fisher' in df.columns:
            return df
            
        # Fisher Transform Calculation
        # Normalize prices to -1 to +1
        # Fisher = 0.5 * ln((1 + X) / (1 - X))
        
        high_roll = df['high'].rolling(self.config.fisher_len).max()
        low_roll = df['low'].rolling(self.config.fisher_len).min()
        
        # Avoid division by zero
        denom = (high_roll - low_roll)
        denom = denom.replace(0, 0.0001)
        
        # Value = 2 * ((Close - Min) / (Max - Min)) - 1
        val = 2.0 * ((df['close'] - low_roll) / denom) - 1.0
        
        # Smooth the value
        val_s = val.ewm(alpha=0.5, adjust=False).mean()
        # Cap to avoid log errors
        val_s = val_s.clip(-0.999, 0.999)
        
        fisher = [0.0] * len(df)
        trigger = [0.0] * len(df)
        
        val_arr = val_s.values
        for i in range(1, len(df)):
            # Fisher Transform
            fisher[i] = 0.5 * np.log((1 + val_arr[i]) / (1 - val_arr[i])) + 0.5 * fisher[i-1]
            trigger[i] = fisher[i-1]
            
        df['fisher'] = fisher
        df['fisher_trigger'] = trigger
        return df

    def evaluate(self, df: pd.DataFrame, idx: int):
        df = self.add_indicators(df)
        row = df.iloc[idx]
        close = row['close']
        atr = row.get('atr', 20)
        
        fish = row.get('fisher', 0.0)
        trig = row.get('fisher_trigger', 0.0)
        
        prev_fish = df['fisher'].iloc[idx-1] if idx > 0 else 0.0
        prev_trig = df['fisher_trigger'].iloc[idx-1] if idx > 0 else 0.0
        
        class Sig: pass
        signal = Sig()
        signal.direction = 'NEUTRAL'
        signal.stop_loss = 0.0
        signal.take_profit_2 = 0.0
        
        # EXTREME REVERSAL LOGIC
        # If Fisher was deeply negative (oversold crash) and crosses up over its trigger
        cross_up = (fish > trig) and (prev_fish <= prev_trig) and (prev_fish < -self.config.fisher_trigger)
        
        # If Fisher was deeply positive (blowoff top) and crosses down
        cross_down = (fish < trig) and (prev_fish >= prev_trig) and (prev_fish > self.config.fisher_trigger)
        
        if cross_up:
            signal.direction = 'LONG'
            signal.stop_loss = close - (atr * self.config.atr_stop_mult)
            signal.take_profit_2 = close + (atr * self.config.atr_tp_mult_2)
            
        elif cross_down:
            signal.direction = 'SHORT'
            signal.stop_loss = close + (atr * self.config.atr_stop_mult)
            signal.take_profit_2 = close - (atr * self.config.atr_tp_mult_2)
            
        return signal
