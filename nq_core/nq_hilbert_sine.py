import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class NQSineConfig:
    atr_stop_mult: float = 2.0
    atr_tp_mult: float = 1.5

class NQ_Hilbert_Sine_Strategy:
    """
    John Ehlers' Hilbert Sine Wave Strategy specifically for the HMM CHOP Regime.
    Executes mean-reversion cyclic trades when the phase crosses.
    Uses Daily Bias alignment when provided.
    """
    def __init__(self, config: NQSineConfig = None):
        self.config = config or NQSineConfig()

    def evaluate(self, df: pd.DataFrame, idx: int, daily_bias: str = 'UNKNOWN'):
        row = df.iloc[idx]
        close = row['close']
        atr = row.get('atr', 20)
        
        class Sig: pass
        signal = Sig()
        signal.direction = 'NEUTRAL'
        signal.stop_loss = 0.0
        signal.take_profit_2 = 0.0
        
        # We assume the suite pre-calculates the crossings
        cross_up = row.get('cross_up', False)
        cross_down = row.get('cross_down', False)
        
        # Execute in direction of daily bias if available, otherwise cyclic trading
        if cross_up and daily_bias in ['BULL', 'UNKNOWN']:
            signal.direction = 'LONG'
            signal.stop_loss = close - (atr * self.config.atr_stop_mult)
            signal.take_profit_2 = close + (atr * self.config.atr_tp_mult)
            
        elif cross_down and daily_bias in ['BEAR', 'UNKNOWN']:
            signal.direction = 'SHORT'
            signal.stop_loss = close + (atr * self.config.atr_stop_mult)
            signal.take_profit_2 = close - (atr * self.config.atr_tp_mult)
            
        return signal
