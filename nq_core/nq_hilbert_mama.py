import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.signal import hilbert

@dataclass
class NQHMM_MamaConfig:
    fast_limit: float = 0.5
    slow_limit: float = 0.05
    atr_stop_mult: float = 2.0
    atr_tp_mult_2: float = 5.0

class NQ_Hilbert_MAMA_Strategy:
    """
    John Ehlers' MESA Adaptive Moving Average (MAMA) Strategy.
    MAMA adapts to the phase rate of change derived from the Hilbert Transform.
    During Chop, MAMA and FAMA stay together (no signals).
    During Trend, Phase freezes, Alpha increases, MAMA pulls away from FAMA (strong trend signal).
    Perfectly aligned for the HMM 'TREND' Regime.
    """
    def __init__(self, config: NQHMM_MamaConfig = None):
        self.config = config or NQHMM_MamaConfig()
        
    def evaluate(self, df: pd.DataFrame, idx: int):
        # MAMA requires the entire array for Hilbert, which we already have computed
        # in the calling script as 'hilbert_phase'. But to calculate MAMA properly,
        # we need the phase rate of change. 
        # Since calculating MAMA bar-by-bar is complex, we assume the backtest script
        # has pre-calculated 'mama' and 'fama' columns for DataFrame-wide speed.
        
        row = df.iloc[idx]
        close = row['close']
        atr = row.get('atr', 20)
        
        mama = row.get('mama', close)
        fama = row.get('fama', close)
        prev_mama = df['mama'].iloc[idx-1] if idx > 0 else close
        prev_fama = df['fama'].iloc[idx-1] if idx > 0 else close
        
        class Sig: pass
        signal = Sig()
        signal.direction = 'NEUTRAL'
        signal.stop_loss = 0.0
        signal.take_profit_2 = 0.0
        
        # Trend Entry Logic: MAMA crosses FAMA
        # MAMA reacts faster to price in a trend
        cross_up = (mama > fama) and (prev_mama <= prev_fama)
        cross_down = (mama < fama) and (prev_mama >= prev_fama)
        
        if cross_up:
            signal.direction = 'LONG'
            signal.stop_loss = close - (atr * self.config.atr_stop_mult)
            signal.take_profit_2 = close + (atr * self.config.atr_tp_mult_2)
            
        elif cross_down:
            signal.direction = 'SHORT'
            signal.stop_loss = close + (atr * self.config.atr_stop_mult)
            signal.take_profit_2 = close - (atr * self.config.atr_tp_mult_2)
            
        return signal
