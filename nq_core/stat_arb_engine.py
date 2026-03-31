import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class StatArbSignal:
    direction: str       # 'LONG_NQ_SHORT_ES', 'SHORT_NQ_LONG_ES', 'NEUTRAL', 'EXIT'
    z_score: float
    spread_value: float
    confidence: float

class StatArbEngine:
    """
    Statistical Arbitrage (Pairs Trading) Engine
    Monitors the historical spread between NQ and ES. 
    If the spread deviates by a specified Z-Score (e.g., > 2.0 or < -2.0),
    a signal is fired to short the outperformer and long the underperformer,
    expecting them to mean-revert back to the 0-line.
    """
    
    def __init__(self, ema_lookback=20, entry_z_score=2.0, exit_z_score=0.5):
        self.ema_lookback = ema_lookback
        self.entry_z_score = entry_z_score
        self.exit_z_score = exit_z_score
        
        self.current_position = None  # None, 'LONG_NQ', 'SHORT_NQ'
        
    def evaluate(self, nq_df: pd.DataFrame, es_df: pd.DataFrame) -> StatArbSignal:
        """
        Calculates the Z-Score of the Spread and outputs a hedge signal.
        Expects correctly aligned Pandas DataFrames.
        """
        # If toggled off or missing data
        if nq_df is None or es_df is None or len(nq_df) < self.ema_lookback:
            return StatArbSignal('NEUTRAL', 0, 0, 0)
            
        # Calculate the raw spread (Using ratio to normalize standard price differences)
        # Ratio = NQ Price / ES Price 
        spread = nq_df['close'] / es_df['close']
        
        # Calculate Rolling Z-Score
        spread_mean = spread.rolling(window=self.ema_lookback).mean()
        spread_std = spread.rolling(window=self.ema_lookback).std()
        
        current_spread = spread.iloc[-1]
        mean = spread_mean.iloc[-1]
        std = spread_std.iloc[-1]
        
        if pd.isna(std) or std == 0:
            return StatArbSignal('NEUTRAL', 0, current_spread, 0)
            
        z_score = (current_spread - mean) / std
        
        # Execution Logic
        signal = 'NEUTRAL'
        conf = 0.0
        
        if self.current_position is None:
            # Spread is too high -> NQ is too expensive relative to ES
            if z_score >= self.entry_z_score:
                signal = 'SHORT_NQ_LONG_ES'
                self.current_position = 'SHORT_NQ'
                conf = min(1.0, z_score / 3.0)
                
            # Spread is too low -> NQ is too cheap relative to ES
            elif z_score <= -self.entry_z_score:
                signal = 'LONG_NQ_SHORT_ES'
                self.current_position = 'LONG_NQ'
                conf = min(1.0, abs(z_score) / 3.0)
                
        else:
            # We are in a position, check for mean-reversion exit
            if self.current_position == 'SHORT_NQ' and z_score <= self.exit_z_score:
                signal = 'EXIT'
                self.current_position = None
            elif self.current_position == 'LONG_NQ' and z_score >= -self.exit_z_score:
                signal = 'EXIT'
                self.current_position = None
                
        return StatArbSignal(signal, z_score, current_spread, conf)
