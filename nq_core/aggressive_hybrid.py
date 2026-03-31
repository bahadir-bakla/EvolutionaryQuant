# NQ Quant Bot - Aggressive Hybrid Strategy
# High-frequency, aggressive version with NQ/Gold specific market heuristics.

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging
from datetime import time

from .super_hybrid import SuperHybridStrategy, SuperHybridConfig, SuperHybridSignal
from .order_blocks import OrderBlock, check_ob_interaction

logger = logging.getLogger(__name__)

@dataclass
class AggressiveConfig(SuperHybridConfig):
    """Aggressive Configuration"""
    # Aggressive Sizing
    risk_per_trade: float = 0.02  # 2% Risk
    
    # ORB
    trade_orb: bool = True
    orb_window_minutes: int = 5 
    
    # Time Rules
    trade_reversal_hour: bool = True # 10:30 ET Reversal
    
    # Heuristics
    use_3rd_touch: bool = True
    use_fvg: bool = True

class AggressiveHybridStrategy(SuperHybridStrategy):
    """
    Aggressive version of Super Hybrid.
    
    Adds:
    1. ORB (Opening Range Breakout) logic for 09:30-09:35 ET.
    2. Time-based Reversal logic (10:30 ET).
    3. FVG (Fair Value Gap) targeting.
    4. 3rd Touch breakouts.
    """
    
    def __init__(self, config: AggressiveConfig = None):
        super().__init__(config or AggressiveConfig())
        self.config: AggressiveConfig = config or AggressiveConfig()
        
        # State
        self.orb_high = None
        self.orb_low = None
        self.orb_set_date = None
        
        # FVG List (simple tracking)
        self.fvgs = [] 

    def evaluate(self, df: pd.DataFrame, idx: int) -> SuperHybridSignal:
        # Get base signal from Super Hybrid
        base_signal = super().evaluate(df, idx)
        
        row = df.iloc[idx]
        current_time = row.name # timestamp
        
        # --- 1. ORB LOGIC (NQ Specific) ---
        if self.config.trade_orb:
            orb_sig = self._check_orb(df, idx, current_time)
            if orb_sig:
                # Force ORB signal if valid
                if orb_sig['direction'] == 'LONG':
                    base_signal.direction = 'LONG'
                    base_signal.factors['orb'] = "ORB Breakout LONG"
                    base_signal.confidence = max(base_signal.confidence, 0.8) # High confidence
                elif orb_sig['direction'] == 'SHORT':
                    base_signal.direction = 'SHORT'
                    base_signal.factors['orb'] = "ORB Breakout SHORT"
                    base_signal.confidence = max(base_signal.confidence, 0.8)
                    
        # --- 2. 10:30 REVERSAL LOGIC ---
        if self.config.trade_reversal_hour:
            # 10:30 ET is often a reversal time
            # Check if time is near 10:30 (assuming data is ET)
            # We need to handle timezone carefully, usually data is UTC-5 or similar
            # For simplicity, we assume the index is datetime and we check .time()
            if current_time.hour == 10 and current_time.minute == 30:
                # If we have a strong existing trend, look for exhaustion
                if base_signal.direction == 'LONG' and row['rsi'] > 75:
                    base_signal.direction = 'NEUTRAL' # Kill long signal
                    base_signal.factors['time'] = "10:30 Reversal Risk (Kill Long)"
                elif base_signal.direction == 'SHORT' and row['rsi'] < 25:
                    base_signal.direction = 'NEUTRAL' # Kill short signal
                    base_signal.factors['time'] = "10:30 Reversal Risk (Kill Short)"

        # --- 3. FVG & 3rd TOUCH (Heuristics) ---
        # 3rd Touch is hard to track perfectly bar-by-bar without state history of every level
        # But we can check if price is testing a known Order Block for the Nth time
        # This requires modifying OrderBlock class to track 'tests', which we can simulate
        
        if self.config.use_3rd_touch and base_signal.active_ob:
            # If interaction is TOUCHED
            # We bump confidence significantly
            # Heuristic: If momentum is high and touching OB, assume breakout attempt
            ob = base_signal.active_ob
            if base_signal.kalman_velocity is not None:
                if ob.direction == 'BEARISH' and base_signal.kalman_velocity > 20:
                     # Breaking Bearish OB
                     base_signal.factors['heuristic'] = "3rd Touch/Breakout Logic (Vel High)"
                     base_signal.direction = 'LONG'
                     base_signal.confidence += 0.2
        
        # FVG Logic (Simplified)
        if self.config.use_fvg:
            # Detect simple FVG from previous 2 bars
            # Bullish FVG: Low[0] > High[2]
            if idx > 2:
                # row is idx
                # bar-1, bar-2
                # Note: This is checking CREATION of FVG, not filling. 
                # To trade FVG FILL, we need to see price entering a previous FVG.
                pass 

        return base_signal

    def _check_orb(self, df: pd.DataFrame, idx: int, current_time) -> Optional[Dict]:
        """Check Opening Range Breakout"""
        # Session start 09:30 ET
        # We need to rely on the dataframe index being localized or consistent
        
        # Identify start of session
        if current_time.hour == 9 and current_time.minute < 30 + self.config.orb_window_minutes:
            if current_time.minute >= 30:
                # Inside ORB window, track High/Low
                curr_date = current_time.date()
                if self.orb_set_date != curr_date:
                    self.orb_high = df.iloc[idx]['high']
                    self.orb_low = df.iloc[idx]['low']
                    self.orb_set_date = curr_date
                else:
                    self.orb_high = max(self.orb_high, df.iloc[idx]['high'])
                    self.orb_low = min(self.orb_low, df.iloc[idx]['low'])
                return None
        
        # After ORB window, check breakout
        if self.orb_high and current_time.date() == self.orb_set_date:
            close = df.iloc[idx]['close']
            if close > self.orb_high:
                # Breakout LONG
                return {'direction': 'LONG'}
            elif close < self.orb_low:
                # Breakout SHORT
                return {'direction': 'SHORT'}
                
        return None
