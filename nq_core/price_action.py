# Price Action Utilities for Gold Strategy
# FVG (Fair Value Gaps) & Liquidity Sweeps

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FairValueGap:
    index: int
    top: float
    bottom: float
    direction: str # 'BULLISH', 'BEARISH'
    filled: bool = False
    
def detect_fvgs(df: pd.DataFrame, min_gap_points: float = 0.5) -> List[FairValueGap]:
    """
    Detect Fair Value Gaps (Imbalances)
    
    Bullish FVG: Low of candle i+1 > High of candle i-1
    Bearish FVG: High of candle i+1 < Low of candle i-1
    (Note: Using i as the middle candle of the 3-candle formation)
    """
    fvgs = []
    
    # Needs at least 3 bars
    if len(df) < 3:
        return []
        
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    # Iterate from index 1 to len-2 (need neighbors)
    for i in range(1, len(df) - 1):
        # Bullish FVG
        # Candle i (middle) is usually large up candle
        # Gap between High[i-1] and Low[i+1]
        prev_high = highs[i-1]
        next_low = lows[i+1]
        
        if next_low > prev_high:
            gap_size = next_low - prev_high
            if gap_size >= min_gap_points:
                fvgs.append(FairValueGap(
                    index=i,
                    top=next_low,
                    bottom=prev_high,
                    direction='BULLISH'
                ))
                
        # Bearish FVG
        # Gap between Low[i-1] and High[i+1]
        prev_low = lows[i-1]
        next_high = highs[i+1]
        
        if next_high < prev_low:
            gap_size = prev_low - next_high
            if gap_size >= min_gap_points:
                fvgs.append(FairValueGap(
                    index=i,
                    top=prev_low,
                    bottom=next_high,
                    direction='BEARISH'
                ))
                
    return fvgs

def check_fvg_interaction(price: float, fvgs: List[FairValueGap]) -> Optional[FairValueGap]:
    """Check if price is filling an active FVG"""
    for fvg in fvgs:
        if fvg.filled:
            continue
            
        # Check fill
        if fvg.direction == 'BULLISH':
            # Price dips into gap
            if fvg.bottom <= price <= fvg.top:
                return fvg
        elif fvg.direction == 'BEARISH':
            # Price rallies into gap
            if fvg.bottom <= price <= fvg.top:
                return fvg
                
    return None
