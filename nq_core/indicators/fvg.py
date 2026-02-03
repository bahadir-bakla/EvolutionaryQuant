# NQ Quant Bot - Fair Value Gaps (FVG)
# Imbalance zones where price tends to return

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class FVGType(Enum):
    BULLISH = "BULLISH"  # Gap up - expect pullback to fill
    BEARISH = "BEARISH"  # Gap down - expect rally to fill


@dataclass
class FairValueGap:
    """Fair Value Gap (Imbalance Zone)"""
    index: int
    fvg_type: FVGType
    top: float         # Top of the gap
    bottom: float      # Bottom of the gap
    midpoint: float    # Midpoint (often acts as magnet)
    size: float        # Size of the gap
    size_percent: float
    filled: bool = False
    fill_percent: float = 0.0
    timestamp: Optional[pd.Timestamp] = None


def detect_fvg(
    df: pd.DataFrame,
    min_gap_percent: float = 0.001,  # 0.1% minimum gap
    lookback: int = 100
) -> List[FairValueGap]:
    """
    Detect Fair Value Gaps (Imbalances)
    
    Bullish FVG: Low of bar1 > High of bar3 (gap between bar2 high reach and bar2 low reach)
    - Current candle (bar3) low is above previous-previous candle (bar1) high
    
    Bearish FVG: High of bar1 < Low of bar3
    - Current candle (bar3) high is below previous-previous candle (bar1) low
    
    Actually using standard definition:
    Bullish FVG: bar[i-2] high < bar[i] low (gap left by strong move up)
    Bearish FVG: bar[i-2] low > bar[i] high (gap left by strong move down)
    """
    fvgs = []
    
    start_idx = max(2, len(df) - lookback)
    
    for i in range(start_idx, len(df)):
        bar_1 = df.iloc[i - 2]  # Two bars ago
        bar_2 = df.iloc[i - 1]  # Previous bar (the impulse)
        bar_3 = df.iloc[i]      # Current bar
        
        # Bullish FVG: Gap left after strong bullish move
        # bar_1 high < bar_3 low (there's a gap)
        if bar_1['high'] < bar_3['low']:
            gap_bottom = bar_1['high']
            gap_top = bar_3['low']
            gap_size = gap_top - gap_bottom
            gap_percent = gap_size / bar_1['high']
            
            if gap_percent >= min_gap_percent:
                fvgs.append(FairValueGap(
                    index=i - 1,  # FVG is at bar_2 (impulse bar)
                    fvg_type=FVGType.BULLISH,
                    top=gap_top,
                    bottom=gap_bottom,
                    midpoint=(gap_top + gap_bottom) / 2,
                    size=gap_size,
                    size_percent=gap_percent,
                    timestamp=df.index[i-1] if isinstance(df.index, pd.DatetimeIndex) else None
                ))
        
        # Bearish FVG: Gap left after strong bearish move
        # bar_1 low > bar_3 high (there's a gap)
        if bar_1['low'] > bar_3['high']:
            gap_top = bar_1['low']
            gap_bottom = bar_3['high']
            gap_size = gap_top - gap_bottom
            gap_percent = gap_size / bar_1['low']
            
            if gap_percent >= min_gap_percent:
                fvgs.append(FairValueGap(
                    index=i - 1,
                    fvg_type=FVGType.BEARISH,
                    top=gap_top,
                    bottom=gap_bottom,
                    midpoint=(gap_top + gap_bottom) / 2,
                    size=gap_size,
                    size_percent=gap_percent,
                    timestamp=df.index[i-1] if isinstance(df.index, pd.DatetimeIndex) else None
                ))
    
    return fvgs


def update_fvg_fill_status(
    fvgs: List[FairValueGap],
    df: pd.DataFrame,
    current_idx: int
) -> List[FairValueGap]:
    """
    Update FVG fill status based on price action
    
    FVG is considered filled when price enters the gap zone
    """
    for fvg in fvgs:
        if fvg.filled:
            continue
        
        # Check if current price has entered the gap
        current_high = df.iloc[current_idx]['high']
        current_low = df.iloc[current_idx]['low']
        
        if fvg.fvg_type == FVGType.BULLISH:
            # Bullish FVG filled when price pulls back into the gap
            if current_low <= fvg.top:
                # Calculate fill percentage
                if current_low <= fvg.bottom:
                    fvg.fill_percent = 1.0
                    fvg.filled = True
                else:
                    fvg.fill_percent = (fvg.top - current_low) / fvg.size
        
        elif fvg.fvg_type == FVGType.BEARISH:
            # Bearish FVG filled when price rallies into the gap
            if current_high >= fvg.bottom:
                if current_high >= fvg.top:
                    fvg.fill_percent = 1.0
                    fvg.filled = True
                else:
                    fvg.fill_percent = (current_high - fvg.bottom) / fvg.size
    
    return fvgs


def get_active_fvgs(
    fvgs: List[FairValueGap],
    current_idx: int,
    max_age: int = 50,
    include_partially_filled: bool = True
) -> List[FairValueGap]:
    """Get active (unfilled or partially filled) FVGs within max_age bars"""
    active = []
    
    for fvg in fvgs:
        age = current_idx - fvg.index
        if age > max_age:
            continue
        
        if not fvg.filled:
            active.append(fvg)
        elif include_partially_filled and fvg.fill_percent < 1.0:
            active.append(fvg)
    
    return active


def check_fvg_interaction(
    price: float,
    fvgs: List[FairValueGap],
    tolerance: float = 0.001
) -> Optional[FairValueGap]:
    """
    Check if price is interacting with any FVG
    
    Returns the FVG if price is inside or near it
    """
    for fvg in fvgs:
        if fvg.filled:
            continue
        
        tolerance_amount = fvg.midpoint * tolerance
        
        # Check if price is inside FVG
        if fvg.bottom - tolerance_amount <= price <= fvg.top + tolerance_amount:
            return fvg
    
    return None


def get_fvg_targets(
    current_price: float,
    fvgs: List[FairValueGap],
    direction: str  # 'long' or 'short'
) -> List[float]:
    """
    Get FVG midpoints as potential targets
    
    For longs: return bearish FVG midpoints above current price
    For shorts: return bullish FVG midpoints below current price
    """
    targets = []
    
    for fvg in fvgs:
        if fvg.filled:
            continue
        
        if direction == 'long' and fvg.fvg_type == FVGType.BEARISH:
            if fvg.midpoint > current_price:
                targets.append(fvg.midpoint)
        
        elif direction == 'short' and fvg.fvg_type == FVGType.BULLISH:
            if fvg.midpoint < current_price:
                targets.append(fvg.midpoint)
    
    # Sort by proximity
    targets.sort(key=lambda x: abs(x - current_price))
    return targets


# === TEST ===
if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing Fair Value Gap detection...")
    ticker = yf.Ticker("NQ=F")
    df = ticker.history(period="3mo", interval="1d")
    df.columns = df.columns.str.lower()
    
    # Detect FVGs
    fvgs = detect_fvg(df, min_gap_percent=0.002)
    
    print(f"Found {len(fvgs)} FVGs")
    
    # Get active FVGs
    active = get_active_fvgs(fvgs, len(df) - 1, max_age=30)
    print(f"Active FVGs (last 30 bars): {len(active)}")
    
    # Show recent FVGs
    print("\nRecent FVGs:")
    for fvg in fvgs[-5:]:
        print(f"  {fvg.fvg_type.value}: {fvg.bottom:.2f} - {fvg.top:.2f} "
              f"(midpoint: {fvg.midpoint:.2f}, size: {fvg.size_percent:.2%})")
