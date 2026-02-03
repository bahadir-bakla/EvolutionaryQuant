# NQ Quant Bot - Market Structure
# Break of Structure (BOS), Change of Character (CHoCH)

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class StructureBias(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class StructureEvent(Enum):
    BOS_BULLISH = "BOS_BULLISH"      # Break of Structure - continuation bullish
    BOS_BEARISH = "BOS_BEARISH"      # Break of Structure - continuation bearish
    CHOCH_BULLISH = "CHOCH_BULLISH"  # Change of Character - reversal to bullish
    CHOCH_BEARISH = "CHOCH_BEARISH"  # Change of Character - reversal to bearish
    NONE = "NONE"


@dataclass
class SwingPoint:
    """Swing high or low point"""
    index: int
    price: float
    is_high: bool  # True = swing high, False = swing low
    timestamp: Optional[pd.Timestamp] = None


@dataclass
class MarketStructure:
    """Market structure state"""
    current_bias: StructureBias
    last_event: StructureEvent
    last_swing_high: Optional[SwingPoint]
    last_swing_low: Optional[SwingPoint]
    higher_highs: int  # Count of consecutive HH
    higher_lows: int   # Count of consecutive HL
    lower_highs: int   # Count of consecutive LH
    lower_lows: int    # Count of consecutive LL
    strength: float    # 0-1 structure strength


def find_swing_points(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3
) -> List[SwingPoint]:
    """
    Find swing highs and lows using pivot detection
    
    A swing high is when 'left_bars' bars on left are lower AND 'right_bars' bars on right are lower
    A swing low is when 'left_bars' bars on left are higher AND 'right_bars' bars on right are higher
    """
    highs = df['high'].values
    lows = df['low'].values
    swing_points = []
    
    for i in range(left_bars, len(df) - right_bars):
        # Check swing high
        is_swing_high = True
        for j in range(1, left_bars + 1):
            if highs[i - j] >= highs[i]:
                is_swing_high = False
                break
        if is_swing_high:
            for j in range(1, right_bars + 1):
                if highs[i + j] >= highs[i]:
                    is_swing_high = False
                    break
        
        if is_swing_high:
            swing_points.append(SwingPoint(
                index=i,
                price=highs[i],
                is_high=True,
                timestamp=df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None
            ))
        
        # Check swing low
        is_swing_low = True
        for j in range(1, left_bars + 1):
            if lows[i - j] <= lows[i]:
                is_swing_low = False
                break
        if is_swing_low:
            for j in range(1, right_bars + 1):
                if lows[i + j] <= lows[i]:
                    is_swing_low = False
                    break
        
        if is_swing_low:
            swing_points.append(SwingPoint(
                index=i,
                price=lows[i],
                is_high=False,
                timestamp=df.index[i] if isinstance(df.index, pd.DatetimeIndex) else None
            ))
    
    # Sort by index
    swing_points.sort(key=lambda x: x.index)
    return swing_points


def analyze_structure(
    swing_points: List[SwingPoint],
    current_price: float,
    lookback: int = 10
) -> MarketStructure:
    """
    Analyze market structure from swing points
    
    Bullish structure: Higher Highs + Higher Lows
    Bearish structure: Lower Highs + Lower Lows
    """
    if len(swing_points) < 4:
        return MarketStructure(
            current_bias=StructureBias.NEUTRAL,
            last_event=StructureEvent.NONE,
            last_swing_high=None,
            last_swing_low=None,
            higher_highs=0, higher_lows=0,
            lower_highs=0, lower_lows=0,
            strength=0.0
        )
    
    # Get recent swings
    recent_swings = swing_points[-lookback:] if len(swing_points) > lookback else swing_points
    
    # Separate highs and lows
    swing_highs = [s for s in recent_swings if s.is_high]
    swing_lows = [s for s in recent_swings if not s.is_high]
    
    # Count structure patterns
    hh_count = 0  # Higher highs
    hl_count = 0  # Higher lows
    lh_count = 0  # Lower highs
    ll_count = 0  # Lower lows
    
    # Analyze highs
    for i in range(1, len(swing_highs)):
        if swing_highs[i].price > swing_highs[i-1].price:
            hh_count += 1
        else:
            lh_count += 1
    
    # Analyze lows
    for i in range(1, len(swing_lows)):
        if swing_lows[i].price > swing_lows[i-1].price:
            hl_count += 1
        else:
            ll_count += 1
    
    # Determine bias
    bullish_score = hh_count + hl_count
    bearish_score = lh_count + ll_count
    
    if bullish_score > bearish_score:
        bias = StructureBias.BULLISH
        strength = bullish_score / (bullish_score + bearish_score + 1)
    elif bearish_score > bullish_score:
        bias = StructureBias.BEARISH
        strength = bearish_score / (bullish_score + bearish_score + 1)
    else:
        bias = StructureBias.NEUTRAL
        strength = 0.5
    
    # Find last event (BOS or CHoCH)
    last_event = StructureEvent.NONE
    
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_high = swing_highs[-1]
        prev_high = swing_highs[-2]
        last_low = swing_lows[-1]
        prev_low = swing_lows[-2]
        
        # Check for BOS (continuation)
        if last_high.price > prev_high.price and bias == StructureBias.BULLISH:
            last_event = StructureEvent.BOS_BULLISH
        elif last_low.price < prev_low.price and bias == StructureBias.BEARISH:
            last_event = StructureEvent.BOS_BEARISH
        
        # Check for CHoCH (reversal)
        # Bullish CHoCH: Was making LL, now making HH
        if prev_low.price < swing_lows[-3].price if len(swing_lows) >= 3 else True:
            if last_high.price > prev_high.price:
                last_event = StructureEvent.CHOCH_BULLISH
        
        # Bearish CHoCH: Was making HH, now making LL  
        if len(swing_highs) >= 3 and prev_high.price > swing_highs[-3].price:
            if last_low.price < prev_low.price:
                last_event = StructureEvent.CHOCH_BEARISH
    
    return MarketStructure(
        current_bias=bias,
        last_event=last_event,
        last_swing_high=swing_highs[-1] if swing_highs else None,
        last_swing_low=swing_lows[-1] if swing_lows else None,
        higher_highs=hh_count,
        higher_lows=hl_count,
        lower_highs=lh_count,
        lower_lows=ll_count,
        strength=strength
    )


def detect_bos_choch(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3
) -> pd.DataFrame:
    """
    Add BOS/CHoCH detection to DataFrame
    
    Returns DataFrame with added columns:
    - swing_high, swing_low: swing point prices
    - structure_bias: current market structure bias
    - structure_event: BOS or CHoCH events
    """
    df = df.copy()
    
    swing_points = find_swing_points(df, left_bars, right_bars)
    
    # Initialize columns
    df['swing_high'] = np.nan
    df['swing_low'] = np.nan
    df['structure_bias'] = 'NEUTRAL'
    df['structure_event'] = 'NONE'
    
    # Mark swing points
    for sp in swing_points:
        if sp.is_high:
            df.iloc[sp.index, df.columns.get_loc('swing_high')] = sp.price
        else:
            df.iloc[sp.index, df.columns.get_loc('swing_low')] = sp.price
    
    # Analyze structure at each point
    for i in range(len(df)):
        relevant_swings = [sp for sp in swing_points if sp.index <= i]
        if len(relevant_swings) >= 4:
            structure = analyze_structure(relevant_swings, df.iloc[i]['close'])
            df.iloc[i, df.columns.get_loc('structure_bias')] = structure.current_bias.value
            df.iloc[i, df.columns.get_loc('structure_event')] = structure.last_event.value
    
    return df


# === TEST ===
if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing Market Structure detection...")
    ticker = yf.Ticker("NQ=F")
    df = ticker.history(period="3mo", interval="1d")
    df.columns = df.columns.str.lower()
    
    # Find swing points
    swings = find_swing_points(df, left_bars=3, right_bars=3)
    print(f"Found {len(swings)} swing points")
    
    # Analyze structure
    structure = analyze_structure(swings, df.iloc[-1]['close'])
    
    print(f"\nMarket Structure:")
    print(f"  Bias: {structure.current_bias.value}")
    print(f"  Last Event: {structure.last_event.value}")
    print(f"  Higher Highs: {structure.higher_highs}")
    print(f"  Higher Lows: {structure.higher_lows}")
    print(f"  Lower Highs: {structure.lower_highs}")
    print(f"  Lower Lows: {structure.lower_lows}")
    print(f"  Strength: {structure.strength:.2f}")
    
    if structure.last_swing_high:
        print(f"  Last Swing High: {structure.last_swing_high.price:.2f}")
    if structure.last_swing_low:
        print(f"  Last Swing Low: {structure.last_swing_low.price:.2f}")
