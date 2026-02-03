# NQ Quant Bot - Pivot Points
# Classic, Camarilla, Fibonacci pivots

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class PivotType(Enum):
    CLASSIC = "classic"
    CAMARILLA = "camarilla"
    FIBONACCI = "fibonacci"
    WOODIE = "woodie"


@dataclass
class PivotLevels:
    """Pivot point levels"""
    pivot: float
    r1: float  # Resistance 1
    r2: float
    r3: float
    r4: Optional[float]  # Camarilla only
    s1: float  # Support 1
    s2: float
    s3: float
    s4: Optional[float]  # Camarilla only
    pivot_type: PivotType
    
    def get_all_levels(self) -> Dict[str, float]:
        """Get all levels as dict"""
        levels = {
            'P': self.pivot,
            'R1': self.r1, 'R2': self.r2, 'R3': self.r3,
            'S1': self.s1, 'S2': self.s2, 'S3': self.s3
        }
        if self.r4 is not None:
            levels['R4'] = self.r4
        if self.s4 is not None:
            levels['S4'] = self.s4
        return levels
    
    def get_nearest_support(self, price: float) -> float:
        """Get nearest support level below price"""
        supports = [self.s1, self.s2, self.s3]
        if self.s4:
            supports.append(self.s4)
        supports = [s for s in supports if s < price]
        return max(supports) if supports else self.s3
    
    def get_nearest_resistance(self, price: float) -> float:
        """Get nearest resistance level above price"""
        resistances = [self.r1, self.r2, self.r3]
        if self.r4:
            resistances.append(self.r4)
        resistances = [r for r in resistances if r > price]
        return min(resistances) if resistances else self.r3


def calculate_classic_pivots(high: float, low: float, close: float) -> PivotLevels:
    """
    Classic Pivot Points
    
    P = (H + L + C) / 3
    R1 = 2P - L
    R2 = P + (H - L)
    R3 = H + 2(P - L)
    S1 = 2P - H
    S2 = P - (H - L)
    S3 = L - 2(H - P)
    """
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    r1 = 2 * pivot - low
    r2 = pivot + range_hl
    r3 = high + 2 * (pivot - low)
    
    s1 = 2 * pivot - high
    s2 = pivot - range_hl
    s3 = low - 2 * (high - pivot)
    
    return PivotLevels(
        pivot=pivot,
        r1=r1, r2=r2, r3=r3, r4=None,
        s1=s1, s2=s2, s3=s3, s4=None,
        pivot_type=PivotType.CLASSIC
    )


def calculate_camarilla_pivots(high: float, low: float, close: float) -> PivotLevels:
    """
    Camarilla Pivot Points - Tighter levels, good for intraday
    
    R4 = C + (H - L) * 1.5/2
    R3 = C + (H - L) * 1.25/4
    R2 = C + (H - L) * 1.166/6
    R1 = C + (H - L) * 1.083/12
    S1 = C - (H - L) * 1.083/12
    S2 = C - (H - L) * 1.166/6
    S3 = C - (H - L) * 1.25/4
    S4 = C - (H - L) * 1.5/2
    """
    range_hl = high - low
    
    r4 = close + range_hl * 1.5 / 2
    r3 = close + range_hl * 1.25 / 4
    r2 = close + range_hl * 1.166 / 6
    r1 = close + range_hl * 1.083 / 12
    
    s1 = close - range_hl * 1.083 / 12
    s2 = close - range_hl * 1.166 / 6
    s3 = close - range_hl * 1.25 / 4
    s4 = close - range_hl * 1.5 / 2
    
    pivot = (high + low + close) / 3
    
    return PivotLevels(
        pivot=pivot,
        r1=r1, r2=r2, r3=r3, r4=r4,
        s1=s1, s2=s2, s3=s3, s4=s4,
        pivot_type=PivotType.CAMARILLA
    )


def calculate_fibonacci_pivots(high: float, low: float, close: float) -> PivotLevels:
    """
    Fibonacci Pivot Points
    
    Uses Fibonacci ratios: 0.382, 0.618, 1.0
    """
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    r1 = pivot + range_hl * 0.382
    r2 = pivot + range_hl * 0.618
    r3 = pivot + range_hl * 1.0
    
    s1 = pivot - range_hl * 0.382
    s2 = pivot - range_hl * 0.618
    s3 = pivot - range_hl * 1.0
    
    return PivotLevels(
        pivot=pivot,
        r1=r1, r2=r2, r3=r3, r4=None,
        s1=s1, s2=s2, s3=s3, s4=None,
        pivot_type=PivotType.FIBONACCI
    )


def calculate_pivots(
    high: float, 
    low: float, 
    close: float,
    pivot_type: PivotType = PivotType.CLASSIC
) -> PivotLevels:
    """Calculate pivot points based on type"""
    if pivot_type == PivotType.CLASSIC:
        return calculate_classic_pivots(high, low, close)
    elif pivot_type == PivotType.CAMARILLA:
        return calculate_camarilla_pivots(high, low, close)
    elif pivot_type == PivotType.FIBONACCI:
        return calculate_fibonacci_pivots(high, low, close)
    else:
        return calculate_classic_pivots(high, low, close)


def get_daily_pivots(df: pd.DataFrame, pivot_type: PivotType = PivotType.CLASSIC) -> Dict[str, PivotLevels]:
    """
    Calculate daily pivot points from intraday data
    
    Returns dict with date as key and PivotLevels as value
    """
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Get previous day's OHLC
    daily = df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    pivots = {}
    for date, row in daily.iterrows():
        pivots[date.strftime('%Y-%m-%d')] = calculate_pivots(
            row['high'], row['low'], row['close'], pivot_type
        )
    
    return pivots


def get_pivot_zone(price: float, pivots: PivotLevels) -> str:
    """
    Determine which pivot zone the price is in
    
    Returns: 'above_r3', 'r2_r3', 'r1_r2', 'pivot_r1', 
             's1_pivot', 's2_s1', 's3_s2', 'below_s3'
    """
    if price >= pivots.r3:
        return 'above_r3'
    elif price >= pivots.r2:
        return 'r2_r3'
    elif price >= pivots.r1:
        return 'r1_r2'
    elif price >= pivots.pivot:
        return 'pivot_r1'
    elif price >= pivots.s1:
        return 's1_pivot'
    elif price >= pivots.s2:
        return 's2_s1'
    elif price >= pivots.s3:
        return 's3_s2'
    else:
        return 'below_s3'


def check_pivot_bounce(
    current_price: float,
    previous_price: float,
    pivots: PivotLevels,
    tolerance: float = 0.002  # 0.2% tolerance
) -> Optional[str]:
    """
    Check if price bounced off a pivot level
    
    Returns: 'bounce_r1', 'bounce_s1', etc. or None
    """
    all_levels = pivots.get_all_levels()
    
    for name, level in all_levels.items():
        level_range = level * tolerance
        
        # Check for bounce (touched and moved away)
        if previous_price <= level + level_range and previous_price >= level - level_range:
            if current_price > previous_price:
                return f'bounce_{name.lower()}_bullish'
            elif current_price < previous_price:
                return f'bounce_{name.lower()}_bearish'
    
    return None


# === TEST ===
if __name__ == "__main__":
    # Test with sample data
    high = 21550.0
    low = 21380.0
    close = 21480.0
    
    print("Testing Pivot Point calculations...")
    print(f"\nInput: H={high}, L={low}, C={close}")
    
    # Classic
    classic = calculate_classic_pivots(high, low, close)
    print(f"\nClassic Pivots:")
    print(f"  Pivot: {classic.pivot:.2f}")
    print(f"  R1: {classic.r1:.2f}, R2: {classic.r2:.2f}, R3: {classic.r3:.2f}")
    print(f"  S1: {classic.s1:.2f}, S2: {classic.s2:.2f}, S3: {classic.s3:.2f}")
    
    # Camarilla
    camarilla = calculate_camarilla_pivots(high, low, close)
    print(f"\nCamarilla Pivots:")
    print(f"  Pivot: {camarilla.pivot:.2f}")
    print(f"  R1: {camarilla.r1:.2f}, R2: {camarilla.r2:.2f}, R3: {camarilla.r3:.2f}, R4: {camarilla.r4:.2f}")
    print(f"  S1: {camarilla.s1:.2f}, S2: {camarilla.s2:.2f}, S3: {camarilla.s3:.2f}, S4: {camarilla.s4:.2f}")
    
    # Zone test
    test_price = 21500.0
    zone = get_pivot_zone(test_price, classic)
    print(f"\nPrice {test_price:.2f} is in zone: {zone}")
