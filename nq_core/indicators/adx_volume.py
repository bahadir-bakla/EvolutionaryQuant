# NQ Quant Bot - ADX (Average Directional Index) & Volume Profile
# Trend strength and volume-based levels

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


class TrendStrength(Enum):
    VERY_STRONG = "VERY_STRONG"  # ADX > 50
    STRONG = "STRONG"            # ADX > 40
    MODERATE = "MODERATE"        # ADX > 25
    WEAK = "WEAK"                # ADX > 20
    NO_TREND = "NO_TREND"        # ADX <= 20


@dataclass
class ADXResult:
    """ADX indicator result"""
    adx: float
    plus_di: float   # +DI
    minus_di: float  # -DI
    trend_strength: TrendStrength
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    crossover: Optional[str]  # '+di_cross_up', '+di_cross_down', None


@dataclass
class VolumeProfileResult:
    """Volume Profile levels"""
    poc: float           # Point of Control - highest volume price
    vah: float           # Value Area High
    val: float           # Value Area Low
    hvn: list            # High Volume Nodes (list of prices)
    lvn: list            # Low Volume Nodes (list of prices)


def calculate_adx(
    df: pd.DataFrame,
    period: int = 14
) -> pd.DataFrame:
    """
    Calculate ADX (Average Directional Index)
    
    ADX measures trend strength (0-100):
    - 0-20: No trend / very weak
    - 20-25: Weak trend
    - 25-40: Moderate trend
    - 40-50: Strong trend
    - 50+: Very strong trend
    
    +DI > -DI = Bullish
    -DI > +DI = Bearish
    """
    df = df.copy()
    
    # Calculate +DM and -DM
    df['high_diff'] = df['high'].diff()
    df['low_diff'] = df['low'].diff().abs() * -1  # Make negative
    
    df['plus_dm'] = np.where(
        (df['high_diff'] > df['low_diff'].abs()) & (df['high_diff'] > 0),
        df['high_diff'],
        0
    )
    
    df['minus_dm'] = np.where(
        (df['low_diff'].abs() > df['high_diff']) & (df['low_diff'] < 0),
        df['low_diff'].abs(),
        0
    )
    
    # Calculate True Range
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    
    # Smooth with Wilder's smoothing (same as EMA with alpha = 1/period)
    alpha = 1 / period
    
    df['atr'] = df['tr'].ewm(alpha=alpha, adjust=False).mean()
    df['plus_dm_smooth'] = df['plus_dm'].ewm(alpha=alpha, adjust=False).mean()
    df['minus_dm_smooth'] = df['minus_dm'].ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate +DI and -DI
    df['plus_di'] = 100 * (df['plus_dm_smooth'] / (df['atr'] + 1e-10))
    df['minus_di'] = 100 * (df['minus_dm_smooth'] / (df['atr'] + 1e-10))
    
    # Calculate DX
    df['dx'] = 100 * (
        (df['plus_di'] - df['minus_di']).abs() / 
        (df['plus_di'] + df['minus_di'] + 1e-10)
    )
    
    # Calculate ADX (smoothed DX)
    df['adx'] = df['dx'].ewm(alpha=alpha, adjust=False).mean()
    
    # Cleanup
    cols_to_keep = ['adx', 'plus_di', 'minus_di']
    
    return df


def get_adx_signal(
    adx: float,
    plus_di: float,
    minus_di: float,
    prev_plus_di: float = None,
    prev_minus_di: float = None
) -> ADXResult:
    """Get ADX signal interpretation"""
    
    # Trend strength
    if adx > 50:
        strength = TrendStrength.VERY_STRONG
    elif adx > 40:
        strength = TrendStrength.STRONG
    elif adx > 25:
        strength = TrendStrength.MODERATE
    elif adx > 20:
        strength = TrendStrength.WEAK
    else:
        strength = TrendStrength.NO_TREND
    
    # Trend direction
    if plus_di > minus_di:
        direction = 'bullish'
    elif minus_di > plus_di:
        direction = 'bearish'
    else:
        direction = 'neutral'
    
    # Crossover detection
    crossover = None
    if prev_plus_di is not None and prev_minus_di is not None:
        if prev_plus_di <= prev_minus_di and plus_di > minus_di:
            crossover = '+di_cross_up'  # Bullish crossover
        elif prev_plus_di >= prev_minus_di and plus_di < minus_di:
            crossover = '+di_cross_down'  # Bearish crossover
    
    return ADXResult(
        adx=adx,
        plus_di=plus_di,
        minus_di=minus_di,
        trend_strength=strength,
        trend_direction=direction,
        crossover=crossover
    )


def calculate_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
    value_area_percent: float = 0.70
) -> VolumeProfileResult:
    """
    Calculate Volume Profile
    
    POC (Point of Control): Price level with highest volume
    VAH (Value Area High): Upper bound of 70% volume concentration
    VAL (Value Area Low): Lower bound of 70% volume concentration
    HVN: High Volume Nodes
    LVN: Low Volume Nodes
    """
    # Create price bins
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min
    bin_size = price_range / num_bins
    
    # Initialize volume at each price level
    volume_profile = np.zeros(num_bins)
    
    # Distribute volume across price levels
    for _, row in df.iterrows():
        bar_low = row['low']
        bar_high = row['high']
        bar_volume = row['volume']
        
        # Find which bins this bar covers
        low_bin = int((bar_low - price_min) / bin_size)
        high_bin = int((bar_high - price_min) / bin_size)
        
        low_bin = max(0, min(low_bin, num_bins - 1))
        high_bin = max(0, min(high_bin, num_bins - 1))
        
        # Distribute volume evenly across covered bins
        num_covered_bins = high_bin - low_bin + 1
        volume_per_bin = bar_volume / num_covered_bins
        
        for b in range(low_bin, high_bin + 1):
            volume_profile[b] += volume_per_bin
    
    # Find POC (highest volume bin)
    poc_bin = np.argmax(volume_profile)
    poc = price_min + (poc_bin + 0.5) * bin_size
    
    # Calculate Value Area (70% of volume)
    total_volume = volume_profile.sum()
    target_volume = total_volume * value_area_percent
    
    # Expand from POC until we capture target volume
    val_bin = poc_bin
    vah_bin = poc_bin
    current_volume = volume_profile[poc_bin]
    
    while current_volume < target_volume:
        # Expand to side with higher volume
        lower_vol = volume_profile[val_bin - 1] if val_bin > 0 else 0
        upper_vol = volume_profile[vah_bin + 1] if vah_bin < num_bins - 1 else 0
        
        if lower_vol >= upper_vol and val_bin > 0:
            val_bin -= 1
            current_volume += lower_vol
        elif vah_bin < num_bins - 1:
            vah_bin += 1
            current_volume += upper_vol
        else:
            break
    
    val = price_min + val_bin * bin_size
    vah = price_min + (vah_bin + 1) * bin_size
    
    # Find HVN and LVN
    mean_volume = volume_profile.mean()
    std_volume = volume_profile.std()
    
    hvn = []
    lvn = []
    
    for i, vol in enumerate(volume_profile):
        price = price_min + (i + 0.5) * bin_size
        if vol > mean_volume + std_volume:
            hvn.append(price)
        elif vol < mean_volume - std_volume:
            lvn.append(price)
    
    return VolumeProfileResult(
        poc=poc,
        vah=vah,
        val=val,
        hvn=hvn[:5],  # Top 5
        lvn=lvn[:5]
    )


# === TEST ===
if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing ADX and Volume Profile...")
    ticker = yf.Ticker("NQ=F")
    df = ticker.history(period="3mo", interval="1d")
    df.columns = df.columns.str.lower()
    
    # Calculate ADX
    df = calculate_adx(df)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    adx_result = get_adx_signal(
        last['adx'], 
        last['plus_di'], 
        last['minus_di'],
        prev['plus_di'],
        prev['minus_di']
    )
    
    print(f"\nADX Analysis:")
    print(f"  ADX: {adx_result.adx:.2f}")
    print(f"  +DI: {adx_result.plus_di:.2f}")
    print(f"  -DI: {adx_result.minus_di:.2f}")
    print(f"  Strength: {adx_result.trend_strength.value}")
    print(f"  Direction: {adx_result.trend_direction}")
    if adx_result.crossover:
        print(f"  Crossover: {adx_result.crossover}")
    
    # Calculate Volume Profile
    vp = calculate_volume_profile(df)
    
    print(f"\nVolume Profile:")
    print(f"  POC: {vp.poc:.2f}")
    print(f"  VAH: {vp.vah:.2f}")
    print(f"  VAL: {vp.val:.2f}")
    print(f"  HVN count: {len(vp.hvn)}")
    print(f"  LVN count: {len(vp.lvn)}")
