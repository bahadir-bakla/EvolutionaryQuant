# NQ Quant Bot - VWAP (Volume Weighted Average Price)
# Institutional level indicator

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class VWAPBias(Enum):
    BULLISH = "BULLISH"      # Price above VWAP
    BEARISH = "BEARISH"      # Price below VWAP
    NEUTRAL = "NEUTRAL"      # At VWAP


@dataclass
class VWAPResult:
    """VWAP calculation result"""
    vwap: float
    upper_band_1: float  # +1 std
    lower_band_1: float  # -1 std
    upper_band_2: float  # +2 std
    lower_band_2: float  # -2 std
    bias: VWAPBias
    distance_percent: float  # Distance from VWAP as %
    zone: str  # 'extreme_oversold', 'oversold', 'neutral', 'overbought', 'extreme_overbought'


def calculate_vwap(
    df: pd.DataFrame,
    reset_daily: bool = True
) -> pd.DataFrame:
    """
    Calculate VWAP with standard deviation bands
    
    VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
    
    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume' columns
        reset_daily: Reset VWAP calculation each day
        
    Returns:
        DataFrame with vwap, upper_band_1, lower_band_1, upper_band_2, lower_band_2
    """
    df = df.copy()
    
    # Typical Price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_volume'] = df['typical_price'] * df['volume']
    
    if reset_daily and isinstance(df.index, pd.DatetimeIndex):
        # Group by date for daily reset
        df['date'] = df.index.date
        groups = df.groupby('date')
        
        vwap_list = []
        std_list = []
        
        for _, group in groups:
            cum_tp_vol = group['tp_volume'].cumsum()
            cum_vol = group['volume'].cumsum()
            vwap = cum_tp_vol / (cum_vol + 1e-10)
            
            # Rolling standard deviation of price from VWAP
            squared_diff = (group['typical_price'] - vwap) ** 2
            variance = squared_diff.expanding().mean()
            std = np.sqrt(variance)
            
            vwap_list.extend(vwap.values)
            std_list.extend(std.values)
        
        df['vwap'] = vwap_list
        df['vwap_std'] = std_list
    else:
        # No daily reset - cumulative
        df['cum_tp_volume'] = df['tp_volume'].cumsum()
        df['cum_volume'] = df['volume'].cumsum()
        df['vwap'] = df['cum_tp_volume'] / (df['cum_volume'] + 1e-10)
        
        squared_diff = (df['typical_price'] - df['vwap']) ** 2
        df['vwap_std'] = np.sqrt(squared_diff.expanding().mean())
    
    # Standard deviation bands
    df['vwap_upper_1'] = df['vwap'] + df['vwap_std']
    df['vwap_lower_1'] = df['vwap'] - df['vwap_std']
    df['vwap_upper_2'] = df['vwap'] + (df['vwap_std'] * 2)
    df['vwap_lower_2'] = df['vwap'] - (df['vwap_std'] * 2)
    
    # Cleanup only temporary columns (not other indicators)
    temp_cols = ['typical_price', 'tp_volume', 'date', 'cum_tp_volume', 'cum_volume']
    for col in temp_cols:
        if col in df.columns:
            df = df.drop(columns=[col], errors='ignore')
    
    return df


def get_vwap_signal(
    price: float,
    vwap: float,
    upper_1: float,
    lower_1: float,
    upper_2: float,
    lower_2: float,
    threshold_percent: float = 0.1  # 0.1% threshold for neutral
) -> VWAPResult:
    """
    Get VWAP signal for current price
    
    Args:
        price: Current close price
        vwap: Current VWAP value
        upper_1, lower_1: ±1 std bands
        upper_2, lower_2: ±2 std bands
        threshold_percent: % threshold for neutral zone
        
    Returns:
        VWAPResult with bias and zone
    """
    distance = price - vwap
    distance_percent = (distance / vwap) * 100 if vwap > 0 else 0
    
    # Determine bias
    if abs(distance_percent) < threshold_percent:
        bias = VWAPBias.NEUTRAL
    elif price > vwap:
        bias = VWAPBias.BULLISH
    else:
        bias = VWAPBias.BEARISH
    
    # Determine zone
    if price >= upper_2:
        zone = 'extreme_overbought'
    elif price >= upper_1:
        zone = 'overbought'
    elif price <= lower_2:
        zone = 'extreme_oversold'
    elif price <= lower_1:
        zone = 'oversold'
    else:
        zone = 'neutral'
    
    return VWAPResult(
        vwap=vwap,
        upper_band_1=upper_1,
        lower_band_1=lower_1,
        upper_band_2=upper_2,
        lower_band_2=lower_2,
        bias=bias,
        distance_percent=distance_percent,
        zone=zone
    )


def check_vwap_cross(
    current_price: float,
    previous_price: float,
    current_vwap: float,
    previous_vwap: float
) -> Tuple[bool, str]:
    """
    Check for VWAP crossover
    
    Returns:
        (is_cross, direction) - direction is 'bullish_cross' or 'bearish_cross' or None
    """
    # Bullish cross: price crosses above VWAP
    if previous_price <= previous_vwap and current_price > current_vwap:
        return True, 'bullish_cross'
    
    # Bearish cross: price crosses below VWAP
    if previous_price >= previous_vwap and current_price < current_vwap:
        return True, 'bearish_cross'
    
    return False, None


# === TEST ===
if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing VWAP calculation...")
    ticker = yf.Ticker("NQ=F")
    df = ticker.history(period="1mo", interval="1h")
    df.columns = df.columns.str.lower()
    
    # Calculate VWAP
    df = calculate_vwap(df, reset_daily=True)
    
    # Get last signal
    last = df.iloc[-1]
    result = get_vwap_signal(
        last['close'], 
        last['vwap'],
        last['vwap_upper_1'],
        last['vwap_lower_1'],
        last['vwap_upper_2'],
        last['vwap_lower_2']
    )
    
    print(f"Price: {last['close']:.2f}")
    print(f"VWAP: {result.vwap:.2f}")
    print(f"Bias: {result.bias.value}")
    print(f"Zone: {result.zone}")
    print(f"Distance: {result.distance_percent:.2f}%")
