# NQ Quant Bot - Multi-Timeframe Analysis
# Weekly + Daily + 4H confluence for higher accuracy

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from enum import Enum
import yfinance as yf


class MTFBias(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"   # All TFs bullish
    BULLISH = "BULLISH"                  # Majority bullish
    NEUTRAL = "NEUTRAL"                  # Mixed
    BEARISH = "BEARISH"                  # Majority bearish
    STRONG_BEARISH = "STRONG_BEARISH"   # All TFs bearish


@dataclass
class TimeframeBias:
    """Bias for a single timeframe"""
    timeframe: str
    trend: str          # 'up', 'down', 'sideways'
    strength: float     # 0-1
    ema_position: str   # 'above', 'below', 'at'
    momentum: str       # 'bullish', 'bearish', 'neutral'
    key_level: Optional[float]  # Nearest S/R level


@dataclass
class MTFAnalysis:
    """Multi-Timeframe Analysis Result"""
    weekly_bias: TimeframeBias
    daily_bias: TimeframeBias
    h4_bias: TimeframeBias
    overall_bias: MTFBias
    alignment_score: float  # 0-1, how aligned are the timeframes
    recommended_direction: str  # 'LONG', 'SHORT', 'WAIT'
    confidence: float


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate EMA"""
    return series.ewm(span=period, adjust=False).mean()


def analyze_single_timeframe(df: pd.DataFrame, timeframe: str) -> TimeframeBias:
    """
    Analyze a single timeframe
    
    Uses:
    - EMA 20/50/200 for trend
    - RSI for momentum
    - Recent highs/lows for S/R
    """
    if len(df) < 50:
        return TimeframeBias(
            timeframe=timeframe,
            trend='sideways',
            strength=0.0,
            ema_position='at',
            momentum='neutral',
            key_level=None
        )
    
    close = df['close']
    high = df['high']
    low = df['low']
    
    # EMAs
    ema_20 = calculate_ema(close, 20)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, min(200, len(df) - 1)) if len(df) > 200 else calculate_ema(close, 50)
    
    current_price = close.iloc[-1]
    current_ema_20 = ema_20.iloc[-1]
    current_ema_50 = ema_50.iloc[-1]
    current_ema_200 = ema_200.iloc[-1]
    
    # Trend determination
    bullish_count = 0
    if current_price > current_ema_20: bullish_count += 1
    if current_price > current_ema_50: bullish_count += 1
    if current_price > current_ema_200: bullish_count += 1
    if current_ema_20 > current_ema_50: bullish_count += 1
    if current_ema_50 > current_ema_200: bullish_count += 1
    
    if bullish_count >= 4:
        trend = 'up'
        strength = bullish_count / 5
    elif bullish_count <= 1:
        trend = 'down'
        strength = (5 - bullish_count) / 5
    else:
        trend = 'sideways'
        strength = 0.5
    
    # EMA position
    if current_price > current_ema_20 * 1.002:
        ema_position = 'above'
    elif current_price < current_ema_20 * 0.998:
        ema_position = 'below'
    else:
        ema_position = 'at'
    
    # RSI for momentum
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    if current_rsi > 60:
        momentum = 'bullish'
    elif current_rsi < 40:
        momentum = 'bearish'
    else:
        momentum = 'neutral'
    
    # Key level (recent swing high/low)
    recent_high = high.iloc[-20:].max()
    recent_low = low.iloc[-20:].min()
    
    if abs(current_price - recent_high) < abs(current_price - recent_low):
        key_level = recent_high
    else:
        key_level = recent_low
    
    return TimeframeBias(
        timeframe=timeframe,
        trend=trend,
        strength=strength,
        ema_position=ema_position,
        momentum=momentum,
        key_level=key_level
    )


def fetch_multi_timeframe_data(symbol: str = "NQ=F") -> Dict[str, pd.DataFrame]:
    """
    Fetch data for multiple timeframes
    
    Returns dict with 'weekly', 'daily', '4h' keys
    """
    ticker = yf.Ticker(symbol)
    
    data = {}
    
    # Weekly (2 years)
    weekly = ticker.history(period="2y", interval="1wk")
    weekly.columns = weekly.columns.str.lower()
    data['weekly'] = weekly
    
    # Daily (1 year)
    daily = ticker.history(period="1y", interval="1d")
    daily.columns = daily.columns.str.lower()
    data['daily'] = daily
    
    # 4H (60 days max for free data)
    h4 = ticker.history(period="60d", interval="1h")
    h4.columns = h4.columns.str.lower()
    # Resample to 4H
    h4_resampled = h4.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    data['4h'] = h4_resampled
    
    return data


def analyze_multi_timeframe(data: Dict[str, pd.DataFrame]) -> MTFAnalysis:
    """
    Perform multi-timeframe analysis
    
    Higher TF = Higher weight in decision
    """
    # Analyze each timeframe
    weekly_bias = analyze_single_timeframe(data['weekly'], 'weekly')
    daily_bias = analyze_single_timeframe(data['daily'], 'daily')
    h4_bias = analyze_single_timeframe(data['4h'], '4h')
    
    # Calculate alignment
    trends = [weekly_bias.trend, daily_bias.trend, h4_bias.trend]
    
    bullish_count = trends.count('up')
    bearish_count = trends.count('down')
    
    # Weighted scoring (weekly = 3x, daily = 2x, 4h = 1x)
    weighted_score = 0
    weights = {'weekly': 3, 'daily': 2, '4h': 1}
    
    for tf, bias in [('weekly', weekly_bias), ('daily', daily_bias), ('4h', h4_bias)]:
        if bias.trend == 'up':
            weighted_score += weights[tf]
        elif bias.trend == 'down':
            weighted_score -= weights[tf]
    
    max_score = sum(weights.values())  # 6
    
    # Overall bias
    if bullish_count == 3:
        overall_bias = MTFBias.STRONG_BULLISH
        alignment_score = 1.0
    elif bullish_count >= 2 and weighted_score > 2:
        overall_bias = MTFBias.BULLISH
        alignment_score = 0.7
    elif bearish_count == 3:
        overall_bias = MTFBias.STRONG_BEARISH
        alignment_score = 1.0
    elif bearish_count >= 2 and weighted_score < -2:
        overall_bias = MTFBias.BEARISH
        alignment_score = 0.7
    else:
        overall_bias = MTFBias.NEUTRAL
        alignment_score = 0.3
    
    # Recommended direction
    if overall_bias in [MTFBias.STRONG_BULLISH, MTFBias.BULLISH]:
        recommended = 'LONG'
        confidence = (weighted_score + max_score) / (2 * max_score)
    elif overall_bias in [MTFBias.STRONG_BEARISH, MTFBias.BEARISH]:
        recommended = 'SHORT'
        confidence = (-weighted_score + max_score) / (2 * max_score)
    else:
        recommended = 'WAIT'
        confidence = 0.3
    
    return MTFAnalysis(
        weekly_bias=weekly_bias,
        daily_bias=daily_bias,
        h4_bias=h4_bias,
        overall_bias=overall_bias,
        alignment_score=alignment_score,
        recommended_direction=recommended,
        confidence=confidence
    )


def get_mtf_filter(mtf_analysis: MTFAnalysis, signal_direction: str) -> Tuple[bool, float]:
    """
    Check if signal aligns with higher timeframe bias
    
    Returns:
        (should_take_trade, confidence_multiplier)
    """
    if signal_direction == 'NEUTRAL':
        return False, 0.0
    
    # Strong alignment
    if mtf_analysis.overall_bias == MTFBias.STRONG_BULLISH and signal_direction == 'LONG':
        return True, 1.5  # Boost confidence
    elif mtf_analysis.overall_bias == MTFBias.STRONG_BEARISH and signal_direction == 'SHORT':
        return True, 1.5
    
    # Normal alignment
    elif mtf_analysis.overall_bias == MTFBias.BULLISH and signal_direction == 'LONG':
        return True, 1.2
    elif mtf_analysis.overall_bias == MTFBias.BEARISH and signal_direction == 'SHORT':
        return True, 1.2
    
    # Counter-trend (risky)
    elif mtf_analysis.overall_bias == MTFBias.STRONG_BULLISH and signal_direction == 'SHORT':
        return False, 0.0  # Don't short in strong uptrend
    elif mtf_analysis.overall_bias == MTFBias.STRONG_BEARISH and signal_direction == 'LONG':
        return False, 0.0  # Don't long in strong downtrend
    
    # Neutral or weak counter
    else:
        return True, 0.8  # Reduced confidence


# === TEST ===
if __name__ == "__main__":
    print("Testing Multi-Timeframe Analysis...")
    
    # Fetch data
    print("\nFetching NQ=F data for multiple timeframes...")
    data = fetch_multi_timeframe_data("NQ=F")
    
    print(f"  Weekly: {len(data['weekly'])} bars")
    print(f"  Daily: {len(data['daily'])} bars")
    print(f"  4H: {len(data['4h'])} bars")
    
    # Analyze
    analysis = analyze_multi_timeframe(data)
    
    print(f"\n{'='*60}")
    print("MULTI-TIMEFRAME ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nWeekly:")
    print(f"  Trend: {analysis.weekly_bias.trend} (strength: {analysis.weekly_bias.strength:.2f})")
    print(f"  EMA Position: {analysis.weekly_bias.ema_position}")
    print(f"  Momentum: {analysis.weekly_bias.momentum}")
    
    print(f"\nDaily:")
    print(f"  Trend: {analysis.daily_bias.trend} (strength: {analysis.daily_bias.strength:.2f})")
    print(f"  EMA Position: {analysis.daily_bias.ema_position}")
    print(f"  Momentum: {analysis.daily_bias.momentum}")
    
    print(f"\n4H:")
    print(f"  Trend: {analysis.h4_bias.trend} (strength: {analysis.h4_bias.strength:.2f})")
    print(f"  EMA Position: {analysis.h4_bias.ema_position}")
    print(f"  Momentum: {analysis.h4_bias.momentum}")
    
    print(f"\n{'='*60}")
    print(f"OVERALL BIAS: {analysis.overall_bias.value}")
    print(f"Alignment Score: {analysis.alignment_score:.2f}")
    print(f"Recommended: {analysis.recommended_direction}")
    print(f"Confidence: {analysis.confidence:.1%}")
    print(f"{'='*60}")
