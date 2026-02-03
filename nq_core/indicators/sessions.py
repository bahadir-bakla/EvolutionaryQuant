# NQ Quant Bot - Session Analysis
# London, New York, Asian session detection

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from enum import Enum
from datetime import time, datetime


class TradingSession(Enum):
    ASIAN = "ASIAN"         # Tokyo: 00:00 - 09:00 UTC
    LONDON = "LONDON"       # London: 08:00 - 17:00 UTC
    NEW_YORK = "NEW_YORK"   # New York: 13:00 - 22:00 UTC
    OVERLAP = "OVERLAP"     # London-NY overlap: 13:00 - 17:00 UTC
    OFF_HOURS = "OFF_HOURS" # Low volatility


@dataclass
class SessionInfo:
    """Current session information"""
    current_session: TradingSession
    session_high: float
    session_low: float
    session_open: float
    is_high_volatility: bool
    time_in_session: float  # 0-1, progress through session
    previous_session_high: Optional[float]
    previous_session_low: Optional[float]


# Session times in UTC
SESSION_TIMES = {
    TradingSession.ASIAN: (time(0, 0), time(9, 0)),
    TradingSession.LONDON: (time(8, 0), time(17, 0)),
    TradingSession.NEW_YORK: (time(13, 0), time(22, 0)),
    TradingSession.OVERLAP: (time(13, 0), time(17, 0)),
}

# NQ futures trades 23 hours (closed 17:00-18:00 ET)
# High volatility times
HIGH_VOLATILITY_TIMES = [
    (time(8, 30), time(9, 30)),   # US Pre-market / London active
    (time(13, 30), time(14, 30)), # NY Open
    (time(14, 0), time(15, 0)),   # Economic releases
]


def get_current_session(timestamp: pd.Timestamp) -> TradingSession:
    """
    Determine current trading session from timestamp
    
    Note: Timestamp should be in UTC
    """
    current_time = timestamp.time()
    
    # Check overlap first (most important)
    overlap_start, overlap_end = SESSION_TIMES[TradingSession.OVERLAP]
    if overlap_start <= current_time <= overlap_end:
        return TradingSession.OVERLAP
    
    # Check New York
    ny_start, ny_end = SESSION_TIMES[TradingSession.NEW_YORK]
    if ny_start <= current_time <= ny_end:
        return TradingSession.NEW_YORK
    
    # Check London
    london_start, london_end = SESSION_TIMES[TradingSession.LONDON]
    if london_start <= current_time <= london_end:
        return TradingSession.LONDON
    
    # Check Asian
    asian_start, asian_end = SESSION_TIMES[TradingSession.ASIAN]
    if asian_start <= current_time <= asian_end:
        return TradingSession.ASIAN
    
    return TradingSession.OFF_HOURS


def is_high_volatility_time(timestamp: pd.Timestamp) -> bool:
    """Check if current time is typically high volatility"""
    current_time = timestamp.time()
    
    for start, end in HIGH_VOLATILITY_TIMES:
        if start <= current_time <= end:
            return True
    
    # Also check if it's overlap (always high vol)
    session = get_current_session(timestamp)
    if session == TradingSession.OVERLAP:
        return True
    
    return False


def calculate_session_levels(
    df: pd.DataFrame,
    session: TradingSession = TradingSession.NEW_YORK
) -> pd.DataFrame:
    """
    Calculate session high/low/open for each bar
    
    Adds columns:
    - session_high, session_low, session_open
    - prev_session_high, prev_session_low
    """
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Initialize columns
    df['session'] = df.index.map(get_current_session).map(lambda x: x.value)
    df['session_high'] = np.nan
    df['session_low'] = np.nan
    df['session_open'] = np.nan
    df['prev_session_high'] = np.nan
    df['prev_session_low'] = np.nan
    
    # Group by date and session
    current_session = None
    session_start_idx = 0
    session_high = 0
    session_low = float('inf')
    session_open = 0
    prev_high = None
    prev_low = None
    
    for i in range(len(df)):
        row_session = get_current_session(df.index[i])
        
        # Session change detected
        if row_session != current_session:
            # Save previous session levels
            if current_session is not None:
                prev_high = session_high
                prev_low = session_low
            
            # Start new session
            current_session = row_session
            session_start_idx = i
            session_high = df.iloc[i]['high']
            session_low = df.iloc[i]['low']
            session_open = df.iloc[i]['open']
        else:
            # Update session levels
            session_high = max(session_high, df.iloc[i]['high'])
            session_low = min(session_low, df.iloc[i]['low'])
        
        # Set values
        df.iloc[i, df.columns.get_loc('session_high')] = session_high
        df.iloc[i, df.columns.get_loc('session_low')] = session_low
        df.iloc[i, df.columns.get_loc('session_open')] = session_open
        
        if prev_high is not None:
            df.iloc[i, df.columns.get_loc('prev_session_high')] = prev_high
            df.iloc[i, df.columns.get_loc('prev_session_low')] = prev_low
    
    return df


def get_session_bias(
    current_price: float,
    session_open: float,
    session_high: float,
    session_low: float
) -> str:
    """
    Determine session bias based on price position
    
    Returns: 'bullish', 'bearish', 'neutral'
    """
    session_range = session_high - session_low
    if session_range == 0:
        return 'neutral'
    
    position = (current_price - session_low) / session_range
    
    if current_price > session_open and position > 0.6:
        return 'bullish'
    elif current_price < session_open and position < 0.4:
        return 'bearish'
    else:
        return 'neutral'


def check_session_breakout(
    current_price: float,
    prev_session_high: float,
    prev_session_low: float,
    threshold_percent: float = 0.002
) -> Optional[str]:
    """
    Check for previous session level breakout
    
    Returns: 'breakout_high', 'breakout_low', or None
    """
    if prev_session_high is None or prev_session_low is None:
        return None
    
    threshold = prev_session_high * threshold_percent
    
    if current_price > prev_session_high + threshold:
        return 'breakout_high'
    elif current_price < prev_session_low - threshold:
        return 'breakout_low'
    
    return None


def get_session_info(
    df: pd.DataFrame,
    current_idx: int
) -> SessionInfo:
    """Get complete session information for current bar"""
    row = df.iloc[current_idx]
    timestamp = df.index[current_idx]
    
    session = get_current_session(timestamp)
    
    return SessionInfo(
        current_session=session,
        session_high=row.get('session_high', 0),
        session_low=row.get('session_low', 0),
        session_open=row.get('session_open', 0),
        is_high_volatility=is_high_volatility_time(timestamp),
        time_in_session=0.5,  # Placeholder
        previous_session_high=row.get('prev_session_high', None),
        previous_session_low=row.get('prev_session_low', None)
    )


# === TEST ===
if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing Session Analysis...")
    ticker = yf.Ticker("NQ=F")
    df = ticker.history(period="5d", interval="1h")
    df.columns = df.columns.str.lower()
    
    # Add session levels
    df = calculate_session_levels(df)
    
    # Get last bar info
    last_idx = len(df) - 1
    session = get_current_session(df.index[last_idx])
    is_hvol = is_high_volatility_time(df.index[last_idx])
    
    print(f"\nCurrent Time (UTC): {df.index[last_idx]}")
    print(f"Session: {session.value}")
    print(f"High Volatility: {is_hvol}")
    print(f"Session High: {df.iloc[last_idx]['session_high']:.2f}")
    print(f"Session Low: {df.iloc[last_idx]['session_low']:.2f}")
    
    # Session distribution
    print(f"\nSession Distribution:")
    print(df['session'].value_counts())
