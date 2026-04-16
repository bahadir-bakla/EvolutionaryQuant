"""
ML Engine — Feature Engineering
================================
Multi-timeframe feature set for Gold M1 scalping.
Builds ~70 features per bar from M1 raw data resampled to 5m and 15m.

Features:
  - Price-action features (candle body/wick ratios, patterns)
  - Momentum (ROC, RSI, CCI)
  - Volatility (ATR, Bollinger, Keltner)
  - Volume (relative volume, OBV delta)
  - Structure (swing highs/lows, fair value gaps, order blocks)
  - Multi-timeframe (5m + 15m versions of core features)
  - Session (time-of-day sine/cosine encoding)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    tp = (high + low + close) / 3.0
    ma = tp.rolling(period).mean()
    md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _obv_delta(close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).cumsum()
    return obv.diff(period)


def _swing_highs(high: pd.Series, lookback: int = 5) -> pd.Series:
    """1 where bar is a swing high (highest in ±lookback bars), else 0."""
    local_max = high.rolling(2 * lookback + 1, center=True).max()
    return (high == local_max).astype(float)


def _swing_lows(low: pd.Series, lookback: int = 5) -> pd.Series:
    local_min = low.rolling(2 * lookback + 1, center=True).min()
    return (low == local_min).astype(float)


def _fvg_bull(high: pd.Series, low: pd.Series) -> pd.Series:
    """Bullish FVG: current bar low > high 2 bars ago."""
    gap = low - high.shift(2)
    return gap.clip(lower=0)   # positive gap size, 0 if no FVG


def _fvg_bear(high: pd.Series, low: pd.Series) -> pd.Series:
    """Bearish FVG: current bar high < low 2 bars ago."""
    gap = low.shift(2) - high
    return gap.clip(lower=0)


def _order_block_bull(open_: pd.Series, close: pd.Series, low: pd.Series, lookback: int = 5) -> pd.Series:
    """Simplified OB: last bearish bar before strong bull impulse."""
    body = (close - open_).abs()
    body_ma = body.rolling(lookback).mean()
    impulse = close > close.shift(1) + body_ma  # strong bull bar
    # OB = low of bar before impulse
    ob_low = low.shift(1)
    return ob_low.where(impulse, 0.0).fillna(0.0)


# -----------------------------------------------------------------------------
# SINGLE-TIMEFRAME FEATURES
# -----------------------------------------------------------------------------

def _build_tf_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Build features for one timeframe.
    df must have columns: open, high, low, close, volume
    Returns DataFrame with prefixed column names.
    """
    o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']

    feat: dict[str, pd.Series] = {}

    # -- Candle structure ------------------------------------------------
    body      = (c - o).abs()
    full_range = (h - l).replace(0, np.nan)
    upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l

    feat['body_ratio']        = body / full_range
    feat['upper_wick_ratio']  = upper_wick / full_range
    feat['lower_wick_ratio']  = lower_wick / full_range
    feat['bull_bar']          = (c > o).astype(float)
    feat['doji']              = (feat['body_ratio'] < 0.1).astype(float)

    # Engulfing pattern
    prev_body  = body.shift(1)
    prev_bull  = (df['close'].shift(1) > df['open'].shift(1))
    bull_engulf = (~prev_bull) & (c > o) & (body > prev_body)
    bear_engulf = (prev_bull) & (c < o) & (body > prev_body)
    feat['bull_engulf'] = bull_engulf.astype(float)
    feat['bear_engulf'] = bear_engulf.astype(float)

    # -- Momentum --------------------------------------------------------
    atr14 = _atr(h, l, c, 14)
    feat['atr14']    = atr14
    feat['rsi14']    = _rsi(c, 14)
    feat['rsi7']     = _rsi(c, 7)
    feat['cci20']    = _cci(h, l, c, 20)
    feat['roc5']     = c.pct_change(5) * 100.0
    feat['roc14']    = c.pct_change(14) * 100.0

    # Normalised price position in ATR units
    feat['close_vs_sma20']  = (c - c.rolling(20).mean()) / atr14.replace(0, np.nan)
    feat['close_vs_sma50']  = (c - c.rolling(50).mean()) / atr14.replace(0, np.nan)
    feat['close_vs_ema9']   = (c - c.ewm(span=9).mean())  / atr14.replace(0, np.nan)

    # -- Volatility ------------------------------------------------------
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    feat['bb_pct']   = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std.replace(0, np.nan))
    feat['bb_width'] = (4 * bb_std) / bb_mid.replace(0, np.nan)
    feat['atr_norm'] = atr14 / c.replace(0, np.nan)  # as % of price

    # -- Volume ----------------------------------------------------------
    # HistData M1 has zero volume — guard with neutral fallback (1.0)
    vol_ma = v.rolling(20).mean()
    feat['vol_ratio']   = (v / vol_ma.replace(0, np.nan)).fillna(1.0)
    feat['obv_delta14'] = _obv_delta(c, v, 14).fillna(0.0)

    # -- Structure -------------------------------------------------------
    # Note: swing_highs/lows use center=True -> NaN at tail — drop them
    feat['fvg_bull_pts']  = _fvg_bull(h, l)
    feat['fvg_bear_pts']  = _fvg_bear(h, l)
    feat['ob_bull_low']   = _order_block_bull(o, c, l, 5)
    # Recent swing distance (no center window -> no NaN)
    feat['dist_from_20h'] = (h.rolling(20).max() - c)  # pts above recent high
    feat['dist_from_20l'] = (c - l.rolling(20).min())  # pts above recent low

    # Highest high / lowest low in last N bars (relative to ATR)
    feat['hh20']  = (h.rolling(20).max() - c) / atr14.replace(0, np.nan)
    feat['ll20']  = (c - l.rolling(20).min()) / atr14.replace(0, np.nan)

    result = pd.DataFrame(feat, index=df.index)
    result.columns = [f"{prefix}_{col}" for col in result.columns]
    return result


# -----------------------------------------------------------------------------
# ICT PRICE ACTION FEATURES (M1 level)
# -----------------------------------------------------------------------------

def _ict_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ICT-inspired price action features computed on M1 bars.
    - Liquidity sweep: wick past N-bar high/low then close back inside
    - Market Structure Break (MSB/BOS): close above prior swing high / below prior swing low
    - Equal highs/lows: price within 0.1 pts of recent high/low (liquidity pool)
    - Premium/Discount: position within recent N-bar range (0=bottom, 1=top)
    - Rejection wick: large wick relative to body (pin bar)
    """
    o, h, l, c = df['open'], df['high'], df['low'], df['close']
    feat: dict[str, pd.Series] = {}

    # Rolling N-bar high/low
    for n in [20, 50]:
        roll_h = h.rolling(n).max().shift(1)   # prior N bars high (not incl. current)
        roll_l = l.rolling(n).min().shift(1)

        # Liquidity sweep bull: wick below roll_l but close above roll_l
        liq_bull = (l < roll_l) & (c > roll_l)
        feat[f'liq_sweep_bull_{n}'] = liq_bull.astype(float)

        # Liquidity sweep bear: wick above roll_h but close below roll_h
        liq_bear = (h > roll_h) & (c < roll_h)
        feat[f'liq_sweep_bear_{n}'] = liq_bear.astype(float)

        # MSB bull: close above prior N-bar high
        feat[f'msb_bull_{n}'] = (c > roll_h).astype(float)
        # MSB bear: close below prior N-bar low
        feat[f'msb_bear_{n}'] = (c < roll_l).astype(float)

        # Premium/Discount (0=discount zone, 1=premium zone)
        rng = (roll_h - roll_l).replace(0, np.nan)
        feat[f'prem_disc_{n}'] = (c - roll_l) / rng

    # Rejection wick ratio (pin bar detector)
    body      = (c - o).abs()
    full_rng  = (h - l).replace(0, np.nan)
    upper_w   = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_w   = pd.concat([o, c], axis=1).min(axis=1) - l
    feat['bull_pin'] = ((lower_w / full_rng) > 0.6) & (c > o)
    feat['bear_pin'] = ((upper_w / full_rng) > 0.6) & (c < o)
    feat['bull_pin'] = feat['bull_pin'].astype(float)
    feat['bear_pin'] = feat['bear_pin'].astype(float)

    # Inside bar (consolidation)
    feat['inside_bar'] = ((h < h.shift(1)) & (l > l.shift(1))).astype(float)

    # Breakout bar: range > 2x rolling avg range
    avg_rng = full_rng.rolling(20).mean()
    feat['breakout_bar'] = (full_rng > 2.0 * avg_rng).astype(float)

    # Equal highs/lows (within 0.2 pts — liquidity pool)
    feat['eq_highs'] = ((h - h.shift(1)).abs() < 0.2).astype(float)
    feat['eq_lows']  = ((l - l.shift(1)).abs() < 0.2).astype(float)

    return pd.DataFrame(feat, index=df.index)


# -----------------------------------------------------------------------------
# HIGHER TIMEFRAME BIAS (H1)
# -----------------------------------------------------------------------------

def _h1_features(m1: pd.DataFrame) -> pd.DataFrame:
    """H1 trend bias aligned back to M1."""
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    h1 = m1.resample('1h').agg(agg).dropna(subset=['open'])

    o, h, l, c = h1['open'], h1['high'], h1['low'], h1['close']
    feat: dict[str, pd.Series] = {}

    # Trend
    ema9  = c.ewm(span=9).mean()
    ema21 = c.ewm(span=21).mean()
    feat['H1_bull_trend']  = (ema9 > ema21).astype(float)
    feat['H1_close_ema9']  = (c - ema9) / c.replace(0, np.nan)
    feat['H1_close_ema21'] = (c - ema21) / c.replace(0, np.nan)
    feat['H1_rsi14']       = _rsi(c, 14)

    atr = _atr(h, l, c, 14)
    feat['H1_atr']         = atr
    feat['H1_atr_norm']    = atr / c.replace(0, np.nan)

    # Candle bias
    feat['H1_bull_bar']    = (c > o).astype(float)
    body = (c - o).abs()
    rng  = (h - l).replace(0, np.nan)
    feat['H1_body_ratio']  = body / rng

    df_h1 = pd.DataFrame(feat, index=h1.index)
    return df_h1.reindex(m1.index, method='ffill')


# -----------------------------------------------------------------------------
# SESSION TIME FEATURES
# -----------------------------------------------------------------------------

def _session_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Encode time-of-day and day-of-week as cyclical features + session flags."""
    # Minutes since midnight (UTC)
    minutes = index.hour * 60 + index.minute

    feat: dict[str, np.ndarray] = {}
    feat['time_sin'] = np.sin(2 * np.pi * minutes / 1440.0)
    feat['time_cos'] = np.cos(2 * np.pi * minutes / 1440.0)

    # Day of week cyclical
    dow = index.dayofweek  # 0=Mon .. 4=Fri
    feat['dow_sin'] = np.sin(2 * np.pi * dow / 5.0)
    feat['dow_cos'] = np.cos(2 * np.pi * dow / 5.0)

    # Session flags (UTC)
    feat['sess_london'] = ((index.hour >= 7) & (index.hour < 11)).astype(float)
    feat['sess_ny']     = ((index.hour >= 13) & (index.hour < 17)).astype(float)
    feat['sess_overlap']= ((index.hour >= 13) & (index.hour < 16)).astype(float)
    feat['sess_asian']  = ((index.hour >= 0)  & (index.hour < 6)).astype(float)

    return pd.DataFrame(feat, index=index)


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

def build_features(m1: pd.DataFrame) -> pd.DataFrame:
    """
    Build full feature matrix from M1 OHLCV data.

    Parameters
    ----------
    m1 : pd.DataFrame
        M1 bars with columns [open, high, low, close, volume]

    Returns
    -------
    pd.DataFrame
        Feature matrix indexed by M1 timestamps.
        Contains ~70 features prefixed by timeframe: m1_, m5_, m15_
        Plus session_ features.
        NaN rows at start due to rolling windows are preserved — caller should dropna().
    """
    m1 = m1.copy()
    m1.columns = m1.columns.str.lower()
    if 'volume' not in m1.columns:
        m1['volume'] = 1.0

    # -- Resample to 5m and 15m -----------------------------------------
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    m5  = m1.resample('5min').agg(agg).dropna(subset=['open'])
    m15 = m1.resample('15min').agg(agg).dropna(subset=['open'])

    # -- Build per-timeframe features ------------------------------------
    f_m1  = _build_tf_features(m1,  'M1')
    f_m5  = _build_tf_features(m5,  'M5')
    f_m15 = _build_tf_features(m15, 'M15')

    # Align M5 and M15 features back to M1 index via forward-fill
    f_m5_aligned  = f_m5.reindex(m1.index, method='ffill')
    f_m15_aligned = f_m15.reindex(m1.index, method='ffill')

    # -- ICT price action features (M1) ---------------------------------
    f_ict = _ict_features(m1)

    # -- H1 trend bias ---------------------------------------------------
    f_h1 = _h1_features(m1)

    # -- Session features ------------------------------------------------
    f_sess = _session_features(m1.index)

    # -- Combine ---------------------------------------------------------
    combined = pd.concat([f_m1, f_m5_aligned, f_m15_aligned, f_ict, f_h1, f_sess], axis=1)
    return combined
