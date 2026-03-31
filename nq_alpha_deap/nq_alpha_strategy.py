"""
NQ Alpha Strategy — EvolutionaryQuant
=======================================
Multi-factor NQ (Nasdaq 100) strategy combining:
  - Pivot Points (Daily R/S levels)
  - 15-min Opening Range Breakout (ORB)
  - Momentum (RSI, ROC, ADX)
  - FVG (Fair Value Gap) + Order Blocks
  - Gap Detection (overnight)
  - Kalman Filter (noise-free trend)
  - Hurst Exponent (regime detection)
  - VWAP (institutional bias)
  - ATR-based dynamic risk

All parameters are DEAP-optimizable via 15-gene genome.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────
# GENOME DEFINITION  — 15 parameters, each gene ∈ [0, 1]
# ─────────────────────────────────────────────────────────────────────
GENE_BOUNDS = [
    (0.20, 1.50),   # [0]  orb_breakout_mult   — ORB breakout threshold (× ATR)
    (0.10, 0.65),   # [1]  orb_rejection_pct    — % reversal back into ORB range
    (3,    20),     # [2]  momentum_roc_period  — Rate of Change period
    (5,    30),     # [3]  momentum_rsi_period  — RSI period
    (52,   75),     # [4]  momentum_rsi_thresh  — RSI overbought threshold
    (0.001, 0.10),  # [5]  kalman_r             — Kalman observation noise
    (0.0001,0.010), # [6]  kalman_q             — Kalman process noise
    (40,   200),    # [7]  hurst_window         — Hurst exponent window
    (0.30, 3.00),   # [8]  fvg_min_size         — FVG min size (× ATR)
    (4,    25),     # [9]  ob_lookback          — Order block lookback bars
    (0.05, 0.60),   # [10] pivot_touch_atr      — Pivot touch tolerance (× ATR)
    (0.05, 0.50),   # [11] gap_min_pct          — Min overnight gap (%)
    (1.00, 6.00),   # [12] tp_atr_mult          — Take profit (× ATR)
    (0.20, 2.00),   # [13] sl_atr_mult          — Stop loss (× ATR)
    (2,    8),      # [14] min_score            — Minimum signal score to enter
]

GENE_NAMES = [
    'orb_breakout_mult', 'orb_rejection_pct',
    'momentum_roc_period', 'momentum_rsi_period', 'momentum_rsi_thresh',
    'kalman_r', 'kalman_q', 'hurst_window',
    'fvg_min_size', 'ob_lookback',
    'pivot_touch_atr', 'gap_min_pct',
    'tp_atr_mult', 'sl_atr_mult', 'min_score',
]
GENOME_SIZE = len(GENE_BOUNDS)


@dataclass
class NQAlphaParams:
    """DEAP-optimizable parameters for the NQ Alpha strategy."""
    orb_breakout_mult:   float = 0.70   # ORB breakout threshold
    orb_rejection_pct:   float = 0.35   # ORB rejection threshold
    momentum_roc_period: int   = 10     # ROC period
    momentum_rsi_period: int   = 14     # RSI period
    momentum_rsi_thresh: float = 60.0   # RSI overbought
    kalman_r:            float = 0.01   # Kalman R (observation noise)
    kalman_q:            float = 0.001  # Kalman Q (process noise)
    hurst_window:        int   = 100    # Hurst window
    fvg_min_size:        float = 1.00   # FVG min size (× ATR)
    ob_lookback:         int   = 10     # OB lookback bars
    pivot_touch_atr:     float = 0.25   # Pivot touch tolerance
    gap_min_pct:         float = 0.15   # Min overnight gap (%)
    tp_atr_mult:         float = 2.50   # TP multiplier
    sl_atr_mult:         float = 1.00   # SL multiplier
    min_score:           int   = 4      # Minimum score to enter


def decode_genome(genome: list) -> NQAlphaParams:
    """Decode a DEAP genome (list of floats ∈ [0,1]) to NQAlphaParams."""
    p = NQAlphaParams()
    names = GENE_NAMES
    bounds = GENE_BOUNDS
    for i, (lo, hi) in enumerate(bounds):
        raw = float(genome[i]) if i < len(genome) else 0.5
        val = lo + abs(raw % 1.0) * (hi - lo)
        setattr(p, names[i], val)
    # Integer clamps
    p.momentum_roc_period = int(np.clip(round(p.momentum_roc_period), 3, 20))
    p.momentum_rsi_period = int(np.clip(round(p.momentum_rsi_period), 5, 30))
    p.hurst_window        = int(np.clip(round(p.hurst_window), 40, 200))
    p.ob_lookback         = int(np.clip(round(p.ob_lookback), 4, 25))
    p.min_score           = int(np.clip(round(p.min_score), 2, 8))
    return p


# ─────────────────────────────────────────────────────────────────────
# INDICATOR LIBRARY
# ─────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df['high'] - df['low']
    hc  = (df['high'] - df['close'].shift(1)).abs()
    lc  = (df['low']  - df['close'].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - 100 / (1 + rs)


def _roc(close: pd.Series, period: int) -> pd.Series:
    return (close / close.shift(period) - 1) * 100


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Simplified ADX (trend strength 0–100)."""
    tr   = _atr(df, 1).rename('tr')
    hh   = df['high'].diff();  ll = -df['low'].diff()
    dmp  = np.where((hh > ll) & (hh > 0), hh, 0.0)
    dmm  = np.where((ll > hh) & (ll > 0), ll, 0.0)
    smtr = pd.Series(tr).rolling(period).sum()
    smdp = pd.Series(dmp).rolling(period).sum()
    smdm = pd.Series(dmm).rolling(period).sum()
    dip  = 100 * smdp / (smtr + 1e-10)
    dim  = 100 * smdm / (smtr + 1e-10)
    dx   = 100 * (dip - dim).abs() / (dip + dim + 1e-10)
    return dx.rolling(period).mean()


def _vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    tp  = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume'].replace(0, 1)
    return (tp * vol).rolling(period).sum() / vol.rolling(period).sum()


def _kalman_filter(series: pd.Series, R: float, Q: float) -> pd.Series:
    """1D Kalman filter — returns smoothed (noise-reduced) price series."""
    n     = len(series)
    xhat  = np.zeros(n)    # state estimate
    P     = np.zeros(n)    # error covariance
    K     = np.zeros(n)    # Kalman gain
    vals  = series.values

    xhat[0] = vals[0]
    P[0]    = 1.0

    for t in range(1, n):
        # Predict
        xhat_p = xhat[t-1]
        P_p    = P[t-1] + Q
        # Update
        K[t]    = P_p / (P_p + R)
        xhat[t] = xhat_p + K[t] * (vals[t] - xhat_p)
        P[t]    = (1 - K[t]) * P_p

    return pd.Series(xhat, index=series.index)


def _hurst_exponent(ts: np.ndarray) -> float:
    """Estimate Hurst exponent via rescaled range (R/S) method.
    H > 0.55  → trending
    H < 0.45  → mean-reverting
    H ≈ 0.5   → random walk
    """
    n = len(ts)
    if n < 20:
        return 0.5
    lags  = range(2, min(n // 2, 50))
    tau   = []
    for lag in lags:
        sub = ts[:lag]
        if sub.std() < 1e-10:
            continue
        rs  = (sub - sub.mean()).cumsum()
        r   = rs.max() - rs.min()
        s   = sub.std()
        tau.append(r / (s + 1e-10))
    if len(tau) < 3:
        return 0.5
    try:
        coef = np.polyfit(np.log(list(lags[:len(tau)])), np.log(tau), 1)
        return float(np.clip(coef[0], 0.0, 1.0))
    except Exception:
        return 0.5


def _daily_pivots(df: pd.DataFrame) -> pd.DataFrame:
    """Compute classic daily pivots (PP, R1, R2, S1, S2) per bar.
    Uses PREVIOUS day's OHLC for current day's pivots.
    Works on any intraday timeframe.
    """
    date_col = df.index.date
    df2 = df.copy()
    df2['_date'] = date_col

    # Daily OHLC
    daily = df2.groupby('_date').agg(
        d_high=('high','max'), d_low=('low','min'), d_close=('close','last')
    ).reset_index()
    daily.columns = ['_date','d_high','d_low','d_close']
    daily['_next_date'] = daily['_date'].shift(-1)

    # Map to bars via bar date
    date_to_prev = {}
    for _, row in daily.iterrows():
        if pd.notna(row['_next_date']):
            date_to_prev[row['_next_date']] = (row['d_high'], row['d_low'], row['d_close'])

    pp  = np.full(len(df2), np.nan)
    r1  = np.full(len(df2), np.nan)
    r2  = np.full(len(df2), np.nan)
    s1  = np.full(len(df2), np.nan)
    s2  = np.full(len(df2), np.nan)

    for i, d in enumerate(date_col):
        if d in date_to_prev:
            h, l, c = date_to_prev[d]
            p       = (h + l + c) / 3
            pp[i]   = p
            r1[i]   = 2*p - l
            r2[i]   = p + (h - l)
            s1[i]   = 2*p - h
            s2[i]   = p - (h - l)

    df2['pivot']   = pp
    df2['r1']      = r1
    df2['r2']      = r2
    df2['s1']      = s1
    df2['s2']      = s2
    df2.drop(columns=['_date'], inplace=True)
    return df2


def _orb_levels(df: pd.DataFrame,
                session_start: Tuple[int,int] = (9, 30),
                orb_minutes: int = 15) -> pd.DataFrame:
    """Compute Opening Range (ORB) high/low per day.
    Returns df with 'orb_high' and 'orb_low' columns.
    Assumes index is DatetimeIndex (timezone-naive, already in US/Eastern or offset).
    """
    df2         = df.copy()
    df2['_date']= df2.index.date
    start_m     = session_start[0] * 60 + session_start[1]
    end_m       = start_m + orb_minutes

    orb_h = {}; orb_l = {}
    for d, grp in df2.groupby('_date'):
        mins = grp.index.hour * 60 + grp.index.minute
        mask = (mins >= start_m) & (mins < end_m)
        sub  = grp[mask]
        if len(sub) > 0:
            orb_h[d] = sub['high'].max()
            orb_l[d] = sub['low'].min()

    df2['orb_high'] = df2['_date'].map(orb_h)
    df2['orb_low']  = df2['_date'].map(orb_l)
    df2.drop(columns=['_date'], inplace=True)
    return df2


def _detect_fvg(df: pd.DataFrame, min_size_pts: float) -> pd.DataFrame:
    """Fair Value Gap (FVG) detection.
    Bullish FVG: candle[i-2].high < candle[i].low  (upside gap)
    Bearish FVG: candle[i-2].low  > candle[i].high (downside gap)
    """
    bull_fvg = np.zeros(len(df), dtype=bool)
    bear_fvg = np.zeros(len(df), dtype=bool)
    bull_fvg_top = np.full(len(df), np.nan)
    bear_fvg_bot = np.full(len(df), np.nan)

    high = df['high'].values
    low  = df['low'].values

    for i in range(2, len(df)):
        gap_up   = low[i]  - high[i-2]
        gap_down = low[i-2] - high[i]
        if gap_up  > min_size_pts:
            bull_fvg[i] = True
            bull_fvg_top[i] = low[i]
        if gap_down > min_size_pts:
            bear_fvg[i] = True
            bear_fvg_bot[i] = high[i]

    df2 = df.copy()
    df2['bull_fvg']     = bull_fvg
    df2['bear_fvg']     = bear_fvg
    df2['bull_fvg_top'] = bull_fvg_top
    df2['bear_fvg_bot'] = bear_fvg_bot
    return df2


def _detect_order_blocks(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Order Block detection:
    Bullish OB: last bearish candle before a strong bullish move
    Bearish OB: last bullish candle before a strong bearish move
    """
    df2    = df.copy()
    n      = len(df2)
    bull_ob = np.zeros(n, dtype=bool)
    bear_ob = np.zeros(n, dtype=bool)
    bull_ob_low  = np.full(n, np.nan)
    bear_ob_high = np.full(n, np.nan)

    close = df2['close'].values
    open_ = df2['open'].values
    high  = df2['high'].values
    low   = df2['low'].values

    for i in range(lookback + 1, n):
        # Potential bullish OB: look back for last bearish candle before price surged
        fwd_high = high[i]
        for j in range(i-1, max(i-lookback, 0), -1):
            if close[j] < open_[j]:  # bearish candle
                if fwd_high > max(high[j:i]):
                    bull_ob[j]    = True
                    bull_ob_low[j] = low[j]
                break
        # Potential bearish OB: look back for last bullish candle before price dropped
        fwd_low = low[i]
        for j in range(i-1, max(i-lookback, 0), -1):
            if close[j] > open_[j]:  # bullish candle
                if fwd_low < min(low[j:i]):
                    bear_ob[j]     = True
                    bear_ob_high[j]= high[j]
                break

    df2['bull_ob']       = bull_ob
    df2['bear_ob']       = bear_ob
    df2['bull_ob_low']   = bull_ob_low
    df2['bear_ob_high']  = bear_ob_high
    return df2


def _detect_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Detect overnight / session gaps."""
    df2 = df.copy()
    prev_close   = df2['close'].shift(1)
    gap_pct      = (df2['open'] - prev_close) / (prev_close + 1e-10) * 100
    df2['gap_pct']      = gap_pct
    df2['gap_up']       = gap_pct > 0
    df2['gap_down']     = gap_pct < 0
    return df2


# ─────────────────────────────────────────────────────────────────────
# MAIN FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────

def add_nq_alpha_features(df: pd.DataFrame, params: NQAlphaParams) -> pd.DataFrame:
    """Single call to add ALL strategy features to df.
    df must have columns: open, high, low, close, volume
    index: DatetimeIndex (timezone-naive)
    """
    df = df.copy()
    df.columns = df.columns.str.lower()

    # ─── Core indicators ──────────────────────────────────────────
    atr_period = 14
    df['atr']  = _atr(df, atr_period)
    df['vwap'] = _vwap(df, 20)
    df['rsi']  = _rsi(df['close'], params.momentum_rsi_period)
    df['roc']  = _roc(df['close'], params.momentum_roc_period)
    df['adx']  = _adx(df, 14)
    df['ema20']= df['close'].ewm(span=20, adjust=False).mean()
    df['ema50']= df['close'].ewm(span=50, adjust=False).mean()

    # ─── Kalman Filter ────────────────────────────────────────────
    df['kalman']       = _kalman_filter(df['close'], params.kalman_r, params.kalman_q)
    df['kalman_slope'] = df['kalman'].diff(3)

    # ─── Hurst Exponent (rolling, optimized) ──────────────────────
    # Calculate every 5 bars to keep runtime manageable, forward-fill between
    hw = params.hurst_window
    close_vals = df['close'].values
    n_bars     = len(df)
    hurst_vals = np.full(n_bars, np.nan)
    step       = max(1, min(10, hw // 10))   # adaptive step: 1/10 of window, min 1, max 10
    for i in range(hw, n_bars, step):
        hurst_vals[i] = _hurst_exponent(close_vals[i-hw:i])
    # Forward-fill NaN gaps
    hurst_series = pd.Series(hurst_vals, index=df.index)
    hurst_series = hurst_series.ffill().fillna(0.5)
    df['hurst']  = hurst_series

    # ─── Pivot Points ─────────────────────────────────────────────
    df = _daily_pivots(df)

    # ─── ORB (15-min) ─────────────────────────────────────────────
    df = _orb_levels(df, session_start=(9, 30), orb_minutes=15)

    # ─── FVG ───────────────────────────────────────────────────────
    fvg_pts = float(params.fvg_min_size) * df['atr'].fillna(5.0)
    # Use mean ATR for a fixed threshold per call
    mean_atr = float(df['atr'].median() or 5.0)
    df = _detect_fvg(df, min_size_pts=params.fvg_min_size * mean_atr)

    # ─── Order Blocks ─────────────────────────────────────────────
    df = _detect_order_blocks(df, params.ob_lookback)

    # ─── Gap ──────────────────────────────────────────────────────
    df = _detect_gap(df)

    # ─── Derived signals ──────────────────────────────────────────
    # ORB breakout / rejection signals
    orb_range = (df['orb_high'] - df['orb_low']).fillna(0)
    thresh    = params.orb_breakout_mult * df['atr'].fillna(5.0)

    df['orb_bull_break']  = (df['close'] > df['orb_high'] + thresh) & \
                             (df['orb_high'].notna()) & (orb_range > 0)
    df['orb_bear_break']  = (df['close'] < df['orb_low']  - thresh) & \
                             (df['orb_low'].notna()) & (orb_range > 0)

    # ORB rejection: price broke out then came back inside
    rej_thresh = params.orb_rejection_pct * orb_range
    df['orb_bull_reject'] = (df['high'] > df['orb_high']) & \
                             (df['close'] < df['orb_high'] - rej_thresh) & \
                             (df['orb_high'].notna())
    df['orb_bear_reject'] = (df['low']  < df['orb_low'])  & \
                             (df['close'] > df['orb_low']  + rej_thresh) & \
                             (df['orb_low'].notna())

    # Kalman trend
    df['kalman_up']   = df['kalman_slope'] > 0
    df['kalman_down'] = df['kalman_slope'] < 0

    # Hurst regime
    df['trending']     = df['hurst'] > 0.55
    df['mean_reverting']= df['hurst'] < 0.45

    # VWAP bias
    df['above_vwap'] = df['close'] > df['vwap']

    # Pivot proximity
    pt = params.pivot_touch_atr * df['atr'].fillna(5.0)
    df['near_pivot']= (
        ((df['close'] - df['pivot']).abs() < pt) |
        ((df['close'] - df['r1']).abs()    < pt) |
        ((df['close'] - df['s1']).abs()    < pt)
    )
    df['near_support']  = ((df['close'] - df['s1']).abs() < pt) | \
                           ((df['close'] - df['s2']).abs() < pt)
    df['near_resistance']= ((df['close'] - df['r1']).abs() < pt) | \
                            ((df['close'] - df['r2']).abs() < pt)

    # Momentum
    df['rsi_bull'] = df['rsi'] > params.momentum_rsi_thresh
    df['rsi_bear'] = df['rsi'] < (100 - params.momentum_rsi_thresh)
    df['roc_bull'] = df['roc'] > 0
    df['roc_bear'] = df['roc'] < 0
    df['adx_strong'] = df['adx'] > 25
    df['ema_bull']   = df['ema20'] > df['ema50']
    df['ema_bear']   = df['ema20'] < df['ema50']

    # Overnight gap signal
    df['gap_up_sig']  = df['gap_pct'] >  params.gap_min_pct
    df['gap_down_sig']= df['gap_pct'] < -params.gap_min_pct

    return df


# ─────────────────────────────────────────────────────────────────────
# SIGNAL SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────

def score_bar(row: pd.Series, params: NQAlphaParams) -> int:
    """
    Compute composite LONG (+) and SHORT (-) signal score for a single bar.
    Score range: -12 to +12  (integer)
    Entry triggered when abs(score) >= params.min_score

    Returns positive for LONG, negative for SHORT.
    """
    long_score  = 0
    short_score = 0

    # ── 1. ORB (weight 2) ──────────────────────────────────────────
    if bool(row.get('orb_bull_break', False)):   long_score  += 2
    if bool(row.get('orb_bear_break', False)):   short_score += 2
    if bool(row.get('orb_bear_reject', False)):  long_score  += 2  # failed breakdown → long
    if bool(row.get('orb_bull_reject', False)):  short_score += 2  # failed breakout  → short

    # ── 2. Pivot Points (weight 1) ─────────────────────────────────
    if bool(row.get('near_support', False)):     long_score  += 1
    if bool(row.get('near_resistance', False)):  short_score += 1

    # ── 3. Momentum (weight 2) ────────────────────────────────────
    if bool(row.get('rsi_bull', False)) and bool(row.get('roc_bull', False)):
        if bool(row.get('ema_bull', False)):     long_score  += 2
    if bool(row.get('rsi_bear', False)) and bool(row.get('roc_bear', False)):
        if bool(row.get('ema_bear', False)):     short_score += 2

    # ── 4. Kalman Filter (weight 1) ───────────────────────────────
    if bool(row.get('kalman_up', False)):        long_score  += 1
    if bool(row.get('kalman_down', False)):      short_score += 1

    # ── 5. Hurst Regime (weight 1 modifier) ───────────────────────
    if bool(row.get('trending', False)):
        # In trending regime, reward trend-following signals
        if long_score  > short_score: long_score  += 1
        if short_score > long_score:  short_score += 1
    if bool(row.get('mean_reverting', False)):
        # In mean-reverting regime, reward reversal signals (pivot/OB)
        if bool(row.get('near_support', False)):    long_score  += 1
        if bool(row.get('near_resistance', False)): short_score += 1

    # ── 6. VWAP Bias (weight 1) ───────────────────────────────────
    if bool(row.get('above_vwap', False)):       long_score  += 1
    else:                                         short_score += 1

    # ── 7. FVG (weight 2) ─────────────────────────────────────────
    if bool(row.get('bull_fvg', False)):         long_score  += 2
    if bool(row.get('bear_fvg', False)):         short_score += 2

    # ── 8. Order Blocks (weight 1) ────────────────────────────────
    if bool(row.get('bull_ob', False)):          long_score  += 1
    if bool(row.get('bear_ob', False)):          short_score += 1

    # ── 9. Gap (weight 1) ─────────────────────────────────────────
    if bool(row.get('gap_up_sig', False)):
        long_score  += 1   # gap up → continuation bias
    if bool(row.get('gap_down_sig', False)):
        short_score += 1   # gap down → continuation bias

    # Net score: positive = long, negative = short
    net = long_score - short_score
    return net


def session_filter(ts) -> bool:
    """True if timestamp is within US Regular Hours: 09:30–16:00 EST.
    Assumes timezone-naive index already in EST offset.
    """
    h = ts.hour; m = ts.minute
    t = h * 60 + m
    return 570 <= t <= 960   # 9:30=570, 16:00=960


# ─────────────────────────────────────────────────────────────────────
# DEFAULT PARAMS REFERENCE
# ─────────────────────────────────────────────────────────────────────
DEFAULT_PARAMS = NQAlphaParams()
