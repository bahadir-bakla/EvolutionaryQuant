"""
NQ Alpha: The Spectral Hunter — EvolutionaryQuant
=================================================
A surgical Nasdaq (NQ) scalper based on Institutional Liquidity Concepts (ICT/SMC).
Focuses on Sweeps of PDH/PDL/Session Levels followed by Market Structure Shifts (MSS).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple

GENE_BOUNDS = [
    (0.30, 2.00),   # [0] fvg_min_size        — FVG min size (× ATR)
    (10,   40),     # [1] ob_lookback         — Order block lookback
    (0.01, 0.25),   # [2] sweep_sensitivity   — Depth of liquidity sweep (× ATR)
    (1.00, 2.00),   # [3] displacement_mult   — Strength of MSS move (× Prev Body)
    (1.50, 8.00),   # [4] tp_atr_mult         — Take profit (× ATR)
    (0.50, 4.00),   # [5] sl_atr_mult         — Stop loss (× ATR)
    (0.10, 0.80),   # [6] meta_bias_threshold — Spectral Bias Filter
    (5,    30),     # [7] rsi_period          — Extra filter for Exhaustion
    (2,    20),     # [8] momentum_period     — ROC Lookback
]

GENE_NAMES = [
    'fvg_min_size', 'ob_lookback', 'sweep_sensitivity', 'displacement_mult',
    'tp_atr_mult', 'sl_atr_mult', 'meta_bias_threshold', 'rsi_period', 'momentum_period'
]
GENOME_SIZE = len(GENE_BOUNDS)


@dataclass
class NQAlphaParams:
    """DEAP-optimizable parameters for 'The Spectral Hunter'."""
    fvg_min_size:        float = 1.00
    ob_lookback:         int   = 15
    sweep_sensitivity:   float = 0.02
    displacement_mult:   float = 1.10
    tp_atr_mult:         float = 3.50
    sl_atr_mult:         float = 2.00
    min_score:           int   = 3
    meta_bias_threshold: float = 0.10
    rsi_period:          int   = 14
    momentum_period:     int   = 10


def decode_genome(genome: list) -> NQAlphaParams:
    p = NQAlphaParams()
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        raw = float(genome[i]) if i < len(genome) else 0.5
        val = lo + abs(raw % 1.0) * (hi - lo)
        setattr(p, GENE_NAMES[i], val)
    p.ob_lookback = int(np.clip(round(p.ob_lookback), 10, 40))
    p.rsi_period  = int(np.clip(round(p.rsi_period), 5, 30))
    return p


# ─────────────────────────────────────────────────────────────────────
# LIQUIDITY ENGINE
# ─────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h_l = df['high'] - df['low']
    h_c = (df['high'] - df['close'].shift(1)).abs()
    l_c = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def _rsi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=n-1, adjust=False).mean()
    ema_down = down.ewm(com=n-1, adjust=False).mean()
    rs = ema_up / ema_down.replace(0, 0.001)
    return 100 - (100 / (1 + rs))

def _liquidity_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates PDH/PDL and Session Levels."""
    df = df.copy()
    if df.empty: return df
    
    df['_date'] = df.index.date
    
    # 1. PDH / PDL (Previous Day High/Low)
    daily = df.groupby('_date').agg({'high':'max', 'low':'min'})
    daily_sh = daily.shift(1).to_dict('index')
    
    df['pdh'] = df['_date'].apply(lambda d: daily_sh[d]['high'] if d in daily_sh else np.nan)
    df['pdl'] = df['_date'].apply(lambda d: daily_sh[d]['low'] if d in daily_sh else np.nan)

    # 2. London/Session High/Low (Simplified: Pre-market high/low)
    # 02:00 -> 09:30 NY Time
    pm_h = {}; pm_l = {}
    for d, grp in df.groupby('_date'):
        mask = (grp.index.hour < 9) | ((grp.index.hour == 9) & (grp.index.minute < 30))
        sub = grp[mask]
        if not sub.empty:
            pm_h[d] = sub['high'].max()
            pm_l[d] = sub['low'].min()
            
    df['session_h'] = df['_date'].map(pm_h)
    df['session_l'] = df['_date'].map(pm_l)
    
    # 3. Standard Daily Pivot Points
    # Using grouped OHLC again to ensure P/R1/S1 are calculated correctly
    daily_ohlc = df.groupby('_date').agg({'high':'max', 'low':'min', 'close':'last'})
    daily_sh = daily_ohlc.shift(1).to_dict('index')
    
    def get_pivot(d, key):
        stats = daily_sh.get(d, None)
        if not stats: return np.nan
        p = (stats['high'] + stats['low'] + stats['close']) / 3
        if key == 'p': return p
        if key == 'r1': return (2 * p) - stats['low']
        if key == 's1': return (2 * p) - stats['high']
        return np.nan

    df['pivot_p']  = df['_date'].apply(lambda d: get_pivot(d, 'p'))
    df['pivot_r1'] = df['_date'].apply(lambda d: get_pivot(d, 'r1'))
    df['pivot_s1'] = df['_date'].apply(lambda d: get_pivot(d, 's1'))

    df.drop(columns=['_date'], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# SIGNAL LOGIC
# ─────────────────────────────────────────────────────────────────────

def add_nq_alpha_features(df: pd.DataFrame, params: NQAlphaParams) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    df['atr'] = _atr(df)
    df['rsi'] = _rsi(df, params.rsi_period)
    df['mom'] = df['close'].diff(params.momentum_period) # Momentum: Price ROC
    df = _liquidity_levels(df)
    
    # ATR value for thresholds
    atrv = df['atr'].fillna(df['atr'].median())
    sens = params.sweep_sensitivity * atrv
    
    # Sweep & Pivot Break Logic
    # Price breaks R1 or S1 with momentum confirmation
    df['pivot_break_up'] = (df['close'] > df['pivot_r1']) & (df['close'].shift(1) <= df['pivot_r1'])
    df['pivot_break_dn'] = (df['close'] < df['pivot_s1']) & (df['close'].shift(1) >= df['pivot_s1'])

    # Liquidity Sweep Tracking (PDH/PDL/Session)
    sw_h = (df['high'] > df['pdh']) | (df['high'] > df['session_h']) | (df['high'] > df['pivot_r1'])
    sw_l = (df['low']  < df['pdl']) | (df['low']  < df['session_l']) | (df['low']  < df['pivot_s1'])

    # Sweep confirmed if price closes back inside the level
    df['sweep_h'] = (sw_h.rolling(3).max() > 0) & (df['close'] < df['high'].rolling(3).max() - sens)
    df['sweep_l'] = (sw_l.rolling(3).max() > 0) & (df['close'] > df['low'].rolling(3).min() + sens)
    
    # Market Structure Shift (MSS)
    # Look back at swing highs/lows (no center=True to avoid future leak)
    df['swing_h'] = df['high'].shift(1).rolling(5).max()
    df['swing_l'] = df['low'].shift(1).rolling(5).min()
    
    # Displacement logic
    body_size = (df['close'] - df['open']).abs()
    prev_body = body_size.shift(1).replace(0, 1e-6)
    df['displaced'] = body_size > (prev_body * params.displacement_mult)
    
    # Fair Value Gaps (Bull/Bear)
    gap_up   = (df['low'] - df['high'].shift(2))
    gap_down = (df['low'].shift(2) - df['high'])
    df['fvg_bull'] = (gap_up > params.fvg_min_size * atrv)
    df['fvg_bear'] = (gap_down > params.fvg_min_size * atrv)

    return df


def score_bar(row: pd.Series, params: NQAlphaParams) -> int:
    """The Hunter Trigger: +/-10 if all structural conditions align."""
    # 1. NY Regular Session (09:30 - 16:00)
    ts = row.name
    t_min = ts.hour * 60 + ts.minute
    if not (570 <= t_min <= 960): return 0
    
    meta  = float(row.get('meta_bias', 0.0))
    mb_th = params.meta_bias_threshold
    
    # LONG: Pivot Break OR Low Sweep + MSS + Momentum
    if (row.get('pivot_break_up', False) or row.get('sweep_l', False)) and row.get('mom', 0) > 0:
        if row.get('displaced', False) and row['close'] > row['swing_h']:
            if meta < -mb_th: return 0
            return 10
        
    # SHORT: Pivot Break OR High Sweep + MSS + Momentum
    if (row.get('pivot_break_dn', False) or row.get('sweep_h', False)) and row.get('mom', 0) < 0:
        if row.get('displaced', False) and row['close'] < row['swing_l']:
            if meta > mb_th: return 0
            return -10
        
    # Fallback to FVG if displacement is extreme
    if row.get('fvg_bull', False) and row['displaced'] and row['close'] > row['swing_h'] and row.get('mom', 0) > 0:
        if meta < -mb_th: return 0
        return 10
    if row.get('fvg_bear', False) and row['displaced'] and row['close'] < row['swing_l'] and row.get('mom', 0) < 0:
        if meta > mb_th: return 0
        return -10
        
    return 0

def session_filter(ts) -> bool:
    h = ts.hour; m = ts.minute
    t = h * 60 + m
    return 570 <= t <= 960

DEFAULT_PARAMS = NQAlphaParams()
