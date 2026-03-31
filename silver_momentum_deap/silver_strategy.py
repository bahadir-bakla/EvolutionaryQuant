"""
Silver Momentum Strategy — EvolutionaryQuant
=============================================
Pure momentum strategy designed for XAGUSD (Silver).

Silver karakteri:
- Gold'dan 3-4x daha volatil
- Güçlü trend/momentum dönemleri
- Industrial demand cycle → trendler uzun sürer
- Countertrend yapma — momentum ile git

Strateji mantığı:
- Multi-period ROC (hızlı/orta/yavaş momentum)
- EMA stack alignment (tam hizalama = en güçlü sinyal)
- MACD crossover + histogram
- ADX trend strength filter
- N-period breakout (price at X-period high/low)
- ATR volatility filter (avoid dead markets)
- Volume surge (momentum confirmation)
- Session filter (London + NY)

12-gene DEAP Genome.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

# ─────────────────────────────────────────────────────────────────────
# GENOME DEFINITION
# ─────────────────────────────────────────────────────────────────────
GENE_BOUNDS = [
    (2,    8),      # [0]  roc_fast_period      Fast ROC
    (5,    20),     # [1]  roc_mid_period        Mid ROC
    (15,   50),     # [2]  roc_slow_period       Slow ROC
    (5,    15),     # [3]  ema_fast              Fast EMA
    (15,   40),     # [4]  ema_mid               Mid EMA
    (40,   100),    # [5]  ema_slow              Slow EMA
    (7,    25),     # [6]  adx_period            ADX period
    (15,   35),     # [7]  adx_threshold         Min ADX for entry
    (10,   40),     # [8]  breakout_period       N-period breakout window
    (1.5,  6.0),    # [9]  tp_atr_mult           Take profit × ATR
    (0.3,  2.0),    # [10] sl_atr_mult           Stop loss × ATR
    (2,    6),      # [11] min_score             Minimum signal score
    (0.0,  0.8),    # [12] meta_bias_threshold   Spectral Bias Filter
]

GENE_NAMES = [
    'roc_fast_period', 'roc_mid_period', 'roc_slow_period',
    'ema_fast', 'ema_mid', 'ema_slow',
    'adx_period', 'adx_threshold', 'breakout_period',
    'tp_atr_mult', 'sl_atr_mult', 'min_score', 'meta_bias_threshold'
]
GENOME_SIZE = len(GENE_BOUNDS)


@dataclass
class SilverMomentumParams:
    """Parameters optimizable by DEAP."""
    roc_fast_period:  int   = 4
    roc_mid_period:   int   = 10
    roc_slow_period:  int   = 25
    ema_fast:         int   = 9
    ema_mid:          int   = 21
    ema_slow:         int   = 55
    adx_period:       int   = 14
    adx_threshold:    float = 20.0
    breakout_period:  int   = 20
    tp_atr_mult:      float = 3.0
    sl_atr_mult:      float = 1.0
    min_score:        int   = 3
    meta_bias_threshold: float = 0.0


def decode_genome(genome: list) -> SilverMomentumParams:
    """Decode DEAP genome [0,1]^12 → SilverMomentumParams."""
    p = SilverMomentumParams()
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        if i >= len(genome):
            break
        raw = float(genome[i])
        val = lo + abs(raw % 1.0) * (hi - lo)
        setattr(p, GENE_NAMES[i], val)
    # Integer conversions
    p.roc_fast_period  = int(np.clip(round(p.roc_fast_period), 2, 8))
    p.roc_mid_period   = int(np.clip(round(p.roc_mid_period),  5, 20))
    p.roc_slow_period  = int(np.clip(round(p.roc_slow_period), 15, 50))
    p.ema_fast         = int(np.clip(round(p.ema_fast),        5, 15))
    p.ema_mid          = int(np.clip(round(p.ema_mid),         15, 40))
    p.ema_slow         = int(np.clip(round(p.ema_slow),        40, 100))
    p.adx_period       = int(np.clip(round(p.adx_period),      7, 25))
    p.breakout_period  = int(np.clip(round(p.breakout_period), 10, 40))
    p.min_score        = int(np.clip(round(p.min_score),       2, 6))
    p.meta_bias_threshold = float(np.clip(p.meta_bias_threshold, 0.0, 0.8))
    return p


# ─────────────────────────────────────────────────────────────────────
# INDICATOR FUNCTIONS
# ─────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df['high'] - df['low']
    hc  = (df['high'] - df['close'].shift(1)).abs()
    lc  = (df['low']  - df['close'].shift(1)).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _roc(close: pd.Series, period: int) -> pd.Series:
    """Rate of Change (%)."""
    return (close / close.shift(period) - 1) * 100


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series,
          fast: int = 12, slow: int = 26, sig: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast  = _ema(close, fast)
    ema_slow  = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal    = _ema(macd_line, sig)
    histogram = macd_line - signal
    return macd_line, signal, histogram


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl   = df['high'] - df['low']
    hc   = (df['high'] - df['close'].shift(1)).abs()
    lc   = (df['low']  - df['close'].shift(1)).abs()
    tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(period).sum()
    hh   = df['high'].diff(); ll = -df['low'].diff()
    dmp  = pd.Series(np.where((hh > ll) & (hh > 0), hh, 0.0), index=df.index).rolling(period).sum()
    dmm  = pd.Series(np.where((ll > hh) & (ll > 0), ll, 0.0), index=df.index).rolling(period).sum()
    dip  = 100 * dmp / (tr + 1e-10)
    dim  = 100 * dmm / (tr + 1e-10)
    dx   = 100 * (dip - dim).abs() / (dip + dim + 1e-10)
    return dx.rolling(period).mean()


def _relative_volume(volume: pd.Series, period: int = 20) -> pd.Series:
    avg = volume.rolling(period).mean()
    return volume / (avg + 1e-10)


def session_filter(ts) -> bool:
    """London (8-12 UTC) or NY (13-20 UTC) session."""
    h = ts.hour
    return (8 <= h < 12) or (13 <= h < 20)  # UTC approximate


# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────

def add_silver_features(df: pd.DataFrame, params: SilverMomentumParams) -> pd.DataFrame:
    """Add all momentum features to XAGUSD dataframe."""
    df = df.copy()
    df.columns = df.columns.str.lower()

    close = df['close']

    # ── ATR ────────────────────────────────────────────────────────
    df['atr'] = _atr(df, 14)

    # ── Multi-period ROC ───────────────────────────────────────────
    df['roc_fast']  = _roc(close, params.roc_fast_period)
    df['roc_mid']   = _roc(close, params.roc_mid_period)
    df['roc_slow']  = _roc(close, params.roc_slow_period)

    # ROC alignment: all pointing same direction
    df['roc_all_bull'] = (df['roc_fast'] > 0) & (df['roc_mid'] > 0) & (df['roc_slow'] > 0)
    df['roc_all_bear'] = (df['roc_fast'] < 0) & (df['roc_mid'] < 0) & (df['roc_slow'] < 0)

    # ── EMA Stack ─────────────────────────────────────────────────
    df['ema_f'] = _ema(close, params.ema_fast)
    df['ema_m'] = _ema(close, params.ema_mid)
    df['ema_s'] = _ema(close, params.ema_slow)

    # Full stack alignment
    df['ema_stack_bull'] = (close > df['ema_f']) & (df['ema_f'] > df['ema_m']) & (df['ema_m'] > df['ema_s'])
    df['ema_stack_bear'] = (close < df['ema_f']) & (df['ema_f'] < df['ema_m']) & (df['ema_m'] < df['ema_s'])

    # Partial alignment (2/3)
    df['ema_partial_bull'] = (close > df['ema_f']) & (df['ema_f'] > df['ema_m'])
    df['ema_partial_bear'] = (close < df['ema_f']) & (df['ema_f'] < df['ema_m'])

    # EMA slope (trend direction)
    df['ema_m_slope'] = df['ema_m'].diff(3) / (df['ema_m'].shift(3) + 1e-10) * 100

    # ── MACD ──────────────────────────────────────────────────────
    df['macd'], df['macd_sig'], df['macd_hist'] = _macd(close, 12, 26, 9)
    df['macd_cross_bull'] = (df['macd'] > df['macd_sig']) & (df['macd'].shift(1) <= df['macd_sig'].shift(1))
    df['macd_cross_bear'] = (df['macd'] < df['macd_sig']) & (df['macd'].shift(1) >= df['macd_sig'].shift(1))
    df['macd_hist_bull']  = (df['macd_hist'] > 0) & (df['macd_hist'] > df['macd_hist'].shift(1))
    df['macd_hist_bear']  = (df['macd_hist'] < 0) & (df['macd_hist'] < df['macd_hist'].shift(1))

    # ── ADX (trend strength) ───────────────────────────────────────
    df['adx']      = _adx(df, params.adx_period)
    df['adx_ok']   = df['adx'] > params.adx_threshold

    # ── N-period Breakout ─────────────────────────────────────────
    bp = params.breakout_period
    df['period_high'] = df['high'].rolling(bp).max().shift(1)
    df['period_low']  = df['low'].rolling(bp).min().shift(1)
    df['breakout_up']  = close > df['period_high']
    df['breakout_down']= close < df['period_low']

    # ── Volume ────────────────────────────────────────────────────
    if 'volume' in df.columns and df['volume'].sum() > 0:
        df['rel_vol'] = _relative_volume(df['volume'], 20)
        df['vol_surge'] = df['rel_vol'] > 1.5
    else:
        df['vol_surge'] = False

    # ── ATR filter (avoid dead markets) ───────────────────────────
    df['atr_pct']    = df['atr'] / (close + 1e-10) * 100
    df['atr_active'] = df['atr_pct'] > 0.05   # min 0.05% ATR

    # ── Gap detection ─────────────────────────────────────────────
    df['gap_pct']     = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-10) * 100
    df['gap_up_cont'] = df['gap_pct'] > 0.20   # gap up → momentum continuation
    df['gap_dn_cont'] = df['gap_pct'] < -0.20

    return df


# ─────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────

def score_bar(row: pd.Series, params: SilverMomentumParams) -> int:
    """
    Composite momentum score for a single bar.
    Positive → LONG, Negative → SHORT.
    Range: -10 to +10
    Entry when abs(score) >= params.min_score AND adx_ok
    """
    # Hard gate: no trade in low-volatility or non-trending market
    if not bool(row.get('adx_ok', False)) or not bool(row.get('atr_active', True)):
        return 0

    meta  = float(row.get('meta_bias', 0.0))
    mb_th = params.meta_bias_threshold

    ls = 0; ss = 0   # long score, short score

    # ── 1. ROC alignment (weight 3 total) ─────────────────────────
    if bool(row.get('roc_all_bull', False)):   ls += 3
    elif bool(row.get('roc_fast', 0) > 0 and row.get('roc_mid', 0) > 0): ls += 1
    if bool(row.get('roc_all_bear', False)):   ss += 3
    elif bool(row.get('roc_fast', 0) < 0 and row.get('roc_mid', 0) < 0): ss += 1

    # ── 2. EMA Stack alignment (weight 3) ─────────────────────────
    if bool(row.get('ema_stack_bull', False)):   ls += 3
    elif bool(row.get('ema_partial_bull', False)): ls += 1
    if bool(row.get('ema_stack_bear', False)):   ss += 3
    elif bool(row.get('ema_partial_bear', False)): ss += 1

    # ── 3. MACD (weight 2) ────────────────────────────────────────
    if bool(row.get('macd_cross_bull', False)): ls += 2
    elif bool(row.get('macd_hist_bull', False)): ls += 1
    if bool(row.get('macd_cross_bear', False)): ss += 2
    elif bool(row.get('macd_hist_bear', False)): ss += 1

    # ── 4. Breakout (weight 2) ─────────────────────────────────────
    if bool(row.get('breakout_up', False)):    ls += 2
    if bool(row.get('breakout_down', False)):  ss += 2

    # ── 5. Volume surge bonus (weight 1) ──────────────────────────
    if bool(row.get('vol_surge', False)):
        if ls > ss: ls += 1
        elif ss > ls: ss += 1

    # ── 6. Gap continuation (weight 1) ────────────────────────────
    if bool(row.get('gap_up_cont', False)):  ls += 1
    if bool(row.get('gap_dn_cont', False)):  ss += 1

    net = ls - ss   # net: positive=LONG, negative=SHORT
    
    # --- SPECTRAL FILTER ---
    if mb_th >= 0.05:
        if net > 0 and meta < -mb_th:
            return 0
        elif net < 0 and meta > mb_th:
            return 0

    return net


DEFAULT_PARAMS = SilverMomentumParams()
