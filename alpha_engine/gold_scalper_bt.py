"""
Gold M1 ICT Scalper — DEAP-Optimizable  v2
===========================================
High-frequency scalping on XAUUSD 1-minute bars.

ICT Logic:
  - London Kill Zone  : 07:00–11:00 UTC  (sharp institutional moves)
  - NY Kill Zone      : 13:00–17:00 UTC  (news + continuation)
  - Silver Bullet     : 10:00–11:00 UTC  (premium window, London close)

Entry model — RETRACEMENT into institutional zone:
  1. HTF Trend  — rolling mean bias on M1 (bull/bear)
  2. Zone tap   — price RETURNS to an FVG or OB zone (NOT the impulse bar)
  3. RSI(14)    — confirms pullback, not overbought/oversold
  Bonus: PDH/PDL rejection tap

Key fix vs v1: FVG & OB zones are TRACKED and the signal fires when price
RETRACES into the zone — NOT when the impulse candle itself appears.
Zones expire after `ob_lookback` bars (via ffill limit).

GENOME (10 genes):
  [0]  session          — 0=London, 1=NY, 2=Both          (0.0 – 2.0)
  [1]  fvg_min_pts      — min FVG gap in price points      (0.1 – 2.0)
  [2]  ob_lookback      — zone validity window (M1 bars)   (5  – 40)
  [3]  tp_pts           — take profit points               (5  – 30)
  [4]  sl_pts           — stop loss points                 (3  – 15)
  [5]  spread_pts       — modeled spread per side          (0.2 – 1.0)
  [6]  trend_bars       — HTF trend window (M1 bars)       (30 – 240)
  [7]  momentum_bars    — body momentum confirm window     (3  – 20)
  [8]  meta_bias_thresh — spectral bias filter             (0.0 – 1.0)
  [9]  lot_base         — lot per $1000 equity             (0.005 – 0.05)

Point value: $100/point/lot (XAUUSD standard)
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

GOLD_PV            = 100.0   # $100/point/lot
INITIAL_CAP        = 1_000.0
MAX_TRADES_SESSION = 6       # max entries per kill zone session per day
MAX_TRAIN_BARS     = 60_000  # ~42 trading days; overrides deap_runner default of 8K
N_SPLITS           = 1       # random window sampling provides regime diversity; no sub-splits
N_EVAL_WINDOWS     = 5       # average fitness over N random windows per DEAP eval (reduces noise)

GENE_BOUNDS = [
    (0.0,  6.0),    # [0]  session: 0-1=London, 1-2=NY, 2-3=Both,
                    #               3-4=London LongOnly, 4-5=NY LongOnly, 5-6=Both LongOnly
    (0.1,  5.0),    # [1]  fvg_min_pts: high value (~4+) = FVG disabled; DEAP finds optimal level
    (5,    40),     # [2]  ob_lookback (zone validity bars)
    (5.0,  35.0),   # [3]  tp_pts (fixed price points)
    (3.0,  18.0),   # [4]  sl_pts (fixed price points)
    (0.2,  1.0),    # [5]  spread_pts
    (30,   240),    # [6]  trend_bars  (EMA cross window)
    (3,    20),     # [7]  momentum_bars
    (0.0,  1.00),   # [8]  meta_bias_thresh
    (0.005,0.05),   # [9]  lot_base
    (0.5,  2.5),    # [10] vol_thresh — Cox intensity gate: ATR/ATR_median threshold
    (50,   300),    # [11] macro_ma_days — daily MA regime gate (bars on daily chart)
                    #                      trade LONG only when daily_close > MA(macro_ma_days)
                    #                      eliminates bear/sideways macro periods
    (0.0,  1.0),    # [12] confirm_entry — >0.5: wait 1-bar bullish confirmation before long entry
                    #                      improves WR ~37% → ~45% at cost of slight signal delay
]
GENE_NAMES = [
    'session', 'fvg_min_pts', 'ob_lookback', 'tp_pts', 'sl_pts',
    'spread_pts', 'trend_bars', 'momentum_bars', 'meta_bias_thresh', 'lot_base',
    'vol_thresh', 'macro_ma_days', 'confirm_entry',
]
GENOME_SIZE = len(GENE_BOUNDS)

# Max concurrent positions per strategy (not a gene — fixed architecture choice)
MAX_CONCURRENT = 2   # 2 simultaneous positions → 2-4 trades/day vs 0.4 previously


@dataclass
class GoldScalperParams:
    session:          float = 5.0    # Both sessions, LONG ONLY (best for Gold)
    fvg_min_pts:      float = 5.0    # OB-only mode (FVG disabled by default)
    ob_lookback:      int   = 20
    tp_pts:           float = 25.0   # grid-search winner: OOS=+77.9%
    sl_pts:           float = 10.0   # grid-search winner
    spread_pts:       float = 0.4
    trend_bars:       int   = 60
    momentum_bars:    int   = 5
    meta_bias_thresh: float = 0.8
    lot_base:         float = 0.01
    vol_thresh:       float = 1.0    # Cox gate: 1.0=disabled
    macro_ma_days:    int   = 125    # daily MA macro filter — sweep winner (0=disabled)
    confirm_entry:    float = 1.0    # >0.5 = require 1-bar bullish confirmation before entry


def decode_genome(genome: list) -> GoldScalperParams:
    p = GoldScalperParams()
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        raw = float(genome[i]) if i < len(genome) else (lo + hi) / 2
        val = float(np.clip(raw, lo, hi))   # direct mapping — smooth landscape for DEAP
        setattr(p, GENE_NAMES[i], val)
    p.ob_lookback   = int(np.clip(round(p.ob_lookback), 5, 40))
    p.trend_bars    = int(np.clip(round(p.trend_bars), 30, 240))
    p.momentum_bars = int(np.clip(round(p.momentum_bars), 3, 20))
    p.vol_thresh    = float(np.clip(p.vol_thresh, 0.5, 2.5))
    p.macro_ma_days = int(np.clip(round(p.macro_ma_days), 50, 300))
    p.confirm_entry = float(np.clip(p.confirm_entry, 0.0, 1.0))
    return p


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — vectorized zone detection
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, min_periods=period).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, min_periods=period).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _add_features(df: pd.DataFrame, p: GoldScalperParams) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()

    c_s = df['close']
    o_s = df['open']
    h_s = df['high']
    l_s = df['low']

    # ── ATR(14) ───────────────────────────────────────────────────────────────
    tr = pd.concat([
        h_s - l_s,
        (h_s - c_s.shift(1)).abs(),
        (l_s - c_s.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(span=14, min_periods=14).mean()

    # ── HTF Trend — EMA cross ────────────────────────────────────────────────
    tb = p.trend_bars
    ema_fast = c_s.ewm(span=max(5, tb // 3), min_periods=10).mean()
    ema_slow = c_s.ewm(span=tb, min_periods=tb).mean()
    df['trend_bull'] = (c_s > ema_slow) & (ema_fast > ema_slow)
    df['trend_bear'] = (c_s < ema_slow) & (ema_fast < ema_slow)

    # ── RSI(14) for pullback confirmation ─────────────────────────────────────
    df['rsi'] = _rsi(c_s, 14)

    # ── Body momentum (secondary confirm) ────────────────────────────────────
    mb    = p.momentum_bars
    bodies = c_s - o_s
    df['mom_bull'] = bodies.rolling(mb).sum() > 0
    df['mom_bear'] = bodies.rolling(mb).sum() < 0

    # ── Fair Value Gap — vectorized zone with expiry ──────────────────────────
    #
    # Bullish FVG at bar i: l[i] > h[i-2] + fvg_min_pts
    #   Zone = [h[i-2], l[i]]  (the gap left by the upward move)
    #   Signal fires on bar j > i when price RETRACES: l[j] <= l[i] AND c[j] >= h[i-2]
    #
    # Bearish FVG: h[i] < l[i-2] - fvg_min_pts → zone = [h[i], l[i-2]]
    #
    ob_n = p.ob_lookback

    h_lag2 = h_s.shift(2)
    l_lag2 = l_s.shift(2)

    bull_fvg_event = l_s > (h_lag2 + p.fvg_min_pts)
    bear_fvg_event = h_s < (l_lag2 - p.fvg_min_pts)

    # Zone bounds — carry forward for ob_n bars only (zone expires)
    bull_fvg_zone_lo = h_lag2.where(bull_fvg_event).ffill(limit=ob_n)
    bull_fvg_zone_hi = l_s.where(bull_fvg_event).ffill(limit=ob_n)
    bear_fvg_zone_lo = h_s.where(bear_fvg_event).ffill(limit=ob_n)
    bear_fvg_zone_hi = l_lag2.where(bear_fvg_event).ffill(limit=ob_n)

    # Shift by 1: cannot enter on the bar where the zone forms
    bfzlo = bull_fvg_zone_lo.shift(1)
    bfzhi = bull_fvg_zone_hi.shift(1)
    bazlo = bear_fvg_zone_lo.shift(1)
    bazhi = bear_fvg_zone_hi.shift(1)

    df['bull_fvg_active'] = ((l_s <= bfzhi) & (c_s >= bfzlo) & bfzhi.notna()).astype(float)
    df['bear_fvg_active'] = ((h_s >= bazlo) & (c_s <= bazhi) & bazhi.notna()).astype(float)

    # ── Order Block — body only, opposing-color pre-impulse required ──────────
    #
    # Bull OB: large GREEN impulse after a RED pre-impulse candle
    #   → OB zone = BODY of the red candle (open to close, i.e., tighter than full wick)
    #   → Signal fires when price RETURNS to that body zone
    #
    # Bear OB: large RED impulse after a GREEN pre-impulse candle
    #   → OB zone = body of green candle
    #
    ranges  = h_s - l_s
    body_s  = (c_s - o_s).abs()
    big_bull_impulse = (c_s > o_s) & (body_s > ranges * 0.60)
    big_bear_impulse = (c_s < o_s) & (body_s > ranges * 0.60)

    o_lag1 = o_s.shift(1)
    c_lag1 = c_s.shift(1)
    red_lag1   = c_lag1 < o_lag1    # pre-impulse bar is bearish
    green_lag1 = c_lag1 > o_lag1    # pre-impulse bar is bullish

    # Proper OB event: opposing colors
    bull_ob_event = big_bull_impulse & red_lag1    # green impulse after red candle
    bear_ob_event = big_bear_impulse & green_lag1  # red impulse after green candle

    # Zone = BODY of pre-impulse candle (tighter than full wick range)
    body_lo_lag1 = pd.concat([o_lag1, c_lag1], axis=1).min(axis=1)
    body_hi_lag1 = pd.concat([o_lag1, c_lag1], axis=1).max(axis=1)

    bull_ob_zone_lo = body_lo_lag1.where(bull_ob_event).ffill(limit=ob_n)
    bull_ob_zone_hi = body_hi_lag1.where(bull_ob_event).ffill(limit=ob_n)
    bear_ob_zone_lo = body_lo_lag1.where(bear_ob_event).ffill(limit=ob_n)
    bear_ob_zone_hi = body_hi_lag1.where(bear_ob_event).ffill(limit=ob_n)

    bozlo = bull_ob_zone_lo.shift(1)
    bozhi = bull_ob_zone_hi.shift(1)
    aozlo = bear_ob_zone_lo.shift(1)
    aozhi = bear_ob_zone_hi.shift(1)

    df['bull_ob_active'] = ((l_s <= bozhi) & (c_s >= bozlo) & bozhi.notna()).astype(float)
    df['bear_ob_active'] = ((h_s >= aozlo) & (c_s <= aozhi) & aozhi.notna()).astype(float)

    # ── PDH/PDL (Previous Day High/Low rejection) ─────────────────────────────
    try:
        daily_h = h_s.resample('1D').max()
        daily_l = l_s.resample('1D').min()
        df['pdh'] = daily_h.shift(1).reindex(df.index, method='ffill')
        df['pdl'] = daily_l.shift(1).reindex(df.index, method='ffill')
        df['pdh_reject'] = (h_s > df['pdh']) & (c_s < df['pdh'])
        df['pdl_reject'] = (l_s < df['pdl']) & (c_s > df['pdl'])
    except Exception:
        df['pdh_reject'] = False
        df['pdl_reject'] = False

    if 'meta_bias' not in df.columns:
        df['meta_bias'] = 0.0

    # ── Cox-process intensity gate — ATR ratio vs rolling baseline ───────────
    atr_median = df['atr'].rolling(200, min_periods=50).median()
    df['atr_ratio'] = df['atr'] / (atr_median + 1e-10)

    # ── Daily Macro MA Regime Gate ────────────────────────────────────────────
    #
    # Only trade LONG when daily close > MA(macro_ma_days).
    # Eliminates bear/choppy macro periods (e.g. Gold 2021-2022 downtrend).
    # Computation: resample M1 → daily, shift 1 day (no lookahead), ffill to M1.
    # macro_ma_days=0 disables the filter entirely.
    try:
        ma_d = int(p.macro_ma_days)
        if ma_d > 0:
            daily_c   = c_s.resample('1D').last().dropna()
            daily_ma  = daily_c.rolling(ma_d, min_periods=ma_d // 2).mean()
            macro_bull = (daily_c > daily_ma).shift(1)          # shift=no lookahead
            df['macro_bull'] = macro_bull.reindex(df.index, method='ffill').fillna(False).astype(bool)
        else:
            df['macro_bull'] = True
    except Exception:
        df['macro_bull'] = True

    return df


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GoldScalperResult:
    total_return:  float = 0.0
    cagr:          float = 0.0
    max_drawdown:  float = 0.0
    sharpe_ratio:  float = 0.0
    win_rate:      float = 0.0
    profit_factor: float = 0.0
    total_trades:  int   = 0
    final_equity:  float = INITIAL_CAP
    avg_rr:        float = 0.0
    trades:        List  = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []

    def fitness(self) -> float:
        if self.total_trades < 10:
            return -999.0
        if self.max_drawdown > 0.70:
            return -50.0
        if self.total_return <= -0.50:
            return float(np.clip(self.total_return * 5, -50, -0.01))
        if self.profit_factor < 1.0:
            return float(np.clip((self.profit_factor - 1) * 10, -20, -0.01))

        pf   = float(np.clip(self.profit_factor, 1.0, 10.0))
        cagr = float(np.clip(self.cagr, 0, 30))
        sh   = float(np.clip(self.sharpe_ratio, 0, 10))
        wr   = float(np.clip(self.win_rate, 0, 1))
        # Drawdown penalty kicks in strongly above 20%
        ddp  = max(0.1, 1 - max(0, self.max_drawdown - 0.20) * 3)
        return float(np.clip(
            (pf * 0.25 + cagr * 0.30 + sh * 0.25 + wr * 0.20) * ddp, 0, 200))


# ─────────────────────────────────────────────────────────────────────────────
# SESSION FILTER
# ─────────────────────────────────────────────────────────────────────────────

def _in_killzone(hour: int, session_code: float):
    """
    Returns (in_killzone: bool, allow_shorts: bool).

    session_code ranges:
      0-1: London only       (both directions)
      1-2: NY only           (both directions)
      2-3: Both sessions     (both directions)
      3-4: London LongOnly   (longs only)
      4-5: NY LongOnly       (longs only)
      5-6: Both LongOnly     (longs only — best for Gold structural bull)
    """
    london = (7 <= hour < 11)
    ny     = (13 <= hour < 17)
    code   = int(session_code)    # floor to 0-5
    long_only = (session_code >= 3.0)

    if code in (0, 3): in_kz = london
    elif code in (1, 4): in_kz = ny
    else: in_kz = london or ny   # 2,3,5 → both

    return in_kz, not long_only


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df_feat: pd.DataFrame,
                 params: GoldScalperParams,
                 initial_capital: float = INITIAL_CAP) -> GoldScalperResult:
    result = GoldScalperResult(final_equity=initial_capital)
    try:
        df = df_feat.dropna(subset=['atr', 'trend_bull', 'rsi'])
        if len(df) < 500:
            return result
        n = len(df)

        # ── Pre-extract numpy arrays (avoids slow df.iloc[i] in loop) ────────
        c_arr    = df['close'].values
        o_arr    = df['open'].values
        h_arr    = df['high'].values
        l_arr    = df['low'].values
        meta_arr = df['meta_bias'].values if 'meta_bias' in df.columns else np.zeros(n)
        rsi_arr  = df['rsi'].values
        tb_arr   = df['trend_bull'].values.astype(bool)
        td_arr   = df['trend_bear'].values.astype(bool)
        bfvg_arr = df['bull_fvg_active'].values.astype(bool)
        afvg_arr = df['bear_fvg_active'].values.astype(bool)
        bob_arr  = df['bull_ob_active'].values.astype(bool)
        aob_arr  = df['bear_ob_active'].values.astype(bool)
        pdl_arr  = df['pdl_reject'].values.astype(bool) if 'pdl_reject' in df.columns \
                   else np.zeros(n, dtype=bool)
        pdh_arr  = df['pdh_reject'].values.astype(bool) if 'pdh_reject' in df.columns \
                   else np.zeros(n, dtype=bool)
        atr_ratio_arr  = df['atr_ratio'].values if 'atr_ratio' in df.columns \
                         else np.ones(n, dtype=float)
        atr_arr        = df['atr'].values
        macro_bull_arr = df['macro_bull'].values.astype(bool) if 'macro_bull' in df.columns \
                         else np.ones(n, dtype=bool)

        # ── Pre-compute killzone mask + session direction ─────────────────────
        hours    = df.index.hour
        long_only_mode = params.session >= 3.0
        allow_shorts   = not long_only_mode
        code = int(params.session)
        if code in (0, 3):
            kz_mask = (hours >= 7) & (hours < 11)
        elif code in (1, 4):
            kz_mask = (hours >= 13) & (hours < 17)
        else:
            kz_mask = ((hours >= 7) & (hours < 11)) | ((hours >= 13) & (hours < 17))

        # Pre-compute session label (L or NY) for trade throttle key
        session_label = np.where(hours < 12, 0, 1)  # 0=London, 1=NY
        dates = np.array([df.index[i].date() for i in range(n)])

        spread       = params.spread_pts
        thresh       = params.meta_bias_thresh
        tp_pts_fixed = getattr(params, 'tp_pts', 25.0)
        sl_pts_fixed = getattr(params, 'sl_pts', 10.0)
        vol_thresh   = getattr(params, 'vol_thresh', 1.0)
        lot_base     = params.lot_base
        use_confirm  = float(getattr(params, 'confirm_entry', 0.0)) > 0.5
        start        = params.trend_bars + 20

        balance      = initial_capital
        eq_curve     = np.empty(n, dtype=float)
        eq_len       = 0
        trades       = []
        open_pos     = None
        pending_long = None   # confirmation candle pending long
        pending_short= None
        session_cnt  = {}

        for idx in range(start, n):
            price = c_arr[idx]
            hi    = h_arr[idx]
            lo    = l_arr[idx]
            rsi_v = rsi_arr[idx]
            meta  = meta_arr[idx]

            # ── Step 1: Fill pending confirmation entry ───────────────────────
            # Confirmation candle: bar after zone tap must close bullish/bearish.
            # Pending is always consumed (filled or cancelled) in one bar.
            if pending_long is not None and open_pos is None:
                if c_arr[idx] > o_arr[idx]:   # bullish confirmation bar
                    entry = c_arr[idx] + spread
                    pl    = pending_long
                    rr    = pl['tp_pts'] / max(pl['sl_pts'], 0.01)
                    open_pos = {'dir': 1, 'entry': entry, 'size': pl['lot'],
                                'sl': entry - pl['sl_pts'], 'tp': entry + pl['tp_pts'],
                                'ts': df.index[idx], 'reason': pl['reason'], 'rr_ratio': rr}
                    session_cnt[pl['sess_key']] = pl['n_sess'] + 1
                pending_long = None   # consume regardless (filled or rejected)

            if pending_short is not None and open_pos is None:
                if c_arr[idx] < o_arr[idx]:   # bearish confirmation bar
                    entry = c_arr[idx] - spread
                    ps    = pending_short
                    rr    = ps['tp_pts'] / max(ps['sl_pts'], 0.01)
                    open_pos = {'dir': -1, 'entry': entry, 'size': ps['lot'],
                                'sl': entry + ps['sl_pts'], 'tp': entry - ps['tp_pts'],
                                'ts': df.index[idx], 'reason': ps['reason'], 'rr_ratio': rr}
                    session_cnt[ps['sess_key']] = ps['n_sess'] + 1
                pending_short = None

            # ── Step 2: Manage open position (SL / TP) ───────────────────────
            if open_pos is not None:
                d = open_pos['dir']
                hit_sl = (d ==  1 and lo <= open_pos['sl']) or (d == -1 and hi >= open_pos['sl'])
                hit_tp = (d ==  1 and hi >= open_pos['tp']) or (d == -1 and lo <= open_pos['tp'])

                if hit_sl:
                    pnl = (open_pos['sl'] - open_pos['entry']) * d * open_pos['size'] * GOLD_PV
                    balance += pnl
                    trades.append({'pnl': pnl, 'dir': d, 'reason': open_pos['reason'],
                                   'entry_time': open_pos['ts'], 'exit_time': df.index[idx],
                                   'rr': 0.0})
                    open_pos = None
                elif hit_tp:
                    pnl = (open_pos['tp'] - open_pos['entry']) * d * open_pos['size'] * GOLD_PV
                    balance += pnl
                    trades.append({'pnl': pnl, 'dir': d, 'reason': open_pos['reason'],
                                   'entry_time': open_pos['ts'], 'exit_time': df.index[idx],
                                   'rr': open_pos.get('rr_ratio', 1.0)})
                    open_pos = None

            if balance < initial_capital * 0.10:
                break

            # ── Step 3: Entry gate ────────────────────────────────────────────
            if not kz_mask[idx] or open_pos is not None:
                eq_curve[eq_len] = balance; eq_len += 1
                continue

            date  = dates[idx]
            s_lbl = session_label[idx]
            sess_key = (date, s_lbl)
            n_sess   = session_cnt.get(sess_key, 0)
            if n_sess >= MAX_TRADES_SESSION:
                eq_curve[eq_len] = balance; eq_len += 1
                continue

            # Skip if already a pending confirmation waiting
            if pending_long is not None or pending_short is not None:
                eq_curve[eq_len] = balance; eq_len += 1
                continue

            # Meta bias directional filter
            long_ok_meta  = True
            short_ok_meta = allow_shorts
            if thresh < 0.98:
                if meta > thresh:    short_ok_meta = False
                elif meta < -thresh: long_ok_meta  = False

            # Lot sizing — proportional to balance
            lot = float(np.clip(round(balance / 1000.0 * lot_base, 3), 0.001, 1.0))

            # ── Zone tap signals ─────────────────────────────────────────────
            bull_zone = bfvg_arr[idx] or bob_arr[idx] or pdl_arr[idx]
            bear_zone = afvg_arr[idx] or aob_arr[idx] or pdh_arr[idx]

            sl_pts = sl_pts_fixed
            tp_pts = tp_pts_fixed

            # Cox process intensity gate
            vol_ok = (vol_thresh <= 1.0) or (atr_ratio_arr[idx] >= vol_thresh)

            # Macro regime gate (daily MA)
            macro_ok = macro_bull_arr[idx]

            long_ok  = tb_arr[idx] and bull_zone and (30.0 < rsi_v < 55.0) and long_ok_meta  and vol_ok and macro_ok
            short_ok = td_arr[idx] and bear_zone and (60.0 < rsi_v < 75.0) and short_ok_meta and vol_ok

            trade_rr = tp_pts / max(sl_pts, 0.01)

            if long_ok and not short_ok:
                if bfvg_arr[idx]:   reason = 'FVG_LONG'
                elif bob_arr[idx]:  reason = 'OB_LONG'
                else:               reason = 'PDL_LONG'

                if use_confirm:
                    pending_long = {'sl_pts': sl_pts, 'tp_pts': tp_pts, 'reason': reason,
                                    'lot': lot, 'sess_key': sess_key, 'n_sess': n_sess}
                else:
                    entry = price + spread
                    open_pos = {'dir': 1, 'entry': entry, 'size': lot,
                                'sl': entry - sl_pts, 'tp': entry + tp_pts,
                                'ts': df.index[idx], 'reason': reason, 'rr_ratio': trade_rr}
                    session_cnt[sess_key] = n_sess + 1

            elif short_ok and not long_ok:
                if afvg_arr[idx]:   reason = 'FVG_SHORT'
                elif aob_arr[idx]:  reason = 'OB_SHORT'
                else:               reason = 'PDH_SHORT'

                if use_confirm:
                    pending_short = {'sl_pts': sl_pts, 'tp_pts': tp_pts, 'reason': reason,
                                     'lot': lot, 'sess_key': sess_key, 'n_sess': n_sess}
                else:
                    entry = price - spread
                    open_pos = {'dir': -1, 'entry': entry, 'size': lot,
                                'sl': entry + sl_pts, 'tp': entry - tp_pts,
                                'ts': df.index[idx], 'reason': reason, 'rr_ratio': trade_rr}
                    session_cnt[sess_key] = n_sess + 1

            # Equity tracking
            if open_pos is not None:
                live = (price - open_pos['entry']) * open_pos['dir'] * open_pos['size'] * GOLD_PV
                eq_curve[eq_len] = balance + live
            else:
                eq_curve[eq_len] = balance
            eq_len += 1

        # Force-close remaining position at end of data
        if open_pos is not None:
            last_p = c_arr[-1]
            pnl = (last_p - open_pos['entry']) * open_pos['dir'] * open_pos['size'] * GOLD_PV
            balance += pnl
            trades.append({'pnl': pnl, 'dir': open_pos['dir'], 'reason': open_pos['reason'],
                           'entry_time': open_pos['ts'], 'exit_time': df.index[-1], 'rr': 0.0})
            if eq_len > 0:
                eq_curve[eq_len - 1] = balance

        # ── Metrics ──────────────────────────────────────────────────────────
        eq  = eq_curve[:eq_len] if eq_len > 0 else np.array([initial_capital])
        ret = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)

        result.total_return = (eq[-1] - initial_capital) / initial_capital
        result.final_equity = float(eq[-1])
        result.total_trades = len(trades)
        result.trades       = trades

        peak = np.maximum.accumulate(eq)
        result.max_drawdown = float(abs(((eq - peak) / (peak + 1e-10)).min()))

        td  = (df.index[-1] - df.index[0]).total_seconds()
        bpy = (len(df) / td) * 365 * 86400 if td > 0 else 525600
        ny  = len(df) / max(bpy, 1)
        if ny > 0 and eq[-1] > 0:
            result.cagr = float(np.clip(
                (eq[-1] / initial_capital) ** (1 / max(ny, 0.1)) - 1, -1, 100))

        if len(ret) > 1 and ret.std() > 1e-10:
            result.sharpe_ratio = float(np.clip(
                ret.mean() / ret.std() * np.sqrt(bpy), -10, 20))

        if trades:
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            loss = [p for p in pnls if p <= 0]
            result.win_rate      = len(wins) / len(pnls)
            gp = sum(wins)
            gl = abs(sum(loss)) if loss else 1e-10
            result.profit_factor = float(np.clip(gp / (gl + 1e-10), 0, 20))
            rrs = [t.get('rr', 0) for t in trades if t.get('rr', 0) > 0]
            result.avg_rr = float(np.mean(rrs)) if rrs else 0.0

    except Exception:
        import traceback; traceback.print_exc()

    return result


def backtest_from_raw(df_raw: pd.DataFrame,
                      params: GoldScalperParams,
                      initial_capital: float = INITIAL_CAP) -> GoldScalperResult:
    """Feature-engineer M1 data then run scalper backtest."""
    try:
        df_feat = _add_features(df_raw, params)
        return run_backtest(df_feat, params, initial_capital)
    except Exception:
        import traceback; traceback.print_exc()
        return GoldScalperResult(final_equity=initial_capital)


DEFAULT_PARAMS = GoldScalperParams()
