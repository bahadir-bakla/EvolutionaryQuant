"""
NQ Liquidity Edge Backtest — DEAP-Optimizable
===============================================
Opening Range (ORB) + Liquidity Sweep reversal strategy for NQ/MNQ.

Logic:
  1. Mark 9:30–10:00 opening range (OR) high/low
  2. Wait for price to false-break (sweep) above/below range
  3. Confirm rejection candle (body < sweep_body_ratio × wick)
  4. Enter counter-direction after confirmation
  5. Exit at TP (take_profit_atr × ATR) or SL (stop_loss_atr × ATR)

GENOME (8 genes):
  [0] sweep_extension_atr — how far price must exceed range (× ATR)  (0.2 – 2.0)
  [1] sweep_body_ratio    — rejection body / wick ratio cap           (0.1 – 0.6)
  [2] stop_loss_atr       — SL × ATR                                  (0.5 – 3.0)
  [3] take_profit_atr     — TP × ATR                                  (1.0 – 6.0)
  [4] or_end_hour         — Opening Range end minute offset (0 = 10:00, 30 = 10:30) (0 – 60)
  [5] session_end_hour    — Last entry hour (11 – 15)                 (11  – 15)
  [6] meta_bias_thresh    — |meta_bias| cap                            (0.0 – 0.7)
  [7] lot_base            — lot per $1000 equity                       (0.01 – 0.05)

Point value: $20 per point per micro-contract (MNQ)
             $200 per point per standard contract (NQ)
Auto-detects by median close price (>5000 = futures).
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

NQ_PV_MICRO   = 2.0    # MNQ $2/point
NQ_PV_FULL    = 20.0   # NQ  $20/point
SLIPPAGE_PCT  = 0.10   # 10% of ATR per side
MAX_TRADES_DAY = 2
INITIAL_CAPITAL = 1_000.0

GENE_BOUNDS = [
    (0.2,   2.0),   # [0] sweep_extension_atr
    (0.10,  0.60),  # [1] sweep_body_ratio
    (0.5,   3.0),   # [2] stop_loss_atr
    (1.0,   6.0),   # [3] take_profit_atr
    (0,    60),     # [4] or_end_offset_min
    (11,   15),     # [5] session_end_hour
    (0.0,   0.70),  # [6] meta_bias_thresh
    (0.01,  0.05),  # [7] lot_base
]
GENE_NAMES = [
    'sweep_extension_atr', 'sweep_body_ratio', 'stop_loss_atr', 'take_profit_atr',
    'or_end_offset_min', 'session_end_hour', 'meta_bias_thresh', 'lot_base',
]
GENOME_SIZE = len(GENE_BOUNDS)


@dataclass
class NQLiquidityParams:
    sweep_extension_atr: float = 0.5
    sweep_body_ratio:    float = 0.30
    stop_loss_atr:       float = 1.0
    take_profit_atr:     float = 2.5
    or_end_offset_min:   int   = 0      # 0 = 10:00, 30 = 10:30
    session_end_hour:    int   = 12
    meta_bias_thresh:    float = 0.20
    lot_base:            float = 0.01


def decode_genome(genome: list) -> NQLiquidityParams:
    p = NQLiquidityParams()
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        raw = float(genome[i]) if i < len(genome) else 0.5
        val = lo + abs(raw % 1.0) * (hi - lo)
        setattr(p, GENE_NAMES[i], val)
    p.or_end_offset_min = int(np.clip(round(p.or_end_offset_min), 0, 60))
    p.session_end_hour  = int(np.clip(round(p.session_end_hour), 11, 15))
    return p


# ─────────────────────────────────────────────────────────────────────
# BACKTEST RESULT
# ─────────────────────────────────────────────────────────────────────

@dataclass
class NQLiquidityResult:
    total_return:  float = 0.0
    cagr:          float = 0.0
    max_drawdown:  float = 0.0
    sharpe_ratio:  float = 0.0
    win_rate:      float = 0.0
    profit_factor: float = 0.0
    total_trades:  int   = 0
    final_equity:  float = INITIAL_CAPITAL
    avg_rr:        float = 0.0
    trades:        List  = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []

    def fitness(self) -> float:
        if self.total_trades < 2:
            return -999.0
        if self.total_return <= 0:
            return float(np.clip(self.total_return * 5, -50, -0.01))
        if self.profit_factor < 1.0:
            return float(np.clip((self.profit_factor - 1) * 10, -20, -0.01))
        if self.max_drawdown > 0.70:
            return -50.0
        pf   = float(np.clip(self.profit_factor, 1.0, 15))
        cagr = float(np.clip(self.cagr, 0, 20))
        sh   = float(np.clip(self.sharpe_ratio, 0, 10))
        wr   = float(np.clip(self.win_rate, 0, 1))
        rr   = float(np.clip(self.avg_rr, 0, 10))
        ddp  = max(0.1, 1 - max(0, self.max_drawdown - 0.15) * 2.5)
        return float(np.clip(
            (pf * 0.25 + cagr * 0.35 + sh * 0.20 + wr * 0.10 + rr * 0.10) * ddp, 0, 200))


# ─────────────────────────────────────────────────────────────────────
# BACKTEST
# ─────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame,
                 params: NQLiquidityParams,
                 initial_capital: float = INITIAL_CAPITAL) -> NQLiquidityResult:
    result = NQLiquidityResult(final_equity=initial_capital)
    try:
        df = df.copy()
        df.columns = df.columns.str.lower()

        # Detect point value
        pv = NQ_PV_FULL if df['close'].median() > 5000 else NQ_PV_MICRO

        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low']  - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        if 'meta_bias' not in df.columns:
            df['meta_bias'] = 0.0

        valid = df.dropna(subset=['atr'])
        if len(valid) < 100:
            return result

        # OR end time: 10:00 + offset
        or_end_h = 10
        or_end_m = params.or_end_offset_min

        balance     = initial_capital
        eq_curve    = [balance]
        trades: List = []
        open_pos    = None
        daily_count: dict = {}

        # Per-day opening range state
        daily_or: dict = {}   # date -> {'high': ..., 'low': ..., 'locked': bool}

        for ts, row in valid.iterrows():
            date  = ts.date()
            hour  = ts.hour
            minute = ts.minute
            t_min = hour * 60 + minute
            price = float(row['close'])
            hi    = float(row['high'])
            lo    = float(row['low'])
            atr   = float(row.get('atr') or price * 0.002)
            meta  = float(row.get('meta_bias', 0.0))
            body  = abs(float(row['close']) - float(row['open']))
            wick_up  = hi - max(float(row['close']), float(row['open']))
            wick_dn  = min(float(row['close']), float(row['open'])) - lo
            slip  = atr * SLIPPAGE_PCT

            # ── Opening Range accumulation ─────────────────────
            is_or = (t_min >= 570) and (hour < or_end_h or
                     (hour == or_end_h and minute < or_end_m))
            if date not in daily_or:
                daily_or[date] = {'high': -np.inf, 'low': np.inf, 'locked': False}
            if is_or and not daily_or[date]['locked']:
                daily_or[date]['high'] = max(daily_or[date]['high'], hi)
                daily_or[date]['low']  = min(daily_or[date]['low'],  lo)
            elif not is_or and not daily_or[date]['locked']:
                # First bar after OR period — lock the range
                if daily_or[date]['high'] > daily_or[date]['low']:
                    daily_or[date]['locked'] = True

            # ── Manage open position ──────────────────────────
            if open_pos is not None:
                d = open_pos['dir']
                pnl = (price - open_pos['entry']) * d * open_pos['size'] * pv

                hit_sl = (d == 1  and lo <= open_pos['sl']) or \
                         (d == -1 and hi >= open_pos['sl'])
                hit_tp = (d == 1  and hi >= open_pos['tp']) or \
                         (d == -1 and lo <= open_pos['tp'])

                if balance + pnl <= 0 or hit_sl:
                    exit_p = open_pos['sl']
                    pnl = (exit_p - open_pos['entry']) * d * open_pos['size'] * pv
                    balance += pnl
                    trades.append({'pnl': pnl, 'rr': 0.0, 'dir': d,
                                   'entry_time': open_pos['ts'], 'exit_time': ts})
                    open_pos = None
                elif hit_tp:
                    exit_p = open_pos['tp']
                    pnl = (exit_p - open_pos['entry']) * d * open_pos['size'] * pv
                    balance += pnl
                    rr_val = params.take_profit_atr / max(params.stop_loss_atr, 1e-6)
                    trades.append({'pnl': pnl, 'rr': rr_val, 'dir': d,
                                   'entry_time': open_pos['ts'], 'exit_time': ts})
                    open_pos = None

            if balance < initial_capital * 0.05:
                break

            # ── Entry Logic ───────────────────────────────────
            n_today = daily_count.get(date, 0)
            or_info = daily_or.get(date, {})
            or_locked = or_info.get('locked', False)
            or_high = or_info.get('high', np.nan)
            or_low  = or_info.get('low', np.nan)

            if (open_pos is None and or_locked and
                    n_today < MAX_TRADES_DAY and
                    hour < params.session_end_hour and
                    not np.isnan(or_high) and not np.isnan(or_low)):

                sweep_ext = params.sweep_extension_atr * atr
                sl_pts    = params.stop_loss_atr   * atr
                tp_pts    = params.take_profit_atr * atr
                # Risk-based sizing: lot_base gene = risk fraction (1-5%)
                # Works for both NQ futures (pv=20) and QQQ shares (pv=1)
                risk_usd = balance * params.lot_base
                sl_usd   = max(sl_pts * pv, 1e-6)
                raw_size = risk_usd / sl_usd
                if pv >= 10:   # NQ futures: size in contracts
                    lot = float(np.clip(round(raw_size, 3), 0.001, 0.5))
                else:          # QQQ/ETF: size in shares
                    lot = float(np.clip(round(raw_size, 1), 1.0,
                                        min(1000.0, balance / max(sl_pts * 50, 1))))

                # Bear sweep: price went above OR high but closed back below
                bear_sweep = (hi > or_high + sweep_ext) and (price < or_high)
                # Body small relative to upper wick (rejection)
                bear_reject = (wick_up > 1e-6) and (body / max(wick_up, 1e-6) < params.sweep_body_ratio)

                # Bull sweep: price went below OR low but closed back above
                bull_sweep = (lo < or_low - sweep_ext) and (price > or_low)
                bull_reject = (wick_dn > 1e-6) and (body / max(wick_dn, 1e-6) < params.sweep_body_ratio)

                if (abs(meta) < params.meta_bias_thresh):
                    if bear_sweep and bear_reject and meta <= 0:
                        entry = price - slip
                        open_pos = {
                            'dir': -1, 'entry': entry,
                            'sl': entry + sl_pts,
                            'tp': entry - tp_pts,
                            'size': lot, 'ts': ts,
                        }
                        daily_count[date] = n_today + 1

                    elif bull_sweep and bull_reject and meta >= 0:
                        entry = price + slip
                        open_pos = {
                            'dir': 1, 'entry': entry,
                            'sl': entry - sl_pts,
                            'tp': entry + tp_pts,
                            'size': lot, 'ts': ts,
                        }
                        daily_count[date] = n_today + 1

            # Equity tracking
            if open_pos is not None:
                live = (price - open_pos['entry']) * open_pos['dir'] * open_pos['size'] * pv
                eq_curve.append(balance + live)
            else:
                eq_curve.append(balance)

        # Close remaining
        if open_pos is not None:
            last_price = float(valid['close'].iloc[-1])
            pnl = (last_price - open_pos['entry']) * open_pos['dir'] * open_pos['size'] * pv
            balance += pnl
            trades.append({'pnl': pnl, 'rr': 0.0, 'dir': open_pos['dir'],
                           'entry_time': open_pos['ts'], 'exit_time': valid.index[-1]})
            eq_curve[-1] = balance

        # Metrics
        eq  = np.array(eq_curve, dtype=float)
        ret = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)

        result.total_return = (eq[-1] - initial_capital) / initial_capital
        result.final_equity = float(eq[-1])
        result.total_trades = len(trades)
        result.trades       = trades

        peak = np.maximum.accumulate(eq)
        result.max_drawdown = float(abs(((eq - peak) / (peak + 1e-10)).min()))

        td  = (valid.index[-1] - valid.index[0]).total_seconds()
        bpy = (len(valid) / td) * 365 * 86400 if td > 0 else 19656
        ny  = len(valid) / max(bpy, 1)
        if ny > 0 and eq[-1] > 0:
            result.cagr = float(np.clip(
                (eq[-1] / initial_capital) ** (1 / max(ny, 0.1)) - 1, -1, 50))

        if len(ret) > 1 and ret.std() > 1e-10:
            result.sharpe_ratio = float(np.clip(
                ret.mean() / ret.std() * np.sqrt(bpy), -10, 10))

        if trades:
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            loss = [p for p in pnls if p <= 0]
            result.win_rate      = len(wins) / len(trades)
            gp = sum(wins)
            gl = abs(sum(loss)) if loss else 1e-10
            result.profit_factor = float(np.clip(gp / (gl + 1e-10), 0, 20))
            rrs = [t.get('rr', 0) for t in trades if t.get('rr', 0) > 0]
            result.avg_rr = float(np.mean(rrs)) if rrs else 0.0

    except Exception:
        import traceback; traceback.print_exc()

    return result


def backtest_from_raw(df_raw: pd.DataFrame,
                      params: NQLiquidityParams,
                      initial_capital: float = INITIAL_CAPITAL) -> NQLiquidityResult:
    return run_backtest(df_raw, params, initial_capital)


DEFAULT_PARAMS = NQLiquidityParams()
