"""
Silver Momentum Backtest Engine — EvolutionaryQuant
====================================================
Optimized single-pass backtest for XAGUSD.
- Kelly + GARCH position sizing (optional)
- Session filter: London + NY
- Single position at a time
- Daily trade limit
- XAG point value: $50/oz per standard lot
"""

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

_DIR    = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
sys.path.insert(0, _DIR)
sys.path.insert(0, _PARENT)

from silver_strategy import SilverMomentumParams, add_silver_features, score_bar, session_filter

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
XAG_PV          = 50.0    # $50 per oz per standard lot (XAGUSD)
MAX_TRADES_DAY  = 4
INITIAL_CAPITAL = 1_000.0
SLIPPAGE_PCT    = 0.08    # 8% of ATR per side


@dataclass
class SilverBacktestResult:
    total_return:  float = 0.0
    cagr:          float = 0.0
    max_drawdown:  float = 0.0
    sharpe_ratio:  float = 0.0
    win_rate:      float = 0.0
    profit_factor: float = 0.0
    total_trades:  int   = 0
    final_equity:  float = 0.0
    avg_rr:        float = 0.0
    trades:        List  = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []

    def fitness(self) -> float:
        if self.total_trades < 4:
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
        ddp  = max(0.1, 1 - max(0, self.max_drawdown - 0.12) * 3)
        return float(np.clip(
            (pf * 0.25 + cagr * 0.35 + sh * 0.20 + wr * 0.10 + rr * 0.10) * ddp,
            0, 200))


def _calc_size(balance: float, sl_pts: float,
               pv: float = XAG_PV,
               risk_pct: float = 0.015,
               min_lot: float = 0.01,
               max_lot: float = 1.0) -> float:
    """ATR-based risk sizing (fallback when Kelly not available)."""
    risk_usd = balance * risk_pct
    sl_usd   = sl_pts * pv
    if sl_usd < 1e-6:
        return min_lot
    raw = risk_usd / sl_usd
    return float(np.clip(round(raw, 3), min_lot, max_lot))


def run_backtest(df_feat: pd.DataFrame,
                 params: SilverMomentumParams,
                 initial_capital: float = INITIAL_CAPITAL,
                 use_kelly_garch: bool = False) -> SilverBacktestResult:
    """
    Single-pass XAGUSD backtest.

    use_kelly_garch: If True, uses Kelly+GARCH sizing (requires kelly_garch_sizer)
    """
    result = SilverBacktestResult(final_equity=initial_capital)
    try:
        warmup = ['atr', 'roc_fast', 'ema_f', 'macd', 'adx']
        valid  = df_feat.dropna(subset=[c for c in warmup if c in df_feat.columns])
        if len(valid) < 50:
            return result

        # Kelly+GARCH sizer setup
        sizer = None
        if use_kelly_garch:
            try:
                sys.path.insert(0, os.path.join(_PARENT, 'kelly_garch_sizer'))
                from position_sizer import KellyGARCHSizer, SizerConfig
                cfg   = SizerConfig(min_lot=0.01, max_lot=1.0)
                sizer = KellyGARCHSizer(cfg)
            except ImportError:
                use_kelly_garch = False

        balance     = initial_capital
        eq_curve    = [balance]
        trades      = []
        open_pos    = None
        daily_count = {}

        for ts, row in valid.iterrows():
            date  = ts.date()
            price = float(row['close'])
            atr   = float(row.get('atr') or price * 0.005)
            slip  = atr * SLIPPAGE_PCT

            # ── Close position ─────────────────────────────────────
            if open_pos is not None:
                d  = open_pos['dir']
                hi = float(row['high'])
                lo = float(row['low'])
                sl = open_pos['sl']
                tp = open_pos['tp']
                s  = open_pos['size']
                closed = False; exit_p = price

                if d == 1:
                    if lo <= sl:   exit_p = sl;  closed = True
                    elif hi >= tp: exit_p = tp;  closed = True
                else:
                    if hi >= sl:   exit_p = sl;  closed = True
                    elif lo <= tp: exit_p = tp;  closed = True

                if closed:
                    pnl = (exit_p - open_pos['entry']) * d * s * XAG_PV - slip * s * XAG_PV
                    balance += pnl
                    rr  = abs(tp - open_pos['entry']) / max(abs(sl - open_pos['entry']), 1e-6)
                    trades.append({'pnl': pnl, 'rr': rr,
                                   'entry_time': open_pos['ts'], 'exit_time': ts,
                                   'size': s, 'dir': d})
                    open_pos = None

            # ── Entry ─────────────────────────────────────────────
            if open_pos is None and session_filter(ts):
                if daily_count.get(date, 0) < MAX_TRADES_DAY:
                    net = score_bar(row, params)
                    if abs(net) >= params.min_score:
                        d      = 1 if net > 0 else -1
                        sl_pts = params.sl_atr_mult * atr
                        tp_pts = params.tp_atr_mult * atr
                        entry  = price + d * slip
                        sl     = entry - d * sl_pts
                        tp     = entry + d * tp_pts

                        if use_kelly_garch and sizer and len(valid) >= 20:
                            size = sizer.compute_lot(
                                capital     = balance,
                                sl_pts      = sl_pts,
                                point_value = XAG_PV,
                                prices      = valid['close'].loc[:ts],
                                trades      = trades,
                            )
                        else:
                            size = _calc_size(balance, sl_pts)

                        open_pos = {'dir': d, 'entry': entry, 'sl': sl,
                                    'tp': tp, 'size': size, 'ts': ts}
                        daily_count[date] = daily_count.get(date, 0) + 1

            # ── Mark-to-market ────────────────────────────────────
            if open_pos is not None:
                live = (price - open_pos['entry']) * open_pos['dir'] * open_pos['size'] * XAG_PV
                eq_curve.append(balance + live)
            else:
                eq_curve.append(balance)

            if balance < initial_capital * 0.05:
                break

        # ── Close remaining ────────────────────────────────────────
        if open_pos is not None:
            fp   = float(valid['close'].iloc[-1])
            atrl = float(valid['atr'].iloc[-1] or fp * 0.005)
            pnl  = (fp - open_pos['entry']) * open_pos['dir'] * open_pos['size'] * XAG_PV \
                   - atrl * SLIPPAGE_PCT * open_pos['size'] * XAG_PV
            balance += pnl
            trades.append({'pnl': pnl, 'rr': 0.0})
            eq_curve[-1] = balance

        # ── Metrics ───────────────────────────────────────────────
        eq  = np.array(eq_curve, dtype=float)
        ret = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)

        result.total_return = (eq[-1] - initial_capital) / initial_capital
        result.final_equity = float(eq[-1])
        result.total_trades = len(trades)
        result.trades       = trades

        peak   = np.maximum.accumulate(eq)
        result.max_drawdown = float(abs(((eq - peak) / (peak + 1e-10)).min()))

        n_bars = len(valid)
        td     = (valid.index[-1] - valid.index[0]).total_seconds()
        bpy    = (n_bars / td) * 365 * 86400 if td > 0 else 78000  # ~5min silver bars/year
        ny     = n_bars / max(bpy, 1)
        if ny > 0 and eq[-1] > 0:
            result.cagr = float(np.clip((eq[-1]/initial_capital)**(1/max(ny,0.1))-1, -1, 50))

        if len(ret) > 1 and ret.std() > 1e-10:
            result.sharpe_ratio = float(np.clip(ret.mean()/ret.std()*np.sqrt(bpy), -10, 10))

        if trades:
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            loss = [p for p in pnls if p <= 0]
            result.win_rate      = len(wins) / len(trades)
            gp = sum(wins); gl = abs(sum(loss)) if loss else 1e-10
            result.profit_factor = float(np.clip(gp/(gl+1e-10), 0, 20))
            rrs = [t.get('rr',0) for t in trades if t.get('rr',0) > 0]
            result.avg_rr        = float(np.mean(rrs)) if rrs else 0.0

    except Exception:
        pass
    return result


def backtest_from_raw(df_raw: pd.DataFrame,
                      params: SilverMomentumParams,
                      initial_capital: float = INITIAL_CAPITAL,
                      use_kelly_garch: bool = False) -> SilverBacktestResult:
    """Convenience: feature-engineer then backtest."""
    try:
        df_feat = add_silver_features(df_raw, params)
        return run_backtest(df_feat, params, initial_capital, use_kelly_garch)
    except Exception:
        return SilverBacktestResult(final_equity=initial_capital)
