"""
ML Engine — Bidirectional ML Scalper Backtest
===============================================
Uses two LightGBM models (long + short) to generate trade signals on M1 bars.

Signal logic:
  - long_proba  >= threshold -> LONG  entry at next bar open + spread
  - short_proba >= threshold -> SHORT entry at next bar open - spread
  - If both fire same bar -> take higher proba, or skip (conflict_skip=True)
  - 1 position at a time

Trade mechanics:
  LONG : SL=entry-sl_pts  TP=entry+tp_pts  PnL=(exit-entry)*lot*PV
  SHORT: SL=entry+sl_pts  TP=entry-tp_pts  PnL=(entry-exit)*lot*PV

GOLD_PV = 100: $1 per 0.01 lot per point
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml_engine.features import build_features

warnings.filterwarnings('ignore')

GOLD_PV = 100.0


# -----------------------------------------------------------------------------
# PARAMS
# -----------------------------------------------------------------------------

@dataclass
class MLScalperParams:
    tp_pts:         float = 6.0
    sl_pts:         float = 3.0
    spread_pts:     float = 0.4
    lot_base:       float = 0.01
    max_hold_bars:  int   = 120
    threshold:      float = 0.70
    session_filter: bool  = True
    conflict_skip:  bool  = False  # if True skip bar when both long+short fire


# -----------------------------------------------------------------------------
# FEATURE BUILDER (cached across calls)
# -----------------------------------------------------------------------------

def _build_signals(
    m1: pd.DataFrame,
    long_bundle: dict,
    short_bundle: dict,
    threshold: float,
    verbose: bool,
) -> tuple[pd.Series, pd.Series]:
    """Return (long_signal, short_signal) boolean Series."""
    if verbose:
        print("  Building features...", end=' ', flush=True)
    X = build_features(m1)
    if verbose:
        print(f"shape={X.shape}")

    warmup = X.isna().any(axis=1)

    def _proba(bundle: dict) -> pd.Series:
        Xb = X.reindex(columns=bundle['features'], fill_value=0.0).fillna(0.0)
        p  = pd.Series(
            bundle['model'].predict_proba(Xb)[:, 1],
            index=m1.index,
        )
        p[warmup] = 0.0
        return p

    long_p  = _proba(long_bundle)
    short_p = _proba(short_bundle)

    long_sig  = (long_p  >= threshold)
    short_sig = (short_p >= threshold)

    if verbose:
        print(f"  Long signals : {long_sig.sum():,}  ({long_sig.mean():.2%})")
        print(f"  Short signals: {short_sig.sum():,}  ({short_sig.mean():.2%})")
        both = (long_sig & short_sig).sum()
        print(f"  Conflicts    : {both:,}")

    return long_p, short_p, long_sig, short_sig


# -----------------------------------------------------------------------------
# BACKTEST ENGINE
# -----------------------------------------------------------------------------

def backtest_ml(
    m1: pd.DataFrame,
    long_bundle: dict,
    short_bundle: dict | None = None,
    params: MLScalperParams | None = None,
    initial_capital: float = 1_000.0,
    verbose: bool = True,
) -> dict:
    """
    Run bidirectional ML scalper backtest.

    If short_bundle is None, only LONG trades are taken (backward compat).
    """
    if params is None:
        params = MLScalperParams()

    threshold = params.threshold

    m1 = m1.copy()
    m1.columns = m1.columns.str.lower()

    # Use long_bundle as fallback if no short_bundle
    if short_bundle is None:
        short_bundle = {'model': None, 'features': long_bundle['features']}

    long_p, short_p, long_sig, short_sig = _build_signals(
        m1, long_bundle, short_bundle if short_bundle['model'] else long_bundle,
        threshold, verbose
    )
    # Disable short if no short model
    if short_bundle['model'] is None:
        short_sig[:] = False

    o_arr = m1['open'].to_numpy(dtype=np.float64)
    h_arr = m1['high'].to_numpy(dtype=np.float64)
    l_arr = m1['low'].to_numpy(dtype=np.float64)
    c_arr = m1['close'].to_numpy(dtype=np.float64)
    ls    = long_sig.to_numpy()
    ss    = short_sig.to_numpy()
    lp    = long_p.to_numpy()
    sp    = short_p.to_numpy()
    idx   = m1.index

    equity   = initial_capital
    trades   = []
    open_pos = None

    for i in range(1, len(m1)):
        # Session filter
        if params.session_filter:
            h = idx[i].hour
            in_session = (7 <= h < 11) or (13 <= h < 17)
            if not in_session and open_pos is None:
                continue

        # Manage open position
        if open_pos is not None:
            bar_hi = h_arr[i]
            bar_lo = l_arr[i]
            held   = i - open_pos['bar_i']
            exit_price  = None
            exit_reason = None
            direction   = open_pos['dir']  # +1 long, -1 short

            if direction == 1:   # LONG
                if bar_hi >= open_pos['tp']:
                    exit_price, exit_reason = open_pos['tp'], 'TP'
                elif bar_lo <= open_pos['sl']:
                    exit_price, exit_reason = open_pos['sl'], 'SL'
                elif held >= params.max_hold_bars:
                    exit_price, exit_reason = c_arr[i], 'TIME'
            else:                # SHORT
                if bar_lo <= open_pos['tp']:
                    exit_price, exit_reason = open_pos['tp'], 'TP'
                elif bar_hi >= open_pos['sl']:
                    exit_price, exit_reason = open_pos['sl'], 'SL'
                elif held >= params.max_hold_bars:
                    exit_price, exit_reason = c_arr[i], 'TIME'

            if exit_price is not None:
                raw_pnl = (exit_price - open_pos['entry']) * direction
                pnl     = raw_pnl * open_pos['lot'] * GOLD_PV
                equity += pnl
                trades.append({
                    'entry_ts':  open_pos['ts'],
                    'exit_ts':   idx[i],
                    'dir':       'LONG' if direction == 1 else 'SHORT',
                    'entry':     open_pos['entry'],
                    'exit':      exit_price,
                    'lot':       open_pos['lot'],
                    'pnl':       round(pnl, 4),
                    'reason':    exit_reason,
                    'hold_bars': held,
                    'proba':     open_pos['proba'],
                })
                open_pos = None

        # Check for new signal (use prev-bar signal → enter this bar's open)
        if open_pos is None:
            want_long  = ls[i - 1]
            want_short = ss[i - 1]

            # Conflict resolution
            if want_long and want_short:
                if params.conflict_skip:
                    want_long = want_short = False
                elif lp[i - 1] >= sp[i - 1]:
                    want_short = False
                else:
                    want_long = False

            if want_long:
                entry = o_arr[i] + params.spread_pts
                open_pos = {
                    'dir': 1, 'entry': entry,
                    'tp':  entry + params.tp_pts,
                    'sl':  entry - params.sl_pts,
                    'lot': params.lot_base, 'ts': idx[i],
                    'bar_i': i, 'proba': float(lp[i - 1]),
                }
            elif want_short:
                entry = o_arr[i] - params.spread_pts
                open_pos = {
                    'dir': -1, 'entry': entry,
                    'tp':  entry - params.tp_pts,
                    'sl':  entry + params.sl_pts,
                    'lot': params.lot_base, 'ts': idx[i],
                    'bar_i': i, 'proba': float(sp[i - 1]),
                }

    # Close at end
    if open_pos is not None:
        exit_price = c_arr[-1]
        direction  = open_pos['dir']
        raw_pnl    = (exit_price - open_pos['entry']) * direction
        pnl        = raw_pnl * open_pos['lot'] * GOLD_PV
        equity += pnl
        trades.append({
            'entry_ts':  open_pos['ts'],
            'exit_ts':   idx[-1],
            'dir':       'LONG' if direction == 1 else 'SHORT',
            'entry':     open_pos['entry'],
            'exit':      exit_price,
            'lot':       open_pos['lot'],
            'pnl':       round(pnl, 4),
            'reason':    'END',
            'hold_bars': len(m1) - 1 - open_pos['bar_i'],
            'proba':     open_pos['proba'],
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {'trades': trades_df, 'equity': pd.Series([initial_capital]), 'metrics': {}}

    # Equity curve
    equity_curve = pd.Series(initial_capital, index=[m1.index[0]])
    cumulative   = trades_df.set_index('exit_ts')['pnl'].cumsum() + initial_capital
    equity_curve = pd.concat([equity_curve, cumulative]).sort_index()

    metrics = _compute_metrics(trades_df, initial_capital, equity_curve)

    if verbose:
        _print_metrics(metrics)

    return {'trades': trades_df, 'equity': equity_curve, 'metrics': metrics}


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------

def _compute_metrics(trades: pd.DataFrame, initial_capital: float,
                     equity: pd.Series) -> dict:
    total  = len(trades)
    wins   = (trades['pnl'] > 0).sum()

    gross_profit = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
    gross_loss   = abs(trades.loc[trades['pnl'] <= 0, 'pnl'].sum())

    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    wr = wins / total if total > 0 else 0.0

    final_equity  = equity.iloc[-1]
    total_return  = (final_equity - initial_capital) / initial_capital * 100

    peak   = equity.cummax()
    dd     = (equity - peak) / peak * 100
    max_dd = dd.min()

    avg_hold = trades['hold_bars'].mean()
    duration = (trades['exit_ts'].max() - trades['entry_ts'].min()).days
    tpd      = total / max(duration, 1)

    by_dir    = trades.groupby('dir')['pnl'].agg(['count', 'sum', 'mean'])
    by_reason = trades.groupby('reason')['pnl'].agg(['count', 'sum', 'mean'])

    return {
        'total_trades':    total,
        'win_rate':        wr,
        'profit_factor':   pf,
        'total_return_pct': total_return,
        'max_dd_pct':      max_dd,
        'avg_hold_bars':   avg_hold,
        'trades_per_day':  tpd,
        'final_equity':    final_equity,
        'by_dir':          by_dir,
        'by_reason':       by_reason,
    }


def _print_metrics(m: dict) -> None:
    print("\n" + "-" * 52)
    print(f"  Trades        : {m['total_trades']:>6}")
    print(f"  Win Rate      : {m['win_rate']:>6.1%}")
    print(f"  Profit Factor : {m['profit_factor']:>6.3f}")
    print(f"  Total Return  : {m['total_return_pct']:>6.1f}%")
    print(f"  Max DD        : {m['max_dd_pct']:>6.1f}%")
    print(f"  Avg Hold      : {m['avg_hold_bars']:>5.0f} min")
    print(f"  Trades/Day    : {m['trades_per_day']:>6.2f}")
    print(f"  Final Equity  : ${m['final_equity']:>8.2f}")
    print("-" * 52)
    if not m['by_dir'].empty:
        print("\n  Direction Breakdown:")
        print(m['by_dir'].to_string())
    if not m['by_reason'].empty:
        print("\n  Exit Breakdown:")
        print(m['by_reason'].to_string())
    print()
