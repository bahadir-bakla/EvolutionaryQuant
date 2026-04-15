"""
ML Engine — ML-Driven Scalper Backtest
========================================
Uses trained LightGBM model to generate trade signals on M1 bars.

Signal logic:
  - At each M1 bar, compute features and get predict_proba
  - If proba >= threshold and no open position: enter LONG at next bar open + spread
  - SL: entry - sl_pts
  - TP: entry + tp_pts
  - Max hold: max_hold_bars (forced exit)
  - Session filter: optional — only trade London/NY overlap

Trade mechanics:
  - 1 position at a time
  - lot size = lot_base (e.g. 0.01)
  - P&L = (exit - entry) * lot * GOLD_PV
  - GOLD_PV = 100 (1 lot = 100 oz, 1pt = $1 per 0.01 lot)

Output: Trades DataFrame + equity curve + metrics
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml_engine.features import build_features

warnings.filterwarnings('ignore')

GOLD_PV = 100.0  # point value: $1 per 0.01 lot per point


# -----------------------------------------------------------------------------
# PARAMS
# -----------------------------------------------------------------------------

@dataclass
class MLScalperParams:
    tp_pts:       float = 5.0
    sl_pts:       float = 3.0
    spread_pts:   float = 0.4
    lot_base:     float = 0.01
    max_hold_bars: int  = 120
    threshold:    float = 0.55
    session_filter: bool = True   # only London + NY


# -----------------------------------------------------------------------------
# BACKTEST ENGINE
# -----------------------------------------------------------------------------

def backtest_ml(
    m1: pd.DataFrame,
    model_bundle: dict,
    params: MLScalperParams | None = None,
    initial_capital: float = 1_000.0,
    verbose: bool = True,
) -> dict:
    """
    Run ML-driven scalper backtest on M1 data.

    Parameters
    ----------
    m1 : pd.DataFrame
        M1 OHLCV bars.
    model_bundle : dict
        Output of load_model() — keys: 'model', 'features', 'threshold'
    params : MLScalperParams, optional
    initial_capital : float

    Returns
    -------
    dict with keys: trades, equity, metrics
    """
    if params is None:
        params = MLScalperParams()

    # Override threshold from bundle if not overridden
    threshold = params.threshold or model_bundle.get('threshold', 0.55)

    m1 = m1.copy()
    m1.columns = m1.columns.str.lower()

    # -- Build features for entire dataset -----------------------------
    if verbose:
        print("  Building features for backtest...", end=' ', flush=True)
    X = build_features(m1)
    X = X.reindex(columns=model_bundle['features'], fill_value=0.0)
    if verbose:
        print(f"shape={X.shape}")

    # -- Generate signals -----------------------------------------------
    if verbose:
        print("  Computing ML signals...", end=' ', flush=True)
    model = model_bundle['model']

    # Fill NaN with 0 for prediction (NaN only in warmup rows)
    X_filled = X.fillna(0.0)
    proba = pd.Series(
        model.predict_proba(X_filled)[:, 1],
        index=m1.index,
        name='proba',
    )
    # Don't trade during NaN warmup period (first ~50 M15 bars = 750 M1 bars)
    warmup_mask = X.isna().any(axis=1)
    proba[warmup_mask] = 0.0

    signals = (proba >= threshold).astype(int)
    if verbose:
        print(f"total signals={signals.sum():,}  ({signals.mean():.2%} of bars)")

    # -- Simulate trades ------------------------------------------------
    o_arr = m1['open'].to_numpy(dtype=np.float64)
    h_arr = m1['high'].to_numpy(dtype=np.float64)
    l_arr = m1['low'].to_numpy(dtype=np.float64)
    c_arr = m1['close'].to_numpy(dtype=np.float64)
    sig   = signals.to_numpy()
    idx   = m1.index

    equity   = initial_capital
    trades   = []
    open_pos = None  # {'entry', 'tp', 'sl', 'lot', 'ts', 'hold'}

    for i in range(1, len(m1)):
        # -- Session filter -------------------------------------------
        if params.session_filter:
            h = idx[i].hour
            in_session = (7 <= h < 11) or (13 <= h < 17)
            if not in_session and open_pos is None:
                continue

        # -- Manage open position ------------------------------------
        if open_pos is not None:
            bar_hi = h_arr[i]
            bar_lo = l_arr[i]
            held   = i - open_pos['bar_i']
            exit_price = None
            exit_reason = None

            # Check TP first (more favourable)
            if bar_hi >= open_pos['tp']:
                exit_price  = open_pos['tp']
                exit_reason = 'TP'
            elif bar_lo <= open_pos['sl']:
                exit_price  = open_pos['sl']
                exit_reason = 'SL'
            elif held >= params.max_hold_bars:
                exit_price  = c_arr[i]
                exit_reason = 'TIME'

            if exit_price is not None:
                pnl = (exit_price - open_pos['entry']) * open_pos['lot'] * GOLD_PV
                equity += pnl
                trades.append({
                    'entry_ts':  open_pos['ts'],
                    'exit_ts':   idx[i],
                    'entry':     open_pos['entry'],
                    'exit':      exit_price,
                    'lot':       open_pos['lot'],
                    'pnl':       pnl,
                    'reason':    exit_reason,
                    'hold_bars': held,
                    'proba':     open_pos['proba'],
                })
                open_pos = None

        # -- Check for new signal -------------------------------------
        if open_pos is None and sig[i - 1] == 1:
            # Enter on current bar open + spread
            entry = o_arr[i] + params.spread_pts
            open_pos = {
                'entry':  entry,
                'tp':     entry + params.tp_pts,
                'sl':     entry - params.sl_pts,
                'lot':    params.lot_base,
                'ts':     idx[i],
                'bar_i':  i,
                'proba':  float(proba.iloc[i - 1]),
            }

    # Close any open position at end
    if open_pos is not None:
        exit_price = c_arr[-1]
        pnl = (exit_price - open_pos['entry']) * open_pos['lot'] * GOLD_PV
        equity += pnl
        trades.append({
            'entry_ts':  open_pos['ts'],
            'exit_ts':   idx[-1],
            'entry':     open_pos['entry'],
            'exit':      exit_price,
            'lot':       open_pos['lot'],
            'pnl':       pnl,
            'reason':    'END',
            'hold_bars': len(m1) - 1 - open_pos['bar_i'],
            'proba':     open_pos['proba'],
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {'trades': trades_df, 'equity': pd.Series([initial_capital]), 'metrics': {}}

    # -- Equity curve ---------------------------------------------------
    equity_curve = pd.Series(initial_capital, index=[m1.index[0]])
    cumulative_pnl = trades_df.set_index('exit_ts')['pnl'].cumsum() + initial_capital
    equity_curve = pd.concat([equity_curve, cumulative_pnl])
    equity_curve.sort_index(inplace=True)

    # -- Metrics --------------------------------------------------------
    metrics = _compute_metrics(trades_df, initial_capital, equity_curve)

    if verbose:
        _print_metrics(metrics)

    return {'trades': trades_df, 'equity': equity_curve, 'metrics': metrics}


def _compute_metrics(trades: pd.DataFrame, initial_capital: float,
                     equity: pd.Series) -> dict:
    total = len(trades)
    wins  = (trades['pnl'] > 0).sum()
    losses = (trades['pnl'] <= 0).sum()

    gross_profit = trades.loc[trades['pnl'] > 0, 'pnl'].sum()
    gross_loss   = abs(trades.loc[trades['pnl'] <= 0, 'pnl'].sum())

    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    wr = wins / total if total > 0 else 0.0

    final_equity = equity.iloc[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100

    # Max drawdown
    peak = equity.cummax()
    dd   = (equity - peak) / peak * 100
    max_dd = dd.min()

    # Avg hold
    avg_hold = trades['hold_bars'].mean()

    # Trades per day
    duration_days = (trades['exit_ts'].max() - trades['entry_ts'].min()).days
    tpd = total / max(duration_days, 1)

    # Reason breakdown
    by_reason = trades.groupby('reason')['pnl'].agg(['count', 'sum', 'mean'])

    return {
        'total_trades': total,
        'win_rate':     wr,
        'profit_factor': pf,
        'total_return_pct': total_return,
        'max_dd_pct':   max_dd,
        'avg_hold_bars': avg_hold,
        'trades_per_day': tpd,
        'final_equity':  final_equity,
        'by_reason':     by_reason,
    }


def _print_metrics(m: dict) -> None:
    print("\n" + "-" * 50)
    print(f"  Trades        : {m['total_trades']:>6}")
    print(f"  Win Rate      : {m['win_rate']:>6.1%}")
    print(f"  Profit Factor : {m['profit_factor']:>6.3f}")
    print(f"  Total Return  : {m['total_return_pct']:>6.1f}%")
    print(f"  Max DD        : {m['max_dd_pct']:>6.1f}%")
    print(f"  Avg Hold (bars): {m['avg_hold_bars']:>5.0f} min")
    print(f"  Trades/Day    : {m['trades_per_day']:>6.2f}")
    print(f"  Final Equity  : ${m['final_equity']:>8.2f}")
    print("-" * 50)
    if not m['by_reason'].empty:
        print("\n  Exit Breakdown:")
        print(m['by_reason'].to_string())
    print()
