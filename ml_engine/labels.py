"""
ML Engine — Triple Barrier Labeling
=====================================
Lopez de Prado's Triple Barrier Method adapted for Gold M1 scalping.

Labels each bar as:
  +1  — upper barrier (TP) hit first  -> BUY signal
  -1  — lower barrier (SL) hit first  -> SELL signal (or no-trade)
   0  — vertical barrier (time) hit   -> no edge / timeout

Entry price: close of label bar + spread
Barriers:
  upper: entry + tp_pts
  lower: entry - sl_pts
  vertical: entry bar + max_bars M1 bars forward

Usage:
    from ml_engine.labels import triple_barrier_labels
    labels = triple_barrier_labels(m1_df, tp_pts=5.0, sl_pts=3.0, max_bars=120)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    m1: pd.DataFrame,
    tp_pts: float = 5.0,
    sl_pts: float = 3.0,
    max_bars: int = 120,
    spread_pts: float = 0.4,
    side: str = "long",
) -> pd.Series:
    """
    Compute triple-barrier labels for every M1 bar.

    Parameters
    ----------
    m1 : pd.DataFrame
        M1 bars with at minimum 'close', 'high', 'low' columns.
    tp_pts : float
        Take-profit distance in price points.
    sl_pts : float
        Stop-loss distance in price points.
    max_bars : int
        Maximum bars to hold before forced exit (vertical barrier).
    spread_pts : float
        Half-spread added to entry price.
    side : str
        "long" only for now (model predicts long setups).

    Returns
    -------
    pd.Series
        Integer labels {-1, 0, +1} indexed like m1.
        Last max_bars rows are NaN (no future data).
    """
    m1 = m1.copy()
    m1.columns = m1.columns.str.lower()

    close = m1['close'].to_numpy(dtype=np.float64)
    high  = m1['high'].to_numpy(dtype=np.float64)
    low   = m1['low'].to_numpy(dtype=np.float64)
    n     = len(close)

    labels = np.full(n, np.nan)

    for i in range(n - max_bars):
        entry  = close[i] + spread_pts

        if side == "long":
            tp_level = entry + tp_pts
            sl_level = entry - sl_pts
        else:
            tp_level = entry - tp_pts
            sl_level = entry + sl_pts

        outcome = 0  # timeout default

        for j in range(i + 1, min(i + max_bars + 1, n)):
            if side == "long":
                if high[j] >= tp_level:
                    outcome = 1
                    break
                if low[j] <= sl_level:
                    outcome = -1
                    break
            else:
                if low[j] <= tp_level:
                    outcome = 1
                    break
                if high[j] >= sl_level:
                    outcome = -1
                    break

        labels[i] = outcome

    return pd.Series(labels, index=m1.index, name='label', dtype='float64')


def barrier_label_stats(labels: pd.Series) -> dict:
    """Print a summary of label distribution."""
    valid = labels.dropna()
    total = len(valid)
    wins     = (valid == 1).sum()
    losses   = (valid == -1).sum()
    timeouts = (valid == 0).sum()

    return {
        'total':        total,
        'win':          wins,
        'loss':         losses,
        'timeout':      timeouts,
        'win_rate':     wins / total if total else 0.0,
        'loss_rate':    losses / total if total else 0.0,
        'timeout_rate': timeouts / total if total else 0.0,
    }
