"""
NQ Alpha Backtest Engine — EvolutionaryQuant
=============================================
Vectorized single-pass backtest for NQAlphaStrategy.
- Single open position at a time
- Max 3 trades per day
- Auto-detects NQ futures ($20/pt) vs QQQ/ETF ($1/share)
- ATR-based SL / TP + slippage
- Session filter: 09:30–16:00 EST only
- Returns BacktestResult dataclass
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List
from nq_alpha_strategy import (
    NQAlphaParams, add_nq_alpha_features, score_bar, session_filter
)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
MAX_TRADES_DAY          = 3
INITIAL_CAPITAL_DEFAULT = 1_000.0
SLIPPAGE_ATR_RATIO      = 0.10   # slippage = 10% of ATR per side


def _detect_instrument(df: pd.DataFrame):
    """Detect if data is NQ futures (~18000-22000) or QQQ/ETF (~260-640).
    Returns (point_value, is_futures).
    """
    med_price = float(df['close'].median())
    if med_price > 5000:
        return 20.0, True    # NQ futures: $20/point per micro-contract
    else:
        return 1.0,  False   # ETF/stock: $1/point per share


@dataclass
class BacktestResult:
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
        """Composite fitness for DEAP. Range: -999 … 200."""
        if self.total_trades < 1:
            return -999.0
        if self.total_return <= 0:
            return float(np.clip(self.total_return * 5, -50, -0.01))
        if self.profit_factor < 1.0:
            return float(np.clip((self.profit_factor - 1) * 10, -20, -0.01))
        if self.max_drawdown > 0.75:
            return -50.0

        pf   = float(np.clip(self.profit_factor, 1.0, 15))
        cagr = float(np.clip(self.cagr, 0, 20))
        sh   = float(np.clip(self.sharpe_ratio, 0, 10))
        wr   = float(np.clip(self.win_rate, 0, 1))
        rr   = float(np.clip(self.avg_rr, 0, 10))
        ddp  = max(0.1, 1 - max(0, self.max_drawdown - 0.15) * 2.5)

        score = (pf * 0.25 + cagr * 0.35 + sh * 0.20 + wr * 0.10 + rr * 0.10) * ddp
        return float(np.clip(score, 0, 200))


def _calc_size(balance: float, sl_pts: float, pv: float,
               risk_pct: float = 0.015) -> float:
    """Risk-based position sizing: risk ~1.5% of balance per trade."""
    risk_usd = balance * risk_pct
    sl_usd   = sl_pts * pv
    if sl_usd < 1e-6:
        sl_usd = 1.0
    raw = risk_usd / sl_usd
    if pv >= 10:    # NQ futures: contracts (0.01 – 0.10)
        return float(np.clip(round(raw, 3), 0.01, 0.10))
    else:            # QQQ shares (1 – max affordable)
        max_shares = balance / (sl_pts * 10 + 1e-6)   # safety cap
        return float(np.clip(round(raw, 1), 1.0, max(1.0, max_shares)))


def run_backtest(df_feat: pd.DataFrame,
                 params: NQAlphaParams,
                 initial_capital: float = INITIAL_CAPITAL_DEFAULT) -> BacktestResult:
    """
    Single-pass backtest on pre-featured dataframe.
    df_feat: output of add_nq_alpha_features()
    """
    result = BacktestResult(final_equity=initial_capital)
    try:
        pv, is_futures = _detect_instrument(df_feat)

        warmup_cols = ['atr']   # fixed: only atr is guaranteed by add_nq_alpha_features
        valid = df_feat.dropna(subset=[c for c in warmup_cols if c in df_feat.columns])
        if len(valid) < 50:
            return result

        balance     = initial_capital
        eq_curve    = [balance]
        trades      = []
        open_pos    = None
        daily_count = {}

        for ts, row in valid.iterrows():
            date  = ts.date()
            price = float(row['close'])
            atr   = float(row.get('atr') or price * 0.002)
            slip  = atr * SLIPPAGE_ATR_RATIO

            # ── Close position ─────────────────────────────────────
            if open_pos is not None:
                d   = open_pos['dir']
                hi  = float(row['high'])
                lo  = float(row['low'])
                sl  = open_pos['sl']
                tp  = open_pos['tp']
                s   = open_pos['size']
                closed = False; exit_p = price

                if d == 1:
                    if lo <= sl:   exit_p = sl;  closed = True
                    elif hi >= tp: exit_p = tp;  closed = True
                else:
                    if hi >= sl:   exit_p = sl;  closed = True
                    elif lo <= tp: exit_p = tp;  closed = True

                if closed:
                    pnl = (exit_p - open_pos['entry']) * d * s * pv - slip * s * pv
                    balance += pnl
                    rr_val  = abs(tp - open_pos['entry']) / max(abs(sl - open_pos['entry']), 1e-6)
                    trades.append({'pnl': pnl, 'rr': rr_val,
                                   'entry_time': open_pos['ts'], 'exit_time': ts})
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
                        size   = _calc_size(balance, sl_pts, pv)
                        open_pos = {'dir': d, 'entry': entry, 'sl': sl,
                                    'tp': tp, 'size': size, 'ts': ts}
                        daily_count[date] = daily_count.get(date, 0) + 1

            # ── Mark-to-market ────────────────────────────────────
            if open_pos is not None:
                live = (price - open_pos['entry']) * open_pos['dir'] * open_pos['size'] * pv
                eq_curve.append(balance + live)
            else:
                eq_curve.append(balance)

            if balance < initial_capital * 0.05:   # ruin stop
                break

        # ── Close remaining ────────────────────────────────────────
        if open_pos is not None:
            fp   = float(valid['close'].iloc[-1])
            atrl = float(valid['atr'].iloc[-1] or fp * 0.002)
            pnl  = (fp - open_pos['entry']) * open_pos['dir'] * open_pos['size'] * pv \
                   - atrl * SLIPPAGE_ATR_RATIO * open_pos['size'] * pv
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

        # CAGR
        n_bars = len(valid)
        td     = (valid.index[-1] - valid.index[0]).total_seconds()
        bpy    = (n_bars / td) * 365 * 86400 if td > 0 else 19656
        ny     = n_bars / max(bpy, 1)
        if ny > 0 and eq[-1] > 0:
            result.cagr = float(np.clip((eq[-1]/initial_capital)**(1/max(ny,0.1))-1, -1, 50))

        # Sharpe
        if len(ret) > 1 and ret.std() > 1e-10:
            result.sharpe_ratio = float(np.clip(
                ret.mean() / ret.std() * np.sqrt(bpy), -10, 10))

        # Win rate + PF
        if trades:
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            loss = [p for p in pnls if p <= 0]
            result.win_rate      = len(wins) / len(trades)
            gp = sum(wins); gl = abs(sum(loss)) if loss else 1e-10
            result.profit_factor = float(np.clip(gp / (gl + 1e-10), 0, 20))
            rrs = [t.get('rr', 0) for t in trades if t.get('rr', 0) > 0]
            result.avg_rr        = float(np.mean(rrs)) if rrs else 0.0

    except Exception:
        pass

    return result


def backtest_from_raw(df_raw: pd.DataFrame,
                      params: NQAlphaParams,
                      initial_capital: float = INITIAL_CAPITAL_DEFAULT) -> BacktestResult:
    """Convenience: feature-engineer then backtest."""
    try:
        df_feat = add_nq_alpha_features(df_raw, params)
        return run_backtest(df_feat, params, initial_capital)
    except Exception:
        return BacktestResult(final_equity=initial_capital)
