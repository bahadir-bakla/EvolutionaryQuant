"""
Gold Sniper Backtest — DEAP-Optimizable
========================================
Wraps GoldSniperStrategyCore with a vectorized backtest loop.

GENOME (9 genes):
  [0] target_atr_mult    — target points = X × ATR           (5  – 35)
  [1] stop_atr_mult      — stop points   = X × ATR           (0.5 – 3)
  [2] basket_loss_pct    — max basket loss as % of balance   (0.1 – 0.5)
  [3] max_bullets        — max open positions in basket      (2  – 12)
  [4] lot_base           — base lot size per $1000 equity    (0.05 – 0.4)
  [5] meta_bias_thresh   — |meta_bias| below this → skip     (0.0 – 0.7)
  [6] min_daily_bias     — reserved (0/1 flag encoded)       (0.0 – 1.0)
  [7] h4_sensitivity     — h4_range fraction for rejection   (0.4 – 0.8)
  [8] sweep_required     — require sweep (vs ob_tap alone)   (0.0 – 1.0)

Point value: $100 per point per standard lot (XAUUSD, 100-oz contract)
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional

_ENGINE = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_ENGINE)
sys.path.insert(0, _ROOT)

from nq_core.gold_sniper_strategy import GoldSniperStrategyCore

GOLD_PV         = 100.0    # $100/point per standard lot
SLIPPAGE_PCT    = 0.05     # 5% of ATR per side
MAX_TRADES_DAY  = 6
INITIAL_CAPITAL = 1_000.0

GENE_BOUNDS = [
    (5.0,  35.0),   # [0] target_atr_mult
    (1.0,   4.0),   # [1] stop_atr_mult
    (0.10,  0.50),  # [2] basket_loss_pct
    (2,    12),     # [3] max_bullets
    (0.005, 0.05),  # [4] lot_base ($/ $1000 equity)
    (0.0,   0.70),  # [5] meta_bias_thresh
    (0.0,   1.0),   # [6] min_daily_bias (0=no filter, 1=strict)
    (0.40,  0.80),  # [7] h4_sensitivity
    (0.0,   1.0),   # [8] sweep_required
]
GENE_NAMES = [
    'target_atr_mult', 'stop_atr_mult', 'basket_loss_pct', 'max_bullets',
    'lot_base', 'meta_bias_thresh', 'min_daily_bias', 'h4_sensitivity', 'sweep_required',
]
GENOME_SIZE = len(GENE_BOUNDS)


@dataclass
class GoldSniperParams:
    target_atr_mult:  float = 18.0
    stop_atr_mult:    float = 1.2
    basket_loss_pct:  float = 0.25
    max_bullets:      int   = 6
    lot_base:         float = 0.01
    meta_bias_thresh: float = 0.15
    min_daily_bias:   float = 0.5
    h4_sensitivity:   float = 0.60
    sweep_required:   float = 0.3


def decode_genome(genome: list) -> GoldSniperParams:
    p = GoldSniperParams()
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        raw = float(genome[i]) if i < len(genome) else 0.5
        val = lo + abs(raw % 1.0) * (hi - lo)
        setattr(p, GENE_NAMES[i], val)
    p.max_bullets = int(np.clip(round(p.max_bullets), 2, 12))
    return p


# ─────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (re-parameterized for DEAP h4_sensitivity)
# ─────────────────────────────────────────────────────────────────────

def _add_features(df: pd.DataFrame, p: GoldSniperParams) -> pd.DataFrame:
    """Build Gold Sniper features on 1H OHLCV data."""
    df = df.copy()
    df.columns = df.columns.str.lower()

    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # Daily bias (24-period rolling on 1H)
    df['d1_close_prev']  = df['close'].shift(24)
    df['d1_close_prev2'] = df['close'].shift(48)
    df['daily_bias_bullish'] = df['d1_close_prev'] > df['d1_close_prev2']
    df['daily_bias_bearish'] = df['d1_close_prev'] < df['d1_close_prev2']

    # 4H context (4 × 1H)
    df['h4_high'] = df['high'].rolling(4).max().shift(1)
    df['h4_low']  = df['low'].rolling(4).min().shift(1)
    h4_range = (df['h4_high'] - df['h4_low']).replace(0, 1e-6)
    s = p.h4_sensitivity
    df['h4_reject_down'] = (df['close'] - df['h4_low']) > (h4_range * s)
    df['h4_reject_up']   = (df['h4_high'] - df['close']) > (h4_range * s)

    # Minor liquidity sweeps (5-period)
    df['minor_high'] = df['high'].rolling(5).max().shift(1)
    df['minor_low']  = df['low'].rolling(5).min().shift(1)
    df['sweep_minor_low']  = (df['low'] < df['minor_low'])  & (df['close'] > df['minor_low'])
    df['sweep_minor_high'] = (df['high'] > df['minor_high']) & (df['close'] < df['minor_high'])

    # Order block taps
    df['is_down'] = df['close'] < df['open']
    df['is_up']   = df['close'] > df['open']
    df['ob_tap_bullish'] = (
        df['is_down'].shift(2) & df['is_up'].shift(1)
        & (df['low'] <= df['low'].shift(2))
        & (df['close'] >= df['close'].shift(1))
    )
    df['ob_tap_bearish'] = (
        df['is_up'].shift(2) & df['is_down'].shift(1)
        & (df['high'] >= df['high'].shift(2))
        & (df['close'] <= df['close'].shift(1))
    )

    if 'meta_bias' not in df.columns:
        df['meta_bias'] = 0.0

    return df


# ─────────────────────────────────────────────────────────────────────
# BACKTEST RESULT
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GoldSniperResult:
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
        if self.total_trades < 3:
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
        ddp  = max(0.1, 1 - max(0, self.max_drawdown - 0.12) * 3)
        return float(np.clip(
            (pf * 0.25 + cagr * 0.40 + sh * 0.20 + wr * 0.15) * ddp, 0, 200))


# ─────────────────────────────────────────────────────────────────────
# BACKTEST LOOP
# ─────────────────────────────────────────────────────────────────────

def run_backtest(df_feat: pd.DataFrame,
                 params: GoldSniperParams,
                 initial_capital: float = INITIAL_CAPITAL) -> GoldSniperResult:
    result = GoldSniperResult(final_equity=initial_capital)
    try:
        valid = df_feat.dropna(subset=['atr', 'daily_bias_bullish'])
        if len(valid) < 100:
            return result

        basket_loss_usd = initial_capital * params.basket_loss_pct
        bot = GoldSniperStrategyCore(
            starting_balance=initial_capital,
            max_basket_loss_usd=basket_loss_usd,
        )
        bot.balance = initial_capital

        eq_curve    = [initial_capital]
        daily_count: dict = {}

        for ts, row in valid.iterrows():
            date  = ts.date()
            hour  = ts.hour
            price = float(row['close'])
            atr   = float(row.get('atr') or price * 0.005)
            meta  = float(row.get('meta_bias', 0.0))
            slip  = atr * SLIPPAGE_PCT

            target_pts = params.target_atr_mult * atr
            stop_pts   = params.stop_atr_mult   * atr

            # Manage open basket (always — even outside session)
            bot.manage_bullets(price, ts, atr, GOLD_PV,
                                target_points=target_pts,
                                stop_points=stop_pts)

            if bot.balance < initial_capital * 0.05:
                break

            # ── ICT Killzone Filter (London+NY = 7-16 UTC) ──────────────────
            # Research: London open (7-11 UTC) + NY Silver Bullet (14-15 UTC)
            # are the highest-probability windows for XAUUSD institutional flow.
            # Asian session (17-6 UTC) is excluded — low liquidity, choppy.
            in_london_ny = (7 <= hour < 16)
            # Silver Bullet premium: 10-11 AM ET = 14-15 UTC
            silver_bullet = (14 <= hour < 15)
            # Outside session: allow position management but skip new entries
            if not in_london_ny:
                eq_curve.append(bot.balance + sum(
                    (price - t['price']) * t['direction'] * t['size'] * GOLD_PV
                    for t in bot.basket))
                continue

            # Entry logic
            n_trades = daily_count.get(date, 0)
            if n_trades < MAX_TRADES_DAY and len(bot.basket) < params.max_bullets:
                lot = min(1.0, max(0.01,
                    round(bot.balance / 1000.0 * params.lot_base, 3)))

                # LONG: daily bullish + 4H reject down + sweep/OB
                if (bool(row.get('daily_bias_bullish')) and
                        bool(row.get('h4_reject_down')) and
                        abs(meta) < params.meta_bias_thresh and
                        meta >= 0):
                    sweep_ok = bool(row.get('sweep_minor_low'))
                    ob_ok    = bool(row.get('ob_tap_bullish'))
                    need_sweep = params.sweep_required >= 0.5
                    if (sweep_ok) or (ob_ok and not need_sweep):
                        if bot.current_direction >= 0:
                            bot.add_trade(price + slip, 1, 'SNIPE_LONG', ts, lot)
                            daily_count[date] = n_trades + 1

                # SHORT: daily bearish + 4H reject up + sweep/OB
                elif (bool(row.get('daily_bias_bearish')) and
                        bool(row.get('h4_reject_up')) and
                        abs(meta) < params.meta_bias_thresh and
                        meta <= 0):
                    sweep_ok = bool(row.get('sweep_minor_high'))
                    ob_ok    = bool(row.get('ob_tap_bearish'))
                    need_sweep = params.sweep_required >= 0.5
                    if (sweep_ok) or (ob_ok and not need_sweep):
                        if bot.current_direction <= 0:
                            bot.add_trade(price - slip, -1, 'SNIPE_SHORT', ts, lot)
                            daily_count[date] = n_trades + 1

            # Equity tracking
            open_pnl = sum(
                (price - t['price']) * t['direction'] * t['size'] * GOLD_PV
                for t in bot.basket
            )
            eq_curve.append(bot.balance + open_pnl)

        # Force-close remaining basket at end
        if bot.basket:
            last_price = float(valid['close'].iloc[-1])
            for t in bot.basket:
                pnl = (last_price - t['price']) * t['direction'] * t['size'] * GOLD_PV
                bot.balance += pnl
                bot.trade_log.append({
                    'entry_time': t['time'], 'exit_time': valid.index[-1],
                    'direction': 'LONG' if t['direction'] == 1 else 'SHORT',
                    'entry_price': t['price'], 'exit_price': last_price,
                    'pnl': pnl, 'reason': 'EOD_CLOSE',
                })
            bot.basket = []
            eq_curve[-1] = bot.balance

        # Metrics
        eq  = np.array(eq_curve, dtype=float)
        ret = np.diff(eq) / (np.abs(eq[:-1]) + 1e-10)
        trades = bot.trade_log

        result.total_return = (eq[-1] - initial_capital) / initial_capital
        result.final_equity = float(eq[-1])
        result.total_trades = len(trades)
        result.trades       = trades

        peak = np.maximum.accumulate(eq)
        result.max_drawdown = float(abs(((eq - peak) / (peak + 1e-10)).min()))

        td  = (valid.index[-1] - valid.index[0]).total_seconds()
        bpy = (len(valid) / td) * 365 * 86400 if td > 0 else 8760
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

    except Exception as exc:
        import traceback; traceback.print_exc()

    return result


def backtest_from_raw(df_raw: pd.DataFrame,
                      params: GoldSniperParams,
                      initial_capital: float = INITIAL_CAPITAL) -> GoldSniperResult:
    """Feature-engineer then backtest."""
    try:
        df_feat = _add_features(df_raw, params)
        return run_backtest(df_feat, params, initial_capital)
    except Exception:
        return GoldSniperResult(final_equity=initial_capital)


DEFAULT_PARAMS = GoldSniperParams()
