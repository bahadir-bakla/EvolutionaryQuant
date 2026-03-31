"""
BACKTEST ENJİNİ v4 — Vectorized + Precompute Cache
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List
import importlib

regime_mod   = importlib.import_module("01_regime_detector")
position_mod = importlib.import_module("02_position_manager")

DataPrecomputer    = regime_mod.DataPrecomputer
InstitutionalKelly = position_mod.InstitutionalKelly
BasketManager      = position_mod.BasketManager


@dataclass
class BacktestResult:
    total_return:  float = 0.0
    cagr:          float = 0.0
    sharpe_ratio:  float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown:  float = 0.0
    win_rate:      float = 0.0
    profit_factor: float = 0.0
    total_trades:  int   = 0
    calmar_ratio:  float = 0.0
    final_equity:  float = 0.0

    def fitness(self) -> float:
        if self.total_trades < 5:
            return -999.0
        if self.max_drawdown > 0.60:
            return -50.0
        if self.total_return <= 0:
            return float(np.clip(self.total_return * 5, -50, -0.01))

        pf     = float(np.clip(self.profit_factor, 0, 10))
        wr     = float(np.clip(self.win_rate, 0, 1))
        cagr   = float(np.clip(self.cagr, 0, 5))
        sharpe = float(np.clip(self.sharpe_ratio, -5, 10))

        score = pf * 0.35 + wr * 0.25 + cagr * 0.25 + sharpe * 0.15

        if self.max_drawdown > 0.20:
            score *= max(0.1, 1 - (self.max_drawdown - 0.20) * 2)

        return float(np.clip(score, 0.01, 20.0))

    def summary(self):
        return (f"Return:{self.total_return*100:.1f}% | "
                f"CAGR:{self.cagr*100:.1f}% | "
                f"DD:{self.max_drawdown*100:.1f}% | "
                f"WR:{self.win_rate*100:.0f}% | "
                f"Trades:{self.total_trades} | "
                f"PF:{self.profit_factor:.2f}")


class InstitutionalBacktester:

    def __init__(
        self,
        initial_capital:    float = 1_000.0,
        commission:         float = 0.0001,
        slippage:           float = 0.0002,
        hurst_window:       int   = 100,
        chop_period:        int   = 14,
        ob_lookback:        int   = 10,
        resistance_lookback:int   = 50,
        score_threshold:    float = 3.0,
        max_layers:         int   = 3,
        dca_step_pct:       float = 0.008,
        target_profit_pct:  float = 0.015,
        stop_loss_pct:      float = 0.025,
        position_pct:       float = 0.15,
        use_session_filter: bool  = True,
    ):
        self.initial_capital    = initial_capital
        self.commission         = commission
        self.slippage           = slippage
        self.position_pct       = position_pct
        self.use_session_filter = use_session_filter
        self.score_threshold    = score_threshold
        self.hurst_window       = hurst_window
        self.chop_period        = chop_period
        self.ob_lookback        = ob_lookback
        self.basket_cfg = {
            'max_layers':        max_layers,
            'dca_step_pct':      dca_step_pct,
            'target_profit_pct': target_profit_pct,
            'stop_loss_pct':     stop_loss_pct,
        }

    def _session_ok(self, bar_time):
        if not self.use_session_filter:
            return True
        try:
            return 7 <= bar_time.hour < 20
        except:
            return True

    def run(self, df: pd.DataFrame, precomp=None, verbose=False) -> BacktestResult:
        if precomp is None:
            precomp = DataPrecomputer(
                df,
                hurst_window = self.hurst_window,
                chop_period  = self.chop_period,
                ob_lookback  = self.ob_lookback,
            )
        return self._run_core(df, precomp, verbose)

    def _run_core(self, df, precomp, verbose=False):
        result = BacktestResult()
        result.final_equity = self.initial_capital

        try:
            equity       = self.initial_capital
            eq_curve     = [equity]
            trades       = []
            kelly        = InstitutionalKelly()
            basket       = BasketManager(**self.basket_cfg)
            n            = min(len(df), len(precomp._kalman_vel))

            for idx in range(150, n):
                price    = df['close'].iloc[idx]
                bar_time = df.index[idx]
                regime   = precomp.regime_at(idx)

                if basket.is_open:
                    ob_fvg = precomp._ob[idx] != 0 or precomp._fvg[idx] != 0
                    if basket.should_dca(price, regime.is_trending, ob_fvg):
                        lot = kelly.lot_size(equity, regime.hurst, regime.kalman_acc) * 0.5
                        basket.add_layer(price, lot, "DCA")

                    why = basket.should_close(price, equity)
                    if why:
                        pnl  = basket.unrealized_pnl(price)
                        pnl -= abs(pnl) * self.commission * 2
                        equity += pnl
                        trades.append({'pnl': pnl, 'r': pnl / (self.initial_capital+1e-10)})
                        kelly.record(pnl, equity * self.position_pct)
                        basket.close()

                elif not basket.is_open:
                    if regime.is_choppy or not self._session_ok(bar_time):
                        eq_curve.append(equity)
                        continue

                    sig = precomp.signal_at(idx, self.score_threshold)
                    if sig.direction != 0:
                        regime_ok = regime.trend_dir == sig.direction or regime.trend_dir == 0
                        if regime_ok:
                            conf  = float(np.clip((sig.score - self.score_threshold) / max(1, 7 - self.score_threshold), 0.3, 1.0))
                            lot   = kelly.lot_size(equity, regime.hurst, regime.kalman_acc, self.initial_capital) * conf
                            entry = price * (1 + self.slippage * sig.direction)
                            basket.open(entry, lot, sig.direction, str(sig.reasons[:2]))

                eq_curve.append(equity)

            if basket.is_open:
                fp  = df['close'].iloc[-1]
                pnl = basket.unrealized_pnl(fp)
                equity += pnl
                trades.append({'pnl': pnl, 'r': pnl/(self.initial_capital+1e-10)})
                eq_curve[-1] = equity

            eq  = np.array(eq_curve)
            ret = np.diff(eq) / (eq[:-1] + 1e-10)

            result.total_return = (eq[-1] - self.initial_capital) / self.initial_capital
            result.final_equity = float(eq[-1])
            peak = np.maximum.accumulate(eq)
            result.max_drawdown = float(abs(((eq - peak) / (peak+1e-10)).min()))

            ppy = self._timeframe(df)
            if len(ret) > 1 and ret.std() > 1e-10:
                result.sharpe_ratio = float(np.clip(ret.mean()/ret.std()*np.sqrt(ppy), -10, 10))
            neg = ret[ret < 0]
            if len(neg) > 1 and neg.std() > 1e-10:
                result.sortino_ratio = float(np.clip(ret.mean()/neg.std()*np.sqrt(ppy), -10, 10))

            ny = len(df) / ppy
            if ny > 0 and eq[-1] > 0:
                result.cagr = float(np.clip((eq[-1]/self.initial_capital)**(1/max(ny,0.1))-1, -1, 10))
            if result.max_drawdown > 0:
                result.calmar_ratio = float(np.clip(result.cagr/result.max_drawdown, -10, 50))

            if trades:
                result.total_trades = len(trades)
                wins   = [t for t in trades if t['pnl'] > 0]
                losses = [t for t in trades if t['pnl'] <= 0]
                result.win_rate      = len(wins) / len(trades)
                gp = sum(t['pnl'] for t in wins)
                gl = abs(sum(t['pnl'] for t in losses)) if losses else 1e-10
                result.profit_factor = float(np.clip(gp/(gl+1e-10), 0, 20))

            if verbose:
                print(result.summary())

        except Exception as e:
            if verbose:
                import traceback; traceback.print_exc()

        return result

    def _timeframe(self, df):
        try:
            d = (df.index[1]-df.index[0]).total_seconds()
            if d <= 900:   return 365*96
            if d <= 3600:  return 365*24
            if d <= 14400: return 365*6
            if d <= 86400: return 365
            return 52
        except:
            return 252