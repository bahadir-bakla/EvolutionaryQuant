"""
Alpha Engine — Portfolio Backtest
===================================
Runs all 5 strategies on their respective instruments and combines
equity curves into a single portfolio result.

Capital allocation (default):
  - Gold Sniper    30%
  - Gold Master    25%
  - Silver         20%
  - NQ Alpha       15%
  - NQ Liquidity   10%

All strategies share the same spectral bias (meta_bias) signal.
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

_ENGINE = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_ENGINE)
sys.path.insert(0, _ENGINE)
sys.path.insert(0, _ROOT)

# Local strategy modules
from gold_sniper_bt  import (backtest_from_raw as sniper_bt,
                              GoldSniperParams, DEFAULT_PARAMS as SNIPER_DEF)
from gold_master_bt  import (backtest_from_raw as master_bt,
                              GoldMasterParams, DEFAULT_PARAMS as MASTER_DEF)
from nq_liquidity_bt import (backtest_from_raw as liq_bt,
                              NQLiquidityParams, DEFAULT_PARAMS as LIQ_DEF)

# Silver (from existing module)
sys.path.insert(0, os.path.join(_ROOT, 'silver_reversion_deap'))
from silver_strategy  import SilverReversionParams, DEFAULT_PARAMS as SILVER_DEF
from backtest_engine  import backtest_from_raw as silver_bt

# NQ Alpha (from existing module, now fixed)
sys.path.insert(0, os.path.join(_ROOT, 'nq_alpha_deap'))
from nq_alpha_strategy import NQAlphaParams, DEFAULT_PARAMS as NQ_ALPHA_DEF
from backtest_engine   import backtest_from_raw as nq_alpha_bt  # type: ignore


# ─────────────────────────────────────────────────────────────────────
# ALLOCATION
# ─────────────────────────────────────────────────────────────────────

DEFAULT_ALLOC = {
    'gold_sniper':  0.30,
    'gold_master':  0.25,
    'silver':       0.20,
    'nq_alpha':     0.15,
    'nq_liquidity': 0.10,
}


@dataclass
class PortfolioResult:
    total_return:  float = 0.0
    cagr:          float = 0.0
    max_drawdown:  float = 0.0
    sharpe_ratio:  float = 0.0
    win_rate:      float = 0.0
    profit_factor: float = 0.0
    total_trades:  int   = 0
    final_equity:  float = 0.0
    per_strategy:  Dict  = field(default_factory=dict)

    def fitness(self) -> float:
        if self.total_trades < 10:
            return -999.0
        if self.total_return <= 0:
            return float(np.clip(self.total_return * 5, -50, -0.01))
        if self.profit_factor < 1.0:
            return float(np.clip((self.profit_factor - 1) * 10, -20, -0.01))
        if self.max_drawdown > 0.60:
            return -50.0
        pf   = float(np.clip(self.profit_factor, 1.0, 15))
        cagr = float(np.clip(self.cagr, 0, 20))
        sh   = float(np.clip(self.sharpe_ratio, 0, 10))
        wr   = float(np.clip(self.win_rate, 0, 1))
        ddp  = max(0.1, 1 - max(0, self.max_drawdown - 0.12) * 3)
        return float(np.clip(
            (pf * 0.25 + cagr * 0.40 + sh * 0.20 + wr * 0.15) * ddp, 0, 200))

    def print_summary(self):
        print(f"\n{'═'*62}")
        print(f"  PORTFOLIO SUMMARY")
        print(f"{'═'*62}")
        print(f"  Total Return   : {self.total_return*100:+.2f}%")
        print(f"  CAGR           : {self.cagr*100:+.2f}%")
        print(f"  Max Drawdown   : {self.max_drawdown*100:.2f}%")
        print(f"  Sharpe Ratio   : {self.sharpe_ratio:.2f}")
        print(f"  Win Rate       : {self.win_rate*100:.1f}%")
        print(f"  Profit Factor  : {self.profit_factor:.2f}")
        print(f"  Total Trades   : {self.total_trades}")
        print(f"  Final Equity   : ${self.final_equity:,.2f}")
        print(f"  Fitness        : {self.fitness():.4f}")
        if self.per_strategy:
            print(f"\n  {'Strategy':<16} {'Return':>8} {'DD':>8} {'Trades':>7} {'Fitness':>9}")
            print(f"  {'─'*52}")
            for name, r in self.per_strategy.items():
                fit = r.fitness() if hasattr(r, 'fitness') else 0
                ret = getattr(r, 'total_return', 0)
                dd  = getattr(r, 'max_drawdown', 0)
                n   = getattr(r, 'total_trades', 0)
                print(f"  {name:<16} {ret*100:>+7.1f}% {dd*100:>7.1f}% {n:>7}  {fit:>8.2f}")
        print(f"{'═'*62}")


# ─────────────────────────────────────────────────────────────────────
# PORTFOLIO RUNNER
# ─────────────────────────────────────────────────────────────────────

def run_portfolio(
    df_gold:      pd.DataFrame,
    df_silver:    pd.DataFrame,
    df_nq:        pd.DataFrame,
    params:       Optional[Dict[str, Any]] = None,
    alloc:        Optional[Dict[str, float]] = None,
    initial_capital: float = 1_000.0,
) -> PortfolioResult:
    """
    Run all 5 strategies and combine results.

    params: dict with keys 'gold_sniper', 'gold_master', 'silver',
                           'nq_alpha', 'nq_liquidity'
            Each value is the respective Params dataclass.
    """
    if alloc is None:
        alloc = DEFAULT_ALLOC
    if params is None:
        params = {}

    p_sniper  = params.get('gold_sniper',  SNIPER_DEF)
    p_master  = params.get('gold_master',  MASTER_DEF)
    p_silver  = params.get('silver',       SILVER_DEF)
    p_nq_alpha= params.get('nq_alpha',     NQ_ALPHA_DEF)
    p_liq     = params.get('nq_liquidity', LIQ_DEF)

    cap_sniper  = initial_capital * alloc.get('gold_sniper',  0.30)
    cap_master  = initial_capital * alloc.get('gold_master',  0.25)
    cap_silver  = initial_capital * alloc.get('silver',       0.20)
    cap_nq_alpha= initial_capital * alloc.get('nq_alpha',     0.15)
    cap_liq     = initial_capital * alloc.get('nq_liquidity', 0.10)

    results: Dict[str, Any] = {}

    print("  Running Gold Sniper...", end=' ', flush=True)
    if not df_gold.empty:
        results['gold_sniper'] = sniper_bt(df_gold, p_sniper, cap_sniper)
        print(f"Ret={results['gold_sniper'].total_return*100:+.1f}%  "
              f"Trades={results['gold_sniper'].total_trades}")
    else:
        print("SKIPPED (no data)")

    print("  Running Gold Master...", end=' ', flush=True)
    if not df_gold.empty:
        results['gold_master'] = master_bt(df_gold, p_master, cap_master)
        print(f"Ret={results['gold_master'].total_return*100:+.1f}%  "
              f"Trades={results['gold_master'].total_trades}")
    else:
        print("SKIPPED (no data)")

    print("  Running Silver Reversion...", end=' ', flush=True)
    if not df_silver.empty:
        results['silver'] = silver_bt(df_silver, p_silver, cap_silver)
        print(f"Ret={results['silver'].total_return*100:+.1f}%  "
              f"Trades={results['silver'].total_trades}")
    else:
        print("SKIPPED (no data)")

    print("  Running NQ Alpha...", end=' ', flush=True)
    if not df_nq.empty:
        results['nq_alpha'] = nq_alpha_bt(df_nq, p_nq_alpha, cap_nq_alpha)
        print(f"Ret={results['nq_alpha'].total_return*100:+.1f}%  "
              f"Trades={results['nq_alpha'].total_trades}")
    else:
        print("SKIPPED (no data)")

    print("  Running NQ Liquidity...", end=' ', flush=True)
    if not df_nq.empty:
        results['nq_liquidity'] = liq_bt(df_nq, p_liq, cap_liq)
        print(f"Ret={results['nq_liquidity'].total_return*100:+.1f}%  "
              f"Trades={results['nq_liquidity'].total_trades}")
    else:
        print("SKIPPED (no data)")

    return _combine(results, alloc, initial_capital)


def _combine(results: Dict, alloc: Dict, initial_capital: float) -> PortfolioResult:
    """Combine per-strategy results into portfolio metrics."""
    port = PortfolioResult(
        final_equity  = initial_capital,
        per_strategy  = results,
    )

    if not results:
        return port

    total_final = 0.0
    all_pnls: list = []
    all_wins: list = []
    all_loss: list = []
    all_trades = 0

    for name, r in results.items():
        cap = initial_capital * alloc.get(name, 0.0)
        total_final += getattr(r, 'final_equity', cap)
        all_trades  += getattr(r, 'total_trades', 0)
        for t in getattr(r, 'trades', []):
            pnl = t.get('pnl', 0)
            all_pnls.append(pnl)
            if pnl > 0:
                all_wins.append(pnl)
            else:
                all_loss.append(pnl)

    port.final_equity  = total_final
    port.total_return  = (total_final - initial_capital) / initial_capital
    port.total_trades  = all_trades

    if all_wins or all_loss:
        port.win_rate = len(all_wins) / max(len(all_pnls), 1)
        gp = sum(all_wins)
        gl = abs(sum(all_loss)) if all_loss else 1e-10
        port.profit_factor = float(np.clip(gp / (gl + 1e-10), 0, 20))

    # Portfolio-level CAGR — use average of strategy CAGRs weighted by allocation
    cagr_sum = 0.0
    dd_max   = 0.0
    sh_sum   = 0.0
    w_sum    = 0.0
    for name, r in results.items():
        w = alloc.get(name, 0.0)
        cagr_sum += getattr(r, 'cagr', 0.0) * w
        dd_max    = max(dd_max, getattr(r, 'max_drawdown', 0.0))
        sh_sum   += getattr(r, 'sharpe_ratio', 0.0) * w
        w_sum    += w

    if w_sum > 0:
        port.cagr         = cagr_sum / w_sum
        port.sharpe_ratio = sh_sum   / w_sum
    port.max_drawdown = dd_max

    return port
