"""
Alpha Engine — Walk-Forward Validation
========================================
Sliding-window walk-forward for any strategy module.

Window scheme (default):
  - Train : 18 months
  - Test  : 6 months
  - Step  : 6 months (non-overlapping test windows)

Each fold:
  1. DEAP optimize on train window
  2. Evaluate best_params on test window (OOS)
  3. Collect OOS metrics

Summary statistics across all folds reveal real-world robustness.
"""

import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_ENGINE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class WFSplit:
    fold:        int
    train_start: str
    train_end:   str
    oos_start:   str
    oos_end:     str
    best_params: Any
    best_fit_is: float
    oos_result:  Any


@dataclass
class WFSummary:
    strategy_name: str
    n_folds:       int
    pos_folds:     int
    oos_returns:   List[float] = field(default_factory=list)
    oos_dds:       List[float] = field(default_factory=list)
    oos_trades:    List[int]   = field(default_factory=list)
    splits:        List[WFSplit] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.pos_folds / max(self.n_folds, 1)

    @property
    def avg_oos_return(self) -> float:
        return float(np.mean(self.oos_returns)) if self.oos_returns else 0.0

    @property
    def avg_oos_dd(self) -> float:
        return float(np.mean(self.oos_dds)) if self.oos_dds else 0.0

    @property
    def median_oos_return(self) -> float:
        return float(np.median(self.oos_returns)) if self.oos_returns else 0.0

    def is_robust(self, min_hit_rate: float = 0.60) -> bool:
        return (self.hit_rate >= min_hit_rate and
                self.avg_oos_return > 0 and
                self.avg_oos_dd < 0.35)

    def print_report(self):
        print(f"\n{'═'*64}")
        print(f"  WALK-FORWARD: {self.strategy_name}")
        print(f"{'═'*64}")
        print(f"  Folds           : {self.n_folds}")
        print(f"  Positive folds  : {self.pos_folds}/{self.n_folds}  "
              f"({self.hit_rate*100:.0f}%)")
        print(f"  Avg OOS Return  : {self.avg_oos_return*100:+.2f}%")
        print(f"  Med OOS Return  : {self.median_oos_return*100:+.2f}%")
        print(f"  Avg OOS DD      : {self.avg_oos_dd*100:.2f}%")
        print(f"\n  {'Fold':<5} {'Train':>22} {'OOS':>22} {'Ret':>8} {'DD':>8} {'Trades':>7}")
        print(f"  {'─'*72}")
        for s in self.splits:
            ret = getattr(s.oos_result, 'total_return', 0) if s.oos_result else 0
            dd  = getattr(s.oos_result, 'max_drawdown', 0) if s.oos_result else 0
            nt  = getattr(s.oos_result, 'total_trades', 0) if s.oos_result else 0
            flag = '[OK]' if ret > 0 else '[FAIL]'
            print(f"  {s.fold:<5} {s.train_start} → {s.train_end}  "
                  f"{s.oos_start} → {s.oos_end}  "
                  f"{ret*100:>+7.1f}% {dd*100:>7.1f}% {nt:>7} {flag}")
        print(f"{'═'*64}")
        verdict = "[OK] ROBUST" if self.is_robust() else "[WARN]  NEEDS MORE WORK"
        print(f"  Verdict: {verdict}")
        print(f"{'═'*64}")


# ─────────────────────────────────────────────────────────────────────
# SPLIT BUILDER
# ─────────────────────────────────────────────────────────────────────

def build_wf_splits(
    df:               pd.DataFrame,
    train_months:     int = 18,
    test_months:      int = 6,
) -> List[Dict]:
    """
    Build sliding-window train/test date pairs.
    Returns list of {'train': df_slice, 'test': df_slice, 'dates': {...}}
    """
    if df.empty or len(df) < 10:
        return []

    splits = []
    start = df.index[0]
    end   = df.index[-1]

    train_delta = pd.DateOffset(months=train_months)
    test_delta  = pd.DateOffset(months=test_months)

    cursor = start
    fold   = 1

    while True:
        train_end = cursor + train_delta
        test_end  = train_end + test_delta

        if test_end > end:
            break

        df_train = df[(df.index >= cursor)     & (df.index < train_end)]
        df_test  = df[(df.index >= train_end)  & (df.index < test_end)]

        if len(df_train) < 100 or len(df_test) < 20:
            cursor = cursor + test_delta
            fold  += 1
            continue

        splits.append({
            'fold':        fold,
            'train':       df_train,
            'test':        df_test,
            'train_start': str(cursor.date()),
            'train_end':   str(train_end.date()),
            'oos_start':   str(train_end.date()),
            'oos_end':     str(test_end.date()),
        })

        cursor = cursor + test_delta
        fold  += 1

    return splits


# ─────────────────────────────────────────────────────────────────────
# WALK-FORWARD RUNNER
# ─────────────────────────────────────────────────────────────────────

def run_walk_forward(
    module,
    df:              pd.DataFrame,
    strategy_name:   str = 'strategy',
    capital:         float = 1_000.0,
    train_months:    int = 18,
    test_months:     int = 6,
    population:      int = 50,
    generations:     int = 60,
    n_splits_inner:  int = 2,
    n_jobs:          Optional[int] = None,
    output_dir:      Optional[str] = None,
    verbose:         bool = True,
) -> WFSummary:
    """
    Full walk-forward optimization + OOS evaluation for any strategy.

    module: any strategy module with GENOME_SIZE, GENE_BOUNDS,
            decode_genome(), backtest_from_raw()
    """
    from deap_runner import run_deap

    splits = build_wf_splits(df, train_months, test_months)

    if not splits:
        raise ValueError(f"Not enough data for walk-forward (need ≥{train_months+test_months}m)")

    summary = WFSummary(strategy_name=strategy_name, n_folds=len(splits), pos_folds=0)

    if verbose:
        print(f"\n{'═'*64}")
        print(f"  WALK-FORWARD: {strategy_name.upper()}")
        print(f"  Folds: {len(splits)}  |  Train: {train_months}m  |  OOS: {test_months}m")
        print(f"{'═'*64}")

    for sp in splits:
        fold  = sp['fold']
        df_tr = sp['train']
        df_oos= sp['test']

        if verbose:
            print(f"\n  Fold {fold}: Train {sp['train_start']}→{sp['train_end']}"
                  f"  OOS {sp['oos_start']}→{sp['oos_end']}")

        # Optimize on train
        try:
            best_params, best_fit, _, _ = run_deap(
                module        = module,
                df_train      = df_tr,
                capital       = capital,
                population    = population,
                generations   = generations,
                n_splits      = n_splits_inner,
                n_jobs        = n_jobs,
                verbose       = False,
                output_dir    = None,
                strategy_name = f'{strategy_name}_fold{fold}',
            )
        except Exception as e:
            if verbose:
                print(f"  [WARN]  Fold {fold} optimization failed: {e}")
            summary.splits.append(WFSplit(
                fold=fold,
                train_start=sp['train_start'], train_end=sp['train_end'],
                oos_start=sp['oos_start'], oos_end=sp['oos_end'],
                best_params=None, best_fit_is=-999.0, oos_result=None,
            ))
            continue

        # OOS evaluation
        try:
            oos_result = module.backtest_from_raw(df_oos, best_params, capital)
        except Exception as e:
            if verbose:
                print(f"  [WARN]  Fold {fold} OOS eval failed: {e}")
            oos_result = None

        oos_ret = getattr(oos_result, 'total_return', 0)
        oos_dd  = getattr(oos_result, 'max_drawdown', 0)
        oos_n   = getattr(oos_result, 'total_trades', 0)

        summary.oos_returns.append(oos_ret)
        summary.oos_dds.append(oos_dd)
        summary.oos_trades.append(oos_n)
        if oos_ret > 0:
            summary.pos_folds += 1

        if verbose:
            flag = '[OK]' if oos_ret > 0 else '[FAIL]'
            print(f"    IS fit={best_fit:>7.3f}  "
                  f"OOS ret={oos_ret*100:>+6.1f}%  "
                  f"dd={oos_dd*100:.1f}%  "
                  f"trades={oos_n}  {flag}")

        summary.splits.append(WFSplit(
            fold=fold,
            train_start=sp['train_start'], train_end=sp['train_end'],
            oos_start=sp['oos_start'], oos_end=sp['oos_end'],
            best_params=best_params, best_fit_is=best_fit,
            oos_result=oos_result,
        ))

    if output_dir:
        _save_wf_report(summary, output_dir)

    if verbose:
        summary.print_report()

    return summary


def _save_wf_report(summary: WFSummary, output_dir: str):
    import json, time
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    report = {
        'strategy':    summary.strategy_name,
        'n_folds':     summary.n_folds,
        'pos_folds':   summary.pos_folds,
        'hit_rate':    summary.hit_rate,
        'avg_oos_ret': summary.avg_oos_return,
        'avg_oos_dd':  summary.avg_oos_dd,
        'robust':      summary.is_robust(),
        'folds': [
            {
                'fold':       s.fold,
                'train':      f"{s.train_start}→{s.train_end}",
                'oos':        f"{s.oos_start}→{s.oos_end}",
                'best_fit_is': s.best_fit_is,
                'oos_return': getattr(s.oos_result, 'total_return', None),
                'oos_dd':     getattr(s.oos_result, 'max_drawdown', None),
                'oos_trades': getattr(s.oos_result, 'total_trades', None),
            }
            for s in summary.splits
        ],
    }
    path = os.path.join(output_dir, f'wf_{summary.strategy_name}_{ts}.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  WF report saved: {path}")
