"""
run_ml_scalper.py  —  Bidirectional ML Gold Scalper
=====================================================
Pipeline:
  1. Load Gold M1 data (2011-2022 train, 2023-2024 OOS)
  2. Train LONG model  (or load if --skip_train)
  3. Train SHORT model (or load if --skip_train)
  4. Backtest OOS with both signals
  5. Print final report

Usage:
    python run_ml_scalper.py                        # full retrain
    python run_ml_scalper.py --skip_train           # load saved models
    python run_ml_scalper.py --skip_train --threshold 0.72
    python run_ml_scalper.py --skip_train --long_only
    python run_ml_scalper.py --skip_train --short_only
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_engine.data_loader import load_gold_years
from ml_engine.trainer import walk_forward_train, load_model, train_both_sides, load_both_models
from ml_engine.backtest_ml import backtest_ml, MLScalperParams


def main():
    parser = argparse.ArgumentParser(description='Bidirectional ML Gold Scalper')
    parser.add_argument('--tp',         type=float, default=6.0,  help='TP pts')
    parser.add_argument('--sl',         type=float, default=3.0,  help='SL pts')
    parser.add_argument('--max_bars',   type=int,   default=120,  help='Max hold bars')
    parser.add_argument('--threshold',  type=float, default=0.70, help='Signal threshold')
    parser.add_argument('--lot',        type=float, default=0.01, help='Lot size')
    parser.add_argument('--no_session', action='store_true',            help='Disable session filter')
    parser.add_argument('--skip_train', action='store_true',            help='Load saved models')
    parser.add_argument('--long_only',  action='store_true',            help='Long only mode')
    parser.add_argument('--short_only', action='store_true',            help='Short only mode')
    parser.add_argument('--oos_years',  type=int, nargs='+', default=None, help='OOS years e.g. --oos_years 2025')
    parser.add_argument('--train_years',type=int, nargs='+', default=None, help='Train years override')
    args = parser.parse_args()

    print("=" * 60)
    print("  ML Gold Scalper v2 — Bidirectional")
    print("=" * 60)
    print(f"  TP={args.tp}  SL={args.sl}  MaxBars={args.max_bars}")
    print(f"  Threshold={args.threshold}  Lot={args.lot}")
    mode = "LONG+SHORT"
    if args.long_only:  mode = "LONG only"
    if args.short_only: mode = "SHORT only"
    print(f"  Mode: {mode}")

    # -- Load data --------------------------------------------------------
    train_years = args.train_years or list(range(2011, 2023))
    oos_years   = args.oos_years   or [2023, 2024]
    print("\n[DATA] Loading Gold M1...")
    m1_train = load_gold_years(years=train_years)
    m1_oos   = load_gold_years(years=oos_years)
    print(f"  Train: {len(m1_train):,} bars  [{m1_train.index[0].date()} -> {m1_train.index[-1].date()}]")
    print(f"  OOS:   {len(m1_oos):,} bars   [{m1_oos.index[0].date()} -> {m1_oos.index[-1].date()}]")
    label = f"OOS {oos_years[0]}" + (f"-{oos_years[-1]}" if len(oos_years) > 1 else "")

    # -- Train ------------------------------------------------------------
    if args.skip_train:
        print("\n[TRAIN] Loading saved models...")
        try:
            long_bundle, short_bundle = load_both_models()
            print("  Loaded: lgbm_gold_long + lgbm_gold_short")
        except FileNotFoundError:
            # Fallback to legacy single model
            long_bundle  = load_model('lgbm_gold')
            short_bundle = None
            print("  Loaded: lgbm_gold (legacy long-only)")
    else:
        print("\n[TRAIN] Training LONG + SHORT models...")
        long_bundle, short_bundle = train_both_sides(m1_train)

    # -- Handle mode flags ------------------------------------------------
    if args.long_only:
        short_bundle = None
    if args.short_only:
        # Use short model as "long" slot and disable long signals
        long_bundle  = short_bundle
        short_bundle = None

    # -- OOS Backtest -----------------------------------------------------
    print("\n[BACKTEST] OOS 2023-2024...")
    params = MLScalperParams(
        tp_pts         = args.tp,
        sl_pts         = args.sl,
        spread_pts     = 0.4,
        lot_base       = args.lot,
        max_hold_bars  = args.max_bars,
        threshold      = args.threshold,
        session_filter = not args.no_session,
    )
    result = backtest_ml(m1_oos, long_bundle, short_bundle, params, initial_capital=1_000.0)

    trades = result['trades']
    m      = result['metrics']

    if trades.empty:
        print("\n  No trades — try lowering --threshold")
        return

    # -- Final Report -----------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  FINAL REPORT — {label}")
    print("=" * 60)
    print(f"  Trades         : {m['total_trades']}")
    print(f"  Win Rate       : {m['win_rate']:.1%}")
    print(f"  Profit Factor  : {m['profit_factor']:.3f}")
    print(f"  Total Return   : {m['total_return_pct']:.1f}%")
    print(f"  Max Drawdown   : {m['max_dd_pct']:.1f}%")
    print(f"  Avg Hold       : {m['avg_hold_bars']:.0f} min")
    print(f"  Trades/Day     : {m['trades_per_day']:.2f}")
    print(f"  Final Equity   : ${m['final_equity']:.2f}")

    # Long vs Short breakdown
    if 'by_dir' in m and not m['by_dir'].empty:
        print(f"\n  Direction P&L:")
        print(m['by_dir'].to_string())

    # Weekly
    trades['week'] = trades['exit_ts'].dt.to_period('W')
    weekly = trades.groupby('week')['pnl'].sum()
    print(f"\n  Weekly P&L (last 8):")
    print(weekly.tail(8).to_string())
    print(f"\n  Avg Weekly P&L : ${weekly.mean():.2f}")
    print(f"  Weekly Std     : ${weekly.std():.2f}")
    print(f"  Positive weeks : {(weekly > 0).mean():.1%}")


if __name__ == '__main__':
    main()
