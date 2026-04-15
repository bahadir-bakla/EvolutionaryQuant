"""
run_ml_scalper.py
==================
End-to-end pipeline:
  1. Load Gold M1 data (2019-2022 train, 2023-2024 OOS)
  2. Train LightGBM with walk-forward CV
  3. Backtest OOS with ML signals
  4. Print final report

Usage:
    python run_ml_scalper.py
    python run_ml_scalper.py --tp 5 --sl 3 --threshold 0.55
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_engine.data_loader import load_gold_years
from ml_engine.trainer import walk_forward_train, load_model
from ml_engine.backtest_ml import backtest_ml, MLScalperParams


def main():
    parser = argparse.ArgumentParser(description='ML Gold Scalper')
    parser.add_argument('--tp',        type=float, default=6.0,  help='Take profit pts')
    parser.add_argument('--sl',        type=float, default=3.0,  help='Stop loss pts')
    parser.add_argument('--max_bars',  type=int,   default=120,  help='Max hold bars')
    parser.add_argument('--threshold', type=float, default=0.65, help='Signal probability threshold')
    parser.add_argument('--lot',       type=float, default=0.01, help='Lot size')
    parser.add_argument('--no_session',action='store_true',       help='Disable session filter')
    parser.add_argument('--skip_train',action='store_true',       help='Skip training, load saved model')
    args = parser.parse_args()

    print("=" * 60)
    print("  ML Gold Scalper — Pattern Recognition Engine")
    print("=" * 60)
    print(f"  TP={args.tp}  SL={args.sl}  MaxBars={args.max_bars}")
    print(f"  Threshold={args.threshold}  Lot={args.lot}")

    # -- Load data -----------------------------------------------------
    print("\n[DATA] Loading Gold M1...")
    m1_train = load_gold_years(years=list(range(2011, 2023)))   # 2011-2022 train (12 yıl)
    m1_oos   = load_gold_years(years=[2023, 2024])              # 2023-2024 OOS
    print(f"  Train: {len(m1_train):,} bars  [{m1_train.index[0].date()} -> {m1_train.index[-1].date()}]")
    print(f"  OOS:   {len(m1_oos):,} bars   [{m1_oos.index[0].date()} -> {m1_oos.index[-1].date()}]")

    # -- Train ---------------------------------------------------------
    if args.skip_train:
        print("\n[TRAIN] Loading saved model...")
        bundle = load_model('lgbm_gold')
    else:
        bundle = walk_forward_train(
            m1_train=m1_train,
            m1_oos=None,   # evaluate separately below for clean OOS
            save_model=True,
            model_name='lgbm_gold',
        )

    # -- OOS Backtest --------------------------------------------------
    print("\n[BACKTEST] OOS 2023-2024...")
    params = MLScalperParams(
        tp_pts        = args.tp,
        sl_pts        = args.sl,
        spread_pts    = 0.4,
        lot_base      = args.lot,
        max_hold_bars = args.max_bars,
        threshold     = args.threshold,
        session_filter = not args.no_session,
    )
    result = backtest_ml(m1_oos, bundle, params, initial_capital=1_000.0)

    trades = result['trades']
    m = result['metrics']

    if trades.empty:
        print("\n  No trades generated — try lowering --threshold")
        return

    # -- Summary --------------------------------------------------------
    print("\n" + "=" * 60)
    print("  FINAL REPORT — OOS 2023-2024")
    print("=" * 60)
    print(f"  Trades         : {m['total_trades']}")
    print(f"  Win Rate       : {m['win_rate']:.1%}")
    print(f"  Profit Factor  : {m['profit_factor']:.3f}")
    print(f"  Total Return   : {m['total_return_pct']:.1f}%")
    print(f"  Max Drawdown   : {m['max_dd_pct']:.1f}%")
    print(f"  Avg Hold       : {m['avg_hold_bars']:.0f} bars = {m['avg_hold_bars']:.0f} min")
    print(f"  Trades/Day     : {m['trades_per_day']:.2f}")
    print(f"  Final Equity   : ${m['final_equity']:.2f}")

    # Weekly P&L
    if not trades.empty:
        trades['week'] = trades['exit_ts'].dt.to_period('W')
        weekly = trades.groupby('week')['pnl'].sum()
        print(f"\n  Weekly P&L (sample):")
        print(weekly.tail(8).to_string())
        print(f"\n  Avg Weekly P&L : ${weekly.mean():.2f}")
        print(f"  Weekly Std     : ${weekly.std():.2f}")


if __name__ == '__main__':
    main()
