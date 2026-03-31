"""
NQ Alpha DEAP — Main Runner
============================
EvolutionaryQuant | NQ Multi-Factor Genetic Optimization

Usage:
    cd c:\\...\\kalman
    python nq_alpha_deap/run_optimizer.py
    python nq_alpha_deap/run_optimizer.py --data cache_qqq_5m.pkl --tf 5m --pop 80 --gen 200
    python nq_alpha_deap/run_optimizer.py --data institutional_engine/cache/NQ_1h.csv --tf 1h

Data sources (auto-detected in priority order):
  1. --data flag (pkl or csv)
  2. cache_qqq_5m.pkl  (root workspace)
  3. cache_qqq_15m.pkl
  4. cache_qqq_1H.pkl
  5. institutional_engine/cache/NQ_1h.csv
"""

import sys, os, argparse, pickle, time
import numpy as np
import pandas as pd
import multiprocessing

_DIR    = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
sys.path.insert(0, _DIR)
sys.path.insert(0, _PARENT)

from nq_alpha_strategy import (
    NQAlphaParams, add_nq_alpha_features, DEFAULT_PARAMS, GENOME_SIZE
)
from backtest_engine  import run_backtest, backtest_from_raw, BacktestResult
from deap_optimizer   import run_deap_optimizer


# ─────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────
def _load_pkl(path: str) -> pd.DataFrame:
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        df = obj
    elif isinstance(obj, dict):
        # Some caches store {'df': ..., 'meta': ...}
        df = obj.get('df', obj.get('data', None))
        if df is None:
            raise ValueError(f"Cannot find DataFrame in pkl dict: {list(obj.keys())}")
    else:
        raise TypeError(f"Unexpected pkl type: {type(obj)}")
    df.columns = df.columns.str.lower()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = df.columns.str.lower()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample to target timeframe."""
    return df.resample(tf).agg(
        open=('open','first'), high=('high','max'),
        low=('low','min'),     close=('close','last'),
        volume=('volume','sum'),
    ).dropna(subset=['open','close'])


def load_data(data_arg: str = None, tf: str = '5m') -> pd.DataFrame:
    """
    Load NQ / QQQ data from various sources.
    tf: target timeframe string ('5m', '15m', '1h')
    """
    # Priority list
    search_paths = []
    if data_arg:
        search_paths.append(data_arg)

    root = _PARENT
    search_paths += [
        os.path.join(root, 'cache_qqq_5m.pkl'),
        os.path.join(root, 'cache_qqq_15m.pkl'),
        os.path.join(root, 'cache_qqq_1H.pkl'),
        os.path.join(root, 'institutional_engine', 'cache', 'NQ_1h.csv'),
    ]

    df = None
    for path in search_paths:
        if not os.path.exists(path):
            continue
        try:
            if path.endswith('.pkl'):
                df = _load_pkl(path)
            else:
                df = _load_csv(path)
            print(f"✅ Data yüklendi: {os.path.basename(path)} "
                  f"({len(df):,} bar,  "
                  f"{df.index[0].date()} → {df.index[-1].date()})")
            break
        except Exception as e:
            print(f"⚠️  {os.path.basename(path)} yüklenemedi: {e}")
            continue

    if df is None:
        raise FileNotFoundError(
            "Hiçbir veri kaynağı bulunamadı! "
            "--data ile bir pkl veya csv belirtin."
        )

    # Ensure volume column
    if 'volume' not in df.columns:
        df['volume'] = 1000

    # Resample if needed
    current_tf = _infer_tf(df)
    if current_tf != tf and tf in ('5m','15m','1h','4h'):
        print(f"🔄 Resample: {current_tf} → {tf}")
        df = _resample(df, tf)

    print(f"📊 Final: {len(df):,} {tf} barları | "
          f"Fiyat aralığı: {df['close'].min():.2f} – {df['close'].max():.2f}")
    return df


def _infer_tf(df: pd.DataFrame) -> str:
    """Guess timeframe from index deltas."""
    if len(df) < 3:
        return 'unknown'
    delta_secs = (df.index[1] - df.index[0]).total_seconds()
    mapping = {300: '5m', 900: '15m', 3600: '1h', 14400: '4h', 86400: '1d'}
    for s, label in mapping.items():
        if abs(delta_secs - s) < s * 0.5:
            return label
    return 'unknown'


# ─────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────
def _print_result(r: BacktestResult, label: str):
    print(f"\n{'═'*58}")
    print(f"  {label}")
    print(f"{'═'*58}")
    print(f"  CAGR           : {r.cagr*100:+.2f}%")
    print(f"  Total Return   : {r.total_return*100:+.2f}%")
    print(f"  Max Drawdown   : {r.max_drawdown*100:.2f}%")
    print(f"  Win Rate       : {r.win_rate*100:.1f}%")
    print(f"  Profit Factor  : {r.profit_factor:.2f}")
    print(f"  Avg R:R        : {r.avg_rr:.2f}")
    print(f"  Sharpe Ratio   : {r.sharpe_ratio:.2f}")
    print(f"  Total Trades   : {r.total_trades}")
    print(f"  Final Equity   : ${r.final_equity:,.2f}")
    print(f"  Fitness        : {r.fitness():.4f}")


def _print_wf_splits(df_raw: pd.DataFrame, params: NQAlphaParams,
                     n_splits: int, oos_pct: float, init_cap: float):
    """Print walk-forward OOS results for best params."""
    from deap_optimizer import build_wf_splits
    raw_splits = build_wf_splits(df_raw, params, n_splits, oos_pct)
    print(f"\n📊 Walk-Forward OOS ({n_splits} splits, OOS={oos_pct*100:.0f}%)")
    print(f"{'─'*58}")
    pos = 0
    for i, raw in enumerate(raw_splits):
        try:
            r = backtest_from_raw(raw, params, init_cap)
            flag = "✅" if r.total_return > 0 else "❌"
            if r.total_return > 0:
                pos += 1
            print(f"  Split {i+1} [{raw.index[0].date()} → {raw.index[-1].date()}]: "
                  f"Ret={r.total_return*100:+.1f}% | "
                  f"PF={r.profit_factor:.2f} | "
                  f"Trades={r.total_trades}  {flag}")
        except Exception as e:
            print(f"  Split {i+1}: ERROR — {e}  ❌")
    pct = pos / max(len(raw_splits), 1) * 100
    print(f"{'─'*58}")
    print(f"  Pozitif splits: {pos}/{len(raw_splits)}  ({pct:.0f}%)  "
          f"{'✅ PORTFOLYO HAZIR' if pos >= len(raw_splits)*0.6 else '⚠️  Daha fazla optimizasyon gerekebilir'}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='NQ Alpha DEAP Optimizer — EvolutionaryQuant')
    parser.add_argument('--data',    type=str,   default=None,  help='Veri dosyası (.pkl veya .csv)')
    parser.add_argument('--tf',      type=str,   default='5m',  help='Timeframe: 5m, 15m, 1h')
    parser.add_argument('--capital', type=float, default=1_000.0)
    parser.add_argument('--pop',     type=int,   default=80,    help='Population size')
    parser.add_argument('--gen',     type=int,   default=200,   help='Generations')
    parser.add_argument('--splits',  type=int,   default=5,     help='WF splits')
    parser.add_argument('--jobs',    type=int,   default=None,  help='CPU cores')
    parser.add_argument('--quick',   action='store_true',       help='Hızlı test (pop=30, gen=20)')
    args = parser.parse_args()

    print("=" * 62)
    print("  🧬  EvolutionaryQuant — NQ Alpha DEAP Optimizer")
    print("=" * 62)
    print(f"  Strategy: 15-min ORB | Pivot | Momentum | FVG | OB")
    print(f"            Kalman Filter | Hurst Exponent | VWAP | Gap")
    print(f"  Genome  : {GENOME_SIZE} parameters")
    print("=" * 62)

    if args.quick:
        args.pop = 30; args.gen = 20; args.splits = 3
        print("⚡ Quick mode aktif (pop=30, gen=20, splits=3)\n")

    # 1. Load data
    print("\n📥 Veri yükleniyor...")
    df_raw = load_data(args.data, args.tf)

    # 2. Default params baseline
    print("\n🔍 Default params baseline testi...")
    try:
        r_def = backtest_from_raw(df_raw, DEFAULT_PARAMS, args.capital)
        _print_result(r_def, "DEFAULT PARAMS (Baseline)")
    except Exception as e:
        print(f"  Baseline test hatası: {e}")
        r_def = BacktestResult()

    # 3. DEAP optimization
    print(f"\n🚀 DEAP Optimizasyon Başlıyor...")
    best_params, best_fit, stats, hof = run_deap_optimizer(
        df_raw          = df_raw,
        initial_capital = args.capital,
        population      = args.pop,
        generations     = args.gen,
        n_splits        = args.splits,
        n_jobs          = args.jobs,
        verbose         = True,
        output_dir      = os.path.join(_DIR, 'outputs'),
    )

    # 4. Full backtest with best params
    print("\n🏆 En İyi Parametre — Full Backtest...")
    try:
        r_best = backtest_from_raw(df_raw, best_params, args.capital)
        _print_result(r_def,  "DEFAULT (Baseline)")
        _print_result(r_best, "OPTİMİZE EDİLMİŞ (Full Data)")
    except Exception as e:
        print(f"  Full backtest hatası: {e}")

    # 5. Walk-forward analysis
    _print_wf_splits(df_raw, best_params, args.splits, 0.30, args.capital)

    # 6. Top 5 genomes
    print("\n🏆 TOP 5 GENOME")
    print("─" * 58)
    from nq_alpha_strategy import decode_genome
    for i, ind in enumerate(list(hof)[:5]):
        p = decode_genome(list(ind))
        print(f"  #{i+1}  Fit:{ind.fitness.values[0]:7.3f} | "
              f"ORB={p.orb_breakout_mult:.2f}x | "
              f"MinScore={p.min_score} | "
              f"TP={p.tp_atr_mult:.1f}x | "
              f"SL={p.sl_atr_mult:.1f}x | "
              f"Hurst={p.hurst_window}")

    print("\n✅ Her şey tamamlandı!")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
