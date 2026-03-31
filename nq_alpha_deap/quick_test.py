"""
NQ Alpha — Quick Test (NO optimization, default params)
=========================================================
EvolutionaryQuant | Hızlı doğrulama scripti

Çalıştır:
    cd c:\\...\\kalman
    python nq_alpha_deap/quick_test.py
    python nq_alpha_deap/quick_test.py --data institutional_engine/cache/NQ_1h.csv
"""

import sys, os, pickle
import pandas as pd
import numpy as np

_DIR    = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
sys.path.insert(0, _DIR)
sys.path.insert(0, _PARENT)

from nq_alpha_strategy import NQAlphaParams, add_nq_alpha_features, DEFAULT_PARAMS
from backtest_engine import run_backtest, backtest_from_raw, BacktestResult


def _load_data(path=None):
    """Load data — same priority as run_optimizer."""
    search = []
    if path:
        search.append(path)
    search += [
        os.path.join(_PARENT, 'cache_qqq_5m.pkl'),
        os.path.join(_PARENT, 'cache_qqq_15m.pkl'),
        os.path.join(_PARENT, 'cache_qqq_1H.pkl'),
        os.path.join(_PARENT, 'institutional_engine', 'cache', 'NQ_1h.csv'),
    ]
    for p in search:
        if not os.path.exists(p):
            continue
        try:
            if p.endswith('.pkl'):
                with open(p, 'rb') as f:
                    obj = pickle.load(f)
                df = obj if isinstance(obj, pd.DataFrame) else obj.get('df')
            else:
                df = pd.read_csv(p, index_col=0, parse_dates=True)
            df.columns = df.columns.str.lower()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if 'volume' not in df.columns:
                df['volume'] = 1000
            print(f"✅ {os.path.basename(p)}: {len(df):,} bars "
                  f"[{df.index[0].date()} → {df.index[-1].date()}]")
            return df
        except Exception as e:
            print(f"⚠️  {os.path.basename(p)}: {e}")
    raise FileNotFoundError("Veri bulunamadı!")


def print_result(r: BacktestResult, label: str):
    print(f"\n{'═'*56}")
    print(f"  {label}")
    print(f"{'═'*56}")
    ok = lambda a, b: "✅" if a > b else "❌"
    print(f"  CAGR           : {r.cagr*100:+.2f}%  {ok(r.cagr, 0.30)}")
    print(f"  Total Return   : {r.total_return*100:+.2f}%")
    print(f"  Max Drawdown   : {r.max_drawdown*100:.2f}%  {ok(0.30, r.max_drawdown)}")
    print(f"  Sharpe Ratio   : {r.sharpe_ratio:.2f}  {ok(r.sharpe_ratio, 1.0)}")
    print(f"  Win Rate       : {r.win_rate*100:.1f}%")
    print(f"  Profit Factor  : {r.profit_factor:.2f}  {ok(r.profit_factor, 1.2)}")
    print(f"  Avg R:R        : {r.avg_rr:.2f}")
    print(f"  Total Trades   : {r.total_trades}")
    print(f"  Final Equity   : ${r.final_equity:,.2f}")
    print(f"  Fitness Score  : {r.fitness():.4f}")


def run_quick_test(data_path=None):
    print("=" * 58)
    print("  🧬  EvolutionaryQuant — NQ Alpha Quick Test")
    print("=" * 58)

    # Load data
    print("\n📥 Veri yükleniyor...")
    df = _load_data(data_path)

    # Test 1: DEFAULT params
    print("\n⚙️  Test 1: DEFAULT params")
    params_def = DEFAULT_PARAMS
    r_def = backtest_from_raw(df, params_def)
    print_result(r_def, "DEFAULT PARAMS")

    # Test 2: Conservative tuned params
    print("\n⚙️  Test 2: Tuned Conservative params")
    params_con = NQAlphaParams(
        orb_breakout_mult   = 0.50,
        orb_rejection_pct   = 0.40,
        momentum_roc_period = 8,
        momentum_rsi_period = 14,
        momentum_rsi_thresh = 58.0,
        kalman_r            = 0.005,
        kalman_q            = 0.0005,
        hurst_window        = 80,
        fvg_min_size        = 0.80,
        ob_lookback         = 12,
        pivot_touch_atr     = 0.30,
        gap_min_pct         = 0.10,
        tp_atr_mult         = 3.00,
        sl_atr_mult         = 1.00,
        min_score           = 3,
    )
    r_con = backtest_from_raw(df, params_con)
    print_result(r_con, "TUNED CONSERVATIVE")

    # Test 3: Aggressive params
    print("\n⚙️  Test 3: Aggressive params")
    params_agg = NQAlphaParams(
        orb_breakout_mult   = 0.30,
        orb_rejection_pct   = 0.25,
        momentum_roc_period = 5,
        momentum_rsi_period = 10,
        momentum_rsi_thresh = 55.0,
        kalman_r            = 0.002,
        kalman_q            = 0.0002,
        hurst_window        = 60,
        fvg_min_size        = 0.50,
        ob_lookback         = 8,
        pivot_touch_atr     = 0.20,
        gap_min_pct         = 0.08,
        tp_atr_mult         = 2.00,
        sl_atr_mult         = 0.70,
        min_score           = 2,
    )
    r_agg = backtest_from_raw(df, params_agg)
    print_result(r_agg, "AGGRESSIVE")

    # Comparison table
    print(f"\n{'═'*56}")
    print("  ÖZET KARŞILAŞTIRMA")
    print(f"{'═'*56}")
    print(f"  {'Metrik':<20} {'Default':>12} {'Conservative':>12} {'Aggressive':>12}")
    print(f"  {'-'*56}")
    rows = [
        ("CAGR %",        r_def.cagr*100,        r_con.cagr*100,        r_agg.cagr*100),
        ("MaxDD %",       r_def.max_drawdown*100, r_con.max_drawdown*100, r_agg.max_drawdown*100),
        ("Sharpe",        r_def.sharpe_ratio,     r_con.sharpe_ratio,    r_agg.sharpe_ratio),
        ("Win Rate %",    r_def.win_rate*100,     r_con.win_rate*100,    r_agg.win_rate*100),
        ("Prof. Factor",  r_def.profit_factor,    r_con.profit_factor,   r_agg.profit_factor),
        ("Trades",        r_def.total_trades,     r_con.total_trades,    r_agg.total_trades),
        ("Fitness",       r_def.fitness(),        r_con.fitness(),       r_agg.fitness()),
    ]
    for name, d, c, a in rows:
        print(f"  {name:<20} {d:>12.2f} {c:>12.2f} {a:>12.2f}")
    print(f"{'═'*56}")

    best_f = max(r_def.fitness(), r_con.fitness(), r_agg.fitness())
    print(f"\n  🏆 En iyi fitness: {best_f:.4f}")
    if best_f > 1.0:
        print("  ✅ Strateji mantıklı görünüyor — DEAP ile optimize etmeye hazır!")
    else:
        print("  ⚠️  Düşük fitness — data kalitesini veya timeframe'i kontrol et.")

    return r_def, r_con, r_agg


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default=None)
    args = p.parse_args()
    run_quick_test(args.data)
