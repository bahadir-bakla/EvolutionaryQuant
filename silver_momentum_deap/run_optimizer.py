"""
Silver Momentum — Main Runner
==============================
EvolutionaryQuant | XAGUSD Pure Momentum DEAP Optimizer

Kullanım:
    # HistData zip/csv ile:
    python silver_momentum_deap/run_optimizer.py --data C:/veri/XAGUSD_2023.zip --tf 5m

    # Klasör dolu zip ile:
    python silver_momentum_deap/run_optimizer.py --data C:/veri/XAGUSD/ --tf 15m

    # Kelly+GARCH sizing ile:
    python silver_momentum_deap/run_optimizer.py --data ... --kelly_garch

    # Hızlı test:
    python silver_momentum_deap/run_optimizer.py --data ... --quick
"""

import sys, os, argparse, multiprocessing
import numpy as np
import pandas as pd

_DIR    = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
sys.path.insert(0, _DIR)
sys.path.insert(0, _PARENT)

from data_loader      import load_histdata, load_histdata_folder, resample_to_tf, load_yfinance_silver
from silver_strategy  import SilverMomentumParams, add_silver_features, decode_genome, DEFAULT_PARAMS, GENOME_SIZE
from backtest_engine  import run_backtest, backtest_from_raw, SilverBacktestResult, INITIAL_CAPITAL
from deap_optimizer   import run_deap_optimizer, build_wf_splits


def _load_data(data_path: str = None, tf: str = '5m') -> pd.DataFrame:
    """Load silver data from HistData or yfinance fallback."""
    tf_str = {'5m': '5min', '15m': '15min', '1h': '1h'}.get(tf, tf)

    if data_path:
        path = os.path.expanduser(data_path)
        if os.path.isdir(path):
            print(f"📁 Klasör yükleniyor: {path}")
            df_m1 = load_histdata_folder(path)
        else:
            print(f"📥 Dosya yükleniyor: {os.path.basename(path)}")
            df_m1 = load_histdata(path)
        print(f"🔄 Resample: M1 → {tf}")
        df = resample_to_tf(df_m1, tf_str)
    else:
        print("📡 YFinance silver data çekiliyor (fallback)...")
        df_raw = load_yfinance_silver(tf=tf, lookback_days=59)
        if df_raw.empty:
            raise ValueError("Veri yüklenemedi! --data ile HistData dosyası belirtin.")
        df = df_raw

    print(f"✅ Final: {len(df):,} {tf} bars | Fiyat: {df['close'].min():.3f} – {df['close'].max():.3f}")
    try:
        sys.path.insert(0, os.path.join(_PARENT, 'spectral_bias_engine'))
        from fft_bias import add_spectral_features
        from hmm_regime import add_regime_features
        from adaptive_meta_labeler import apply_adaptive_meta_labels
        
        print("🎧 Spectral Regime Modülü (Meta-Bias) Dataframe'e ekleniyor...")
        df = add_spectral_features(df, int(800 / int(tf.replace('m','')))) if 'm' in tf else add_spectral_features(df, window_size=60)
        df = add_regime_features(df, lookback=500)
        df = apply_adaptive_meta_labels(df)
        print("🎧 Spectral Özellikler eklendi!")
    except Exception as e:
        print(f"⚠️ Spectral Bias Engine hata verdi ({e}), bu modül devre dışı.")
        df['meta_bias'] = 0.0

    return df

def _print_result(r: SilverBacktestResult, label: str):
    print(f"\n{'═'*58}")
    print(f"  {label}")
    print(f"{'═'*58}")
    ok = lambda a, b: "✅" if a > b else "❌"
    print(f"  CAGR           : {r.cagr*100:+.2f}%  {ok(r.cagr, 0.30)}")
    print(f"  Total Return   : {r.total_return*100:+.2f}%")
    print(f"  Max Drawdown   : {r.max_drawdown*100:.2f}%  {ok(0.25, r.max_drawdown)}")
    print(f"  Sharpe Ratio   : {r.sharpe_ratio:.2f}  {ok(r.sharpe_ratio, 1.0)}")
    print(f"  Win Rate       : {r.win_rate*100:.1f}%")
    print(f"  Profit Factor  : {r.profit_factor:.2f}  {ok(r.profit_factor, 1.2)}")
    print(f"  Avg R:R        : {r.avg_rr:.2f}")
    print(f"  Total Trades   : {r.total_trades}")
    print(f"  Final Equity   : ${r.final_equity:,.2f}")
    print(f"  Fitness        : {r.fitness():.4f}")


def _print_wf(df_raw, params, n_splits, init_cap, use_kg):
    splits = build_wf_splits(df_raw, n_splits)
    print(f"\n📊 Walk-Forward OOS ({n_splits} splits)")
    print("─" * 58)
    pos = 0
    for i, raw in enumerate(splits):
        try:
            r   = backtest_from_raw(raw, params, init_cap, use_kg)
            flg = "✅" if r.total_return > 0 else "❌"
            if r.total_return > 0: pos += 1
            print(f"  Split {i+1} [{raw.index[0].date()}→{raw.index[-1].date()}]: "
                  f"Ret={r.total_return*100:+.1f}% | PF={r.profit_factor:.2f} | "
                  f"T={r.total_trades}  {flg}")
        except Exception as e:
            print(f"  Split {i+1}: ERROR {e}  ❌")
    pct = pos / max(len(splits), 1) * 100
    print("─" * 58)
    print(f"  Pozitif: {pos}/{len(splits)}  ({pct:.0f}%)  "
          f"{'✅ HAZIR' if pct >= 60 else '⚠️  Daha optimize gerekebilir'}")


def main():
    parser = argparse.ArgumentParser(description='Silver Momentum DEAP — EvolutionaryQuant')
    parser.add_argument('--data',       type=str,   default=None)
    parser.add_argument('--tf',         type=str,   default='5m', choices=['5m','15m','1h'])
    parser.add_argument('--capital',    type=float, default=INITIAL_CAPITAL)
    parser.add_argument('--pop',        type=int,   default=80)
    parser.add_argument('--gen',        type=int,   default=200)
    parser.add_argument('--splits',     type=int,   default=5)
    parser.add_argument('--jobs',       type=int,   default=None)
    parser.add_argument('--kelly_garch',action='store_true', help='Kelly+GARCH sizing aktif')
    parser.add_argument('--quick',      action='store_true', help='Hizli test')
    args = parser.parse_args()

    print("=" * 62)
    print("  🥈  EvolutionaryQuant — Silver Momentum DEAP")
    print("=" * 62)
    print(f"  Strategy: ROC stack | EMA alignment | MACD | ADX")
    print(f"            Breakout | Volume surge | Session filter")
    print(f"  Sizing  : {'Kelly + GARCH(1,1)' if args.kelly_garch else 'ATR-based (1.5% risk)'}")
    print(f"  Genome  : {GENOME_SIZE} parameters")
    print("=" * 62)

    if args.quick:
        args.pop = 30; args.gen = 20; args.splits = 3
        print("⚡ Quick mode (pop=30, gen=20, splits=3)\n")

    # 1. Load data
    print("\n📥 Veri yükleniyor...")
    df_raw = _load_data(args.data, args.tf)

    # 2. Baseline
    print("\n🔍 DEFAULT params baseline...")
    try:
        r_def = backtest_from_raw(df_raw, DEFAULT_PARAMS, args.capital, args.kelly_garch)
        _print_result(r_def, "DEFAULT PARAMS (Baseline)")
    except Exception as e:
        print(f"  Baseline hatası: {e}")

    # 3. DEAP
    print(f"\n🚀 DEAP Optimizasyon Başlıyor...")
    best_params, best_fit, stats, hof = run_deap_optimizer(
        df_raw          = df_raw,
        initial_capital = args.capital,
        population      = args.pop,
        generations     = args.gen,
        n_splits        = args.splits,
        n_jobs          = args.jobs,
        use_kelly_garch = args.kelly_garch,
        verbose         = True,
        output_dir      = os.path.join(_DIR, 'outputs'),
    )

    # 4. Full backtest comparison
    print("\n🏆 Sonuçlar:")
    try:
        r_best = backtest_from_raw(df_raw, best_params, args.capital, False)
        _print_result(r_best, "OPTİMİZE — ATR Sizing")
    except Exception as e:
        print(f"  Backtest hatası: {e}")

    if args.kelly_garch:
        try:
            r_kg = backtest_from_raw(df_raw, best_params, args.capital, True)
            _print_result(r_kg, "OPTİMİZE — Kelly+GARCH Sizing")
        except Exception as e:
            print(f"  Kelly+GARCH backtest hatası: {e}")

    # 5. Walk-forward
    _print_wf(df_raw, best_params, args.splits, args.capital, args.kelly_garch)

    # 6. Top 5
    print("\n🏆 TOP 5 GENOME")
    print("─" * 60)
    for i, ind in enumerate(list(hof)[:5]):
        p = decode_genome(list(ind))
        print(f"  #{i+1}  Fit:{ind.fitness.values[0]:7.3f} | "
              f"ROC={p.roc_fast_period}/{p.roc_mid_period}/{p.roc_slow_period} | "
              f"ADX>={p.adx_threshold:.0f} | TP={p.tp_atr_mult:.1f}x | "
              f"SL={p.sl_atr_mult:.1f}x | Score>={p.min_score}")
    print("\n✅ Tamamlandi!")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
