"""
============================================================
LiquidityEdge XAU/USD — Ana çalıştırıcı
============================================================
Kullanım:
    python run_optimizer.py --data XAUUSD_1h.csv
    python run_optimizer.py --data cache/ --tf 1h --generations 200 --capital 1000
    python run_optimizer.py --data XAUUSD_1h.csv --xgb  # XGBoost filtresi de eğit
============================================================
"""

import os
import sys
import json
import argparse
import multiprocessing
from datetime import datetime

import numpy as np
import pandas as pd

# Modülleri yükle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_engine  import (decode, run_backtest, DEFAULT_PARAMS,
                               add_liquidity_features, LiquidityXGBFilter)
from deap_optimizer   import run_deap


# ──────────────────────────────────────────────────────────
# Veri yükleme
# ──────────────────────────────────────────────────────────

def load_data(path: str, tf: str = '1h') -> pd.DataFrame:
    """CSV dosyasından veya cache klasöründen veri yükle."""
    if os.path.isfile(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    elif os.path.isdir(path):
        for fname in [f'XAUUSD_{tf}.csv', f'xauusd_{tf}.csv',
                      f'XAUUSD_{tf.upper()}.csv']:
            fp = os.path.join(path, fname)
            if os.path.exists(fp):
                df = pd.read_csv(fp, index_col=0, parse_dates=True)
                break
        else:
            raise FileNotFoundError(
                f"Cache klasöründe XAUUSD_{tf}.csv bulunamadı: {path}")
    else:
        raise FileNotFoundError(f"Veri dosyası/klasörü bulunamadı: {path}")

    df.columns = df.columns.str.lower()
    required   = ['open', 'high', 'low', 'close', 'volume']
    missing    = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    print(f"📊 Veri: {len(df):,} bar  "
          f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'spectral_bias_engine'))
        from fft_bias import add_spectral_features
        from hmm_regime import add_regime_features
        from adaptive_meta_labeler import apply_adaptive_meta_labels
        
        print("🎧 Spectral Regime Engine (Meta-Bias) özellikleri hesaplanıyor, lütfen bekleyin...")
        df = add_spectral_features(df, window_size=60)
        df = add_regime_features(df, lookback=500)
        df = apply_adaptive_meta_labels(df)
        print("🎧 Spectral özellikler Dataframe'e eklendi!")
    except Exception as e:
        print(f"⚠️ Spectral Bias özellikleri hesaplanamadı ({e}), bu modül devre dışı bırakıldı.")
        df['meta_bias'] = 0.0

    return df


# ──────────────────────────────────────────────────────────
# Rapor
# ──────────────────────────────────────────────────────────

def print_report(r, label="", initial_capital=1_000, leverage=30):
    """Backtest sonuçlarını formatlı yaz."""
    lev_ret = r.total_return * leverage
    print(f"\n{'═'*64}")
    print(f"📈 {label} — BACKTEST SONUÇLARI")
    print(f"{'═'*64}")
    print(f"  Başlangıç Sermaye : ${initial_capital:,.0f}")
    print(f"  Kaldıraç          : 1:{leverage:.0f}")
    print(f"  {'─'*40}")
    print(f"  CAGR              : %{r.cagr*100:+.2f}")
    print(f"  Toplam Getiri     : %{r.total_return*100:+.2f}")
    print(f"  Lev. Getiri       : %{lev_ret*100:+.2f}")
    print(f"  Max Drawdown      : %{r.max_drawdown*100:.2f}")
    print(f"  Sharpe Ratio      : {r.sharpe_ratio:.3f}")
    print(f"  Win Rate          : %{r.win_rate*100:.1f}")
    print(f"  Profit Factor     : {r.profit_factor:.2f}")
    print(f"  Ortalama R:R      : {r.avg_rr:.2f}")
    print(f"  Toplam İşlem      : {r.total_trades}")
    print(f"  Son Bakiye        : ${r.final_equity:,.2f}")
    print(f"  {'─'*40}")

    feq = initial_capital * (1 + lev_ret)
    if feq > initial_capital:
        print(f"  💰 ${initial_capital:,} → ${feq:,.0f}  ({feq/initial_capital:.1f}x)")
        flag = "🎯 ULAŞILDI!" if feq >= initial_capital * 10 else f"→ {feq/initial_capital:.1f}x"
        print(f"  10x Hedef         : {flag}")
    else:
        print(f"  ❌ ${initial_capital:,} → ${feq:,.0f}")
    print(f"{'═'*64}")


def compare_results(r_default, r_opt, initial_capital=1_000):
    """Default vs Optimized karşılaştırması."""
    print(f"\n{'═'*64}")
    print("📊 KARŞILAŞTIRMA: Default Params vs Optimized")
    print(f"{'═'*64}")
    print(f"  {'Metrik':<22} {'Default':>12} {'Optimized':>12} {'Delta':>8}")
    print(f"  {'─'*58}")
    metrics = [
        ("CAGR %",         r_default.cagr*100,         r_opt.cagr*100,         True),
        ("Win Rate %",     r_default.win_rate*100,      r_opt.win_rate*100,     True),
        ("Profit Factor",  r_default.profit_factor,     r_opt.profit_factor,    True),
        ("Max Drawdown %", r_default.max_drawdown*100,  r_opt.max_drawdown*100, False),
        ("Sharpe Ratio",   r_default.sharpe_ratio,      r_opt.sharpe_ratio,     True),
        ("Avg R:R",        r_default.avg_rr,            r_opt.avg_rr,           True),
        ("Total Trades",   float(r_default.total_trades), float(r_opt.total_trades), None),
        ("Final $",        r_default.final_equity,      r_opt.final_equity,     True),
    ]
    for lbl, dv, ov, higher_better in metrics:
        delta = ov - dv
        if higher_better is True:
            flag = "✅" if delta > 0 else "❌"
        elif higher_better is False:
            flag = "✅" if delta < 0 else "❌"
        else:
            flag = "ℹ️ "
        print(f"  {lbl:<22} {dv:>12.2f} {ov:>12.2f} {flag} {delta:>+7.2f}")
    print(f"{'═'*64}")


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LiquidityEdge XAU/USD DEAP Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python run_optimizer.py --data XAUUSD_1h.csv
  python run_optimizer.py --data ../institutional_engine/cache --tf 1h
  python run_optimizer.py --data XAUUSD_1h.csv --generations 300 --population 80
  python run_optimizer.py --data XAUUSD_1h.csv --xgb  # XGBoost filtresi de eğit
        """)
    parser.add_argument('--data',        type=str,   required=True,
                        help='CSV dosyası veya cache klasörü')
    parser.add_argument('--tf',          type=str,   default='1h',
                        help='Timeframe (default: 1h)')
    parser.add_argument('--capital',     type=float, default=1_000.0,
                        help='Başlangıç sermayesi $ (default: 1000)')
    parser.add_argument('--leverage',    type=float, default=30.0,
                        help='Kaldıraç (default: 30)')
    parser.add_argument('--generations', type=int,   default=200,
                        help='DEAP nesil sayısı (default: 200)')
    parser.add_argument('--population',  type=int,   default=70,
                        help='Popülasyon (default: 70)')
    parser.add_argument('--splits',      type=int,   default=5,
                        help='Walk-forward split sayısı (default: 5)')
    parser.add_argument('--jobs',        type=int,   default=None,
                        help='CPU çekirdek (default: auto)')
    parser.add_argument('--xgb',         action='store_true',
                        help='XGBoost sinyal filtresi de eğit')
    parser.add_argument('--top',         type=int,   default=5,
                        help='Top-N göster (default: 5)')
    args = parser.parse_args()

    print("\n" + "💧" * 32)
    print("  LIQUIDITYEDGE XAU/USD DEAP OPTIMIZER")
    print("💧" * 32 + "\n")

    # Veri
    df = load_data(args.data, args.tf)

    # DEAP config
    n_jobs = args.jobs or min(11, multiprocessing.cpu_count())
    config = {
        'population_size':   args.population,
        'n_generations':     args.generations,
        'hall_of_fame_size': 20,
        'crossover_prob':    0.70,
        'mutation_prob':     0.25,
        'tournament_size':   5,
        'initial_capital':   args.capital,
        'train_pct':         0.70,
        'n_splits':          args.splits,
        'stagnation_limit':  25,
        'n_jobs':            n_jobs,
        'verbose':           True,
        'log_every':         10,
    }

    # Optimize
    best_params, history = run_deap(df, config)

    # Sonuçları backtest et
    r_opt     = run_backtest(df, best_params, args.capital)
    r_default = run_backtest(df, DEFAULT_PARAMS, args.capital)

    print_report(r_default, "DEFAULT PARAMS",  args.capital, args.leverage)
    print_report(r_opt,     "OPTİMİZE EDİLMİŞ", args.capital, args.leverage)
    compare_results(r_default, r_opt, args.capital)

    # Top-N
    print(f"\n🏆 TOP-{args.top}")
    print("-" * 64)
    for i, ind in enumerate(list(history['hof'])[:args.top]):
        p = decode(list(ind))
        print(f"#{i+1:2d}  Fit:{ind.fitness.values[0]:7.4f}  "
              f"ob={p['ob_lookback']:2d}  "
              f"tp={p['tp_atr_mult']:.1f}x  "
              f"sl={p['sl_atr_mult']:.2f}x  "
              f"lot={p['lot_size']:.3f}  "
              f"ema={p['ema_trend_period']:2d}  "
              f"sweep={p['sweep_margin']:.2f}")

    # XGBoost filtresi (opsiyonel)
    xgb_acc = None
    if args.xgb:
        print("\n🤖 XGBoost Sinyal Filtresi Eğitimi...")
        xgb_filter = LiquidityXGBFilter()
        xgb_acc    = xgb_filter.train(df, best_params)
        if xgb_acc:
            print(f"   ✅ XGB Accuracy: {xgb_acc:.1%}")

    # Kaydet
    os.makedirs("outputs", exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f"outputs/liquidity_edge_{ts}.json"
    save_data = {
        'timestamp':    ts,
        'fitness':      float(hof_fitness := history['hof'][0].fitness.values[0]),
        'best_params':  best_params,
        'backtest': {
            'total_return':  r_opt.total_return,
            'cagr':          r_opt.cagr,
            'max_drawdown':  r_opt.max_drawdown,
            'sharpe_ratio':  r_opt.sharpe_ratio,
            'win_rate':      r_opt.win_rate,
            'profit_factor': r_opt.profit_factor,
            'avg_rr':        r_opt.avg_rr,
            'total_trades':  r_opt.total_trades,
            'final_equity':  r_opt.final_equity,
        },
        'default_backtest': {
            'total_return':  r_default.total_return,
            'profit_factor': r_default.profit_factor,
            'win_rate':      r_default.win_rate,
        },
        'xgb_accuracy': xgb_acc,
        'config':       {k: v for k, v in config.items() if k != 'hof'},
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\n✅ Sonuç kaydedildi: {path}")

    return best_params, history


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
