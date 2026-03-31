"""
GoldMaster DEAP — Optimizer Runner
===================================
XAU/USD Triple-Tap + FVG + Momentum strategy optimized via DEAP.

Usage:
    python 03_run.py --data path/to/XAUUSD_1h.csv
    python 03_run.py --data path/to/cache --tf 1h --capital 1000

Data format (CSV): datetime index, open, high, low, close, volume columns.
"""

import os, sys, json, multiprocessing, argparse
import pandas as pd
import numpy as np
import importlib
from datetime import datetime

# Klasörü path'e ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_cache(cache_dir: str, tf: str) -> pd.DataFrame:
    path = os.path.join(cache_dir, f"XAUUSD_{tf}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache yok: {path}\nÖnce institutional_engine'i çalıştır!")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def full_report(df, params, initial_capital=1_000, leverage=30.0):
    gm  = importlib.import_module("01_goldmaster_backtest")
    GMParams = gm.GMParams
    GoldMasterBacktester = gm.GoldMasterBacktester

    p   = GMParams(**params)
    bt  = GoldMasterBacktester(initial_capital)
    r   = bt.run(df, p, verbose=False)

    lret = r.total_return * leverage
    feq  = initial_capital * (1 + lret)

    print("\n" + "═"*62)
    print("📈 GOLDMASTER — BACKTEST RAPORU")
    print("═"*62)
    print(f"  Başlangıç      : ${initial_capital:,.0f}")
    print(f"  Kaldıraç       : 1:{leverage:.0f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  CAGR           : %{r.cagr*100:.2f}")
    print(f"  Toplam Getiri  : %{r.total_return*100:.2f}")
    print(f"  Lev. Getiri    : %{lret*100:.2f}")
    print(f"  Max Drawdown   : %{r.max_drawdown*100:.2f}")
    print(f"  Sharpe Ratio   : {r.sharpe_ratio:.3f}")
    print(f"  Win Rate       : %{r.win_rate*100:.1f}")
    print(f"  Profit Factor  : {r.profit_factor:.2f}")
    print(f"  Toplam İşlem   : {r.total_trades}")
    print(f"  ─────────────────────────────────────────")

    if feq > initial_capital:
        print(f"  💰 ${initial_capital:,.0f} → ${feq:,.0f}  ({feq/initial_capital:.1f}x)")
        if r.cagr > 0:
            daily = (1 + r.cagr * leverage) ** (1/365) - 1
            eq60  = initial_capital * (1 + daily) ** 60
            print(f"  📅 60 gün proj. : ${eq60:,.0f}")
        flag = "✅ ULAŞILDI!" if feq >= initial_capital * 10 else f"Şu an {feq/initial_capital:.1f}x"
        print(f"  🎯 10x hedef    : {flag}")
    else:
        print(f"  ❌ ${initial_capital:,.0f} → ${feq:,.0f}")
    print("═"*62)
    return r


def mtf_report(cache_dir, params, initial_capital=1_000, leverage=30.0):
    gm  = importlib.import_module("01_goldmaster_backtest")
    GMParams = gm.GMParams
    GoldMasterBacktester = gm.GoldMasterBacktester

    p  = GMParams(**params)
    bt = GoldMasterBacktester(initial_capital)

    print("\n🔀 MULTI-TIMEFRAME DOĞRULAMA")
    print("-"*62)
    for tf in ["1h", "4h", "1D"]:
        try:
            df_tf = load_cache(cache_dir, tf)
            r     = bt.run(df_tf, p)
            lret  = r.total_return * leverage
            flag  = "✅" if lret > 0 else "❌"
            print(f"  {flag} {tf:5s}: Getiri=%{lret*100:6.1f} | "
                  f"DD=%{r.max_drawdown*100:5.1f} | "
                  f"PF={r.profit_factor:.2f} | "
                  f"WR=%{r.win_rate*100:.0f} | "
                  f"Trades={r.total_trades}")
        except Exception as e:
            print(f"  ❌ {tf}: {e}")


def main():
    parser = argparse.ArgumentParser(description="GoldMaster DEAP Optimizer")
    parser.add_argument('--data', type=str, required=False,
                        default=None,
                        help='CSV dosyası veya cache klasörü yolu')
    parser.add_argument('--tf', type=str, default='1h', choices=['1h', '4h', '1D'],
                        help='Timeframe (default: 1h)')
    parser.add_argument('--capital', type=float, default=1_000,
                        help='Başlangıç sermayesi USD (default: 1000)')
    parser.add_argument('--leverage', type=float, default=30.0,
                        help='Kaldıraç (default: 30)')
    parser.add_argument('--generations', type=int, default=150,
                        help='DEAP nesil sayısı (default: 150)')
    parser.add_argument('--population', type=int, default=60,
                        help='Popülasyon büyüklüğü (default: 60)')
    parser.add_argument('--jobs', type=int, default=None,
                        help='CPU çekirdek sayısı (default: auto)')
    args = parser.parse_args()

    # Veri yolu belirle
    if args.data is None:
        # Varsayılan: script'in yanındaki cache klasörü
        script_dir = os.path.dirname(os.path.abspath(__file__))
        CACHE_DIR  = os.path.join(script_dir, '..', 'institutional_engine', 'cache')
        OPT_TF     = args.tf
    elif os.path.isdir(args.data):
        CACHE_DIR = args.data
        OPT_TF    = args.tf
    else:
        CACHE_DIR = None
        OPT_TF    = args.tf

    OWN_CAPITAL = args.capital
    LEVERAGE    = args.leverage

    print("🥇 GOLDMASTER DEAP OPTİMİZASYONU\n")

    # Veri yükle
    if CACHE_DIR:
        df = load_cache(CACHE_DIR, OPT_TF)
    elif args.data and os.path.isfile(args.data):
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
        df.columns = df.columns.str.lower()
    else:
        raise FileNotFoundError(
            "Veri bulunamadı!\n"
            "  python 03_run.py --data XAUUSD_1h.csv\n"
            "  python 03_run.py --data institutional_engine/cache --tf 1h"
        )

    print(f"📊 {OPT_TF}: {len(df):,} bar  "
          f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}\n")

    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'spectral_bias_engine'))
        from fft_bias import add_spectral_features
        from hmm_regime import add_regime_features
        from adaptive_meta_labeler import apply_adaptive_meta_labels
        
        print("🎧 Spectral Regime Modülü (Meta-Bias) Dataframe'e ekleniyor...")
        df = add_spectral_features(df, window_size=60)
        df = add_regime_features(df, lookback=500)
        df = apply_adaptive_meta_labels(df)
        print("🎧 Spectral Özellikler eklendi!\n")
    except Exception as e:
        print(f"⚠️ Spectral Bias Engine hata verdi ({e}), bu modül devre dışı.\n")
        df['meta_bias'] = 0.0


    # Config
    config = {
        'population_size':   60,
        'n_generations':     150,
        'hall_of_fame_size': 15,
        'crossover_prob':    0.70,
        'mutation_prob':     0.25,
        'tournament_size':   5,
        'initial_capital':   OWN_CAPITAL,
        'commission':        0.0001,
        'slippage':          0.0002,
        'train_pct':         0.70,
        'n_splits':          5,
        'stagnation_limit':  15,
        'n_jobs':            11,
        'verbose':           True,
        'log_every':         5,
    }

    # Optimize
    deap_mod = importlib.import_module("02_deap_optimizer")
    optimizer = deap_mod.GoldMasterDEAP(df, config)
    best_params, history = optimizer.run()

    # Raporlar
    full_report(df, best_params, OWN_CAPITAL, LEVERAGE)
    mtf_report(CACHE_DIR, best_params, OWN_CAPITAL, LEVERAGE)

    # Kaydet
    os.makedirs("outputs", exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f"outputs/goldmaster_params_{ts}.json"
    with open(path, 'w') as f:
        json.dump({'timestamp': ts,
                   'fitness': history['hof'][0].fitness.values[0],
                   'params': best_params}, f, indent=2)
    print(f"\n✅ Kaydedildi: {path}")

    # Top 5
    print("\n🏆 TOP 5")
    print("-"*62)
    for i, ind in enumerate(list(history['hof'])[:5]):
        p = deap_mod.decode(list(ind))
        print(f"#{i+1}  Fit:{ind.fitness.values[0]:.4f}  "
              f"Taps={p['min_taps']}  "
              f"TP={p['target_atr_mult']:.1f}x  "
              f"SL={p['stop_atr_mult']:.1f}x  "
              f"Lot={p['lot_size']:.2f}  "
              f"Growth={p['growth_factor']:.2f}  "
              f"FVG={'✅' if p['fvg_required'] else '❌'}")

    return best_params, history


if __name__ == '__main__':
    multiprocessing.freeze_support()
    best_params, history = main()
