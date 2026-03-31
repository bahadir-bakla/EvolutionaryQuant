"""
=============================================================
run_holy_trinity_deap.py — ANA ÇALIŞTIRICI
=============================================================
python run_holy_trinity_deap.py
=============================================================
"""

import os, sys, json, multiprocessing
import pandas as pd
import numpy as np
from datetime import datetime
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ht = importlib.import_module("holy_trinity_deap")


def load_cache(cache_dir, tf):
    path = os.path.join(cache_dir, f"XAUUSD_{tf}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cache yok: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def fetch_nq_from_cache_or_yf(cache_dir, tf="1h"):
    """NQ verisini cache'den veya yfinance'tan çek"""
    nq_path = os.path.join(cache_dir, f"NQ_{tf}.csv")

    if os.path.exists(nq_path):
        print(f"   ⚡ NQ cache bulundu: {nq_path}")
        return pd.read_csv(nq_path, index_col=0, parse_dates=True)

    print("   📥 NQ verisi yfinance'tan çekiliyor...")
    try:
        import yfinance as yf
        ticker = yf.Ticker("NQ=F")
        df = ticker.history(period="730d", interval="1h")
        if df.empty:
            raise ValueError("NQ verisi boş")
        df.columns = df.columns.str.lower()
        df.index.name = 'timestamp'
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df = ht.add_nq_features(df)
        df.to_csv(nq_path)
        print(f"   ✅ NQ: {len(df):,} bar kaydedildi → {nq_path}")
        return df
    except Exception as e:
        print(f"   ⚠️  NQ çekilemedi: {e}")
        print("   📋 Gold verisi NQ proxy olarak kullanılacak")
        return None


def align_dataframes(gold_df, nq_df):
    """İki dataframe'i ortak index'e göre hizala"""
    if nq_df is None:
        return gold_df, gold_df  # NQ yoksa Gold'u proxy olarak kullan
    common = gold_df.index.intersection(nq_df.index)
    if len(common) < 100:
        print(f"   ⚠️  Ortak bar az ({len(common)}), Gold proxy kullanılıyor")
        return gold_df, gold_df
    return gold_df.loc[common], nq_df.loc[common]


def full_report(gold_df, nq_df, params, capital=1_000, leverage=30.0):
    r = ht.run_holy_trinity(gold_df, nq_df, params, capital)
    lret = r.total_return * leverage
    feq  = capital * (1 + lret)

    print("\n" + "═"*62)
    print("🏆 HOLY TRINITY V7 — BACKTEST RAPORU")
    print("═"*62)
    print(f"  Başlangıç        : ${capital:,.0f}")
    print(f"  Kaldıraç         : 1:{leverage:.0f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  CAGR             : %{r.cagr*100:.2f}")
    print(f"  Toplam Getiri    : %{r.total_return*100:.2f}")
    print(f"  Kaldıraçlı Ret  : %{lret*100:.2f}")
    print(f"  Max Drawdown     : %{r.max_drawdown*100:.2f}")
    print(f"  Sharpe Ratio     : {r.sharpe_ratio:.3f}")
    print(f"  Win Rate         : %{r.win_rate*100:.1f}")
    print(f"  Profit Factor    : {r.profit_factor:.2f}")
    print(f"  Toplam İşlem     : {r.total_trades}")
    print(f"  Fitness          : {r.fitness():.4f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Gold PnL         : ${r.gold_pnl:,.2f}")
    print(f"  NQ PnL           : ${r.nq_pnl:,.2f}")
    print(f"  ─────────────────────────────────────────")
    if feq > capital:
        print(f"  💰 ${capital:,.0f} → ${feq:,.0f}  ({feq/capital:.1f}x)")
        if r.cagr > 0:
            daily = (1 + r.cagr * leverage) ** (1/365) - 1
            eq60  = capital * (1 + daily)**60
            print(f"  📅 60 gün proj.  : ${eq60:,.0f}")
        flag = "✅ ULAŞILDI!" if feq >= capital*10 else f"Şu an {feq/capital:.1f}x (hedef 10x)"
        print(f"  🎯 10x hedef     : {flag}")
    else:
        print(f"  ❌ ${capital:,.0f} → ${feq:,.0f}")
    print("═"*62)
    return r


def print_params(params):
    print("\n📋 OPTİMAL PARAMETRELER")
    print("-"*45)
    sections = {
        'Gold Sniper': ['gs_lot','gs_hard_stop','gs_target_pts','gs_stop_pts','gs_stale_hours','gs_point_value'],
        'NQ Basket':   ['nq_lot','nq_tp_pts','nq_stop_pts'],
        'Escape Vel.': ['safe_threshold','growth_factor'],
        'Risk Mgmt':   ['dd_threshold','cooldown_bars'],
    }
    for section, keys in sections.items():
        print(f"\n  {section}:")
        for k in keys:
            if k in params:
                print(f"    {k:<20} : {params[k]}")


def main():
    print("🏆 HOLY TRINITY V7 — DEAP OPTİMİZASYONU\n")

    # ══════════════════════════════════════════════════════
    # ⚙️  AYARLAR
    # ══════════════════════════════════════════════════════
    CACHE_DIR  = "C:/Users/9bakl/OneDrive/Masaüstü/kalman/institutional_engine/cache"
    CAPITAL    = 1_000.0
    LEVERAGE   = 30.0
    # ══════════════════════════════════════════════════════

    # 1. Gold verisi (cache'den)
    print("📂 Veri yükleniyor...")
    gold_df = load_cache(CACHE_DIR, "1h")
    gold_df = ht.add_gold_sniper_features(gold_df)
    print(f"   ✅ Gold: {len(gold_df):,} bar")

    # 2. NQ verisi
    nq_df_raw = fetch_nq_from_cache_or_yf(CACHE_DIR, "1h")
    if nq_df_raw is not None and 'vwap' not in nq_df_raw.columns:
        nq_df_raw = ht.add_nq_features(nq_df_raw)

    gold_aligned, nq_aligned = align_dataframes(gold_df, nq_df_raw)
    print(f"   ✅ Ortak: {len(gold_aligned):,} bar\n")

    # 3. DEAP config
    config = {
        'population_size':   60,
        'n_generations':     200,
        'hall_of_fame_size': 15,
        'crossover_prob':    0.70,
        'mutation_prob':     0.25,
        'tournament_size':   5,
        'initial_capital':   CAPITAL,
        'train_pct':         0.70,
        'n_splits':          5,
        'stagnation_limit':  20,
        'n_jobs':            11,
        'verbose':           True,
        'log_every':         10,
    }

    # 4. Optimize et
    optimizer = ht.HolyTrinityDEAP(gold_aligned, nq_aligned, config)
    best_params, history = optimizer.run()

    # 5. Tam rapor
    full_report(gold_aligned, nq_aligned, best_params, CAPITAL, LEVERAGE)
    print_params(best_params)

    # 6. Karşılaştır: V7 varsayılan vs optimize
    print("\n📊 V7 DEFAULT vs OPTİMİZE KARŞILAŞTIRMA")
    print("-"*62)
    v7_default = {
        'gs_lot': 0.02, 'gs_hard_stop': 500, 'gs_target_pts': 140,
        'gs_stop_pts': 12, 'gs_stale_hours': 72, 'nq_lot': 0.03,
        'nq_tp_pts': 40, 'nq_stop_pts': 150, 'safe_threshold': 3000,
        'growth_factor': 0.40, 'dd_threshold': 0.30, 'cooldown_bars': 48,
        'gs_point_value': 100,
    }
    r_def = ht.run_holy_trinity(gold_aligned, nq_aligned, v7_default, CAPITAL)
    r_opt = ht.run_holy_trinity(gold_aligned, nq_aligned, best_params, CAPITAL)

    print(f"  {'Metrik':<20} {'V7 Default':>15} {'Optimize':>15} {'Fark':>10}")
    print(f"  {'-'*60}")
    for label, d, o in [
        ("CAGR %",     r_def.cagr*100,          r_opt.cagr*100),
        ("Total Ret %", r_def.total_return*100,  r_opt.total_return*100),
        ("MaxDD %",     r_def.max_drawdown*100,  r_opt.max_drawdown*100),
        ("Win Rate %",  r_def.win_rate*100,       r_opt.win_rate*100),
        ("Profit Factor", r_def.profit_factor,   r_opt.profit_factor),
        ("Trades",      r_def.total_trades,       r_opt.total_trades),
        ("Fitness",     r_def.fitness(),          r_opt.fitness()),
    ]:
        diff = o - d
        flag = "✅" if diff > 0 else "❌"
        print(f"  {label:<20} {d:>15.2f} {o:>15.2f} {flag}{diff:>8.2f}")

    # 7. Kaydet
    os.makedirs("outputs", exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = f"outputs/holy_trinity_v7_opt_{ts}.json"
    with open(path, 'w') as f:
        json.dump({
            'timestamp':     ts,
            'fitness':       history['hof'][0].fitness.values[0],
            'best_params':   best_params,
            'v7_default':    v7_default,
        }, f, indent=2)
    print(f"\n✅ Kaydedildi: {path}")

    # 8. Top 5
    print("\n🏆 TOP 5 PARAMETRE SETİ")
    print("-"*62)
    for i, ind in enumerate(list(history['hof'])[:5]):
        p = ht.decode(list(ind))
        print(f"#{i+1}  Fit:{ind.fitness.values[0]:.4f}  "
              f"gs_lot={p['gs_lot']:.3f}  "
              f"gs_tp={p['gs_target_pts']:.0f}pts  "
              f"gs_sl={p['gs_stop_pts']:.0f}pts  "
              f"nq_tp={p['nq_tp_pts']:.0f}pts  "
              f"safe=${p['safe_threshold']:.0f}")

    return best_params, history


if __name__ == '__main__':
    multiprocessing.freeze_support()
    best_params, history = main()