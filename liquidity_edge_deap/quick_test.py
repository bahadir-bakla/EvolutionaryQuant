"""
LiquidityEdge — Hızlı Test & Default Sonuçlar
================================================
DEAP çalıştırmadan önce sistemin çalışıp çalışmadığını
ve mantıklı sonuçlar üretip üretmediğini test eder.

Kullanım:
    python quick_test.py

~30 saniye içinde sonuç çıkar.
"""

import sys, os
import pandas as pd
import numpy as np

# Modül yolu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_engine import (
    run_backtest, add_liquidity_features,
    DEFAULT_PARAMS, decode, GENE_BOUNDS, GENOME_NAMES
)

# ──────────────────────────────────────────────────────────
# Veri yükle (institutional_engine cache)
# ──────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'institutional_engine', 'cache'
)

print("=" * 64)
print("💧 LIQUIDITYEDGE — HIZLI TEST")
print("=" * 64)

csv_path = os.path.join(CACHE_DIR, 'XAUUSD_1h.csv')
if not os.path.exists(csv_path):
    # Fallback: üst dizindeki DAT_MT CSV
    parent = os.path.dirname(os.path.abspath(__file__)) + '/..'
    for yr in ['2024', '2023', '2022']:
        p = os.path.join(parent, f'DAT_MT_XAUUSD_M1_{yr}.csv')
        if os.path.exists(p):
            csv_path = p
            break

print(f"\n📂 Veri: {csv_path}")
df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
df.columns = df.columns.str.lower()

# Kolon normalizasyonu
if 'vol' in df.columns and 'volume' not in df.columns:
    df['volume'] = df['vol']
if 'volume' not in df.columns:
    df['volume'] = 1000

# M1 ise 1h'ye resample et
try:
    freq_sec = (df.index[1] - df.index[0]).total_seconds()
    if freq_sec < 3600:
        print(f"  → M1 veri tespit edildi, 1H'ye resample ediliyor...")
        df = df.resample('1h').agg({
            'open':   'first',
            'high':   'max',
            'low':    'min',
            'close':  'last',
            'volume': 'sum'
        }).dropna()
except:
    pass

df.dropna(subset=['open','high','low','close'], inplace=True)
print(f"  → {len(df):,} bar  {df.index[0].date()} → {df.index[-1].date()}")

# ──────────────────────────────────────────────────────────
# TEST 1: Feature engineering
# ──────────────────────────────────────────────────────────
print("\n[1/4] Feature engineering test...")
try:
    df2 = add_liquidity_features(df, DEFAULT_PARAMS)
    print(f"  ✅ {len(df2.columns)} feature üretildi")
    
    # Sinyal istatistikleri
    n = len(df2)
    print(f"\n  📊 Sinyal Özeti (toplam {n:,} bar):")
    for col, label in [
        ('ob_bull',         'Bullish OB         '),
        ('ob_bear',         'Bearish OB         '),
        ('sweep_low_bull',  'Liq. Sweep Bull    '),
        ('sweep_high_bear', 'Liq. Sweep Bear    '),
        ('fvg_bull',        'FVG Bullish        '),
        ('fvg_bear',        'FVG Bearish        '),
        ('trend_up',        'Trend UP (EMA stack)'),
        ('trend_dn',        'Trend DN (EMA stack)'),
        ('rsi_bull_div',    'RSI Bull Divergence'),
        ('rsi_bear_div',    'RSI Bear Divergence'),
    ]:
        if col in df2.columns:
            cnt  = int(df2[col].sum())
            pct  = cnt / n * 100
            bar  = '█' * int(pct * 2) if pct < 50 else '█' * 50
            print(f"    {label}: {cnt:5d} ({pct:5.2f}%)  {bar}")
except Exception as e:
    print(f"  ❌ HATA: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────────────────
# TEST 2: Default parametreler ile backtest
# ──────────────────────────────────────────────────────────
print("\n[2/4] Default parametreler ile backtest...")
r_def = run_backtest(df, DEFAULT_PARAMS, initial_capital=1_000.0, verbose=False)

print(f"\n  📈 DEFAULT PARAMS SONUÇLARI")
print(f"  {'─'*40}")
print(f"  CAGR           : %{r_def.cagr*100:+.2f}")
print(f"  Toplam Getiri  : %{r_def.total_return*100:+.2f}")
print(f"  Max Drawdown   : %{r_def.max_drawdown*100:.2f}")
print(f"  Sharpe Ratio   : {r_def.sharpe_ratio:.3f}")
print(f"  Win Rate       : %{r_def.win_rate*100:.1f}")
print(f"  Profit Factor  : {r_def.profit_factor:.2f}")
print(f"  Avg R:R        : {r_def.avg_rr:.2f}")
print(f"  Toplam İşlem   : {r_def.total_trades}")
print(f"  Fitness Skoru  : {r_def.fitness():.4f}")
print(f"  Son Bakiye     : ${r_def.final_equity:,.2f}")

# ──────────────────────────────────────────────────────────
# TEST 3: Manuel birkaç parametre deneme (quick sweep)
# ──────────────────────────────────────────────────────────
print("\n[3/4] Hızlı parametre tarama (10 rastgele genome)...")

import random
random.seed(42)
best_fit   = r_def.fitness()
best_params = DEFAULT_PARAMS.copy()
results = []

for trial in range(10):
    genome = [random.uniform(0, 1) for _ in range(len(GENE_BOUNDS))]
    p = decode(genome)
    r = run_backtest(df, p, initial_capital=1_000.0)
    f = r.fitness()
    results.append((f, p, r))
    marker = " ✅" if f > best_fit else ""
    print(f"  Trial {trial+1:2d}: Fit={f:7.3f} | "
          f"CAGR={r.cagr*100:+5.1f}% | "
          f"PF={r.profit_factor:.2f} | "
          f"WR={r.win_rate*100:.0f}% | "
          f"Trades={r.total_trades:3d}{marker}")
    if f > best_fit:
        best_fit    = f
        best_params = p

print(f"\n  🏆 En iyi rastgele fitness: {best_fit:.4f}")
results.sort(key=lambda x: x[0], reverse=True)

# ──────────────────────────────────────────────────────────
# TEST 4: Veri bölümü testi (walk-forward sanity)
# ──────────────────────────────────────────────────────────
print("\n[4/4] Walk-forward sanity check...")
n_splits = 4
sz = len(df) // n_splits
split_results = []
for i in range(n_splits):
    s = i * sz
    e = min(s + sz, len(df))
    df_split = df.iloc[s:e]
    r = run_backtest(df_split, DEFAULT_PARAMS, initial_capital=1_000.0)
    split_results.append(r)
    flag = "✅" if r.total_return > 0 else "❌"
    print(f"  Split {i+1} ({df_split.index[0].date()} → {df_split.index[-1].date()}): "
          f"Getiri=%{r.total_return*100:+.1f} | "
          f"PF={r.profit_factor:.2f} | "
          f"Trades={r.total_trades} {flag}")

pos_splits = sum(1 for r in split_results if r.total_return > 0)
print(f"\n  {'─'*50}")
print(f"  Pozitif split: {pos_splits}/{n_splits}")
if pos_splits >= 3:
    print("  ✅ Sistem tutarlı — DEAP'e hazır!")
elif pos_splits >= 2:
    print("  ⚠️  Kısmi tutarlılık — DEAP geliştirebilir")
else:
    print("  ❌ Dikkat: Default params zayıf — DEAP kritik!")

# ──────────────────────────────────────────────────────────
# ÖZET
# ──────────────────────────────────────────────────────────
print("\n" + "═" * 64)
print("📋 ÖZET")
print("═" * 64)
print(f"  Sistem           : LiquidityEdge XAU/USD v1.0")
print(f"  Veri             : {len(df):,} bar 1H XAUUSD")
print(f"  Default Fitness  : {r_def.fitness():.4f}")
print(f"  En İyi Random    : {best_fit:.4f}")

if r_def.fitness() > 0:
    print(f"\n  ✅ Sistem çalışıyor!")
    print(f"  → DEAP optimize etmek için:")
    print(f"     python run_optimizer.py --data ../institutional_engine/cache")
else:
    print(f"\n  ⚠️  Default params negatif fitness — sinyal koşulları çok sıkı olabilir")
    print(f"  → DEAP bunu optimize edecek. Çalıştır:")
    print(f"     python run_optimizer.py --data ../institutional_engine/cache --generations 150")

print("═" * 64)
