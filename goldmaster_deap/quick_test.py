"""
GoldMaster DEAP — Hızlı Test
=================================
Optimum parametrelerle backtest yapıp sonuçları gösterir.

Kullanım:
    python quick_test.py
"""
import sys, os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib
gm = importlib.import_module("01_goldmaster_backtest")
GMParams           = gm.GMParams
GoldMasterBacktester = gm.GoldMasterBacktester
add_features       = gm.add_features

# ── Veri ──────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'institutional_engine', 'cache')
CSV_PATH  = os.path.join(CACHE_DIR, 'XAUUSD_1h.csv')

print("=" * 64)
print("🥇 GOLDMASTER DEAP — HIZLI TEST")
print("=" * 64)

df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
df.columns = df.columns.str.lower()
if 'volume' not in df.columns:
    df['volume'] = 1000
df.dropna(subset=['open','high','low','close'], inplace=True)
print(f"\n📂 {len(df):,} bar  {df.index[0].date()} → {df.index[-1].date()}")

# ── Optimum vs Default ────────────────────────────────────
import json
opt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'outputs', 'goldmaster_params_20260315_152849.json')

bt = GoldMasterBacktester(initial_capital=1_000.0)

# Default
p_default = GMParams()
r_default  = bt.run(df, p_default)

print(f"\n  📈 DEFAULT PARAMS")
print(f"  CAGR           : %{r_default.cagr*100:+.2f}")
print(f"  Toplam Getiri  : %{r_default.total_return*100:+.2f}")
print(f"  Max Drawdown   : %{r_default.max_drawdown*100:.2f}")
print(f"  Sharpe         : {r_default.sharpe_ratio:.3f}")
print(f"  Win Rate       : %{r_default.win_rate*100:.1f}")
print(f"  Profit Factor  : {r_default.profit_factor:.2f}")
print(f"  Toplam İşlem   : {r_default.total_trades}")
print(f"  Fitness        : {r_default.fitness():.4f}")
print(f"  Son Bakiye     : ${r_default.final_equity:,.2f}")

# Optimum (JSON'dan)
if os.path.exists(opt_path):
    with open(opt_path) as f:
        data = json.load(f)
    p_opt = GMParams(**data['params'])
    r_opt = bt.run(df, p_opt)

    print(f"\n  🏆 OPTİMUM PARAMS (Fitness={data['fitness']:.4f})")
    for k, v in data['params'].items():
        print(f"     {k:<22}: {v}")
    print(f"\n  CAGR           : %{r_opt.cagr*100:+.2f}")
    print(f"  Toplam Getiri  : %{r_opt.total_return*100:+.2f}")
    print(f"  Max Drawdown   : %{r_opt.max_drawdown*100:.2f}")
    print(f"  Sharpe         : {r_opt.sharpe_ratio:.3f}")
    print(f"  Win Rate       : %{r_opt.win_rate*100:.1f}")
    print(f"  Profit Factor  : {r_opt.profit_factor:.2f}")
    print(f"  Toplam İşlem   : {r_opt.total_trades}")
    print(f"  Fitness        : {r_opt.fitness():.4f}")
    print(f"  Son Bakiye     : ${r_opt.final_equity:,.2f}")

# Walk-forward
print(f"\n  Walk-Forward (4 split):")
n_splits = 4
sz = len(df) // n_splits
for i in range(n_splits):
    df_s = df.iloc[i*sz:(i+1)*sz]
    p = GMParams(**data['params']) if os.path.exists(opt_path) else GMParams()
    r = bt.run(df_s, p)
    flag = "✅" if r.total_return > 0 else "❌"
    print(f"  Split {i+1} ({df_s.index[0].date()} → {df_s.index[-1].date()}): "
          f"Getiri=%{r.total_return*100:+.1f} | PF={r.profit_factor:.2f} | "
          f"Trades={r.total_trades} {flag}")

print("=" * 64)
