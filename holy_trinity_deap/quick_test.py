"""
Holy Trinity V7 DEAP — Hızlı Test
====================================
Gold Sniper + NQ Golden Basket optimum parametrelerle test.

Kullanım:
    python quick_test.py
"""
import sys, os, json
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from holy_trinity_deap import run_holy_trinity, BacktestResult, decode as ht_decode

# ── Veri ──────────────────────────────────────────────────
CACHE_DIR = os.path.join(PARENT_DIR, 'institutional_engine', 'cache')

print("=" * 64)
print("🏆 HOLY TRINITY V7 DEAP — HIZLI TEST")
print("=" * 64)

# Gold
from nq_core.gold_sniper_strategy import add_gold_sniper_features
gold_csv = os.path.join(CACHE_DIR, 'XAUUSD_1h.csv')
gold_df  = pd.read_csv(gold_csv, index_col=0, parse_dates=True)
gold_df.columns = gold_df.columns.str.lower()
if 'volume' not in gold_df.columns: gold_df['volume'] = 1000
gold_df = add_gold_sniper_features(gold_df)
print(f"\n📂 Gold: {len(gold_df):,} bar  {gold_df.index[0].date()} → {gold_df.index[-1].date()}")

# NQ
nq_csv = os.path.join(CACHE_DIR, 'NQ_1h.csv')
if os.path.exists(nq_csv):
    nq_df = pd.read_csv(nq_csv, index_col=0, parse_dates=True)
    nq_df.columns = nq_df.columns.str.lower()
    if 'volume' not in nq_df.columns: nq_df['volume'] = 1000
    # NQ sweep features
    tp  = (nq_df['high'] + nq_df['low'] + nq_df['close']) / 3
    vol = nq_df['volume'].replace(0, 1)
    nq_df['vwap']           = (tp*vol).rolling(20).sum() / vol.rolling(20).sum()
    nq_df['htf_swing_high'] = nq_df['high'].rolling(24).max().shift(1)
    nq_df['htf_swing_low']  = nq_df['low'].rolling(24).min().shift(1)
    nq_df['sweep_low']      = (nq_df['low']  < nq_df['htf_swing_low'])  & (nq_df['close'] > nq_df['htf_swing_low'])
    nq_df['sweep_high']     = (nq_df['high'] > nq_df['htf_swing_high']) & (nq_df['close'] < nq_df['htf_swing_high'])
    print(f"📂 NQ  : {len(nq_df):,} bar  {nq_df.index[0].date()} → {nq_df.index[-1].date()}")
else:
    print("⚠️  NQ verisi bulunamadı, sadece Gold kullanılıyor")
    nq_df = gold_df.copy()

# Ortak index
common  = gold_df.index.intersection(nq_df.index)
gold_df = gold_df.loc[common]
nq_df   = nq_df.loc[common]
print(f"🔗 Ortak: {len(common):,} bar\n")

# ── V7 Default ───────────────────────────────────────────
v7_default = {
    'gs_lot': 0.02, 'gs_hard_stop': 500, 'gs_target_pts': 140,
    'gs_stop_pts': 12, 'gs_stale_hours': 72, 'nq_lot': 0.03,
    'nq_tp_pts': 40, 'nq_stop_pts': 150, 'safe_threshold': 3000,
    'growth_factor': 0.40, 'dd_threshold': 0.30,
    'cooldown_bars': 48, 'gs_point_value': 100,
}
r_def = run_holy_trinity(gold_df, nq_df, v7_default, 1_000.0)
print(f"  📈 V7 DEFAULT PARAMS")
print(f"  CAGR           : %{r_def.cagr*100:+.2f}")
print(f"  Toplam Getiri  : %{r_def.total_return*100:+.2f}")
print(f"  Max Drawdown   : %{r_def.max_drawdown*100:.2f}")
print(f"  Sharpe         : {r_def.sharpe_ratio:.3f}")
print(f"  Win Rate       : %{r_def.win_rate*100:.1f}")
print(f"  Profit Factor  : {r_def.profit_factor:.2f}")
print(f"  Gold PnL       : ${r_def.gold_pnl:,.2f}")
print(f"  NQ PnL         : ${r_def.nq_pnl:,.2f}")
print(f"  Toplam İşlem   : {r_def.total_trades}")
print(f"  Fitness        : {r_def.fitness():.4f}")
print(f"  Son Bakiye     : ${r_def.final_equity:,.2f}")

# ── Optimum ───────────────────────────────────────────────
best_json = None
for fname in ['holy_trinity_v7_opt_20260315_222726.json',
              'holy_trinity_v7_opt_20260315_165547.json',
              'ht_v7_20260315_225420.json']:
    fp = os.path.join(SCRIPT_DIR, 'outputs', fname)
    if os.path.exists(fp):
        with open(fp) as f:
            d = json.load(f)
        if best_json is None or d.get('fitness', 0) > best_json.get('fitness', 0):
            best_json = d
            best_json['_file'] = fname

if best_json:
    opt_p = best_json.get('best_params', best_json.get('params', v7_default))
    # Eksik key varsa default'tan al
    for k in v7_default:
        if k not in opt_p:
            opt_p[k] = v7_default[k]
    r_opt = run_holy_trinity(gold_df, nq_df, opt_p, 1_000.0)
    print(f"\n  🏆 OPTİMUM PARAMS ({best_json['_file']}, Fitness={best_json['fitness']:.4f})")
    for k, v in opt_p.items():
        print(f"     {k:<22}: {v:.4f}" if isinstance(v, float) else f"     {k:<22}: {v}")
    print(f"\n  CAGR           : %{r_opt.cagr*100:+.2f}")
    print(f"  Toplam Getiri  : %{r_opt.total_return*100:+.2f}")
    print(f"  Max Drawdown   : %{r_opt.max_drawdown*100:.2f}")
    print(f"  Sharpe         : {r_opt.sharpe_ratio:.3f}")
    print(f"  Win Rate       : %{r_opt.win_rate*100:.1f}")
    print(f"  Profit Factor  : {r_opt.profit_factor:.2f}")
    print(f"  Fitness        : {r_opt.fitness():.4f}")
    print(f"  Son Bakiye     : ${r_opt.final_equity:,.2f}")

# Walk-forward splits
print(f"\n  Walk-Forward (4 split):")
params_to_test = opt_p if best_json else v7_default
n_splits = 4
sz = len(common) // n_splits
for i in range(n_splits):
    comm_s   = common[i*sz:(i+1)*sz]
    gdf_s    = gold_df.loc[comm_s]
    ndf_s    = nq_df.loc[comm_s]
    r = run_holy_trinity(gdf_s, ndf_s, params_to_test, 1_000.0)
    flag = "✅" if r.total_return > 0 else "❌"
    print(f"  Split {i+1} ({comm_s[0].date()} → {comm_s[-1].date()}): "
          f"Getiri=%{r.total_return*100:+.1f} | PF={r.profit_factor:.2f} | "
          f"Trades={r.total_trades} {flag}")

print("=" * 64)
