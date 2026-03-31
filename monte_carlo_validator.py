"""
Monte Carlo Portfolio Validator v2 — HIZLI
===========================================
Features onceden hesaplanir, her simde sadece backtest dongusu kosar.
500 sim ~2-5 dakikada tamamlanir (v1: 30+ dakika)

Kullanim: python monte_carlo_validator.py
"""

import sys, os, warnings, json
import numpy as np
import pandas as pd
from datetime import timedelta
import importlib.util

warnings.filterwarnings('ignore')

ROOT       = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR  = os.path.join(ROOT, 'institutional_engine', 'cache')
CAPITAL    = 1_000.0
N_SIMS     = 500
MIN_DAYS   = 365
MAX_DAYS   = 365 * 3
SEED       = 42


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("=" * 68)
print("  MONTE CARLO PORTFOLIO VALIDATOR v2 (HIZLI)")
print(f"  {N_SIMS} Rastgele Pencere | {MIN_DAYS//365}-{MAX_DAYS//365} Yil | 3 Algo")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────
# VERI + FEATURES — ONCE HESAPLA
# ─────────────────────────────────────────────────────────────────
print("\n[0/5] Veri & feature engineering (bir kez)...")

gold_csv = os.path.join(CACHE_DIR, 'XAUUSD_1h.csv')
df_raw = pd.read_csv(gold_csv, index_col=0, parse_dates=True)
df_raw.columns = df_raw.columns.str.lower()
if 'volume' not in df_raw.columns: df_raw['volume'] = 1000
df_raw.dropna(subset=['open','high','low','close'], inplace=True)
print(f"  Raw gold: {len(df_raw):,} bar")

# ── GoldMaster features ────────────────────────────────────────
df_gm = None
try:
    gm_mod = load_module('gm_bt',
               os.path.join(ROOT, 'goldmaster_deap', '01_goldmaster_backtest.py'))
    gm_opt_path = None
    gm_out = os.path.join(ROOT, 'goldmaster_deap', 'outputs')
    if os.path.exists(gm_out):
        files = sorted([f for f in os.listdir(gm_out)
                        if f.startswith('goldmaster_params_') and f.endswith('.json')],
                       reverse=True)
        if files:
            gm_opt_path = os.path.join(gm_out, files[0])
    if gm_opt_path and os.path.exists(gm_opt_path):
        with open(gm_opt_path) as f: d = json.load(f)
        gm_params = gm_mod.GMParams(**d['params'])
        print(f"  GoldMaster: {files[0]}  fitness={d['fitness']:.4f}")
    else:
        gm_params = gm_mod.GMParams()
        print("  GoldMaster: default params")
    bt_gm = gm_mod.GoldMasterBacktester(initial_capital=CAPITAL)
    df_gm = ('goldmaster', gm_params, bt_gm, df_raw)
except Exception as e:
    print(f"  GoldMaster HATA: {e}")

# ── LiquidityEdge features (ONCE hesapla) ─────────────────────
df_le_feat = None
le_params  = None
run_le_bt  = None
try:
    le_mod = load_module('le_bt',
               os.path.join(ROOT, 'liquidity_edge_deap', 'backtest_engine.py'))
    run_le_bt = le_mod.run_backtest

    le_out = os.path.join(ROOT, 'liquidity_edge_deap', 'outputs')
    le_best = None
    if os.path.exists(le_out):
        files = sorted([f for f in os.listdir(le_out)
                        if f.startswith('liquidity_edge_') and f.endswith('.json')
                        and 'monte_carlo' not in f], reverse=True)
        if files:
            with open(os.path.join(le_out, files[0])) as f:
                d = json.load(f)
            le_best = d
            print(f"  LiquidityEdge: {files[0]}  fitness={d.get('fitness', 0):.4f}")

    le_params = le_best['best_params'] if le_best and 'best_params' in le_best \
                else le_mod.DEFAULT_PARAMS

    # Features bir kez hesapla
    print("  LiquidityEdge feature engineering...", end=' ', flush=True)
    df_le_feat = le_mod.add_liquidity_features(df_raw.copy(), le_params)
    print(f"OK ({len(df_le_feat.columns)} kolon)")
except Exception as e:
    print(f"\n  LiquidityEdge HATA: {e}")

# ── GoldSniper features (ONCE hesapla) ────────────────────────
df_gs_feat = None
try:
    sys.path.insert(0, ROOT)
    from nq_core.gold_sniper_strategy import add_gold_sniper_features
    print("  GoldSniper feature engineering...", end=' ', flush=True)
    df_gs_feat = add_gold_sniper_features(df_raw.copy())
    tp  = (df_gs_feat['high'] + df_gs_feat['low'] + df_gs_feat['close']) / 3
    vol = df_gs_feat['volume'].replace(0, 1)
    df_gs_feat['vwap'] = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()
    ema50 = df_gs_feat['close'].ewm(span=50, adjust=False).mean()
    df_gs_feat['_bias'] = np.where(ema50 > ema50.shift(5), 1,
                           np.where(ema50 < ema50.shift(5), -1, 0))
    print(f"OK ({len(df_gs_feat.columns)} kolon)")
except Exception as e:
    print(f"\n  GoldSniper HATA: {e}")

# ─────────────────────────────────────────────────────────────────
# HIZLI BACKTEST FONKSIYONLARI (feature engineering yok!)
# ─────────────────────────────────────────────────────────────────
POINT_VALUE = 100.0

def bt_le_fast(df_feat_slice: pd.DataFrame, p: dict, cap: float = CAPITAL):
    """LiquidityEdge fast backtest — feature engineering yok."""
    balance  = cap
    eq       = [balance]
    trades   = []
    open_pos = None
    MIN_SCORE = 4

    for i in range(50, len(df_feat_slice)):
        row   = df_feat_slice.iloc[i]
        price = float(row['close'])
        atr   = float(row.get('atr', price * 0.005)) or price * 0.005

        if open_pos is not None:
            moved = (price - open_pos['e']) * open_pos['d']
            if moved <= -open_pos['sl']:
                pnl = -open_pos['sl'] * open_pos['s'] * POINT_VALUE
                balance += pnl; trades.append(pnl); open_pos = None
            elif moved >= open_pos['tp']:
                pnl = open_pos['tp'] * open_pos['s'] * POINT_VALUE
                balance += pnl; trades.append(pnl); open_pos = None

        if open_pos is None:
            pb  = max(0, int((balance - cap) // 1000))
            lot = min(round(p['lot_size'] * (1 + pb * p['growth_factor']), 3), 2.0)
            sl  = max(atr * p['sl_atr_mult'], 1.0)
            tp  = max(atr * p['tp_atr_mult'], sl * 1.5)

            ls = ss = 0
            if row.get('sweep_low_bull', False):  ls += 2
            if row.get('ob_bull', False):          ls += 2
            if row.get('pivot_bull', False):       ls += 2
            if row.get('trend_up', False):         ls += 1
            if row.get('fvg_bull', False):         ls += 1
            rsi = float(row.get('rsi', 50))
            if rsi < p['rsi_os_level'] + 15:       ls += 1
            if row.get('above_vwap', False):       ls += 1
            if row.get('rsi_bull_div', False):     ls += 1

            if row.get('sweep_high_bear', False):  ss += 2
            if row.get('ob_bear', False):          ss += 2
            if row.get('pivot_bear', False):       ss += 2
            if row.get('trend_dn', False):         ss += 1
            if row.get('fvg_bear', False):         ss += 1
            if rsi > p['rsi_ob_level'] - 15:       ss += 1
            if row.get('below_vwap', False):       ss += 1
            if row.get('rsi_bear_div', False):     ss += 1

            mb_th = p.get('meta_bias_threshold', 0.0)
            meta = float(row.get('meta_bias', 0.0))
            if mb_th >= 0.05:
                if meta < -mb_th: ls = 0
                if meta > mb_th:  ss = 0

            if ls >= MIN_SCORE and ls > ss:
                open_pos = {'e': price, 'd': 1, 's': lot, 'sl': sl, 'tp': tp}
            elif ss >= MIN_SCORE and ss > ls:
                open_pos = {'e': price, 'd': -1, 's': lot, 'sl': sl, 'tp': tp}

        eq.append(balance)

    if open_pos:
        fp = float(df_feat_slice['close'].iloc[-1])
        pnl = (fp - open_pos['e']) * open_pos['d'] * open_pos['s'] * POINT_VALUE
        balance += pnl; trades.append(pnl); eq[-1] = balance

    eq  = np.array(eq)
    ret = (eq[-1] - cap) / cap
    pk  = np.maximum.accumulate(eq)
    dd  = float(abs(((eq - pk) / (pk + 1e-10)).min()))
    ny  = len(eq) / (365 * 24)
    cagr = float((eq[-1] / cap) ** (1/max(ny, 0.1)) - 1) if eq[-1] > 0 else -1
    wins  = [t for t in trades if t > 0]
    loses = [t for t in trades if t <= 0]
    gp = sum(wins); gl = abs(sum(loses)) if loses else 1e-10
    pf = float(np.clip(gp / (gl + 1e-10), 0, 20))
    return {'cagr': cagr, 'total_return': ret, 'max_drawdown': dd,
            'profit_factor': pf, 'total_trades': len(trades)}


def bt_gs_fast(df_feat_slice: pd.DataFrame, cap: float = CAPITAL,
               gs_lot=0.0133, tp_pts=149.6, sl_pts=15.5,
               hard=843.45, stale=24.0, gf=0.20, pval=106.76):
    """GoldSniper fast backtest — feature engineering yok."""
    balance  = cap
    eq       = [balance]
    trades   = []
    open_pos = None

    for i in range(50, len(df_feat_slice)):
        row   = df_feat_slice.iloc[i]
        price = float(row['close'])

        if open_pos is not None:
            moved    = (price - open_pos['e']) * open_pos['d']
            pnl_live = moved * open_pos['s'] * pval
            bi       = i - open_pos['bi']
            closed = False
            if moved >= tp_pts:       pnl = tp_pts * open_pos['s'] * pval; closed = True
            elif moved <= -sl_pts:    pnl = -sl_pts * open_pos['s'] * pval; closed = True
            elif pnl_live <= -hard:   pnl = -hard; closed = True
            elif bi >= stale:         pnl = pnl_live; closed = True
            if closed:
                balance += pnl; trades.append(pnl); open_pos = None

        if open_pos is None:
            pb  = max(0, int((balance - cap) // 1000))
            lot = min(round(gs_lot * (1 + pb * gf), 3), 0.50)
            bias_b = bool(row.get('daily_bias_bullish',
                          row.get('_bias', 0) > 0))
            bias_e = bool(row.get('daily_bias_bearish',
                          row.get('_bias', 0) < 0))
            sw_b  = bool(row.get('sweep_minor_low', False))
            sw_e  = bool(row.get('sweep_minor_high', False))
            h4_b  = bool(row.get('h4_reject_up', False))
            h4_e  = bool(row.get('h4_reject_down', False))
            ob_b  = bool(row.get('ob_tap_bullish', False))
            ob_e  = bool(row.get('ob_tap_bearish', False))
            vwap  = float(row.get('vwap', price))
            av    = price > vwap
            ls = (2 if bias_b else 0) + (2 if sw_b else 0) + \
                 (1 if h4_b  else 0) + (1 if ob_b else 0) + (1 if av else 0)
            ss = (2 if bias_e else 0) + (2 if sw_e else 0) + \
                 (1 if h4_e  else 0) + (1 if ob_e else 0) + (0 if av else 1)
            mb_th = 0.54 # typical evolved gs meta_bias filter average
            meta = float(row.get('meta_bias', 0.0))
            if mb_th >= 0.05:
                if meta < -mb_th: ls = 0
                if meta > mb_th:  ss = 0

            if ls >= 4 and ls > ss:
                open_pos = {'e': price, 'd': 1,  's': lot, 'bi': i}
            elif ss >= 4 and ss > ls:
                open_pos = {'e': price, 'd': -1, 's': lot, 'bi': i}

        eq.append(balance)

    eq  = np.array(eq)
    ret = (eq[-1] - cap) / cap
    pk  = np.maximum.accumulate(eq)
    dd  = float(abs(((eq - pk) / (pk + 1e-10)).min()))
    ny  = len(eq) / (365 * 24)
    cagr = float((eq[-1] / cap) ** (1/max(ny, 0.1)) - 1) if eq[-1] > 0 else -1
    wins  = [t for t in trades if t > 0]
    loses = [t for t in trades if t <= 0]
    gp = sum(wins); gl = abs(sum(loses)) if loses else 1e-10
    pf = float(np.clip(gp / (gl + 1e-10), 0, 20))
    return {'cagr': cagr, 'total_return': ret, 'max_drawdown': dd,
            'profit_factor': pf, 'total_trades': len(trades)}


# ─────────────────────────────────────────────────────────────────
# MONTE CARLO
# ─────────────────────────────────────────────────────────────────
print(f"\n[1/5] Monte Carlo ({N_SIMS} iterasyon)...")
rng = np.random.default_rng(SEED)

# Hangi feature df'leri mevcut?
feat_dfs = {}
if df_gm:          feat_dfs['GoldMaster']    = df_raw
if df_le_feat is not None: feat_dfs['LiquidityEdge'] = df_le_feat
if df_gs_feat is not None: feat_dfs['GoldSniper']    = df_gs_feat

ref_df   = df_raw
start_ts = ref_df.index[0]
end_ts   = ref_df.index[-1]
total_td = (end_ts - start_ts).days

mc_results = {name: [] for name in feat_dfs}

for sim_i in range(N_SIMS):
    max_start = total_td - MIN_DAYS
    if max_start <= 0: break
    offset  = int(rng.integers(0, max_start))
    win_len = int(rng.integers(MIN_DAYS, min(MAX_DAYS, total_td - offset)))
    ws = start_ts + timedelta(days=offset)
    we = ws       + timedelta(days=win_len)

    # GoldMaster
    if df_gm:
        sl = df_raw.loc[ws:we]
        if len(sl) >= 200:
            try:
                _, p, b, _ = df_gm
                r = b.run(sl, p)
                if hasattr(r, 'cagr'):
                    mc_results['GoldMaster'].append({
                        'cagr': r.cagr, 'total_return': r.total_return,
                        'max_drawdown': r.max_drawdown,
                        'profit_factor': r.profit_factor,
                        'total_trades': r.total_trades,
                    })
            except Exception: pass

    # LiquidityEdge (feature df dilimle)
    if df_le_feat is not None:
        sl = df_le_feat.loc[ws:we]
        if len(sl) >= 200:
            try:
                mc_results['LiquidityEdge'].append(bt_le_fast(sl, le_params, CAPITAL))
            except Exception: pass

    # GoldSniper (feature df dilimle)
    if df_gs_feat is not None:
        sl = df_gs_feat.loc[ws:we]
        if len(sl) >= 200:
            try:
                mc_results['GoldSniper'].append(bt_gs_fast(sl, CAPITAL))
            except Exception: pass

    if (sim_i + 1) % 100 == 0 or sim_i == 0:
        print(f"  {sim_i+1:3}/{N_SIMS} ...", end='\r')

print(f"  {N_SIMS}/{N_SIMS} tamamlandi!       ")

# ─────────────────────────────────────────────────────────────────
# ANALIZ
# ─────────────────────────────────────────────────────────────────
print(f"\n[2/5] Istatistiksel ozet:\n")

mc_summary = {}
for name, sims in mc_results.items():
    valid = [s for s in sims if s.get('total_trades', 0) >= 5]
    if not valid:
        print(f"  {name}: Yeterli veri yok\n"); continue

    returns = [s['total_return']  for s in valid]
    cagrs   = [s['cagr']          for s in valid]
    dds     = [s['max_drawdown']  for s in valid]
    pfs     = [min(s['profit_factor'], 20) for s in valid]

    p_profit   = np.mean([r > 0 for r in returns]) * 100
    med_cagr   = np.median(cagrs)  * 100
    p5_cagr    = np.percentile(cagrs, 5)  * 100
    p95_cagr   = np.percentile(cagrs, 95) * 100
    med_dd     = np.median(dds)    * 100
    worst_dd   = np.percentile(dds, 95) * 100
    med_pf     = np.median(pfs)

    grade = "A+" if p_profit > 80 and med_pf > 1.3 else \
            "A"  if p_profit > 70 and med_pf > 1.2 else \
            "B"  if p_profit > 60 else "C"

    mc_summary[name] = dict(n=len(valid), p_profit=p_profit,
                            med_cagr=med_cagr, p5_cagr=p5_cagr, p95_cagr=p95_cagr,
                            med_dd=med_dd, worst_dd=worst_dd, med_pf=med_pf, grade=grade)

    print(f"  {'─'*58}")
    print(f"  {name:<18}  NOT: {grade}   n={len(valid)}")
    print(f"  {'─'*58}")
    print(f"  P(Kazanc penceresi) : %{p_profit:.1f}")
    print(f"  Medyan CAGR         : %{med_cagr:+.1f}")
    print(f"  En kotu %5 CAGR     : %{p5_cagr:+.1f}")
    print(f"  En iyi  %95 CAGR    : %{p95_cagr:+.1f}")
    print(f"  Medyan Max DD       : %{med_dd:.1f}")
    print(f"  Worst-case DD (%95) : %{worst_dd:.1f}")
    print(f"  Medyan PF           : {med_pf:.3f}\n")

# ─────────────────────────────────────────────────────────────────
# PORTFOLIO OZET
# ─────────────────────────────────────────────────────────────────
print(f"\n[3/5] Portfolio ozeti:")
if mc_summary:
    avg_pp  = np.mean([v['p_profit'] for v in mc_summary.values()])
    avg_cg  = np.mean([v['med_cagr'] for v in mc_summary.values()])
    avg_dd  = np.mean([v['med_dd']   for v in mc_summary.values()])
    avg_pf  = np.mean([v['med_pf']   for v in mc_summary.values()])
    best    = max(mc_summary, key=lambda k: mc_summary[k]['p_profit'])
    p_grade = "A+" if avg_pp > 80 and avg_pf > 1.3 else \
              "A"  if avg_pp > 70 else "B"

    print(f"\n  {'━'*58}")
    print(f"  PORTFOLIO MONTE CARLO ({N_SIMS} sim, {MIN_DAYS//365}-{MAX_DAYS//365}yr)")
    print(f"  {'━'*58}")
    print(f"  Ort. P(Kazanc pencere) : %{avg_pp:.1f}")
    print(f"  Ort. Medyan CAGR       : %{avg_cg:+.1f}")
    print(f"  Ort. Medyan Max DD     : %{avg_dd:.1f}")
    print(f"  Ort. Medyan PF         : {avg_pf:.3f}")
    print(f"  En guclu algo          : {best}")
    print(f"  PORTFOLIO NOTU         : {p_grade}")
    flag = ("ROBUST - GitHub'a hazir!" if avg_pp >= 70
            else "Iyilestirme gerekebilir")
    print(f"\n  {flag}")

# Kaydet
out_dir = os.path.join(ROOT, 'liquidity_edge_deap', 'outputs')
os.makedirs(out_dir, exist_ok=True)
out_json = os.path.join(out_dir, 'monte_carlo_results.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump({'n_simulations': N_SIMS,
               'window': f'{MIN_DAYS//365}-{MAX_DAYS//365}yr',
               'algos': list(feat_dfs.keys()),
               'summary': mc_summary}, f, ensure_ascii=False, indent=2)
print(f"\n  Sonuclar: {out_json}")
print("=" * 68)
