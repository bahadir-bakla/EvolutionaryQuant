"""
GoldSniper Standalone DEAP Optimizer
======================================
Strateji: GoldSniperStrategyCore (NQ bagimsiz Gold-only mod)
Genome   : 9 parametre (TP, SL, lot, hard stop, stale, pval, growth, min_score)
Fitness  : Walk-forward 4 split composite (CAGR * PF * WR_pen * DD_pen)

Kullanim:
    cd c:\\...\\holy_trinity_deap
    python run_gold_sniper_deap.py
"""

import sys, os, json, time, warnings, random
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import os

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from nq_core.gold_sniper_strategy import add_gold_sniper_features

# ─────────────────────────────────────────────────────────────────
# GENOME TANIMI
# ─────────────────────────────────────────────────────────────────
# [0] gs_target_pts  50  - 350    TP mesafesi (points)
# [1] gs_stop_pts    4   - 35     SL mesafesi (points)
# [2] gs_hard_stop   200 - 2000   Sabit dolar stop
# [3] gs_stale_hours 8   - 200    Stale kapatma suresi (bar)
# [4] gs_point_value 80  - 150    Kontrat degeri
# [5] gs_lot         0.01- 0.15   Baslangic lot
# [6] growth_factor  0.05- 0.60   Compounding carpani
# [7] min_score_long 3   - 5      Minimum long sinyal puani
# [8] min_score_short 3  - 5      Minimum short sinyal puani

GENE_BOUNDS = [
    (50.0,  350.0),   # [0] gs_target_pts
    (4.0,    35.0),   # [1] gs_stop_pts
    (200.0, 2000.0),  # [2] gs_hard_stop
    (8.0,   200.0),   # [3] gs_stale_hours
    (80.0,  150.0),   # [4] gs_point_value
    (0.01,   0.15),   # [5] gs_lot
    (0.05,   0.60),   # [6] growth_factor
    (3.0,    5.0),    # [7] min_score_long
    (3.0,    5.0),    # [8] min_score_short
    (0.0,    0.8),    # [9] meta_bias_threshold
]
GENOME_SIZE = len(GENE_BOUNDS)
GENOME_NAMES = [
    'gs_target_pts', 'gs_stop_pts', 'gs_hard_stop', 'gs_stale_hours',
    'gs_point_value', 'gs_lot', 'growth_factor', 'min_score_long', 'min_score_short', 'meta_bias_threshold'
]

# DEAP ayarlari
POPULATION  = 70
GENERATIONS = 200
CXPB, MUTPB = 0.65, 0.30
CAPITAL  = 1_000.0
PVAL_DEF = 100.0   # sabit point value — optimizasyon dışı

# ─────────────────────────────────────────────────────────────────
# VERI HAZIRLAMA (global — worker'lara gecilir)
# ─────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(PARENT_DIR, 'institutional_engine', 'cache')
CSV_PATH  = os.path.join(CACHE_DIR, 'XAUUSD_1h.csv')

# Global degiskenler — worker process'leri icin module scope'ta tanimli olmali
_df_feat  = None
WF_SPLITS = []
N_WF      = 3   # 4'ten 3'e düşürüldü — her split daha uzun, daha fazla trade

def _load_data():
    global _df_feat, WF_SPLITS
    _df_raw = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    _df_raw.columns = _df_raw.columns.str.lower()
    if 'volume' not in _df_raw.columns: _df_raw['volume'] = 1000
    _df_raw.dropna(subset=['open','high','low','close'], inplace=True)

    _df_feat = add_gold_sniper_features(_df_raw.copy())
    tp  = (_df_feat['high'] + _df_feat['low'] + _df_feat['close']) / 3
    vol = _df_feat['volume'].replace(0, 1)
    _df_feat['vwap'] = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()
    _ema50 = _df_feat['close'].ewm(span=50, adjust=False).mean()
    _df_feat['_bias'] = np.where(_ema50 > _ema50.shift(5), 1,
                        np.where(_ema50 < _ema50.shift(5), -1, 0))

    try:
        sys.path.insert(0, os.path.join(PARENT_DIR, 'spectral_bias_engine'))
        from fft_bias import add_spectral_features
        from hmm_regime import add_regime_features
        from adaptive_meta_labeler import apply_adaptive_meta_labels
        print("   🎧 Spectral Regime Modülü (Meta-Bias) hesaplanıyor...")
        _df_feat = add_spectral_features(_df_feat, window_size=60)
        _df_feat = add_regime_features(_df_feat, lookback=500)
        _df_feat = apply_adaptive_meta_labels(_df_feat)
    except Exception as e:
        print(f"   ⚠️ Spectral özellikler eklenemedi ({e})")
        _df_feat['meta_bias'] = 0.0

    wf_len    = len(_df_feat) // N_WF
    WF_SPLITS = [(i * wf_len, (i + 1) * wf_len) for i in range(N_WF)]
    return _df_feat


# ─────────────────────────────────────────────────────────────────
# DECODE
# ─────────────────────────────────────────────────────────────────
def decode(genome: list) -> dict:
    p = {}
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        raw = float(genome[i])
        p[GENOME_NAMES[i]] = lo + abs(raw % 1.0) * (hi - lo)
    return {
        'gs_target_pts' : float(np.clip(p['gs_target_pts'],  50, 350)),
        'gs_stop_pts'   : float(np.clip(p['gs_stop_pts'],    4,   35)),
        'gs_hard_stop'  : float(np.clip(p['gs_hard_stop'],  200, 2000)),
        'gs_stale_hours': float(np.clip(p['gs_stale_hours'],  8, 200)),
        'gs_point_value': float(np.clip(p['gs_point_value'],  80, 150)),
        'gs_lot'        : float(np.clip(p['gs_lot'],        0.01, 0.15)),
        'growth_factor' : float(np.clip(p['growth_factor'], 0.05, 0.60)),
        'min_score_long' : int(np.clip(round(p['min_score_long']),  3, 5)),
        'min_score_short': int(np.clip(round(p['min_score_short']), 3, 5)),
        'meta_bias_threshold': float(np.clip(p['meta_bias_threshold'], 0.0, 0.8)),
    }


# ─────────────────────────────────────────────────────────────────
# BACKTEST (feature-free — df zaten hazir)
# ─────────────────────────────────────────────────────────────────
def run_gs_backtest(df_slice: pd.DataFrame, params: dict,
                    cap: float = CAPITAL) -> dict:
    tp_pts  = params['gs_target_pts']
    sl_pts  = params['gs_stop_pts']
    hard    = params['gs_hard_stop']
    stale   = params['gs_stale_hours']
    pval    = params['gs_point_value']
    gs_lot  = params['gs_lot']
    gf      = params['growth_factor']
    ml      = params['min_score_long']
    ms      = params['min_score_short']

    balance  = cap
    eq       = [balance]
    trades   = []
    open_pos = None

    for i in range(50, len(df_slice)):
        row   = df_slice.iloc[i]
        price = float(row['close'])

        if open_pos is not None:
            moved    = (price - open_pos['e']) * open_pos['d']
            pnl_live = moved * open_pos['s'] * pval
            bi       = i - open_pos['bi']
            closed = False
            if   moved    >= tp_pts: pnl = tp_pts * open_pos['s'] * pval; closed = True
            elif moved    <= -sl_pts: pnl = -sl_pts * open_pos['s'] * pval; closed = True
            elif pnl_live <= -hard:  pnl = -hard; closed = True
            elif bi        >= stale: pnl = pnl_live; closed = True
            if closed:
                balance += pnl
                trades.append({
                    'pnl': pnl, 
                    'rr': tp_pts / max(sl_pts, 1),
                    'entry_time': df_slice.index[open_pos['bi']],
                    'size': open_pos['s']
                })
                open_pos = None

        if open_pos is None:
            pb  = max(0, int((balance - cap) // 1000))
            lot = min(round(gs_lot * (1 + pb * gf), 3), 0.50)

            bias_b = bool(row.get('daily_bias_bullish',
                          row.get('_bias', 0) > 0))
            bias_e = bool(row.get('daily_bias_bearish',
                          row.get('_bias', 0) < 0))
            sw_b  = bool(row.get('sweep_minor_low',  False))
            sw_e  = bool(row.get('sweep_minor_high', False))
            h4_b  = bool(row.get('h4_reject_up',   False))
            h4_e  = bool(row.get('h4_reject_down', False))
            ob_b  = bool(row.get('ob_tap_bullish',  False))
            ob_e  = bool(row.get('ob_tap_bearish',  False))
            vwap_ = float(row.get('vwap', price))
            av    = price > vwap_

            ls = (2 if bias_b else 0) + (2 if sw_b else 0) + \
                 (1 if h4_b  else 0) + (1 if ob_b else 0) + (1 if av else 0)
            ss = (2 if bias_e else 0) + (2 if sw_e else 0) + \
                 (1 if h4_e  else 0) + (1 if ob_e else 0) + (0 if av else 1)

            mb_th = params.get('meta_bias_threshold', 0.0)
            meta  = float(row.get('meta_bias', 0.0))
            if mb_th >= 0.05:
                if meta < -mb_th: ls = 0
                if meta > mb_th:  ss = 0

            if ls >= ml and ls > ss:
                open_pos = {'e': price, 'd': 1,  's': lot, 'bi': i}
            elif ss >= ms and ss > ls:
                open_pos = {'e': price, 'd': -1, 's': lot, 'bi': i}

        eq.append(balance)

    if open_pos:
        fp  = float(df_slice['close'].iloc[-1])
        pnl = (fp - open_pos['e']) * open_pos['d'] * open_pos['s'] * pval
        balance += pnl; trades.append({'pnl': pnl, 'rr': 0}); eq[-1] = balance

    eq = np.array(eq)
    tr = (eq[-1] - cap) / cap
    pk = np.maximum.accumulate(eq)
    dd = float(abs(((eq - pk) / (pk + 1e-10)).min()))
    ny = len(eq) / (365 * 24)
    cagr = float((eq[-1] / cap) ** (1/max(ny, 0.1)) - 1) if eq[-1] > 0 else -1

    n  = len(trades)
    wins  = [t['pnl'] for t in trades if t['pnl'] > 0]
    loses = [t['pnl'] for t in trades if t['pnl'] <= 0]
    wr  = len(wins) / n if n > 0 else 0
    gp  = sum(wins); gl = abs(sum(loses)) if loses else 1e-10
    pf  = float(np.clip(gp / (gl + 1e-10), 0, 20))
    rrs = [t['rr'] for t in trades if t.get('rr', 0) > 0]
    rr  = float(np.mean(rrs)) if rrs else 0

    return {'cagr': cagr, 'total_return': tr, 'max_drawdown': dd,
            'win_rate': wr, 'profit_factor': pf, 'avg_rr': rr,
            'total_trades': n, 'final_equity': float(eq[-1])}


def fitness_from_result(r: dict) -> float:
    if r['total_trades'] < 4:     return -999.0   # 8'den 4'e düşürüldü
    if r['total_return']  <= 0:   return float(np.clip(r['total_return'] * 5, -50, -0.01))
    if r['profit_factor'] < 1.0:  return float(np.clip((r['profit_factor'] - 1) * 10, -20, -0.01))
    if r['max_drawdown']  > 0.75: return -50.0   # 0.70'ten 0.75'e

    pf     = float(np.clip(r['profit_factor'], 1, 15))
    cagr   = float(np.clip(r['cagr'], 0, 20))
    sharpe = float(np.clip(r.get('sharpe', 0), 0, 10))
    rr     = float(np.clip(r['avg_rr'], 0, 20))
    dd_pen = max(0.1, 1 - max(0, r['max_drawdown'] - 0.15) * 2)

    score = (pf * 0.30 + cagr * 0.35 + rr * 0.20 + r['win_rate'] * 0.15) * dd_pen
    return float(np.clip(score, 0, 100))


# ─────────────────────────────────────────────────────────────────
# WALK-FORWARD FITNESS
# ─────────────────────────────────────────────────────────────────
def evaluate_genome(genome):
    try:
        p = decode(genome)
        fits = []
        for start_i, end_i in WF_SPLITS:
            sl = _df_feat.iloc[start_i:end_i]
            if len(sl) < 200: continue
            r = run_gs_backtest(sl, p, CAPITAL)
            fits.append(fitness_from_result(r))
        if not fits: return (-999.0,)
        # En kotu split'i de dikkate al
        score = np.mean(fits) * 0.7 + min(fits) * 0.3
        return (float(score),)
    except Exception:
        return (-999.0,)


# ─────────────────────────────────────────────────────────────────
# DEAP KURULUM — module scope'ta (worker'lar icin gerekli)
# ─────────────────────────────────────────────────────────────────
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("pip install deap")
    sys.exit(1)

if not hasattr(creator, 'FitnessGS'):
    creator.create('FitnessGS', base.Fitness, weights=(1.0,))
if not hasattr(creator, 'IndividualGS'):
    creator.create('IndividualGS', list, fitness=creator.FitnessGS)

toolbox = base.Toolbox()
toolbox.register('individual', tools.initIterate, creator.IndividualGS,
                 lambda: [random.random() for _ in range(GENOME_SIZE)])
toolbox.register('population', tools.initRepeat, list, toolbox.individual)
toolbox.register('evaluate', evaluate_genome)
toolbox.register('mate',    tools.cxBlend, alpha=0.35)
toolbox.register('mutate',  tools.mutGaussian, mu=0, sigma=0.25, indpb=0.30)  # 0.15 -> 0.25
toolbox.register('select',  tools.selTournament, tournsize=4)


# ─────────────────────────────────────────────────────────────────
# WINDOWS MULTIPROCESSING GEREKTIRIYOR: if __name__ == '__main__'
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Veri yukle
    print("=" * 64)
    print("🏆 GOLD SNIPER DEAP OPTİMİZASYONU")
    print(f"   Populasyon : {POPULATION}")
    print(f"   Nesil      : {GENERATIONS}")
    print("=" * 64)
    print("\nVeri & feature engineering (bir kez)...", end=' ', flush=True)
    _load_data()
    print(f"OK — {len(_df_feat):,} bar  {_df_feat.index[0].date()} → {_df_feat.index[-1].date()}")

    N_CORES = max(1, min(os.cpu_count() - 1, 8))  # thread sayisi
    print(f"CPU : {N_CORES} thread (ThreadPoolExecutor) 🚀")
    print(f"Veri: {len(_df_feat):,} bar | WF splits: {N_WF}")
    print(f"{'─'*64}")

    pop  = toolbox.population(n=POPULATION)
    hof  = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('max', np.max)
    stats.register('avg', np.mean)

    t0 = time.time()

    # ThreadPoolExecutor: thread'ler _df_feat'e erisebilir (memory paylasilir)
    with ThreadPoolExecutor(max_workers=N_CORES) as executor:
        def parallel_eval(genomes):
            return list(executor.map(toolbox.evaluate, genomes))

        for gen in range(GENERATIONS):
            invalid = [ind for ind in pop if not ind.fitness.valid]
            if invalid:
                results = parallel_eval(invalid)
                for ind, fit in zip(invalid, results):
                    ind.fitness.values = fit

            hof.update(pop)

            record  = stats.compile(pop)
            elapsed = time.time() - t0
            eta     = (elapsed / (gen + 1)) * (GENERATIONS - gen - 1)
            if gen % 5 == 0:
                print(f"Gen {gen:3d}: Max={record['max']:7.3f} | "
                      f"Avg={record['avg']:7.3f} | "
                      f"{elapsed/60:.1f}dk | ~{eta/60:.0f}dk kaldi")

            if gen > 0 and gen % 40 == 0:
                toolbox.unregister('mutate')
                toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.20, indpb=0.30)
                print(f"  Sigma reset: 0.20")

            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values
            for mut in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mut)
                    del mut.fitness.values
            pop[:] = offspring

    elapsed_total = (time.time() - t0) / 60
    print(f"\nTamamlandi ({elapsed_total:.1f} dk)  Fitness: {hof[0].fitness.values[0]:.4f}")

    # ─────────────────────────────────────────────────────────────────
    # SONUCLAR
    # ─────────────────────────────────────────────────────────────────
    best_genome = hof[0]
    best_p      = decode(best_genome)
    best_fit    = hof[0].fitness.values[0]

    print("\n" + "=" * 64)
    print("🏆 EN İYİ GOLD SNIPER PARAMETRELERİ:")
    print("=" * 64)
    for k, v in best_p.items():
        print(f"   {k:<20}: {v}")

    r_full = run_gs_backtest(_df_feat, best_p, CAPITAL)
    r_def  = run_gs_backtest(_df_feat, {
        'gs_target_pts': 140, 'gs_stop_pts': 12, 'gs_hard_stop': 500,
        'gs_stale_hours': 72, 'gs_point_value': 100, 'gs_lot': 0.02,
        'growth_factor': 0.40, 'min_score_long': 4, 'min_score_short': 4,
    }, CAPITAL)

    def print_res(r, label):
        print(f"\n{'═'*56}")
        print(f"  {label}")
        print(f"{'═'*56}")
        print(f"  CAGR          : %{r['cagr']*100:+.2f}")
        print(f"  Toplam Getiri : %{r['total_return']*100:+.2f}")
        print(f"  Max Drawdown  : %{r['max_drawdown']*100:.2f}")
        print(f"  Win Rate      : %{r['win_rate']*100:.1f}")
        print(f"  Profit Factor : {r['profit_factor']:.2f}")
        print(f"  Avg R:R       : {r['avg_rr']:.2f}")
        print(f"  Toplam Islem  : {r['total_trades']}")
        print(f"  Son Bakiye    : ${r['final_equity']:,.2f}")

    print_res(r_def, "DEFAULT PARAMS")
    print_res(r_full, "OPTİMİZE EDİLMİS")

    print(f"\n  Walk-Forward (4 split):")
    pos = 0
    for i, (s, e) in enumerate(WF_SPLITS):
        sl = _df_feat.iloc[s:e]
        r  = run_gs_backtest(sl, best_p, CAPITAL)
        ok = r['total_return'] > 0
        if ok: pos += 1
        flag = "✅" if ok else "❌"
        print(f"  Split {i+1} ({sl.index[0].date()} → {sl.index[-1].date()}): "
              f"Getiri=%{r['total_return']*100:+.1f} | "
              f"PF={r['profit_factor']:.2f} | "
              f"Trades={r['total_trades']} {flag}")
    print(f"\n  Pozitif split: {pos}/4  {'✅ PORTFOLYO HAZIR' if pos >= 3 else '⚠️'}")

    out_dir = os.path.join(SCRIPT_DIR, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(out_dir, f'gold_sniper_params_{ts}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'fitness': best_fit, 'best_params': best_p,
                   'backtest': r_full,
                   'top5': [decode(h) for h in hof]}, f, ensure_ascii=False, indent=2)
    print(f"\n  Kaydedildi: {out}")
    print("=" * 64)
