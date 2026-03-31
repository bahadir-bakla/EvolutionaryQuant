"""
Silver Momentum DEAP Optimizer — EvolutionaryQuant
===================================================
Walk-forward DEAP optimization for XAGUSD momentum strategy.
"""

import sys, os, random, time, json, multiprocessing
import numpy as np
from copy import deepcopy
from datetime import datetime

_DIR    = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
sys.path.insert(0, _DIR)
sys.path.insert(0, _PARENT)

from silver_strategy import SilverMomentumParams, decode_genome, GENOME_SIZE
from backtest_engine import run_backtest, SilverBacktestResult
from silver_strategy import add_silver_features

from deap import base, creator, tools

if not hasattr(creator, 'FitnessSilver'):
    creator.create('FitnessSilver', base.Fitness, weights=(1.0,))
if not hasattr(creator, 'IndividualSilver'):
    creator.create('IndividualSilver', list, fitness=creator.FitnessSilver)

# ── Worker globals ─────────────────────────────────────────────────
_WF_RAW_SPLITS: list = []
_INIT_CAP:  float     = 1_000.0
_USE_KG:    bool      = False


def _init_worker(wf_splits, init_cap, use_kg):
    global _WF_RAW_SPLITS, _INIT_CAP, _USE_KG
    sys.path.insert(0, _DIR); sys.path.insert(0, _PARENT)
    _WF_RAW_SPLITS = wf_splits
    _INIT_CAP      = init_cap
    _USE_KG        = use_kg


def evaluate_genome(genome) -> tuple:
    try:
        params = decode_genome(genome)
        scores = []
        for raw in _WF_RAW_SPLITS:
            try:
                feat = add_silver_features(raw, params)
                r    = run_backtest(feat, params, _INIT_CAP, _USE_KG)
                scores.append(r.fitness())
            except Exception:
                scores.append(-999.0)

        if not scores:
            return (-999.0,)

        mean_s  = float(np.mean(scores))
        std_s   = float(np.std(scores))
        worst_s = float(np.min(scores))
        consis  = max(-1.0, min(1.0, 1 - std_s / (abs(mean_s) + 1e-6)))
        blended = mean_s * 0.50 + worst_s * 0.25 + mean_s * 0.25 * max(0, consis)
        return (float(np.clip(blended, -999, 200)),)
    except Exception:
        return (-999.0,)


def _proxy(g): return evaluate_genome(g)


def build_wf_splits(df_raw, n_splits: int = 5, oos_pct: float = 0.30):
    n    = len(df_raw)
    size = n // n_splits
    splits = []
    for i in range(n_splits):
        start     = i * size
        end       = (i + 1) * size
        oos_start = start + int(size * (1 - oos_pct))
        sl        = df_raw.iloc[oos_start:end].copy()
        if len(sl) >= 100:
            splits.append(sl)
    return splits


def run_deap_optimizer(df_raw,
                       initial_capital: float = 1_000.0,
                       population:  int = 80,
                       generations: int = 200,
                       n_splits:    int = 5,
                       n_jobs:      int = None,
                       use_kelly_garch: bool = False,
                       verbose:     bool = True,
                       output_dir:  str = None):

    if n_jobs is None:
        n_jobs = max(1, min(multiprocessing.cpu_count() - 1, 10))
    if output_dir is None:
        output_dir = os.path.join(_DIR, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    raw_splits = build_wf_splits(df_raw, n_splits)
    if verbose:
        print(f"📊 Walk-forward: {len(raw_splits)} OOS splits")
        for i, sl in enumerate(raw_splits):
            print(f"   Split {i+1}: {len(sl):,} bars [{sl.index[0].date()} → {sl.index[-1].date()}]")

    toolbox = base.Toolbox()
    toolbox.register('attr_float', random.uniform, 0.0, 1.0)
    toolbox.register('individual', tools.initRepeat, creator.IndividualSilver,
                     toolbox.attr_float, n=GENOME_SIZE)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('select', tools.selTournament, tournsize=5)
    toolbox.register('mate',   tools.cxBlend, alpha=0.3)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.25, indpb=0.3)

    pool = multiprocessing.Pool(n_jobs, _init_worker,
                                (raw_splits, initial_capital, use_kelly_garch))

    pop  = toolbox.population(n=population)
    hof  = tools.HallOfFame(15)
    stat = tools.Statistics(key=lambda ind: ind.fitness.values[0]
                            if ind.fitness.valid else -999.0)
    stat.register('max', np.max)
    stat.register('avg', np.mean)

    if verbose:
        kgstr = " | Kelly+GARCH" if use_kelly_garch else ""
        print(f"\n🧬 Pop:{population} | Gen:{generations} | CPUs:{n_jobs}{kgstr}")
        print("─" * 60)

    fits = pool.map(_proxy, pop)
    for ind, f in zip(pop, fits):
        ind.fitness.values = f
    hof.update(pop)

    best_fit = hof[0].fitness.values[0]
    stag = 0; sigma = 0.25
    stats_log = []; t0 = time.time()
    rec = stat.compile(pop)
    if verbose:
        print(f"Gen   0: Max={rec['max']:7.3f} | Avg={rec['avg']:7.3f}")

    for gen in range(1, generations + 1):
        offspring = list(map(deepcopy, toolbox.select(pop, len(pop))))
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.70:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for mut in offspring:
            if random.random() < 0.25:
                toolbox.mutate(mut)
                del mut.fitness.values
                for j in range(len(mut)):
                    mut[j] = float(np.clip(mut[j], 0.0, 1.0))

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        if invalid:
            fits = pool.map(_proxy, invalid)
            for ind, f in zip(invalid, fits):
                ind.fitness.values = f

        pop[:] = tools.selBest(pop, 10) + offspring[10:]
        hof.update(pop)
        rec = stat.compile(pop)
        stats_log.append({'gen': gen, 'max': float(rec['max']), 'avg': float(rec['avg'])})

        curr = hof[0].fitness.values[0]
        if curr > best_fit + 0.005:
            best_fit = curr; stag = 0; sigma = max(0.10, sigma * 0.95)
        else:
            stag += 1
        if stag >= 20:
            sigma = min(0.65, sigma * 1.6)
            fresh = toolbox.population(n=20)
            frits = pool.map(_proxy, fresh)
            for ind, f in zip(fresh, frits):
                ind.fitness.values = f
            pop = tools.selBest(pop, len(pop)-20) + fresh
            stag = 0
            toolbox.unregister('mutate')
            toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=sigma, indpb=0.3)

        if verbose and gen % 10 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / gen * (generations - gen)
            print(f"Gen {gen:3d}: Max={rec['max']:7.3f} | Avg={rec['avg']:7.3f} | "
                  f"sigma={sigma:.2f} | {elapsed/60:.1f}dk | ~{eta/60:.0f}dk")

    pool.close(); pool.join()
    elapsed = time.time() - t0
    if verbose:
        print(f"\n✅ Bitti! ({elapsed/60:.1f} dk) | Best: {hof[0].fitness.values[0]:.4f}")

    best_params = decode_genome(list(hof[0]))

    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    out  = os.path.join(output_dir, f'silver_momentum_{ts}.json')
    data = {
        'timestamp':   ts,
        'fitness':     float(hof[0].fitness.values[0]),
        'kelly_garch': use_kelly_garch,
        'best_params': best_params.__dict__,
        'top5': [decode_genome(list(h)).__dict__ for h in list(hof)[:5]],
        'stats_tail': stats_log[-50:],
    }
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    if verbose:
        print(f"📁 Kaydedildi: {out}")

    return best_params, float(hof[0].fitness.values[0]), stats_log, hof
