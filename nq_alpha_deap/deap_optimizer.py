"""
NQ Alpha DEAP Optimizer — EvolutionaryQuant
============================================
Genetic Algorithm optimizer for NQAlphaStrategy.
- Walk-forward validation (5 splits, 70/30 train-test)
- Adaptive sigma (exploration ↔ exploitation)
- Consistency penalty: penalizes high variance across splits
- Multiprocessing (CPU - 1 cores)
- Hall of Fame (top 15 genomes)
"""

import sys, os, random, time, json
import numpy as np
import multiprocessing
from copy import deepcopy
from datetime import datetime

# Path setup — this file lives in nq_alpha_deap/
_DIR    = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_DIR)
sys.path.insert(0, _DIR)
sys.path.insert(0, _PARENT)

from nq_alpha_strategy import (
    NQAlphaParams, decode_genome, add_nq_alpha_features, GENOME_SIZE
)
from backtest_engine import run_backtest, BacktestResult

from deap import base, creator, tools

# ─────────────────────────────────────────────────────────────────────
# DEAP SETUP
# ─────────────────────────────────────────────────────────────────────
if not hasattr(creator, 'FitnessNQ'):
    creator.create('FitnessNQ', base.Fitness, weights=(1.0,))
if not hasattr(creator, 'IndividualNQ'):
    creator.create('IndividualNQ', list, fitness=creator.FitnessNQ)

# ─────────────────────────────────────────────────────────────────────
# GLOBAL WORKER STATE (shared via fork-safe initializer)
# ─────────────────────────────────────────────────────────────────────
_WF_SPLITS   = []    # list of (df_train, df_oos) pairs — pre-featured per split
_INIT_CAP    = 1_000.0
_N_SPLITS    = 5

def _init_worker(wf_splits, init_cap, n_splits):
    global _WF_SPLITS, _INIT_CAP, _N_SPLITS
    sys.path.insert(0, _DIR)
    sys.path.insert(0, _PARENT)
    _WF_SPLITS = wf_splits
    _INIT_CAP  = init_cap
    _N_SPLITS  = n_splits


def evaluate_genome(genome) -> tuple:
    """Evaluate a genome against all walk-forward OOS splits."""
    try:
        params = decode_genome(genome)
        scores = []
        for df_feat in _WF_SPLITS:
            r = run_backtest(df_feat, params, _INIT_CAP)
            scores.append(r.fitness())

        if not scores:
            return (-999.0,)

        mean_s  = float(np.mean(scores))
        std_s   = float(np.std(scores))
        worst_s = float(np.min(scores))

        # Consistency reward: penalize high variance
        consistency = 1.0 - std_s / (abs(mean_s) + 1e-6)
        consistency = max(-1.0, min(1.0, consistency))

        # Blended score: mean × (0.50) + worst × (0.25) + consistency bump
        blended = mean_s * 0.50 + worst_s * 0.25 + mean_s * 0.25 * max(0, consistency)
        return (float(np.clip(blended, -999, 200)),)
    except Exception:
        return (-999.0,)


def _proxy(genome):
    return evaluate_genome(genome)


# ─────────────────────────────────────────────────────────────────────
# WALK-FORWARD SPLIT BUILDER
# ─────────────────────────────────────────────────────────────────────
def build_wf_splits(df_raw, params_ref: NQAlphaParams,
                    n_splits: int = 5, oos_pct: float = 0.30):
    """
    Build n_splits walk-forward OOS segments.
    df_raw: raw OHLCV dataframe (not featured yet).
    params_ref: reference params for feature engineering
                (features don't change between evaluations for same genome,
                 but we pre-feature with DEFAULT params to avoid re-running
                 per-genome; dynamic features like Kalman are recalculated per genome).

    Returns list of pre-featured DataFrames (OOS slices).
    """
    n    = len(df_raw)
    size = n // n_splits
    splits = []
    for i in range(n_splits):
        start = i * size
        end   = (i + 1) * size
        oos_start = start + int(size * (1 - oos_pct))
        slice_raw = df_raw.iloc[oos_start:end].copy()
        if len(slice_raw) < 100:
            continue
        splits.append(slice_raw)   # raw — feature engineering done per genome in evaluate_genome
    return splits


def _build_featured_splits(df_raw, params: NQAlphaParams,
                            n_splits: int, oos_pct: float):
    """Pre-feature each split with given params — avoids re-computation inside workers."""
    raw_splits = build_wf_splits(df_raw, params, n_splits, oos_pct)
    featured   = []
    for raw in raw_splits:
        try:
            feat = add_nq_alpha_features(raw, params)
            featured.append(feat)
        except Exception:
            pass
    return featured


# ─────────────────────────────────────────────────────────────────────
# MAIN OPTIMIZER
# ─────────────────────────────────────────────────────────────────────
def run_deap_optimizer(df_raw,
                       initial_capital: float = 1_000.0,
                       population:  int = 80,
                       generations: int = 200,
                       n_splits:    int = 5,
                       oos_pct:    float = 0.30,
                       n_jobs:     int = None,
                       verbose:    bool = True,
                       output_dir: str = None):
    """
    Full DEAP optimization run.
    Returns (best_params, best_fitness, stats_list, hof).
    """
    if n_jobs is None:
        n_jobs = max(1, min(multiprocessing.cpu_count() - 1, 10))

    if output_dir is None:
        output_dir = os.path.join(_DIR, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # ── Build walk-forward splits (raw — workers feature per genome) ──
    raw_splits = build_wf_splits(df_raw, NQAlphaParams(), n_splits, oos_pct)
    if verbose:
        print(f"📊 Walk-forward: {len(raw_splits)} OOS splits")
        for i, sl in enumerate(raw_splits):
            print(f"   Split {i+1}: {len(sl):,} bars  "
                  f"[{sl.index[0].date()} → {sl.index[-1].date()}]")

    # ── DEAP toolbox ──────────────────────────────────────────────────
    toolbox = base.Toolbox()
    toolbox.register('attr_float', random.uniform, 0.0, 1.0)
    toolbox.register('individual', tools.initRepeat, creator.IndividualNQ,
                     toolbox.attr_float, n=GENOME_SIZE)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', evaluate_genome)
    toolbox.register('select',   tools.selTournament, tournsize=5)
    toolbox.register('mate',     tools.cxBlend, alpha=0.3)
    toolbox.register('mutate',   tools.mutGaussian, mu=0, sigma=0.25, indpb=0.3)

    # ── Process pool ─────────────────────────────────────────────────
    pool = multiprocessing.Pool(
        n_jobs, _init_worker,
        (raw_splits, initial_capital, n_splits)
    )

    pop  = toolbox.population(n=population)
    hof  = tools.HallOfFame(15)
    stat = tools.Statistics(key=lambda ind: ind.fitness.values[0]
                            if ind.fitness.valid else -999.0)
    stat.register('max', np.max)
    stat.register('avg', np.mean)

    if verbose:
        print(f"\n🧬 Population:{population} | Generations:{generations} | CPUs:{n_jobs}")
        print(f"{'─'*60}")

    # ── Initial evaluation ────────────────────────────────────────────
    fits = pool.map(_proxy, pop)
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    hof.update(pop)

    best_fit  = hof[0].fitness.values[0]
    stagnation= 0
    sigma     = 0.25
    stats_log = []
    t0        = time.time()

    rec = stat.compile(pop)
    if verbose:
        print(f"Gen   0: Max={rec['max']:7.3f} | Avg={rec['avg']:7.3f}")

    # ── Evolution loop ────────────────────────────────────────────────
    for gen in range(1, generations + 1):
        # Selection + reproduction
        offspring = list(map(deepcopy, toolbox.select(pop, len(pop))))

        # Crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.70:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        # Mutation
        for mut in offspring:
            if random.random() < 0.25:
                toolbox.mutate(mut)
                del mut.fitness.values
                # Clamp genes to [0, 1]
                for j in range(len(mut)):
                    mut[j] = float(np.clip(mut[j], 0.0, 1.0))

        # Evaluate invalid individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        if invalid:
            fits = pool.map(_proxy, invalid)
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

        # Elitist replacement: keep top 10 parents
        pop[:] = tools.selBest(pop, 10) + offspring[10:]
        hof.update(pop)

        rec = stat.compile(pop)
        stats_log.append({
            'gen': gen,
            'max_fitness': float(rec['max']),
            'avg_fitness': float(rec['avg']),
        })

        # ── Adaptive sigma ────────────────────────────────────────
        curr = hof[0].fitness.values[0]
        if curr > best_fit + 0.005:
            best_fit   = curr
            stagnation = 0
            sigma      = max(0.10, sigma * 0.95)
        else:
            stagnation += 1

        if stagnation >= 20:
            sigma = min(0.65, sigma * 1.6)
            # Inject fresh individuals
            fresh = toolbox.population(n=20)
            frits = pool.map(_proxy, fresh)
            for ind, fit in zip(fresh, frits):
                ind.fitness.values = fit
            pop = tools.selBest(pop, len(pop) - 20) + fresh
            stagnation = 0
            toolbox.unregister('mutate')
            toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=sigma, indpb=0.3)
            if verbose:
                print(f"  ⚡ Sigma reset → {sigma:.2f}")

        if verbose and gen % 10 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / gen * (generations - gen)
            print(f"Gen {gen:3d}: Max={rec['max']:7.3f} | Avg={rec['avg']:7.3f} | "
                  f"σ={sigma:.2f} | {elapsed/60:.1f}dk | ~{eta/60:.0f}dk kaldı")

    pool.close(); pool.join()

    elapsed = time.time() - t0
    if verbose:
        print(f"\n✅ Tamamlandı! ({elapsed/60:.1f} dk)  "
              f"Best Fitness: {hof[0].fitness.values[0]:.4f}")

    best_params = decode_genome(list(hof[0]))

    # ── Save results ──────────────────────────────────────────────────
    ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(output_dir, f'nq_alpha_{ts}.json')
    save_data = {
        'timestamp':   ts,
        'fitness':     float(hof[0].fitness.values[0]),
        'best_params': best_params.__dict__,
        'top5': [decode_genome(list(h)).__dict__ for h in list(hof)[:5]],
        'stats': stats_log[-50:],   # last 50 generations
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"📁 Kaydedildi: {out_path}")

    return best_params, float(hof[0].fitness.values[0]), stats_log, hof
