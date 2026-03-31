"""
============================================================
LiquidityEdge XAU/USD — DEAP Genetic Algorithm Optimizer
============================================================
Backtest engine'deki 15 parametreyi evrimsel süreçle optimize eder.

Çalıştırma:
    python deap_optimizer.py --data XAUUSD_1h.csv
    python deap_optimizer.py --data cache/ --tf 1h --generations 200
============================================================
"""

import random
import time
import numpy as np
import multiprocessing
import sys
import os
from copy import deepcopy
from typing import Tuple, List

# ── DEAP ─────────────────────────────────────────────────
from deap import base, creator, tools

# ── Backtest Engine ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_engine import (
    GENE_BOUNDS, GENOME_SIZE, GENOME_NAMES,
    decode, run_backtest, DEFAULT_PARAMS
)

# ── DEAP creator (global, bir kez) ───────────────────────
if not hasattr(creator, 'FitnessMax'):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, 'Individual'):
    creator.create("Individual", list, fitness=creator.FitnessMax)

# ── Worker globals (multiprocessing için) ─────────────────
_SPLITS   = None   # [(df_test1, df_test2, ...)]
_INIT_CAP = 1_000.0


def _init_worker(splits, init_cap):
    global _SPLITS, _INIT_CAP
    _SPLITS   = splits
    _INIT_CAP = init_cap


def evaluate_genome(genome: list) -> Tuple[float,]:
    """Bir genome'u N walk-forward split üzerinde değerlendir."""
    global _SPLITS, _INIT_CAP
    try:
        p = decode(genome)
        scores = []
        for df_test in _SPLITS:
            r = run_backtest(df_test, p, _INIT_CAP)
            scores.append(r.fitness())

        tm   = float(np.mean(scores))
        ts   = float(np.std(scores))
        # Consistency bonusu — tutarsız strateji cezalandırılır
        cons = 1 - (ts / (abs(tm) + 1e-10))
        fit  = tm * (0.6 + 0.4 * max(0.0, cons))
        return (float(np.clip(fit, -999.0, 200.0)),)
    except Exception:
        return (-999.0,)


def _get_fitness(ind):
    return evaluate_genome(list(ind))


def _setup_toolbox(toolbox, sigma: float = 0.3, tournament_size: int = 5):
    """DEAP toolbox operatörlerini kur."""
    if hasattr(toolbox, 'attr_float'):
        toolbox.unregister('attr_float')
        toolbox.unregister('individual')
        toolbox.unregister('population')
        toolbox.unregister('evaluate')
        toolbox.unregister('select')
        toolbox.unregister('mate')
        toolbox.unregister('mutate')

    toolbox.register("attr_float",  random.uniform, 0.0, 1.0)
    toolbox.register("individual",  tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=GENOME_SIZE)
    toolbox.register("population",  tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate",    evaluate_genome)
    toolbox.register("select",      tools.selTournament, tournsize=tournament_size)
    toolbox.register("mate",        tools.cxBlend, alpha=0.3)
    toolbox.register("mutate",      tools.mutGaussian, mu=0, sigma=sigma, indpb=0.3)


def run_deap(df, config: dict) -> Tuple[dict, dict]:
    """
    Ana DEAP optimizasyon döngüsü.
    Returns: (best_params, history)
    """
    cfg    = config
    n_jobs = min(cfg.get('n_jobs', 8), multiprocessing.cpu_count())

    # Walk-forward splits
    n    = cfg.get('n_splits', 5)
    tp   = cfg.get('train_pct', 0.70)
    sz   = len(df) // n
    splits = []
    for i in range(n - 1):
        s  = i * sz
        e  = s + sz
        te = s + int(sz * tp)
        splits.append(df.iloc[te:e].copy())

    toolbox = base.Toolbox()
    _setup_toolbox(toolbox, sigma=0.3,
                   tournament_size=cfg.get('tournament_size', 5))

    print("=" * 64)
    print("💧 LIQUIDITYEDGE XAU/USD — DEAP OPTİMİZASYONU")
    print(f"   Veri         : {len(df):,} bar")
    print(f"   Splits       : {len(splits)} walk-forward")
    print(f"   Popülasyon   : {cfg['population_size']}")
    print(f"   Nesil        : {cfg['n_generations']}")
    print(f"   CPU          : {n_jobs} core 🚀")
    print(f"   Genome       : {GENOME_SIZE} gen")
    print("=" * 64)
    print("  Optimize  :", ', '.join(GENOME_NAMES[:7]))
    print("            :", ', '.join(GENOME_NAMES[7:]))
    print("-" * 64)

    pool = multiprocessing.Pool(
        processes=n_jobs,
        initializer=_init_worker,
        initargs=(splits, cfg.get('initial_capital', 1_000.0))
    )

    pop_size = cfg['population_size']
    n_gen    = cfg['n_generations']
    hof_size = cfg.get('hall_of_fame_size', 20)
    cx_prob  = cfg.get('crossover_prob', 0.70)
    mut_prob = cfg.get('mutation_prob', 0.25)
    stag_lim = cfg.get('stagnation_limit', 25)
    log_ev   = cfg.get('log_every', 10)
    stats_log = []

    try:
        pop  = toolbox.population(n=pop_size)
        hof  = tools.HallOfFame(hof_size)
        sf   = tools.Statistics(
            key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -999)
        sf.register("max", np.max)
        sf.register("avg", np.mean)
        sf.register("std", np.std)

        # Gen 0
        fits = pool.map(_get_fitness, pop)
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit
        hof.update(pop)

        rec  = sf.compile(pop)
        best = hof[0].fitness.values[0]
        stag = 0
        sigma = 0.3
        t0   = time.time()

        print(f"Gen   0: Max={rec['max']:7.3f} | Avg={rec['avg']:7.3f}")
        stats_log.append({'gen': 0, 'max': rec['max'], 'avg': rec['avg']})

        for gen in range(1, n_gen + 1):
            # Selection + crossover + mutation
            offspring = list(map(deepcopy, toolbox.select(pop, pop_size)))
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cx_prob:
                    toolbox.mate(c1, c2)
                    del c1.fitness.values, c2.fitness.values
            for m in offspring:
                if random.random() < mut_prob:
                    toolbox.mutate(m)
                    del m.fitness.values
                    for j in range(len(m)):
                        m[j] = float(np.clip(m[j], 0.0, 1.0))

            # Sadece değerlendirmesi silinenleri değerlendir
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            if invalid:
                fits = pool.map(_get_fitness, invalid)
                for ind, fit in zip(invalid, fits):
                    ind.fitness.values = fit

            # Elite + random replacement
            pop[:] = tools.selBest(pop, 10) + offspring[10:]
            hof.update(pop)
            rec = sf.compile(pop)
            stats_log.append({
                'gen': gen, 'max': rec['max'],
                'avg': rec['avg'], 'std': rec['std']
            })

            # Stagnation kontrolü
            curr = hof[0].fitness.values[0]
            if curr > best + 0.001:
                best  = curr
                stag  = 0
                sigma = max(0.08, sigma * 0.95)
            else:
                stag += 1

            if stag >= stag_lim:
                sigma = min(0.65, sigma * 1.6)
                # Taze kan injection
                ni = toolbox.population(n=25)
                nf = pool.map(_get_fitness, ni)
                for ind, fit in zip(ni, nf):
                    ind.fitness.values = fit
                pop = tools.selBest(pop, pop_size - 25) + ni
                stag = 0
                _setup_toolbox(toolbox, sigma=sigma,
                               tournament_size=cfg.get('tournament_size', 5))
                if cfg.get('verbose', True):
                    print(f"  ⚡ Sigma reset: {sigma:.3f}")

            if cfg.get('verbose', True) and gen % log_ev == 0:
                el  = time.time() - t0
                eta = (el / gen) * (n_gen - gen)
                print(f"Gen {gen:3d}: Max={rec['max']:7.3f} | "
                      f"Avg={rec['avg']:7.3f} | "
                      f"Sigma={sigma:.3f} | "
                      f"{el:.0f}s | ~{eta/60:.1f}dk")

    finally:
        pool.close()
        pool.join()

    el = time.time() - t0
    best_params = decode(list(hof[0]))

    if cfg.get('verbose', True):
        print(f"\n✅ Tamamlandı! ({el/60:.1f} dakika)")
        print(f"   En iyi fitness : {hof[0].fitness.values[0]:.4f}")
        print("\n" + "═" * 64)
        print("🏆 EN İYİ LiquidityEdge PARAMETRELER:")
        print("═" * 64)
        for k, v in best_params.items():
            print(f"   {k:<22} : {v}")
        print("═" * 64)

    return best_params, {
        'stats': stats_log,
        'hof':   hof,
        'elapsed_sec': el,
    }
