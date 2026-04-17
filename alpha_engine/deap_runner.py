"""
Alpha Engine — Universal DEAP Optimizer
=========================================
One optimizer that works for any strategy module.
Strategy module must expose:
  - GENOME_SIZE : int
  - GENE_BOUNDS : list of (lo, hi) tuples
  - decode_genome(genome) -> Params
  - backtest_from_raw(df, params, initial_capital) -> Result with .fitness()

Usage:
    from alpha_engine.deap_runner import run_deap

    best_params, best_fit, stats = run_deap(
        module     = gold_sniper_bt,   # any strategy module
        df_train   = df,
        capital    = 1000.0,
        population = 80,
        generations= 150,
        n_jobs     = 4,
    )
"""

import os
import sys
import time
import json
import random
import numpy as np
from copy import deepcopy
from typing import Any, Optional

try:
    from deap import base, creator, tools, algorithms
    _DEAP_OK = True
except ImportError:
    _DEAP_OK = False


# ─────────────────────────────────────────────────────────────────────
# EVAL FUNCTION — closure-based (no pickle required)
# ─────────────────────────────────────────────────────────────────────

def _make_eval_fn(module, df, capital, n_splits, max_train_bars=None, n_eval_windows=1):
    """Return a closure that evaluates one genome. No pickling needed.

    max_train_bars:  if set, randomly sample a window of N bars from the full df
                     each evaluation call — covers all market regimes across the
                     full training set while keeping speed constant.
    n_eval_windows:  number of random windows to average per eval (reduces noise).
                     Setting this to 3+ dramatically improves fitness signal quality.
    """
    n = len(df)

    def _eval(individual):
        try:
            genome = list(individual)
            params = module.decode_genome(genome)

            # Multi-window evaluation: average fitness over n_eval_windows random slices.
            # This eliminates the high variance of single-window evaluation and allows
            # DEAP to reliably distinguish good params from regime-overfitted ones.
            if max_train_bars and n > max_train_bars and n_eval_windows > 1:
                fits = []
                for _ in range(n_eval_windows):
                    start = random.randint(0, n - max_train_bars)
                    df_use = df.iloc[start:start + max_train_bars]
                    r = module.backtest_from_raw(df_use, params, capital)
                    fits.append(r.fitness())
                valid = [f for f in fits if f > -990.0]
                if not valid:
                    return (-999.0,)
                return (float(np.mean(valid)),)

            # Single-window path (n_eval_windows=1 or small dataset)
            if max_train_bars and n > max_train_bars:
                start = random.randint(0, n - max_train_bars)
                df_use = df.iloc[start:start + max_train_bars]
            else:
                df_use = df

            if n_splits <= 1:
                r = module.backtest_from_raw(df_use, params, capital)
                return (r.fitness(),)

            chunk = len(df_use) // n_splits
            fits  = []
            for i in range(n_splits):
                sl = df_use.iloc[i * chunk:(i + 1) * chunk]
                if len(sl) < 50:
                    continue
                r = module.backtest_from_raw(sl, params, capital)
                fits.append(r.fitness())

            # Average only VALID splits (skip -999 = insufficient trades).
            valid = [f for f in fits if f > -990.0]
            if not valid:
                return (-999.0,)
            return (float(np.mean(valid)),)
        except Exception:
            return (-999.0,)
    return _eval


# ─────────────────────────────────────────────────────────────────────
# DEAP SETUP
# ─────────────────────────────────────────────────────────────────────

def _setup_deap(module, eval_fn):
    """Create DEAP toolbox for a strategy module."""
    if not _DEAP_OK:
        raise ImportError("pip install deap")

    gene_bounds = module.GENE_BOUNDS

    # Re-create creator if needed (avoids DEAP clash if called multiple times)
    if not hasattr(creator, 'FitnessMax_Alpha'):
        creator.create('FitnessMax_Alpha', base.Fitness, weights=(1.0,))
    if not hasattr(creator, 'Individual_Alpha'):
        creator.create('Individual_Alpha', list, fitness=creator.FitnessMax_Alpha)

    toolbox = base.Toolbox()

    def make_individual():
        genes = [random.uniform(lo, hi) for lo, hi in gene_bounds]
        return creator.Individual_Alpha(genes)

    toolbox.register('individual', make_individual)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    toolbox.register('evaluate', eval_fn)
    toolbox.register('map',    map)          # builtin map — no pickling needed
    toolbox.register('mate',   tools.cxBlend, alpha=0.4)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=0.1, indpb=0.3)
    toolbox.register('select', tools.selTournament, tournsize=4)

    return toolbox


def _clamp_individual(individual, gene_bounds):
    """Clamp genes to valid bounds after mutation/crossover."""
    for i, (lo, hi) in enumerate(gene_bounds):
        individual[i] = float(np.clip(individual[i], lo, hi))
    return individual


# ─────────────────────────────────────────────────────────────────────
# MAIN OPTIMIZER
# ─────────────────────────────────────────────────────────────────────

def run_deap(
    module,
    df_train:       Any,
    capital:        float = 1_000.0,
    population:     int   = 80,
    generations:    int   = 100,
    n_splits:       int   = 1,
    n_jobs:         Optional[int] = None,   # kept for API compat, ignored
    max_train_bars: Optional[int] = None,   # None → use module.MAX_TRAIN_BARS or 8_000
    verbose:        bool  = True,
    output_dir:     Optional[str] = None,
    strategy_name:  str   = 'strategy',
):
    """
    Run DEAP genetic optimization on any strategy module.

    Returns: (best_params, best_fitness, stats_list, hall_of_fame)
    """
    if not _DEAP_OK:
        raise ImportError("pip install deap")

    # Resolve max_train_bars, n_splits, n_eval_windows from module constants when not passed
    if max_train_bars is None:
        max_train_bars = getattr(module, 'MAX_TRAIN_BARS', 8_000)
    # Module can override n_splits (e.g. M1 scalper uses N_SPLITS=1 for speed)
    n_splits = getattr(module, 'N_SPLITS', n_splits)
    # N_EVAL_WINDOWS: how many random windows to average per fitness eval (reduces noise)
    n_eval_windows = getattr(module, 'N_EVAL_WINDOWS', 1)

    # Build a closure so no pickling is needed — module stays in this process
    eval_fn = _make_eval_fn(module, df_train, capital, n_splits, max_train_bars, n_eval_windows)
    toolbox = _setup_deap(module, eval_fn)
    gene_bounds = module.GENE_BOUNDS

    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(10)

    stats_log = []
    t0 = time.time()

    actual_bars = min(len(df_train), max_train_bars) if max_train_bars else len(df_train)
    if verbose:
        print(f"\n  Strategy: {strategy_name}", flush=True)
        print(f"  Genome  : {module.GENOME_SIZE} genes  Bars: {actual_bars:,}", flush=True)
        print(f"  Pop/Gen : {population}/{generations}  Splits: {n_splits}  EvalWins: {n_eval_windows}", flush=True)
        print(f"  {'Gen':>4}  {'Max':>8}  {'Avg':>8}  {'Time':>7}", flush=True)
        print(f"  {'-'*36}", flush=True)

    for gen in range(1, generations + 1):
        # Evaluate
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        hof.update(pop)
        best_fit = max(ind.fitness.values[0] for ind in pop)
        avg_fit  = float(np.mean([ind.fitness.values[0] for ind in pop]))

        stats_log.append({'gen': gen, 'max_fitness': best_fit, 'avg_fitness': avg_fit})

        if verbose and (gen % 10 == 0 or gen == 1 or gen == generations):
            elapsed = time.time() - t0
            print(f"  {gen:>4}  {best_fit:>8.3f}  {avg_fit:>8.3f}  {elapsed:>6.1f}s")

        # Evolve
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(deepcopy, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                _clamp_individual(child1, gene_bounds)
                _clamp_individual(child2, gene_bounds)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                _clamp_individual(mutant, gene_bounds)
                del mutant.fitness.values

        # Re-evaluate offspring without fitness
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        new_fits = list(toolbox.map(toolbox.evaluate, invalid))
        for ind, fit in zip(invalid, new_fits):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        # Early stop if all -999
        if gen >= 10 and best_fit == -999.0:
            if verbose:
                print(f"  Early stop: fitness stuck at -999 after {gen} gens")
            break

    best_genome = list(hof[0])
    best_params = module.decode_genome(best_genome)
    best_fit    = hof[0].fitness.values[0]

    if verbose:
        print(f"\n  Best fitness : {best_fit:.4f}")

    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        out = {
            'strategy':   strategy_name,
            'timestamp':  ts,
            'fitness':    best_fit,
            'best_params': vars(best_params) if hasattr(best_params, '__dict__') else {},
            'top5': [
                vars(module.decode_genome(list(ind))) if hasattr(module.decode_genome(list(ind)), '__dict__') else {}
                for ind in list(hof)[:5]
            ],
            'stats': stats_log,
        }
        path = os.path.join(output_dir, f'{strategy_name}_{ts}.json')
        with open(path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        if verbose:
            print(f"  Saved: {path}")

    return best_params, best_fit, stats_log, hof


# ─────────────────────────────────────────────────────────────────────
# CONVENIENCE: RUN ALL STRATEGIES
# ─────────────────────────────────────────────────────────────────────

def optimize_all(
    df_gold:     Any,
    df_silver:   Any,
    df_nq:       Any,
    capital:     float = 1_000.0,
    population:  int   = 60,
    generations: int   = 80,
    n_splits:    int   = 3,
    n_jobs:      Optional[int] = None,
    output_dir:  Optional[str] = None,
):
    """Optimize all 5 strategies sequentially. Returns dict of best_params."""
    _ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
    _ROOT_DIR   = os.path.dirname(_ENGINE_DIR)
    sys.path.insert(0, _ENGINE_DIR)
    sys.path.insert(0, _ROOT_DIR)

    import importlib
    results = {}

    strategies = [
        ('gold_sniper',  'gold_sniper_bt',  df_gold,   _ENGINE_DIR),
        ('gold_master',  'gold_master_bt',  df_gold,   _ENGINE_DIR),
        ('gold_scalper', 'gold_scalper_bt', df_gold,   _ENGINE_DIR),
        ('silver',       None,              df_silver, None),
        ('nq_alpha',     None,              df_nq,     None),
        ('nq_liquidity', 'nq_liquidity_bt', df_nq,     _ENGINE_DIR),
    ]

    for name, mod_name, df, search_dir in strategies:
        if df is None or (hasattr(df, 'empty') and df.empty):
            print(f"\n[WARN]  {name}: No data — skipping.")
            continue

        print(f"\n{'═'*60}")
        print(f"  OPTIMIZING: {name.upper()}")
        print(f"{'═'*60}")

        if mod_name:
            if search_dir:
                sys.path.insert(0, search_dir)
            mod = importlib.import_module(mod_name)
        elif name == 'silver':
            sys.path.insert(0, os.path.join(_ROOT_DIR, 'silver_reversion_deap'))
            # Wrap silver modules to match expected interface
            mod = _SilverModule()
        elif name == 'nq_alpha':
            sys.path.insert(0, os.path.join(_ROOT_DIR, 'nq_alpha_deap'))
            mod = _NQAlphaModule()
        else:
            continue

        try:
            best_params, best_fit, stats, hof = run_deap(
                module        = mod,
                df_train      = df,
                capital       = capital,
                population    = population,
                generations   = generations,
                n_splits      = n_splits,
                n_jobs        = n_jobs,
                verbose       = True,
                output_dir    = output_dir or os.path.join(_ROOT_DIR, 'alpha_engine', 'outputs'),
                strategy_name = name,
            )
            results[name] = best_params
            print(f"  [OK] {name}: fitness={best_fit:.3f}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback; traceback.print_exc()

    return results


# ─────────────────────────────────────────────────────────────────────
# ADAPTER WRAPPERS (make Silver + NQ Alpha compatible)
# ─────────────────────────────────────────────────────────────────────

class _SilverModule:
    """Adapter wrapping silver_reversion_deap to universal interface."""
    def __init__(self):
        import importlib, sys, os
        _r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(_r, 'silver_reversion_deap'))
        self._strat = importlib.import_module('silver_strategy')
        self._bt    = importlib.import_module('backtest_engine')
        self.GENOME_SIZE = self._strat.GENOME_SIZE
        self.GENE_BOUNDS = self._strat.GENE_BOUNDS

    def decode_genome(self, genome):
        return self._strat.decode_genome(genome)

    def backtest_from_raw(self, df, params, capital):
        return self._bt.backtest_from_raw(df, params, capital)


class _NQAlphaModule:
    """Adapter wrapping nq_alpha_deap to universal interface."""
    def __init__(self):
        import importlib, sys, os
        _r = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, os.path.join(_r, 'nq_alpha_deap'))
        self._strat = importlib.import_module('nq_alpha_strategy')
        self._bt    = importlib.import_module('backtest_engine')
        self.GENOME_SIZE = self._strat.GENOME_SIZE
        self.GENE_BOUNDS = self._strat.GENE_BOUNDS

    def decode_genome(self, genome):
        return self._strat.decode_genome(genome)

    def backtest_from_raw(self, df, params, capital):
        return self._bt.backtest_from_raw(df, params, capital)
