"""
DEAP OPTİMİZATÖR — Precompute Cache + Multiprocessing
"""

import random
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Tuple, List
import time
import multiprocessing
import importlib

from deap import base, creator, tools

if not hasattr(creator, 'FitnessMax'):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, 'Individual'):
    creator.create("Individual", list, fitness=creator.FitnessMax)

GENOME_SIZE = 11
GENE_BOUNDS = [
    (2.0,  6.0),    # score_threshold — düşürüldü (2→6)
    (50,   200),    # hurst_window
    (10,   30),     # chop_period
    (5,    25),     # ob_lookback
    (20,   80),     # resistance_lookback
    (0.003,0.02),   # dca_step_pct
    (0.008,0.04),   # target_profit_pct
    (0.010,0.04),   # stop_loss_pct
    (0.05, 0.25),   # position_pct
    (1,    3),      # max_layers
    (0,    1),      # use_session_filter
]


def decode_genome(genome: List[float]) -> dict:
    p = {}
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        raw  = genome[i] if i < len(genome) else 0.5
        p[i] = lo + abs(raw % 1.0) * (hi - lo)
    return {
        'score_threshold':      float(np.clip(p[0], 2.0, 6.0)),
        'hurst_window':         int(np.clip(p[1], 50, 200)),
        'chop_period':          int(np.clip(p[2], 10, 30)),
        'ob_lookback':          int(np.clip(p[3], 5, 25)),
        'resistance_lookback':  int(np.clip(p[4], 20, 80)),
        'dca_step_pct':         float(np.clip(p[5], 0.003, 0.02)),
        'target_profit_pct':    float(np.clip(p[6], 0.008, 0.04)),
        'stop_loss_pct':        float(np.clip(p[7], 0.010, 0.04)),
        'position_pct':         float(np.clip(p[8], 0.05, 0.25)),
        'max_layers':           int(np.clip(p[9], 1, 3)),
        'use_session_filter':   bool(p[10] > 0.5),
    }


# Global worker state
_SPLITS      = None
_PRECOMPUTED = None
_INIT_CAP    = 1_000
_COMMISSION  = 0.0001
_SLIPPAGE    = 0.0002


def _init_worker(splits, init_cap, commission, slippage):
    global _SPLITS, _PRECOMPUTED, _INIT_CAP, _COMMISSION, _SLIPPAGE
    import importlib
    regime_mod      = importlib.import_module("01_regime_detector")
    DataPrecomputer = regime_mod.DataPrecomputer

    _SPLITS     = splits
    _INIT_CAP   = init_cap
    _COMMISSION = commission
    _SLIPPAGE   = slippage

    # Her split için precompute bir kez yap
    _PRECOMPUTED = []
    for sp in splits:
        _PRECOMPUTED.append(DataPrecomputer(sp['test']))


def evaluate_genome(genome: list) -> Tuple[float,]:
    global _SPLITS, _PRECOMPUTED, _INIT_CAP, _COMMISSION, _SLIPPAGE
    try:
        params = decode_genome(genome)
        import importlib
        bt_mod = importlib.import_module("03_backtest_engine")
        Backtester = bt_mod.InstitutionalBacktester

        bt = Backtester(
            initial_capital     = _INIT_CAP,
            commission          = _COMMISSION,
            slippage            = _SLIPPAGE,
            hurst_window        = params['hurst_window'],
            chop_period         = params['chop_period'],
            ob_lookback         = params['ob_lookback'],
            resistance_lookback = params['resistance_lookback'],
            score_threshold     = params['score_threshold'],
            max_layers          = params['max_layers'],
            dca_step_pct        = params['dca_step_pct'],
            target_profit_pct   = params['target_profit_pct'],
            stop_loss_pct       = params['stop_loss_pct'],
            position_pct        = params['position_pct'],
            use_session_filter  = params['use_session_filter'],
        )

        scores = []
        for i, sp in enumerate(_SPLITS):
            precomp = _PRECOMPUTED[i] if _PRECOMPUTED else None
            r       = bt.run(sp['test'], precomp=precomp)
            scores.append(r.fitness())

        tm   = float(np.mean(scores))
        ts   = float(np.std(scores))
        cons = 1 - (ts / (abs(tm) + 1e-10))
        fit  = tm * (0.6 + 0.4 * max(0, cons))
        return (float(np.clip(fit, -999, 100)),)

    except Exception:
        return (-999.0,)


def _get_fitness(ind):
    return evaluate_genome(list(ind))


class InstitutionalDEAPOptimizer:

    def __init__(self, df: pd.DataFrame, config: dict = None):
        self.df     = df
        self.config = config or self._default_config()
        self._prepare_splits()
        self.toolbox          = base.Toolbox()
        self.generation_stats = []
        self.start_time       = None

    def _default_config(self):
        return {
            'population_size':   50,
            'n_generations':     150,
            'hall_of_fame_size': 15,
            'crossover_prob':    0.70,
            'mutation_prob':     0.25,
            'tournament_size':   5,
            'initial_capital':   1_000,
            'commission':        0.0001,
            'slippage':          0.0002,
            'train_pct':         0.70,
            'n_splits':          4,
            'stagnation_limit':  15,
            'n_jobs':            11,
            'verbose':           True,
            'log_every':         5,
        }

    def _prepare_splits(self):
        total = len(self.df)
        n     = self.config.get('n_splits', 4)
        sz    = total // n
        tp    = self.config.get('train_pct', 0.70)
        self.splits = []
        for i in range(n - 1):
            s  = i * sz
            e  = s + sz
            te = s + int(sz * tp)
            self.splits.append({
                'train': self.df.iloc[s:te].copy(),
                'test':  self.df.iloc[te:e].copy(),
            })

    def _setup_toolbox(self, sigma=0.3):
        tb = self.toolbox
        tb.register("attr_float", random.uniform, 0.0, 1.0)
        tb.register("individual", tools.initRepeat, creator.Individual,
                    tb.attr_float, n=GENOME_SIZE)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("evaluate",   evaluate_genome)
        tb.register("select",     tools.selTournament, tournsize=self.config['tournament_size'])
        tb.register("mate",       tools.cxBlend, alpha=0.3)
        tb.register("mutate",     tools.mutGaussian, mu=0, sigma=sigma, indpb=0.3)

    def run(self):
        self.start_time = time.time()
        cfg    = self.config
        n_jobs = min(cfg.get('n_jobs', 1), multiprocessing.cpu_count())
        self._setup_toolbox(0.3)

        if cfg['verbose']:
            print("=" * 62)
            print("🏛️  INSTITUTIONAL DEAP v4")
            print(f"   Popülasyon : {cfg['population_size']}")
            print(f"   Nesil      : {cfg['n_generations']}")
            print(f"   Veri       : {len(self.df):,} bar")
            print(f"   CPU        : {n_jobs} core 🚀")
            print(f"   Splits     : {len(self.splits)} walk-forward")
            print("=" * 62)

        pool = multiprocessing.Pool(
            processes   = n_jobs,
            initializer = _init_worker,
            initargs    = (self.splits, cfg['initial_capital'],
                           cfg['commission'], cfg['slippage'])
        )

        try:
            pop = self.toolbox.population(n=cfg['population_size'])
            hof = tools.HallOfFame(cfg['hall_of_fame_size'])
            sf  = tools.Statistics(
                key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -999
            )
            sf.register("max", np.max)
            sf.register("avg", np.mean)
            sf.register("std", np.std)

            fits = pool.map(_get_fitness, pop)
            for ind, fit in zip(pop, fits):
                ind.fitness.values = fit
            hof.update(pop)

            best   = hof[0].fitness.values[0]
            stag   = 0
            sigma  = 0.3
            rec    = sf.compile(pop)
            if cfg['verbose']:
                print(f"Gen   0: Max={rec['max']:.3f} | Avg={rec['avg']:.3f}")
            self.generation_stats.append({'gen':0,'max_fitness':rec['max'],'avg_fitness':rec['avg'],'std_fitness':rec['std']})

            for gen in range(1, cfg['n_generations']+1):
                off = list(map(deepcopy, self.toolbox.select(pop, len(pop))))
                for c1, c2 in zip(off[::2], off[1::2]):
                    if random.random() < cfg['crossover_prob']:
                        self.toolbox.mate(c1, c2)
                        del c1.fitness.values, c2.fitness.values
                for m in off:
                    if random.random() < cfg['mutation_prob']:
                        self.toolbox.mutate(m)
                        del m.fitness.values
                        for i in range(len(m)):
                            m[i] = float(np.clip(m[i], 0.0, 1.0))

                inv  = [ind for ind in off if not ind.fitness.valid]
                if inv:
                    fits = pool.map(_get_fitness, inv)
                    for ind, fit in zip(inv, fits):
                        ind.fitness.values = fit

                pop[:] = tools.selBest(pop, 8) + off[8:]
                hof.update(pop)
                rec = sf.compile(pop)
                self.generation_stats.append({'gen':gen,'max_fitness':rec['max'],'avg_fitness':rec['avg'],'std_fitness':rec['std']})

                curr = hof[0].fitness.values[0]
                if curr > best + 0.001:
                    best = curr; stag = 0
                    sigma = max(0.1, sigma * 0.95)
                else:
                    stag += 1

                if stag >= cfg['stagnation_limit']:
                    sigma = min(0.6, sigma * 1.5)
                    ni = self.toolbox.population(n=15)
                    nf = pool.map(_get_fitness, ni)
                    for ind, fit in zip(ni, nf):
                        ind.fitness.values = fit
                    pop = tools.selBest(pop, len(pop)-15) + ni
                    stag = 0
                    self._setup_toolbox(sigma)
                    if cfg['verbose']:
                        print(f"  ⚡ Stagnation! Sigma={sigma:.2f}")

                if cfg['verbose'] and gen % cfg['log_every'] == 0:
                    el  = time.time() - self.start_time
                    eta = (el/gen) * (cfg['n_generations']-gen)
                    print(f"Gen {gen:3d}: Max={rec['max']:7.3f} | Avg={rec['avg']:7.3f} | "
                          f"Sigma={sigma:.2f} | {el:.0f}s | ~{eta/60:.0f}dk kaldı")

        finally:
            pool.close()
            pool.join()

        best_params = decode_genome(list(hof[0]))
        el = time.time() - self.start_time

        if cfg['verbose']:
            print(f"\n✅ Tamamlandı! ({el/60:.1f} dakika)")
            print(f"   Fitness: {hof[0].fitness.values[0]:.4f}")
            print("\n🏆 EN İYİ PARAMETRELER:")
            for k, v in best_params.items():
                print(f"   {k:<25} : {v}")

        return best_params, {'stats': self.generation_stats, 'hof': hof}