"""
DEAP Optimizer — GoldMaster Parametreleri
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

# ─────────────────────────────────────────────────────────
# GENOME — Her gen ne anlama geliyor
# ─────────────────────────────────────────────────────────
# [0]  min_taps         2-5      kaç dokunuş
# [1]  tap_atr_mult     0.3-1.0  dokunuş eşiği
# [2]  momentum_thresh  0.05-0.5 kırılım eşiği
# [3]  fvg_required     0 or 1   FVG şart mı
# [4]  target_atr_mult  1.5-6.0  TP çarpanı
# [5]  stop_atr_mult    0.5-2.5  SL çarpanı
# [6]  lot_size         0.01-0.2 başlangıç lot
# [7]  position_pct     0.05-0.3 sermaye yüzdesi
# [8]  growth_factor    0.1-0.5  compounding faktörü
# [9]  htf_window       24-100   HTF penceresi

GENE_BOUNDS = [
    (2,    5),      # min_taps
    (0.3,  1.0),    # tap_atr_mult
    (0.05, 0.50),   # momentum_thresh
    (0,    1),      # fvg_required
    (1.5,  6.0),    # target_atr_mult
    (0.5,  2.5),    # stop_atr_mult
    (0.01, 0.20),   # lot_size
    (0.05, 0.30),   # position_pct
    (0.10, 0.50),   # growth_factor
    (24,   100),    # htf_window
    (0.0,  0.8),    # meta_bias_threshold
]
GENOME_SIZE = len(GENE_BOUNDS)


def decode(genome: list) -> dict:
    p = {}
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        raw  = genome[i] if i < len(genome) else 0.5
        p[i] = lo + abs(raw % 1.0) * (hi - lo)
    return {
        'min_taps':         int(np.clip(p[0], 2, 5)),
        'tap_atr_mult':     float(np.clip(p[1], 0.3, 1.0)),
        'momentum_thresh':  float(np.clip(p[2], 0.05, 0.5)),
        'fvg_required':     bool(p[3] > 0.5),
        'target_atr_mult':  float(np.clip(p[4], 1.5, 6.0)),
        'stop_atr_mult':    float(np.clip(p[5], 0.5, 2.5)),
        'lot_size':         float(np.clip(p[6], 0.01, 0.20)),
        'position_pct':     float(np.clip(p[7], 0.05, 0.30)),
        'growth_factor':    float(np.clip(p[8], 0.10, 0.50)),
        'htf_window':       int(np.clip(p[9], 24, 100)),
        'meta_bias_threshold': float(np.clip(p[10], 0.0, 0.8)),
    }


# Global worker state
_SPLITS     = None
_INIT_CAP   = 1_000
_COMMISSION = 0.0001
_SLIPPAGE   = 0.0002


def _init_worker(splits, init_cap, commission, slippage):
    global _SPLITS, _INIT_CAP, _COMMISSION, _SLIPPAGE
    _SPLITS     = splits
    _INIT_CAP   = init_cap
    _COMMISSION = commission
    _SLIPPAGE   = slippage


def evaluate_genome(genome: list) -> Tuple[float,]:
    global _SPLITS, _INIT_CAP, _COMMISSION, _SLIPPAGE
    try:
        import importlib
        gm_mod  = importlib.import_module("01_goldmaster_backtest")
        GMParams = gm_mod.GMParams
        GoldMasterBacktester = gm_mod.GoldMasterBacktester

        params  = decode(genome)
        p       = GMParams(**params)
        bt      = GoldMasterBacktester(_INIT_CAP, _COMMISSION, _SLIPPAGE)

        scores = []
        for sp in _SPLITS:
            r = bt.run(sp['test'], p)
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


class GoldMasterDEAP:

    def __init__(self, df: pd.DataFrame, config: dict = None):
        self.df     = df
        self.config = config or self._default()
        self._splits()
        self.toolbox = base.Toolbox()
        self.stats   = []

    def _default(self):
        return {
            'population_size':   60,
            'n_generations':     150,
            'hall_of_fame_size': 15,
            'crossover_prob':    0.70,
            'mutation_prob':     0.25,
            'tournament_size':   5,
            'initial_capital':   1_000,
            'commission':        0.0001,
            'slippage':          0.0002,
            'train_pct':         0.70,
            'n_splits':          5,
            'stagnation_limit':  15,
            'n_jobs':            11,
            'verbose':           True,
            'log_every':         5,
        }

    def _splits(self):
        n  = self.config['n_splits']
        sz = len(self.df) // n
        tp = self.config['train_pct']
        self.splits = []
        for i in range(n - 1):
            s  = i * sz; e = s + sz; te = s + int(sz * tp)
            self.splits.append({
                'train': self.df.iloc[s:te].copy(),
                'test':  self.df.iloc[te:e].copy(),
            })

    def _setup(self, sigma=0.3):
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
        cfg    = self.config
        n_jobs = min(cfg['n_jobs'], multiprocessing.cpu_count())
        self._setup(0.3)
        t0 = time.time()

        if cfg['verbose']:
            print("=" * 62)
            print("🥇 GOLDMASTER DEAP OPTİMİZASYONU")
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

            fits = pool.map(_get_fitness, pop)
            for ind, fit in zip(pop, fits):
                ind.fitness.values = fit
            hof.update(pop)

            best = hof[0].fitness.values[0]
            stag = 0; sigma = 0.3

            rec = sf.compile(pop)
            if cfg['verbose']:
                print(f"Gen   0: Max={rec['max']:.3f} | Avg={rec['avg']:.3f}")
            self.stats.append({'gen':0,'max_fitness':rec['max'],'avg_fitness':rec['avg']})

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

                inv = [ind for ind in off if not ind.fitness.valid]
                if inv:
                    fits = pool.map(_get_fitness, inv)
                    for ind, fit in zip(inv, fits):
                        ind.fitness.values = fit

                pop[:] = tools.selBest(pop, 8) + off[8:]
                hof.update(pop)
                rec = sf.compile(pop)
                self.stats.append({'gen':gen,'max_fitness':rec['max'],'avg_fitness':rec['avg']})

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
                    stag = 0; self._setup(sigma)
                    if cfg['verbose']:
                        print(f"  ⚡ Sigma={sigma:.2f}")

                if cfg['verbose'] and gen % cfg['log_every'] == 0:
                    el  = time.time() - t0
                    eta = (el/gen) * (cfg['n_generations'] - gen)
                    print(f"Gen {gen:3d}: Max={rec['max']:7.3f} | Avg={rec['avg']:7.3f} | "
                          f"Sigma={sigma:.2f} | {el:.0f}s | ~{eta/60:.0f}dk kaldı")

        finally:
            pool.close(); pool.join()

        best_params = decode(list(hof[0]))
        el = time.time() - t0

        if cfg['verbose']:
            print(f"\n✅ Tamamlandı! ({el/60:.1f} dakika)")
            print(f"   Fitness: {hof[0].fitness.values[0]:.4f}")
            print("\n🏆 EN İYİ PARAMETRELER:")
            for k, v in best_params.items():
                print(f"   {k:<20} : {v}")

        return best_params, {'stats': self.stats, 'hof': hof}
