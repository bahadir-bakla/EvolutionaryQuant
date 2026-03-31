"""
=============================================================
HOLY TRINITY V7 — DEAP OPTİMİZATÖRÜ
=============================================================
Çalışan sistemi daha iyi hale getiriyoruz.

DEAP şunları optimize eder:
  Gold Sniper  : lot, hard_stop, target_pts, stop_pts, stale_h
  Golden Basket: nq_lot, nq_tp, nq_stop
  Escape Vel.  : safe_threshold, growth_factor
  Risk Mgmt    : dd_threshold, cooldown_bars

Veri: institutional_engine cache (1h XAUUSD)
     + yfinance'tan NQ + SI anlık
=============================================================
"""

import random
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Tuple, List
import time
import multiprocessing
import os
import sys
import json
from datetime import datetime
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────
# GENOME — 13 parametre
# ─────────────────────────────────────────────────────────
#
# [0]  gs_lot           0.01 - 0.10   Gold Sniper başlangıç lot
# [1]  gs_hard_stop     200  - 1000   Basket hard stop ($)
# [2]  gs_target_pts    80   - 200    TP hedef (puan)
# [3]  gs_stop_pts      8    - 30     SL (puan)
# [4]  gs_stale_hours   24   - 120    Bayat işlem öldürücü (saat)
# [5]  nq_lot           0.01 - 0.10   NQ başlangıç lot
# [6]  nq_tp_pts        20   - 80     NQ TP hedef (puan)
# [7]  nq_stop_pts      80   - 250    NQ basket stop (puan)
# [8]  safe_threshold   1000 - 5000   Escape vel. eşiği ($)
# [9]  growth_factor    0.20 - 0.60   Lot büyüme çarpanı
# [10] dd_threshold     0.15 - 0.40   Circuit breaker DD oranı
# [11] cooldown_bars    24   - 96     Soğuma süresi (bar)
# [12] gs_point_value   80   - 120    Gold point değeri ($)

GENE_BOUNDS = [
    (0.01, 0.10),   # gs_lot
    (200,  1000),   # gs_hard_stop
    (80,   200),    # gs_target_pts
    (8,    30),     # gs_stop_pts
    (24,   120),    # gs_stale_hours
    (0.01, 0.10),   # nq_lot
    (20,   80),     # nq_tp_pts
    (80,   250),    # nq_stop_pts
    (1000, 5000),   # safe_threshold
    (0.20, 0.60),   # growth_factor
    (0.15, 0.40),   # dd_threshold
    (24,   96),     # cooldown_bars
    (80,   120),    # gs_point_value
]
GENOME_SIZE = len(GENE_BOUNDS)


def decode(genome: list) -> dict:
    p = {}
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        raw  = float(genome[i]) if i < len(genome) else 0.5
        p[i] = lo + abs(raw % 1.0) * (hi - lo)
    return {
        'gs_lot':           float(np.clip(p[0], 0.01, 0.10)),
        'gs_hard_stop':     float(np.clip(p[1], 200, 1000)),
        'gs_target_pts':    float(np.clip(p[2], 80, 200)),
        'gs_stop_pts':      float(np.clip(p[3], 8, 30)),
        'gs_stale_hours':   float(np.clip(p[4], 24, 120)),
        'nq_lot':           float(np.clip(p[5], 0.01, 0.10)),
        'nq_tp_pts':        float(np.clip(p[6], 20, 80)),
        'nq_stop_pts':      float(np.clip(p[7], 80, 250)),
        'safe_threshold':   float(np.clip(p[8], 1000, 5000)),
        'growth_factor':    float(np.clip(p[9], 0.20, 0.60)),
        'dd_threshold':     float(np.clip(p[10], 0.15, 0.40)),
        'cooldown_bars':    int(np.clip(p[11], 24, 96)),
        'gs_point_value':   float(np.clip(p[12], 80, 120)),
    }


# ─────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────

def add_gold_sniper_features(df):
    df = df.copy()
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low']  - df['close'].shift(1)).abs()
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()

    df['d1_close_prev']  = df['close'].shift(24)
    df['d1_close_prev2'] = df['close'].shift(48)
    df['daily_bias_bullish'] = df['d1_close_prev'] > df['d1_close_prev2']
    df['daily_bias_bearish'] = df['d1_close_prev'] < df['d1_close_prev2']

    df['h4_high'] = df['high'].rolling(4).max().shift(1)
    df['h4_low']  = df['low'].rolling(4).min().shift(1)
    h4r = df['h4_high'] - df['h4_low']
    df['h4_reject_down'] = (df['close'] - df['h4_low'])  > (h4r * 0.6)
    df['h4_reject_up']   = (df['h4_high'] - df['close']) > (h4r * 0.6)

    df['minor_high'] = df['high'].rolling(5).max().shift(1)
    df['minor_low']  = df['low'].rolling(5).min().shift(1)
    df['sweep_minor_low']  = (df['low']  < df['minor_low'])  & (df['close'] > df['minor_low'])
    df['sweep_minor_high'] = (df['high'] > df['minor_high']) & (df['close'] < df['minor_high'])

    is_down = df['close'] < df['open']
    is_up   = df['close'] > df['open']
    df['ob_tap_bullish'] = (is_down.shift(2) & is_up.shift(1) &
                            (df['low'] <= df['low'].shift(2)) &
                            (df['close'] >= df['close'].shift(1)))
    df['ob_tap_bearish'] = (is_up.shift(2) & is_down.shift(1) &
                            (df['high'] >= df['high'].shift(2)) &
                            (df['close'] <= df['close'].shift(1)))
    return df


def add_nq_features(df):
    df = df.copy()
    tp  = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume'].replace(0, 1)
    df['vwap'] = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()

    df['htf_swing_high'] = df['high'].rolling(24).max().shift(1)
    df['htf_swing_low']  = df['low'].rolling(24).min().shift(1)

    df['sweep_low']  = (df['low']  < df['htf_swing_low'])  & (df['close'] > df['htf_swing_low'])
    df['sweep_high'] = (df['high'] > df['htf_swing_high']) & (df['close'] < df['htf_swing_high'])
    return df


@dataclass
class BacktestResult:
    total_return:  float = 0.0
    cagr:          float = 0.0
    max_drawdown:  float = 0.0
    sharpe_ratio:  float = 0.0
    win_rate:      float = 0.0
    profit_factor: float = 0.0
    total_trades:  int   = 0
    final_equity:  float = 0.0
    gold_pnl:      float = 0.0
    nq_pnl:        float = 0.0

    def fitness(self) -> float:
        if self.total_trades < 5:
            return -999.0
        if self.total_return <= 0:
            return float(np.clip(self.total_return * 5, -50, -0.01))
        if self.profit_factor < 1.0:
            return float(np.clip((self.profit_factor - 1) * 10, -20, -0.01))
        if self.max_drawdown > 0.70:
            return -50.0

        pf     = float(np.clip(self.profit_factor, 1.0, 15))
        wr     = float(np.clip(self.win_rate, 0, 1))
        cagr   = float(np.clip(self.cagr, 0, 20))
        sharpe = float(np.clip(self.sharpe_ratio, 0, 10))
        dd_pen = max(0.1, 1 - max(0, self.max_drawdown - 0.15) * 3)

        score = (pf * 0.35 + wr * 0.15 + cagr * 0.35 + sharpe * 0.15) * dd_pen
        return float(np.clip(score, 0.0, 100.0))


def run_holy_trinity(gold_df: pd.DataFrame, nq_df: pd.DataFrame,
                     params: dict, initial_capital: float = 1_000.0) -> BacktestResult:
    """
    Holy Trinity V7 backtest — Gold Sniper + Golden Basket NQ
    gold_df ve nq_df aynı index'e sahip olmalı
    """
    result = BacktestResult()
    result.final_equity = initial_capital

    try:
        # Ortak index
        common = gold_df.index.intersection(nq_df.index)
        if len(common) < 100:
            return result
        gdf = gold_df.loc[common]
        ndf = nq_df.loc[common]

        balance       = initial_capital
        peak_equity   = balance
        eq_curve      = [balance]
        all_trades    = []
        gold_pnl_total= 0.0
        nq_pnl_total  = 0.0

        # Gold Sniper state
        gs_basket        = []
        gs_direction     = 0
        gs_zone_hist     = {'sup_taps': 0, 'res_taps': 0,
                             'last_sup': None, 'last_res': None}

        # NQ Golden Basket state
        nq_basket        = []
        nq_direction     = 0
        nq_halted        = False
        nq_cooldown      = 0

        p = params

        for i in range(50, len(common)):
            bar_time = common[i]
            gc_price = float(gdf['close'].iloc[i])
            nq_price = float(ndf['close'].iloc[i]) if i < len(ndf) else 0
            gc_atr   = float(gdf['atr'].iloc[i]) if gdf['atr'].iloc[i] > 0 else gc_price * 0.005
            gc_row   = gdf.iloc[i]

            # ── ESCAPE VELOCITY ──────────────────────────
            profit_blocks = max(0, int((balance - initial_capital) // 1000))
            if balance < p['safe_threshold']:
                multiplier = 0.1
            else:
                multiplier = max(0.1, balance / 1000.0)

            # ── CIRCUIT BREAKER ───────────────────────────
            if balance > peak_equity:
                peak_equity = balance
            dd = (peak_equity - balance) / (peak_equity + 1e-10)
            if dd >= p['dd_threshold'] and not nq_halted:
                # NQ'yu durdur
                if nq_basket:
                    nq_pnl = sum((nq_price - t['price']) * t['dir'] * t['size'] * 20.0
                                 for t in nq_basket)
                    balance      += nq_pnl
                    nq_pnl_total += nq_pnl
                    nq_basket     = []
                nq_halted   = True
                nq_cooldown = p['cooldown_bars']

            if nq_cooldown > 0:
                nq_cooldown -= 1
                if nq_cooldown == 0:
                    nq_halted = False

            # ── GOLD SNIPER ───────────────────────────────
            gs_lot = max(0.01, round(p['gs_lot'] * multiplier, 3))

            # Basket PnL kontrol
            if gs_basket:
                gs_pnl = sum((gc_price - t['price']) * t['dir'] * t['size'] * p['gs_point_value']
                             for t in gs_basket)
                # Hard stop
                if gs_pnl <= -p['gs_hard_stop'] * multiplier:
                    balance      += gs_pnl
                    gold_pnl_total += gs_pnl
                    all_trades.append({'pnl': gs_pnl, 'src': 'gs_hardstop'})
                    gs_basket = []; gs_direction = 0
                else:
                    # Bireysel bullet yönetimi
                    active = []
                    for t in gs_basket:
                        pts = (gc_price - t['price']) * t['dir']
                        pnl = pts * t['size'] * p['gs_point_value']
                        # Stale killer
                        stale = False
                        try:
                            stale = (bar_time - t['time']).total_seconds() > p['gs_stale_hours'] * 3600
                        except:
                            pass

                        if pts <= -p['gs_stop_pts'] or stale:
                            balance      += pnl
                            gold_pnl_total += pnl
                            all_trades.append({'pnl': pnl, 'src': 'gs_sl'})
                        elif pts >= p['gs_target_pts']:
                            balance      += pnl
                            gold_pnl_total += pnl
                            all_trades.append({'pnl': pnl, 'src': 'gs_tp'})
                        else:
                            active.append(t)
                    gs_basket = active
                    if not gs_basket:
                        gs_direction = 0

            # Yeni Gold Sniper girişi
            if len(gs_basket) < 8:
                # Zone history güncelle
                gs_sup = float(gc_row.get('h4_low', 0) or 0)
                gs_res = float(gc_row.get('h4_high', 0) or 0)
                if gs_zone_hist['last_sup'] != gs_sup:
                    gs_zone_hist['sup_taps'] = 0
                    gs_zone_hist['last_sup'] = gs_sup
                if gs_zone_hist['last_res'] != gs_res:
                    gs_zone_hist['res_taps'] = 0
                    gs_zone_hist['last_res'] = gs_res
                tap_t = gc_atr * 0.5
                if gs_sup and abs(float(gc_row.get('low', gc_price)) - gs_sup) <= tap_t:
                    gs_zone_hist['sup_taps'] += 1
                if gs_res and abs(float(gc_row.get('high', gc_price)) - gs_res) <= tap_t:
                    gs_zone_hist['res_taps'] += 1

                db  = bool(gc_row.get('daily_bias_bullish', False))
                dbr = bool(gc_row.get('daily_bias_bearish', False))
                h4d = bool(gc_row.get('h4_reject_down', False))
                h4u = bool(gc_row.get('h4_reject_up', False))
                swl = bool(gc_row.get('sweep_minor_low', False))
                swh = bool(gc_row.get('sweep_minor_high', False))
                obt = bool(gc_row.get('ob_tap_bullish', False))
                obb = bool(gc_row.get('ob_tap_bearish', False))

                if db and h4d and (swl or obt) and gs_direction >= 0:
                    gs_basket.append({'price': gc_price, 'dir': 1,
                                      'size': gs_lot, 'time': bar_time})
                    gs_direction = 1
                elif dbr and h4u and (swh or obb) and gs_direction <= 0:
                    gs_basket.append({'price': gc_price, 'dir': -1,
                                      'size': gs_lot, 'time': bar_time})
                    gs_direction = -1

            # ── GOLDEN BASKET NQ ──────────────────────────
            if not nq_halted and nq_price > 0:
                nq_lot = max(0.01, round(p['nq_lot'] * multiplier, 3))
                nq_tp_usd = nq_lot * 20.0 * p['nq_tp_pts']
                nq_stop_usd = nq_lot * 20.0 * p['nq_stop_pts']

                if nq_basket:
                    nq_pnl = sum((nq_price - t['price']) * t['dir'] * t['size'] * 20.0
                                 for t in nq_basket)
                    if nq_pnl >= nq_tp_usd:
                        balance      += nq_pnl
                        nq_pnl_total += nq_pnl
                        all_trades.append({'pnl': nq_pnl, 'src': 'nq_tp'})
                        nq_basket = []; nq_direction = 0
                    elif nq_pnl <= -nq_stop_usd:
                        balance      += nq_pnl
                        nq_pnl_total += nq_pnl
                        all_trades.append({'pnl': nq_pnl, 'src': 'nq_stop'})
                        nq_basket = []; nq_direction = 0

                if not nq_basket:
                    nq_row  = ndf.iloc[i]
                    prev_nq = ndf.iloc[i-1]
                    sw_low  = bool(nq_row.get('sweep_low', False))
                    sw_high = bool(nq_row.get('sweep_high', False))
                    vwap_v  = float(nq_row.get('vwap', nq_price))
                    prev_vwap = float(prev_nq.get('vwap', vwap_v))

                    if sw_low and float(prev_nq['close']) < prev_vwap and float(nq_row['close']) > vwap_v:
                        nq_basket.append({'price': nq_price, 'dir': 1,
                                          'size': nq_lot, 'time': bar_time})
                        nq_direction = 1
                    elif sw_high and float(prev_nq['close']) > prev_vwap and float(nq_row['close']) < vwap_v:
                        nq_basket.append({'price': nq_price, 'dir': -1,
                                          'size': nq_lot, 'time': bar_time})
                        nq_direction = -1

            eq_curve.append(balance)

        # Açık pozisyonları kapat
        if gs_basket:
            fp  = float(gdf['close'].iloc[-1])
            pnl = sum((fp - t['price']) * t['dir'] * t['size'] * p['gs_point_value']
                      for t in gs_basket)
            balance      += pnl
            gold_pnl_total += pnl
            all_trades.append({'pnl': pnl, 'src': 'eod'})
            eq_curve[-1] = balance

        if nq_basket and not nq_halted:
            fp  = float(ndf['close'].iloc[-1])
            pnl = sum((fp - t['price']) * t['dir'] * t['size'] * 20.0
                      for t in nq_basket)
            balance      += pnl
            nq_pnl_total += pnl
            all_trades.append({'pnl': pnl, 'src': 'eod'})
            eq_curve[-1] = balance

        # ── METRİKLER ────────────────────────────────────
        eq  = np.array(eq_curve)
        ret = np.diff(eq) / (eq[:-1] + 1e-10)

        result.total_return  = (eq[-1] - initial_capital) / initial_capital
        result.final_equity  = float(eq[-1])
        result.gold_pnl      = gold_pnl_total
        result.nq_pnl        = nq_pnl_total

        peak = np.maximum.accumulate(eq)
        result.max_drawdown  = float(abs(((eq - peak) / (peak + 1e-10)).min()))

        ppy = 365 * 24  # 1h veri
        if len(ret) > 1 and ret.std() > 1e-10:
            result.sharpe_ratio = float(np.clip(
                ret.mean() / ret.std() * np.sqrt(ppy), -10, 10))

        ny = len(eq) / ppy
        if ny > 0 and eq[-1] > 0:
            result.cagr = float(np.clip(
                (eq[-1] / initial_capital) ** (1/max(ny, 0.1)) - 1, -1, 20))

        if all_trades:
            result.total_trades = len(all_trades)
            wins   = [t for t in all_trades if t['pnl'] > 0]
            losses = [t for t in all_trades if t['pnl'] <= 0]
            result.win_rate = len(wins) / len(all_trades)
            gp = sum(t['pnl'] for t in wins)
            gl = abs(sum(t['pnl'] for t in losses)) if losses else 1e-10
            result.profit_factor = float(np.clip(gp / (gl + 1e-10), 0, 20))

    except Exception as e:
        pass

    return result


# ─────────────────────────────────────────────────────────
# DEAP
# ─────────────────────────────────────────────────────────

from deap import base, creator, tools

if not hasattr(creator, 'FitnessMax'):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, 'Individual'):
    creator.create("Individual", list, fitness=creator.FitnessMax)

_SPLITS       = None
_GOLD_SPLITS  = None
_NQ_SPLITS    = None
_INIT_CAP     = 1_000.0


def _init_worker(gold_splits, nq_splits, init_cap):
    global _GOLD_SPLITS, _NQ_SPLITS, _INIT_CAP
    _GOLD_SPLITS = gold_splits
    _NQ_SPLITS   = nq_splits
    _INIT_CAP    = init_cap


def evaluate_genome(genome: list) -> Tuple[float,]:
    global _GOLD_SPLITS, _NQ_SPLITS, _INIT_CAP
    try:
        params = decode(genome)
        scores = []
        for gdf, ndf in zip(_GOLD_SPLITS, _NQ_SPLITS):
            r = run_holy_trinity(gdf, ndf, params, _INIT_CAP)
            scores.append(r.fitness())
        tm   = float(np.mean(scores))
        ts   = float(np.std(scores))
        cons = 1 - (ts / (abs(tm) + 1e-10))
        fit  = tm * (0.6 + 0.4 * max(0, cons))
        return (float(np.clip(fit, -999, 200)),)
    except Exception:
        return (-999.0,)


def _get_fitness(ind):
    return evaluate_genome(list(ind))


class HolyTrinityDEAP:

    def __init__(self, gold_df, nq_df, config=None):
        self.gold_df = gold_df
        self.nq_df   = nq_df
        self.config  = config or self._default()
        self._prepare_splits()
        self.toolbox = base.Toolbox()
        self.stats   = []

    def _default(self):
        return {
            'population_size':   60,
            'n_generations':     200,
            'hall_of_fame_size': 15,
            'crossover_prob':    0.70,
            'mutation_prob':     0.25,
            'tournament_size':   5,
            'initial_capital':   1_000.0,
            'train_pct':         0.70,
            'n_splits':          5,
            'stagnation_limit':  20,
            'n_jobs':            11,
            'verbose':           True,
            'log_every':         10,
        }

    def _prepare_splits(self):
        n  = self.config['n_splits']
        sz = len(self.gold_df) // n
        tp = self.config['train_pct']
        self.gold_splits = []
        self.nq_splits   = []
        for i in range(n - 1):
            s = i*sz; e = s+sz; te = s+int(sz*tp)
            self.gold_splits.append(self.gold_df.iloc[te:e].copy())
            self.nq_splits.append(self.nq_df.iloc[te:e].copy())

    def _setup(self, sigma=0.3):
        tb = self.toolbox
        tb.register("attr_float", random.uniform, 0.0, 1.0)
        tb.register("individual", tools.initRepeat, creator.Individual,
                    tb.attr_float, n=GENOME_SIZE)
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("evaluate",   evaluate_genome)
        tb.register("select",     tools.selTournament,
                    tournsize=self.config['tournament_size'])
        tb.register("mate",       tools.cxBlend, alpha=0.3)
        tb.register("mutate",     tools.mutGaussian, mu=0, sigma=sigma, indpb=0.3)

    def run(self):
        cfg    = self.config
        n_jobs = min(cfg['n_jobs'], multiprocessing.cpu_count())
        self._setup(0.3)
        t0 = time.time()

        if cfg['verbose']:
            print("=" * 62)
            print("🏆 HOLY TRINITY V7 — DEAP OPTİMİZASYONU")
            print(f"   Gold veri    : {len(self.gold_df):,} bar")
            print(f"   NQ veri      : {len(self.nq_df):,} bar")
            print(f"   Popülasyon   : {cfg['population_size']}")
            print(f"   Nesil        : {cfg['n_generations']}")
            print(f"   CPU          : {n_jobs} core 🚀")
            print(f"   Splits       : {len(self.gold_splits)} walk-forward")
            print("=" * 62)
            print("  Optimize: gs_lot, gs_hard_stop, gs_target_pts,")
            print("            nq_lot, nq_tp_pts, escape_vel, circuit_breaker")
            print("-" * 62)

        pool = multiprocessing.Pool(
            processes   = n_jobs,
            initializer = _init_worker,
            initargs    = (self.gold_splits, self.nq_splits,
                           cfg['initial_capital'])
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

            best = hof[0].fitness.values[0]
            stag = 0; sigma = 0.3
            rec  = sf.compile(pop)
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

                pop[:] = tools.selBest(pop, 10) + off[10:]
                hof.update(pop)
                rec = sf.compile(pop)
                self.stats.append({
                    'gen': gen,
                    'max_fitness': rec['max'],
                    'avg_fitness': rec['avg'],
                    'std_fitness': rec['std'],
                })

                curr = hof[0].fitness.values[0]
                if curr > best + 0.001:
                    best = curr; stag = 0
                    sigma = max(0.1, sigma * 0.95)
                else:
                    stag += 1

                if stag >= cfg['stagnation_limit']:
                    sigma = min(0.6, sigma * 1.5)
                    ni = self.toolbox.population(n=20)
                    nf = pool.map(_get_fitness, ni)
                    for ind, fit in zip(ni, nf):
                        ind.fitness.values = fit
                    pop = tools.selBest(pop, len(pop)-20) + ni
                    stag = 0; self._setup(sigma)
                    if cfg['verbose']:
                        print(f"  ⚡ Sigma={sigma:.2f}")

                if cfg['verbose'] and gen % cfg['log_every'] == 0:
                    el  = time.time() - t0
                    eta = (el/gen) * (cfg['n_generations'] - gen)
                    print(f"Gen {gen:3d}: Max={rec['max']:7.3f} | "
                          f"Avg={rec['avg']:7.3f} | "
                          f"Sigma={sigma:.2f} | "
                          f"{el:.0f}s | ~{eta/60:.0f}dk kaldı")

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