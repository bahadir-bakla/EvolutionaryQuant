"""
Holy Trinity V7 DEAP — Runner
==============================
Gold Sniper (XAUUSD) + NQ Golden Basket dual-asset optimizer.

Usage:
    python run_optimizer.py --gold XAUUSD_1h.csv --nq NQ_1h.csv
    python run_optimizer.py --gold cache/ --tf 1h

Uses the original nq_core engine classes (GoldSniperStrategyCore, GoldenStrategyCore).
"""
import sys, os, json, multiprocessing, random, time
import pandas as pd
import numpy as np
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass

# nq_core'u bul: script'in üst dizininde olmalı
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR  = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PARENT_DIR)

from nq_core.gold_sniper_strategy import GoldSniperStrategyCore, add_gold_sniper_features
from nq_core.golden_strategy      import GoldenStrategyCore, add_golden_features

from deap import base, creator, tools

if not hasattr(creator, 'FitnessMax'):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, 'Individual'):
    creator.create("Individual", list, fitness=creator.FitnessMax)

GENE_BOUNDS = [
    (0.01, 0.10),  # gs_lot
    (200,  1000),  # gs_hard_stop
    (80,   200),   # gs_target_pts
    (8,    30),    # gs_stop_pts
    (0.01, 0.10),  # nq_lot
    (20,   80),    # nq_tp_pts
    (80,   250),   # nq_stop_pts
    (1000, 5000),  # safe_threshold
    (0.20, 0.60),  # growth_factor
    (0.15, 0.40),  # dd_threshold
]
GENOME_SIZE = len(GENE_BOUNDS)


def decode(genome):
    p = {}
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        raw  = float(genome[i]) if i < len(genome) else 0.5
        p[i] = lo + abs(raw % 1.0) * (hi - lo)
    return {
        'gs_lot':         float(np.clip(p[0], 0.01, 0.10)),
        'gs_hard_stop':   float(np.clip(p[1], 200, 1000)),
        'gs_target_pts':  float(np.clip(p[2], 80, 200)),
        'gs_stop_pts':    float(np.clip(p[3], 8, 30)),
        'nq_lot':         float(np.clip(p[4], 0.01, 0.10)),
        'nq_tp_pts':      float(np.clip(p[5], 20, 80)),
        'nq_stop_pts':    float(np.clip(p[6], 80, 250)),
        'safe_threshold': float(np.clip(p[7], 1000, 5000)),
        'growth_factor':  float(np.clip(p[8], 0.20, 0.60)),
        'dd_threshold':   float(np.clip(p[9], 0.15, 0.40)),
    }


def add_nq_features(df):
    df = df.copy()
    tp  = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume'].replace(0, 1)
    df['vwap']           = (tp*vol).rolling(20).sum() / vol.rolling(20).sum()
    df['htf_swing_high'] = df['high'].rolling(24).max().shift(1)
    df['htf_swing_low']  = df['low'].rolling(24).min().shift(1)
    df['sweep_low']  = (df['low']  < df['htf_swing_low'])  & (df['close'] > df['htf_swing_low'])
    df['sweep_high'] = (df['high'] > df['htf_swing_high']) & (df['close'] < df['htf_swing_high'])
    return df


@dataclass
class Result:
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

    def fitness(self):
        if self.total_trades < 5:    return -999.0
        if self.total_return <= 0:   return float(np.clip(self.total_return*5, -50, -0.01))
        if self.profit_factor < 1.0: return float(np.clip((self.profit_factor-1)*10, -20, -0.01))
        if self.max_drawdown > 0.70: return -50.0
        pf   = float(np.clip(self.profit_factor, 1.0, 15))
        wr   = float(np.clip(self.win_rate, 0, 1))
        cagr = float(np.clip(self.cagr, 0, 20))
        sh   = float(np.clip(self.sharpe_ratio, 0, 10))
        ddp  = max(0.1, 1 - max(0, self.max_drawdown - 0.15) * 3)
        return float(np.clip((pf*0.35+wr*0.15+cagr*0.35+sh*0.15)*ddp, 0, 100))


def run_backtest(gdf, ndf, params, initial_capital=1_000.0):
    result = Result(final_equity=initial_capital)
    try:
        common = gdf.index.intersection(ndf.index)
        if len(common) < 100: return result
        gdf = gdf.loc[common]; ndf = ndf.loc[common]

        balance = initial_capital; peak_eq = balance
        eq_curve = [balance]; all_trades = []
        gold_pnl_total = 0.0; nq_pnl_total = 0.0
        NQ_PV = 20.0; GS_PV = 100.0

        gold_bot = GoldSniperStrategyCore(
            lot_size=params['gs_lot'],
            starting_balance=initial_capital,
            max_basket_loss_usd=params['gs_hard_stop'],
        )
        nq_bot = GoldenStrategyCore(
            target_profit_usd=params['nq_tp_pts']*params['nq_lot']*NQ_PV,
            max_layers=5, dca_step_points=15,
            lot_size=params['nq_lot'], starting_balance=initial_capital,
        )
        nq_halted = False; nq_cooldown = 0

        for i in range(50, len(common)):
            bt = common[i]
            gc = float(gdf['close'].iloc[i])
            nq = float(ndf['close'].iloc[i])
            gr = gdf.iloc[i]; nr = ndf.iloc[i]

            if balance < params['safe_threshold']:
                mult = 0.1
            else:
                pb   = max(0, int((balance - initial_capital)//1000))
                mult = 1.0 + pb * params['growth_factor']

            if balance > peak_eq:
                peak_eq = balance
                if nq_cooldown <= 0: nq_halted = False
            dd = (peak_eq - balance) / (peak_eq + 1e-10)
            if dd >= params['dd_threshold'] and not nq_halted:
                if nq_bot.basket:
                    pnl = nq_bot.calculate_basket_pnl(nq, NQ_PV)
                    balance += pnl; nq_pnl_total += pnl
                    all_trades.append({'pnl': pnl})
                    nq_bot.clear_basket(pnl)
                nq_halted = True; nq_cooldown = 48
            if nq_cooldown > 0: nq_cooldown -= 1

            # Gold Sniper
            gold_bot.lot_size            = max(0.01, round(params['gs_lot']*mult, 3))
            gold_bot.max_basket_loss_usd = max(200, params['gs_hard_stop']*mult)
            realized = gold_bot.manage_bullets(
                gc, bt, float(gr.get('atr', gc*0.005)),
                GS_PV, params['gs_target_pts'], params['gs_stop_pts'])
            if realized:
                balance += realized; gold_pnl_total += realized
                all_trades.append({'pnl': realized})

            if len(gold_bot.basket) < 10:
                gold_bot.reset_zone_history(
                    float(gr.get('h4_low', 0) or 0),
                    float(gr.get('h4_high', 0) or 0))
                gold_bot.check_zone_interactions(
                    float(gr.get('low', gc)), float(gr.get('high', gc)),
                    float(gr.get('h4_low', 0) or 0),
                    float(gr.get('h4_high', 0) or 0),
                    float(gr.get('atr', gc*0.005)))
                lot = max(0.01, round(params['gs_lot']*mult, 3))
                if gr.get('daily_bias_bullish') and gr.get('h4_reject_down'):
                    if gr.get('sweep_minor_low') or gr.get('ob_tap_bullish'):
                        if gold_bot.current_direction >= 0:
                            gold_bot.add_trade(gc, 1, "LONG", bt, lot)
                elif gr.get('daily_bias_bearish') and gr.get('h4_reject_up'):
                    if gr.get('sweep_minor_high') or gr.get('ob_tap_bearish'):
                        if gold_bot.current_direction <= 0:
                            gold_bot.add_trade(gc, -1, "SHORT", bt, lot)

            # NQ
            if not nq_halted:
                nq_bot.lot_size = max(0.01, round(params['nq_lot']*mult, 3))
                nq_bot.target_profit_usd = max(10, params['nq_tp_pts']*nq_bot.lot_size*NQ_PV)
                try:
                    action = nq_bot.check_basket_logic(
                        nq, bt, NQ_PV,
                        htf_high=float(nr.get('htf_swing_high', 0) or 0),
                        htf_low =float(nr.get('htf_swing_low',  0) or 0))
                except:
                    action = None
                if action in ["CLOSE_ALL","CLOSE_ALL_HTF","MARGIN_CALL"]:
                    pnl = nq_bot.calculate_basket_pnl(nq, NQ_PV)
                    balance += pnl; nq_pnl_total += pnl
                    all_trades.append({'pnl': pnl})
                    nq_bot.clear_basket(pnl)
                    if action == "MARGIN_CALL":
                        nq_halted = True; nq_cooldown = 48
                elif action == "DCA":
                    nq_bot.add_trade(nq, nq_bot.current_direction, "DCA", bt)
                elif action == "PYRAMID":
                    nq_bot.add_trade(nq, nq_bot.current_direction, "PYRAMID", bt)

                if not nq_bot.basket:
                    pnr = ndf.iloc[i-1]
                    vc  = float(nr.get('vwap', nq))
                    vp  = float(pnr.get('vwap', vc))
                    if nr.get('sweep_low') and float(pnr['close']) < vp and float(nr['close']) > vc:
                        nq_bot.add_trade(nq, 1, "INITIAL_LONG", bt)
                    elif nr.get('sweep_high') and float(pnr['close']) > vp and float(nr['close']) < vc:
                        nq_bot.add_trade(nq, -1, "INITIAL_SHORT", bt)

            eq_curve.append(balance)

        if gold_bot.basket:
            fp = float(gdf['close'].iloc[-1])
            pnl = gold_bot.calculate_basket_pnl(fp, GS_PV)
            balance += pnl; gold_pnl_total += pnl
            all_trades.append({'pnl': pnl}); eq_curve[-1] = balance

        if nq_bot.basket and not nq_halted:
            fp = float(ndf['close'].iloc[-1])
            pnl = nq_bot.calculate_basket_pnl(fp, NQ_PV)
            balance += pnl; nq_pnl_total += pnl
            all_trades.append({'pnl': pnl}); eq_curve[-1] = balance

        eq  = np.array(eq_curve)
        ret = np.diff(eq) / (eq[:-1] + 1e-10)
        result.total_return = (eq[-1]-initial_capital)/initial_capital
        result.final_equity = float(eq[-1])
        result.gold_pnl = gold_pnl_total; result.nq_pnl = nq_pnl_total
        peak = np.maximum.accumulate(eq)
        result.max_drawdown = float(abs(((eq-peak)/(peak+1e-10)).min()))
        ppy = 365*24
        if len(ret)>1 and ret.std()>1e-10:
            result.sharpe_ratio = float(np.clip(ret.mean()/ret.std()*np.sqrt(ppy),-10,10))
        ny = len(eq)/ppy
        if ny>0 and eq[-1]>0:
            result.cagr = float(np.clip((eq[-1]/initial_capital)**(1/max(ny,0.1))-1,-1,20))
        if all_trades:
            result.total_trades = len(all_trades)
            wins = [t for t in all_trades if t['pnl']>0]
            losses = [t for t in all_trades if t['pnl']<=0]
            result.win_rate = len(wins)/len(all_trades)
            gp = sum(t['pnl'] for t in wins)
            gl = abs(sum(t['pnl'] for t in losses)) if losses else 1e-10
            result.profit_factor = float(np.clip(gp/(gl+1e-10),0,20))
    except Exception as e:
        pass
    return result


_GOLD_SPLITS = None; _NQ_SPLITS = None; _INIT_CAP = 1_000.0

def _init_worker(gs, ns, ic):
    global _GOLD_SPLITS, _NQ_SPLITS, _INIT_CAP
    sys.path.insert(0, _PARENT_DIR)
    _GOLD_SPLITS=gs; _NQ_SPLITS=ns; _INIT_CAP=ic

def evaluate_genome(genome):
    try:
        p = decode(genome)
        scores = [run_backtest(g,n,p,_INIT_CAP).fitness()
                  for g,n in zip(_GOLD_SPLITS,_NQ_SPLITS)]
        tm=np.mean(scores); ts=np.std(scores)
        cons=1-(ts/(abs(tm)+1e-10))
        return (float(np.clip(tm*(0.6+0.4*max(0,cons)),-999,200)),)
    except:
        return (-999.0,)

def _get_fitness(ind): return evaluate_genome(list(ind))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Holy Trinity V7 DEAP Optimizer")
    parser.add_argument('--gold', type=str, default=None, help='XAUUSD CSV veya cache klasörü')
    parser.add_argument('--nq',   type=str, default=None, help='NQ CSV dosyası (opsiyonel)')
    parser.add_argument('--tf',   type=str, default='1h')
    parser.add_argument('--capital', type=float, default=1_000.0)
    args = parser.parse_args()

    print("🏆 HOLY TRINITY V7 — DEAP (Orijinal Engine)\n")

    # Data yolu: --gold argümanı yoksa cache klasörüne bak
    cache_dir = os.path.join(_PARENT_DIR, "institutional_engine", "cache")
    if args.gold and os.path.isdir(args.gold):
        cache_dir = args.gold
    gold_csv = os.path.join(cache_dir, "XAUUSD_1h.csv") if not (args.gold and os.path.isfile(args.gold)) else args.gold
    gold_df  = pd.read_csv(gold_csv, index_col=0, parse_dates=True)
    gold_df  = add_gold_sniper_features(gold_df)
    print(f"✅ Gold: {len(gold_df):,} bar")

    nq_path = args.nq or os.path.join(cache_dir, "NQ_1h.csv")
    if os.path.exists(nq_path):
        nq_df = pd.read_csv(nq_path, index_col=0, parse_dates=True)
        if 'vwap' not in nq_df.columns: nq_df = add_nq_features(nq_df)
    else:
        try:
            import yfinance as yf
            nq_df = yf.Ticker("NQ=F").history(period="730d", interval="1h")
            nq_df.columns = nq_df.columns.str.lower()
            if nq_df.index.tz is not None: nq_df.index = nq_df.index.tz_convert(None)
            nq_df = add_nq_features(nq_df)
            nq_df.to_csv(nq_path)
        except:
            nq_df = gold_df.copy()
    print(f"✅ NQ: {len(nq_df):,} bar")

    common  = gold_df.index.intersection(nq_df.index)
    gold_df = gold_df.loc[common]; nq_df = nq_df.loc[common]
    print(f"✅ Ortak: {len(gold_df):,} bar\n")

    # Splits
    n=5; sz=len(gold_df)//n; tp=0.70
    gs=[gold_df.iloc[i*sz+int(sz*tp):(i+1)*sz].copy() for i in range(n-1)]
    ns=[nq_df.iloc[i*sz+int(sz*tp):(i+1)*sz].copy()   for i in range(n-1)]

    n_jobs = min(11, multiprocessing.cpu_count())
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=GENOME_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_genome)
    toolbox.register("select", tools.selTournament, tournsize=5)
    toolbox.register("mate",   tools.cxBlend, alpha=0.3)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.3)

    pool = multiprocessing.Pool(n_jobs, _init_worker, (gs, ns, 1_000.0))

    pop  = toolbox.population(n=60)
    hof  = tools.HallOfFame(15)
    sf   = tools.Statistics(key=lambda ind: ind.fitness.values[0] if ind.fitness.valid else -999)
    sf.register("max", np.max); sf.register("avg", np.mean)

    print(f"🧬 Pop:60 | Nesil:200 | CPU:{n_jobs} 🚀")

    fits = pool.map(_get_fitness, pop)
    for ind,fit in zip(pop,fits): ind.fitness.values=fit
    hof.update(pop)
    rec=sf.compile(pop); best=hof[0].fitness.values[0]; stag=0; sigma=0.3; t0=time.time()
    print(f"Gen   0: Max={rec['max']:.3f} | Avg={rec['avg']:.3f}")
    stats=[]

    for gen in range(1, 201):
        off = list(map(deepcopy, toolbox.select(pop, len(pop))))
        for c1,c2 in zip(off[::2],off[1::2]):
            if random.random()<0.7: toolbox.mate(c1,c2); del c1.fitness.values,c2.fitness.values
        for m in off:
            if random.random()<0.25:
                toolbox.mutate(m); del m.fitness.values
                for i in range(len(m)): m[i]=float(np.clip(m[i],0,1))
        inv=[ind for ind in off if not ind.fitness.valid]
        if inv:
            fits=pool.map(_get_fitness,inv)
            for ind,fit in zip(inv,fits): ind.fitness.values=fit
        pop[:]=tools.selBest(pop,10)+off[10:]
        hof.update(pop); rec=sf.compile(pop)
        stats.append({'gen':gen,'max_fitness':rec['max'],'avg_fitness':rec['avg']})

        curr=hof[0].fitness.values[0]
        if curr>best+0.001: best=curr; stag=0; sigma=max(0.1,sigma*0.95)
        else: stag+=1
        if stag>=20:
            sigma=min(0.6,sigma*1.5)
            ni=toolbox.population(n=20); nf=pool.map(_get_fitness,ni)
            for ind,fit in zip(ni,nf): ind.fitness.values=fit
            pop=tools.selBest(pop,len(pop)-20)+ni; stag=0
            toolbox.unregister("mutate")
            toolbox.register("mutate",tools.mutGaussian,mu=0,sigma=sigma,indpb=0.3)
            print(f"  ⚡ Sigma={sigma:.2f}")

        if gen%10==0:
            el=time.time()-t0; eta=(el/gen)*(200-gen)
            print(f"Gen {gen:3d}: Max={rec['max']:7.3f} | Avg={rec['avg']:7.3f} | "
                  f"Sigma={sigma:.2f} | {el:.0f}s | ~{eta/60:.0f}dk kaldı")

    pool.close(); pool.join()

    best_params = decode(list(hof[0]))
    el = time.time()-t0
    print(f"\n✅ Tamamlandı! ({el/60:.1f}dk)  Fitness:{hof[0].fitness.values[0]:.4f}")

    # Rapor
    r   = run_backtest(gold_df, nq_df, best_params, 1_000)
    v7  = {'gs_lot':0.02,'gs_hard_stop':500,'gs_target_pts':140,'gs_stop_pts':12,
           'nq_lot':0.03,'nq_tp_pts':40,'nq_stop_pts':150,'safe_threshold':3000,
           'growth_factor':0.40,'dd_threshold':0.30}
    rv7 = run_backtest(gold_df, nq_df, v7, 1_000)

    print("\n"+"═"*60)
    print("📈 SONUÇ KARŞILAŞTIRMA")
    print("═"*60)
    print(f"  {'':20s} {'V7 Default':>15} {'Optimize':>15}")
    for lbl,d,o in [
        ("CAGR %",          rv7.cagr*100,         r.cagr*100),
        ("Getiri %",        rv7.total_return*100,  r.total_return*100),
        ("MaxDD %",         rv7.max_drawdown*100,  r.max_drawdown*100),
        ("Win Rate %",      rv7.win_rate*100,       r.win_rate*100),
        ("Profit Factor",   rv7.profit_factor,      r.profit_factor),
        ("Final $",         rv7.final_equity,       r.final_equity),
    ]:
        flag="✅" if o>d else "❌"
        print(f"  {lbl:<20} {d:>15.2f} {o:>15.2f} {flag}")
    print("═"*60)

    os.makedirs("outputs", exist_ok=True)
    ts=datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"outputs/ht_v7_{ts}.json",'w') as f:
        json.dump({'fitness':hof[0].fitness.values[0],'params':best_params},f,indent=2)
    print(f"✅ Kaydedildi")

    print("\n🏆 TOP 5")
    for i,ind in enumerate(list(hof)[:5]):
        p=decode(list(ind))
        print(f"#{i+1} Fit:{ind.fitness.values[0]:.4f} "
              f"gs_lot={p['gs_lot']:.3f} gs_tp={p['gs_target_pts']:.0f} "
              f"nq_tp={p['nq_tp_pts']:.0f} safe=${p['safe_threshold']:.0f}")
    return best_params

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()