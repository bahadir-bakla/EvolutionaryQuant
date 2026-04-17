"""
Alpha Engine — Master CLI Runner
==================================
Single entry point for all optimization and validation tasks.

Usage:
  # 1. Quick baseline test (all strategies, default params):
  python alpha_engine/run_master.py --mode baseline

  # 2. DEAP optimize Gold Sniper only:
  python alpha_engine/run_master.py --mode deap --strategy gold_sniper

  # 3. DEAP optimize ALL strategies:
  python alpha_engine/run_master.py --mode deap --strategy all

  # 4. Walk-forward for Gold Master:
  python alpha_engine/run_master.py --mode wf --strategy gold_master

  # 5. Walk-forward ALL + report:
  python alpha_engine/run_master.py --mode wf --strategy all

  # 6. Full pipeline (deap then wf):
  python alpha_engine/run_master.py --mode full --strategy all

  # 7. Portfolio baseline:
  python alpha_engine/run_master.py --mode portfolio

Common flags:
  --years  2019 2020 2021 2022    Gold/Silver data years
  --capital 1000                  Starting capital per strategy
  --pop 80                        DEAP population size
  --gen 150                       DEAP generations
  --jobs 4                        Parallel CPU cores
  --quick                         Quick mode (pop=30, gen=30)
"""

import os
import sys
import argparse
import multiprocessing

_ENGINE = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_ENGINE)
sys.path.insert(0, _ENGINE)
sys.path.insert(0, _ROOT)


def _parse():
    p = argparse.ArgumentParser(description='Alpha Engine Master Runner')
    p.add_argument('--mode',     default='baseline',
                   choices=['baseline', 'deap', 'wf', 'full', 'portfolio'],
                   help='Run mode')
    p.add_argument('--strategy', default='all',
                   help='Strategy to run: gold_sniper|gold_master|silver|nq_alpha|nq_liquidity|nq_smc|all')
    p.add_argument('--years',    nargs='+', type=int, default=None,
                   help='Gold data years, e.g. 2019 2020 2021 2022')
    p.add_argument('--capital',  type=float, default=1_000.0)
    p.add_argument('--pop',      type=int,   default=80)
    p.add_argument('--gen',      type=int,   default=120)
    p.add_argument('--wf_train', type=int,   default=18, help='WF train window (months)')
    p.add_argument('--wf_test',  type=int,   default=6,  help='WF test window (months)')
    p.add_argument('--jobs',     type=int,   default=None)
    p.add_argument('--quick',    action='store_true', help='pop=30, gen=30, quick test')
    p.add_argument('--data_root',type=str,   default=_ROOT)
    return p.parse_args()


def _load_data(args):
    from data_loader import get_gold_1h, get_gold_m1, get_silver_1h, get_nq_5m

    years = args.years or list(range(2019, 2026))
    root  = args.data_root
    strat = getattr(args, 'strategy', 'all')

    print("\n=== Loading Market Data ===")
    df_gold = df_silver = df_nq = df_gold_m1 = None

    # Gold M1 only loaded when scalper is requested (heavy ~1.4M bars)
    need_m1 = strat in ('gold_scalper', 'all')

    try:
        df_gold = get_gold_1h(years=years, root_dir=root)
    except Exception as e:
        print(f"  Gold 1H: {e}")

    if need_m1:
        try:
            df_gold_m1 = get_gold_m1(years=years, root_dir=root)
        except Exception as e:
            print(f"  Gold M1: {e}")
            df_gold_m1 = df_gold  # fallback to 1H if M1 unavailable

    try:
        df_silver = get_silver_1h(root_dir=root)
    except Exception as e:
        print(f"  Silver: {e}")

    try:
        df_nq = get_nq_5m()
    except Exception as e:
        print(f"  NQ: {e}")

    return (
        df_gold      if df_gold    is not None else _empty(),
        df_silver    if df_silver  is not None else _empty(),
        df_nq        if df_nq      is not None else _empty(),
        df_gold_m1   if df_gold_m1 is not None else _empty(),
    )


def _empty():
    import pandas as pd
    return pd.DataFrame()


def _get_module(name: str):
    """Return (module, df_key) for a strategy name."""
    import importlib
    mapping = {
        'gold_sniper':  ('gold_sniper_bt',  'gold'),
        'gold_master':  ('gold_master_bt',  'gold'),
        'gold_scalper': ('gold_scalper_bt', 'gold_m1'),
        'nq_liquidity': ('nq_liquidity_bt', 'nq'),
    }
    if name in mapping:
        mod_name, dk = mapping[name]
        sys.path.insert(0, _ENGINE)
        return importlib.import_module(mod_name), dk

    if name == 'silver':
        sys.path.insert(0, os.path.join(_ROOT, 'silver_reversion_deap'))
        from deap_runner import _SilverModule
        return _SilverModule(), 'silver'

    if name == 'nq_alpha':
        sys.path.insert(0, os.path.join(_ROOT, 'nq_alpha_deap'))
        from deap_runner import _NQAlphaModule
        return _NQAlphaModule(), 'nq'

    if name == 'nq_smc':
        _smc_dir = os.path.join(_ROOT, 'nq_smc_deap')
        sys.path.insert(0, _smc_dir)
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location('nq_smc_bt', os.path.join(_smc_dir, 'nq_smc_bt.py'))
        mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(mod)
        return mod, 'nq'

    raise ValueError(f"Unknown strategy: {name}")


# ─────────────────────────────────────────────────────────────────────
# MODES
# ─────────────────────────────────────────────────────────────────────

def mode_baseline(args, df_gold, df_silver, df_nq, df_gold_m1=None):
    """Run all strategies with default params and print results."""
    import pandas as pd
    from gold_sniper_bt  import backtest_from_raw as sniper_bt,  DEFAULT_PARAMS as S_DEF
    from gold_master_bt  import backtest_from_raw as master_bt,  DEFAULT_PARAMS as M_DEF
    from nq_liquidity_bt import backtest_from_raw as liq_bt,     DEFAULT_PARAMS as L_DEF

    sys.path.insert(0, os.path.join(_ROOT, 'silver_reversion_deap'))
    sys.path.insert(0, os.path.join(_ROOT, 'nq_alpha_deap'))
    from silver_strategy   import DEFAULT_PARAMS as SIL_DEF
    from backtest_engine   import backtest_from_raw as silver_bt

    # Need to handle nq_alpha import name collision
    import importlib
    nq_alpha_bt_mod = importlib.import_module('backtest_engine')
    nq_a_bt  = nq_alpha_bt_mod.backtest_from_raw
    nqa_def_mod = importlib.import_module('nq_alpha_strategy')
    NQA_DEF  = nqa_def_mod.DEFAULT_PARAMS

    # NQ SMC
    _smc_mod, _ = _get_module('nq_smc')
    smc_bt  = _smc_mod.backtest_from_raw
    SMC_DEF = _smc_mod.DEFAULT_PARAMS

    tasks = [
        ('Gold Sniper',   df_gold,   sniper_bt, S_DEF),
        ('Gold Master',   df_gold,   master_bt, M_DEF),
        ('Silver Rev.',   df_silver, silver_bt, SIL_DEF),
        ('NQ Alpha',      df_nq,     nq_a_bt,  NQA_DEF),
        ('NQ Liquidity',  df_nq,     liq_bt,   L_DEF),
        ('NQ SMC',        df_nq,     smc_bt,   SMC_DEF),
    ]

    print(f"\n{'═'*66}")
    print(f"  BASELINE RESULTS (DEFAULT PARAMS)")
    print(f"{'═'*66}")
    print(f"  {'Strategy':<16} {'Return':>8} {'CAGR':>7} {'DD':>7} "
          f"{'Sharpe':>7} {'PF':>6} {'Trades':>7} {'Fitness':>9}")
    print(f"  {'─'*66}")

    for label, df, bt_fn, params in tasks:
        if df is None or (hasattr(df, 'empty') and df.empty):
            print(f"  {label:<16} {'NO DATA':>56}")
            continue
        try:
            r = bt_fn(df, params, args.capital)
            fit = r.fitness() if hasattr(r, 'fitness') else 0
            print(f"  {label:<16} "
                  f"{r.total_return*100:>+7.1f}% "
                  f"{r.cagr*100:>+6.1f}% "
                  f"{r.max_drawdown*100:>6.1f}% "
                  f"{r.sharpe_ratio:>7.2f} "
                  f"{r.profit_factor:>6.2f} "
                  f"{r.total_trades:>7} "
                  f"{fit:>9.2f}")
        except Exception as e:
            print(f"  {label:<16} ERROR: {e}")

    print(f"{'═'*66}")


def mode_deap(args, df_gold, df_silver, df_nq, df_gold_m1, strategy_filter: str = 'all'):
    from deap_runner import run_deap, optimize_all

    out_dir = os.path.join(_ENGINE, 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    data_map = {'gold': df_gold, 'silver': df_silver, 'nq': df_nq, 'gold_m1': df_gold_m1}

    if strategy_filter == 'all':
        optimize_all(
            df_gold=df_gold, df_silver=df_silver, df_nq=df_nq,
            capital=args.capital, population=args.pop,
            generations=args.gen, n_splits=3,
            n_jobs=args.jobs, output_dir=out_dir,
        )
        return

    mod, dk = _get_module(strategy_filter)
    df = data_map.get(dk)
    if df is None or (hasattr(df, 'empty') and df.empty):
        print(f"[WARN]  No data for {strategy_filter}")
        return

    best_params, best_fit, stats, hof = run_deap(
        module        = mod,
        df_train      = df,
        capital       = args.capital,
        population    = args.pop,
        generations   = args.gen,
        n_splits      = 3,
        n_jobs        = args.jobs,
        verbose       = True,
        output_dir    = out_dir,
        strategy_name = strategy_filter,
    )
    print(f"\n  [OK] Best fitness: {best_fit:.4f}")
    print(f"   Params: {vars(best_params) if hasattr(best_params, '__dict__') else best_params}")


def mode_walk_forward(args, df_gold, df_silver, df_nq, strategy_filter: str = 'all'):
    from walk_forward import run_walk_forward

    STRATEGIES = ['gold_sniper', 'gold_master', 'silver', 'nq_alpha', 'nq_liquidity', 'nq_smc']
    targets = STRATEGIES if strategy_filter == 'all' else [strategy_filter]
    data_map = {'gold': df_gold, 'silver': df_silver, 'nq': df_nq}
    out_dir  = os.path.join(_ENGINE, 'outputs')

    for name in targets:
        try:
            mod, dk = _get_module(name)
        except ValueError as e:
            print(f"[WARN]  {e}")
            continue

        df = data_map.get(dk)
        if df is None or (hasattr(df, 'empty') and df.empty):
            print(f"[WARN]  No data for {name}")
            continue

        run_walk_forward(
            module          = mod,
            df              = df,
            strategy_name   = name,
            capital         = args.capital,
            train_months    = args.wf_train,
            test_months     = args.wf_test,
            population      = args.pop,
            generations     = args.gen,
            n_splits_inner  = 2,
            n_jobs          = args.jobs,
            output_dir      = out_dir,
            verbose         = True,
        )


def mode_portfolio(args, df_gold, df_silver, df_nq):
    from portfolio_bt import run_portfolio
    print("\n=== Portfolio Backtest (Default Params) ===")
    r = run_portfolio(
        df_gold=df_gold, df_silver=df_silver, df_nq=df_nq,
        initial_capital=args.capital,
    )
    r.print_summary()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    args = _parse()
    if args.quick:
        args.pop = 30
        args.gen = 30
        print("[FAST] Quick mode: pop=30, gen=30")

    print("=" * 66)
    print("  EvolutionaryQuant — Alpha Engine Master Runner")
    print("=" * 66)
    print(f"  Mode      : {args.mode}")
    print(f"  Strategy  : {args.strategy}")
    print(f"  Capital   : ${args.capital:,.2f}")
    if args.mode in ('deap', 'wf', 'full'):
        print(f"  Pop/Gen   : {args.pop}/{args.gen}")

    df_gold, df_silver, df_nq, df_gold_m1 = _load_data(args)

    if args.mode == 'baseline':
        mode_baseline(args, df_gold, df_silver, df_nq, df_gold_m1)

    elif args.mode == 'deap':
        mode_deap(args, df_gold, df_silver, df_nq, df_gold_m1, args.strategy)

    elif args.mode == 'wf':
        mode_walk_forward(args, df_gold, df_silver, df_nq, args.strategy)

    elif args.mode == 'full':
        print("\n[1/2] DEAP Optimization...")
        mode_deap(args, df_gold, df_silver, df_nq, df_gold_m1, args.strategy)
        print("\n[2/2] Walk-Forward Validation...")
        mode_walk_forward(args, df_gold, df_silver, df_nq, args.strategy)

    elif args.mode == 'portfolio':
        mode_portfolio(args, df_gold, df_silver, df_nq)

    print("\n  [DONE]")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
