"""
Alpha-Forge Main Orchestrator
Ties together: Data → Oracle → Evolution → Validation → Execution
"""

import os
import sys
import json
import time
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.pipeline import AlphaForgeDataPipeline
from src.features.oracle import OracleLayer
from src.evolution.optimizer import DEAPEvolutionEngine
from src.utils.validation import WalkForwardValidator, MonteCarloSimulator
from src.execution.risk import RiskManager, PositionSizer
from src.execution.trade_executor import TradeExecutor
from src.utils.database import TradeLogger
from src.utils.monitoring import SystemMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("alphaforgedb.log"),
    ],
)
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_mt4_csv(filepath: str) -> pd.DataFrame:
    """MT4 CSV formatini yukle: DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL"""
    import glob
    df = pd.read_csv(filepath, header=None,
                     names=["date","time","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"],
                                     format="%Y.%m.%d %H:%M", errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime")
    df = df[["open","high","low","close","volume"]].astype(float)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def _load_gold_data(root_dir: str) -> Optional[pd.DataFrame]:
    """Tum MT4 XAUUSD M1 CSV dosyalarini birlestirir."""
    frames = []
    for yr in range(2019, 2026):
        path = os.path.join(root_dir, f"DAT_MT_XAUUSD_M1_{yr}.csv")
        if os.path.exists(path):
            df = _load_mt4_csv(path)
            frames.append(df)
            logger.info(f"  Gold {yr}: {len(df):,} bar")
    if not frames:
        return None
    return pd.concat(frames).sort_index()


def _load_nq_data(root_dir: str) -> Optional[pd.DataFrame]:
    """QQQ 5m cache'den yukle (Alpaca)."""
    cache = os.path.join(root_dir, "cache_qqq_5m.pkl")
    if os.path.exists(cache):
        df = pd.read_pickle(cache)
        logger.info(f"  QQQ 5m: {len(df):,} bar")
        return df
    return None


def run_phase1_data(config: dict, root_dir: str = "..") -> tuple:
    """
    Phase 1: Veri yukleme.
    Once MT4 CSV + QQQ pickle deneyin, yoksa MT5 pipeline fallback.
    Returns: (nq_df, gold_df)
    """
    logger.info("=" * 60)
    logger.info("PHASE 1: Data Acquisition")
    logger.info("=" * 60)

    gold_df = _load_gold_data(root_dir)
    nq_df   = _load_nq_data(root_dir)

    if nq_df is not None and gold_df is not None:
        logger.info(f"Yerel veri bulundu: Gold {len(gold_df):,} bar | NQ {len(nq_df):,} bar")
        return nq_df, gold_df

    # Fallback: MT5 pipeline (NQ icin)
    logger.warning("Yerel veri bulunamadi — MT5 pipeline deneniyor...")
    pipeline = AlphaForgeDataPipeline(config={
        "instruments": [config["instruments"][0]["name"]],
        "timeframes": ["5m"],
        "context_length": config["data"]["context_length"],
        "forecast_horizon": config["data"]["forecast_horizon"],
    })
    results = pipeline.run_full_pipeline()
    instrument = config["instruments"][0]["name"]
    path = f"data/processed/{instrument}_5m_processed.parquet"
    if os.path.exists(path):
        nq_df = pd.read_parquet(path)
        return nq_df, None
    return None, None


def run_phase2_oracle(df: pd.DataFrame, instrument: str = "NQ",
                      cache_dir: str = "data") -> pd.DataFrame:
    """Phase 2: Build oracle features — uses pkl cache if available."""
    logger.info("=" * 60)
    logger.info("PHASE 2: Oracle Feature Engineering")
    logger.info("=" * 60)

    os.makedirs(cache_dir, exist_ok=True)
    # Cache key: instrument + data fingerprint (last timestamp + row count)
    last_ts = str(df.index[-1])[:10].replace("-", "")
    cache_path = os.path.join(cache_dir, f"oracle_{instrument.lower()}_{last_ts}.pkl")

    if os.path.exists(cache_path):
        try:
            cached = pd.read_pickle(cache_path)
            logger.info(f"Oracle cache hit: {cache_path} ({len(cached):,} rows)")
            return cached
        except Exception as e:
            logger.warning(f"Cache load failed ({e}), recomputing...")

    oracle = OracleLayer()
    oracle_df = oracle.build_oracle_features(df, instrument)

    try:
        oracle_df.to_pickle(cache_path)
        logger.info(f"Oracle cached: {cache_path}")
    except Exception as e:
        logger.warning(f"Cache save failed: {e}")

    return oracle_df


def run_phase3_evolution(oracle_df: pd.DataFrame, config: dict,
                          gold_oracle_df: pd.DataFrame = None) -> dict:
    """Phase 3: DEAP Strategy Evolution (multi-asset)."""
    logger.info("=" * 60)
    logger.info("PHASE 3: DEAP Strategy Evolution")
    if gold_oracle_df is not None:
        logger.info("  Mode: MULTI-ASSET  (Gold + NQ Breakdown + NQ Wick)")
    else:
        logger.info("  Mode: NQ only  (Gold verisi bulunamadi)")
    logger.info("=" * 60)

    engine = DEAPEvolutionEngine(
        oracle_df,
        config=config.get("evolution", {}),
        gold_df=gold_oracle_df,
    )

    results = engine.run()
    return results


def run_phase4_validation(oracle_df: pd.DataFrame, best_params: dict,
                           gold_oracle_df: pd.DataFrame = None) -> dict:
    """Phase 4: Walk-Forward + Monte Carlo."""
    logger.info("=" * 60)
    logger.info("PHASE 4: Validation")
    logger.info("=" * 60)

    from src.evolution.optimizer import (MultiAssetBacktester, StrategyBacktesterLegacy,
                                          AdvancedFitnessCalculator)

    if gold_oracle_df is not None:
        backtester = MultiAssetBacktester(gold_oracle_df, oracle_df)
    else:
        backtester = StrategyBacktesterLegacy(oracle_df)
    fitness_calc = AdvancedFitnessCalculator()

    # Walk-Forward — max 20 window, step büyük tut
    total_bars = len(oracle_df)
    step_bars  = max(200, total_bars // 20)
    wf = WalkForwardValidator(train_bars=1000, test_bars=250, step_bars=step_bars)

    def optimize_func(data):
        return best_params

    def backtest_func(data, params):
        if gold_oracle_df is not None:
            bt = MultiAssetBacktester(gold_oracle_df, data)
        else:
            bt = StrategyBacktesterLegacy(data)
        return bt.run(params)

    wf_results = wf.run(oracle_df, optimize_func, backtest_func)

    # Monte Carlo
    _, trades = backtester.run(best_params)
    mc = MonteCarloSimulator(n_simulations=5000)
    mc_results = mc.run_permutation_test(trades)

    return {
        "walk_forward": wf_results,
        "monte_carlo": mc_results,
    }


def print_final_report(evolution_results: dict, validation_results: dict):
    """Print comprehensive final report."""
    logger.info("=" * 70)
    logger.info("ALPHA-FORGE FINAL REPORT")
    logger.info("=" * 70)

    # Evolution
    params = evolution_results["best_params"]
    metrics = evolution_results["metrics"]

    logger.info("OPTIMIZED PARAMETERS:")
    for k, v in params.items():
        logger.info(f"  {k:30s}: {v}")

    logger.info(f"\nPERFORMANCE METRICS:")
    logger.info(f"  Return:        {metrics.get('total_return', 0):+.2f}%")
    logger.info(f"  Win Rate:      {metrics.get('win_rate', 0)*100:.1f}%")
    logger.info(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    logger.info(f"  Sortino:       {metrics.get('sortino', 0):.2f}")
    logger.info(f"  Sharpe:        {metrics.get('sharpe', 0):.2f}")
    logger.info(f"  Max Drawdown:  {metrics.get('max_dd', 0)*100:.1f}%")
    logger.info(f"  Calmar:        {metrics.get('calmar', 0):.2f}")
    logger.info(f"  Trades:        {metrics.get('trade_count', 0)}")

    # Validation
    if "walk_forward" in validation_results:
        wf = validation_results["walk_forward"]
        logger.info(f"\nWALK-FORWARD ANALYSIS:")
        logger.info(f"  Windows:          {wf['total_windows']}")
        logger.info(f"  Profitable:       {wf['profitable_windows']} ({wf['profitability_rate']:.1f}%)")
        logger.info(f"  Total Return:     {wf['total_return']:+.2f}%")
        logger.info(f"  Avg Return/Win:   {wf['avg_return']:+.2f}%")

    if "monte_carlo" in validation_results:
        mc = validation_results["monte_carlo"]
        logger.info(f"\nMONTE CARLO ({mc['n_simulations']} simulations):")
        logger.info(f"  P(Profit):        {mc['probability_profit']*100:.1f}%")
        logger.info(f"  P(Ruin):          {mc['probability_ruin']*100:.1f}%")
        logger.info(f"  Median Balance:   ${mc['final_balance']['median']:,.2f}")
        logger.info(f"  Median Max DD:    {mc['max_drawdown']['median']*100:.1f}%")

    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Alpha-Forge Trading System")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file")
    parser.add_argument("--skip-data", action="store_true", help="Skip data fetching")
    parser.add_argument("--skip-evolution", action="store_true", help="Skip DEAP evolution")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation")
    parser.add_argument("--load-params", type=str, help="Load params from JSON file")
    parser.add_argument("--live", action="store_true", help="Run live trading mode")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("ALPHA-FORGE: AI-Hybrid Quant Trading System")
    logger.info("=" * 60)

    config     = load_config(args.config)
    instrument = config["instruments"][0]["name"]

    # Proje kok dizini (alpha_forge'un bir ust seviyesi)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Phase 1: Data
    if args.skip_data:
        logger.info("Skipping data phase, loading from cache...")
        path = f"data/processed/{instrument}_5m_processed.parquet"
        if os.path.exists(path):
            nq_df   = pd.read_parquet(path)
            gold_df = _load_gold_data(root_dir)
        else:
            logger.error("No cached data found. Run without --skip-data first.")
            return
    else:
        nq_df, gold_df = run_phase1_data(config, root_dir)
        if nq_df is None:
            logger.error("Data pipeline failed — NQ verisi bulunamadi")
            return

    # Phase 2: Oracle (NQ + Gold ayri ayri)
    # Gold: M1 → 1H resample (Sniper/Master 1H sinyali bekliyor, M1 degil)
    if gold_df is not None:
        gold_1h = gold_df.resample("1h").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}
        ).dropna()
        logger.info(f"Gold M1->1H: {len(gold_df):,} -> {len(gold_1h):,} bar")
    else:
        gold_1h = None

    oracle_df      = run_phase2_oracle(nq_df, instrument, cache_dir="data")
    # XAUUSD_1H suffix kullan — M1 oracle cache ile karismasin
    gold_oracle_df = run_phase2_oracle(gold_1h, "XAUUSD_1H", cache_dir="data") if gold_1h is not None else None
    if gold_oracle_df is not None:
        logger.info(f"Gold oracle hazir: {len(gold_oracle_df):,} bar")

    # Phase 3: Evolution
    if args.load_params:
        with open(args.load_params, "r") as f:
            best_params = json.load(f)
        evolution_results = {"best_params": best_params, "metrics": {}}
        logger.info(f"Params yuklendi: {args.load_params}")
    elif args.skip_evolution:
        evolution_results = {"best_params": {}, "metrics": {}}
        best_params = {}
    else:
        evolution_results = run_phase3_evolution(oracle_df, config, gold_oracle_df)
        best_params = evolution_results["best_params"]

    # Phase 4: Validation
    if args.skip_validation or not best_params:
        validation_results = {}
    else:
        validation_results = run_phase4_validation(oracle_df, best_params, gold_oracle_df)

    # Final Report
    print_final_report(evolution_results, validation_results)

    # Save everything
    os.makedirs("outputs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": ts,
        "instrument": instrument,
        "params": evolution_results.get("best_params", {}),
        "metrics": evolution_results.get("metrics", {}),
        "validation": validation_results,
    }
    with open(f"outputs/report_{ts}.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nReport saved: outputs/report_{ts}.json")

    # Live mode
    if args.live and best_params:
        logger.info("\nStarting LIVE trading mode...")
        risk_mgr = RiskManager(config["risk"])
        position_sizer = PositionSizer(config["position_sizing"])
        executor = TradeExecutor(config.get("execution", {}))
        trade_logger = TradeLogger()
        monitor = SystemMonitor()

        executor.connect()
        try:
            while True:
                monitor.update()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Live trading stopped")
        finally:
            executor.disconnect()


if __name__ == "__main__":
    main()
