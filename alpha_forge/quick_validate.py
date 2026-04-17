"""
Best params → Walk-Forward (15 window) + Monte Carlo → Rapor
"""
import json, sys, os, logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PARAMS_FILE = "outputs/deap_results_20260406_181722.json"
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Load params ───────────────────────────────────────────────────────────────
with open(PARAMS_FILE) as f:
    data = json.load(f)
best_params = data["best_params"]
logger.info(f"Params: fitness={data['fitness']:.4f} | return={data['metrics']['total_return']:+.1f}% | trades={data['metrics']['trade_count']}")

# ── Load raw data ─────────────────────────────────────────────────────────────
def load_mt4_csv(path):
    df = pd.read_csv(path, header=None, names=["date","time","open","high","low","close","volume"])
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M", errors="coerce")
    df = df.dropna(subset=["datetime"]).set_index("datetime")
    return df[["open","high","low","close","volume"]].astype(float).sort_index()

def load_gold(root):
    frames = []
    for yr in range(2019, 2026):
        for ext in ["csv", "txt"]:
            p = os.path.join(root, f"DAT_MT_XAUUSD_M1_{yr}.{ext}")
            if os.path.exists(p):
                frames.append(load_mt4_csv(p))
                logger.info(f"  Gold {yr}: loaded")
                break
    return pd.concat(frames).sort_index() if frames else None

logger.info("Loading data...")
gold_m1 = load_gold(ROOT_DIR)
nq_5m   = pd.read_pickle(os.path.join(ROOT_DIR, "cache_qqq_5m.pkl"))
logger.info(f"Gold M1: {len(gold_m1):,} | NQ 5m: {len(nq_5m):,}")

# Resample gold M1 → 1H
gold_1h = gold_m1.resample("1h").agg(
    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
).dropna()
logger.info(f"Gold 1H: {len(gold_1h):,} bars")

# ── Oracle with correct cache key ─────────────────────────────────────────────
from src.features.oracle import OracleLayer

def get_oracle(df, instrument, suffix=""):
    os.makedirs("data", exist_ok=True)
    last_ts = str(df.index[-1])[:10].replace("-","")
    cp = f"data/oracle_{instrument.lower()}{suffix}_{last_ts}.pkl"
    if os.path.exists(cp):
        cached = pd.read_pickle(cp)
        if len(cached) > len(df) * 2:
            logger.warning(f"Cache stale (rows mismatch), recomputing {cp}")
            os.remove(cp)
        else:
            logger.info(f"Oracle cache: {cp} ({len(cached):,} rows)")
            return cached
    logger.info(f"Computing oracle for {instrument}{suffix}...")
    o = OracleLayer().build_oracle_features(df, instrument)
    o.to_pickle(cp)
    logger.info(f"Saved: {cp}")
    return o

gold_oracle = get_oracle(gold_1h, "XAUUSD", suffix="_1h")
nq_oracle   = get_oracle(nq_5m,   "NQ")
logger.info(f"Gold oracle 1H: {len(gold_oracle):,} | NQ oracle: {len(nq_oracle):,}")

# ── Align by datetime ─────────────────────────────────────────────────────────
common_start = max(nq_oracle.index[0],   gold_oracle.index[0])
common_end   = min(nq_oracle.index[-1],  gold_oracle.index[-1])
nq_aligned   = nq_oracle[(nq_oracle.index   >= common_start) & (nq_oracle.index   <= common_end)]
gold_aligned = gold_oracle[(gold_oracle.index >= common_start) & (gold_oracle.index <= common_end)]
logger.info(f"Aligned: {common_start.date()} -> {common_end.date()} | NQ {len(nq_aligned):,} | Gold {len(gold_aligned):,}")

# ── Walk-Forward (datetime-based windows) ─────────────────────────────────────
from src.evolution.optimizer import MultiAssetBacktester
from src.utils.validation import MonteCarloSimulator

total_days   = (common_end - common_start).days
window_days  = max(30, total_days // 15)
logger.info(f"Walk-Forward: ~15 windows x {window_days} days | total {total_days} days")

wf_returns    = []
wf_trades_all = []

for i in range(15):
    w_start = common_start + pd.Timedelta(days=i * window_days)
    w_split = w_start      + pd.Timedelta(days=int(window_days * 0.75))
    w_end   = w_start      + pd.Timedelta(days=window_days)
    if w_end > common_end:
        break

    nq_test   = nq_aligned[(nq_aligned.index     >= w_split) & (nq_aligned.index   < w_end)]
    gold_test = gold_aligned[(gold_aligned.index >= w_split) & (gold_aligned.index < w_end)]

    if len(nq_test) < 100 or len(gold_test) < 20:
        logger.info(f"  Window {i+1:02d}: skip (NQ={len(nq_test)} Gold={len(gold_test)})")
        continue

    try:
        bt = MultiAssetBacktester(gold_test, nq_test)
        bal, trades = bt.run(best_params)
        ret = (bal / 950.0 - 1) * 100
        wf_returns.append(ret)
        wf_trades_all.extend(trades)
        tag = "WIN " if ret > 0 else "LOSS"
        logger.info(f"  Window {i+1:02d} [{w_split.date()}->{w_end.date()}]: {tag} {ret:+.1f}% | ${bal:.0f} | {len(trades)} trades")
    except Exception as e:
        logger.warning(f"  Window {i+1:02d}: ERROR - {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
logger.info("=" * 65)
logger.info("WALK-FORWARD SUMMARY")
if wf_returns:
    n_win = sum(1 for r in wf_returns if r > 0)
    avg_r = np.mean(wf_returns)
    logger.info(f"  Windows          : {len(wf_returns)}")
    logger.info(f"  Profitable       : {n_win}/{len(wf_returns)} ({100*n_win/len(wf_returns):.0f}%)")
    logger.info(f"  Avg return       : {avg_r:+.1f}%")
    logger.info(f"  Min / Max        : {min(wf_returns):+.1f}% / {max(wf_returns):+.1f}%")
    logger.info(f"  Std dev          : {np.std(wf_returns):.1f}%")
    logger.info(f"  Total WF trades  : {len(wf_trades_all)}")

    # Projection
    test_days  = window_days * 0.25
    daily_ret  = avg_r / test_days if test_days > 0 else 0
    monthly    = daily_ret * 30
    d60        = (1 + daily_ret/100) ** 60 * 1000
    logger.info(f"\n  PROJECTION (test window ~{test_days:.0f} days):")
    logger.info(f"  Est daily return : {daily_ret:+.3f}%")
    logger.info(f"  Est monthly      : {monthly:+.1f}%")
    logger.info(f"  $1k after 60d    : ${d60:,.0f}")
    logger.info(f"  $1k after 6mo    : ${(1+daily_ret/100)**180*1000:,.0f}")
else:
    logger.error("No windows ran!")

# ── Monte Carlo ───────────────────────────────────────────────────────────────
if wf_trades_all:
    logger.info("=" * 65)
    logger.info("MONTE CARLO (2000 sims)")
    mc = MonteCarloSimulator(n_simulations=2000)
    mc_res = mc.run_permutation_test(wf_trades_all)
    logger.info(f"  P(Profit)        : {mc_res['probability_profit']*100:.1f}%")
    logger.info(f"  P(Ruin)          : {mc_res['probability_ruin']*100:.1f}%")
    logger.info(f"  Median balance   : ${mc_res['final_balance']['median']:,.0f}")
    logger.info(f"  5th pct          : ${mc_res['final_balance']['percentile_5']:,.0f}")
    logger.info(f"  95th pct         : ${mc_res['final_balance']['percentile_95']:,.0f}")
    logger.info(f"  Median Max DD    : {mc_res['max_drawdown']['median']*100:.1f}%")

logger.info("=" * 65)
logger.info("DONE")
