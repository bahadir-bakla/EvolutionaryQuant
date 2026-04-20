"""
Algo Director — Gold Scalper + NQ Liquidity Live Trader
=========================================================
Algo Core live execution engine.  Completely independent from
ml_scalper_live.py (different MAGIC number: 20260419).

Strategies
----------
  gold_scalper   ICT M1 scalper on XAUUSD  (London + NY kill zones)
  nq_liquidity   ORB reversal on NQ100     (Opening Range sweep)

Intelligence Layer
------------------
  MiroFish       Background crowd-simulation sentiment (4-hour cache)
  News feed      Headline refresh every 30 min

Usage
-----
  python algo_director.py                 # live mode
  python algo_director.py --paper         # paper trading
  python algo_director.py --disable-nq    # gold only
  python algo_director.py --disable-mirofish
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from alpha_engine.gold_scalper_bt import (
    GoldScalperParams,
    _add_features as gold_add_features,
    _in_killzone,
    GOLD_PV,
)
from alpha_engine.nq_liquidity_bt import NQLiquidityParams, NQ_PV_FULL, NQ_PV_MICRO
from mt5_bridge.connector import MT5Connector
from mt5_bridge.order_manager import MT5OrderManager

try:
    from alpha_forge.src.intelligence.mirofish_bridge import MiroFishBridge
    MIROFISH_OK = True
except ImportError:
    MIROFISH_OK = False

try:
    from alpha_forge.src.intelligence.news_feed import NewsFeed
    NEWSFEED_OK = True
except ImportError:
    NEWSFEED_OK = False

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / 'logs' / 'algo_director'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"live_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('algo_director')

# ─── Constants ───────────────────────────────────────────────────────────────
MAGIC        = 20260419          # Algo Core — distinct from ML Scalper (20260418)
GOLD_SYMBOL  = 'XAUUSD'
NQ_SYMBOL    = 'NAS100'          # Adjust to your broker (NAS100, USTECH100, NQ100)
WARMUP_BARS  = 500
POLL_SECONDS = 5
STATE_FILE   = Path(__file__).parent / 'logs' / 'algo_director' / 'algo_state.json'
OUTPUTS_DIR  = Path(__file__).parent / 'alpha_engine' / 'outputs'


# ─── DEAP Param Loader ───────────────────────────────────────────────────────

def _load_best_params(strategy: str) -> dict:
    """Return best_params dict from the latest DEAP output JSON."""
    files = sorted(OUTPUTS_DIR.glob(f'{strategy}_*.json'))
    if not files:
        log.warning(f"No DEAP output for {strategy} — using defaults")
        return {}
    latest = files[-1]
    log.info(f"Loading {strategy} params from {latest.name}")
    try:
        return json.loads(latest.read_text()).get('best_params', {})
    except Exception as e:
        log.warning(f"Param load failed: {e}")
        return {}


def _make_gold_params() -> GoldScalperParams:
    raw = _load_best_params('gold_scalper')
    p = GoldScalperParams()
    INT_FIELDS = {'ob_lookback', 'trend_bars', 'momentum_bars', 'macro_ma_days'}
    for k, v in raw.items():
        if hasattr(p, k):
            setattr(p, k, int(round(float(v))) if k in INT_FIELDS else float(v))
    return p


def _make_nq_params() -> NQLiquidityParams:
    raw = _load_best_params('nq_liquidity')
    p = NQLiquidityParams()
    INT_FIELDS = {'or_end_offset_min', 'session_end_hour'}
    for k, v in raw.items():
        if hasattr(p, k):
            setattr(p, k, int(round(float(v))) if k in INT_FIELDS else float(v))
    return p


# ─── Sentiment Cache (thread-safe) ───────────────────────────────────────────

class SentimentCache:
    """MiroFish result refreshed in a background thread every 4 hours."""

    TTL = 4 * 3600

    def __init__(self) -> None:
        self._score   = 0.0
        self._action  = 'FLAT'
        self._summary = ''
        self._ts      = 0.0
        self._lock    = threading.Lock()

    def update(self, score: float, action: str, summary: str) -> None:
        with self._lock:
            self._score   = score
            self._action  = action
            self._summary = summary
            self._ts      = time.time()

    @property
    def score(self) -> float:
        with self._lock:
            return self._score

    @property
    def action(self) -> str:
        with self._lock:
            return self._action

    @property
    def stale(self) -> bool:
        with self._lock:
            return (time.time() - self._ts) > self.TTL

    def to_dict(self) -> dict:
        with self._lock:
            return {
                'score':   round(self._score, 3),
                'action':  self._action,
                'summary': self._summary[:200],
                'age_min': round((time.time() - self._ts) / 60, 1),
            }


_sentiment = SentimentCache()


def _mirofish_worker(headlines: list, market_ctx: dict) -> None:
    try:
        bridge = MiroFishBridge()
        result = bridge.query(headlines, market_ctx, sim_rounds=20)
        _sentiment.update(result.score, result.action, result.summary)
        log.info(f"MiroFish: score={result.score:.3f} action={result.action}")
    except Exception as e:
        log.warning(f"MiroFish worker error: {e}")
        # Hata durumunda timestamp'i güncelle — 4 saat retry olmaz (rate limit koruması)
        d = _sentiment.to_dict()
        _sentiment.update(d['score'], d['action'], d.get('summary', ''))


def _trigger_mirofish(gold_m1: pd.DataFrame, headlines: list) -> None:
    """Kick off a background MiroFish query if sentiment is stale."""
    if not MIROFISH_OK or not _sentiment.stale:
        return
    try:
        close = gold_m1['close']
        price = float(close.iloc[-1])
        delta = close.diff()
        gain  = delta.clip(lower=0).ewm(com=13, min_periods=14).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=13, min_periods=14).mean()
        rsi   = float((100 - 100 / (1 + gain / (loss + 1e-10))).iloc[-1])
        tr    = pd.concat([
            gold_m1['high'] - gold_m1['low'],
            (gold_m1['high'] - close.shift(1)).abs(),
            (gold_m1['low']  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = float(tr.ewm(span=14, min_periods=14).mean().iloc[-1])

        ctx = {
            'instrument': 'XAUUSD',
            'price': round(price, 2),
            'trend': 'bullish' if _sentiment.score >= 0 else 'bearish',
            'rsi':   round(rsi, 1),
            'atr':   round(atr, 2),
        }
        threading.Thread(
            target=_mirofish_worker, args=(headlines, ctx), daemon=True
        ).start()
        log.info("MiroFish refresh triggered (background)")
    except Exception as e:
        log.warning(f"MiroFish trigger error: {e}")


# ─── Strategy State ───────────────────────────────────────────────────────────

@dataclass
class StrategyState:
    name:           str
    symbol:         str
    equity:         float = 10_000.0
    open_ticket:    Optional[int] = None
    open_dir:       Optional[str] = None
    open_entry:     float = 0.0
    open_tp:        float = 0.0
    open_sl:        float = 0.0
    open_lot:       float = 0.01
    open_bar_count: int   = 0
    last_bar_time:  Optional[pd.Timestamp] = None
    trade_log:      list  = field(default_factory=list)


# ─── Gold Scalper Signal ─────────────────────────────────────────────────────

def gold_scalper_signal(
    m1: pd.DataFrame,
    params: GoldScalperParams,
) -> tuple[Optional[str], str]:
    """
    Run _add_features on recent M1 data and check the last completed bar (-2).
    Returns (direction, reason) or (None, '').
    """
    try:
        df = gold_add_features(m1.copy(), params)
        df = df.dropna(subset=['atr', 'trend_bull', 'rsi'])
        if len(df) < params.trend_bars + 20:
            return None, ''

        row  = df.iloc[-2]
        hour = df.index[-2].hour

        in_kz, allow_shorts = _in_killzone(hour, params.session)
        if not in_kz:
            return None, ''

        rsi_v    = float(row['rsi'])
        meta     = float(row.get('meta_bias', 0.0))
        thresh   = params.meta_bias_thresh
        vol_ok   = float(row.get('atr_ratio', 1.0)) >= (
            params.vol_thresh if getattr(params, 'vol_thresh', 1.0) > 1.0 else 0.0
        )
        macro_ok = bool(row.get('macro_bull', True))

        bull_zone = bool(row.get('bull_fvg_active')) or bool(row.get('bull_ob_active')) or bool(row.get('pdl_reject'))
        bear_zone = bool(row.get('bear_fvg_active')) or bool(row.get('bear_ob_active')) or bool(row.get('pdh_reject'))

        long_ok_meta  = True
        short_ok_meta = allow_shorts
        if thresh < 0.98:
            if meta > thresh:
                short_ok_meta = False
            elif meta < -thresh:
                long_ok_meta = False

        trend_bull = bool(row.get('trend_bull', False))
        trend_bear = bool(row.get('trend_bear', False))

        long_ok  = trend_bull and bull_zone and (30.0 < rsi_v < 55.0) and long_ok_meta  and vol_ok and macro_ok
        short_ok = trend_bear and bear_zone and (60.0 < rsi_v < 75.0) and short_ok_meta and vol_ok

        if long_ok and not short_ok:
            if bool(row.get('bull_fvg_active')): return 'LONG', 'FVG_LONG'
            if bool(row.get('bull_ob_active')):  return 'LONG', 'OB_LONG'
            return 'LONG', 'PDL_LONG'

        if short_ok and not long_ok:
            if bool(row.get('bear_fvg_active')): return 'SHORT', 'FVG_SHORT'
            if bool(row.get('bear_ob_active')):  return 'SHORT', 'OB_SHORT'
            return 'SHORT', 'PDH_SHORT'

    except Exception as e:
        log.warning(f"gold_scalper_signal error: {e}")

    return None, ''


# ─── NQ Liquidity Signal ─────────────────────────────────────────────────────

def nq_liquidity_signal(
    bars_5m: pd.DataFrame,
    params: NQLiquidityParams,
    trade_count_today: int,
) -> Optional[str]:
    """
    Process today's 5-min bars for NQ ORB sweep reversal.
    Returns 'LONG', 'SHORT', or None.

    Opening Range: 9:30 – (10:00 + or_end_offset_min) in broker bar time.
    """
    try:
        if bars_5m.empty or len(bars_5m) < 20:
            return None
        if trade_count_today >= 2:
            return None

        df = bars_5m.copy()
        df.columns = df.columns.str.lower()

        tr   = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low']  - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_s = tr.rolling(14).mean()

        # Use last completed bar
        row   = df.iloc[-2]
        ts    = df.index[-2]
        today = ts.date()

        # Determine OR window (in minutes-of-day)
        or_start_min = 9 * 60 + 30    # 9:30
        or_end_min   = 10 * 60 + int(params.or_end_offset_min)  # 10:00 + offset

        # Build today's OR from today's bars
        today_bars = df[df.index.date == today]
        if today_bars.empty:
            return None

        or_high = -np.inf
        or_low  = np.inf
        or_locked = False

        for bar_ts, bar_row in today_bars.iterrows():
            bm = bar_ts.hour * 60 + bar_ts.minute
            if or_start_min <= bm < or_end_min:
                or_high = max(or_high, float(bar_row['high']))
                or_low  = min(or_low,  float(bar_row['low']))
            elif bm >= or_end_min and or_high > or_low:
                or_locked = True

        if not or_locked or or_high == -np.inf or or_low == np.inf:
            return None

        t_min = ts.hour * 60 + ts.minute
        if t_min >= params.session_end_hour * 60:
            return None

        atr = float(atr_s.iloc[-2])
        if atr <= 0 or np.isnan(atr):
            return None

        hi    = float(row['high'])
        lo    = float(row['low'])
        price = float(row['close'])
        o_px  = float(row['open'])
        body  = abs(price - o_px)
        wick_up = hi - max(price, o_px)
        wick_dn = min(price, o_px) - lo
        meta    = float(row.get('meta_bias', 0.0))

        sweep_ext = params.sweep_extension_atr * atr

        bear_sweep  = (hi > or_high + sweep_ext) and (price < or_high)
        bear_reject = (wick_up > 1e-6) and (body / max(wick_up, 1e-6) < params.sweep_body_ratio)
        bull_sweep  = (lo < or_low - sweep_ext) and (price > or_low)
        bull_reject = (wick_dn > 1e-6) and (body / max(wick_dn, 1e-6) < params.sweep_body_ratio)

        if abs(meta) < params.meta_bias_thresh:
            if bear_sweep and bear_reject and meta <= 0:
                return 'SHORT'
            if bull_sweep and bull_reject and meta >= 0:
                return 'LONG'

    except Exception as e:
        log.warning(f"nq_liquidity_signal error: {e}")

    return None


# ─── Position Helpers ─────────────────────────────────────────────────────────

def sync_equity(state: StrategyState, paper: bool) -> None:
    if paper:
        return
    try:
        import MetaTrader5 as mt5
        info = mt5.account_info()
        if info:
            state.equity = info.equity
    except Exception:
        pass


def check_position(
    order_mgr: MT5OrderManager,
    state: StrategyState,
    point_value: float,
    paper: bool,
) -> bool:
    """
    Returns True if position is still open in MT5.
    On close: appends trade record, clears state.
    """
    if state.open_ticket is None:
        return False

    if paper:
        state.open_bar_count += 1
        return True

    try:
        positions = order_mgr.get_positions(symbol=state.symbol)
        tickets   = {p['ticket'] for p in positions if p.get('magic') == MAGIC}
    except Exception:
        return True

    if state.open_ticket not in tickets:
        # MT5 closed the position (TP/SL hit server-side)
        try:
            import MetaTrader5 as mt5
            tick    = mt5.symbol_info_tick(state.symbol)
            last_px = tick.bid if tick else state.open_entry
        except Exception:
            last_px = state.open_entry

        if state.open_dir == 'LONG':
            raw_pnl = last_px - state.open_entry
        else:
            raw_pnl = state.open_entry - last_px

        pnl = round(raw_pnl * state.open_lot * point_value, 2)
        log.info(f"[{state.name}] Position {state.open_ticket} closed | PnL ${pnl:+.2f}")

        state.trade_log.append({
            'ticket':    state.open_ticket,
            'dir':       state.open_dir,
            'entry':     state.open_entry,
            'exit':      round(last_px, 2),
            'lot':       state.open_lot,
            'pnl':       pnl,
            'bars_held': state.open_bar_count,
            'close_ts':  datetime.now(timezone.utc).isoformat(),
            'reason':    'MT5_CLOSE',
        })
        state.open_ticket    = None
        state.open_dir       = None
        state.open_bar_count = 0
        return False

    state.open_bar_count += 1
    return True


def open_trade(
    order_mgr: MT5OrderManager,
    state: StrategyState,
    direction: str,
    tp_pts: float,
    sl_pts: float,
    lot: float,
    reason: str,
    point_value: float,
    paper: bool,
    current_price: float = 0.0,
) -> None:
    """Send market order (or simulate in paper mode)."""
    if not paper:
        try:
            import MetaTrader5 as mt5
            tick = mt5.symbol_info_tick(state.symbol)
            if tick is None:
                log.error(f"[{state.name}] Cannot get tick for {state.symbol}")
                return
            entry = tick.ask if direction == 'LONG' else tick.bid
        except Exception as e:
            log.error(f"[{state.name}] MT5 tick error: {e}")
            return
    else:
        entry = current_price if current_price > 0 else 1.0

    sl = round(entry - sl_pts, 2) if direction == 'LONG' else round(entry + sl_pts, 2)
    tp = round(entry + tp_pts, 2) if direction == 'LONG' else round(entry - tp_pts, 2)

    log.info(
        f"[{state.name}] SIGNAL {direction} | entry={entry:.2f} "
        f"sl={sl:.2f} tp={tp:.2f} lot={lot} reason={reason}"
    )

    if paper:
        state.open_ticket    = -1
        state.open_dir       = direction
        state.open_entry     = entry
        state.open_sl        = sl
        state.open_tp        = tp
        state.open_lot       = lot
        state.open_bar_count = 0
        return

    ticket = order_mgr.open_order(
        symbol=state.symbol,
        direction=direction,
        volume=lot,
        stop_loss=sl,
        take_profit=tp,
        magic=MAGIC,
    )

    if ticket:
        state.open_ticket    = ticket
        state.open_dir       = direction
        state.open_entry     = entry
        state.open_sl        = sl
        state.open_tp        = tp
        state.open_lot       = lot
        state.open_bar_count = 0
        log.info(f"[{state.name}] Order placed: ticket={ticket}")
    else:
        log.error(f"[{state.name}] Order placement failed")


# ─── State Persistence ────────────────────────────────────────────────────────

def _trade_stats(trade_log: list) -> dict:
    trades = trade_log[-200:]
    wins   = [t for t in trades if t.get('pnl', 0) > 0]
    losses = [t for t in trades if t.get('pnl', 0) <= 0]
    gp = sum(t.get('pnl', 0) for t in wins)
    gl = abs(sum(t.get('pnl', 0) for t in losses))
    return {
        'total_trades': len(trades),
        'wins':         len(wins),
        'losses':       len(losses),
        'win_rate':     len(wins) / len(trades) if trades else 0.0,
        'gross_profit': round(gp, 2),
        'gross_loss':   round(gl, 2),
        'profit_factor': round(gp / gl, 3) if gl > 0 else 0.0,
        'total_pnl':    round(gp - gl, 2),
    }


def save_state(
    gold: StrategyState,
    nq: StrategyState,
    gold_p: GoldScalperParams,
    nq_p: NQLiquidityParams,
    mode: str,
) -> None:
    def _pos(s: StrategyState) -> dict:
        return {
            'active':    s.open_ticket is not None,
            'ticket':    s.open_ticket,
            'direction': s.open_dir,
            'entry':     s.open_entry,
            'tp':        s.open_tp,
            'sl':        s.open_sl,
            'lot':       s.open_lot,
            'bars_held': s.open_bar_count,
        }

    data = {
        'updated_at':  datetime.now(timezone.utc).isoformat(),
        'mode':        mode,
        'sentiment':   _sentiment.to_dict(),
        'gold_scalper': {
            'symbol':        GOLD_SYMBOL,
            'equity':        gold.equity,
            'open_position': _pos(gold),
            'stats':         _trade_stats(gold.trade_log),
            'trades':        gold.trade_log[-100:],
            'params': {
                'session':          gold_p.session,
                'tp_pts':           gold_p.tp_pts,
                'sl_pts':           gold_p.sl_pts,
                'meta_bias_thresh': gold_p.meta_bias_thresh,
                'lot_base':         gold_p.lot_base,
            },
        },
        'nq_liquidity': {
            'symbol':        NQ_SYMBOL,
            'equity':        nq.equity,
            'open_position': _pos(nq),
            'stats':         _trade_stats(nq.trade_log),
            'trades':        nq.trade_log[-100:],
            'params': {
                'sweep_extension_atr': nq_p.sweep_extension_atr,
                'stop_loss_atr':       nq_p.stop_loss_atr,
                'take_profit_atr':     nq_p.take_profit_atr,
                'or_end_offset_min':   nq_p.or_end_offset_min,
                'session_end_hour':    nq_p.session_end_hour,
            },
        },
    }
    STATE_FILE.write_text(json.dumps(data, indent=2, default=str))


# ─── News Feed ────────────────────────────────────────────────────────────────

def _fetch_headlines() -> list:
    if not NEWSFEED_OK:
        return []
    try:
        feed = NewsFeed()
        items = feed.fetch(instrument='XAUUSD', limit=15)
        return [i.title for i in items]
    except Exception:
        return []


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run_live(args: argparse.Namespace) -> None:
    log.info("=" * 60)
    log.info("  Algo Director — Gold Scalper + NQ Liquidity")
    log.info("=" * 60)
    log.info(f"  Paper mode  : {args.paper}")
    log.info(f"  NQ enabled  : {not args.disable_nq}")
    log.info(f"  MiroFish    : {not args.disable_mirofish and MIROFISH_OK}")

    gold_params = _make_gold_params()
    nq_params   = _make_nq_params()

    log.info(f"  Gold  params: session={gold_params.session:.2f}  "
             f"tp={gold_params.tp_pts:.1f}  sl={gold_params.sl_pts:.1f}  "
             f"lot_base={gold_params.lot_base:.4f}")
    log.info(f"  NQ    params: sweep_ext={nq_params.sweep_extension_atr:.3f}  "
             f"sl={nq_params.stop_loss_atr:.3f}  tp={nq_params.take_profit_atr:.3f}")
    log.info("")

    connector = MT5Connector()
    order_mgr = MT5OrderManager()

    if not args.paper:
        if not connector.connect():
            log.error("MT5 connection failed — exiting")
            return
        log.info("MT5 connected")

    gold_state = StrategyState(name='gold_scalper', symbol=GOLD_SYMBOL)
    nq_state   = StrategyState(name='nq_liquidity',  symbol=NQ_SYMBOL)

    use_nq       = not args.disable_nq
    use_mirofish = not args.disable_mirofish

    nq_trades_today: int  = 0
    nq_last_date: Optional[date] = None
    news_headlines: list  = []
    last_news_ts:   float = 0.0

    log.info("Entering main loop... (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(POLL_SECONDS)

            if args.paper:
                log.debug("[PAPER] No live MT5 data — idle")
                continue

            # ── Fetch XAUUSD M1 ─────────────────────────────────────────────
            gold_m1 = connector.fetch_data(GOLD_SYMBOL, '1m', bars=WARMUP_BARS)
            if gold_m1.empty or len(gold_m1) < 200:
                log.warning("Insufficient XAUUSD data, skipping")
                continue

            gold_m1.columns = gold_m1.columns.str.lower()
            sync_equity(gold_state, args.paper)
            nq_state.equity = gold_state.equity   # shared account

            # ── Refresh news every 30 min ────────────────────────────────────
            if time.time() - last_news_ts > 1800:
                news_headlines = _fetch_headlines()
                last_news_ts   = time.time()

            # ── MiroFish background refresh ──────────────────────────────────
            if use_mirofish:
                _trigger_mirofish(gold_m1, news_headlines)

            # ── Gold Scalper ─────────────────────────────────────────────────
            gold_latest = gold_m1.index[-2]
            if gold_latest != gold_state.last_bar_time:
                gold_state.last_bar_time = gold_latest

                # Manage open position
                if gold_state.open_ticket is not None:
                    check_position(order_mgr, gold_state, GOLD_PV, args.paper)

                # Entry gate
                if gold_state.open_ticket is None:
                    direction, reason = gold_scalper_signal(gold_m1, gold_params)

                    # MiroFish veto: block if sentiment strongly contradicts
                    if direction and use_mirofish and abs(_sentiment.score) > 0.4:
                        mf_action = _sentiment.action
                        if (direction == 'LONG' and mf_action == 'SHORT') or \
                           (direction == 'SHORT' and mf_action == 'LONG'):
                            log.info(
                                f"[gold_scalper] {direction} vetoed by MiroFish "
                                f"({mf_action} score={_sentiment.score:.2f})"
                            )
                            direction = None

                    if direction:
                        lot = max(0.01, round(gold_state.equity / 1000.0 * gold_params.lot_base, 2))
                        open_trade(
                            order_mgr, gold_state, direction,
                            gold_params.tp_pts, gold_params.sl_pts,
                            lot, reason, GOLD_PV, args.paper,
                            current_price=float(gold_m1['close'].iloc[-2]),
                        )

                save_state(gold_state, nq_state, gold_params, nq_params,
                           'paper' if args.paper else 'live')

            # ── NQ Liquidity ─────────────────────────────────────────────────
            if use_nq:
                nq_5m = connector.fetch_data(NQ_SYMBOL, '5m', bars=300)

                if not nq_5m.empty and len(nq_5m) > 20:
                    nq_5m.columns = nq_5m.columns.str.lower()
                    pv = NQ_PV_FULL if nq_5m['close'].median() > 5000 else NQ_PV_MICRO

                    today = nq_5m.index[-2].date()
                    if today != nq_last_date:
                        nq_trades_today = 0
                        nq_last_date    = today

                    # Manage open NQ position
                    if nq_state.open_ticket is not None:
                        check_position(order_mgr, nq_state, pv, args.paper)

                    # Entry gate
                    if nq_state.open_ticket is None:
                        direction = nq_liquidity_signal(nq_5m, nq_params, nq_trades_today)

                        if direction:
                            tr   = pd.concat([
                                nq_5m['high'] - nq_5m['low'],
                                (nq_5m['high'] - nq_5m['close'].shift(1)).abs(),
                                (nq_5m['low']  - nq_5m['close'].shift(1)).abs(),
                            ], axis=1).max(axis=1)
                            atr = float(tr.rolling(14).mean().iloc[-2])
                            sl_pts = nq_params.stop_loss_atr * atr
                            tp_pts = nq_params.take_profit_atr * atr
                            risk   = nq_state.equity * nq_params.lot_base
                            lot    = max(0.001, round(risk / max(sl_pts * pv, 1e-6), 3))

                            open_trade(
                                order_mgr, nq_state, direction,
                                tp_pts, sl_pts, lot,
                                'NQ_SWEEP', pv, args.paper,
                                current_price=float(nq_5m['close'].iloc[-2]),
                            )
                            nq_trades_today += 1

                    save_state(gold_state, nq_state, gold_params, nq_params,
                               'paper' if args.paper else 'live')

    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        if not args.paper:
            connector.disconnect()
        log.info("Session ended")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    global NQ_SYMBOL
    p = argparse.ArgumentParser(description='Algo Director — Gold Scalper + NQ Liquidity')
    p.add_argument('--paper',            action='store_true', help='Paper trading — no real orders')
    p.add_argument('--disable-nq',       action='store_true', help='Disable NQ Liquidity strategy')
    p.add_argument('--disable-mirofish', action='store_true', help='Disable MiroFish filter')
    p.add_argument('--nq-symbol',        default=NQ_SYMBOL,   help=f'MT5 NQ symbol (default: {NQ_SYMBOL})')
    args = p.parse_args()

    # Allow broker-specific NQ symbol override
    NQ_SYMBOL = args.nq_symbol

    run_live(args)


if __name__ == '__main__':
    main()
