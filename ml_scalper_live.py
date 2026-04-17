"""
ML Scalper — VPS Live Trader
==============================
Bağımsız çalışan ML Gold Scalper live execution modülü.
Algo Core / Mirofish pipeline'ından tamamen ayrı çalışır.

Mimari:
  - Her M1 bar kapanışında sinyal üretir
  - LightGBM LONG + SHORT modelleri (lgbm_gold_long/short.pkl)
  - Kelly equity-% cap ile lot sizing
  - MT5'e SL/TP ile market order gönderir
  - Tek pozisyon kuralı (1 trade at a time)

Kullanım (VPS'te):
  python ml_scalper_live.py                        # varsayılan ayarlarla
  python ml_scalper_live.py --threshold 0.75
  python ml_scalper_live.py --kelly 0.25 --kelly_max_risk 1.5
  python ml_scalper_live.py --paper                # paper trading (order yok)

Gereksinimler:
  MetaTrader5, lightgbm, pandas, numpy
  MT5 terminal açık ve logon yapılmış olmalı
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from ml_engine.backtest_ml import MLScalperParams, kelly_lot
from ml_engine.features import build_features
from ml_engine.trainer import load_both_models, load_model
from mt5_bridge.connector import MT5Connector
from mt5_bridge.order_manager import MT5OrderManager

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / 'logs' / 'ml_scalper'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"live_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger('ml_scalper_live')

# ─── Constants ───────────────────────────────────────────────────────────────
SYMBOL       = 'XAUUSD'
MAGIC        = 20260418        # unique ID — ML Scalper trades
WARMUP_BARS  = 600             # bars needed for features (M15 needs 50*15=750 M1 bars)
GOLD_PV      = 100.0
POLL_SECONDS = 5               # how often to check for new bar


# ─── State ───────────────────────────────────────────────────────────────────
@dataclass
class LiveState:
    equity:           float  = 1_000.0
    last_bar_time:    Optional[pd.Timestamp] = None
    open_ticket:      Optional[int]          = None
    open_dir:         Optional[str]          = None   # 'LONG' or 'SHORT'
    open_entry:       float  = 0.0
    open_tp:          float  = 0.0
    open_sl:          float  = 0.0
    open_lot:         float  = 0.01
    open_bar_count:   int    = 0
    trade_log:        list   = field(default_factory=list)


# ─── Signal Engine ───────────────────────────────────────────────────────────

def generate_signal(
    m1: pd.DataFrame,
    long_bundle: dict,
    short_bundle: dict,
    params: MLScalperParams,
) -> tuple[bool, bool, float, float]:
    """
    Build features on latest M1 data and return signals.
    Returns: (want_long, want_short, long_proba, short_proba)
    Signal is based on the LAST COMPLETED bar (index -2 to avoid lookahead).
    """
    X = build_features(m1)
    last_idx = -2   # last completed bar

    def _proba(bundle: dict) -> float:
        Xb = X.reindex(columns=bundle['features'], fill_value=0.0).fillna(0.0)
        if Xb.iloc[last_idx].isna().any():
            return 0.0
        row = Xb.iloc[[last_idx]]
        return float(bundle['model'].predict_proba(row)[0, 1])

    long_p  = _proba(long_bundle)
    short_p = _proba(short_bundle) if short_bundle['model'] else 0.0

    want_long  = long_p  >= params.threshold
    want_short = short_p >= params.threshold

    # Session filter
    now_h = m1.index[-1].hour
    in_session = (7 <= now_h < 11) or (13 <= now_h < 17)
    if not in_session:
        want_long = want_short = False

    # Conflict resolution — take higher proba
    if want_long and want_short:
        if long_p >= short_p:
            want_short = False
        else:
            want_long = False

    return want_long, want_short, long_p, short_p


# ─── Position Management ──────────────────────────────────────────────────────

def sync_equity(connector: MT5Connector, state: LiveState) -> None:
    """Sync equity from MT5 account info."""
    import MetaTrader5 as mt5
    info = mt5.account_info()
    if info:
        state.equity = info.equity
        log.debug(f"Equity synced: ${state.equity:,.2f}")


def check_open_position(
    order_mgr: MT5OrderManager,
    state: LiveState,
    params: MLScalperParams,
    current_bar_count: int,
) -> bool:
    """
    Check if our open position is still alive in MT5.
    MT5 handles TP/SL hits automatically — we just need to detect close.
    Returns True if position still open.
    """
    if state.open_ticket is None:
        return False

    positions = order_mgr.get_positions(symbol=SYMBOL)
    tickets   = {p['ticket'] for p in positions if p.get('magic') == MAGIC}

    if state.open_ticket not in tickets:
        # Position closed (TP or SL hit by MT5)
        log.info(f"Position {state.open_ticket} closed by MT5 (TP/SL/manual)")
        state.trade_log.append({
            'ticket':   state.open_ticket,
            'dir':      state.open_dir,
            'entry':    state.open_entry,
            'close_ts': datetime.now(timezone.utc).isoformat(),
            'reason':   'MT5_CLOSE',
        })
        state.open_ticket = None
        state.open_dir    = None
        return False

    # Time-based exit: max_hold_bars
    state.open_bar_count += 1
    if state.open_bar_count >= params.max_hold_bars:
        log.info(f"Time exit: {state.open_ticket} held {state.open_bar_count} bars")
        order_mgr.close_position(state.open_ticket)
        state.open_ticket = None
        state.open_dir    = None
        return False

    return True


def open_trade(
    order_mgr: MT5OrderManager,
    state: LiveState,
    direction: str,
    params: MLScalperParams,
    win_prob: float,
    paper: bool,
) -> None:
    """Calculate lot size and send order to MT5."""
    import MetaTrader5 as mt5

    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        log.error("Could not get tick for XAUUSD")
        return

    if direction == 'LONG':
        entry = tick.ask
        sl    = round(entry - params.sl_pts, 2)
        tp    = round(entry + params.tp_pts, 2)
    else:
        entry = tick.bid
        sl    = round(entry + params.sl_pts, 2)
        tp    = round(entry - params.tp_pts, 2)

    # Lot sizing
    if params.kelly_fraction > 0:
        lot = kelly_lot(
            state.equity, win_prob,
            params.tp_pts, params.sl_pts,
            params.kelly_fraction, params.kelly_max_lot,
            params.kelly_max_risk_pct,
        )
    else:
        lot = params.lot_base

    log.info(
        f"SIGNAL {direction} | entry={entry} sl={sl} tp={tp} "
        f"lot={lot} proba={win_prob:.3f} equity=${state.equity:,.0f}"
    )

    if paper:
        log.info("[PAPER] Order NOT sent")
        state.open_ticket   = -1    # dummy
        state.open_dir      = direction
        state.open_entry    = entry
        state.open_sl       = sl
        state.open_tp       = tp
        state.open_lot      = lot
        state.open_bar_count = 0
        return

    ticket = order_mgr.open_order(
        symbol=SYMBOL,
        direction=direction,
        volume=lot,
        stop_loss=sl,
        take_profit=tp,
        magic=MAGIC,
    )

    if ticket:
        state.open_ticket   = ticket
        state.open_dir      = direction
        state.open_entry    = entry
        state.open_sl       = sl
        state.open_tp       = tp
        state.open_lot      = lot
        state.open_bar_count = 0
        log.info(f"Order placed: ticket={ticket}")
    else:
        log.error("Order placement failed")


# ─── Main Loop ───────────────────────────────────────────────────────────────

def run_live(args: argparse.Namespace) -> None:
    log.info("=" * 60)
    log.info("  ML Gold Scalper — Live Trader")
    log.info("=" * 60)
    log.info(f"  Symbol    : {SYMBOL}")
    log.info(f"  Threshold : {args.threshold}")
    log.info(f"  Kelly     : {args.kelly} (max_lot={args.kelly_max}, max_risk={args.kelly_max_risk}%)")
    log.info(f"  Paper     : {args.paper}")
    log.info("")

    params = MLScalperParams(
        tp_pts             = args.tp,
        sl_pts             = args.sl,
        spread_pts         = 0.4,
        lot_base           = args.lot,
        max_hold_bars      = args.max_bars,
        threshold          = args.threshold,
        session_filter     = True,
        kelly_fraction     = args.kelly,
        kelly_max_lot      = args.kelly_max,
        kelly_max_risk_pct = args.kelly_max_risk,
    )

    # Load models
    log.info("Loading models...")
    try:
        long_bundle, short_bundle = load_both_models()
        log.info("Loaded lgbm_gold_long + lgbm_gold_short")
    except FileNotFoundError:
        long_bundle  = load_model('lgbm_gold')
        short_bundle = {'model': None, 'features': long_bundle['features']}
        log.info("Loaded lgbm_gold (long-only fallback)")

    # Connect to MT5
    connector = MT5Connector()
    order_mgr = MT5OrderManager()

    if not args.paper:
        if not connector.connect():
            log.error("MT5 connection failed — exiting")
            return
        log.info("MT5 connected")

    state = LiveState()
    bar_count = 0
    log.info("Entering main loop... (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(POLL_SECONDS)

            # ── Fetch latest M1 bars ──────────────────────────────────────
            if not args.paper:
                m1 = connector.fetch_data(SYMBOL, '1m', bars=WARMUP_BARS)
                if m1.empty or len(m1) < 200:
                    log.warning("Insufficient data, skipping")
                    continue
                sync_equity(connector, state)
            else:
                # Paper mode: dummy check, assume data is available
                log.debug("[PAPER] Skipping live data fetch")
                continue

            # ── New bar check ─────────────────────────────────────────────
            latest_bar = m1.index[-2]   # last completed bar
            if latest_bar == state.last_bar_time:
                continue   # same bar, no action

            state.last_bar_time = latest_bar
            bar_count += 1

            # ── Manage open position ──────────────────────────────────────
            if state.open_ticket is not None:
                still_open = check_open_position(order_mgr, state, params, bar_count)
                if still_open:
                    continue   # wait for current trade to close

            # ── Generate signal ───────────────────────────────────────────
            want_long, want_short, long_p, short_p = generate_signal(
                m1, long_bundle, short_bundle, params
            )

            if want_long:
                open_trade(order_mgr, state, 'LONG',  params, long_p,  args.paper)
            elif want_short:
                open_trade(order_mgr, state, 'SHORT', params, short_p, args.paper)

    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        if not args.paper:
            connector.disconnect()
        log.info(f"Session ended. Total bars processed: {bar_count}")
        log.info(f"Final equity: ${state.equity:,.2f}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='ML Gold Scalper — Live Trader')
    parser.add_argument('--tp',             type=float, default=6.0,   help='TP pts (default 6.0)')
    parser.add_argument('--sl',             type=float, default=3.0,   help='SL pts (default 3.0)')
    parser.add_argument('--max_bars',       type=int,   default=120,   help='Max hold bars (default 120)')
    parser.add_argument('--threshold',      type=float, default=0.75,  help='Signal threshold (default 0.75)')
    parser.add_argument('--lot',            type=float, default=0.01,  help='Base lot size (default 0.01)')
    parser.add_argument('--kelly',          type=float, default=0.25,  help='Kelly fraction (default 0.25)')
    parser.add_argument('--kelly_max',      type=float, default=1.0,   help='Kelly hard max lot (default 1.0)')
    parser.add_argument('--kelly_max_risk', type=float, default=1.5,   help='Max %% equity/trade (default 1.5)')
    parser.add_argument('--paper',          action='store_true',       help='Paper trading — no real orders')
    args = parser.parse_args()
    run_live(args)


if __name__ == '__main__':
    main()
