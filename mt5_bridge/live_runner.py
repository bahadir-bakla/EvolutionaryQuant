# MT5 Live Runner (Gold & Silver Edition)
# Runs strategies in a loop, fetching live data and executing trades.

import time
import sys
import logging
import joblib
from pathlib import Path
import threading
import time
import threading
import time
from scipy.signal import hilbert
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from mt5_bridge.connector import MT5Connector
from mt5_bridge.order_manager import MT5OrderManager
from nq_core.super_hybrid import SuperHybridStrategy, SuperHybridConfig
from nq_core.silver_edge import SilverEdgeStrategy, SilverEdgeConfig
from nq_core.gold_price_action import GoldPriceActionStrategy
from nq_core.reversion_scalper import ReversionScalper, ReversionConfig
from nq_core.optimized_strategy import add_optimized_indicators
from nq_core.nq_algorithmic_suite import NQAlgorithmicSuite
from tui.web_news import WebNewsAggregator

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mt5_live.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MT5_Live_Runner")

# --- CONFIGURATION ---
SYMBOLS = {
    'GOLD': 'XAUUSD',   # Gold 1H
    'SILVER': 'XAGUSD', # Silver 1H
    'GOLD_15M': 'XAUUSD', # Gold 15m Sniper
    'GOLD_REV': 'XAUUSD',  # Gold 1m Reversion
    'NQ_SUITE': 'USTEC'    # Nasdaq 5m Master HMM Suite
}

TIMEFRAMES = {
    'GOLD': '1h',
    'SILVER': '1h',
    'GOLD_15M': '15m',
    'GOLD_REV': '1m',
    'NQ_SUITE': '5m'
}

# Magic Numbers (ID for each strategy)
MAGIC_GOLD = 1001
MAGIC_SILVER = 1002
MAGIC_GOLD_15M = 1003
MAGIC_GOLD_REV = 1004
MAGIC_NQ_SUITE = 1005

# Strategy Configs
gold_config = SuperHybridConfig(
    atr_stop_mult=1.5,
    atr_tp_mult_1=1.0,   # Was 2.0 — Gold 1H ATR ~20pt, old TP ~40pt unreachable
    atr_tp_mult_2=2.0    # Was 4.0
)
gold_strategy = SuperHybridStrategy(gold_config)

# Silver Edge V3 (5-Layer: HTF + Gold Correlation + Session + Kalman+ + Dynamic Risk)
silver_strategy = SilverEdgeStrategy(SilverEdgeConfig())

gold_pa_strategy = GoldPriceActionStrategy() # Sniper V2

# Gold Reversion Config
gold_rev_config = ReversionConfig(
    streak_threshold=5, 
    retrace_pct=0.5,
    use_smart_targets=True,
    use_rsi_filter=True # Turbo Mode
)
gold_rev_strategy = ReversionScalper(gold_rev_config)

# Nasdaq Algorithmic Suite (MAMA, Sine, Breakdown)
nq_suite_strategy = NQAlgorithmicSuite()

# --- AI META LABELER ---
META_LABELER = None
meta_path = Path(__file__).parent.parent / 'models' / 'gold_meta_labeler_v1.pkl'
if meta_path.exists():
    try:
        META_LABELER = joblib.load(meta_path)
        logger.info(f"Loaded AI Meta-Labeler from {meta_path.name}")
    except Exception as e:
        logger.error(f"Failed to load Meta-Labeler: {e}")

# --- HMM REGIME DIRECTOR ---
REGIME_MODEL = None
regime_path = Path(__file__).parent.parent / 'models' / 'nq_hilbert_hmm.pkl'
if regime_path.exists():
    try:
        REGIME_MODEL = joblib.load(regime_path)
        logger.info(f"Loaded Hilbert-HMM Regime Director")
    except Exception as e:
        logger.error(f"Failed to load HMM Regime Director: {e}")

global_current_regime = "UNKNOWN"
global_daily_bias = "UNKNOWN"
global_news_sentiment = "UNKNOWN"
global_news_is_critical = False

def update_macro_news():
    """Background thread to update Global News Sentiment every 15 mins."""
    global global_news_sentiment, global_news_is_critical
    aggregator = WebNewsAggregator()
    while True:
        try:
            status, color = aggregator.get_market_sentiment()
            global_news_sentiment = status
            global_news_is_critical = "CRITICAL" in status or "ELEVATED" in status
            logger.info(f"📰 [WEB NEWS AI] Global Sentiment is: {global_news_sentiment}")
        except Exception as e:
            logger.error(f"Failed to fetch News Sentiment: {e}")
        time.sleep(900) # 15 minutes

def update_macro_regime():
    """Background thread to update the HMM Macro Regime every hour."""
    global global_current_regime, global_daily_bias
    while True:
        if REGIME_MODEL is not None:
            try:
                import yfinance as yf
                import numpy as np
                df = yf.Ticker("NQ=F").history(period="60d", interval="1h")
                if not df.empty:
                    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                    # Engineer HMM Features
                    df['returns'] = np.log(df['close'] / df['close'].shift(1))
                    df['tr1'] = df['high'] - df['low']
                    df['tr2'] = abs(df['high'] - df['close'].shift(1))
                    df['tr3'] = abs(df['low'] - df['close'].shift(1))
                    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
                    df['atr'] = df['tr'].rolling(14).mean()
                    
                    sma_len = 20
                    df['sma'] = df['close'].rolling(sma_len).mean()
                    df['detrended'] = df['close'] - df['sma']
                    df['detrended'].fillna(0, inplace=True)
                    analytic_signal = hilbert(df['detrended'].values)
                    df['hilbert_phase'] = np.angle(analytic_signal)
                    df['hilbert_sine'] = np.sin(df['hilbert_phase'])
                    df['hilbert_leadsine'] = np.sin(df['hilbert_phase'] + np.pi/4)
                    
                    df.dropna(inplace=True)
                    
                    features = REGIME_MODEL['features']
                    X = df[features].copy()
                    X_scaled = REGIME_MODEL['scaler'].transform(X)
                    
                    hmm = REGIME_MODEL['model']
                    latest_state = hmm.predict(X_scaled)[-1]
                    
                    global_current_regime = REGIME_MODEL['state_mapping'][latest_state]
                    logger.info(f"🌍 [MACRO REGIME UPDATED] NQ is currently in: {global_current_regime}")
                    
                df_1d = yf.Ticker("NQ=F").history(period="6mo", interval="1d")
                if not df_1d.empty:
                    df_1d['ema_20'] = df_1d['Close'].ewm(span=20, adjust=False).mean()
                    df_1d['ema_50'] = df_1d['Close'].ewm(span=50, adjust=False).mean()
                    global_daily_bias = 'BULL' if df_1d['ema_20'].iloc[-1] > df_1d['ema_50'].iloc[-1] else 'BEAR'
                    logger.info(f"📅 [DAILY BIAS UPDATED] NQ Bias is: {global_daily_bias}")
                    
            except Exception as e:
                logger.error(f"Failed to update HMM Regime: {e}")
                
        # Update once per hour (3600 seconds)
        time.sleep(3600)

# Start the regime director daemon thread
regime_thread = threading.Thread(target=update_macro_regime, daemon=True)
regime_thread.start()

# Start the news scraper daemon thread
news_thread = threading.Thread(target=update_macro_news, daemon=True)
news_thread.start()

last_processed_bar = {
    'GOLD': None,
    'SILVER': None,
    'GOLD_15M': None,
    'GOLD_REV': None,
    'NQ_SUITE': None
}

def run_strategy(connector, order_manager, asset_name):
    """Execute strategy for a single asset"""
    symbol = SYMBOLS[asset_name]
    tf = TIMEFRAMES[asset_name]
    magic = 0
    strategy = None

    if asset_name == 'GOLD':
        magic = MAGIC_GOLD
        strategy = gold_strategy
    elif asset_name == 'SILVER':
        magic = MAGIC_SILVER
        strategy = silver_strategy
    elif asset_name == 'GOLD_15M':
        magic = MAGIC_GOLD_15M
        strategy = gold_pa_strategy
    elif asset_name == 'GOLD_REV':
        magic = MAGIC_GOLD_REV
        strategy = gold_rev_strategy
    elif asset_name == 'NQ_SUITE':
        magic = MAGIC_NQ_SUITE
        strategy = nq_suite_strategy
        
    logger.info(f"Analyzing {asset_name} ({symbol}) on {tf}...")
    
    # 1. Fetch Data
    # Fetch 500 bars to ensure indicators have enough history
    df = connector.fetch_data(symbol, tf, 500)
    
    if df.empty:
        logger.warning(f"No data for {symbol}")
        return

    # Check for new bar (Don't trade same bar twice)
    last_time = df.index[-1]
    last_processed = last_processed_bar.get(asset_name) # Safe get
    if last_processed == last_time:
        logger.info(f"  Already processed bar {last_time}. Waiting...")
        return
        
    last_processed_bar[asset_name] = last_time
    
    # 2. Indicators
    if asset_name == 'GOLD_15M':
         df['tr'] = pd.concat([
            df['high'] - df['low'], 
            (df['high'] - df['close'].shift()).abs(), 
            (df['low'] - df['close'].shift()).abs()
        ], axis=1).max(axis=1)
         df['atr'] = df['tr'].rolling(14).mean()
         df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    elif asset_name == 'GOLD_REV':
         pass
    elif asset_name == 'NQ_SUITE':
         df = strategy.prepare_data(df)
    else:
        df = add_optimized_indicators(df)

    # 3. Strategy Signal
    current_idx = len(df) - 2 # Trade Closed Candle
    
    # Check current position for Reversion State Logic
    positions = order_manager.get_positions(symbol)
    pos_info = None
    if asset_name == 'GOLD_REV':
        for p in positions:
            if p['magic'] == magic:
                pos_info = {'direction': 'LONG' if p['type']==0 else 'SHORT', 
                            'tp': p['tp'], 'sl': p['sl']}
                break
                
    if asset_name == 'GOLD_REV':
        # Calculate Pivots (15m)
        df_15m = connector.fetch_data(symbol, '15m', 200)
        pivots = []
        if not df_15m.empty:
             for i in range(2, len(df_15m)-2):
                 if df_15m['low'].iloc[i] < df_15m['low'].iloc[i-1] and \
                    df_15m['low'].iloc[i] < df_15m['low'].iloc[i+1]:
                     pivots.append(df_15m['low'].iloc[i])
            
        signal = strategy.evaluate(df, current_idx, pos_info, key_levels=pivots)
    elif asset_name == 'NQ_SUITE':
        signal, active_bot = strategy.evaluate(df, current_idx, global_current_regime, global_daily_bias)
        if signal.direction != 'NEUTRAL':
            signal.reason = f"Auth: HMM Regime -> {global_current_regime} | Bot: {active_bot} | Bias: {global_daily_bias}"
    elif asset_name == 'SILVER':
        # Silver Edge V3: Feed Gold 1H + Silver 4H for multi-layer intelligence
        gold_df = connector.fetch_data('XAUUSD', '1h', 100)
        htf_df = connector.fetch_data(symbol, '4h', 100)
        signal = strategy.evaluate(df, current_idx, htf_df=htf_df, gold_df=gold_df)
    else:
        signal = strategy.evaluate(df, current_idx)
    
    # --- AI META-LABELER INTERCEPTION ---
    if asset_name == 'GOLD' and signal.direction in ['LONG', 'SHORT'] and META_LABELER is not None:
        try:
            row = df.iloc[current_idx]
            feat_atr = getattr(row, 'atr', row['high'] - row['low'])
            feat_rsi = getattr(row, 'rsi_14', 50.0)
            feat_kalman_vel = getattr(row, 'kalman_vel', 0.0)
            ema_200 = df['close'].ewm(span=200).mean().iloc[current_idx]
            feat_ema_dist = (row['close'] - ema_200) / ema_200
            feat_hour = df.index[current_idx].hour
            feat_day = df.index[current_idx].weekday()
            
            X_df = pd.DataFrame([{
                'f_atr': feat_atr,
                'f_rsi': feat_rsi,
                'f_kalman_vel': feat_kalman_vel,
                'f_ema_dist': feat_ema_dist,
                'f_hour': feat_hour,
                'f_day': feat_day
            }])
            
            # Predict Probability
            model = META_LABELER['model']
            threshold = META_LABELER['threshold']
            proba = model.predict_proba(X_df)[0][1] # Win probability
            
            if proba < threshold:
                logger.warning(f"  [AI FILTER] Trade Rejected! Win Prob {proba*100:.1f}% < {threshold*100:.1f}%")
                # Intercept the trade. If we are in a position, just exit it (since primary wanted to reverse).
                # If we are flat, remain flat.
                signal.direction = 'EXIT'
            else:
                logger.info(f"  [AI APPROVED] Trade Accepted! Win Prob {proba*100:.1f}% >= {threshold*100:.1f}%")
                
        except Exception as e:
            logger.error(f"Failed to run Meta-Labeler inference: {e}")
            
    logger.info(f"  Signal: {signal.direction} | Reason: {getattr(signal, 'reason', '')}")
    
    # Execute Execution Logic
    # Check current position in MT5
    current_pos_dir = 'NEUTRAL'
    current_ticket = None
    
    for p in positions:
        if p['magic'] == magic:
            current_pos_dir = 'LONG' if p['type'] == 0 else 'SHORT'
            current_ticket = p['ticket']
            break
            
    # Logic:
    # If Signal != Current -> Close Current, Open New
    
    if signal.direction == 'NEUTRAL':
        pass
        
    elif signal.direction == 'EXIT':
         if current_pos_dir != 'NEUTRAL':
            order_manager.close_position(current_ticket)
            
    elif signal.direction != current_pos_dir:
        logger.info(f"  Action: {current_pos_dir} -> {signal.direction}")
        
        # Close existing
        if current_pos_dir != 'NEUTRAL':
            order_manager.close_position(current_ticket)
            
        # Open new
        if signal.direction in ['LONG', 'SHORT']:
            lots = 0.10
            
            # Dynamic sizing via confidence for Silver Edge V3
            if asset_name == 'SILVER' and getattr(signal, 'confidence', 0) >= 0.7:
                lots = 0.20  # High confidence = double size
                logger.info(f"  HIGH CONFIDENCE ({signal.confidence:.2f}): Size {lots}")
                
            order_manager.open_order(
                symbol=symbol,
                direction=signal.direction,
                volume=lots,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit if hasattr(signal, 'take_profit') else signal.take_profit_1, 
                magic=magic
            )
            
    elif signal.direction == current_pos_dir:
        logger.info(f"  Holding {current_pos_dir}. No Change.")


def main_loop():
    connector = MT5Connector()
    manager = MT5OrderManager()
    
    if not connector.connect():
        logger.error("Failed to connect to MT5")
        return

    logger.info("=== MT5 LIVE RUNNER STARTED ===")
    logger.info(f"Trading: {list(SYMBOLS.keys())}")
    
    try:
        while True:
            if not connector.connected:
                connector.connect()
                
            run_strategy(connector, manager, 'GOLD')
            run_strategy(connector, manager, 'SILVER')
            run_strategy(connector, manager, 'GOLD_15M')
            run_strategy(connector, manager, 'GOLD_REV')
            run_strategy(connector, manager, 'NQ_SUITE')
            
            logger.info("Sleeping 60s...")
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Stopping...")
        connector.disconnect()

if __name__ == "__main__":
    main_loop()
