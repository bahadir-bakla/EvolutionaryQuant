# MT5 Connector
# Handles connection to MetaTrader 5 Terminal and Data Fetching
# Requirements: pip install MetaTrader5

import MetaTrader5 as mt5
import pandas as pd
import sys
import logging
from datetime import datetime
import pytz

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5Connector:
    """
    Bridge between Python and local MT5 Terminal.
    """
    def __init__(self):
        self.connected = False
        
    def connect(self, login: int = None, password: str = None, server: str = None):
        """Initialize connection to MT5 terminal"""
        # Try to initialize with credentials if provided
        if login and password and server:
            logger.info(f"Connecting to {server} with account {login}...")
            authorized = mt5.initialize(login=login, password=password, server=server)
        else:
            # Try to connect to existing open terminal
            logger.info("Connecting to open terminal...")
            authorized = mt5.initialize()
            
        if not authorized:
            logger.error(f"MT5 initialization failed. Error: {mt5.last_error()}")
            mt5.shutdown()
            self.connected = False
            return False
        
        # Check if terminal is actually connected to a server
        logger.info(f"MT5 Initialized. Terminal Info: {mt5.terminal_info()}")
        logger.info(f"MT5 Version: {mt5.version()}")
        self.connected = True
        return True
        
    def disconnect(self):
        mt5.shutdown()
        self.connected = False
        logger.info("MT5 Disconnected")

    def get_symbol_info(self, symbol: str):
        """Get symbol specification"""
        if not self.connected:
            if not self.connect(): return None
            
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol not found: {symbol}")
            return None
        return info

    def fetch_data(self, symbol: str, timeframe_str: str, bars: int = 1000) -> pd.DataFrame:
        """
        Fetch OHLCV data from MT5.
        
        Args:
            symbol: e.g. "XAUUSD", "NQ100", "US500" (Depends on broker!)
            timeframe_str: "1m", "5m", "15m", "1h", "4h", "d1"
        """
        if not self.connected:
            if not self.connect(): return pd.DataFrame()
            
        # Map timeframe string to MT5 constant
        tf_map = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "30m": mt5.TIMEFRAME_M30,
            "1h": mt5.TIMEFRAME_H1,
            "4h": mt5.TIMEFRAME_H4,
            "d1": mt5.TIMEFRAME_D1
        }
        
        mt5_tf = tf_map.get(timeframe_str.lower())
        if not mt5_tf:
            logger.error(f"Invalid timeframe: {timeframe_str}")
            return pd.DataFrame()
            
        # Ensure symbol is selected in Market Watch
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return pd.DataFrame()
            
        # Fetch rates
        # copy_rates_from_pos(symbol, timeframe, start_pos, count)
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, bars)
        
        if rates is None or len(rates) == 0:
            logger.error(f"No data received for {symbol}")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        
        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Set index
        df.set_index('time', inplace=True)
        
        # Rename columns to match our strategy format (lowercase)
        # MT5 columns: time, open, high, low, close, tick_volume, spread, real_volume
        df.rename(columns={
            'tick_volume': 'volume', 
            # some brokers use real_volume for stocks, tick for fx
            # we'll use tick_volume mostly for cfds
        }, inplace=True)
        
        # Drop unused
        if 'spread' in df.columns: df.drop(columns=['spread'], inplace=True)
        if 'real_volume' in df.columns: df.drop(columns=['real_volume'], inplace=True)
        
        # Ensure float columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(float)
                
        return df

    def get_market_book(self, symbol: str):
        # Advanced: For scalping if needed
        pass
