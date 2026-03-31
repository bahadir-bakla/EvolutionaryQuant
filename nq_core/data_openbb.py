# NQ Quant Bot - OpenBB Data Integration
# Fetch advanced data using OpenBB Platform

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Standard imports
try:
    from openbb import obb
    OPENBB_AVAILABLE = True
except ImportError:
    OPENBB_AVAILABLE = False
    print("OpenBB not installed or not found.")

def check_openbb():
    """Check if OpenBB is ready"""
    if not OPENBB_AVAILABLE:
        print("[-] OpenBB SDK is missing. Please install it.")
        return False
    
    # helper to suppress login/telemetry noise
    # users often don't have a PAT setup
    return True

def fetch_historical_intraday(
    symbol: str, 
    start_date: str, 
    end_date: str, 
    interval: str = "5m", 
    provider: str = "yfinance"
):
    """
    Fetch historical data using OpenBB
    
    Providers: yfinance, polygon, fmp, alpha_vantage
    """
    if not check_openbb():
        return None
    
    print(f"[*] OpenBB Fetching: {symbol} ({interval}) from {start_date} to {end_date} via {provider}")
    
    try:
        # OpenBB v4 syntax
        df = obb.equity.price.historical(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            provider=provider
        ).to_df()
        
        # Clean up
        df.columns = df.columns.str.lower()
        print(f"[+] Fetched {len(df)} rows.")
        return df
        
    except Exception as e:
        print(f"[-] Error fetching data: {e}")
        return None

def fetch_futures_historical(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "5m"
):
    """
    Fetch futures data
    OpenBB might treat futures differently or require specific providers
    """
    return fetch_historical_intraday(symbol, start_date, end_date, interval, provider="yfinance")

if __name__ == "__main__":
    # Test
    if OPENBB_AVAILABLE:
        # Test 1: Recent Intraday (Should work with yfinance)
        print("\nTest 1: Recent Intraday NVDA (5d)...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        nvda = fetch_historical_intraday(
            "NVDA", 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d'), 
            "15m"
        )
        if nvda is not None:
            print(nvda.head())
        else:
            print("Failed to fetch recent data.")

        # Test 2: Historical Daily (2021)
        print("\nTest 2: Historical Daily NQ=F (2021)...")
        nq = fetch_historical_intraday(
            "NQ=F", 
            "2021-01-01", 
            "2021-01-10", 
            "1d"
        )
        if nq is not None:
            print(nq.head())
        else:
            print("Failed to fetch historical daily data.")
