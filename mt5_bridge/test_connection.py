# Test MT5 Connection
# Run this to verify you can talk to the terminal

from connector import MT5Connector
import time

def test():
    print("Initializing Connector...")
    connector = MT5Connector()
    
    # Credentials from user
    login = 10009695146
    password = "0kK_IyYb"
    server = "MetaQuotes-Demo"
    
    if connector.connect(login=login, password=password, server=server):
        print("[OK] SUCCESS: Connected to MT5")
        
        # Test Data Fetch (Gold)
        # Note: Symbol names depend on broker (e.g. "XAUUSD", "Gold", "GC=F" is Yahoo not MT5)
        # Common MT5 symbols: "XAUUSD", "US500", "USTEC", "Nasdaq"
        symbol = "XAUUSD" # Try generic Gold
        print(f"Attempting to fetch {symbol} data...")
        
        df = connector.fetch_data(symbol, "1h", 10)
        
        if not df.empty:
            print(f"[OK] Data Received:\n{df.tail()}")
        else:
            print(f"[ERROR] Failed to fetch {symbol}. Check Market Watch or Symbol Name.")
            print("Trying 'Gold'...")
            df = connector.fetch_data("Gold", "1h", 10)
            if not df.empty:
                print(f"[OK] Data Received for 'Gold':\n{df.tail()}")
            
        connector.disconnect()
    else:
        print("[ERROR] FAILED: Could not connect to MT5. Is it running?")

if __name__ == "__main__":
    test()
