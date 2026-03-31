# Gold 15m Price Action Strategy (Liquidity & Structure)

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PADelegationSignal:
    direction: str
    confidence: float
    stop_loss: float
    take_profit: float
    reason: str

class GoldPriceActionStrategy:
    """
    15m Strategy for Gold (XAUUSD) focused on:
    1. Liquidity Sweeps (Fakeouts below Swing Lows / above Swing Highs)
    2. 3rd Touch Rejections (Trendline/Zone holds)
    3. FVG Confluence
    """
    
    def __init__(self):
        self.swing_lookback = 20
        
    def find_4h_order_blocks(self, df: pd.DataFrame, current_idx: int) -> List[dict]:
        """
        Approximation of 4H Order Blocks using 15m data.
        1. Resample last ~5 days of data to 4H.
        2. Identify OBs.
        """
        # Take enough data to form 4H candles
        lookback_bars = 4 * 4 * 10 # 4 bars/hr * 4 hrs * 10 candles
        if current_idx < lookback_bars:
            return []
            
        slice_15m = df.iloc[current_idx-lookback_bars : current_idx].copy()
        
        # Resample to 4H
        # Note: Index must be datetime
        df_4h = slice_15m.resample('4h').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
        }).dropna()
        
        obs = []
        # Simple OB Detection on 4H
        # Bullish OB: The last Red candle before a Green candle that broke structure/had momentum
        if len(df_4h) < 3: return []
        
        for i in range(len(df_4h) - 5, len(df_4h) - 1):
            curr = df_4h.iloc[i]
            next_c = df_4h.iloc[i+1]
            
            # Simple definition: Down candle followed by Up candle that engulfed
            is_red = curr['close'] < curr['open']
            is_green_next = next_c['close'] > next_c['open']
            englufing = next_c['close'] > curr['high']
            
            if is_red and is_green_next and englufing:
                obs.append({
                    'type': 'BULLISH',
                    'top': curr['high'],
                    'bottom': curr['low'],
                    'time': df_4h.index[i]
                })
        
        return obs

    def evaluate(self, df: pd.DataFrame, idx: int) -> PADelegationSignal:
        row = df.iloc[idx]
        
        signal = PADelegationSignal('NEUTRAL', 0.0, 0.0, 0.0, "")
        atr = row.get('atr', 2.0)
        
        # 1. Check if we are in a 4H Order Block (Support)
        obs_4h = self.find_4h_order_blocks(df, idx)
        in_bullish_ob = False
        
        for ob in obs_4h:
            if ob['type'] == 'BULLISH':
                # Price dipped into OB
                if row['low'] <= ob['top'] and row['low'] >= ob['bottom']:
                    in_bullish_ob = True
                    break
        
        # 2. 15m Triggers
        # A. FVG (Simple: 3 bar pattern)
        is_fvg_bullish = False
        if idx > 2:
            # Candle i-2 High < Candle i Low (Gap Up? No, Gap Down filled?)
            # ICT Bullish FVG: Candle 1 High < Candle 3 Low
            # Here we want to see if we just FILLED a FVG?
            # Or if we just formed a reversal?
            pass
        window = 20
        start_idx = max(0, idx - window)
        search_slice = df.iloc[start_idx:idx]
        
        swing_low = search_slice['low'].min()
        swing_high = search_slice['high'].max()
        
        bullish_sweep = (row['low'] < swing_low) and (row['close'] > swing_low)
        bearish_sweep = (row['high'] > swing_high) and (row['close'] < swing_high)
        
        # Placeholder for EMA, assuming it's calculated elsewhere or passed in `df`
        # For now, let's use a simple moving average as a stand-in if 'ema' column doesn't exist
        ema = row.get('ema', df['close'].iloc[max(0, idx-10):idx].mean()) # Example: 10-period SMA if no EMA
        
        # C. Oversold (RSI/Stoch)
        rsi = row.get('rsi', 50)
        is_oversold = rsi < 35
        
        # LOGIC: Sniper V2 (Trend Following Liquidity Sweeps)
        # We only take trades WITH the major trend (EMA 200)
        # 4. Filter: Trend (EMA 200) - ESSENTIAL for Profitability
        ema_period = 200
        # Calculate EMA dynamically if not in df
        # Note: This is slow in a loop but accurate for the specific index
        if 'ema_200' in df.columns:
            ema = df.iloc[idx]['ema_200']
        else:
             # Fallback: Simple moving average of last 200 if valid
             if idx >= ema_period: # Use >= for correct calculation at idx=ema_period
                 ema = df['close'].ewm(span=ema_period, adjust=False).mean().iloc[idx] # adjust=False for traditional EMA
             else:
                 ema = df['close'].iloc[idx] # Not enough data for EMA, use current close as fallback
        
        # This filters out "catching falling knives"
        
        if bullish_sweep:
            if row['close'] > ema: # Only Buy Dips in Uptrend
                if row['close'] > row['open']: 
                    signal.direction = 'LONG'
                    signal.confidence = 0.9
                    signal.stop_loss = row['low'] - (atr * 1.5)
                    signal.take_profit = swing_high
                    signal.reason = "Trend Follow: Liquidity Sweep + EMA Support"
                    return signal

        elif bearish_sweep:
            if row['close'] < ema: # Only Sell Rallies in Downtrend
                if row['close'] < row['open']:
                    signal.direction = 'SHORT'
                    signal.confidence = 0.9
                    signal.stop_loss = row['high'] + (atr * 1.5)
                    signal.take_profit = swing_low
                    signal.reason = "Trend Follow: Liquidity Sweep + EMA Resistance"
                    return signal
                     
        return signal
