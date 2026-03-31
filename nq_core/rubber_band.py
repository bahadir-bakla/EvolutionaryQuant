import pandas as pd
from dataclasses import dataclass

@dataclass
class RubberBandSignal:
    direction: str # LONG, SHORT, NEUTRAL
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str

class RubberBandStrategy:
    def __init__(self, rsi_length=3, ema_target=20):
        self.rsi_length = rsi_length # Hyper sensitive
        self.ema_target = ema_target
        
    def evaluate(self, df: pd.DataFrame, idx: int) -> RubberBandSignal:
        signal = RubberBandSignal('NEUTRAL', 0.0, 0.0, 0.0, "")
        
        # Need enough data
        if idx < self.ema_target + 5:
            return signal
            
        # 1. Indicators
        lookback = 100
        start = max(0, idx - lookback)
        slice_df = df.iloc[start : idx+1].copy()
        
        # EMA Target
        slice_df['ema'] = slice_df['close'].ewm(span=self.ema_target, adjust=False).mean()
        
        # RSI 3
        delta = slice_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_length).mean()
        rs = gain / loss
        slice_df['rsi'] = 100 - (100 / (1 + rs))
        
        curr = slice_df.iloc[-1]
        prev = slice_df.iloc[-2]
        
        # 2. Logic (Mean Reversion)
        
        # LONG: RSI was < 10, now > 10 (Hook up)
        if prev['rsi'] < 10 and curr['rsi'] > 10:
             # Filter: Ensure we have room to target (Price < EMA)
             if curr['close'] < curr['ema']:
                 signal.direction = 'LONG'
                 signal.stop_loss = slice_df.iloc[-3:]['low'].min() # Low of last 3 candles
                 signal.take_profit = curr['ema']
                 signal.reason = f"Fear Snap (RSI {int(prev['rsi'])}->{int(curr['rsi'])})"
                 return signal
                 
        # SHORT: RSI was > 90, now < 90 (Hook down)
        if prev['rsi'] > 90 and curr['rsi'] < 90:
             # Filter: Ensure we have room to target (Price > EMA)
             if curr['close'] > curr['ema']:
                 signal.direction = 'SHORT'
                 signal.stop_loss = slice_df.iloc[-3:]['high'].max() # High of last 3 candles
                 signal.take_profit = curr['ema']
                 signal.reason = f"Greed Snap (RSI {int(prev['rsi'])}->{int(curr['rsi'])})"
                 return signal
                 
        return signal
