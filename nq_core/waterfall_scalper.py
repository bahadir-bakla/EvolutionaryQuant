import pandas as pd
from dataclasses import dataclass

@dataclass
class WaterfallSignal:
    direction: str # SHORT, NEUTRAL
    stop_loss: float
    take_profit: float
    reason: str

class WaterfallScalper:
    def __init__(self, multiplier=3.0, tp_points=20.0, sl_points=40.0, volume_mult=0.0, use_ema_filter=False, use_vwap_filter=False):
        self.multiplier = multiplier
        self.tp_points = tp_points
        self.sl_points = sl_points
        self.volume_mult = volume_mult
        self.use_ema_filter = use_ema_filter
        self.use_vwap_filter = use_vwap_filter
        
    def evaluate(self, df: pd.DataFrame, idx: int) -> WaterfallSignal:
        signal = WaterfallSignal('NEUTRAL', 0.0, 0.0, "")
        
        # Need history
        lookback = 50
        if idx < lookback:
            return signal
            
        # Get candles
        c = df.iloc[idx]
        a = df.iloc[idx-1]
        b = df.iloc[idx-2]
        
        # Calculate Bodies
        body_c = abs(c['open'] - c['close'])
        body_a = abs(a['open'] - a['close'])
        body_b = abs(b['open'] - b['close'])
        
        # Directions
        is_c_red = c['close'] < c['open']
        is_a_red = a['close'] < a['open']
        
        # Logic
        threshold_size = body_b * self.multiplier
        if body_b == 0: threshold_size = 1.0 
        
        if is_a_red and body_a >= threshold_size:
            # VOLUME FILTER
            vol_ok = True
            if self.volume_mult > 0:
                vol_slice = df['volume'].iloc[idx-1-20 : idx-1]
                avg_vol = vol_slice.mean()
                if a['volume'] < (avg_vol * self.volume_mult):
                    vol_ok = False
                    
            # EMA FILTER
            ema_ok = True
            if self.use_ema_filter:
                ema_slice = df['close'].iloc[max(0, idx-50):idx]
                ema_val = ema_slice.mean() 
                if c['close'] > ema_val: 
                    ema_ok = False
                    
            # VWAP FILTER
            vwap_ok = True
            if self.use_vwap_filter:
                # Expecting 'vwap' column in DF. 
                # If not present, we can't filter (or assume True).
                if 'vwap' in df.columns:
                    vwap_val = df.iloc[idx]['vwap']
                    if c['close'] > vwap_val: # Price above VWAP = Bullish?
                        # For Waterfall (Short), we want Price < VWAP (Bearish Trend)
                        vwap_ok = False
                else:
                    # Fallback: Rolling VWAP (50 period) approximation
                    # Simple VWAP = Sum(P*V) / Sum(V)
                    v = df['volume'].iloc[idx-50:idx+1]
                    p = df['close'].iloc[idx-50:idx+1]
                    pv = (p * v).sum()
                    vol_sum = v.sum()
                    if vol_sum > 0:
                        rolling_vwap = pv / vol_sum
                        if c['close'] > rolling_vwap:
                            vwap_ok = False
            
            # 2. C must be Red (Follow through) + Filters
            if is_c_red and vol_ok and ema_ok and vwap_ok:
                # Setup Confirmed
                entry = c['close']
                signal.direction = 'SHORT'
                signal.stop_loss = entry + self.sl_points
                signal.take_profit = entry - self.tp_points
                signal.reason = f"Waterfall: Drop {body_a:.2f} (> {self.multiplier}x)"
                if self.volume_mult > 0: signal.reason += " + Vol"
                if self.use_ema_filter: signal.reason += " + Trend"
                if self.use_vwap_filter: signal.reason += " + VWAP"
                
        return signal
