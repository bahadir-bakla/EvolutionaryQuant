# Reversion Scalper Strategy (User Logic)
# 1. Detect 7 consecutive red candles (Oversold).
# 2. Buy at Close. Target 50% Fib Retracement.
# 3. At 50%, Flip to Short. Target Original Low.

import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class ReversionConfig:
    streak_threshold: int = 7 # 7 for NQ, 5 for Gold
    retrace_pct: float = 0.5
    use_smart_targets: bool = True
    use_rsi_filter: bool = False
    rsi_period: int = 14
    rsi_limit: int = 30
    
@dataclass
class ReversionSignal:
    direction: str # LONG, SHORT, EXIT, NEUTRAL
    stop_loss: float
    take_profit: float
    reason: str
    target_type: str = "FIXED" # FIXED or SMART

class ReversionScalper:
    def __init__(self, config: ReversionConfig):
        self.config = config
        self.streak_threshold = config.streak_threshold
        self.retrace_pct = config.retrace_pct
        # State Tracking
        self.state = 'NEUTRAL' # NEUTRAL, LONG_PHASE, SHORT_PHASE
        self.drop_high = 0.0
        self.drop_low = 0.0
        self.retrace_level = 0.0
        
    def evaluate(self, df: pd.DataFrame, idx: int, position: dict = None, key_levels: list = None) -> ReversionSignal:
        row = df.iloc[idx]
        # Current Candle Color
        is_red = row['close'] < row['open']
        is_green = row['close'] > row['open']
        
        signal = ReversionSignal('NEUTRAL', 0.0, 0.0, "")
        
        # 0. Check if we are currently in a trade
        # The backtester/live runner usually handles "Is in trade?", but for this specific 
        # "Flip" logic, we need to know WHERE we are in the sequence.
        # We will assume the `position` arg tells us if we have a live trade.
        
        has_position = position is not None and position['direction'] != 'NEUTRAL'
        pos_dir = position['direction'] if has_position else 'NEUTRAL'
        
        # 1. NEUTRAL STATE: Look for 7 Red Candles
        if not has_position:
            self.state = 'NEUTRAL' # Reset state if flat
            
            # Lookback to count reds
            # We need strictly consecutive reds INCLUDING current?
            # Or consecutive reds PREVIOUSLY and we enter now?
            # User: "art arda 6-7 kırmızı mum kapanmışsa" (if 6-7 red candles CLOSED)
            # So check last 7 candles (idx-6 to idx)
            
            # Slice last N candles
            window = self.streak_threshold
            if idx < window: return signal
            
            recent = df.iloc[idx-window+1 : idx+1] # +1 to include current
            
            # Check if all are red
            # Note: This is computationally standard for scalping
            all_red = all(c['close'] < c['open'] for _, c in recent.iterrows())
            
            # RSI Filter Check
            rsi_ok = True
            if self.config.use_rsi_filter and all_red:
                # Calculate RSI on the fly (simple approx or using history)
                # Need > 14 bars
                if idx > self.config.rsi_period + 1:
                    window_df = df.iloc[max(0, idx-50):idx+1].copy()
                    delta = window_df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/self.config.rsi_period, adjust=False).mean()
                    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/self.config.rsi_period, adjust=False).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    curr_rsi = rsi.iloc[-1]
                    
                    # If Red Streak, we want RSI to be OVERSOLD (< 30)
                    if curr_rsi > self.config.rsi_limit:
                        rsi_ok = False
            
            if all_red and rsi_ok:
                # Setup Found!
                self.drop_high = recent.iloc[0]['high'] # Top of the drop
                self.drop_low = recent.iloc[-1]['low']  # Bottom of the drop
                
                # Dynamic Retrace Level
                self.retrace_level = self.drop_low + (self.drop_high - self.drop_low) * self.retrace_pct
                
                # Signal LONG
                signal.direction = 'LONG'
                signal.stop_loss = self.drop_low - (self.drop_high - self.drop_low) * 0.5 # Wide Stop
                signal.take_profit = self.retrace_level
                signal.reason = f"{self.streak_threshold} Red Streak -> {int(self.retrace_pct*100)}% Retrace"
                if self.config.use_rsi_filter:
                    signal.reason += " + RSI Filter"
                
                self.state = 'LONG_PHASE'
                return signal

        # 2. LONG PHASE: Waiting for Retrace
        elif pos_dir == 'LONG':
            # Check if we hit target? 
            # The backtester usually exits for us at TP.
            # But we want to FLIP to SHORT immediately at TP.
            # This is tricky in a standard "Evaluating Candle Close" loop because TP happens intra-candle.
            
            # If High >= Retrace Level, we logicially hit it.
            if row['high'] >= self.retrace_level:
                # We hit the Retrace level.
                # Signal FLIP TO SHORT
                signal.direction = 'SHORT'
                signal.stop_loss = self.drop_high # Stop at the top (100% retrace)
                
                # DEFAULT TARGET: The Drop Low
                target_price = self.drop_low
                tgt_type = "FIXED"
                
                # SMART TARGET: Use Nearest Key Pivot Level if available
                if self.config.use_smart_targets and key_levels:
                    # Find pivots BELOW the current price (Retrace Level)
                    deeper_pivots = [p for p in key_levels if p < self.drop_low]
                    if deeper_pivots:
                        target_price = max(deeper_pivots) # Nearest below
                        signal.reason = "Flip Short -> Smart Target (Major Pivot)"
                        tgt_type = "SMART"
                    else:
                        signal.reason = "Flip Short -> Target Low"
                else:
                    signal.reason = "Flip Short -> Target Low"
                
                signal.take_profit = target_price
                signal.target_type = tgt_type
                self.state = 'SHORT_PHASE'
                return signal

        # 3. SHORT PHASE: Waiting for Low
        elif pos_dir == 'SHORT':
            # Target is dynamic (set at entry).
            # If Low <= TP, we assume TP hit.
            # Usually handled by Position Dict check, but here for signal generation:
            pass # No new signal needed until TP hit logic in runner
                
        return signal
