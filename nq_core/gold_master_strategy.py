import pandas as pd
import numpy as np

class GoldMasterStrategyCore:
    def __init__(self, target_pips=100.0, stop_pips=50.0, lot_size=0.1, starting_balance=1000.0):
        # We use pips/points for absolute targets since these are high R:R trades, not basket scalps
        self.target_points = target_pips
        self.stop_points = stop_pips
        self.lot_size = lot_size
        self.initial_balance = starting_balance
        self.balance = starting_balance
        
        self.open_position = None # We hold 1 massive conviction trade, no complex DCA basket
        self.trade_log = []
        
        # State Machine memory for Triple-Taps
        self.zone_history = {
            'support_taps': 0,
            'resistance_taps': 0,
            'last_support_level': None,
            'last_resistance_level': None
        }

    def reset_zone_history(self, support, resistance):
        if self.zone_history['last_support_level'] != support:
            self.zone_history['support_taps'] = 0
            self.zone_history['last_support_level'] = support
            
        if self.zone_history['last_resistance_level'] != resistance:
            self.zone_history['resistance_taps'] = 0
            self.zone_history['last_resistance_level'] = resistance

    def check_zone_interactions(self, current_low, current_high, support, resistance, atr):
        # Track taps within 0.5 ATR of the HTF zone
        tap_threshold = atr * 0.5
        
        # Support Interaction Check
        if support and abs(current_low - support) <= tap_threshold:
            self.zone_history['support_taps'] += 1
            
        # Resistance Interaction Check
        if resistance and abs(current_high - resistance) <= tap_threshold:
            self.zone_history['resistance_taps'] += 1

    def manage_position(self, current_price, current_time, point_value=100.0):
        if not self.open_position:
            return None
            
        points_moved = (current_price - self.open_position['price']) * self.open_position['direction']
        pnl = points_moved * self.open_position['size'] * point_value
        
        # Hard SL and TP for Master Strategy
        if points_moved <= -self.stop_points:
            return "STOP_LOSS"
            
        if points_moved >= self.target_points:
            return "TAKE_PROFIT"
            
        # Margin call
        if self.balance + pnl <= 0:
            return "MARGIN_CALL"
            
        return "HOLD"
        
    def execute_close(self, current_price, current_time, reason, point_value=100.0):
        points_moved = (current_price - self.open_position['price']) * self.open_position['direction']
        pnl = points_moved * self.open_position['size'] * point_value
        
        self.balance += pnl
        self.trade_log.append({
            'entry_time': self.open_position['time'],
            'exit_time': current_time,
            'direction': "LONG" if self.open_position['direction'] == 1 else "SHORT",
            'entry_price': self.open_position['price'],
            'exit_price': current_price,
            'pnl': pnl,
            'reason': reason
        })
        self.open_position = None
        return pnl

    def execute_open(self, price, direction, reason, time, size):
        self.open_position = {
            'price': price,
            'direction': direction,
            'time': time,
            'size': size,
            'reason': reason
        }


def add_gold_master_features(df):
    """ Calculate FVGs, OBs, Momentum, and Triple-Bases """
    df = df.copy()
    
    # 1. MTF Context (Approximating 4H zones on 15m/5m timeframe)
    # Using rolling 48 periods for 4H equivalent on a 5m chart
    df['htf_support'] = df['low'].rolling(48).min().shift(1)
    df['htf_resistance'] = df['high'].rolling(48).max().shift(1)
    
    # 2. Fair Value Gaps (FVG)
    # Bullish FVG: Low of candle 3 is higher than High of candle 1
    df['fvg_bullish'] = df['low'] > df['high'].shift(2)
    # Bearish FVG: High of candle 3 is lower than Low of candle 1
    df['fvg_bearish'] = df['high'] < df['low'].shift(2)
    
    # 3. Momentum (Rate of Change)
    # 10-period momentum to confirm explosive movement escaping a zone
    df['momentum_roc'] = df['close'].pct_change(periods=10) * 100
    
    # ATR for sizing targets
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift(1)), 
                          abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    
    return df
