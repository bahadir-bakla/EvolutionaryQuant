import pandas as pd
import numpy as np

class GoldSniperStrategyCore:
    def __init__(self, lot_size=0.1, starting_balance=1000.0, target_pnl_usd=5000.0, max_basket_loss_usd=500.0):
        self.lot_size = lot_size
        self.initial_balance = starting_balance
        self.balance = starting_balance
        self.max_basket_loss_usd = max_basket_loss_usd  # IMPROVE-2: Hard dollar cap
        
        # We hold a basket of sniper entries all pointing to the Macro Target
        self.basket = []
        self.current_direction = 0
        self.trade_log = []
        self.target_pnl_usd = target_pnl_usd # Major swing target PnL to close the whole basket
        
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
        tap_threshold = atr * 0.5
        if support and abs(current_low - support) <= tap_threshold:
            self.zone_history['support_taps'] += 1
        if resistance and abs(current_high - resistance) <= tap_threshold:
            self.zone_history['resistance_taps'] += 1

    def calculate_basket_pnl(self, current_price, point_value=100):
        total = 0
        for t in self.basket:
            pts = (current_price - t['price']) * t['direction']
            total += pts * t['size'] * point_value
        return total

    def add_trade(self, price, direction, reason, time, size):
        self.basket.append({
            'price': price,
            'direction': direction,
            'reason': reason,
            'time': time,
            'size': size,
            'max_points': 0  # Track maximum excursion in favor
        })
        self.current_direction = direction

    def manage_bullets(self, current_price, current_time, current_atr, point_value=100.0,
                       target_points=None, stop_points=None):
        if not self.basket:
            return 0
            
        pnl_realized = 0
        active_bullets = []
        new_pyramid_bullets = []
        
        # Use precise explicit targets if provided, otherwise fallback to macro dynamic ATR calculation
        TARGET_POINTS = target_points if target_points is not None else current_atr * 20.0  
        STOP_POINTS = stop_points if stop_points is not None else min(15.0, current_atr * 1.5)
        
        # BUG-3 FIX: Check basket-wide hard dollar cap FIRST
        basket_floating_pnl = self.calculate_basket_pnl(current_price, point_value)
        if basket_floating_pnl <= -self.max_basket_loss_usd:
            # EMERGENCY: Close entire basket at once
            pnl_realized += basket_floating_pnl
            self.balance += basket_floating_pnl
            for t in self.basket:
                self.trade_log.append({
                    'entry_time': t['time'], 'exit_time': current_time,
                    'direction': "LONG" if t['direction'] == 1 else "SHORT",
                    'entry_price': t['price'], 'exit_price': current_price,
                    'pnl': (current_price - t['price']) * t['direction'] * t['size'] * point_value,
                    'reason': "BASKET_HARD_STOP"
                })
            self.basket = []
            self.current_direction = 0
            return pnl_realized
        
        for t in self.basket:
            pts_in_favor = (current_price - t['price']) * t['direction']
            open_pnl = pts_in_favor * t['size'] * point_value
                
            # Add missing trailing max_points
            if pts_in_favor > t.get('max_points', 0):
                t['max_points'] = pts_in_favor
                
            # Trailing Stop Mechanics (Locking in profit if macro trend starts)
            if t['max_points'] >= TARGET_POINTS * 0.5:
                # Trigger Risk-Free Pyramid if not already triggered
                if not t.get('pyramided', False):
                    t['pyramided'] = True
                    new_pyramid_bullets.append({
                        'price': current_price,
                        'direction': t['direction'],
                        'reason': 'PYRAMID_ADD',
                        'time': current_time,
                        'size': t['size'],
                        'max_points': 0,
                        'pyramided': True  # Do not recursively pyramid
                    })
                    
                # Once halfway to target, lock stop loss at Break Even (+10% of target to cover spread)
                # Note: STOP_POINTS is a positive distance scalar against the trade entry. Negative distance means lock-in profit.
                dynamic_stop = -TARGET_POINTS * 0.10
                
                # If trade hit 100% of target, trail dynamically below the trend
                if t['max_points'] >= TARGET_POINTS:
                    dynamic_stop = -(t['max_points'] - (TARGET_POINTS * 0.5))
            else:
                dynamic_stop = STOP_POINTS
                
            # Exit 1: STOP LOSS or TRAILING STOP HIT
            if pts_in_favor <= -dynamic_stop:  
                pnl_realized += open_pnl
                self.balance += open_pnl
                self.trade_log.append({
                    'entry_time': t['time'], 'exit_time': current_time,
                    'direction': "LONG" if t['direction'] == 1 else "SHORT",
                    'entry_price': t['price'], 'exit_price': current_price,
                    'pnl': open_pnl, 'reason': "STOP_LOSS_OR_TRAIL"
                })
            # Exit 2: MACRO TARGET HIT (Removed static exit to allow Trailing Stop to ride the wave)
            # We now rely solely on Exit 1 (Trailing Stop) or Exit 3 (Time limit) to close winning trades.
            # IMPROVE-3: Exit 3: STALE TRADE KILLER (72h = 72 bars on 1H)
            elif hasattr(current_time, 'hour') and hasattr(t['time'], 'hour') and (current_time - t['time']).total_seconds() > 72 * 3600:
                pnl_realized += open_pnl
                self.balance += open_pnl
                self.trade_log.append({
                    'entry_time': t['time'], 'exit_time': current_time,
                    'direction': "LONG" if t['direction'] == 1 else "SHORT",
                    'entry_price': t['price'], 'exit_price': current_price,
                    'pnl': open_pnl, 'reason': "STALE_72H_KILL"
                })
            else:
                active_bullets.append(t)
                
        # Add any new pyramids to the active basket
        active_bullets.extend(new_pyramid_bullets)
        
        self.basket = active_bullets
        if not self.basket:
            self.current_direction = 0
            
        return pnl_realized


def add_gold_sniper_features(df):
    """ Calculate Daily Bias, 4H Rejections, and LTF Minor Sweeps natively on 1H data """
    df = df.copy()
    
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift(1)), 
                          abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # 1. DAILY MACRO CONTEXT (1H chart -> 24 periods per day)
    # Roll 24 periods to get yesterday's close, high, and low.
    df['d1_close_prev'] = df['close'].shift(24)
    df['d1_close_prev2'] = df['close'].shift(48)
    df['d1_high_prev'] = df['high'].rolling(24).max().shift(24)
    df['d1_low_prev'] = df['low'].rolling(24).min().shift(24)
    
    # Bullish Bias: Last day closed higher than the day before
    df['daily_bias_bullish'] = df['d1_close_prev'] > df['d1_close_prev2']
    df['daily_bias_bearish'] = df['d1_close_prev'] < df['d1_close_prev2']

    # 2. 4H CONTEXT & REJECTIONS (1H chart -> 4 periods)
    df['h4_high'] = df['high'].rolling(4).max().shift(1)
    df['h4_low'] = df['low'].rolling(4).min().shift(1)
    
    # Rejection Wicks on the 4H zone
    h4_range = df['h4_high'] - df['h4_low']
    df['h4_reject_down'] = (df['close'] - df['h4_low']) > (h4_range * 0.6) # Buying pressure
    df['h4_reject_up'] = (df['h4_high'] - df['close']) > (h4_range * 0.6) # Selling pressure
    
    # 3. MINOR LIQUIDITY SWEEPS (1H 5-period window)
    df['minor_high'] = df['high'].rolling(5).max().shift(1)
    df['minor_low'] = df['low'].rolling(5).min().shift(1)
    
    df['sweep_minor_low'] = (df['low'] < df['minor_low']) & (df['close'] > df['minor_low'])
    df['sweep_minor_high'] = (df['high'] > df['minor_high']) & (df['close'] < df['minor_high'])
    
    # 4. ORDER BLOCK TOUCH (1H)
    df['is_down'] = df['close'] < df['open']
    df['is_up'] = df['close'] > df['open']
    df['ob_tap_bullish'] = df['is_down'].shift(2) & df['is_up'].shift(1) & (df['low'] <= df['low'].shift(2)) & (df['close'] >= df['close'].shift(1))
    df['ob_tap_bearish'] = df['is_up'].shift(2) & df['is_down'].shift(1) & (df['high'] >= df['high'].shift(2)) & (df['close'] <= df['close'].shift(1))

    return df
