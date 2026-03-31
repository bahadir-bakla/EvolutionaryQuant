import pandas as pd
import numpy as np

class GoldenStrategyCore:
    def __init__(self, target_profit_usd=1500, max_layers=8, dca_step_points=25, pyramid_step_points=15, lot_size=15, starting_balance=100000, max_basket_loss_usd=300.0):
        self.target_profit_usd = target_profit_usd
        self.max_layers = max_layers
        self.dca_step_points = dca_step_points
        self.pyramid_step_points = pyramid_step_points
        self.lot_size = lot_size
        self.initial_balance = starting_balance
        self.balance = starting_balance
        self.max_basket_loss_usd = max_basket_loss_usd  # Hard dollar cap
        
        self.basket = []
        self.current_direction = 0  # 1 for LONG, -1 for SHORT
        self.highest_basket_pnl = 0.0  # Track max profit for trailing stops
        
    def add_trade(self, price, direction, reason, time):
        # Prevent adding trades if we are busted
        if self.balance <= 0:
            return False
            
        self.basket.append({
            'price': price,
            'direction': direction,
            'reason': reason,
            'time': time,
            'size': self.lot_size
        })
        self.current_direction = direction
        return True
        
    def calculate_basket_pnl(self, current_price, point_value=20):
        total = 0
        for t in self.basket:
            pts = (current_price - t['price']) * t['direction']
            total += pts * t['size'] * point_value
        return total
        
    def check_basket_logic(self, current_price, current_time, point_value=20, htf_high=None, htf_low=None, portfolio_equity=None):
        if not self.basket:
            return None # Neutral
            
        pnl = self.calculate_basket_pnl(current_price, point_value)
        
        # Track max PnL for Trailing Stop
        if pnl > self.highest_basket_pnl:
            self.highest_basket_pnl = pnl
        
        # Margin call / Account Blown Check — use GLOBAL portfolio equity if available
        equity = (portfolio_equity + pnl) if portfolio_equity is not None else (self.balance + pnl)
        if equity <= 0:
            return "MARGIN_CALL"
        
        # IMPROVE-2: Hard dollar loss cap on NQ basket
        if pnl <= -self.max_basket_loss_usd:
            return "HARD_STOP"
            
        # 0. ABYSS RIDER (Dynamic Trailing Stop to catch 600-800 Point Macro Trends)
        # If profit explodes to 3x the baseline target (e.g. 120+ points), activate trailing mode
        if self.highest_basket_pnl >= (self.target_profit_usd * 3):
            # Trail by 1.5x the baseline target (e.g. 60 points). 
            # So if NQ crashes 800 points, we only close when it bounces 60 points from the absolute bottom.
            if pnl <= self.highest_basket_pnl - (self.target_profit_usd * 1.5):
                return "CLOSE_ALL"
            # Bypassing the static targets below to allow infinite running
        else:
            # 1. Standard USD Target Profit (Rescue Mode for Heavy DCA)
            if pnl >= self.target_profit_usd and len(self.basket) >= 3:
                return "CLOSE_ALL"
                
            # 1b. Huge Standard Target for surgical entries 
            # (If it fails to hit Abyss mode but gets lucky, though Abyss usually catches it first)
            if pnl >= (self.target_profit_usd * 5) and len(self.basket) < 3:
                return "CLOSE_ALL"
            
        # 2. HTF (4H) Target Profit (If provided and we're in profit)
        if pnl > 0 and htf_high is not None and htf_low is not None:
            # If LONG and price reached near HTF Swing High / Resistance
            if self.current_direction == 1 and current_price >= (htf_high - 5):
                return "CLOSE_ALL_HTF"
            # If SHORT and price reached near HTF Swing Low / Support
            elif self.current_direction == -1 and current_price <= (htf_low + 5):
                return "CLOSE_ALL_HTF"
            
        if len(self.basket) < self.max_layers:
            last_price = self.basket[-1]['price']
            points_moved = (current_price - last_price) * self.current_direction
            
            if points_moved <= -self.dca_step_points:
                return "DCA"
            elif points_moved >= self.pyramid_step_points:
                return "PYRAMID"
                
        return "HOLD"
        
    def clear_basket(self, realized_pnl):
        self.balance += realized_pnl
        self.basket = []
        self.current_direction = 0
        self.highest_basket_pnl = 0.0

def add_golden_features(df):
    """ Calculate Liquidity Grabs, OBs, and VWAP """
    df = df.copy()
    
    # 20-period Swing High/Low for Liquidity points (LTF)
    df['swing_high'] = df['high'].rolling(20).max().shift(1)
    df['swing_low'] = df['low'].rolling(20).min().shift(1)
    
    # HTF (approx 4H) Swing High/Low for Liquidity Targets
    # 5m chart: 12 candles/hr * 4 = 48 candles. 20-period on 4H = 20 * 48 = 960 candles
    df['htf_swing_high'] = df['high'].rolling(960).max().shift(1)
    df['htf_swing_low'] = df['low'].rolling(960).min().shift(1)
    
    # ATR
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['close'].shift(1)), 
                          abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    
    # Liquidity Sweep (Sweep & Reject)
    # Price pokes above swing high but closes below it (or just forms a pinbar-like structure)
    df['sweep_high'] = (df['high'] > df['swing_high']) & (df['close'] < df['swing_high'])
    df['sweep_low'] = (df['low'] < df['swing_low']) & (df['close'] > df['swing_low'])
    
    # VWAP (Simplified session VWAP for intraday data)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['tp_v'] = df['typical_price'] * df['volume']
    
    if hasattr(df.index, 'date'):
        df['date'] = df.index.date
        df['cum_vol'] = df.groupby('date')['volume'].cumsum()
        df['cum_tp_v'] = df.groupby('date')['tp_v'].cumsum()
        df['vwap'] = df['cum_tp_v'] / (df['cum_vol'] + 1e-10)
    else:
        df['vwap'] = df['close'].rolling(50).mean() # fallback
        
    # Order Block Definition: Last down candle before strong up move (Bullish OB)
    df['is_down'] = df['close'] < df['open']
    df['is_up'] = df['close'] > df['open']
    
    # Very simple Institutional Engulfing approximation mapping
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(20).mean()
    
    # Strong candle = body is 1.5x larger than average
    df['strong_up'] = df['is_up'] & (df['body'] > df['avg_body'] * 1.5)
    df['strong_down'] = df['is_down'] & (df['body'] > df['avg_body'] * 1.5)
    
    df['bullish_ob_formed'] = df['strong_up'] & df['is_down'].shift(1)
    df['bearish_ob_formed'] = df['strong_down'] & df['is_up'].shift(1)
    
    return df
