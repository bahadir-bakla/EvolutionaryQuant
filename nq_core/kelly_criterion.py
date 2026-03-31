import math
import numpy as np

class DynamicKellySizer:
    """
    Dynamic Kelly Criterion Sizer
    f* = W - ((1 - W) / R)
    Where:
    W = Empirical Win Rate (0.0 to 1.0)
    R = Average Reward-to-Risk Ratio
    
    If the algorithm doesn't have enough history, it forcefully degrades 
    to a predefined 'Safe Base Multiplier'.
    """
    
    def __init__(self, safe_base_multiplier=0.10, max_kelly_fraction=0.50, min_history=10):
        self.safe_base_multiplier = safe_base_multiplier
        self.max_kelly_fraction = max_kelly_fraction
        self.min_history = min_history
        self.use_kelly = False  # The Graceful Degradation Toggle
        
        # Track historical trades for dynamic calculation
        self.wins = 0
        self.losses = 0
        self.total_reward = 0.0
        self.total_risk_taken = 0.0

    def toggle_kelly(self, state: bool):
        self.use_kelly = state

    def record_trade(self, pnl_usd: float, risk_usd: float):
        """ Feed real-time trade results into the memory bank. """
        if pnl_usd > 0:
            self.wins += 1
            self.total_reward += pnl_usd
        else:
            self.losses += 1
            # Prevent negative zero, etc
            actual_loss = abs(pnl_usd) if pnl_usd < 0 else risk_usd
            self.total_risk_taken += actual_loss
            
    def calculate_kelly_fraction(self) -> float:
        """ 
        Calculates the exact Kelly fraction. 
        Returns the safe_base_multiplier if toggled off or if history is insufficient.
        """
        if not self.use_kelly:
            return self.safe_base_multiplier
            
        total_trades = self.wins + self.losses
        
        if total_trades < self.min_history:
            return self.safe_base_multiplier
            
        win_rate = self.wins / total_trades
        
        # Average winner vs Average loser
        avg_win = self.total_reward / self.wins if self.wins > 0 else 0
        avg_loss = self.total_risk_taken / self.losses if self.losses > 0 else 1.0 # default to 1 to avoid div0
        
        if avg_loss == 0:
            return self.max_kelly_fraction  # Infinite R:R means max aggression
            
        r_ratio = avg_win / avg_loss
        
        if r_ratio <= 0:
            return self.safe_base_multiplier / 2.0 # Defensive scaling if strategy is bleeding
            
        # The Kelly Formula
        kelly_pct = win_rate - ((1 - win_rate) / r_ratio)
        
        # Sanitize and Clamp the result
        if kelly_pct <= 0:
            return 0.01  # Absolute minimum survival lot
            
        # Optional Half-Kelly for safety
        half_kelly = kelly_pct / 2.0
        
        # Clamp to max allowed fraction
        final_fraction = min(half_kelly, self.max_kelly_fraction)
        
        return round(final_fraction, 3)

    def get_lot_multiplier(self, current_equity: float, base_equity: float = 1000.0) -> float:
        """ Combines global equity scaling with the Kelly Edge. """
        kelly_f = self.calculate_kelly_fraction()
        global_scale = max(0.1, current_equity / base_equity)
        
        # The scaling factor is the empirical Edge (Kelly) multiplied by Account Growth
        return round(kelly_f * global_scale, 2)
