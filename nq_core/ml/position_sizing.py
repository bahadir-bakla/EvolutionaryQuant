# NQ Quant Bot - Adaptive Position Sizing
# Dynamic sizing based on confidence, regime, and drawdown

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class SizingMode(Enum):
    FIXED = "FIXED"           # Fixed % of capital
    KELLY = "KELLY"           # Kelly Criterion
    ADAPTIVE = "ADAPTIVE"     # Confidence-based
    REGIME = "REGIME"         # Regime-based adjustment


@dataclass
class PositionSize:
    """Position sizing result"""
    size_percent: float       # % of capital to risk
    size_dollars: float       # Dollar amount
    size_contracts: int       # Number of contracts (for futures)
    risk_per_trade: float     # Dollar risk on this trade
    confidence_mult: float    # Confidence multiplier applied
    regime_mult: float        # Regime multiplier applied
    drawdown_mult: float      # Drawdown protection multiplier
    rationale: str


class AdaptivePositionSizer:
    """
    Adaptive Position Sizing with multiple factors
    
    Factors:
    1. Base size (Kelly or fixed %)
    2. Signal confidence
    3. Market regime
    4. Current drawdown
    5. Volatility
    """
    
    def __init__(
        self,
        base_risk_percent: float = 0.02,  # 2% base risk
        max_risk_percent: float = 0.05,   # 5% max risk
        use_kelly: bool = True,
        kelly_fraction: float = 0.5       # Half-Kelly
    ):
        self.base_risk = base_risk_percent
        self.max_risk = max_risk_percent
        self.use_kelly = use_kelly
        self.kelly_fraction = kelly_fraction
        
        # Track performance for Kelly
        self.wins = []
        self.losses = []
        
    def update_performance(self, pnl: float):
        """Update win/loss history"""
        if pnl > 0:
            self.wins.append(pnl)
        elif pnl < 0:
            self.losses.append(abs(pnl))
    
    def calculate_kelly(self) -> float:
        """
        Calculate Kelly Criterion
        
        Kelly % = W - (1-W)/R
        W = win rate
        R = avg win / avg loss
        """
        if len(self.wins) < 5 or len(self.losses) < 5:
            return self.base_risk
        
        win_rate = len(self.wins) / (len(self.wins) + len(self.losses))
        avg_win = np.mean(self.wins)
        avg_loss = np.mean(self.losses)
        
        if avg_loss == 0:
            return self.base_risk
        
        win_loss_ratio = avg_win / avg_loss
        
        kelly = win_rate - (1 - win_rate) / win_loss_ratio
        kelly = max(0, kelly) * self.kelly_fraction  # Half-Kelly
        
        return min(kelly, self.max_risk)
    
    def get_confidence_multiplier(self, confidence: float) -> float:
        """
        Adjust size based on signal confidence
        
        confidence: 0-1 probability
        """
        if confidence >= 0.9:
            return 1.5  # Very high confidence
        elif confidence >= 0.75:
            return 1.2
        elif confidence >= 0.6:
            return 1.0
        elif confidence >= 0.5:
            return 0.8
        else:
            return 0.5  # Low confidence
    
    def get_regime_multiplier(self, regime: str, signal: str) -> float:
        """
        Adjust size based on market regime alignment
        
        regime: 'BULL', 'BEAR', 'CHOPPY'
        signal: 'LONG', 'SHORT'
        """
        # Aligned with trend = larger size
        if regime == 'BULL' and signal == 'LONG':
            return 1.3
        elif regime == 'BEAR' and signal == 'SHORT':
            return 1.3
        # Counter-trend = smaller size
        elif regime == 'BULL' and signal == 'SHORT':
            return 0.5
        elif regime == 'BEAR' and signal == 'LONG':
            return 0.5
        # Choppy = reduced size
        elif regime == 'CHOPPY':
            return 0.7
        else:
            return 1.0
    
    def get_drawdown_multiplier(
        self, 
        current_equity: float, 
        peak_equity: float
    ) -> float:
        """
        Reduce size during drawdown
        
        Protect capital when losing
        """
        if peak_equity <= 0:
            return 1.0
        
        drawdown = (peak_equity - current_equity) / peak_equity
        
        if drawdown >= 0.15:  # 15%+ drawdown
            return 0.25       # Cut size to 25%
        elif drawdown >= 0.10:  # 10-15%
            return 0.5
        elif drawdown >= 0.05:  # 5-10%
            return 0.75
        else:
            return 1.0
    
    def get_volatility_adjustment(
        self, 
        current_atr: float, 
        avg_atr: float
    ) -> float:
        """
        Adjust for current volatility
        
        High volatility = smaller size
        """
        if avg_atr == 0:
            return 1.0
        
        vol_ratio = current_atr / avg_atr
        
        if vol_ratio >= 2.0:  # Very high volatility
            return 0.5
        elif vol_ratio >= 1.5:
            return 0.7
        elif vol_ratio <= 0.5:  # Very low volatility
            return 1.2
        else:
            return 1.0
    
    def calculate_position_size(
        self,
        capital: float,
        signal: str,              # 'LONG', 'SHORT'
        confidence: float,        # 0-1
        regime: str,              # 'BULL', 'BEAR', 'CHOPPY'
        current_equity: float,
        peak_equity: float,
        current_atr: float,
        avg_atr: float,
        entry_price: float,
        stop_loss: float,
        contract_size: float = 20.0  # NQ = $20 per point
    ) -> PositionSize:
        """
        Calculate optimal position size
        """
        # Base risk
        if self.use_kelly:
            base_risk = self.calculate_kelly()
        else:
            base_risk = self.base_risk
        
        # Get multipliers
        conf_mult = self.get_confidence_multiplier(confidence)
        regime_mult = self.get_regime_multiplier(regime, signal)
        dd_mult = self.get_drawdown_multiplier(current_equity, peak_equity)
        vol_mult = self.get_volatility_adjustment(current_atr, avg_atr)
        
        # Combined multiplier
        combined_mult = conf_mult * regime_mult * dd_mult * vol_mult
        
        # Final risk percent
        final_risk = base_risk * combined_mult
        final_risk = min(final_risk, self.max_risk)  # Cap at max
        final_risk = max(final_risk, 0.005)  # Minimum 0.5%
        
        # Dollar risk
        dollar_risk = capital * final_risk
        
        # Calculate position size based on stop distance
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance > 0:
            # For futures: position size in contracts
            dollar_per_point = contract_size
            points_risk = stop_distance
            max_contracts = dollar_risk / (points_risk * dollar_per_point)
            contracts = max(1, int(max_contracts))
            
            actual_dollar_risk = contracts * points_risk * dollar_per_point
            actual_percent = actual_dollar_risk / capital
        else:
            contracts = 1
            actual_dollar_risk = dollar_risk
            actual_percent = final_risk
        
        # Build rationale
        rationale_parts = []
        if conf_mult != 1.0:
            rationale_parts.append(f"conf={conf_mult:.1f}x")
        if regime_mult != 1.0:
            rationale_parts.append(f"regime={regime_mult:.1f}x")
        if dd_mult != 1.0:
            rationale_parts.append(f"dd_prot={dd_mult:.1f}x")
        if vol_mult != 1.0:
            rationale_parts.append(f"vol={vol_mult:.1f}x")
        
        rationale = f"Base {base_risk:.1%} " + " ".join(rationale_parts)
        
        return PositionSize(
            size_percent=actual_percent,
            size_dollars=contracts * entry_price * contract_size,
            size_contracts=contracts,
            risk_per_trade=actual_dollar_risk,
            confidence_mult=conf_mult,
            regime_mult=regime_mult,
            drawdown_mult=dd_mult,
            rationale=rationale
        )


class AntiMartingale:
    """
    Anti-Martingale position sizing
    
    Increase size during winning streaks
    Decrease size during losing streaks
    """
    
    def __init__(
        self,
        base_size: float = 1.0,
        increment: float = 0.25,
        max_multiplier: float = 2.0,
        min_multiplier: float = 0.5
    ):
        self.base_size = base_size
        self.increment = increment
        self.max_mult = max_multiplier
        self.min_mult = min_multiplier
        self.current_mult = 1.0
        self.streak = 0
    
    def update(self, is_win: bool) -> float:
        """Update streak and return new multiplier"""
        if is_win:
            if self.streak >= 0:
                self.streak += 1
            else:
                self.streak = 1
        else:
            if self.streak <= 0:
                self.streak -= 1
            else:
                self.streak = -1
        
        # Calculate multiplier
        if self.streak > 0:
            # Winning streak - increase
            self.current_mult = min(
                self.max_mult,
                1.0 + (self.streak * self.increment)
            )
        elif self.streak < 0:
            # Losing streak - decrease
            self.current_mult = max(
                self.min_mult,
                1.0 + (self.streak * self.increment)
            )
        else:
            self.current_mult = 1.0
        
        return self.current_mult
    
    def get_size(self) -> float:
        """Get current size multiplier"""
        return self.base_size * self.current_mult


# === TEST ===
if __name__ == "__main__":
    print("Testing Adaptive Position Sizing...")
    
    sizer = AdaptivePositionSizer(
        base_risk_percent=0.02,
        max_risk_percent=0.05,
        use_kelly=True
    )
    
    # Simulate some trades
    sample_trades = [500, -200, 300, 400, -150, 600, -100, 200]
    for pnl in sample_trades:
        sizer.update_performance(pnl)
    
    # Calculate position for a trade
    position = sizer.calculate_position_size(
        capital=100000,
        signal='LONG',
        confidence=0.75,
        regime='BULL',
        current_equity=98000,  # Slight drawdown
        peak_equity=100000,
        current_atr=150,
        avg_atr=120,
        entry_price=21000,
        stop_loss=20850,
        contract_size=20
    )
    
    print(f"\n{'='*60}")
    print("POSITION SIZE CALCULATION")
    print(f"{'='*60}")
    print(f"Signal: LONG at 21000, Stop: 20850")
    print(f"Risk %: {position.size_percent:.2%}")
    print(f"Risk $: ${position.risk_per_trade:,.2f}")
    print(f"Contracts: {position.size_contracts}")
    print(f"Position Value: ${position.size_dollars:,.2f}")
    print(f"\nMultipliers:")
    print(f"  Confidence: {position.confidence_mult:.1f}x")
    print(f"  Regime: {position.regime_mult:.1f}x")
    print(f"  Drawdown: {position.drawdown_mult:.1f}x")
    print(f"\nRationale: {position.rationale}")
    
    # Anti-Martingale test
    print(f"\n{'='*60}")
    print("ANTI-MARTINGALE TEST")
    print(f"{'='*60}")
    
    am = AntiMartingale()
    results = ['W', 'W', 'W', 'L', 'L', 'W', 'W']
    
    for i, r in enumerate(results):
        mult = am.update(r == 'W')
        print(f"  Trade {i+1}: {r} -> Multiplier: {mult:.2f}x")
