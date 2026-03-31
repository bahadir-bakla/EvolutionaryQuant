# NQ Quant Bot - Gold Master Strategy
# 15m Execution + 4H Structure + Price Action (FVG, OB, 3rd Touch)

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging

from .kalman_predict import KalmanPredictor, get_kalman_signal
from .optimized_strategy import OptimizedSignal
from .order_blocks import detect_order_blocks, check_ob_interaction, OrderBlock
from .price_action import detect_fvgs, check_fvg_interaction, FairValueGap
from .brain import QuantBrain, MarketRegime

logger = logging.getLogger(__name__)

@dataclass
class GoldMasterConfig:
    """Configuration for Gold Master"""
    symbol: str = "GC=F" 
    contract_value: float = 100.0 
    
    # Risk
    atr_stop_mult: float = 2.0  # Widen stops for Gold volatility
    atr_tp_mult_1: float = 2.0
    atr_tp_mult_2: float = 4.0
    
    # Filters
    min_adx: float = 20.0       # Ignore weak trends
    
    # 4H Approximation
    trend_ema_period: int = 200 
    
    # Weights
    weight_bias: float = 2.0
    weight_ob: float = 3.0
    weight_fvg: float = 2.5
    weight_3rd_touch: float = 4.0 
    
    # Signal Threshold
    min_score_long: float = 5.5 # Stricter
    min_score_short: float = 5.5


class GoldMasterStrategy:
    """
    Gold Master Strategy
    
    Logic:
    1. Bias (4H Proxy): Price > EMA 200 (15m) -> Bullish Bias
    2. Setup:
       - Retest of Bullish Order Block
       - Fill of Bullish FVG
       - 3rd Touch Breakout of Resistance
    """
    
    def __init__(self, config: GoldMasterConfig = None):
        self.config = config or GoldMasterConfig()
        
        self.kalman = KalmanPredictor(process_noise=0.005, measurement_noise=0.05)
        self.order_blocks: List[OrderBlock] = []
        self.fvgs: List[FairValueGap] = []
        
        # State
        self.last_ob_update = 0
        self.brain = QuantBrain()
        
    def _update_structure(self, df: pd.DataFrame, idx: int):
        """Update OBs and FVGs periodically"""
        if idx - self.last_ob_update > 5: # Update every 5 bars
            # Lookback 100 bars for structure
            subset = df.iloc[max(0, idx-100):idx+1]
            self.order_blocks = detect_order_blocks(subset, lookback=20)
            self.fvgs = detect_fvgs(subset)
            self.last_ob_update = idx
            
    def evaluate(self, df: pd.DataFrame, idx: int) -> OptimizedSignal:
        self._update_structure(df, idx)
        
        row = df.iloc[idx]
        price = row['close']
        
        # 1. Higher Timeframe Bias (Approximation)
        ema_period = self.config.trend_ema_period
        ema_col = f'ema_{ema_period}'
        
        # Check if pre-calculated
        if ema_col in df.columns:
            ema_trend = row[ema_col]
        else:
             # Calculate dynamically (slower)
             # Note: For optimization, pre-calculate in dataframe
             ema_trend = df['close'].ewm(span=ema_period).mean().iloc[idx]
        
        bias = 'BULLISH' if price > ema_trend else 'BEARISH'
        
        # 2. Pullback Trigger (Stochastics)
        stoch_k = row.get('stoch_k', 50)
        
        # 3. Momentum Confirmation (Kalman)
        kalman_pred = self.kalman.update(price)
        k_sig, _ = get_kalman_signal(self.kalman)
        
        # SCORING
        long_score = 0.0
        short_score = 0.0
        factors = {}
        
        # A. TREND ALIGNMENT
        if bias == 'BULLISH':
            long_score += 2.0
            factors['bias'] = "Uptrend (Price > EMA200)"
        else:
            short_score += 2.0
            factors['bias'] = "Downtrend (Price < EMA200)"
            
        # B. PULLBACK ENTRY
        # Buy Dip in Uptrend
        if bias == 'BULLISH':
            if stoch_k < 30: 
                long_score += 3.0
                factors['setup'] = "Oversold Pullback"
            elif stoch_k < 50 and k_sig == 'LONG': # Early turn up
                long_score += 2.0
                factors['setup'] = "Momentum turn in Dip"
                
        # Sell Rally in Downtrend
        if bias == 'BEARISH':
            if stoch_k > 70:
                short_score += 3.0
                factors['setup'] = "Overbought Rally"
            elif stoch_k > 50 and k_sig == 'SHORT':
                short_score += 2.0
                factors['setup'] = "Momentum turn in Rally"
                
        # C. STRUCTURE BONUS (OB/FVG)
        active_ob, _ = check_ob_interaction(price, self.order_blocks)
        active_fvg = check_fvg_interaction(price, self.fvgs)
        
        if active_ob:
            if bias == 'BULLISH' and active_ob.direction == 'BULLISH':
                long_score += 2.0
                factors['confluence'] = "Key Support (OB)"
            elif bias == 'BEARISH' and active_ob.direction == 'BEARISH':
                short_score += 2.0
                factors['confluence'] = "Key Resistance (OB)"
                
        if active_fvg:
            if bias == 'BULLISH' and active_fvg.direction == 'BULLISH':
                 long_score += 1.5
                 factors['confluence'] = factors.get('confluence', '') + " + FVG Support"
            elif bias == 'BEARISH' and active_fvg.direction == 'BEARISH':
                 short_score += 1.5
                 factors['confluence'] = factors.get('confluence', '') + " + FVG Resistance"
                 
        # D. ADX FILTER (Avoid Chop)
        adx = row.get('adx', 25)
        if adx < 20: 
            # In chop, penalize unless at extreme structure
            if not active_ob:
                long_score -= 2.0
                short_score -= 2.0
                
        # FINAL SIGNAL
        # Threshold: Bias(2) + Pullback(3) = 5.0 (Minimum for entry)
        threshold = 5.0
        
        direction = 'NEUTRAL'
        confidence = 0.0
        
        if long_score >= threshold and long_score > short_score:
            direction = 'LONG'
            confidence = min(1.0, long_score / 10.0)
        elif short_score >= threshold and short_score > long_score:
            direction = 'SHORT'
            confidence = min(1.0, short_score / 10.0)
            
        # Levels (Wider for Gold)
        atr = row.get('atr', price * 0.005)
        
        if direction == 'LONG':
            stop = price - (atr * 2.0)
            tp1 = price + (atr * 3.0) # Aim for 1.5R minimum
            tp2 = price + (atr * 5.0)
            tp3 = price + (atr * 8.0)
        elif direction == 'SHORT':
            stop = price + (atr * 2.0)
            tp1 = price - (atr * 3.0)
            tp2 = price - (atr * 5.0)
            tp3 = price - (atr * 8.0)
        else:
            stop = tp1 = tp2 = tp3 = price
            
        risk = abs(price - stop)
        reward = abs(tp1 - price)
        rr = reward / (risk + 1e-10)
        
        return OptimizedSignal(
            direction=direction,
            confidence=confidence,
            entry=price,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            kalman_velocity=kalman_pred.velocity,
            vwap_dist=0,
            ema_trend=0,
            volatility_state="NORMAL",
            risk_reward=rr,
            factors=factors
        )
