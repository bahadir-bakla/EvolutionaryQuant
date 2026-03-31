# Super Hybrid Strategy - Restored Gold Master
# Combines Kalman filtering, VWAP, EMA stacking, Momentum, and Order Blocks into a single scoring system.

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import logging

from .brain import QuantBrain, MarketRegime, BrainState
from .kalman_predict import KalmanPredictor, get_kalman_signal
from .optimized_strategy import OptimizedSignal
from .order_blocks import OrderBlock, check_ob_interaction, detect_order_blocks

logger = logging.getLogger(__name__)

@dataclass
class SuperHybridConfig:
    """Configuration for Super Hybrid Strategy"""
    # Base Weights
    weight_kalman: float = 3.0
    weight_vwap: float = 2.0
    weight_ema: float = 2.0
    weight_momentum: float = 1.5
    
    # Confluence Weights
    weight_ob: float = 1.5
    weight_rsi: float = 1.0
    
    # Thresholds
    min_score_long: float = 4.0
    min_score_short: float = 4.0
    
    # Risk
    atr_stop_mult: float = 1.5
    atr_tp_mult_1: float = 2.0
    atr_tp_mult_2: float = 4.0

@dataclass
class SuperHybridSignal(OptimizedSignal):
    """Extended signal with OB info"""
    active_ob: Optional[OrderBlock] = None
    ob_confluence: float = 0.0

class SuperHybridStrategy:
    """
    The original "Super Hybrid" Gold Algorithmic Master.
    Scores entries based on dynamic Kalman calculations, VWAP dist,
    Trend Stack, Momentum, and structural Order Blocks.
    """
    def __init__(self, config: SuperHybridConfig = None):
        self.config = config or SuperHybridConfig()
        
        # Core Predictors
        self.kalman = KalmanPredictor(process_noise=0.005, measurement_noise=0.05)
        self.brain = QuantBrain(hurst_window=100)
        
        # State
        self.order_blocks: List[OrderBlock] = []

    def evaluate(self, df: pd.DataFrame, idx: int) -> SuperHybridSignal:
        row = df.iloc[idx]
        price = row['close']
        timestamp = df.index[idx]
        
        # Update State
        brain_state = self.brain.update(price, timestamp)
        kalman_pred = self.kalman.update(price)
        
        long_score = 0.0
        short_score = 0.0
        factors = {}
        
        # a. Kalman
        k_sig, k_weight = get_kalman_signal(self.kalman)
        if k_sig == 'LONG':
            long_score += self.config.weight_kalman * k_weight
            factors['kalman'] = f"LONG (v={kalman_pred.velocity:.2f})"
        elif k_sig == 'SHORT':
            short_score += self.config.weight_kalman * k_weight
            factors['kalman'] = f"SHORT (v={kalman_pred.velocity:.2f})"
            
        # b. VWAP
        vwap_dist = row.get('vwap_dist', 0)
        if vwap_dist > 0.2:
            long_score += self.config.weight_vwap
            factors['vwap'] = f"LONG (+{vwap_dist:.2f}%)"
        elif vwap_dist < -0.2:
            short_score += self.config.weight_vwap
            factors['vwap'] = f"SHORT ({vwap_dist:.2f}%)"
            
        # c. EMA Stack
        ema_stack = row.get('ema_stack', 0)
        if ema_stack == 2:
            long_score += self.config.weight_ema
            factors['ema'] = "LONG (Stacked)"
        elif ema_stack == -2:
            short_score += self.config.weight_ema
            factors['ema'] = "SHORT (Stacked)"
            
        # d. Momentum
        mom_5d = row.get('return_5d', 0)
        if mom_5d > 1.0:
            long_score += self.config.weight_momentum
            factors['mom'] = f"LONG (+{mom_5d:.2f}%)"
        elif mom_5d < -1.0:
            short_score += self.config.weight_momentum
            factors['mom'] = f"SHORT ({mom_5d:.2f}%)"
                
        # e. Order Blocks & RSI
        self.order_blocks = self._update_order_blocks(df, idx)
        
        active_ob, interaction = check_ob_interaction(price, self.order_blocks)
        if active_ob and interaction in ['TOUCHED', 'INSIDE']:
            if active_ob.direction == 'BULLISH':
                long_score += self.config.weight_ob
                factors['ob'] = f"LONG (Bullish OB {active_ob.low}-{active_ob.high})"
            elif active_ob.direction == 'BEARISH':
                short_score += self.config.weight_ob
                factors['ob'] = f"SHORT (Bearish OB {active_ob.low}-{active_ob.high})"
        
        rsi = row.get('rsi', 50)
        if rsi < 30:
            long_score += self.config.weight_rsi
            factors['rsi'] = f"LONG (Oversold {rsi:.0f})"
        elif rsi > 70:
            short_score += self.config.weight_rsi
            factors['rsi'] = f"SHORT (Overbought {rsi:.0f})"
            
        # Final Decision
        params = self.config
        if long_score >= params.min_score_long and long_score > short_score:
            direction = 'LONG'
            confidence = min(1.0, long_score / 15.0)
        elif short_score >= params.min_score_short and short_score > long_score:
            direction = 'SHORT'
            confidence = min(1.0, short_score / 15.0)
        else:
            direction = 'NEUTRAL'
            confidence = 0.0
            
        # Levels
        atr = row.get('atr', price * 0.002)
        if direction == 'LONG':
            stop = price - (atr * params.atr_stop_mult)
            tp1 = price + (atr * params.atr_tp_mult_1)
            tp2 = price + (atr * params.atr_tp_mult_2)
            tp3 = price + (atr * params.atr_tp_mult_2 * 1.5)
        elif direction == 'SHORT':
            stop = price + (atr * params.atr_stop_mult)
            tp1 = price - (atr * params.atr_tp_mult_1)
            tp2 = price - (atr * params.atr_tp_mult_2)
            tp3 = price - (atr * params.atr_tp_mult_2 * 1.5)
        else:
            stop = tp1 = tp2 = tp3 = price
            
        risk = abs(price - stop)
        reward = abs(tp1 - price)
        rr = reward / (risk + 1e-10)

        return SuperHybridSignal(
            direction=direction,
            confidence=confidence,
            entry=price,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            kalman_velocity=kalman_pred.velocity,
            vwap_dist=vwap_dist,
            ema_trend=ema_stack,
            volatility_state="NORMAL",
            risk_reward=rr,
            factors=factors,
            active_ob=active_ob,
            ob_confluence=0.0
        )

    def _update_order_blocks(self, df: pd.DataFrame, current_idx: int) -> List[OrderBlock]:
        if not self.order_blocks or current_idx % 50 == 0:
             subset = df.iloc[:current_idx+1]
             self.order_blocks = detect_order_blocks(subset, lookback=20)
        return self.order_blocks
