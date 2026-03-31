# NQ Quant Bot - Hyper Hybrid Strategy
# Extends Super Hybrid with Advanced Quant Tools (Stochastics, POC, Volume Profile)

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging

from .super_hybrid import SuperHybridStrategy, SuperHybridConfig, SuperHybridSignal
from .brain import MarketRegime

logger = logging.getLogger(__name__)

@dataclass
class HyperConfig(SuperHybridConfig):
    """Configuration for Hyper Hybrid"""
    # New Weights
    weight_stoch: float = 1.5
    weight_poc: float = 2.0
    
    # Quant Thresholds
    stoch_overbought: float = 80.0
    stoch_oversold: float = 20.0
    poc_tolerance: float = 0.001 # 0.1% distance

class HyperHybridStrategy(SuperHybridStrategy):
    """
    Hyper Hybrid: Super Hybrid + Quant Tools.
    
    Adds:
    1. Stochastic Oscillator: Timing filter (don't buy top/sell bottom).
    2. POC (Point of Control): Volume profile support/resistance interaction.
    """
    
    def __init__(self, config: HyperConfig = None):
        super().__init__(config or HyperConfig())
        self.config: HyperConfig = config or HyperConfig()

    def evaluate(self, df: pd.DataFrame, idx: int) -> SuperHybridSignal:
        # Get base signal (Gold Logic + ML + OB)
        # Note: We can't just call super().evaluate() and modify it because 
        # super().evaluate() does the scoring internally and returns a final object.
        # We need to re-implement the scoring logic OR hook into it.
        # For full control, we'll re-implement the scoring phase but reuse components.
        
        # ... Or better, we inherit the heavy lifting but re-calculate score?
        # Let's copy the logic to ensure we integrate deep into the scoring matrix.
        # This is cleaner than monkey-patching the result.
        
        row = df.iloc[idx]
        price = row['close']
        timestamp = df.index[idx]
        
        # Predictors
        brain_state = self.brain.update(price, timestamp)
        kalman_pred = self.kalman.update(price)
        
        long_score = 0.0
        short_score = 0.0
        factors = {}
        
        # --- BASE LOGIC (Re-impl to add new factors) ---
        
        # 1. Kalman
        from .kalman_predict import get_kalman_signal
        k_sig, k_weight = get_kalman_signal(self.kalman)
        if k_sig == 'LONG':
            long_score += self.config.weight_kalman * k_weight
            factors['kalman'] = f"LONG (v={kalman_pred.velocity:.2f})"
        elif k_sig == 'SHORT':
            short_score += self.config.weight_kalman * k_weight
            factors['kalman'] = f"SHORT (v={kalman_pred.velocity:.2f})"
            
        # 2. VWAP
        vwap_dist = row.get('vwap_dist', 0)
        if vwap_dist > 0.2:
            long_score += self.config.weight_vwap
            factors['vwap'] = f"LONG (+{vwap_dist:.2f}%)"
        elif vwap_dist < -0.2:
            short_score += self.config.weight_vwap
            factors['vwap'] = f"SHORT ({vwap_dist:.2f}%)"
            
        # 3. EMA
        ema_stack = row.get('ema_stack', 0)
        if ema_stack == 2:
            long_score += self.config.weight_ema
            factors['ema'] = "LONG (Stacked)"
        elif ema_stack == -2:
            short_score += self.config.weight_ema
            factors['ema'] = "SHORT (Stacked)"
            
        # 4. Momentum
        mom_5d = row.get('return_5d', 0)
        if mom_5d > 1.0:
            long_score += self.config.weight_momentum
            factors['mom'] = f"LONG (+{mom_5d:.2f}%)"
        elif mom_5d < -1.0:
            short_score += self.config.weight_momentum
            factors['mom'] = f"SHORT ({mom_5d:.2f}%)"

        # --- NEW QUANT LOGIC ---
        
        # 5. Stochastic Oscillator
        stoch_k = row.get('stoch_k', 50)
        
        # Timing Penalty: If buying at overbought, reduce score
        if k_sig == 'LONG' and stoch_k > self.config.stoch_overbought:
             long_score -= 1.5 # Heavy penalty
             factors['stoch'] = "Overbought (Penalty)"
        elif k_sig == 'SHORT' and stoch_k < self.config.stoch_oversold:
             short_score -= 1.5
             factors['stoch'] = "Oversold (Penalty)"
             
        # Timing Boost: Crossing up from oversold
        if stoch_k < 30 and k_sig == 'LONG':
             long_score += self.config.weight_stoch
             factors['stoch'] = "LONG (Oversold Buy)"
        elif stoch_k > 70 and k_sig == 'SHORT':
             short_score += self.config.weight_stoch
             factors['stoch'] = "SHORT (Overbought Sell)"
             
        # 6. POC Interaction
        poc = row.get('poc_level', 0)
        if poc > 0:
            dist_to_poc = (price - poc) / poc
            if abs(dist_to_poc) < self.config.poc_tolerance:
                 # At POC
                 if k_sig == 'LONG' and brain_state.regime == MarketRegime.TRENDING_BULLISH:
                     long_score += 1.5
                     factors['poc'] = "LONG (Bounce off POC)"
                 elif k_sig == 'SHORT' and brain_state.regime == MarketRegime.TRENDING_BEARISH:
                     short_score += 1.5
                     factors['poc'] = "SHORT (Reject off POC)"

        # --- ML & OB ---
        
        # Regime
        regime = "UNKNOWN"
        if self.regime_detector and self.is_trained:
            try:
                lookback = 50 
                if idx >= lookback:
                    subset = df.iloc[idx-lookback : idx+1]
                    regime_state = self.regime_detector.predict(subset)
                    regime = regime_state.current_regime.value
                    if regime == "CHOPPY":
                        # Reduce Trend scores
                        if 'ema' in factors:
                             if factors['ema'].startswith('LONG'): long_score -= 1.0
                             else: short_score -= 1.0
                             factors['regime'] = "Choppy (Trend Reduced)"
            except: pass
            
        # Order Blocks
        self.order_blocks = self._update_order_blocks(df, idx)
        from .order_blocks import check_ob_interaction
        active_ob, interaction = check_ob_interaction(price, self.order_blocks)
        if active_ob and interaction in ['TOUCHED', 'INSIDE']:
            if active_ob.direction == 'BULLISH':
                long_score += self.config.weight_ob
                factors['ob'] = "LONG (Bullish OB)"
            elif active_ob.direction == 'BEARISH':
                short_score += self.config.weight_ob
                factors['ob'] = "SHORT (Bearish OB)"
                
        # ML Filter
        ml_conf = 0.5
        if self.classifier and self.is_trained:
            try:
                if idx >= 60:
                    subset = df.iloc[idx-60 : idx+1]
                    ml_pred = self.classifier.predict(subset)
                    ml_conf = ml_pred.probability
                    
                    if ml_pred.signal == 'LONG' and ml_conf > 0.6:
                        long_score += self.config.weight_ml
                        factors['ml'] = f"LONG (Prob {ml_conf:.2f})"
                    elif ml_pred.signal == 'SHORT' and ml_conf > 0.6:
                        short_score += self.config.weight_ml
                        factors['ml'] = f"SHORT (Prob {ml_conf:.2f})"
            except: pass
            
        # Final Signal
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
            ml_confidence=ml_conf,
            hmm_regime=str(regime),
            active_ob=active_ob,
            ob_confluence=0.0
        )
