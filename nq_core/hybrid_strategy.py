# NQ Quant Bot - Hybrid Strategy (Kalman + ML)
# Combines the robust Kalman Filter base with ML confirmation and Regime Detection

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nq_core.brain import QuantBrain, BrainState, MarketRegime
from nq_core.ml.xgboost_classifier import SignalClassifier, XGBOOST_AVAILABLE
from nq_core.ml.hmm_regime import get_regime_detector, HMM_AVAILABLE
from nq_core.ml.position_sizing import AdaptivePositionSizer, PositionSize
from nq_core.order_blocks import OrderBlock, detect_order_blocks, check_ob_interaction, get_active_order_blocks

@dataclass
class HybridSignal:
    direction: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    regime: str
    ml_confirmation: str
    position_size: Optional[PositionSize]
    factors: Dict[str, str]

class HybridStrategy:
    """
    Hybrid Strategy: Kalman Filter Base + ML Enhancements
    
    Logic:
    1. Base Signal: QuantBrain (Kalman Velocity + Z-Score)
    2. Confirmation: XGBoost Classifier
    3. Regime Filter: HMM (Trend vs Chop)
    4. Sizing: Adaptive based on Regime & Confidence
    """
    
    def __init__(self, train_on_init: bool = True):
        # Base Component
        self.brain = QuantBrain(hurst_window=50)
        
        # ML Components
        self.classifier = SignalClassifier() if XGBOOST_AVAILABLE else None
        self.regime_detector = get_regime_detector(use_hmm=HMM_AVAILABLE)
        self.position_sizer = AdaptivePositionSizer(use_kelly=True)
        
        self.is_trained = False
        self.train_on_init = train_on_init
        
    def train(self, df: pd.DataFrame):
        """Train ML components"""
        if self.classifier:
            print("  Training XGBoost Classifier...")
            self.classifier.train(df)
            
        if hasattr(self.regime_detector, 'train'):
            print("  Training Regime Detector...")
            self.regime_detector.train(df)
            
        self.is_trained = True
        
    def evaluate(self, df: pd.DataFrame, idx: int, capital: float = 100000) -> HybridSignal:
        """Generate hybrid signal"""
        if not self.is_trained and self.train_on_init:
            pass 

        row = df.iloc[idx]
        price = row['close']
        timestamp = df.index[idx]
        
        # 1. Update Brain (Base Signal)
        brain_state = self.brain.update(price, timestamp)
        
        # Base Logic from QuantBrain/Confluence
        # Velocity threshold
        vel_threshold = price * 0.0001
        
        base_dir = "NEUTRAL"
        base_conf = 0.0
        factors = {}
        
        # === 3. REGIME FILTER (First Priority) ===
        regime = "UNKNOWN"
        if self.regime_detector and (getattr(self.regime_detector, 'is_trained', True)):
             reg_state = self.regime_detector.predict(df.iloc[:idx+1])
             regime = reg_state.current_regime.value

        # === 1. BASE SIGNAL GENERATION (Regime Dependent) ===
        
        # Scenario A: CHOPPY Market
        if regime == "CHOPPY":
            # STRICT FILTER: No Trend Following
            # Only strong Mean Reversion allowed
            
            if abs(brain_state.z_score) > 2.5: # Stricter Z-Score (was 2.0)
                if brain_state.z_score > 2.5:
                    base_dir = "SHORT"
                    base_conf = 0.7 
                    factors['mean_rev'] = f"Chop Extreme Overbought (Z={brain_state.z_score:.2f})"
                elif brain_state.z_score < -2.5:
                    base_dir = "LONG"
                    base_conf = 0.7
                    factors['mean_rev'] = f"Chop Extreme Oversold (Z={brain_state.z_score:.2f})"
            else:
                factors['regime_filter'] = "Choppy - Waiting for Extremes"
                
        # Scenario B: TRENDING Market (Bull/Bear)
        else:
            # Allow Trend Following (Velocity)
            if brain_state.kalman_velocity > vel_threshold:
                if regime == "BULL":
                    base_dir = "LONG"
                    base_conf = 0.8
                    factors['trend_align'] = "Bull Trend + Velocity"
                elif regime == "BEAR":
                    # Counter-trend velocity in Bear market - cautious
                    base_dir = "LONG"
                    base_conf = 0.6
                    factors['trend_counter'] = "Bear Market Correction"
                    
            elif brain_state.kalman_velocity < -vel_threshold:
                if regime == "BEAR":
                    base_dir = "SHORT"
                    base_conf = 0.8
                    factors['trend_align'] = "Bear Trend + Velocity"
                elif regime == "BULL":
                     base_dir = "SHORT"
                     base_conf = 0.6
                     factors['trend_counter'] = "Bull Market Pullback"
            
            # Allow Pullback Entries (Z-Score against trend)
            if base_dir == "NEUTRAL":
                if regime == "BULL" and -2.0 < brain_state.z_score < -1.0:
                    base_dir = "LONG"
                    base_conf = 0.75
                    factors['pullback'] = "Bull Flag Entry"
                elif regime == "BEAR" and 1.0 < brain_state.z_score < 2.0:
                    base_dir = "SHORT"
                    base_conf = 0.75
                    factors['pullback'] = "Bear Flag Entry"

        # === 2. ML CONFIRMATION ===
        ml_signal = "NEUTRAL"
        if self.classifier and self.classifier.is_trained:
             # Need a slice of df up to idx
             pred = self.classifier.predict(df.iloc[:idx+1])
             ml_signal = pred.signal
             
             if ml_signal == base_dir and base_dir != "NEUTRAL":
                 base_conf += 0.1
                 factors['ml_confirm'] = f"XGBoost Confirmed ({pred.probability:.1%})"
             elif ml_signal != "NEUTRAL" and ml_signal != base_dir:
                 # If we are in Trend Alignment, ignore weak ML counter-signals
                 if "trend_align" in factors and pred.probability < 0.7:
                     pass
                 else:
                     base_conf -= 0.2
                     factors['ml_conflict'] = f"XGBoost says {ml_signal}"

        # Threshold check
        final_dir = "NEUTRAL"
        if base_conf >= 0.65: # High threshold for quality
            final_dir = base_dir
            
        # 4. Levels & Sizing
        atr = row.get('atr', price * 0.01)
        
        stop_loss = price
        tp1 = tp2 = tp3 = price
        pos_size = None
        
        if final_dir == "LONG":
            stop_loss = price - (atr * 2.0)
            tp1 = price + (atr * 1.5)
            tp2 = price + (atr * 3.0) 
            tp3 = price + (atr * 5.0)
        elif final_dir == "SHORT":
            stop_loss = price + (atr * 2.0)
            tp1 = price - (atr * 1.5)
            tp2 = price - (atr * 3.0)
            tp3 = price - (atr * 5.0)
            
        if final_dir != "NEUTRAL":
            pos_size = self.position_sizer.calculate_position_size(
                capital=capital,
                signal=final_dir,
                confidence=base_conf,
                regime=regime,
                current_equity=capital, # Simplified for step-eval
                peak_equity=capital,
                current_atr=atr,
                avg_atr=atr, # approx
                entry_price=price,
                stop_loss=stop_loss
            )

        return HybridSignal(
            direction=final_dir,
            confidence=base_conf,
            entry=price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            regime=regime,
            ml_confirmation=ml_signal,
            position_size=pos_size,
            factors=factors
        )
