# NQ Quant Bot - Neural Scalper Strategy
# High-frequency, ML-enhanced scalping engine for 1m/5m timeframes.

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging

from .brain import QuantBrain
from .kalman_predict import KalmanPredictor
from .ml.xgboost_classifier import SignalClassifier, XGBOOST_AVAILABLE
from .order_blocks import detect_order_blocks, check_ob_interaction, OrderBlock

logger = logging.getLogger(__name__)

@dataclass
class ScalperConfig:
    """Configuration for Neural Scalper"""
    # Timeframe
    timeframe: str = "5m" # or "1m"
    
    # ML Weights
    min_ml_confidence: float = 0.52 # Lowered from 0.65 to get more trades
    
    # Momentum
    vel_threshold: float = 5.0 # Lowered from 15.0 to catch standard 5m moves
    use_micro_obs: bool = True
    
    # Risk (Tight Scalping)
    atr_stop_mult: float = 1.2
    atr_tp_mult: float = 2.5 # Aim for bigger runners
    risk_per_trade: float = 0.025 # Aggressive sizing

@dataclass
class ScalperSignal:
    direction: str # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float
    entry: float
    stop_loss: float
    take_profit: float
    factors: Dict[str, str]

class NeuralScalperStrategy:
    """
    Neural Scalper: High-Frequency ML Strategy.
    
    Logic:
    1. Instant Momentum: Kalman Velocity (1-bar) to detect bursts.
    2. Micro-Structure: Validates momentum against recent Order Blocks.
    3. Neural Filter: XGBoost classifier trained on 1m/5m price action.
    """
    
    def __init__(self, config: ScalperConfig = None):
        self.config = config or ScalperConfig()
        
        # Components
        self.kalman = KalmanPredictor(process_noise=0.01, measurement_noise=0.05) # Faster reaction
        self.classifier = SignalClassifier() if XGBOOST_AVAILABLE else None
        
        # State
        self.order_blocks: List[OrderBlock] = []
        self.is_trained = False
        
    def train(self, df: pd.DataFrame):
        """Train ML on high-frequency data"""
        if len(df) < 200:
            logger.warning("Not enough data for Neural Scalper training (need 200+)")
            return
            
        if self.classifier:
            try:
                # Target: Price move > 0.1% in next 3 bars (Scalp target)
                future_ret = df['close'].shift(-3) / df['close'] - 1
                threshold = 0.001 # 0.1%
                
                labels = pd.Series(1, index=df.index) # Default Neutral
                labels[future_ret > threshold] = 2 # Long
                labels[future_ret < -threshold] = 0 # Short
                
                # Train with these scalping labels
                # We need to hack the classifier trigger because it usually generates its own labels
                # But here we want customs. The classifier.train method does feature creation & fit.
                
                # For simplicity/speed, we'll let the classifier generate its own labels 
                # but we'll set the threshold parameter if we could. 
                # Since SignalClassifier.train takes df and generates labels internally, 
                # we rely on its default logic but passing a smaller threshold via a custom method or 
                # just relying on the default create_labels logic which uses 0.5%.
                # Let's use the standard train for now.
                self.classifier.train(df, forward_periods=3, threshold=0.001)
                
            except Exception as e:
                logger.warning(f"Neural Scalper ML Training failed: {e}")
        
        self.is_trained = True

    def evaluate(self, df: pd.DataFrame, idx: int) -> ScalperSignal:
        row = df.iloc[idx]
        price = row['close']
        
        # 1. Instant Momentum (Kalman)
        kalman_pred = self.kalman.update(price)
        velocity = kalman_pred.velocity
        
        factors = {}
        direction = 'NEUTRAL'
        confidence = 0.0
        
        # Velocity Trigger
        if velocity > self.config.vel_threshold:
            raw_dir = 'LONG'
            factors['vel'] = f"High Velocity (+{velocity:.1f})"
        elif velocity < -self.config.vel_threshold:
            raw_dir = 'SHORT'
            factors['vel'] = f"High Velocity ({velocity:.1f})"
        else:
            raw_dir = 'NEUTRAL'
            
        # 2a. VWAP Trend Filter (New)
        # Only take longs above VWAP, shorts below VWAP (Trend following scalping)
        vwap_dist = row.get('vwap_dist', 0) # Calculated in optimized_indicators
        # vwap_dist is % diff. Positive = Price > VWAP
        
        if raw_dir == 'LONG' and vwap_dist < -0.05: # Price below VWAP significantly
             raw_dir = 'NEUTRAL' # Don't buy below VWAP (unless huge mean rev, but this is trend scalp)
             factors['vwap'] = "Below VWAP filter"
        elif raw_dir == 'SHORT' and vwap_dist > 0.05:
             raw_dir = 'NEUTRAL'
             factors['vwap'] = "Above VWAP filter"
             
        # 2b. Micro-Structure (Order Blocks)
        if self.config.use_micro_obs and raw_dir != 'NEUTRAL':
            self._update_obs(df, idx)
            active_ob, interaction = check_ob_interaction(price, self.order_blocks)
            
            if active_ob and interaction in ['TOUCHED', 'INSIDE']:
                # Rejects trade if hitting opposing OB
                if raw_dir == 'LONG' and active_ob.direction == 'BEARISH':
                    raw_dir = 'NEUTRAL'
                    factors['ob'] = "Blocked by Bearish OB"
                elif raw_dir == 'SHORT' and active_ob.direction == 'BULLISH':
                    raw_dir = 'NEUTRAL'
                    factors['ob'] = "Blocked by Bullish OB"
                # Boost if bouncing off support/resistance
                elif raw_dir == 'LONG' and active_ob.direction == 'BULLISH':
                    confidence += 0.2
                    factors['ob'] = "Bounce off Bullish OB"
                elif raw_dir == 'SHORT' and active_ob.direction == 'BEARISH':
                    confidence += 0.2
                    factors['ob'] = "Rejection from Bearish OB"
        
        # 3. Neural Filter
        ml_conf = 0.5
        if self.classifier and self.is_trained:
            try:
                lookback = 60
                if idx >= lookback:
                    subset = df.iloc[idx-lookback : idx+1]
                    ml_pred = self.classifier.predict(subset)
                    
                    if ml_pred.signal == raw_dir:
                        # Confirm signal
                        ml_conf = ml_pred.probability
                        if ml_conf > self.config.min_ml_confidence:
                            direction = raw_dir
                            confidence += ml_conf
                            factors['ml'] = f"Confirmed ({ml_conf:.2f})"
                        else:
                             factors['ml'] = f"Weak Confirm ({ml_conf:.2f})"
                             direction = 'NEUTRAL' # Filter out weak ML confirmation
                    else:
                        # ML disagrees
                        direction = 'NEUTRAL'
                        factors['ml'] = f"Disagree ({ml_pred.signal})"
            except Exception:
                pass
        else:
            # Fallback if no ML
            if confidence > 0.1: # If OB confirmed
                direction = raw_dir
        
        # Risk Management
        atr = row.get('atr', price * 0.001)
        if direction == 'LONG':
            stop = price - (atr * self.config.atr_stop_mult)
            tp = price + (atr * self.config.atr_tp_mult)
        elif direction == 'SHORT':
            stop = price + (atr * self.config.atr_stop_mult)
            tp = price - (atr * self.config.atr_tp_mult)
        else:
            stop = tp = price
            
        return ScalperSignal(
            direction=direction,
            confidence=confidence,
            entry=price,
            stop_loss=stop,
            take_profit=tp,
            factors=factors
        )

    def _update_obs(self, df, idx):
        # Update rarely to save speed, or on every bar for scalping precision?
        # For scalping, every 10 bars is enough for 5m OBs
        if not self.order_blocks or idx % 10 == 0:
            if idx > 50:
                subset = df.iloc[idx-50:idx+1]
                self.order_blocks = detect_order_blocks(subset, lookback=10)
