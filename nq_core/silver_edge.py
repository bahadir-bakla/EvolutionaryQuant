# Silver Edge Strategy (V3)
# 5-Layer Intelligence Architecture for Silver Trading
# Layers: HTF Bias | Gold Correlation | Session Filter | Enhanced Kalman | Dynamic Risk

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from .kalman_predict import KalmanPredictor
from .optimized_strategy import OptimizedSignal


@dataclass
class SilverEdgeConfig:
    # Layer 1: HTF Bias
    htf_ema_period: int = 50          # 4H EMA period for trend direction
    
    # Layer 2: Gold Correlation
    gold_lookback: int = 3            # Check last N bars of Gold
    use_gold_filter: bool = True
    
    # Layer 3: Session Filter
    use_session_filter: bool = True
    
    # Layer 4: Kalman + Structure
    min_adx: float = 25.0             # Relaxed from 30 (V2 was too strict)
    min_body_ratio: float = 0.3       # Candle body vs range (filter dojis)
    divergence_lookback: int = 5      # Bars to check for divergence
    
    # Layer 5: Dynamic Risk
    base_atr_stop: float = 1.5        # SL: UNCHANGED
    base_atr_tp: float = 1.5          # Base TP (from V2 optimization)
    
    # Scoring
    score_threshold: float = 5.0


class SilverEdgeStrategy:
    """
    Silver Edge V3: 5-Layer Decision Engine
    
    Layer 1 - HTF Bias:     4H EMA trend → Only trade WITH the trend
    Layer 2 - Gold Filter:  Gold direction must confirm Silver direction
    Layer 3 - Session:      Avoid low-vol Asian hours, favor NY/London
    Layer 4 - Enhanced Kalman: Velocity + Acceleration + Divergence + Candle Structure
    Layer 5 - Dynamic Risk: Confidence-based TP/Lot scaling
    """
    
    def __init__(self, config: SilverEdgeConfig = None):
        self.config = config or SilverEdgeConfig()
        self.kalman = KalmanPredictor(process_noise=0.01, measurement_noise=0.1)
        
        # Divergence tracking
        self._vel_history: List[float] = []
        self._price_history: List[float] = []
        
    def evaluate(
        self, 
        df: pd.DataFrame, 
        idx: int,
        htf_df: Optional[pd.DataFrame] = None,    # 4H data for HTF bias
        gold_df: Optional[pd.DataFrame] = None     # Gold data for correlation
    ) -> OptimizedSignal:
        
        row = df.iloc[idx]
        price = row['close']
        open_price = row['open']
        high = row['high']
        low = row['low']
        
        # ============================================================
        # LAYER 4: Enhanced Kalman (run first to get base signals)
        # ============================================================
        pred = self.kalman.update(price)
        kalman_vel = pred.velocity
        kalman_acc = pred.acceleration
        
        adx = row.get('adx', 0)
        rsi = row.get('rsi', 50)
        
        # Track history for divergence
        self._vel_history.append(kalman_vel)
        self._price_history.append(price)
        if len(self._vel_history) > 20:
            self._vel_history.pop(0)
            self._price_history.pop(0)
        
        long_score = 0.0
        short_score = 0.0
        factors = {}
        
        # --- 4A. ADX Trend Gate (Relaxed from V2) ---
        if adx < self.config.min_adx:
            factors['adx'] = f"Weak ({adx:.1f})"
            long_score -= 3.0
            short_score -= 3.0
        else:
            factors['adx'] = f"Strong ({adx:.1f})"
            # Progressive bonus: ADX 25=+1, ADX 40=+3, ADX 60=+5
            adx_bonus = min(5.0, (adx - 20) / 8.0)
            if kalman_vel > 0: 
                long_score += adx_bonus
            elif kalman_vel < 0: 
                short_score += adx_bonus
                
        # --- 4B. Velocity Core ---
        if kalman_vel > 0.1:
            vel_score = 3.0 * min(abs(kalman_vel) * 5, 3.0)  # Capped
            long_score += vel_score
            factors['vel'] = f"Positive ({kalman_vel:.3f})"
        elif kalman_vel < -0.1:
            vel_score = 3.0 * min(abs(kalman_vel) * 5, 3.0)
            short_score += vel_score
            factors['vel'] = f"Negative ({kalman_vel:.3f})"
            
        # --- 4C. Acceleration Boost ---
        is_rocket = False
        if kalman_vel > 0 and kalman_acc > 0.01:
            long_score += 3.0
            factors['acc'] = f"Accelerating (+{kalman_acc:.4f}) 🚀"
            is_rocket = True
        elif kalman_vel < 0 and kalman_acc < -0.01:
            short_score += 3.0
            factors['acc'] = f"Accelerating ({kalman_acc:.4f}) 🚀"
            is_rocket = True
            
        # --- 4D. Kalman Divergence Detection ---
        lb = self.config.divergence_lookback
        if len(self._vel_history) >= lb:
            price_change = price - self._price_history[-lb]
            vel_change = kalman_vel - self._vel_history[-lb]
            
            # Bearish Divergence: Price up but momentum fading
            if price_change > 0 and vel_change < -0.05:
                long_score -= 3.0
                factors['div'] = "Bear Div ⚠️"
            # Bullish Divergence: Price down but momentum building
            elif price_change < 0 and vel_change > 0.05:
                short_score -= 3.0
                factors['div'] = "Bull Div ⚠️"
                
        # --- 4E. Candle Body Ratio ---
        candle_range = high - low
        body = abs(price - open_price)
        body_ratio = body / candle_range if candle_range > 0 else 0
        
        if body_ratio < self.config.min_body_ratio:
            # Doji / Indecision → weaken signal
            long_score -= 1.0
            short_score -= 1.0
            factors['candle'] = f"Doji ({body_ratio:.2f})"
        elif body_ratio > 0.7:
            # Strong candle → boost confidence
            if price > open_price:
                long_score += 1.5
            else:
                short_score += 1.5
            factors['candle'] = f"Strong ({body_ratio:.2f})"
            
        # --- 4F. RSI Filter ---
        if rsi > 70 and not is_rocket:
            long_score -= 2.0
            factors['rsi'] = f"OB ({rsi:.0f})"
        elif rsi < 30 and not is_rocket:
            short_score -= 2.0
            factors['rsi'] = f"OS ({rsi:.0f})"
            
        # ============================================================
        # LAYER 1: HTF Bias (4H EMA Trend)
        # ============================================================
        htf_bias = 'NEUTRAL'
        if htf_df is not None and len(htf_df) >= self.config.htf_ema_period:
            ema = htf_df['close'].ewm(span=self.config.htf_ema_period, adjust=False).mean().iloc[-1]
            htf_price = htf_df['close'].iloc[-1]
            
            if htf_price > ema * 1.001:  # 0.1% above = UP
                htf_bias = 'UP'
                short_score -= 3.0  # Penalize contra-trend
                factors['htf'] = f"4H UP (> EMA{self.config.htf_ema_period})"
            elif htf_price < ema * 0.999:  # 0.1% below = DOWN
                htf_bias = 'DOWN'
                long_score -= 3.0   # Penalize contra-trend
                factors['htf'] = f"4H DOWN (< EMA{self.config.htf_ema_period})"
            else:
                htf_bias = 'FLAT'
                long_score -= 1.0
                short_score -= 1.0
                factors['htf'] = "4H FLAT"
        else:
            factors['htf'] = "No HTF Data"
        
        # ============================================================
        # LAYER 2: Gold-Silver Correlation Filter
        # ============================================================
        if self.config.use_gold_filter and gold_df is not None and len(gold_df) >= self.config.gold_lookback + 1:
            gold_now = gold_df['close'].iloc[-1]
            gold_prev = gold_df['close'].iloc[-self.config.gold_lookback - 1]
            gold_change = (gold_now - gold_prev) / gold_prev
            
            if gold_change > 0.001:  # Gold rising
                gold_dir = 'UP'
                short_score -= 2.0  # Don't short Silver when Gold rises
                factors['gold'] = f"Gold UP (+{gold_change:.3%})"
            elif gold_change < -0.001:  # Gold falling
                gold_dir = 'DOWN'
                long_score -= 2.0   # Don't long Silver when Gold falls
                factors['gold'] = f"Gold DOWN ({gold_change:.3%})"
            else:
                factors['gold'] = "Gold FLAT"
        else:
            factors['gold'] = "No Gold Data"
        
        # ============================================================
        # LAYER 3: Session Filter
        # ============================================================
        if self.config.use_session_filter:
            bar_time = df.index[idx]
            hour = bar_time.hour if hasattr(bar_time, 'hour') else 12
            
            if 0 <= hour < 7:      # Asian
                long_score -= 3.0
                short_score -= 3.0
                factors['session'] = "Asian (Low Vol) ⛔"
            elif 7 <= hour < 13:   # London
                long_score += 1.5
                short_score += 1.5
                factors['session'] = "London ✅"
            elif 13 <= hour < 20:  # NY
                long_score += 2.0
                short_score += 2.0
                factors['session'] = "NY ✅✅"
            else:                   # Late NY
                factors['session'] = "Late NY"
        
        # ============================================================
        # DECISION
        # ============================================================
        direction = 'NEUTRAL'
        confidence = 0.0
        
        if long_score >= self.config.score_threshold and long_score > short_score:
            direction = 'LONG'
            confidence = min(1.0, long_score / 15.0)
        elif short_score >= self.config.score_threshold and short_score > long_score:
            direction = 'SHORT'
            confidence = min(1.0, short_score / 15.0)
            
        # ============================================================
        # LAYER 5: Dynamic Risk Manager
        # ============================================================
        atr = row.get('atr', price * 0.005)
        stop_mult = self.config.base_atr_stop  # FIXED — never changes
        
        # TP scales with confidence
        if confidence >= 0.7:
            tp_mult = self.config.base_atr_tp * 1.5   # High confidence → wider TP
            factors['risk'] = f"Aggressive (conf={confidence:.2f})"
        elif confidence >= 0.4:
            tp_mult = self.config.base_atr_tp           # Normal
            factors['risk'] = f"Normal (conf={confidence:.2f})"
        else:
            tp_mult = self.config.base_atr_tp * 0.75    # Low confidence → tight TP
            factors['risk'] = f"Conservative (conf={confidence:.2f})"
        
        if is_rocket:
            tp_mult *= 1.3  # Rocket bonus (controlled)
            factors['mode'] = "ROCKET 🚀"
        
        # Calculate levels
        if direction == 'LONG':
            stop = price - (atr * stop_mult)
            tp1 = price + (atr * tp_mult)
            tp2 = price + (atr * tp_mult * 2.0)
            tp3 = price + (atr * tp_mult * 3.0)
        elif direction == 'SHORT':
            stop = price + (atr * stop_mult)
            tp1 = price - (atr * tp_mult)
            tp2 = price - (atr * tp_mult * 2.0)
            tp3 = price - (atr * tp_mult * 3.0)
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
            kalman_velocity=kalman_vel,
            vwap_dist=0,
            ema_trend=0,
            volatility_state="ROCKET" if is_rocket else htf_bias,
            risk_reward=rr,
            factors=factors
        )
