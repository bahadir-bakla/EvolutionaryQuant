# Silver Velocity Strategy (v2)
# Enhancing Silver Bullet with "Rocket Logic"
# If Momentum is strong (Velocity) AND increasing (Acceleration), we go Aggressive.

import pandas as pd
from dataclasses import dataclass
from .kalman_predict import KalmanPredictor, get_kalman_signal
from .optimized_strategy import OptimizedSignal

@dataclass
class SilverVelocityConfig:
    # Trend thresholds
    min_adx: float = 30.0 # Stricter Trend (Was 25)
    
    # Kalman Thresholds
    high_velocity_thresh: float = 0.5 # Stricter Rocket (Was implicit/lower)
    
    # Risk
    base_atr_stop: float = 1.5
    base_atr_tp: float = 1.5  # Was 3.0 — too far, trades reversed to SL
    
    # Aggression
    rocket_size_mult: float = 2.0 # More aggressive if we are right

class SilverVelocityStrategy:
    """
    Silver Velocity:
    1. Base: Trend Following (ADX + Kalman).
    2. Boost: If Acceleration aligns with Velocity -> "Rocket Mode".
       - Wider Targets
       - Tighter Trailing Stop (Parabolic protection) (Handled in runner config usually, but we signal it)
    """
    
    def __init__(self, config: SilverVelocityConfig = None):
        self.config = config or SilverVelocityConfig()
        self.kalman = KalmanPredictor(process_noise=0.01, measurement_noise=0.1) 
        
    def evaluate(self, df: pd.DataFrame, idx: int) -> OptimizedSignal:
        row = df.iloc[idx]
        price = row['close']
        
        # 1. Indicators
        pred = self.kalman.update(price)
        kalman_vel = pred.velocity
        kalman_acc = pred.acceleration
        
        adx = row.get('adx', 0)
        rsi = row.get('rsi', 50)
        
        # 2. Logic
        long_score = 0.0
        short_score = 0.0
        factors = {}
        
        # A. TREND METER (ADX)
        if adx < self.config.min_adx:
            factors['adx'] = f"Weak ({adx:.1f})"
            long_score -= 5.0
            short_score -= 5.0
        else:
            factors['adx'] = f"Strong ({adx:.1f})"
            if kalman_vel > 0: long_score += 2.0
            elif kalman_vel < 0: short_score += 2.0
            
        # B. VELOCITY CORE
        if kalman_vel > 0.1:
            long_score += 3.0 * abs(kalman_vel) * 5 # Scale score by velocity magnitude
            factors['vel'] = f"Positive ({kalman_vel:.3f})"
        elif kalman_vel < -0.1:
            short_score += 3.0 * abs(kalman_vel) * 5
            factors['vel'] = f"Negative ({kalman_vel:.3f})"
            
        # C. ACCELERATION BOOST (The "Rocket")
        is_rocket = False
        if kalman_vel > 0 and kalman_acc > 0.01:
             long_score += 3.0
             factors['acc'] = f"Accelerating (+{kalman_acc:.4f}) 🚀"
             is_rocket = True
        elif kalman_vel < 0 and kalman_acc < -0.01:
             short_score += 3.0
             factors['acc'] = f"Accelerating ({kalman_acc:.4f}) 🚀"
             is_rocket = True
             
        # D. RSI CHECK (Don't buy top tick unless Rocket)
        if rsi > 70 and not is_rocket:
             long_score -= 2.0
             factors['rsi'] = "OB (No Rocket)"
        elif rsi < 30 and not is_rocket:
             short_score -= 2.0
             factors['rsi'] = "OS (No Rocket)"
             
        # 3. Decision
        direction = 'NEUTRAL'
        confidence = 0.0
        score_thresh = 5.0
        
        if long_score >= score_thresh and long_score > short_score:
            direction = 'LONG'
            confidence = min(1.0, long_score / 15.0)
        elif short_score >= score_thresh and short_score > long_score:
            direction = 'SHORT'
            confidence = min(1.0, short_score / 15.0)
            
        # 4. Levels & Sizing
        atr = row.get('atr', price * 0.005)
        
        # Dynamic Risk Profile
        stop_mult = self.config.base_atr_stop
        tp_mult = self.config.base_atr_tp
        
        if is_rocket:
            # Rocket: wider targets but still reachable
            tp_mult *= 1.5  # Was 2.0 — made TP 6x ATR, unreachable
            factors['mode'] = "ROCKET (Extended TPs)"
            
        if direction == 'LONG':
            stop = price - (atr * stop_mult)
            tp1 = price + (atr * tp_mult)
            tp2 = price + (atr * tp_mult * 2.0)
            tp3 = price + (atr * tp_mult * 4.0)
        elif direction == 'SHORT':
            stop = price + (atr * stop_mult)
            tp1 = price - (atr * tp_mult)
            tp2 = price - (atr * tp_mult * 2.0)
            tp3 = price - (atr * tp_mult * 4.0)
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
            volatility_state="ACCELERATING" if is_rocket else "NORMAL",
            risk_reward=rr,
            factors=factors
        )
