# Silver Bullet Strategy (Pure Momentum)
# Silver "siklemez" anything but Trend + Momentum.
# Logic: Kalman Velocity + ADX + RSI. No Order Blocks. No Fading.

import pandas as pd
from dataclasses import dataclass
from .kalman_predict import KalmanPredictor, get_kalman_signal
from .optimized_strategy import OptimizedSignal

@dataclass
class SilverConfig:
    # Trend thresholds
    min_adx: float = 25.0
    rsi_long_min: float = 50.0
    rsi_short_max: float = 50.0
    
    # Aggressive Stops for Momentum
    atr_stop_mult: float = 2.0
    atr_trail_mult: float = 1.5
    
    # Weights for Confidence
    weight_kalman: float = 5.0
    weight_adx: float = 3.0
    weight_rsi: float = 2.0

class SilverBulletStrategy:
    """
    Silver Bullet: Pure Momentum.
    "Trend is King".
    """
    
    def __init__(self, config: SilverConfig = None):
        self.config = config or SilverConfig()
        self.kalman = KalmanPredictor(process_noise=0.01, measurement_noise=0.1) # Tuned for Silver Volatility
        
    def evaluate(self, df: pd.DataFrame, idx: int) -> OptimizedSignal:
        row = df.iloc[idx]
        price = row['close']
        
        # 1. Indicators
        # Kalman
        pred = self.kalman.update(price)
        k_sig, k_weight = get_kalman_signal(self.kalman)
        kalman_vel = pred.velocity
        
        # ADX (Trend Strength)
        adx = row.get('adx', 0)
        
        # RSI (Momentum)
        rsi = row.get('rsi', 50)
        
        # 2. Logic
        long_score = 0.0
        short_score = 0.0
        factors = {}
        
        # Only trade if Trend is present
        if adx < self.config.min_adx:
            factors['adx'] = f"Chop ({adx:.1f})"
            # Penalize, but maybe less severe?
            long_score -= 2.0
            short_score -= 2.0
        else:
            factors['adx'] = f"Trending ({adx:.1f})"
            if k_sig == 'LONG':
                long_score += self.config.weight_adx
            elif k_sig == 'SHORT':
                short_score += self.config.weight_adx
                
        # Kalman Velocity
        if k_sig == 'LONG':
            long_score += self.config.weight_kalman * k_weight
            factors['kalman'] = f"LONG (Vel {kalman_vel:.2f})"
        elif k_sig == 'SHORT':
            short_score += self.config.weight_kalman * k_weight
            factors['kalman'] = f"SHORT (Vel {kalman_vel:.2f})"
            
        # RSI Confirmation
        if rsi > self.config.rsi_long_min:
            long_score += self.config.weight_rsi
            factors['rsi'] = f"Bullish ({rsi:.0f})"
        elif rsi < self.config.rsi_short_max:
            short_score += self.config.weight_rsi
            factors['rsi'] = f"Bearish ({rsi:.0f})"
            
        # 3. Decision
        min_score = 4.0 # Lowered from 6.0
        
        direction = 'NEUTRAL'
        confidence = 0.0
        
        if long_score >= min_score and long_score > short_score:
            direction = 'LONG'
            confidence = min(1.0, long_score / 10.0)
        elif short_score >= min_score and short_score > long_score:
            direction = 'SHORT'
            confidence = min(1.0, short_score / 10.0)
            
        # 4. Exits (Trend Following = Loose TP, Tightish Trail)
        atr = row.get('atr', price * 0.01)
        
        if direction == 'LONG':
            stop = price - (atr * self.config.atr_stop_mult)
            # Let winners run - wide TPs
            tp1 = price + (atr * 4.0) 
            tp2 = price + (atr * 8.0)
            tp3 = price + (atr * 12.0)
        elif direction == 'SHORT':
            stop = price + (atr * self.config.atr_stop_mult)
            tp1 = price - (atr * 4.0)
            tp2 = price - (atr * 8.0)
            tp3 = price - (atr * 12.0)
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
            volatility_state="NORMAL",
            risk_reward=rr,
            factors=factors
        )
