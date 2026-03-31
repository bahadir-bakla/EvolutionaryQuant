import pandas as pd
import numpy as np
from dataclasses import dataclass
from nq_core.kalman_predict import KalmanPredictor

@dataclass
class NQKalmanTrendConfig:
    vel_threshold: float = 2.0  # Velocity needed to confirm trend acceleration
    atr_stop_mult: float = 2.0  # Wide stop for NQ noise
    atr_tp_mult_2: float = 4.0  # Large target for trend home-runs

class NQ_Kalman_Trend_Strategy:
    """
    Specifically designed for the HMM Trend Regime on Nasdaq.
    Uses a Zero-Lag Kalman Filter to detect true price acceleration,
    ignoring minor pullbacks and executing only when momentum is mathematically accelerating.
    """
    def __init__(self, config: NQKalmanTrendConfig = None):
        self.config = config or NQKalmanTrendConfig()
        self.predictors = {} # Dict of kalman predictors per symbol/timeframe, simple here
        self.kalman = KalmanPredictor(process_noise=0.01, measurement_noise=0.1) # Tuned slightly faster for NQ
        
    def evaluate(self, df: pd.DataFrame, idx: int):
        row = df.iloc[idx]
        close = row['close']
        atr = row.get('atr', 20)
        
        # Update Kalman Filter
        pred = self.kalman.update(close)
        
        class Sig: pass
        signal = Sig()
        signal.direction = 'NEUTRAL'
        signal.stop_loss = 0.0
        signal.take_profit_2 = 0.0
        
        # We need to wait for Kalman to warm up
        if idx < 20:
            return signal
            
        # Velocity Breakout Condition
        # If velocity is highly positive and price closes near its high
        is_strong_green = close > row['open'] and (row['high'] - close) < (atr * 0.2)
        is_strong_red = close < row['open'] and (close - row['low']) < (atr * 0.2)
        
        if pred.velocity > self.config.vel_threshold and is_strong_green:
            signal.direction = 'LONG'
            signal.stop_loss = close - (atr * self.config.atr_stop_mult)
            signal.take_profit_2 = close + (atr * self.config.atr_tp_mult_2)
            
        elif pred.velocity < -self.config.vel_threshold and is_strong_red:
            signal.direction = 'SHORT'
            signal.stop_loss = close + (atr * self.config.atr_stop_mult)
            signal.take_profit_2 = close - (atr * self.config.atr_tp_mult_2)
            
        return signal
