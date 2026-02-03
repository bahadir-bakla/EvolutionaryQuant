# Adaptive Kalman Filter
# Hurst Exponent'e göre dinamik R/Q ayarlı 2-State Kalman

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from filterpy.kalman import KalmanFilter
import logging

logger = logging.getLogger(__name__)

@dataclass
class KalmanState:
    """Kalman filter output state"""
    price: float          # Smoothed price (position)
    velocity: float       # Price velocity (momentum)
    future_prices: List[float]  # Future predictions
    r_value: float        # Current R (measurement noise)
    q_value: float        # Current Q (process noise)


class AdaptiveKalman:
    """
    2-State Adaptive Kalman Filter
    
    State Vector: [price, velocity]
    - price: Mevcut fiyat tahmini
    - velocity: Fiyatın değişim hızı (momentum göstergesi)
    
    Adaptasyon:
    - Hurst > 0.6 (Trend): R düşür -> Fiyata hızlı tepki
    - Hurst < 0.4 (Choppy): R artır -> Gürültüyü filtrele
    """
    
    def __init__(
        self,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-3,
        trending_r_mult: float = 0.1,
        choppy_r_mult: float = 10.0
    ):
        self.base_q = process_noise
        self.base_r = measurement_noise
        self.trending_r_mult = trending_r_mult
        self.choppy_r_mult = choppy_r_mult
        
        # FilterPy Kalman Filter
        self.kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # State transition matrix (physics model: x = x0 + v*dt)
        self.kf.F = np.array([
            [1., 1.],  # position = prev_position + velocity
            [0., 1.]   # velocity = prev_velocity (constant velocity model)
        ])
        
        # Measurement function (we only measure price)
        self.kf.H = np.array([[1., 0.]])
        
        # Initial covariance
        self.kf.P *= 1000.
        
        # Process noise
        self.kf.Q = np.eye(2) * self.base_q
        
        # Measurement noise
        self.kf.R = np.array([[self.base_r]])
        
        # Initialize state
        self.kf.x = np.array([[0.], [0.]])
        
        self.initialized = False
        self.prices_history: List[float] = []
        
    def reset(self, initial_price: float):
        """Reset filter with initial price"""
        self.kf.x = np.array([[initial_price], [0.]])
        self.kf.P *= 1000.
        self.initialized = True
        self.prices_history = [initial_price]
        
    def adapt_to_regime(self, hurst_value: float, regime: str):
        """
        Rejime göre Kalman parametrelerini ayarla
        
        Args:
            hurst_value: 0-1 arası Hurst exponent
            regime: TRENDING, CHOPPY, NEUTRAL
        """
        if regime == "TRENDING":
            # Trend var: Fiyata hızlı tepki ver
            self.kf.R = np.array([[self.base_r * self.trending_r_mult]])
            self.kf.Q = np.eye(2) * (self.base_q * 2)  # Daha hızlı adaptasyon
            logger.debug(f"Kalman adapted to TRENDING: R={self.kf.R[0,0]:.6f}")
            
        elif regime == "CHOPPY":
            # Yatay/Testere: Gürültüyü filtrele
            self.kf.R = np.array([[self.base_r * self.choppy_r_mult]])
            self.kf.Q = np.eye(2) * (self.base_q * 0.5)  # Daha yavaş değişim
            logger.debug(f"Kalman adapted to CHOPPY: R={self.kf.R[0,0]:.6f}")
            
        else:  # NEUTRAL
            self.kf.R = np.array([[self.base_r]])
            self.kf.Q = np.eye(2) * self.base_q
            
    def update(self, price: float) -> KalmanState:
        """
        Yeni fiyat ile Kalman filter güncelle
        
        Args:
            price: Yeni gözlemlenen fiyat
            
        Returns:
            KalmanState: Güncel durum
        """
        if not self.initialized:
            self.reset(price)
            
        # Predict step
        self.kf.predict()
        
        # Update step
        self.kf.update(price)
        
        self.prices_history.append(price)
        
        # Extract state
        estimated_price = self.kf.x[0, 0]
        velocity = self.kf.x[1, 0]
        
        # Future predictions
        future_prices = self._predict_future(steps=3)
        
        return KalmanState(
            price=estimated_price,
            velocity=velocity,
            future_prices=future_prices,
            r_value=self.kf.R[0, 0],
            q_value=self.kf.Q[0, 0]
        )
    
    def _predict_future(self, steps: int = 3) -> List[float]:
        """Gelecek fiyat tahmini"""
        predictions = []
        state = self.kf.x.copy()
        
        for _ in range(steps):
            state = self.kf.F @ state
            predictions.append(state[0, 0])
            
        return predictions
    
    def batch_process(
        self, 
        prices: np.ndarray,
        hurst_values: Optional[np.ndarray] = None,
        regimes: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch fiyat işleme
        
        Args:
            prices: Fiyat dizisi
            hurst_values: (optional) Her bar için Hurst değeri
            regimes: (optional) Her bar için rejim
            
        Returns:
            Tuple[smoothed_prices, velocities]
        """
        n = len(prices)
        smoothed = np.zeros(n)
        velocities = np.zeros(n)
        
        self.reset(prices[0])
        
        for i, price in enumerate(prices):
            # Dinamik adaptasyon (eğer hurst verilmişse)
            if hurst_values is not None and regimes is not None:
                if i > 0:  # İlk bardan sonra
                    self.adapt_to_regime(hurst_values[i], regimes[i])
            
            state = self.update(price)
            smoothed[i] = state.price
            velocities[i] = state.velocity
            
        return smoothed, velocities


# === TEST ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Simüle data: Önce yatay, sonra trend
    np.random.seed(42)
    t = np.linspace(0, 20, 400)
    
    # Choppy phase (ilk 200 bar)
    choppy = np.sin(t[:200]) * 3 + 100 + np.random.randn(200) * 0.8
    
    # Trending phase (son 200 bar)
    trend = np.linspace(100, 130, 200) + np.random.randn(200) * 0.8
    
    prices = np.concatenate([choppy, trend])
    
    # Basit rejim simülasyonu
    regimes = ["CHOPPY"] * 200 + ["TRENDING"] * 200
    hurst_vals = np.array([0.35] * 200 + [0.7] * 200)
    
    # Kalman uygula
    kalman = AdaptiveKalman()
    smoothed, velocities = kalman.batch_process(prices, hurst_vals, regimes)
    
    # Plot
    plt.figure(figsize=(14, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(prices, label='Raw Price', alpha=0.5)
    plt.plot(smoothed, label='Kalman Smoothed', linewidth=2)
    plt.axvline(200, color='red', linestyle='--', alpha=0.5, label='Regime Change')
    plt.legend()
    plt.title('Adaptive Kalman Filter')
    
    plt.subplot(2, 1, 2)
    plt.plot(velocities, label='Velocity (Momentum)', color='purple')
    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.legend()
    plt.title('Price Velocity')
    
    plt.tight_layout()
    plt.show()
