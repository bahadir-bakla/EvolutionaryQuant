# QuantBrain - Market Regime Detection & Decision Engine
# Kalman + Hurst birleştiren ana karar merkezi

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging

from .hurst import calculate_hurst, HurstResult
from .kalman import AdaptiveKalman, KalmanState

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Piyasa rejimi"""
    TRENDING_BULLISH = "TRENDING_BULLISH"
    TRENDING_BEARISH = "TRENDING_BEARISH"
    CHOPPY = "CHOPPY"
    NEUTRAL = "NEUTRAL"


@dataclass
class BrainState:
    """QuantBrain output state"""
    timestamp: Optional[pd.Timestamp] = None
    price: float = 0.0
    
    # Kalman outputs
    kalman_price: float = 0.0
    kalman_velocity: float = 0.0
    future_predictions: List[float] = field(default_factory=list)
    
    # Hurst outputs
    hurst_value: float = 0.5
    hurst_regime: str = "NEUTRAL"
    hurst_confidence: float = 0.0
    
    # Z-Score (fiyatın Kalman'dan sapması)
    z_score: float = 0.0
    
    # Final regime
    regime: MarketRegime = MarketRegime.NEUTRAL
    
    # Trading bias
    bias: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    bias_strength: float = 0.0  # 0-1


class QuantBrain:
    """
    Ana Karar Merkezi
    
    Görevler:
    1. Hurst ile rejim tespit
    2. Kalman'ı rejime göre adapte et
    3. Z-Score sapma hesapla
    4. Trading bias belirle
    """
    
    def __init__(
        self,
        hurst_window: int = 100,
        z_score_window: int = 20,
        trending_threshold: float = 0.6,
        choppy_threshold: float = 0.4
    ):
        self.hurst_window = hurst_window
        self.z_score_window = z_score_window
        self.trending_threshold = trending_threshold
        self.choppy_threshold = choppy_threshold
        
        # Components
        self.kalman = AdaptiveKalman()
        
        # History
        self.prices: List[float] = []
        self.kalman_prices: List[float] = []
        self.states: List[BrainState] = []
        
    def reset(self):
        """Reset brain state"""
        self.prices = []
        self.kalman_prices = []
        self.states = []
        self.kalman = AdaptiveKalman()
        
    def update(self, price: float, timestamp: Optional[pd.Timestamp] = None) -> BrainState:
        """
        Yeni fiyat ile brain'i güncelle
        
        Args:
            price: Yeni fiyat
            timestamp: Zaman damgası
            
        Returns:
            BrainState: Güncel durum
        """
        self.prices.append(price)
        
        # 1. Hurst Exponent hesapla
        hurst_result = self._calculate_hurst()
        
        # 2. Kalman'ı rejime göre adapte et
        self.kalman.adapt_to_regime(hurst_result.value, hurst_result.regime)
        
        # 3. Kalman güncelle
        kalman_state = self.kalman.update(price)
        self.kalman_prices.append(kalman_state.price)
        
        # 4. Z-Score hesapla
        z_score = self._calculate_z_score(price, kalman_state.price)
        
        # 5. Final rejim belirle
        regime = self._determine_regime(hurst_result, kalman_state.velocity)
        
        # 6. Trading bias belirle
        bias, bias_strength = self._determine_bias(
            kalman_state.velocity, 
            z_score, 
            regime
        )
        
        state = BrainState(
            timestamp=timestamp,
            price=price,
            kalman_price=kalman_state.price,
            kalman_velocity=kalman_state.velocity,
            future_predictions=kalman_state.future_prices,
            hurst_value=hurst_result.value,
            hurst_regime=hurst_result.regime,
            hurst_confidence=hurst_result.confidence,
            z_score=z_score,
            regime=regime,
            bias=bias,
            bias_strength=bias_strength
        )
        
        self.states.append(state)
        return state
    
    def _calculate_hurst(self) -> HurstResult:
        """Hurst hesapla"""
        if len(self.prices) < self.hurst_window:
            return HurstResult(0.5, "NEUTRAL", 0.0, 0.0)
        
        return calculate_hurst(
            np.array(self.prices),
            window=self.hurst_window
        )
    
    def _calculate_z_score(self, price: float, kalman_price: float) -> float:
        """
        Z-Score: Fiyatın Kalman eğrisinden sapması
        
        Z > 2: Aşırı alım (potential short)
        Z < -2: Aşırı satım (potential long)
        """
        if len(self.kalman_prices) < self.z_score_window:
            return 0.0
        
        # Son N bardaki sapmaların std'si
        recent_kalman = np.array(self.kalman_prices[-self.z_score_window:])
        recent_prices = np.array(self.prices[-self.z_score_window:])
        
        deviations = recent_prices - recent_kalman
        std = np.std(deviations)
        
        if std < 1e-10:
            return 0.0
            
        current_deviation = price - kalman_price
        z_score = current_deviation / std
        
        return float(z_score)
    
    def _determine_regime(self, hurst: HurstResult, velocity: float) -> MarketRegime:
        """Final rejim belirleme"""
        if hurst.regime == "TRENDING":
            if velocity > 0:
                return MarketRegime.TRENDING_BULLISH
            else:
                return MarketRegime.TRENDING_BEARISH
        elif hurst.regime == "CHOPPY":
            return MarketRegime.CHOPPY
        else:
            return MarketRegime.NEUTRAL
    
    def _determine_bias(
        self, 
        velocity: float, 
        z_score: float,
        regime: MarketRegime
    ) -> Tuple[str, float]:
        """
        Trading bias belirleme
        
        Trend rejiminde: Velocity yönüne git
        Choppy rejimde: Z-Score tersine git (mean reversion)
        """
        if regime in [MarketRegime.TRENDING_BULLISH, MarketRegime.TRENDING_BEARISH]:
            # Trend-following
            if velocity > 0.1:
                return "BULLISH", min(abs(velocity) * 10, 1.0)
            elif velocity < -0.1:
                return "BEARISH", min(abs(velocity) * 10, 1.0)
            else:
                return "NEUTRAL", 0.0
                
        elif regime == MarketRegime.CHOPPY:
            # Mean reversion
            if z_score > 2.0:
                return "BEARISH", min(abs(z_score) / 4, 1.0)  # Overbought -> Short
            elif z_score < -2.0:
                return "BULLISH", min(abs(z_score) / 4, 1.0)  # Oversold -> Long
            else:
                return "NEUTRAL", 0.0
        
        return "NEUTRAL", 0.0
    
    def get_summary(self) -> Dict:
        """Son durumun özeti"""
        if not self.states:
            return {}
        
        last = self.states[-1]
        return {
            "regime": last.regime.value,
            "bias": last.bias,
            "bias_strength": f"{last.bias_strength:.1%}",
            "hurst": f"{last.hurst_value:.3f}",
            "z_score": f"{last.z_score:.2f}",
            "kalman_velocity": f"{last.kalman_velocity:.4f}",
            "future_3_bar": last.future_predictions[-1] if last.future_predictions else None
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Tüm state'leri DataFrame'e çevir"""
        if not self.states:
            return pd.DataFrame()
        
        data = []
        for s in self.states:
            data.append({
                'timestamp': s.timestamp,
                'price': s.price,
                'kalman_price': s.kalman_price,
                'velocity': s.kalman_velocity,
                'hurst': s.hurst_value,
                'hurst_regime': s.hurst_regime,
                'z_score': s.z_score,
                'regime': s.regime.value,
                'bias': s.bias,
                'bias_strength': s.bias_strength
            })
        
        return pd.DataFrame(data)


# === TEST ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Simüle data
    np.random.seed(42)
    
    # 3 faz: Choppy -> Trend Up -> Choppy
    choppy1 = np.sin(np.linspace(0, 6, 100)) * 3 + 100 + np.random.randn(100) * 0.5
    trend = np.linspace(100, 120, 150) + np.random.randn(150) * 0.8
    choppy2 = np.sin(np.linspace(0, 8, 100)) * 4 + 120 + np.random.randn(100) * 0.5
    
    prices = np.concatenate([choppy1, trend, choppy2])
    
    # Brain çalıştır
    brain = QuantBrain()
    
    for i, price in enumerate(prices):
        state = brain.update(price)
        
    # Summary
    print("=== QuantBrain Summary ===")
    print(brain.get_summary())
    
    # Plot
    df = brain.to_dataframe()
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    # Price + Kalman
    axes[0].plot(df['price'], label='Price', alpha=0.5)
    axes[0].plot(df['kalman_price'], label='Kalman', linewidth=2)
    axes[0].set_title('Price vs Kalman')
    axes[0].legend()
    
    # Hurst
    axes[1].plot(df['hurst'], color='purple')
    axes[1].axhline(0.5, color='black', linestyle='--', alpha=0.3)
    axes[1].axhline(0.6, color='green', linestyle='--', alpha=0.3, label='Trend threshold')
    axes[1].axhline(0.4, color='red', linestyle='--', alpha=0.3, label='Choppy threshold')
    axes[1].set_title('Hurst Exponent')
    axes[1].legend()
    
    # Z-Score
    axes[2].plot(df['z_score'], color='orange')
    axes[2].axhline(2, color='red', linestyle='--', alpha=0.3)
    axes[2].axhline(-2, color='green', linestyle='--', alpha=0.3)
    axes[2].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[2].set_title('Z-Score (Deviation from Kalman)')
    
    # Bias Strength
    colors = ['green' if b == 'BULLISH' else 'red' if b == 'BEARISH' else 'gray' 
              for b in df['bias']]
    axes[3].bar(range(len(df)), df['bias_strength'], color=colors, alpha=0.7)
    axes[3].set_title('Trading Bias Strength')
    
    plt.tight_layout()
    plt.show()
