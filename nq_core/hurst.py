# Hurst Exponent Calculator
# R/S Analysis ile piyasa rejimi tespiti

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class HurstResult:
    """Hurst Exponent calculation result"""
    value: float                    # Hurst değeri (0-1)
    regime: str                     # TRENDING, CHOPPY, NEUTRAL
    confidence: float               # Hesaplama güvenilirliği
    r_squared: float               # Fit kalitesi

def calculate_hurst(
    series: np.ndarray,
    window: int = 100,
    min_lag: int = 2,
    max_lag: int = 20
) -> HurstResult:
    """
    R/S (Rescaled Range) Analysis ile Hurst Exponent hesapla.
    
    Hurst Exponent yorumu:
    - H < 0.5: Mean-Reverting (Testere/Yatay piyasa) -> Counter-trend stratejisi
    - H = 0.5: Random Walk (Rastgele yürüyüş) -> Sinyal yok
    - H > 0.5: Trending (Trend piyasası) -> Trend-following stratejisi
    
    Args:
        series: Fiyat serisi (numpy array)
        window: Hesaplama pencere boyutu
        min_lag: Minimum lag değeri
        max_lag: Maximum lag değeri
        
    Returns:
        HurstResult: Hurst değeri ve rejim bilgisi
    """
    
    # Yeterli veri kontrolü
    if len(series) < window:
        logger.warning(f"Yetersiz veri: {len(series)} < {window}")
        return HurstResult(
            value=0.5, 
            regime="NEUTRAL", 
            confidence=0.0,
            r_squared=0.0
        )
    
    # Son 'window' kadar veriyi al
    data = np.array(series[-window:])
    
    # Log returns (daha stabil)
    returns = np.diff(np.log(data + 1e-10))
    
    lags = range(min_lag, min(max_lag, len(returns) // 2))
    rs_values = []
    
    for lag in lags:
        # R/S hesaplama
        rs = _calculate_rs(returns, lag)
        if rs > 0:
            rs_values.append((lag, rs))
    
    if len(rs_values) < 3:
        return HurstResult(
            value=0.5,
            regime="NEUTRAL",
            confidence=0.0,
            r_squared=0.0
        )
    
    # Log-log regression
    log_lags = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])
    
    # Linear fit: log(R/S) = H * log(n) + c
    hurst, intercept, r_squared = _linear_regression(log_lags, log_rs)
    
    # Hurst değerini sınırla (0, 1)
    hurst = np.clip(hurst, 0.0, 1.0)
    
    # Rejim belirleme
    if hurst > 0.6:
        regime = "TRENDING"
    elif hurst < 0.4:
        regime = "CHOPPY"
    else:
        regime = "NEUTRAL"
    
    # Confidence = R² değeri
    confidence = max(0.0, min(1.0, r_squared))
    
    return HurstResult(
        value=float(hurst),
        regime=regime,
        confidence=confidence,
        r_squared=r_squared
    )

def _calculate_rs(returns: np.ndarray, lag: int) -> float:
    """R/S (Rescaled Range) hesapla"""
    n = len(returns)
    num_segments = n // lag
    
    if num_segments < 1:
        return 0.0
    
    rs_list = []
    
    for i in range(num_segments):
        segment = returns[i * lag:(i + 1) * lag]
        
        if len(segment) < 2:
            continue
            
        # Mean-adjusted cumulative deviations
        mean_ret = np.mean(segment)
        deviations = segment - mean_ret
        cumsum = np.cumsum(deviations)
        
        # Range
        R = np.max(cumsum) - np.min(cumsum)
        
        # Standard deviation
        S = np.std(segment, ddof=1)
        
        if S > 1e-10:
            rs_list.append(R / S)
    
    return np.mean(rs_list) if rs_list else 0.0

def _linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Simple linear regression: y = slope * x + intercept"""
    n = len(x)
    
    if n < 2:
        return 0.5, 0.0, 0.0
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    if denominator < 1e-10:
        return 0.5, 0.0, 0.0
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # R² hesapla
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    
    r_squared = 1 - (ss_res / (ss_tot + 1e-10)) if ss_tot > 1e-10 else 0.0
    
    return slope, intercept, r_squared


# === TEST ===
if __name__ == "__main__":
    # Test: Random walk -> H ≈ 0.5
    np.random.seed(42)
    random_walk = np.cumsum(np.random.randn(200)) + 100
    result = calculate_hurst(random_walk)
    print(f"Random Walk: H={result.value:.3f}, Regime={result.regime}")
    
    # Test: Trending -> H > 0.5
    trend = np.linspace(100, 150, 200) + np.random.randn(200) * 0.5
    result = calculate_hurst(trend)
    print(f"Trending: H={result.value:.3f}, Regime={result.regime}")
    
    # Test: Mean-reverting -> H < 0.5
    mean_rev = np.sin(np.linspace(0, 10, 200)) * 5 + 100 + np.random.randn(200) * 0.3
    result = calculate_hurst(mean_rev)
    print(f"Mean-Rev: H={result.value:.3f}, Regime={result.regime}")
