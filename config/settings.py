# NQ Quant Bot - Configuration

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class KalmanConfig:
    """Kalman Filter Configuration"""
    # Base noise parameters
    process_noise: float = 1e-5      # Q - süreç gürültüsü
    measurement_noise: float = 1e-3   # R - ölçüm gürültüsü
    
    # Adaptive parameters
    trending_r_multiplier: float = 0.1    # Trend: daha hassas
    choppy_r_multiplier: float = 10.0     # Yatay: daha sağır
    
    # Future prediction steps
    future_steps: int = 3

@dataclass
class HurstConfig:
    """Hurst Exponent Configuration"""
    window: int = 100              # Hesaplama penceresi
    min_lag: int = 2               # Minimum lag
    max_lag: int = 20              # Maximum lag
    
    # Rejim eşikleri
    trending_threshold: float = 0.6     # H > 0.6 = Trend
    choppy_threshold: float = 0.4       # H < 0.4 = Mean Revert

@dataclass 
class OrderBlockConfig:
    """Order Block Detection Configuration"""
    lookback: int = 20             # Geriye bakış periyodu
    body_threshold: float = 1.5    # Ortalamadan X kat büyük body
    volume_threshold: float = 1.5  # Ortalamadan X kat büyük hacim

@dataclass
class RiskConfig:
    """Risk Management Configuration"""
    max_risk_per_trade: float = 0.02    # Trade başına max %2
    max_daily_risk: float = 0.05        # Günlük max %5
    atr_stop_multiplier: float = 2.0    # ATR * 2 = Stop mesafesi

@dataclass
class TradingConfig:
    """Main Trading Configuration"""
    symbol: str = "NQ=F"                 # Nasdaq Futures
    timeframe: str = "1d"                # Günlük
    data_provider: str = "yfinance"      # OpenBB provider
    
    # Sub-configs
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    hurst: HurstConfig = field(default_factory=HurstConfig)
    order_block: OrderBlockConfig = field(default_factory=OrderBlockConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    # Confluence requirements
    min_confluence_score: int = 3        # Minimum 3 faktör

# Default config
DEFAULT_CONFIG = TradingConfig()
