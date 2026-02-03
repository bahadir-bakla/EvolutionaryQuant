# Confluence Engine
# Çoklu sinyalleri birleştirip final trading kararı

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import logging

from .brain import BrainState, MarketRegime
from .order_blocks import OrderBlock, check_ob_interaction, calculate_ob_confluence

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Sinyal tipi"""
    STRONG_LONG = "STRONG_LONG"
    LONG = "LONG"
    WEAK_LONG = "WEAK_LONG"
    NEUTRAL = "NEUTRAL"
    WEAK_SHORT = "WEAK_SHORT"
    SHORT = "SHORT"
    STRONG_SHORT = "STRONG_SHORT"


@dataclass
class ConfluenceFactor:
    """Tek bir confluence faktörü"""
    name: str
    value: float          # -1 (bearish) to +1 (bullish)
    weight: float         # Ağırlık
    active: bool          # Bu faktör aktif mi?
    description: str = ""


@dataclass
class Signal:
    """Final trading sinyali"""
    timestamp: Optional[pd.Timestamp] = None
    signal_type: SignalType = SignalType.NEUTRAL
    direction: str = "NEUTRAL"   # LONG, SHORT, NEUTRAL
    
    # Confluence details
    confluence_score: float = 0.0
    factors: List[ConfluenceFactor] = field(default_factory=list)
    active_factors: int = 0
    
    # Entry/Exit levels
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit_1: float = 0.0
    take_profit_2: float = 0.0
    
    # Risk
    risk_reward_ratio: float = 0.0
    confidence: float = 0.0


class ConfluenceEngine:
    """
    Confluence (Birleşim) Motoru
    
    Kural: Minimum 3 faktör aynı yönde olmalı
    
    Faktörler:
    1. Rejim (Hurst) - Trend mi Yatay mı?
    2. Velocity (Kalman) - Momentum yönü
    3. Z-Score - Mean reversion sinyali
    4. Order Block - Kurumsal bölge
    5. RSI (OpenBB'den) - Aşırı alım/satım
    6. Price vs Kalman - Fiyat pozisyonu
    """
    
    def __init__(
        self,
        min_confluence: int = 3,
        weights: Optional[Dict[str, float]] = None
    ):
        self.min_confluence = min_confluence
        
        # Default ağırlıklar
        self.weights = weights or {
            'regime': 1.5,
            'velocity': 1.0,
            'z_score': 1.2,
            'order_block': 1.5,
            'price_position': 0.8,
            'rsi': 1.0
        }
        
    def evaluate(
        self,
        brain_state: BrainState,
        order_blocks: List[OrderBlock],
        rsi: Optional[float] = None,
        atr: Optional[float] = None
    ) -> Signal:
        """
        Confluence değerlendirmesi yap
        
        Args:
            brain_state: QuantBrain çıktısı
            order_blocks: Aktif Order Blocklar
            rsi: RSI değeri (0-100)
            atr: ATR değeri (stop hesabı için)
            
        Returns:
            Signal: Trading sinyali
        """
        factors: List[ConfluenceFactor] = []
        
        # 1. Rejim Faktörü
        regime_factor = self._evaluate_regime(brain_state)
        factors.append(regime_factor)
        
        # 2. Velocity (Momentum) Faktörü
        velocity_factor = self._evaluate_velocity(brain_state)
        factors.append(velocity_factor)
        
        # 3. Z-Score Faktörü
        z_score_factor = self._evaluate_z_score(brain_state)
        factors.append(z_score_factor)
        
        # 4. Order Block Faktörü
        ob_factor = self._evaluate_order_blocks(brain_state.price, order_blocks)
        factors.append(ob_factor)
        
        # 5. Price Position (Fiyat Kalman'ın üstünde/altında)
        price_pos_factor = self._evaluate_price_position(brain_state)
        factors.append(price_pos_factor)
        
        # 6. RSI (varsa)
        if rsi is not None:
            rsi_factor = self._evaluate_rsi(rsi)
            factors.append(rsi_factor)
        
        # Confluence hesapla
        signal = self._calculate_signal(factors, brain_state, atr)
        signal.timestamp = brain_state.timestamp
        
        return signal
    
    def _evaluate_regime(self, state: BrainState) -> ConfluenceFactor:
        """Rejim değerlendirmesi"""
        if state.regime == MarketRegime.TRENDING_BULLISH:
            value = 1.0
            desc = "Bullish Trend"
        elif state.regime == MarketRegime.TRENDING_BEARISH:
            value = -1.0
            desc = "Bearish Trend"
        elif state.regime == MarketRegime.CHOPPY:
            value = 0.0
            desc = "Choppy Market"
        else:
            value = 0.0
            desc = "Neutral"
            
        return ConfluenceFactor(
            name="regime",
            value=value,
            weight=self.weights['regime'],
            active=abs(value) > 0.3,
            description=desc
        )
    
    def _evaluate_velocity(self, state: BrainState) -> ConfluenceFactor:
        """Velocity (momentum) değerlendirmesi"""
        vel = state.kalman_velocity
        
        # Normalize: -1 to 1
        normalized = np.tanh(vel * 5)  
        
        if normalized > 0.3:
            desc = f"Bullish Momentum ({vel:.4f})"
        elif normalized < -0.3:
            desc = f"Bearish Momentum ({vel:.4f})"
        else:
            desc = "Neutral Momentum"
            
        return ConfluenceFactor(
            name="velocity",
            value=normalized,
            weight=self.weights['velocity'],
            active=abs(normalized) > 0.3,
            description=desc
        )
    
    def _evaluate_z_score(self, state: BrainState) -> ConfluenceFactor:
        """Z-Score mean reversion sinyali"""
        z = state.z_score
        
        # Mean reversion: Z > 2 -> Short signal, Z < -2 -> Long signal
        if z > 2.0:
            value = -min(1.0, (z - 2) / 2)  # Overbought -> Bearish
            desc = f"Overbought (Z={z:.2f})"
        elif z < -2.0:
            value = min(1.0, (-z - 2) / 2)  # Oversold -> Bullish
            desc = f"Oversold (Z={z:.2f})"
        else:
            value = 0.0
            desc = f"Normal range (Z={z:.2f})"
            
        return ConfluenceFactor(
            name="z_score",
            value=value,
            weight=self.weights['z_score'],
            active=abs(z) > 2.0,
            description=desc
        )
    
    def _evaluate_order_blocks(
        self, 
        price: float, 
        order_blocks: List[OrderBlock]
    ) -> ConfluenceFactor:
        """Order Block etkileşimi"""
        if not order_blocks:
            return ConfluenceFactor(
                name="order_block",
                value=0.0,
                weight=self.weights['order_block'],
                active=False,
                description="No active OB"
            )
        
        # Fiyat OB içinde mi?
        ob, interaction = check_ob_interaction(price, order_blocks)
        
        if ob and interaction in ["INSIDE", "TOUCHED"]:
            if ob.direction == "BULLISH":
                value = ob.strength
                desc = f"At Bullish OB (str={ob.strength:.2f})"
            else:
                value = -ob.strength
                desc = f"At Bearish OB (str={ob.strength:.2f})"
            active = True
        else:
            # Yakın OB var mı?
            bullish_conf = calculate_ob_confluence(price, order_blocks, "BULLISH")
            bearish_conf = calculate_ob_confluence(price, order_blocks, "BEARISH")
            
            value = bullish_conf - bearish_conf
            active = abs(value) > 0.3
            desc = f"OB Confluence: Bull={bullish_conf:.2f}, Bear={bearish_conf:.2f}"
            
        return ConfluenceFactor(
            name="order_block",
            value=value,
            weight=self.weights['order_block'],
            active=active,
            description=desc
        )
    
    def _evaluate_price_position(self, state: BrainState) -> ConfluenceFactor:
        """Fiyatın Kalman'a göre pozisyonu"""
        diff_pct = (state.price - state.kalman_price) / state.kalman_price
        
        # Normalize
        value = np.tanh(diff_pct * 10)
        
        if value > 0.2:
            desc = f"Price above Kalman (+{diff_pct:.2%})"
        elif value < -0.2:
            desc = f"Price below Kalman ({diff_pct:.2%})"
        else:
            desc = "Price at Kalman"
            
        return ConfluenceFactor(
            name="price_position",
            value=value,
            weight=self.weights['price_position'],
            active=abs(value) > 0.2,
            description=desc
        )
    
    def _evaluate_rsi(self, rsi: float) -> ConfluenceFactor:
        """RSI değerlendirmesi"""
        if rsi > 70:
            value = -(rsi - 70) / 30  # Overbought -> Bearish
            desc = f"RSI Overbought ({rsi:.1f})"
        elif rsi < 30:
            value = (30 - rsi) / 30   # Oversold -> Bullish
            desc = f"RSI Oversold ({rsi:.1f})"
        else:
            value = 0.0
            desc = f"RSI Neutral ({rsi:.1f})"
            
        return ConfluenceFactor(
            name="rsi",
            value=value,
            weight=self.weights['rsi'],
            active=rsi > 70 or rsi < 30,
            description=desc
        )
    
    def _calculate_signal(
        self, 
        factors: List[ConfluenceFactor],
        state: BrainState,
        atr: Optional[float]
    ) -> Signal:
        """Final sinyal hesapla"""
        
        # Ağırlıklı toplam
        weighted_sum = sum(f.value * f.weight for f in factors if f.active)
        total_weight = sum(f.weight for f in factors if f.active)
        
        active_factors = sum(1 for f in factors if f.active)
        
        if total_weight > 0:
            confluence_score = weighted_sum / total_weight
        else:
            confluence_score = 0.0
        
        # Sinyal tipi belirleme
        if active_factors >= self.min_confluence:
            if confluence_score > 0.6:
                signal_type = SignalType.STRONG_LONG
                direction = "LONG"
            elif confluence_score > 0.3:
                signal_type = SignalType.LONG
                direction = "LONG"
            elif confluence_score > 0.1:
                signal_type = SignalType.WEAK_LONG
                direction = "LONG"
            elif confluence_score < -0.6:
                signal_type = SignalType.STRONG_SHORT
                direction = "SHORT"
            elif confluence_score < -0.3:
                signal_type = SignalType.SHORT
                direction = "SHORT"
            elif confluence_score < -0.1:
                signal_type = SignalType.WEAK_SHORT
                direction = "SHORT"
            else:
                signal_type = SignalType.NEUTRAL
                direction = "NEUTRAL"
        else:
            signal_type = SignalType.NEUTRAL
            direction = "NEUTRAL"
        
        # Entry/Exit seviyeleri
        entry_price = state.price
        
        if atr and direction != "NEUTRAL":
            stop_distance = atr * 2
            tp1_distance = atr * 2
            tp2_distance = atr * 4
            
            if direction == "LONG":
                stop_loss = entry_price - stop_distance
                take_profit_1 = entry_price + tp1_distance
                take_profit_2 = entry_price + tp2_distance
            else:
                stop_loss = entry_price + stop_distance
                take_profit_1 = entry_price - tp1_distance
                take_profit_2 = entry_price - tp2_distance
                
            risk_reward = tp1_distance / stop_distance if stop_distance > 0 else 0
        else:
            stop_loss = 0.0
            take_profit_1 = 0.0
            take_profit_2 = 0.0
            risk_reward = 0.0
        
        return Signal(
            signal_type=signal_type,
            direction=direction,
            confluence_score=confluence_score,
            factors=factors,
            active_factors=active_factors,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            risk_reward_ratio=risk_reward,
            confidence=min(1.0, active_factors / self.min_confluence)
        )


# === TEST ===
if __name__ == "__main__":
    from .brain import QuantBrain, MarketRegime
    
    # Simüle state
    test_state = BrainState(
        price=105.0,
        kalman_price=103.0,
        kalman_velocity=0.3,
        hurst_value=0.65,
        hurst_regime="TRENDING",
        z_score=-0.5,
        regime=MarketRegime.TRENDING_BULLISH,
        bias="BULLISH",
        bias_strength=0.7
    )
    
    # Simüle OB
    test_obs = [
        OrderBlock(index=10, low=102, high=104, direction="BULLISH", 
                  strength=0.8, volume_ratio=2.0, body_ratio=2.0)
    ]
    
    # Confluence Engine
    engine = ConfluenceEngine(min_confluence=3)
    signal = engine.evaluate(test_state, test_obs, rsi=35, atr=2.0)
    
    print("=== Confluence Signal ===")
    print(f"Direction: {signal.direction}")
    print(f"Type: {signal.signal_type.value}")
    print(f"Score: {signal.confluence_score:.2f}")
    print(f"Active Factors: {signal.active_factors}")
    print(f"Confidence: {signal.confidence:.1%}")
    print(f"\nEntry: {signal.entry_price:.2f}")
    print(f"Stop Loss: {signal.stop_loss:.2f}")
    print(f"TP1: {signal.take_profit_1:.2f}")
    print(f"R:R = {signal.risk_reward_ratio:.1f}")
    print("\n--- Factors ---")
    for f in signal.factors:
        status = "✓" if f.active else "✗"
        print(f"  {status} {f.name}: {f.value:+.2f} (w={f.weight}) - {f.description}")
