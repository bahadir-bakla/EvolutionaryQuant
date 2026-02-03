# NQ Quant Bot - Enhanced Confluence Engine v2
# Integrates ALL advanced indicators

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nq_core.brain import QuantBrain, BrainState
from nq_core.order_blocks import OrderBlock, get_active_order_blocks
from nq_core.indicators.vwap import calculate_vwap, get_vwap_signal, VWAPBias
from nq_core.indicators.pivots import calculate_pivots, get_pivot_zone, PivotType
from nq_core.indicators.market_structure import find_swing_points, analyze_structure, StructureBias
from nq_core.indicators.sessions import get_current_session, is_high_volatility_time, TradingSession
from nq_core.indicators.fvg import detect_fvg, get_active_fvgs, check_fvg_interaction, FVGType
from nq_core.indicators.adx_volume import calculate_adx, get_adx_signal, calculate_volume_profile, TrendStrength


class SignalStrength(Enum):
    VERY_STRONG = "VERY_STRONG"  # 8+ factors
    STRONG = "STRONG"            # 6-7 factors
    MODERATE = "MODERATE"        # 4-5 factors
    WEAK = "WEAK"                # 2-3 factors
    NO_SIGNAL = "NO_SIGNAL"      # 0-1 factors


@dataclass
class ConfluenceFactor:
    """Individual confluence factor"""
    name: str
    signal: str  # 'LONG', 'SHORT', 'NEUTRAL'
    weight: float
    confidence: float
    reason: str


@dataclass
class EnhancedSignal:
    """Enhanced trading signal with all factors"""
    direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    strength: SignalStrength
    confluence_score: float
    active_factors: int
    total_factors: int
    confidence: float
    
    # Price levels
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # Factor breakdown
    factors: List[ConfluenceFactor]
    
    # Key levels
    vwap: float
    pivot: float
    poc: float  # Volume Profile POC
    nearest_support: float
    nearest_resistance: float
    
    # Context
    session: str
    is_high_volatility: bool
    market_structure: str
    trend_strength: str


class EnhancedConfluenceEngine:
    """
    Enhanced Confluence Engine v2
    
    Evaluates 12+ factors:
    1. Brain State (Kalman + Hurst)
    2. Order Blocks
    3. VWAP Bias
    4. Pivot Zone
    5. Market Structure (BOS/CHoCH)
    6. Session Timing
    7. Fair Value Gaps
    8. ADX Trend Strength
    9. Volume Profile (POC proximity)
    10. RSI
    11. Z-Score
    12. Velocity
    """
    
    # Factor weights
    WEIGHTS = {
        'brain_regime': 1.0,
        'brain_velocity': 0.8,
        'brain_zscore': 0.9,
        'order_block': 1.2,
        'vwap_bias': 1.5,      # VWAP is critical for institutional
        'vwap_zone': 0.7,
        'pivot_zone': 1.0,
        'market_structure': 1.3,  # BOS/CHoCH very important
        'session': 0.6,
        'fvg': 0.9,
        'adx_strength': 0.8,
        'adx_direction': 1.0,
        'volume_poc': 0.7,
        'rsi': 0.6,
    }
    
    def __init__(
        self,
        min_factors: int = 4,
        min_score: float = 3.0,
        pivot_type: PivotType = PivotType.CLASSIC
    ):
        self.min_factors = min_factors
        self.min_score = min_score
        self.pivot_type = pivot_type
        
    def evaluate(
        self,
        df: pd.DataFrame,
        current_idx: int,
        brain_state: BrainState,
        order_blocks: List[OrderBlock],
        pivots: 'PivotLevels',
        fvgs: List['FairValueGap'],
        rsi: Optional[float] = None,
        atr: Optional[float] = None
    ) -> EnhancedSignal:
        """
        Evaluate all confluence factors and generate signal
        """
        row = df.iloc[current_idx]
        price = row['close']
        timestamp = df.index[current_idx]
        
        factors = []
        long_score = 0.0
        short_score = 0.0
        
        # === 1. BRAIN STATE (Kalman + Hurst) ===
        
        # Regime
        if brain_state.regime == 'TRENDING':
            if brain_state.bias == 'bullish':
                factor = ConfluenceFactor('brain_regime', 'LONG', self.WEIGHTS['brain_regime'], 0.8, 'Trending bullish regime')
                long_score += self.WEIGHTS['brain_regime']
            elif brain_state.bias == 'bearish':
                factor = ConfluenceFactor('brain_regime', 'SHORT', self.WEIGHTS['brain_regime'], 0.8, 'Trending bearish regime')
                short_score += self.WEIGHTS['brain_regime']
            else:
                factor = ConfluenceFactor('brain_regime', 'NEUTRAL', self.WEIGHTS['brain_regime'], 0.5, 'Trending neutral')
            factors.append(factor)
        
        # Velocity
        if brain_state.kalman_velocity > 5:
            factor = ConfluenceFactor('brain_velocity', 'LONG', self.WEIGHTS['brain_velocity'], 0.7, f'Strong upward velocity: {brain_state.kalman_velocity:.1f}')
            long_score += self.WEIGHTS['brain_velocity']
            factors.append(factor)
        elif brain_state.kalman_velocity < -5:
            factor = ConfluenceFactor('brain_velocity', 'SHORT', self.WEIGHTS['brain_velocity'], 0.7, f'Strong downward velocity: {brain_state.kalman_velocity:.1f}')
            short_score += self.WEIGHTS['brain_velocity']
            factors.append(factor)
        
        # Z-Score
        if brain_state.z_score < -2:
            factor = ConfluenceFactor('brain_zscore', 'LONG', self.WEIGHTS['brain_zscore'], 0.8, f'Oversold z-score: {brain_state.z_score:.2f}')
            long_score += self.WEIGHTS['brain_zscore']
            factors.append(factor)
        elif brain_state.z_score > 2:
            factor = ConfluenceFactor('brain_zscore', 'SHORT', self.WEIGHTS['brain_zscore'], 0.8, f'Overbought z-score: {brain_state.z_score:.2f}')
            short_score += self.WEIGHTS['brain_zscore']
            factors.append(factor)
        
        # === 2. VWAP ===
        if 'vwap' in df.columns:
            vwap = row['vwap']
            vwap_signal = get_vwap_signal(
                price, vwap,
                row.get('vwap_upper_1', vwap * 1.01),
                row.get('vwap_lower_1', vwap * 0.99),
                row.get('vwap_upper_2', vwap * 1.02),
                row.get('vwap_lower_2', vwap * 0.98)
            )
            
            # VWAP Bias (critical)
            if vwap_signal.bias == VWAPBias.BULLISH:
                factor = ConfluenceFactor('vwap_bias', 'LONG', self.WEIGHTS['vwap_bias'], 0.9, f'Price above VWAP ({vwap_signal.distance_percent:.2f}%)')
                long_score += self.WEIGHTS['vwap_bias']
                factors.append(factor)
            elif vwap_signal.bias == VWAPBias.BEARISH:
                factor = ConfluenceFactor('vwap_bias', 'SHORT', self.WEIGHTS['vwap_bias'], 0.9, f'Price below VWAP ({vwap_signal.distance_percent:.2f}%)')
                short_score += self.WEIGHTS['vwap_bias']
                factors.append(factor)
            
            # VWAP Zone (mean reversion at extremes)
            if vwap_signal.zone == 'extreme_oversold':
                factor = ConfluenceFactor('vwap_zone', 'LONG', self.WEIGHTS['vwap_zone'], 0.7, 'At -2 std band (mean reversion)')
                long_score += self.WEIGHTS['vwap_zone']
                factors.append(factor)
            elif vwap_signal.zone == 'extreme_overbought':
                factor = ConfluenceFactor('vwap_zone', 'SHORT', self.WEIGHTS['vwap_zone'], 0.7, 'At +2 std band (mean reversion)')
                short_score += self.WEIGHTS['vwap_zone']
                factors.append(factor)
        else:
            vwap = price
        
        # === 3. PIVOT POINTS ===
        if pivots:
            pivot_zone = get_pivot_zone(price, pivots)
            
            if pivot_zone in ['above_r3', 'r2_r3']:
                factor = ConfluenceFactor('pivot_zone', 'LONG', self.WEIGHTS['pivot_zone'], 0.6, f'Strong bullish pivot zone: {pivot_zone}')
                long_score += self.WEIGHTS['pivot_zone'] * 0.5  # Reduced because might be extended
                factors.append(factor)
            elif pivot_zone in ['below_s3', 's3_s2']:
                factor = ConfluenceFactor('pivot_zone', 'SHORT', self.WEIGHTS['pivot_zone'], 0.6, f'Strong bearish pivot zone: {pivot_zone}')
                short_score += self.WEIGHTS['pivot_zone'] * 0.5
                factors.append(factor)
            elif pivot_zone == 'pivot_r1':
                # Near pivot - potential support
                factor = ConfluenceFactor('pivot_zone', 'LONG', self.WEIGHTS['pivot_zone'], 0.5, 'Above pivot, support zone')
                long_score += self.WEIGHTS['pivot_zone'] * 0.3
                factors.append(factor)
            elif pivot_zone == 's1_pivot':
                factor = ConfluenceFactor('pivot_zone', 'SHORT', self.WEIGHTS['pivot_zone'], 0.5, 'Below pivot, resistance zone')
                short_score += self.WEIGHTS['pivot_zone'] * 0.3
                factors.append(factor)
        
        # === 4. MARKET STRUCTURE ===
        if 'structure_bias' in df.columns:
            structure = df.iloc[current_idx]['structure_bias']
            
            if structure == 'BULLISH':
                factor = ConfluenceFactor('market_structure', 'LONG', self.WEIGHTS['market_structure'], 0.85, 'Bullish market structure (HH/HL)')
                long_score += self.WEIGHTS['market_structure']
                factors.append(factor)
            elif structure == 'BEARISH':
                factor = ConfluenceFactor('market_structure', 'SHORT', self.WEIGHTS['market_structure'], 0.85, 'Bearish market structure (LH/LL)')
                short_score += self.WEIGHTS['market_structure']
                factors.append(factor)
        
        # === 5. SESSION ===
        session = get_current_session(timestamp)
        is_hvol = is_high_volatility_time(timestamp)
        
        if session in [TradingSession.OVERLAP, TradingSession.NEW_YORK]:
            # During active sessions, trend following works better
            factor = ConfluenceFactor('session', 'NEUTRAL', self.WEIGHTS['session'], 0.6, f'Active session: {session.value}')
            factors.append(factor)
        
        # === 6. FAIR VALUE GAPS ===
        active_fvgs = get_active_fvgs(fvgs, current_idx, max_age=30)
        fvg_interaction = check_fvg_interaction(price, active_fvgs)
        
        if fvg_interaction:
            if fvg_interaction.fvg_type == FVGType.BULLISH:
                # Price retracing into bullish FVG = buy opportunity
                factor = ConfluenceFactor('fvg', 'LONG', self.WEIGHTS['fvg'], 0.75, f'At bullish FVG: {fvg_interaction.bottom:.0f}-{fvg_interaction.top:.0f}')
                long_score += self.WEIGHTS['fvg']
                factors.append(factor)
            elif fvg_interaction.fvg_type == FVGType.BEARISH:
                factor = ConfluenceFactor('fvg', 'SHORT', self.WEIGHTS['fvg'], 0.75, f'At bearish FVG: {fvg_interaction.bottom:.0f}-{fvg_interaction.top:.0f}')
                short_score += self.WEIGHTS['fvg']
                factors.append(factor)
        
        # === 7. ADX ===
        if 'adx' in df.columns:
            adx = row['adx']
            plus_di = row['plus_di']
            minus_di = row['minus_di']
            
            adx_result = get_adx_signal(adx, plus_di, minus_di)
            
            # ADX Direction
            if adx_result.trend_direction == 'bullish' and adx > 20:
                factor = ConfluenceFactor('adx_direction', 'LONG', self.WEIGHTS['adx_direction'], 0.7, f'+DI > -DI with ADX={adx:.1f}')
                long_score += self.WEIGHTS['adx_direction']
                factors.append(factor)
            elif adx_result.trend_direction == 'bearish' and adx > 20:
                factor = ConfluenceFactor('adx_direction', 'SHORT', self.WEIGHTS['adx_direction'], 0.7, f'-DI > +DI with ADX={adx:.1f}')
                short_score += self.WEIGHTS['adx_direction']
                factors.append(factor)
            
            # ADX Strength
            if adx_result.trend_strength in [TrendStrength.STRONG, TrendStrength.VERY_STRONG]:
                factor = ConfluenceFactor('adx_strength', 'NEUTRAL', self.WEIGHTS['adx_strength'], 0.6, f'Strong trend: {adx_result.trend_strength.value}')
                factors.append(factor)
        else:
            adx_result = None
        
        # === 8. ORDER BLOCKS ===
        for ob in order_blocks:
            if ob.direction == 'bullish' and ob.low <= price <= ob.high:
                factor = ConfluenceFactor('order_block', 'LONG', self.WEIGHTS['order_block'], 0.8, f'Inside bullish OB: {ob.low:.0f}-{ob.high:.0f}')
                long_score += self.WEIGHTS['order_block']
                factors.append(factor)
                break
            elif ob.direction == 'bearish' and ob.low <= price <= ob.high:
                factor = ConfluenceFactor('order_block', 'SHORT', self.WEIGHTS['order_block'], 0.8, f'Inside bearish OB: {ob.low:.0f}-{ob.high:.0f}')
                short_score += self.WEIGHTS['order_block']
                factors.append(factor)
                break
        
        # === 9. RSI ===
        if rsi is not None:
            if rsi < 30:
                factor = ConfluenceFactor('rsi', 'LONG', self.WEIGHTS['rsi'], 0.65, f'RSI oversold: {rsi:.1f}')
                long_score += self.WEIGHTS['rsi']
                factors.append(factor)
            elif rsi > 70:
                factor = ConfluenceFactor('rsi', 'SHORT', self.WEIGHTS['rsi'], 0.65, f'RSI overbought: {rsi:.1f}')
                short_score += self.WEIGHTS['rsi']
                factors.append(factor)
        
        # === CALCULATE FINAL SIGNAL ===
        
        total_score = max(long_score, short_score)
        net_score = long_score - short_score
        
        # Count aligned factors
        long_factors = len([f for f in factors if f.signal == 'LONG'])
        short_factors = len([f for f in factors if f.signal == 'SHORT'])
        
        # Determine direction
        if net_score > self.min_score and long_factors >= self.min_factors:
            direction = 'LONG'
            active_factors = long_factors
        elif net_score < -self.min_score and short_factors >= self.min_factors:
            direction = 'SHORT'
            active_factors = short_factors
        else:
            direction = 'NEUTRAL'
            active_factors = 0
        
        # Determine strength
        max_factors = max(long_factors, short_factors)
        if max_factors >= 8:
            strength = SignalStrength.VERY_STRONG
        elif max_factors >= 6:
            strength = SignalStrength.STRONG
        elif max_factors >= 4:
            strength = SignalStrength.MODERATE
        elif max_factors >= 2:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NO_SIGNAL
        
        # Calculate price levels
        if atr and atr > 0:
            if direction == 'LONG':
                stop_loss = price - (atr * 2.5)
                tp1 = price + (atr * 2)
                tp2 = price + (atr * 3)
                tp3 = price + (atr * 4)
            elif direction == 'SHORT':
                stop_loss = price + (atr * 2.5)
                tp1 = price - (atr * 2)
                tp2 = price - (atr * 3)
                tp3 = price - (atr * 4)
            else:
                stop_loss = price
                tp1 = tp2 = tp3 = price
        else:
            stop_loss = price
            tp1 = tp2 = tp3 = price
        
        # Get key levels
        nearest_support = pivots.get_nearest_support(price) if pivots else price * 0.99
        nearest_resistance = pivots.get_nearest_resistance(price) if pivots else price * 1.01
        
        return EnhancedSignal(
            direction=direction,
            strength=strength,
            confluence_score=abs(net_score),
            active_factors=active_factors,
            total_factors=len(factors),
            confidence=min(1.0, abs(net_score) / 10),
            entry=price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            factors=factors,
            vwap=vwap,
            pivot=pivots.pivot if pivots else price,
            poc=price,  # Will be updated with volume profile
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
            session=session.value,
            is_high_volatility=is_hvol,
            market_structure=df.iloc[current_idx].get('structure_bias', 'NEUTRAL'),
            trend_strength=adx_result.trend_strength.value if adx_result else 'UNKNOWN'
        )


# === TEST ===
if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing Enhanced Confluence Engine v2...")
    
    # Fetch data
    ticker = yf.Ticker("NQ=F")
    df = ticker.history(period="3mo", interval="1d")
    df.columns = df.columns.str.lower()
    
    # Add all indicators
    from nq_core.indicators import calculate_vwap, calculate_adx, detect_fvg
    from nq_core.indicators.market_structure import detect_bos_choch
    from nq_core.indicators.pivots import calculate_pivots
    from nq_core.order_blocks import detect_order_blocks, get_active_order_blocks
    
    print("Calculating indicators...")
    df = calculate_vwap(df, reset_daily=False)
    df = calculate_adx(df)
    df = detect_bos_choch(df)
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # Other indicators
    order_blocks = detect_order_blocks(df)
    fvgs = detect_fvg(df)
    
    # Get yesterday's OHLC for pivots
    prev_day = df.iloc[-2]
    pivots = calculate_pivots(prev_day['high'], prev_day['low'], prev_day['close'])
    
    # Initialize
    brain = QuantBrain()
    engine = EnhancedConfluenceEngine(min_factors=3, min_score=2.5)
    
    # Process last bar
    current_idx = len(df) - 1
    row = df.iloc[current_idx]
    
    brain_state = brain.update(row['close'], df.index[current_idx])
    active_obs = get_active_order_blocks(order_blocks, current_idx, max_age=30)
    
    signal = engine.evaluate(
        df, current_idx, brain_state, active_obs, pivots, fvgs,
        rsi=row['rsi'], atr=row['atr']
    )
    
    print(f"\n{'='*60}")
    print("ENHANCED SIGNAL ANALYSIS")
    print(f"{'='*60}")
    print(f"Price: {row['close']:.2f}")
    print(f"Direction: {signal.direction}")
    print(f"Strength: {signal.strength.value}")
    print(f"Confluence Score: {signal.confluence_score:.2f}")
    print(f"Active Factors: {signal.active_factors}/{signal.total_factors}")
    print(f"Confidence: {signal.confidence:.1%}")
    
    print(f"\nKey Levels:")
    print(f"  Entry: {signal.entry:.2f}")
    print(f"  Stop Loss: {signal.stop_loss:.2f}")
    print(f"  TP1: {signal.take_profit_1:.2f}")
    print(f"  TP2: {signal.take_profit_2:.2f}")
    print(f"  VWAP: {signal.vwap:.2f}")
    print(f"  Pivot: {signal.pivot:.2f}")
    
    print(f"\nContext:")
    print(f"  Session: {signal.session}")
    print(f"  High Volatility: {signal.is_high_volatility}")
    print(f"  Market Structure: {signal.market_structure}")
    print(f"  Trend Strength: {signal.trend_strength}")
    
    print(f"\nFactor Breakdown:")
    for f in signal.factors:
        print(f"  [{f.signal:^7}] {f.name}: {f.reason} (weight: {f.weight:.1f})")
