"""
NQ Wick Reversal Strategy - DEAP Compatible Version
This version is designed to work with the existing DEAP optimization framework
by exposing parameters that can be evolved and providing a standard interface.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
from 01_strategy_interface import IStrategy, Signal, SignalType

@dataclass
class NQWickReversalParams:
    """Parameters that can be evolved by DEAP"""
    # US Session parameters
    session_start_hour: int = 9          # 9:30 AM EST
    session_start_minute: int = 30
    range_duration_minutes: int = 15     # First 15 minutes
    
    # Wick reversal parameters
    wick_threshold_ratio: float = 0.3    # Minimum wick size as ratio of candle body
    
    # Profit target
    profit_target_points: float = 20.0   # 20 units profit target
    
    # Stop loss
    stop_loss_atr_mult: float = 1.5      # ATR multiplier for stop loss
    
    # Position sizing
    base_lot_size: float = 0.01
    max_lot_size: float = 0.05
    
    # ATR period
    atr_period: int = 14

class NQWickReversalStrategy(IStrategy):
    """
    NQ Wick Reversal Strategy compatible with DEAP framework
    - Identifies US session opening range (first 15 minutes)
    - Looks for wick rejections at liquidation points (range highs/lows)
    - Enters in opposite direction of wick rejection
    - Fixed 20 point profit target
    """
    
    def __init__(self, params: NQWickReversalParams = None):
        self.params = params or NQWickReversalParams()
        self.name = "NQ_Wick_Reversal"
        self.daily_range_high = None
        self.daily_range_low = None
        self.range_established = False
        self.last_range_date = None
        
    def get_required_indicators(self) -> List[str]:
        """Return list of indicators this strategy requires"""
        return ['atr', 'body_size', 'upper_wick', 'lower_wick', 
                'upper_wick_ratio', 'lower_wick_ratio', 'is_bullish', 'is_bearish']
    
    def add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add required indicators for the strategy"""
        df = data.copy()
        
        # Calculate ATR for dynamic stop loss
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(self.params.atr_period).mean()
        
        # Calculate candle properties
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        
        # Wick strength as ratio of wick to body (avoid division by zero)
        df['upper_wick_ratio'] = np.where(
            df['body_size'] > 0, 
            df['upper_wick'] / df['body_size'], 
            0
        )
        df['lower_wick_ratio'] = np.where(
            df['body_size'] > 0, 
            df['lower_wick'] / df['body_size'], 
            0
        )
        
        return df
    
    def is_us_session_open(self, timestamp) -> bool:
        """Check if we're in the US session opening range"""
        hour = timestamp.hour
        minute = timestamp.minute
        
        start_time = self.params.session_start_hour * 60 + self.params.session_start_minute
        current_time = hour * 60 + minute
        end_time = start_time + self.params.range_duration_minutes
        
        return start_time <= current_time < end_time
    
    def update_daily_range(self, df: pd.DataFrame, idx: int):
        """Update the daily opening range based on first 15 minutes"""
        current_date = df.index[idx].date()
        
        # Reset range tracking for new day
        if self.last_range_date != current_date:
            self.daily_range_high = None
            self.daily_range_low = None
            self.range_established = False
            self.last_range_date = current_date
        
        if idx < self.params.range_duration_minutes:
            # Still in the opening range period
            if not self.range_established:
                # Initialize range
                period_data = df.iloc[:idx+1]
                self.daily_range_high = period_data['high'].max()
                self.daily_range_low = period_data['low'].min()
            else:
                # Update range
                self.daily_range_high = max(self.daily_range_high, df.iloc[idx]['high'])
                self.daily_range_low = min(self.daily_range_low, df.iloc[idx]['low'])
        elif idx == self.params.range_duration_minutes:
            # Just finished establishing the range
            period_data = df.iloc[:idx+1]
            self.daily_range_high = period_data['high'].max()
            self.daily_range_low = period_data['low'].min()
            self.range_established = True
    
    def generate_signals(self, data: pd.DataFrame, current_idx: int) -> Signal:
        """Generate trading signal at current index"""
        # Initialize neutral signal
        signal = Signal()
        signal.signal_type = SignalType.NEUTRAL
        signal.price = 0.0
        signal.stop_loss = 0.0
        signal.take_profit = 0.0
        signal.position_size = self.params.base_lot_size
        signal.metadata = {}
        
        if current_idx < 1:  # Need at least one previous candle
            return signal
            
        row = df := self.add_indicators(data) if not hasattr(self, '_cached_indicators') else self._cached_indicators
        if not hasattr(self, '_cached_indicators') or len(self._cached_indicators) != len(data):
            self._cached_indicators = self.add_indicators(data)
            row = self._cached_indicators.iloc[current_idx]
        else:
            row = self._cached_indicators.iloc[current_idx]
            
        prev_row = self._cached_indicators.iloc[current_idx-1]
        
        # Update daily range
        self.update_daily_range(self._cached_indicators, current_idx)
        
        # Only trade after range is established
        if not self.range_established:
            return signal
            
        # Only trade during US session after opening range
        if not self.is_us_session_open(row.name):
            return signal
        
        # Check for wick rejections at range levels
        return self.check_wick_rejection(row, prev_row)
    
    def check_wick_rejection(self, row, prev_row) -> Signal:
        """Check for wick rejections at range highs/lows"""
        signal = Signal()
        signal.signal_type = SignalType.NEUTRAL
        
        # Avoid division by zero
        if prev_row['body_size'] == 0:
            return signal
            
        # Check for upper wick rejection at range high (potential short)
        if (row['high'] > self.daily_range_high and 
            prev_row['high'] > self.daily_range_high and
            prev_row['upper_wick_ratio'] > self.params.wick_threshold_ratio and
            prev_row['is_bearish']):  # Previous candle closed bearish after wick
            
            signal.signal_type = SignalType.SHORT
            signal.price = prev_row['close']
            signal.stop_loss = self.daily_range_high + (row['atr'] * self.params.stop_loss_atr_mult)
            signal.take_profit = signal.price - self.params.profit_target_points
            signal.position_size = self.calculate_position_size(row['atr'])
            signal.metadata = {
                'signal_type': 'WICK_REJECTION_HIGH',
                'range_high': self.daily_range_high,
                'range_low': self.daily_range_low,
                'wick_ratio': prev_row['upper_wick_ratio']
            }
            
        # Check for lower wick rejection at range low (potential long)
        elif (row['low'] < self.daily_range_low and 
              prev_row['low'] < self.daily_range_low and
              prev_row['lower_wick_ratio'] > self.params.wick_threshold_ratio and
              prev_row['is_bullish']):  # Previous candle closed bullish after wick
            
            signal.signal_type = SignalType.LONG
            signal.price = prev_row['close']
            signal.stop_loss = self.daily_range_low - (row['atr'] * self.params.stop_loss_atr_mult)
            signal.take_profit = signal.price + self.params.profit_target_points
            signal.position_size = self.calculate_position_size(row['atr'])
            signal.metadata = {
                'signal_type': 'WICK_REJECTION_LOW',
                'range_high': self.daily_range_high,
                'range_low': self.daily_range_low,
                'wick_ratio': prev_row['lower_wick_ratio']
            }
        
        return signal
    
    def calculate_position_size(self, atr_value: float) -> float:
        """Calculate dynamic position size based on ATR volatility"""
        # Normalize ATR (typical NQ 5m ATR is around 10-20 points)
        volatility_factor = min(2.0, max(0.5, atr_value / 15.0))
        raw_size = self.params.base_lot_size * (0.5 + volatility_factor * 0.5)  # Scale 0.5x to 1.5x base
        return min(self.params.max_lot_size, max(0.005, raw_size))  # Clamp between 0.005 and max_lot_size
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameter values for DEAP optimization"""
        return {
            'session_start_hour': self.params.session_start_hour,
            'session_start_minute': self.params.session_start_minute,
            'range_duration_minutes': self.params.range_duration_minutes,
            'wick_threshold_ratio': self.params.wick_threshold_ratio,
            'profit_target_points': self.params.profit_target_points,
            'stop_loss_atr_mult': self.params.stop_loss_atr_mult,
            'base_lot_size': self.params.base_lot_size,
            'max_lot_size': self.params.max_lot_size,
            'atr_period': self.params.atr_period
        }
    
    def set_parameters(self, params: Dict[str, Any]):
        """Set parameters from DEAP optimization"""
        for key, value in params.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)