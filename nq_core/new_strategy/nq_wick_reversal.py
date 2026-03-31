import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class NQWickReversalConfig:
    # US Session parameters (EST)
    session_start_hour: int = 9    # 9:30 AM EST
    session_start_minute: int = 30
    range_duration_minutes: int = 15  # First 15 minutes
    # Wick reversal parameters
    wick_threshold_ratio: float = 0.3  # Minimum wick size as ratio of candle body
    # Profit target
    profit_target_points: float = 20.0  # 20 units profit target
    # Stop loss
    stop_loss_atr_mult: float = 1.5  # ATR multiplier for stop loss
    # Position sizing
    base_lot_size: float = 0.01
    max_lot_size: float = 0.05

class NQ_Wick_Reversal_Strategy:
    """
    NQ Wick Reversal Strategy:
    - Identifies US session opening range (first 15 minutes)
    - Looks for wick rejections at liquidation points (range highs/lows)
    - Enters in opposite direction of wick rejection
    - Fixed 20 point profit target
    """
    
    def __init__(self, config: NQWickReversalConfig = None):
        self.config = config or NQWickReversalConfig()
        self.daily_range_high = None
        self.daily_range_low = None
        self.range_established = False
        
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add required indicators for the strategy"""
        # Calculate ATR for dynamic stop loss
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(14).mean()
        
        # Calculate candle properties
        df['body_size'] = np.abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        
        # Wick strength as ratio of wick to body
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
        # Convert to EST if needed (assuming timestamp is in UTC or already adjusted)
        # For simplicity, we'll assume the data is already in EST or we're checking hour/minute
        hour = timestamp.hour
        minute = timestamp.minute
        
        start_time = self.config.session_start_hour * 60 + self.config.session_start_minute
        current_time = hour * 60 + minute
        end_time = start_time + self.config.range_duration_minutes
        
        return start_time <= current_time < end_time
    
    def update_daily_range(self, df: pd.DataFrame, idx: int):
        """Update the daily opening range based on first 15 minutes"""
        if idx < self.config.range_duration_minutes:
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
        elif idx == self.config.range_duration_minutes:
            # Just finished establishing the range
            period_data = df.iloc[:idx+1]
            self.daily_range_high = period_data['high'].max()
            self.daily_range_low = period_data['low'].min()
            self.range_established = True
    
    def evaluate(self, df: pd.DataFrame, idx: int):
        """Evaluate strategy at given index"""
        # Initialize signal object
        class Sig: pass
        signal = Sig()
        signal.direction = 'NEUTRAL'
        signal.entry_price = 0.0
        signal.stop_loss = 0.0
        signal.take_profit = 0.0
        signal.lot_size = self.config.base_lot_size
        
        if idx < 1:  # Need at least one previous candle
            return signal
            
        row = df.iloc[idx]
        prev_row = df.iloc[idx-1]
        
        # Update daily range if we're still in establishment period
        self.update_daily_range(df, idx)
        
        # Only trade after range is established
        if not self.range_established:
            return signal
            
        # Only trade during US session after opening range
        if not self.is_us_session_open(row.name):
            return signal
        
        # Check for wick rejections at range levels
        signal = self.check_wick_rejection(row, prev_row, signal)
        
        return signal
    
    def check_wick_rejection(self, row, prev_row, signal):
        """Check for wick rejections at range highs/lows"""
        # Avoid division by zero
        if prev_row['body_size'] == 0:
            return signal
            
        # Check for upper wick rejection at range high (potential short)
        if (row['high'] > self.daily_range_high and 
            prev_row['high'] > self.daily_range_high and
            prev_row['upper_wick_ratio'] > self.config.wick_threshold_ratio and
            prev_row['is_bearish']):  # Previous candle closed bearish after wick
            
            signal.direction = 'SHORT'
            signal.entry_price = prev_row['close']
            signal.stop_loss = self.daily_range_high + (row['atr'] * self.config.stop_loss_atr_mult)
            signal.take_profit = signal.entry_price - self.config.profit_target_points
            
            # Dynamic lot size based on ATR volatility
            volatility_factor = min(2.0, row['atr'] / 15.0)  # Normalize ATR
            signal.lot_size = min(
                self.config.max_lot_size,
                self.config.base_lot_size * (1.0 + volatility_factor)
            )
            
        # Check for lower wick rejection at range low (potential long)
        elif (row['low'] < self.daily_range_low and 
              prev_row['low'] < self.daily_range_low and
              prev_row['lower_wick_ratio'] > self.config.wick_threshold_ratio and
              prev_row['is_bullish']):  # Previous candle closed bullish after wick
            
            signal.direction = 'LONG'
            signal.entry_price = prev_row['close']
            signal.stop_loss = self.daily_range_low - (row['atr'] * self.config.stop_loss_atr_mult)
            signal.take_profit = signal.entry_price + self.config.profit_target_points
            
            # Dynamic lot size based on ATR volatility
            volatility_factor = min(2.0, row['atr'] / 15.0)  # Normalize ATR
            signal.lot_size = min(
                self.config.max_lot_size,
                self.config.base_lot_size * (1.0 + volatility_factor)
            )
        
        return signal

# Example usage function
def run_wick_reversal_backtest(df: pd.DataFrame):
    """Example backtest runner for the wick reversal strategy"""
    strategy = NQ_Wick_Reversal_Strategy()
    df_with_indicators = strategy.add_indicators(df.copy())
    
    balance = 1000.0
    trades = []
    
    for i in range(len(df_with_indicators)):
        signal = strategy.evaluate(df_with_indicators, i)
        
        if signal.direction != 'NEUTRAL':
            # Simulate trade execution
            entry_price = signal.entry_price
            exit_price = None
            pnl = 0.0
            
            # Check if stop loss or take profit was hit in subsequent candles
            for j in range(i+1, min(i+50, len(df_with_indicators))):  # Max 50 bars lookahead
                future_row = df_with_indicators.iloc[j]
                
                if signal.direction == 'LONG':
                    if future_row['low'] <= signal.stop_loss:
                        exit_price = signal.stop_loss
                        break
                    elif future_row['high'] >= signal.take_profit:
                        exit_price = signal.take_profit
                        break
                else:  # SHORT
                    if future_row['high'] >= signal.stop_loss:
                        exit_price = signal.stop_loss
                        break
                    elif future_row['low'] <= signal.take_profit:
                        exit_price = signal.take_profit
                        break
            
            # If neither hit, close at last price
            if exit_price is None:
                exit_price = df_with_indicators.iloc[-1]['close']
            
            # Calculate P&L
            if signal.direction == 'LONG':
                pnl = (exit_price - entry_price) * 20.0 * signal.lot_size  # NQ $20/point
            else:
                pnl = (entry_price - exit_price) * 20.0 * signal.lot_size
            
            balance += pnl
            trades.append({
                'entry_time': df_with_indicators.index[i],
                'direction': signal.direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'balance': balance,
                'lot_size': signal.lot_size
            })
    
    return balance, trades