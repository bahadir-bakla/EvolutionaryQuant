"""
Spectral Bias Engine — EvolutionaryQuant
=========================================
Fourier-based cycle extraction for dominant market frequency.

Processes price history using Fast Fourier Transform (FFT) to isolate 
the top N dominant wave frequencies (cycles). Reconstructs a composite 
wave and measures its current instantaneous slope (derivative) to output 
a mathematical directional bias (+1 Bullish, -1 Bearish).
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from collections import deque

class SpectralCyclePredictor:
    """
    Online rolling FFT predictor for Market Bias.
    """
    def __init__(self, 
                 window_size: int = 120, 
                 top_n_cycles: int = 3,
                 min_period: int = 5,
                 max_period: int = 60,
                 detrend: bool = True):
        """
        window_size:   Lookback window for FFT (must be stationary enough)
        top_n_cycles:  How many dominant frequencies to blend
        min_period:    Ignore noise cycles shorter than this (e.g. 5 days)
        max_period:    Ignore macro cycles longer than this
        detrend:       Whether to remove linear trend before FFT
        """
        self.window_size  = window_size
        self.top_n_cycles = top_n_cycles
        self.min_period   = min_period
        self.max_period   = max_period
        self.detrend      = detrend
        
        self.cycle_periods= []   # last detected dominant periods
        self.signal_history= deque(maxlen=window_size)

    def _extract_cycles(self, prices: np.ndarray) -> Tuple[float, float, float]:
        """
        Core FFT Logic.
        Returns: current_cycle_value, next_cycle_value, predicted_slope
        """
        n = len(prices)
        if n < 20:
            return 0.0, 0.0, 0.0

        # Optional: Detrending
        if self.detrend:
            x = np.arange(n)
            coefs = np.polyfit(x, prices, 1)
            poly  = np.poly1d(coefs)
            trend = poly(x)
            detrended = prices - trend
        else:
            detrended = prices - np.mean(prices)

        # Apply Hamming window to reduce edge effects (spectral leakage)
        window = np.hamming(n)
        d_windowed = detrended * window

        # Perform FFT
        fft_result = np.fft.rfft(d_windowed)
        frequencies = np.fft.rfftfreq(n, d=1.0)  # cycles per bar

        # Calculate Power Spectral Density (PSD)
        psd = np.abs(fft_result) ** 2

        # Filter out DC component (freq=0) and frequencies outside bounds
        valid_mask = (frequencies > 0)
        
        # Convert constraints from periods to frequencies
        # freq = 1 / period
        max_f = 1.0 / self.min_period
        min_f = 1.0 / self.max_period if self.max_period <= n else 1.0 / n
        
        valid_mask &= (frequencies >= min_f) & (frequencies <= max_f)

        if not np.any(valid_mask):
            return 0.0, 0.0, 0.0

        psd[~valid_mask] = 0.0
        
        # Sort indices by descending power
        peak_indices = np.argsort(psd)[::-1]
        top_indices  = peak_indices[:self.top_n_cycles]
        
        self.cycle_periods = [int(1.0 / frequencies[i]) for i in top_indices if frequencies[i] > 0]

        # Reconstruct composite wave using only top N frequencies
        composite_current = 0.0
        composite_next    = 0.0
        
        for idx in top_indices:
            if psd[idx] == 0:
                continue
            
            amp = np.abs(fft_result[idx]) / n * 2.0
            phase = np.angle(fft_result[idx])
            freq = frequencies[idx]
            
            # evaluate wave at t = n - 1 (current bar)
            # Standard DFT formula: Re( A * exp(i * (2*pi*f*t + phase)) )
            val_now   = amp * np.cos(2.0 * np.pi * freq * (n - 1) + phase)
            # evaluate wave at t = n (predicted next bar)
            val_next  = amp * np.cos(2.0 * np.pi * freq * n + phase)
            
            composite_current += val_now
            composite_next    += val_next

        slope = composite_next - composite_current
        return composite_current, composite_next, slope

    def predict(self, recent_prices: np.ndarray) -> dict:
        """
        Run spectral analysis and return bias prediction.
        recent_prices: 1D numpy array of length `self.window_size`
        """
        if len(recent_prices) != self.window_size:
            # Fallback if not enough data
            recent_prices = np.asarray(recent_prices)
        
        c_now, c_next, dP = self._extract_cycles(recent_prices)
        
        # Determine bias direction:
        # +1 if wave is sloping up, -1 if sloping down
        # Confidence logic: if slope is steep, higher confidence
        if abs(dP) > 1e-6:
            direction = 1 if dP > 0 else -1
            strength  = min(abs(dP) / np.std(recent_prices) * 10, 1.0) if np.std(recent_prices) > 0 else 0
        else:
            direction = 0
            strength  = 0.0
            
        return {
            'bias_direction': direction,
            'bias_strength':  float(strength),
            'dominant_periods': self.cycle_periods,
            'cycle_value_current': c_now,
            'cycle_value_forecast': c_next
        }

def add_spectral_features(df: pd.DataFrame, window_size: int = 120, col_name: str = 'close', step: int = 10) -> pd.DataFrame:
    """
    Pandas convenience function. 
    Optimized: Calculates every `step` bars and forward fills.
    """
    df = df.copy()
    predictor = SpectralCyclePredictor(window_size=window_size)
    
    bias_dir = np.full(len(df), np.nan)
    bias_str = np.full(len(df), np.nan)
    periods  = np.full(len(df), np.nan)
    
    prices = df[col_name].values
    for i in range(window_size, len(prices), step):
        windowed = prices[i-window_size:i]
        res = predictor.predict(windowed)
        bias_dir[i] = res['bias_direction']
        bias_str[i] = res['bias_strength']
        if res['dominant_periods']:
            periods[i]  = res['dominant_periods'][0]
            
    df['spectral_bias']     = pd.Series(bias_dir).ffill().fillna(0.0).values
    df['spectral_strength'] = pd.Series(bias_str).ffill().fillna(0.0).values
    df['dominant_cycle']    = pd.Series(periods).ffill().fillna(0.0).values
    
    return df
