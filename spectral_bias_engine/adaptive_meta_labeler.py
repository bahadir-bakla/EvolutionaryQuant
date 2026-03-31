"""
Adaptive Meta Labeler — EvolutionaryQuant
==========================================
Contextual Bandit / Online Weighting Engine

Dynamically weights the predictions of 'fft_bias' and 'hmm_regime'
based on how correct they were yesterday. Outputs a final "meta_bias" score 
(-1.0 to 1.0) for strategies to consume.

This substitutes massive RL overhead with an ultra-fast, robust PnL-based 
Thompson Sampling or simple Exponential Moving Average (EMA) of accuracy.
"""

import numpy as np
import pandas as pd
from typing import Dict

class AdaptiveRegimeLabeler:
    """
    Online learning agent that weights different directional models 
    based on out-of-sample forward returns.
    """
    def __init__(self, learning_rate: float = 0.10):
        # Initial equal weights [FFT, HMM]
        self.weights = np.array([0.5, 0.5])
        self.learning_rate = learning_rate
        
        # Keep track of yesterday's raw predictions to score them today
        self.last_predictions = np.array([0.0, 0.0])
        self.initialized = False

    def update(self, actual_return: float):
        """
        Rewards/punishes the weights based on exactly what happened.
        actual_return is (Close_Today - Close_Yesterday).
        """
        if not self.initialized:
            self.initialized = True
            return

        # Direction of market (-1 or 1 or 0)
        market_dir = np.sign(actual_return) if abs(actual_return) > 1e-6 else 0.0
        if market_dir == 0:
            return

        # Core logic: if model predicted SAME direction as market, increase its weight
        # If it predicted opposite, decrease its weight
        for i in range(len(self.weights)):
            model_dir = np.sign(self.last_predictions[i])
            
            # Did it get the direction right? (1 for correct, -1 for wrong, 0 for neutral)
            accuracy = model_dir * market_dir
            
            # Simple gradient update step
            self.weights[i] += self.learning_rate * accuracy
            
        # Normalize weights back to [0, 1] sum=1 so they don't blow up
        self.weights = np.clip(self.weights, 0.01, 1.0) # floor at 0.01 to prevent death
        self.weights = self.weights / np.sum(self.weights)

    def predict(self, fft_bias: float, hmm_label: str) -> float:
        """
        Blend outputs based on current weights.
        
        fft_bias: -1.0 to 1.0 depending on cycle derivative
        hmm_label: 'bull', 'bear', 'ranging'
        """
        # Convert text label to float score
        hmm_score = 1.0 if hmm_label == 'bull' else (-1.0 if hmm_label == 'bear' else 0.0)
        
        # Save raw predictions BEFORE blending
        raw_preds = np.array([fft_bias, hmm_score])
        self.last_predictions = raw_preds
        
        # Final dot product
        meta_score = np.dot(self.weights, raw_preds)
        
        # Clip to safe bounds
        return float(np.clip(meta_score, -1.0, 1.0))

def apply_adaptive_meta_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates the online learning process historically.
    Requires 'spectral_bias' and 'hmm_regime' columns to exist.
    """
    if 'spectral_bias' not in df.columns or 'hmm_regime' not in df.columns:
        raise ValueError("DataFrame missing required columns ('spectral_bias', 'hmm_regime')")

    df = df.copy()
    labeler = AdaptiveRegimeLabeler(learning_rate=0.05)
    
    meta_scores = np.zeros(len(df))
    fft_weights = np.zeros(len(df))
    hmm_weights = np.zeros(len(df))
    
    # Calculate simple 1-bar forward returns to act as truth label
    returns = df['close'].diff().shift(-1).fillna(0).values
    
    fft_biases  = df['spectral_bias'].values
    hmm_regimes = df['hmm_regime'].values
    
    for i in range(len(df)):
        # 1. Update based on *yesterday's* prediction against *today's* actual return
        # Since 'returns' looks forward, at index `i-1` returns[i-1] is the change happening closing at `i`
        if i > 0:
            actual = returns[i-1]
            labeler.update(actual)
            
        # 2. Predict today
        f_b = fft_biases[i]
        h_r = hmm_regimes[i]
        score = labeler.predict(f_b, h_r)
        
        # 3. Store
        meta_scores[i] = score
        fft_weights[i] = labeler.weights[0]
        hmm_weights[i] = labeler.weights[1]
        
    df['meta_bias']  = meta_scores
    df['w_fft']      = fft_weights
    df['w_hmm']      = hmm_weights
    
    return df
