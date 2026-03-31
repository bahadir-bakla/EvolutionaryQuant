"""
Regime Detection — EvolutionaryQuant
=====================================
Detects market regimes using Gaussian Mixture Models (as a robust HMM proxy).

Extracts 3 latent regimes based on:
1. Log Returns (Direction)
2. Range / True Range (Volatility)

Identified Regimes (automatically sorted):
- Ranging / Low Vol (Consolidation)
- Trending Bull (High return, Med/High Vol)
- Trending Bear (Low return, Med/High Vol)
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress sklearn warnings if matrix is ill-conditioned on small datasets
warnings.filterwarnings('ignore', category=UserWarning)

class RegimeDetector:
    """
    Fits a 3-State Gaussian Mixture Model to classify current market environment.
    State 0: Ranging (Low Vol)
    State 1: Bearish Trend (High Vol, Negative Returns)
    State 2: Bullish Trend (High Vol, Positive Returns)
    """
    def __init__(self, n_regimes: int = 3, lookback: int = 1500):
        self.n_regimes = n_regimes
        self.lookback  = lookback
        self.gmm       = GaussianMixture(n_components=n_regimes, covariance_type='full', max_iter=200, random_state=42)
        self.scaler    = StandardScaler()
        self.fitted    = False
        
        # State mapping dictionary 
        self.state_map = {0: 'Unknown', 1: 'Unknown', 2: 'Unknown'}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create Volatility and Return features for HMM/GMM clustering."""
        feat = pd.DataFrame(index=df.index)
        
        # Log Returns
        feat['returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        
        # True Range Volatility proxy
        if 'high' in df.columns and 'low' in df.columns:
            feat['tr_pct'] = (df['high'] - df['low']) / df['close']
        else:
            feat['tr_pct'] = feat['returns'].abs()
            
        # Optional: Smoothed measures so regime isn't just 1-bar noise
        feat['ret_ma'] = feat['returns'].rolling(window=5, min_periods=1).mean()
        feat['vol_ma'] = feat['tr_pct'].rolling(window=5, min_periods=1).mean()
        
        return feat[['ret_ma', 'vol_ma']].dropna()

    def fit(self, df: pd.DataFrame):
        """Fit distribution on historical rolling data."""
        # Use only recent lookback to capture modern market condition
        historical = df.tail(self.lookback).copy()
        
        # Need at least 50 bars to fit a 3-state GMM
        if len(historical) < 50:
            return
            
        X = self._prepare_features(historical)
        if len(X) < 50:
            return

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit GMM
        self.gmm.fit(X_scaled)
        
        # Predict on the same training set to find which state is which
        hidden_states = self.gmm.predict(X_scaled)
        historical.loc[X.index, 'state'] = hidden_states
        historical.loc[X.index, 'log_ret'] = X['ret_ma']
        historical.loc[X.index, 'vol']     = X['vol_ma']
        
        # Map states intelligently
        # State with lowest avg Volatility = Ranging
        # Out of remaining two, state with higher average return = Bull, other = Bear
        
        state_stats = historical.groupby('state')[['vol', 'log_ret']].mean()
        
        # If model collapsed and didn't find all 3 states, abort mapping
        if len(state_stats) < self.n_regimes:
            self.fitted = True
            return

        # 1. Ranging is lowest vol
        range_state = state_stats['vol'].idxmin()
        
        # 2. Bull/Bear are the others
        others = [s for s in state_stats.index if s != range_state]
        
        if state_stats.loc[others[0], 'log_ret'] > state_stats.loc[others[1], 'log_ret']:
            bull_state = others[0]
            bear_state = others[1]
        else:
            bull_state = others[1]
            bear_state = others[0]

        self.state_map[range_state] = 'ranging'
        self.state_map[bull_state]  = 'bull'
        self.state_map[bear_state]  = 'bear'
        
        self.fitted = True

    def predict_current_state(self, df_recent: pd.DataFrame) -> dict:
        """
        Takes recent history (e.g. 5-10 bars), returns dict with current regime mapping.
        """
        if not self.fitted:
            return {'state_id': 0, 'label': 'ranging', 'confidence': 0.0}
            
        X = self._prepare_features(df_recent)
        if len(X) == 0:
            return {'state_id': 0, 'label': 'ranging', 'confidence': 0.0}
            
        # Get the VERY LAST row 
        x_last = X.iloc[[-1]]
        x_scaled = self.scaler.transform(x_last)
        
        # Predict probabilities
        probs = self.gmm.predict_proba(x_scaled)[0]
        state_id = int(np.argmax(probs))
        conf     = float(probs[state_id])
        
        label = self.state_map.get(state_id, 'ranging')

        return {
            'state_id': state_id,
            'label': label,
            'confidence': conf
        }

def add_regime_features(df: pd.DataFrame, lookback: int = 1500) -> pd.DataFrame:
    """
    Pandas convenience function. Batch labels a historical DataFrame.
    Retrains on expanding window (slow) or rolling (very slow).
    For speed, we'll train once on the FIRST `lookback` portion and 
    update the model periodically (e.g., every 500 bars).
    """
    df = df.copy()
    detector = RegimeDetector(n_regimes=3, lookback=lookback)
    
    state_labels = []
    state_confs  = []
    
    # Needs at least 50 bars to start
    # We will do a staggered block update for speed 
    update_frequency = 500
    
    for i in range(len(df)):
        if i < 50:
            state_labels.append('ranging')
            state_confs.append(0.0)
            continue
            
        # Retrain every `update_frequency` bars
        if i == 50 or (i % update_frequency == 0):
            # Train using data up to `i`
            train_window = df.iloc[max(0, i-lookback):i]
            detector.fit(train_window)
            
        res = detector.predict_current_state(df.iloc[max(0, i-10):i+1])
        state_labels.append(res['label'])
        state_confs.append(res['confidence'])
        
    df['hmm_regime'] = state_labels
    df['hmm_confidence'] = state_confs
    
    return df
