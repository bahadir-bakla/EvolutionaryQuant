# NQ Quant Bot - Hidden Markov Model for Regime Detection
# 3 Hidden States: Bull, Bear, Choppy

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("hmmlearn not installed. Run: pip install hmmlearn")


class MarketRegime(Enum):
    BULL = "BULL"           # Trending up, momentum strategy
    BEAR = "BEAR"           # Trending down, short or hedge
    CHOPPY = "CHOPPY"       # Range-bound, mean reversion


@dataclass
class RegimeState:
    """Current regime state from HMM"""
    current_regime: MarketRegime
    regime_probability: float
    bull_prob: float
    bear_prob: float
    choppy_prob: float
    regime_duration: int  # Bars in current regime
    transition_prob: float  # Probability of regime change


class HMMRegimeDetector:
    """
    Hidden Markov Model for Market Regime Detection
    
    Uses returns and volatility as observables
    3 hidden states: Bull, Bear, Choppy
    """
    
    def __init__(self, n_states: int = 3, lookback: int = 252):
        if not HMM_AVAILABLE:
            raise ImportError("hmmlearn is required. Install with: pip install hmmlearn")
        
        self.n_states = n_states
        self.lookback = lookback
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.is_trained = False
        self.state_mapping = {}  # Maps HMM states to regime names
        
        # Suppress convergence warnings
        from warnings import simplefilter
        simplefilter(action='ignore', category=FutureWarning)
        simplefilter(action='ignore', category=UserWarning)
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare observation features for HMM
        
        Features:
        1. Daily returns
        2. Rolling volatility
        3. Return momentum (5-day)
        """
        close = df['close']
        
        # Daily returns
        returns = close.pct_change()
        
        # 10-day volatility
        volatility = returns.rolling(10).std()
        
        # 5-day momentum
        momentum = close.pct_change(5)
        
        # Combine
        features = pd.DataFrame({
            'returns': returns,
            'volatility': volatility,
            'momentum': momentum
        })
        
        # Handle inf/nan
        features = features.replace([np.inf, -np.inf], np.nan).dropna()
        
        return features.values
    
    def train(self, df: pd.DataFrame) -> None:
        """Train HMM on historical data"""
        X = self.prepare_features(df)
        
        if len(X) < 50:
            raise ValueError("Insufficient data for training (need at least 50 bars)")
        
        # Fit model
        self.model.fit(X)
        self.is_trained = True
        
        # Identify states by their characteristics
        # Get mean returns for each state
        state_returns = []
        hidden_states = self.model.predict(X)
        features_df = pd.DataFrame(X, columns=['returns', 'volatility', 'momentum'])
        
        for state in range(self.n_states):
            mask = hidden_states == state
            mean_return = features_df.loc[mask, 'returns'].mean()
            mean_vol = features_df.loc[mask, 'volatility'].mean()
            state_returns.append((state, mean_return, mean_vol))
        
        # Sort by mean return
        state_returns.sort(key=lambda x: x[1], reverse=True)
        
        # Map states: highest return = Bull, lowest = Bear, middle = Choppy
        self.state_mapping = {
            state_returns[0][0]: MarketRegime.BULL,
            state_returns[2][0]: MarketRegime.BEAR,
            state_returns[1][0]: MarketRegime.CHOPPY
        }
        
    def predict(self, df: pd.DataFrame) -> RegimeState:
        """Predict current regime"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.prepare_features(df)
        
        # Get most likely state sequence
        hidden_states = self.model.predict(X)
        
        # Get probabilities for current state
        probs = self.model.predict_proba(X)
        current_probs = probs[-1]
        current_state = hidden_states[-1]
        
        # Map to regime
        current_regime = self.state_mapping[current_state]
        
        # Calculate regime duration
        duration = 1
        for i in range(len(hidden_states) - 2, -1, -1):
            if hidden_states[i] == current_state:
                duration += 1
            else:
                break
        
        # Get transition probability (from transition matrix)
        trans_matrix = self.model.transmat_
        transition_prob = 1 - trans_matrix[current_state, current_state]
        
        # Map probabilities to regimes
        bull_state = [s for s, r in self.state_mapping.items() if r == MarketRegime.BULL][0]
        bear_state = [s for s, r in self.state_mapping.items() if r == MarketRegime.BEAR][0]
        choppy_state = [s for s, r in self.state_mapping.items() if r == MarketRegime.CHOPPY][0]
        
        return RegimeState(
            current_regime=current_regime,
            regime_probability=current_probs[current_state],
            bull_prob=current_probs[bull_state],
            bear_prob=current_probs[bear_state],
            choppy_prob=current_probs[choppy_state],
            regime_duration=duration,
            transition_prob=transition_prob
        )
    
    def get_regime_history(self, df: pd.DataFrame) -> pd.Series:
        """Get regime for each bar in history"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.prepare_features(df)
        hidden_states = self.model.predict(X)
        
        # Map to regime names
        regimes = [self.state_mapping[s].value for s in hidden_states]
        
        # Create series with proper index
        idx = df.index[len(df) - len(regimes):]
        return pd.Series(regimes, index=idx)
    
    def get_strategy_recommendation(self, regime_state: RegimeState) -> dict:
        """
        Get trading strategy recommendation based on regime
        """
        recommendations = {
            MarketRegime.BULL: {
                'strategy': 'TREND_FOLLOWING',
                'bias': 'LONG',
                'position_size_mult': 1.2,  # Larger positions
                'stop_loss_mult': 1.5,      # Wider stops
                'take_profit_mult': 3.0,    # Larger targets
                'description': 'Uptrend - Look for pullback longs, trail stops'
            },
            MarketRegime.BEAR: {
                'strategy': 'TREND_FOLLOWING',
                'bias': 'SHORT',
                'position_size_mult': 1.0,
                'stop_loss_mult': 1.5,
                'take_profit_mult': 2.5,
                'description': 'Downtrend - Short rallies, protect capital'
            },
            MarketRegime.CHOPPY: {
                'strategy': 'MEAN_REVERSION',
                'bias': 'NEUTRAL',
                'position_size_mult': 0.7,  # Smaller positions
                'stop_loss_mult': 1.0,      # Tighter stops
                'take_profit_mult': 1.5,    # Quick profits
                'description': 'Range-bound - Fade extremes, quick exits'
            }
        }
        
        return recommendations[regime_state.current_regime]


# Simple fallback when hmmlearn not available
class SimpleRegimeDetector:
    """
    Simple regime detection without HMM
    Uses moving averages and volatility
    """
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
    
    def detect(self, df: pd.DataFrame) -> RegimeState:
        """Detect regime using simple rules"""
        close = df['close']
        
        # Trend detection
        ema_20 = close.ewm(span=20).mean()
        ema_50 = close.ewm(span=50).mean()
        
        current_price = close.iloc[-1]
        current_ema20 = ema_20.iloc[-1]
        current_ema50 = ema_50.iloc[-1]
        
        # Volatility
        returns = close.pct_change()
        current_vol = returns.iloc[-20:].std()
        avg_vol = returns.iloc[-100:].std() if len(df) > 100 else current_vol
        
        # Determine regime
        if current_price > current_ema20 > current_ema50:
            if current_vol < avg_vol * 1.2:
                regime = MarketRegime.BULL
                prob = 0.8
            else:
                regime = MarketRegime.CHOPPY
                prob = 0.6
        elif current_price < current_ema20 < current_ema50:
            if current_vol < avg_vol * 1.2:
                regime = MarketRegime.BEAR
                prob = 0.8
            else:
                regime = MarketRegime.CHOPPY
                prob = 0.6
        else:
            regime = MarketRegime.CHOPPY
            prob = 0.7
        
        # Calculate approximate probabilities
        if regime == MarketRegime.BULL:
            bull_prob, bear_prob, choppy_prob = 0.7, 0.1, 0.2
        elif regime == MarketRegime.BEAR:
            bull_prob, bear_prob, choppy_prob = 0.1, 0.7, 0.2
        else:
            bull_prob, bear_prob, choppy_prob = 0.2, 0.2, 0.6
        
        return RegimeState(
            current_regime=regime,
            regime_probability=prob,
            bull_prob=bull_prob,
            bear_prob=bear_prob,
            choppy_prob=choppy_prob,
            regime_duration=1,
            transition_prob=0.1
        )


def get_regime_detector(use_hmm: bool = True):
    """Factory function to get appropriate detector"""
    if use_hmm and HMM_AVAILABLE:
        return HMMRegimeDetector()
    else:
        return SimpleRegimeDetector()


# === TEST ===
if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing HMM Regime Detection...")
    
    # Fetch data
    ticker = yf.Ticker("NQ=F")
    df = ticker.history(period="2y", interval="1d")
    df.columns = df.columns.str.lower()
    print(f"Fetched {len(df)} bars")
    
    if HMM_AVAILABLE:
        print("\nUsing HMM Regime Detector...")
        detector = HMMRegimeDetector()
        detector.train(df)
    else:
        print("\nUsing Simple Regime Detector (hmmlearn not installed)...")
        detector = SimpleRegimeDetector()
    
    # Get current regime
    if HMM_AVAILABLE:
        regime = detector.predict(df)
        
        # Get history
        regime_history = detector.get_regime_history(df)
        regime_counts = regime_history.value_counts()
        print("\nRegime Distribution:")
        for r, count in regime_counts.items():
            print(f"  {r}: {count} bars ({count/len(regime_history)*100:.1f}%)")
    else:
        regime = detector.detect(df)
    
    print(f"\n{'='*60}")
    print("CURRENT REGIME ANALYSIS")
    print(f"{'='*60}")
    print(f"Regime: {regime.current_regime.value}")
    print(f"Confidence: {regime.regime_probability:.1%}")
    print(f"Bull Prob: {regime.bull_prob:.1%}")
    print(f"Bear Prob: {regime.bear_prob:.1%}")
    print(f"Choppy Prob: {regime.choppy_prob:.1%}")
    print(f"Duration: {regime.regime_duration} bars")
    print(f"Transition Prob: {regime.transition_prob:.1%}")
    
    # Get recommendation
    if HMM_AVAILABLE:
        rec = detector.get_strategy_recommendation(regime)
        print(f"\nStrategy Recommendation:")
        print(f"  Strategy: {rec['strategy']}")
        print(f"  Bias: {rec['bias']}")
        print(f"  Position Size Mult: {rec['position_size_mult']}")
        print(f"  Description: {rec['description']}")
