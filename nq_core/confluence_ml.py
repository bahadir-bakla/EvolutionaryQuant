# NQ Quant Bot - ML-Enhanced Confluence Engine v3
# Integrates ML, Multi-TF, HMM, and Adaptive Sizing

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nq_core.confluence_v2 import EnhancedConfluenceEngine, EnhancedSignal
from nq_core.brain import QuantBrain, BrainState
from nq_core.order_blocks import detect_order_blocks, get_active_order_blocks
from nq_core.indicators import calculate_vwap, calculate_adx, detect_fvg, calculate_pivots
from nq_core.indicators.market_structure import detect_bos_choch

# ML modules
from nq_core.ml.multi_timeframe import (
    fetch_multi_timeframe_data, analyze_multi_timeframe, get_mtf_filter, MTFBias
)
from nq_core.ml.xgboost_classifier import SignalClassifier, XGBOOST_AVAILABLE
from nq_core.ml.hmm_regime import get_regime_detector, MarketRegime, HMM_AVAILABLE
from nq_core.ml.position_sizing import AdaptivePositionSizer, PositionSize


@dataclass
class MLEnhancedSignal:
    """ML-enhanced trading signal"""
    # Base signal
    direction: str
    confidence: float
    entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # ML enhancements
    ml_signal: str        # XGBoost prediction
    ml_probability: float
    
    # Regime
    regime: str
    regime_confidence: float
    
    # Multi-TF
    mtf_bias: str
    mtf_alignment: float
    mtf_filter_passed: bool
    
    # Position sizing
    position_size: Optional[PositionSize]
    
    # Combined score
    combined_score: float
    
    # Metadata
    factors: Dict[str, Any]


class MLConfluenceEngine:
    """
    ML-Enhanced Confluence Engine v3
    
    Layers:
    1. Multi-Timeframe Filter (Weekly/Daily/4H alignment)
    2. HMM Regime Detection (Bull/Bear/Choppy)
    3. XGBoost Signal Classification
    4. Technical Confluence (from v2)
    5. Adaptive Position Sizing
    """
    
    def __init__(
        self,
        use_xgboost: bool = True,
        use_hmm: bool = True,
        min_factors: int = 4,
        min_score: float = 2.5
    ):
        # Technical confluence
        self.tech_engine = EnhancedConfluenceEngine(
            min_factors=min_factors,
            min_score=min_score
        )
        
        # ML components
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        self.use_hmm = use_hmm
        
        if self.use_xgboost:
            self.classifier = SignalClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1
            )
            self.ml_trained = False
        else:
            self.classifier = None
            self.ml_trained = False
        
        # Regime detector
        self.regime_detector = get_regime_detector(use_hmm=use_hmm)
        self.regime_trained = False
        
        # Position sizer
        self.position_sizer = AdaptivePositionSizer(
            base_risk_percent=0.02,
            max_risk_percent=0.05,
            use_kelly=True
        )
        
        # State
        self.mtf_analysis = None
        self.current_regime = None
        self.peak_equity = 100000
        
    def train(self, df: pd.DataFrame):
        """Train ML models on historical data"""
        if self.use_xgboost:
            print("Training XGBoost classifier...")
            results = self.classifier.walk_forward_train(df, n_splits=5)
            avg_acc = np.mean([r.accuracy for r in results])
            print(f"  Average accuracy: {avg_acc:.1%}")
            self.ml_trained = True
        
        if self.use_hmm and hasattr(self.regime_detector, 'train'):
            print("Training HMM regime detector...")
            self.regime_detector.train(df)
            self.regime_trained = True
    
    def update_mtf(self, symbol: str = "NQ=F"):
        """Update multi-timeframe analysis"""
        print("Fetching multi-timeframe data...")
        mtf_data = fetch_multi_timeframe_data(symbol)
        self.mtf_analysis = analyze_multi_timeframe(mtf_data)
        print(f"  MTF Bias: {self.mtf_analysis.overall_bias.value}")
    
    def evaluate(
        self,
        df: pd.DataFrame,
        current_idx: int,
        brain_state: BrainState,
        order_blocks: List,
        pivots,
        fvgs: List,
        capital: float = 100000,
        rsi: Optional[float] = None,
        atr: Optional[float] = None
    ) -> MLEnhancedSignal:
        """
        Generate ML-enhanced signal
        """
        row = df.iloc[current_idx]
        price = row['close']
        
        # === 1. TECHNICAL CONFLUENCE ===
        tech_signal = self.tech_engine.evaluate(
            df, current_idx, brain_state, order_blocks, pivots, fvgs, rsi, atr
        )
        
        # === 2. XGBoost PREDICTION ===
        if self.ml_trained and self.use_xgboost:
            ml_pred = self.classifier.predict(df.iloc[:current_idx+1])
            ml_signal = ml_pred.signal
            ml_prob = ml_pred.probability
        else:
            ml_signal = 'NEUTRAL'
            ml_prob = 0.5
        
        # === 3. REGIME DETECTION ===
        if self.regime_trained or not hasattr(self.regime_detector, 'train'):
            if hasattr(self.regime_detector, 'predict'):
                regime_state = self.regime_detector.predict(df.iloc[:current_idx+1])
            else:
                regime_state = self.regime_detector.detect(df.iloc[:current_idx+1])
            regime = regime_state.current_regime.value
            regime_conf = regime_state.regime_probability
        else:
            regime = 'CHOPPY'
            regime_conf = 0.5
        
        # === 4. MULTI-TIMEFRAME FILTER ===
        if self.mtf_analysis:
            mtf_passed, mtf_mult = get_mtf_filter(self.mtf_analysis, tech_signal.direction)
            mtf_bias = self.mtf_analysis.overall_bias.value
            mtf_align = self.mtf_analysis.alignment_score
        else:
            mtf_passed = True
            mtf_mult = 1.0
            mtf_bias = 'NEUTRAL'
            mtf_align = 0.5
        
        # === 5. COMBINED SIGNAL ===
        
        # Vote counting
        votes = {
            'LONG': 0,
            'SHORT': 0,
            'NEUTRAL': 0
        }
        
        # Technical signal (weight: 3)
        votes[tech_signal.direction] += 3 * tech_signal.confidence
        
        # ML signal (weight: 2)
        votes[ml_signal] += 2 * ml_prob
        
        # Regime bias (weight: 1.5)
        if regime == 'BULL':
            votes['LONG'] += 1.5 * regime_conf
        elif regime == 'BEAR':
            votes['SHORT'] += 1.5 * regime_conf
        
        # MTF bias (weight: 2)
        if mtf_passed:
            if mtf_bias in ['STRONG_BULLISH', 'BULLISH']:
                votes['LONG'] += 2 * mtf_align
            elif mtf_bias in ['STRONG_BEARISH', 'BEARISH']:
                votes['SHORT'] += 2 * mtf_align
        
        # Determine final direction
        max_vote = max(votes.values())
        final_direction = [k for k, v in votes.items() if v == max_vote][0]
        
        # Combined score
        combined_score = max_vote
        
        # MTF Filter override
        if not mtf_passed and final_direction != 'NEUTRAL':
            # Counter-trend to strong MTF - reduce confidence or skip
            combined_score *= 0.5
        
        # Final confidence
        total_votes = sum(votes.values()) + 0.01
        final_confidence = votes[final_direction] / total_votes
        
        # === 6. POSITION SIZING ===
        if final_direction != 'NEUTRAL' and atr:
            if final_direction == 'LONG':
                stop = price - (atr * 2.5)
                tp1 = price + (atr * 2)
                tp2 = price + (atr * 3)
                tp3 = price + (atr * 4)
            else:
                stop = price + (atr * 2.5)
                tp1 = price - (atr * 2)
                tp2 = price - (atr * 3)
                tp3 = price - (atr * 4)
            
            position = self.position_sizer.calculate_position_size(
                capital=capital,
                signal=final_direction,
                confidence=final_confidence,
                regime=regime,
                current_equity=capital,
                peak_equity=self.peak_equity,
                current_atr=atr,
                avg_atr=df['atr'].mean() if 'atr' in df.columns else atr,
                entry_price=price,
                stop_loss=stop
            )
        else:
            stop = price
            tp1 = tp2 = tp3 = price
            position = None
        
        return MLEnhancedSignal(
            direction=final_direction,
            confidence=final_confidence,
            entry=price,
            stop_loss=stop,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            ml_signal=ml_signal,
            ml_probability=ml_prob,
            regime=regime,
            regime_confidence=regime_conf,
            mtf_bias=mtf_bias,
            mtf_alignment=mtf_align,
            mtf_filter_passed=mtf_passed,
            position_size=position,
            combined_score=combined_score,
            factors={
                'tech_direction': tech_signal.direction,
                'tech_confidence': tech_signal.confidence,
                'tech_factors': tech_signal.active_factors,
                'ml_long_prob': ml_prob if ml_signal == 'LONG' else 0,
                'ml_short_prob': ml_prob if ml_signal == 'SHORT' else 0,
                'votes': votes
            }
        )


def run_ml_backtest(period: str = '2y'):
    """Run backtest with ML-enhanced engine"""
    import yfinance as yf
    from nq_core.backtest import NQBacktestEngine
    
    print('='*60)
    print('NQ QUANT BOT - ML ENHANCED BACKTEST')
    print('='*60)
    
    # Fetch data
    print('\n[1] Fetching NQ=F data...')
    ticker = yf.Ticker('NQ=F')
    df = ticker.history(period=period, interval='1d')
    df.columns = df.columns.str.lower()
    print(f'    Fetched {len(df)} bars')
    
    # Add indicators
    print('[2] Calculating indicators...')
    
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
    
    # VWAP, ADX, Structure
    df = calculate_vwap(df, reset_daily=False)
    df = calculate_adx(df)
    df = detect_bos_choch(df)
    
    # Detect order blocks and FVGs
    print('[3] Detecting patterns...')
    order_blocks = detect_order_blocks(df)
    fvgs = detect_fvg(df)
    
    # Initialize ML engine
    print('[4] Initializing ML engine...')
    ml_engine = MLConfluenceEngine(
        use_xgboost=XGBOOST_AVAILABLE,
        use_hmm=HMM_AVAILABLE,
        min_factors=3,
        min_score=2.0
    )
    
    # Train on first 60% of data
    train_size = int(len(df) * 0.6)
    train_df = df.iloc[:train_size]
    
    print(f'[5] Training on {train_size} bars...')
    ml_engine.train(train_df)
    
    # Update MTF
    print('[6] Updating multi-timeframe analysis...')
    ml_engine.update_mtf('NQ=F')
    
    # Generate signals on test data
    print('[7] Generating ML-enhanced signals...')
    brain = QuantBrain(hurst_window=50)
    
    signals = []
    for i in range(train_size, len(df)):
        row = df.iloc[i]
        
        brain_state = brain.update(row['close'], df.index[i])
        active_obs = get_active_order_blocks(order_blocks, i, max_age=30)
        
        if i > 0:
            prev = df.iloc[i-1]
            pivots = calculate_pivots(prev['high'], prev['low'], prev['close'])
        else:
            pivots = None
        
        signal = ml_engine.evaluate(
            df.iloc[:i+1], i, brain_state, active_obs, pivots, fvgs,
            capital=100000,
            rsi=row['rsi'] if pd.notna(row['rsi']) else None,
            atr=row['atr'] if pd.notna(row['atr']) else None
        )
        
        signals.append({
            'signal': signal.direction,
            'stop_loss': signal.stop_loss,
            'tp1': signal.take_profit_1,
            'atr': row['atr'] if pd.notna(row['atr']) else 0,
            'confidence': signal.confidence,
            'ml_signal': signal.ml_signal,
            'regime': signal.regime,
            'mtf_passed': signal.mtf_filter_passed
        })
    
    signals_df = pd.DataFrame(signals, index=df.index[train_size:])
    
    # Signal distribution
    signal_counts = signals_df['signal'].value_counts()
    print('\n    Signal distribution:')
    for sig, count in signal_counts.items():
        pct = count / len(signals_df) * 100
        print(f'      {sig}: {count} ({pct:.1f}%)')
    
    # Regime distribution
    regime_counts = signals_df['regime'].value_counts()
    print('\n    Regime distribution:')
    for reg, count in regime_counts.items():
        pct = count / len(signals_df) * 100
        print(f'      {reg}: {count} ({pct:.1f}%)')
    
    # Run backtest
    print('\n[8] Running backtest...')
    test_df = df.iloc[train_size:]
    backtest = NQBacktestEngine(initial_capital=100000, use_kelly=True)
    result = backtest.run(test_df, signals_df)
    
    print(result)
    
    return result, signals_df


# === TEST ===
if __name__ == "__main__":
    result, signals = run_ml_backtest('2y')
