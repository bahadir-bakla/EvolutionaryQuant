# NQ Quant Bot - XGBoost Signal Classification
# Machine Learning for signal prediction

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Run: pip install xgboost")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib


@dataclass
class MLPrediction:
    """ML model prediction result"""
    signal: str  # 'LONG', 'SHORT', 'NEUTRAL'
    probability: float  # Confidence 0-1
    long_prob: float
    short_prob: float
    neutral_prob: float
    features_used: int


@dataclass
class MLModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1: float
    feature_importance: Dict[str, float]


class SignalClassifier:
    """
    XGBoost-based signal classifier
    
    Features:
    - Technical indicators (RSI, MACD, BB, etc.)
    - Price action features (returns, volatility)
    - Market structure features
    - Volume features
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        min_child_weight: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8
    ):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='multi:softprob',
            num_class=3,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ML features from OHLCV data
        
        Returns DataFrame with all features
        """
        features = pd.DataFrame(index=df.index)
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # === PRICE FEATURES ===
        # Returns
        features['return_1d'] = close.pct_change(1)
        features['return_3d'] = close.pct_change(3)
        features['return_5d'] = close.pct_change(5)
        features['return_10d'] = close.pct_change(10)
        
        # Volatility
        features['volatility_10d'] = close.pct_change().rolling(10).std()
        features['volatility_20d'] = close.pct_change().rolling(20).std()
        
        # Range
        features['daily_range'] = (high - low) / close
        features['range_ma_ratio'] = features['daily_range'] / features['daily_range'].rolling(20).mean()
        
        # === TREND FEATURES ===
        # EMAs
        ema_8 = close.ewm(span=8).mean()
        ema_21 = close.ewm(span=21).mean()
        ema_50 = close.ewm(span=50).mean()
        
        features['ema_8_dist'] = (close - ema_8) / close
        features['ema_21_dist'] = (close - ema_21) / close
        features['ema_50_dist'] = (close - ema_50) / close
        features['ema_8_21_cross'] = (ema_8 - ema_21) / close
        features['ema_21_50_cross'] = (ema_21 - ema_50) / close
        
        # === MOMENTUM FEATURES ===
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_ma'] = features['rsi'].rolling(10).mean()
        features['rsi_std'] = features['rsi'].rolling(10).std()
        
        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal_line = macd.ewm(span=9).mean()
        features['macd'] = macd / close
        features['macd_signal'] = signal_line / close
        features['macd_hist'] = (macd - signal_line) / close
        
        # Rate of Change
        features['roc_5'] = (close - close.shift(5)) / close.shift(5)
        features['roc_10'] = (close - close.shift(10)) / close.shift(10)
        
        # === VOLATILITY FEATURES ===
        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        features['atr_ratio'] = atr / close
        features['atr_change'] = atr.pct_change(5)
        
        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        features['bb_position'] = (close - bb_mid) / (bb_std * 2 + 1e-10)
        features['bb_width'] = (bb_std * 4) / bb_mid
        
        # === VOLUME FEATURES ===
        features['volume_ma_ratio'] = volume / volume.rolling(20).mean()
        features['volume_change'] = volume.pct_change(1)
        features['volume_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()
        
        # === PATTERN FEATURES ===
        # Candle patterns
        body = close - df['open']
        features['body_ratio'] = body / (high - low + 1e-10)
        features['upper_wick'] = (high - pd.concat([close, df['open']], axis=1).max(axis=1)) / (high - low + 1e-10)
        features['lower_wick'] = (pd.concat([close, df['open']], axis=1).min(axis=1) - low) / (high - low + 1e-10)
        
        # Higher highs / Lower lows
        features['hh'] = (high > high.rolling(5).max().shift(1)).astype(int)
        features['ll'] = (low < low.rolling(5).min().shift(1)).astype(int)
        
        # === MARKET STRUCTURE ===
        # Distance from recent high/low
        features['dist_from_high_20'] = (high.rolling(20).max() - close) / close
        features['dist_from_low_20'] = (close - low.rolling(20).min()) / close
        
        # Store feature names
        self.feature_names = list(features.columns)
        
        return features
    
    def create_labels(
        self, 
        df: pd.DataFrame, 
        forward_periods: int = 5,
        threshold: float = 0.005  # 0.5% move threshold
    ) -> pd.Series:
        """
        Create labels based on future returns
        
        0 = SHORT (price drops > threshold)
        1 = NEUTRAL (price moves < threshold)
        2 = LONG (price rises > threshold)
        """
        future_return = df['close'].shift(-forward_periods) / df['close'] - 1
        
        labels = pd.Series(index=df.index, dtype=int)
        labels[future_return > threshold] = 2   # LONG
        labels[future_return < -threshold] = 0  # SHORT
        labels[(future_return >= -threshold) & (future_return <= threshold)] = 1  # NEUTRAL
        
        return labels
    
    def train(
        self, 
        df: pd.DataFrame, 
        forward_periods: int = 5,
        threshold: float = 0.005,
        validation_split: float = 0.2
    ) -> MLModelMetrics:
        """
        Train the classifier
        """
        # Create features and labels
        features = self.create_features(df)
        labels = self.create_labels(df, forward_periods, threshold)
        
        # Align and drop NaN
        data = pd.concat([features, labels.rename('label')], axis=1).dropna()
        
        X = data[self.feature_names]
        y = data['label']
        
        # Time series split (no lookahead bias)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_val_scaled)
        
        # Feature importance
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return MLModelMetrics(
            accuracy=accuracy_score(y_val, y_pred),
            precision=precision_score(y_val, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y_val, y_pred, average='weighted', zero_division=0),
            f1=f1_score(y_val, y_pred, average='weighted', zero_division=0),
            feature_importance=importance
        )
    
    def predict(self, df: pd.DataFrame) -> MLPrediction:
        """
        Make prediction for the latest bar
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self.create_features(df)
        
        # Get last row
        X = features.iloc[-1:].copy()
        
        # Ensure we have all features
        missing = set(self.feature_names) - set(X.columns)
        for col in missing:
            X[col] = 0
        
        X = X[self.feature_names]
        
        # Handle NaN and Inf values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Clip extreme values
        X = X.clip(-1e10, 1e10)
        
        try:
            X_scaled = self.scaler.transform(X)
            
            # Predict probabilities
            probs = self.model.predict_proba(X_scaled)[0]
            pred_class = self.model.predict(X_scaled)[0]
        except Exception as e:
            # Fallback to neutral on error
            return MLPrediction(
                signal='NEUTRAL',
                probability=0.34,
                long_prob=0.33,
                short_prob=0.33,
                neutral_prob=0.34,
                features_used=len(self.feature_names)
            )
        
        # Map to signal
        signal_map = {0: 'SHORT', 1: 'NEUTRAL', 2: 'LONG'}
        signal = signal_map[pred_class]
        
        return MLPrediction(
            signal=signal,
            probability=probs[pred_class],
            long_prob=probs[2],
            short_prob=probs[0],
            neutral_prob=probs[1],
            features_used=len(self.feature_names)
        )
    
    def walk_forward_train(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        forward_periods: int = 5,
        threshold: float = 0.005
    ) -> List[MLModelMetrics]:
        """
        Walk-forward training to prevent overfitting
        """
        features = self.create_features(df)
        labels = self.create_labels(df, forward_periods, threshold)
        
        data = pd.concat([features, labels.rename('label')], axis=1).dropna()
        X = data[self.feature_names]
        y = data['label']
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        results = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            self.model.fit(X_train_scaled, y_train, verbose=False)
            y_pred = self.model.predict(X_val_scaled)
            
            results.append(MLModelMetrics(
                accuracy=accuracy_score(y_val, y_pred),
                precision=precision_score(y_val, y_pred, average='weighted', zero_division=0),
                recall=recall_score(y_val, y_pred, average='weighted', zero_division=0),
                f1=f1_score(y_val, y_pred, average='weighted', zero_division=0),
                feature_importance={}
            ))
        
        self.is_trained = True
        return results
    
    def save(self, path: str):
        """Save model and scaler"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }, path)
    
    def load(self, path: str):
        """Load model and scaler"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.is_trained = True


# === TEST ===
if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing XGBoost Signal Classifier...")
    
    # Fetch data
    ticker = yf.Ticker("NQ=F")
    df = ticker.history(period="2y", interval="1d")
    df.columns = df.columns.str.lower()
    print(f"Fetched {len(df)} bars")
    
    # Train classifier
    classifier = SignalClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1
    )
    
    print("\nTraining with walk-forward validation...")
    results = classifier.walk_forward_train(df, n_splits=5)
    
    print("\nWalk-Forward Results:")
    for i, r in enumerate(results):
        print(f"  Split {i+1}: Accuracy={r.accuracy:.1%}, F1={r.f1:.2f}")
    
    # Average performance
    avg_acc = np.mean([r.accuracy for r in results])
    avg_f1 = np.mean([r.f1 for r in results])
    print(f"\nAverage: Accuracy={avg_acc:.1%}, F1={avg_f1:.2f}")
    
    # Full training for feature importance
    print("\nFull training for feature importance...")
    metrics = classifier.train(df)
    
    print("\nTop 10 Features:")
    for i, (feat, imp) in enumerate(list(metrics.feature_importance.items())[:10]):
        print(f"  {i+1}. {feat}: {imp:.4f}")
    
    # Make prediction
    pred = classifier.predict(df)
    print(f"\nCurrent Prediction:")
    print(f"  Signal: {pred.signal}")
    print(f"  Confidence: {pred.probability:.1%}")
    print(f"  Long Prob: {pred.long_prob:.1%}")
    print(f"  Short Prob: {pred.short_prob:.1%}")
