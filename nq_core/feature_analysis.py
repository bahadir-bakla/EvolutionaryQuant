# NQ Quant Bot - Feature Importance Analysis
# Find which indicators actually work

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_feature_importance(symbol: str = "NQ=F", period: str = "1y"):
    """Analyze which features are actually predictive"""
    
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Fetch data
    print(f"\nFetching {symbol} daily data...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1d")
    df.columns = df.columns.str.lower()
    print(f"Bars: {len(df)}")
    
    # Create all features
    print("\nCreating features...")
    features = pd.DataFrame(index=df.index)
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # === Price Features ===
    features['return_1d'] = close.pct_change(1)
    features['return_3d'] = close.pct_change(3)
    features['return_5d'] = close.pct_change(5)
    
    # === EMAs ===
    features['ema_8_dist'] = (close - close.ewm(span=8).mean()) / close
    features['ema_21_dist'] = (close - close.ewm(span=21).mean()) / close
    features['ema_50_dist'] = (close - close.ewm(span=50).mean()) / close
    
    # === RSI ===
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    features['rsi'] = 100 - (100 / (1 + rs))
    
    # === VWAP distance ===
    tp = (high + low + close) / 3
    vwap = (tp * volume).cumsum() / volume.cumsum()
    features['vwap_dist'] = (close - vwap) / close
    
    # === Momentum ===
    features['momentum_3'] = close.pct_change(3)
    features['momentum_5'] = close.pct_change(5)
    
    # === Volatility ===
    features['volatility'] = close.pct_change().rolling(10).std()
    
    # === ATR ===
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    features['atr_ratio'] = tr.rolling(14).mean() / close
    
    # === Volume ===
    features['volume_ratio'] = volume / volume.rolling(20).mean()
    
    # === Structure ===
    features['hh'] = (high > high.rolling(5).max().shift(1)).astype(int)
    features['ll'] = (low < low.rolling(5).min().shift(1)).astype(int)
    
    # === Bollinger Bands ===
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    features['bb_position'] = (close - bb_mid) / (bb_std * 2 + 1e-10)
    
    # Create labels (future return)
    forward_return = close.shift(-5) / close - 1
    labels = (forward_return > 0.005).astype(int)  # 1 if up > 0.5%
    
    # Prepare data
    data = pd.concat([features, labels.rename('label')], axis=1).dropna()
    X = data.drop('label', axis=1)
    y = data['label']
    
    print(f"Samples: {len(X)}")
    
    # Method 1: Correlation with future returns
    print("\n" + "-"*60)
    print("METHOD 1: Correlation with 5-day forward return")
    print("-"*60)
    
    correlations = {}
    for col in X.columns:
        corr = X[col].corr(forward_return.loc[X.index])
        correlations[col] = corr
    
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n{'Feature':<25} {'Correlation':>12} {'Useful?':>10}")
    print("-"*50)
    useful_features = []
    for feat, corr in sorted_corr:
        useful = "YES" if abs(corr) > 0.05 else "NO"
        if abs(corr) > 0.05:
            useful_features.append(feat)
        print(f"{feat:<25} {corr:>12.4f} {useful:>10}")
    
    # Method 2: XGBoost importance
    print("\n" + "-"*60)
    print("METHOD 2: XGBoost Feature Importance")
    print("-"*60)
    
    try:
        from xgboost import XGBClassifier
        
        model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        
        # Handle NaN
        X_clean = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        model.fit(X_clean, y)
        
        importance = dict(zip(X.columns, model.feature_importances_))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n{'Feature':<25} {'Importance':>12} {'Useful?':>10}")
        print("-"*50)
        xgb_useful = []
        for feat, imp in sorted_imp:
            useful = "YES" if imp > 0.05 else "NO"
            if imp > 0.05:
                xgb_useful.append(feat)
            print(f"{feat:<25} {imp:>12.4f} {useful:>10}")
        
    except ImportError:
        print("XGBoost not installed")
        xgb_useful = []
    
    # Summary
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    # Combine both methods
    all_useful = set(useful_features) | set(xgb_useful)
    useless = set(X.columns) - all_useful
    
    print("\n[+] KEEP these indicators:")
    for f in sorted(all_useful):
        print(f"   - {f}")
    
    print("\n[-] REMOVE these indicators (not predictive):")
    for f in sorted(useless):
        print(f"   - {f}")
    
    # RSI specific check
    rsi_corr = correlations.get('rsi', 0)
    rsi_imp = importance.get('rsi', 0) if 'importance' in dir() else 0
    print(f"\n[i] RSI Analysis:")
    print(f"   Correlation: {rsi_corr:.4f}")
    print(f"   XGBoost Importance: {rsi_imp:.4f}")
    if abs(rsi_corr) < 0.03 and rsi_imp < 0.05:
        print("   [!] RSI is NOT useful for this strategy!")
    else:
        print("   RSI has some value")
    
    return {
        'useful': list(all_useful),
        'useless': list(useless),
        'correlations': dict(sorted_corr),
        'xgb_importance': dict(sorted_imp) if 'sorted_imp' in dir() else {}
    }


if __name__ == "__main__":
    result = analyze_feature_importance("NQ=F", "1y")
