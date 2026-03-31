# ES-NQ Correlation Scalper Logic

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class CorrelationSignal:
    timestamp: pd.Timestamp
    direction: str # 'LONG', 'SHORT', 'NEUTRAL'
    es_price: float
    nq_price: float
    spread_zscore: float
    confidence: float
    stop_loss: float
    take_profit: float

class CorrelationScanner:
    """
    Scans for ES leading NQ.
    Logic:
    1. Calculate rolling correlation (must be high, e.g., > 0.8).
    2. Calculate Z-Score of the Spread (ES_norm - NQ_norm).
    3. Signal if Spread diverge significantly while correlation remains high.
    """
    
    def __init__(self, window: int = 20, z_threshold: float = 2.0):
        self.window = window
        self.z_threshold = z_threshold
        
    def align_data(self, es_df: pd.DataFrame, nq_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge ES and NQ on timestamp inner join.
        """
        # Ensure datetimes
        es_df.index = pd.to_datetime(es_df.index)
        nq_df.index = pd.to_datetime(nq_df.index)
        
        # Rename cols
        es = es_df[['close']].rename(columns={'close': 'es_close'})
        nq = nq_df[['close']].rename(columns={'close': 'nq_close'})
        
        # Merge
        merged = pd.merge(es, nq, left_index=True, right_index=True, how='inner')
        return merged
        
    def calculate_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Z-Score and Correlation stats.
        """
        # Returns
        df['es_ret'] = df['es_close'].pct_change()
        df['nq_ret'] = df['nq_close'].pct_change()
        
        # Normalized Accumulation (Cumulative Sum of returns roughly)
        # Better: Rolling Z-Score of price ratio or difference in returns?
        # Let's use spread of normalized prices
        
        # Rolling Correlation
        df['corr'] = df['es_ret'].rolling(self.window).corr(df['nq_ret'])
        
        # Spread: We want to know if NQ is "cheap" relative to ES
        # Beta calculation: NQ = beta * ES + alpha
        # Residual = NQ_actual - (beta * ES)
        # If Residual is very negative -> NQ is cheap (BUY)
        # If Residual is very positive -> NQ is expensive (SELL)
        
        # Simple approach: Spread of standardized returns
        # Z-Score of the difference between NQ and ES moves over window
        
        # Let's map cumulative returns
        df['es_cum'] = (1 + df['es_ret']).cumprod()
        df['nq_cum'] = (1 + df['nq_ret']).cumprod()
        
        # Ratio
        df['ratio'] = df['nq_close'] / df['es_close']
        
        # Rolling Z-Score of Ratio
        r_mean = df['ratio'].rolling(self.window).mean()
        r_std = df['ratio'].rolling(self.window).std()
        df['z_score'] = (df['ratio'] - r_mean) / r_std
        
        return df

    def get_signal(self, row: pd.Series) -> str:
        """
        Trade NQ based on Z-Score.
        If Z-Score < -2: NQ is Undervalued relative to ES -> LONG NQ
        If Z-Score > +2: NQ is Overvalued relative to ES -> SHORT NQ
        Verification: Correlation must be > 0.8
        """
        corr = row.get('corr', 0)
        z = row.get('z_score', 0)
        
        if corr < 0.8:
            return 'NEUTRAL' # Decoupled, unsafe
            
        if z < -self.z_threshold:
            return 'LONG' # NQ Lagging Down (Cheap)
        elif z > self.z_threshold:
            return 'SHORT' # NQ Lagging Up (Expensive)
            
        return 'NEUTRAL'
