import pandas as pd
import numpy as np
from scipy.signal import hilbert

from nq_core.nq_hilbert_mama import NQ_Hilbert_MAMA_Strategy
from nq_core.nq_hilbert_sine import NQ_Hilbert_Sine_Strategy
from nq_core.nq_extreme_breakdown import NQ_Extreme_Breakdown_Strategy

class NQAlgorithmicSuite:
    """
    The Master Controller for the Nasdaq Quantitative Edge.
    Calculates complex mathematical phases (Hilbert, MAMA, ATRs) purely on 5-minute data.
    Routes data to the mathematically optimal strategy entirely based on the Macro HMM Regime.
    """
    def __init__(self):
        self.strat_trend = NQ_Hilbert_MAMA_Strategy()
        self.strat_chop = NQ_Hilbert_Sine_Strategy()
        self.strat_extreme = NQ_Extreme_Breakdown_Strategy()

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates all required technical structures in one pass.
        WARNING: This dataset MUST be 5-minute data. Phase vectors are extremely
        sensitive to timeframe changes and will fail out of distribution if fed 1M data.
        """
        if len(df) < 60:
            return df
            
        df = df.copy()
        
        # 0. Base ATR
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        
        # 1. Hilbert Transform & MAMA Vectors
        sma_len = 20
        df['sma'] = df['close'].rolling(sma_len).mean()
        df['detrended'] = df['close'] - df['sma']
        df['detrended'].fillna(0, inplace=True)
        
        analytic_signal = hilbert(df['detrended'].values)
        df['hilbert_phase'] = np.angle(analytic_signal)
        
        df['phase_delta'] = df['hilbert_phase'].diff()
        # Correct bound mapping wrapping pi/-pi
        df['phase_delta'] = df['phase_delta'].apply(lambda x: x + 2*np.pi if x < -np.pi else (x - 2*np.pi if x > np.pi else x))
        df['phase_delta'] = df['phase_delta'].abs()

        df['sine'] = np.sin(df['hilbert_phase'])
        df['leadsine'] = np.sin(df['hilbert_phase'] + np.pi/4) 
        df['sine_prev'] = df['sine'].shift(1)
        df['leadsine_prev'] = df['leadsine'].shift(1)
        
        df['cross_up'] = (df['sine'] > df['leadsine']) & (df['sine_prev'] <= df['leadsine_prev'])
        df['cross_down'] = (df['sine'] < df['leadsine']) & (df['sine_prev'] >= df['leadsine_prev'])
        
        # Recursive MAMA Array calculation (fastest way outside of cython)
        fast_limit = 0.5
        slow_limit = 0.05
        mamas = [0.0]*len(df)
        famas = [0.0]*len(df)
        closes = df['close'].values
        deltas = df['phase_delta'].fillna(0).values
        
        for i in range(len(df)):
            if i == 0:
                mamas[i] = closes[i]
                famas[i] = closes[i]
                continue
            phase_diff = deltas[i]
            if phase_diff > 0:
                alpha = fast_limit / (phase_diff + 0.0001)
            else:
                alpha = fast_limit
            alpha = max(slow_limit, min(fast_limit, alpha))
            mamas[i] = alpha * closes[i] + (1 - alpha) * mamas[i-1]
            famas[i] = (alpha / 2) * mamas[i] + (1 - (alpha / 2)) * famas[i-1]
            
        df['mama'] = mamas
        df['fama'] = famas

        # 2. Extreme Macro Breakdown Supports
        df['macro_low'] = df['low'].shift(1).rolling(50).min()
        
        return df

    def evaluate(self, df: pd.DataFrame, idx: int, current_regime: str, daily_bias: str = 'UNKNOWN'):
        """ Routes index and data to the active bot based on structural regime. """
        
        active_bot = "None"
        class Sig: pass
        signal = Sig()
        signal.direction = 'NEUTRAL'
        signal.stop_loss = 0.0
        signal.take_profit_2 = 0.0
        
        # Safety catch
        if len(df) < 50 or idx < 50:
            return signal, active_bot
            
        if 'TREND' in current_regime:
            active_bot = "Hilbert MAMA (Trend)"
            signal = self.strat_trend.evaluate(df, idx)
            
        elif 'CHOP' in current_regime:
            active_bot = "Hilbert SineWave (Chop)"
            signal = self.strat_chop.evaluate(df, idx, daily_bias)
            
        elif 'EXTREME' in current_regime:
            active_bot = "Macro Breakdown (Crash)"
            signal = self.strat_extreme.evaluate(df, idx)
            
        return signal, active_bot
