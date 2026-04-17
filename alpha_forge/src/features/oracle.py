"""
Alpha-Forge Oracle Layer
Aggregates AI signals: TimesFM forecast, MiroFish sentiment, Gemma regime
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class OracleLayer:
    """
    AI Feature Oracle — combines all AI model outputs into
    a single feature DataFrame for strategy evolution.
    """

    def __init__(self):
        pass

    @staticmethod
    def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators without external dependencies."""
        df = df.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]

        # RSI
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["bb_upper"] = sma20 + 2 * std20
        df["bb_lower"] = sma20 - 2 * std20
        df["bb_mid"] = sma20
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-10)

        # ADX
        plus_dm = h.diff()
        minus_dm = -l.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr1 = h - l
        tr2 = (h - c.shift()).abs()
        tr3 = (l - c.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / 14).mean() / (atr14 + 1e-10)
        minus_di = 100 * minus_dm.ewm(alpha=1 / 14).mean() / (atr14 + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df["adx"] = dx.ewm(alpha=1 / 14).mean()

        # ATR
        df["atr"] = atr14

        # SMAs
        df["sma_20"] = c.rolling(20).mean()
        df["sma_50"] = c.rolling(50).mean()
        df["sma_200"] = c.rolling(200).mean()
        df["sma_diff"] = (df["sma_20"] - df["sma_50"]) / (df["sma_50"] + 1e-10) * 100
        df["price_vs_200sma"] = (c - df["sma_200"]) / (df["sma_200"] + 1e-10) * 100

        # Volume
        df["volume_ratio"] = v / (v.rolling(20).mean() + 1e-10)

        # Returns
        df["returns"] = c.pct_change()
        df["log_returns"] = np.log(c / c.shift(1))

        return df

    @staticmethod
    def compute_timesfm_features(
        df: pd.DataFrame,
        context_length: int = 64,
        horizon: int = 10,
    ) -> pd.DataFrame:
        """
        Generate TimesFM-like features using statistical forecasting.
        In production, replace with actual TimesFM API calls.
        """
        features = pd.DataFrame(index=df.index)
        c = df["close"].values

        for i in range(context_length, len(c)):
            context = c[i - context_length : i]

            # Simple trend forecast (linear regression)
            x = np.arange(len(context))
            slope, intercept = np.polyfit(x, context, 1)
            forecast = slope * np.arange(horizon) + intercept

            current = context[-1]
            forecast_end = forecast[-1]
            change_pct = (forecast_end - current) / current * 100

            # Momentum
            mid = len(forecast) // 2
            first_half = np.mean(forecast[:mid])
            second_half = np.mean(forecast[mid:])
            momentum = (second_half - first_half) / (first_half + 1e-10) * 100

            # Curvature
            curvature = np.polyfit(x, context, 2)[0] if len(context) >= 3 else 0

            # Signal
            if change_pct > 0.1:
                signal = min(1.0, change_pct / 2.0)
            elif change_pct < -0.1:
                signal = max(-1.0, change_pct / 2.0)
            else:
                signal = 0.0

            ts = df.index[i]
            features.loc[ts, "timesfm_trend_slope"] = slope
            features.loc[ts, "timesfm_price_change_pct"] = change_pct
            features.loc[ts, "timesfm_forecast_volatility"] = (
                np.std(forecast) / (np.mean(forecast) + 1e-10) * 100
            )
            features.loc[ts, "timesfm_momentum"] = momentum
            features.loc[ts, "timesfm_curvature"] = curvature
            features.loc[ts, "timesfm_confidence"] = 1.0 / (1.0 + np.var(forecast))
            features.loc[ts, "timesfm_forecast_high"] = np.max(forecast)
            features.loc[ts, "timesfm_forecast_low"] = np.min(forecast)
            features.loc[ts, "timesfm_forecast_range"] = (
                np.max(forecast) - np.min(forecast)
            )
            features.loc[ts, "timesfm_signal"] = signal

        return features

    @staticmethod
    def compute_mirofish_features(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Simulate MiroFish multi-agent sentiment.
        In production, replace with actual MiroFish server calls.
        """
        features = pd.DataFrame(index=df.index)

        returns = df["close"].pct_change().fillna(0)
        vol_ratio = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)

        sentiment = np.zeros(len(df))
        stress = np.zeros(len(df))
        fear_greed = np.full(len(df), 50.0)

        for i in range(1, len(df)):
            # Market signal impact
            market_signal = returns.iloc[i] * 50
            volume_impact = (vol_ratio.iloc[i] - 1) * 10

            # Herd behavior
            prev_sentiment = sentiment[i - 1]
            herd = prev_sentiment * 0.7

            # Update sentiment
            sentiment[i] = np.clip(
                prev_sentiment * 0.8 + market_signal * 0.1 + herd * 0.1,
                -1.0,
                1.0,
            )

            # Fear/Greed
            fear_greed[i] = 50 + sentiment[i] * 50

            # Stress (increases with volatility)
            stress[i] = min(1.0, abs(returns.iloc[i]) * 100 + stress[i - 1] * 0.9)

        features["mirofish_sentiment"] = sentiment
        features["mirofish_fear_greed"] = fear_greed
        features["mirofish_avg_stress"] = stress
        features["mirofish_net_positions"] = sentiment * 100

        return features

    @staticmethod
    def compute_gemma_regime(df: pd.DataFrame) -> pd.DataFrame:
        """
        Rule-based regime detection (Gemma 4 proxy).
        In production, replace with actual Gemma 4 API calls.
        """
        features = pd.DataFrame(index=df.index)

        regimes = []
        confidences = []
        trend_strengths = []

        for i in range(len(df)):
            adx = df.iloc[i].get("adx", 25)
            bb_width = df.iloc[i].get("bb_width", 0.02)
            rsi = df.iloc[i].get("rsi", 50)

            if adx > 30:
                regime = "TREND_UP" if rsi > 50 else "TREND_DOWN"
                confidence = min(0.9, adx / 50)
            elif adx < 20:
                regime = "RANGE"
                confidence = 0.7
            elif bb_width > 0.05:
                regime = "VOLATILE"
                confidence = 0.6
            else:
                regime = "TRANSITION"
                confidence = 0.5

            regimes.append(regime)
            confidences.append(confidence)
            trend_strengths.append(min(1.0, adx / 50))

        regime_map = {
            "TREND_UP": 1.0,
            "TREND_DOWN": -1.0,
            "RANGE": 0.0,
            "VOLATILE": 0.5,
            "TRANSITION": 0.25,
        }

        features["gemma_regime"] = regimes
        features["gemma_confidence"] = confidences
        features["gemma_trend_strength"] = trend_strengths
        features["gemma_regime_numeric"] = [regime_map[r] for r in regimes]

        return features

    @staticmethod
    def compute_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Gold Reversion Scalper icin gerekli ozellikler:
        - red_streak: art arda kapanan kirmizi mum sayisi
        - rsi (14): zaten hesaplaniyor, burada tekrar hesaplanmaz
        - reversion_setup: streak >= 4 ve rsi < 35 ise True
        - drop_range: son streak icindeki dusus mesafesi (ATR ile normalize)
        """
        df = df.copy()

        # Red streak
        is_red = (df["close"] < df["open"]).astype(int)
        streak = 0
        streaks = []
        for v in is_red:
            streak = streak + 1 if v else 0
            streaks.append(streak)
        df["red_streak"] = streaks

        # Reversion setup flag (hizli filtre — DEAP streak threshold > 3)
        rsi_col = df["rsi"] if "rsi" in df.columns else pd.Series(50, index=df.index)
        df["reversion_setup"] = (
            (df["red_streak"] >= 4) & (rsi_col < 38)
        ).astype(float)

        # Drop range (son 6 bar icindeki high-low mesafesi, ATR ile normalize)
        atr_col = df["atr"] if "atr" in df.columns else pd.Series(1.0, index=df.index)
        df["drop_range_norm"] = (
            df["high"].rolling(6).max() - df["low"].rolling(6).min()
        ) / (atr_col + 1e-10)

        return df

    def build_oracle_features(
        self,
        df: pd.DataFrame,
        instrument: str = "NQ",
    ) -> pd.DataFrame:
        """Build complete oracle feature DataFrame."""
        logger.info(f"Building oracle for {instrument}: {len(df)} bars")

        # Technical indicators
        oracle_df = self.compute_technical_indicators(df)

        # Reversion features (Gold streak + setup flag)
        logger.info("Computing reversion features...")
        oracle_df = self.compute_reversion_features(oracle_df)

        # TimesFM features
        logger.info("Computing TimesFM features...")
        tsfm = self.compute_timesfm_features(oracle_df)
        oracle_df = oracle_df.join(tsfm, how="left")

        # MiroFish sentiment
        logger.info("Computing MiroFish sentiment...")
        mf = self.compute_mirofish_features(oracle_df)
        oracle_df = oracle_df.join(mf, how="left")

        # Gemma regime
        logger.info("Computing Gemma regime...")
        gr = self.compute_gemma_regime(oracle_df)
        oracle_df = oracle_df.join(gr, how="left")

        # Forward-fill regime features
        regime_cols = [c for c in oracle_df.columns if c.startswith("gemma_")]
        oracle_df[regime_cols] = oracle_df[regime_cols].ffill()

        logger.info(
            f"Oracle built: {len(oracle_df)} rows, {len(oracle_df.columns)} columns"
        )
        return oracle_df

    @staticmethod
    def get_oracle_signal(oracle_df: pd.DataFrame, idx: int) -> Dict:
        """Extract oracle signal at a specific index."""
        row = oracle_df.iloc[idx]
        return {
            "timesfm_signal": row.get("timesfm_signal", 0.0),
            "mirofish_sentiment": row.get("mirofish_sentiment", 0.0),
            "gemma_regime": row.get("gemma_regime_numeric", 0.0),
            "gemma_confidence": row.get("gemma_confidence", 0.5),
            "rsi": row.get("rsi", 50),
            "adx": row.get("adx", 25),
            "macd_hist": row.get("macd_hist", 0),
            "bb_width": row.get("bb_width", 0.02),
            "volume_ratio": row.get("volume_ratio", 1.0),
            "atr": row.get("atr", 15),
        }
