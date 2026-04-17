"""
Alpha-Forge Data Pipeline
MT5 integration, data cleaning, windowing for TimesFM
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from scipy import stats
import os
import logging

logger = logging.getLogger(__name__)


class MT5DataFetcher:
    """MetaTrader 5 data fetching with robust error handling."""

    INSTRUMENTS = {
        "NQ": {"symbol": "NAS100", "point": 0.25, "lot_size": 1.0},
        "ES": {"symbol": "US500", "point": 0.25, "lot_size": 1.0},
        "XAUUSD": {"symbol": "XAUUSD", "point": 0.01, "lot_size": 100.0},
        "XAGUSD": {"symbol": "XAGUSD", "point": 0.001, "lot_size": 5000.0},
    }

    TIMEFRAMES = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }

    def __init__(self):
        self.connected = False

    def connect(self) -> bool:
        if not mt5.initialize():
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False
        info = mt5.terminal_info()
        logger.info(f"MT5 Connected: {info.name} (login: {info.login})")
        self.connected = True
        return True

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5 Disconnected")

    def fetch_rates(
        self,
        instrument: str,
        timeframe: str = "5m",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        count: int = 10000,
    ) -> Optional[pd.DataFrame]:
        if not self.connected and not self.connect():
            return None

        symbol = self.INSTRUMENTS[instrument]["symbol"]
        tf = self.TIMEFRAMES[timeframe]

        if start_date and end_date:
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        else:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            logger.warning(f"No data for {symbol} {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        logger.info(f"Fetched {len(df)} bars for {instrument} {timeframe}")
        return df[["open", "high", "low", "close", "volume"]]


class DataCleaner:
    """Production-grade data cleaning pipeline."""

    @staticmethod
    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df[~df.index.duplicated(keep="first")]
        removed = before - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        return df

    @staticmethod
    def fill_gaps(df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
        full_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
        df = df.reindex(full_index)
        df["open"] = df["open"].ffill()
        df["high"] = df["high"].ffill()
        df["low"] = df["low"].ffill()
        df["close"] = df["close"].ffill()
        df["volume"] = df["volume"].fillna(0)
        return df.dropna()

    @staticmethod
    def clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        df["high"] = df[["high", "low"]].max(axis=1)
        df["low"] = df[["high", "low"]].min(axis=1)
        df["high"] = df[["high", "open", "close"]].max(axis=1)
        df["low"] = df[["low", "open", "close"]].min(axis=1)
        return df

    @staticmethod
    def detect_outliers_zscore(
        df: pd.DataFrame, column: str = "close", threshold: float = 3.0
    ) -> pd.Series:
        z = np.abs(stats.zscore(df[column].dropna()))
        outliers = pd.Series(False, index=df.index)
        outliers[df[column].dropna().index[z > threshold]] = True
        return outliers

    @staticmethod
    def detect_price_spikes(
        df: pd.DataFrame, max_change_pct: float = 5.0
    ) -> pd.Series:
        return df["close"].pct_change().abs() * 100 > max_change_pct

    @staticmethod
    def clean_pipeline(
        df: pd.DataFrame,
        freq: str = "5min",
        outlier_threshold: float = 3.0,
    ) -> pd.DataFrame:
        logger.info(f"Cleaning: {len(df)} bars")
        df = DataCleaner.remove_duplicates(df)
        df = DataCleaner.clean_ohlc(df)

        outliers = DataCleaner.detect_outliers_zscore(df, "close", outlier_threshold)
        spikes = DataCleaner.detect_price_spikes(df)
        all_outliers = outliers | spikes

        if all_outliers.any():
            logger.info(f"Detected {all_outliers.sum()} outliers")
            df.loc[all_outliers, "close"] = np.nan
            df["close"] = df["close"].interpolate(method="linear")
            for col in ["open", "high", "low"]:
                df.loc[all_outliers, col] = np.nan
                df[col] = df[col].interpolate(method="linear")

        df = DataCleaner.fill_gaps(df, freq)
        logger.info(f"Cleaned: {len(df)} bars")
        return df


class TimesFMWindowCreator:
    """Creates normalized time-series windows for TimesFM."""

    def __init__(
        self,
        context_length: int = 512,
        forecast_horizon: int = 96,
        stride: int = 1,
    ):
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride

    def create_windows(
        self,
        df: pd.DataFrame,
        target_column: str = "close",
    ) -> Tuple[np.ndarray, np.ndarray]:
        series = df[target_column].values
        series = self._normalize_rolling(series)

        X, y = [], []
        total = self.context_length + self.forecast_horizon

        for i in range(0, len(series) - total + 1, self.stride):
            window = series[i : i + total]
            X.append(window[: self.context_length])
            y.append(window[self.context_length :])

        return np.array(X), np.array(y)

    @staticmethod
    def _normalize_rolling(series: np.ndarray) -> np.ndarray:
        window = min(100, len(series) // 4)
        rm = pd.Series(series).rolling(window, min_periods=1).mean().values
        rs = pd.Series(series).rolling(window, min_periods=1).std().values
        rs = np.where(rs == 0, 1, rs)
        return (series - rm) / rs

    def create_multi_feature_windows(
        self,
        df: pd.DataFrame,
        feature_columns: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if feature_columns is None:
            feature_columns = ["close", "volume"]

        df = df.copy()
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(20).std()
        df["volume_ratio"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
        df = df.fillna(0)

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        features = scaler.fit_transform(df[feature_columns].values)

        X, y = [], []
        total = self.context_length + self.forecast_horizon

        for i in range(0, len(features) - total + 1, self.stride):
            window = features[i : i + total]
            X.append(window[: self.context_length])
            y.append(window[self.context_length :, 0])

        return np.array(X), np.array(y)


class AlphaForgeDataPipeline:
    """Main data pipeline orchestrator."""

    def __init__(self, config: dict = None):
        self.config = config or {
            "instruments": ["NQ", "ES", "XAUUSD", "XAGUSD"],
            "timeframes": ["5m", "1h"],
            "context_length": 512,
            "forecast_horizon": 96,
        }
        self.fetcher = MT5DataFetcher()
        self.cleaner = DataCleaner()
        self.window_creator = TimesFMWindowCreator(
            context_length=self.config["context_length"],
            forecast_horizon=self.config["forecast_horizon"],
        )

    def process_instrument(
        self,
        instrument: str,
        timeframe: str = "5m",
        use_cached: bool = True,
    ) -> Optional[dict]:
        cache_path = f"data/processed/{instrument}_{timeframe}_processed.parquet"

        if use_cached and os.path.exists(cache_path):
            logger.info(f"Loading cached: {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            logger.info(f"Fetching {instrument} {timeframe}...")
            df = self.fetcher.fetch_rates(instrument, timeframe)
            if df is None:
                return None

            logger.info(f"Cleaning {instrument} {timeframe}...")
            df = self.cleaner.clean_pipeline(df)

            os.makedirs("data/processed", exist_ok=True)
            df.to_parquet(cache_path)

        logger.info(f"Creating windows for {instrument} {timeframe}...")
        X, y = self.window_creator.create_windows(df)
        np.save(f"data/processed/{instrument}_{timeframe}_X.npy", X)
        np.save(f"data/processed/{instrument}_{timeframe}_y.npy", y)

        return {
            "instrument": instrument,
            "timeframe": timeframe,
            "bars": len(df),
            "windows": len(X),
            "date_range": {"start": str(df.index[0]), "end": str(df.index[-1])},
            "stats": {
                "mean_close": float(df["close"].mean()),
                "std_close": float(df["close"].std()),
            },
        }

    def run_full_pipeline(self) -> dict:
        self.fetcher.connect()
        results = {}

        for instrument in self.config["instruments"]:
            for tf in self.config["timeframes"]:
                result = self.process_instrument(instrument, tf)
                if result:
                    results[f"{instrument}_{tf}"] = result
                    logger.info(
                        f"OK {instrument}_{tf}: {result['bars']} bars, {result['windows']} windows"
                    )

        self.fetcher.disconnect()

        os.makedirs("data/processed", exist_ok=True)
        import json

        with open("data/processed/metadata.json", "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "config": self.config,
                    "results": results,
                },
                f,
                indent=2,
            )

        logger.info(f"Pipeline complete: {len(results)} datasets")
        return results
