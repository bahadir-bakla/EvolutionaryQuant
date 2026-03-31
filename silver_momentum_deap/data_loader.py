"""
HistData.com Silver Data Loader — EvolutionaryQuant
====================================================
Supports HistData.com XAGUSD M1 zip/CSV format:

Format 1 (most common):
  YYYYMMDD,HHMMSS,Open,High,Low,Close,Volume
  20231016,093000,23.1234,23.1456,23.1123,23.1345,234

Format 2 (some symbols):
  YYYYMMDD HHMMSS,Open,High,Low,Close,Volume

Usage:
    from data_loader import load_histdata, resample_to_tf
    df_5m  = load_histdata('XAGUSD_2023.csv')
    df_5m  = resample_to_tf(df_5m, '5min')
    df_15m = resample_to_tf(df_5m, '15min')

    # Or load from zip:
    df_5m = load_histdata('XAGUSD_2023.zip')

    # Or load from folder (all zips/csvs in folder):
    df_5m = load_histdata_folder('C:/data/XAGUSD/')
"""

import os
import zipfile
import io
import glob
import numpy as np
import pandas as pd
from typing import Union, List, Optional


# ─────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────

def _parse_histdata_bytes(content: bytes) -> pd.DataFrame:
    """Parse raw HistData CSV bytes → OHLCV DataFrame."""
    # Try both comma and semicolon separators
    for sep in [',', ';']:
        try:
            df = pd.read_csv(
                io.StringIO(content.decode('utf-8', errors='ignore')),
                sep=sep, header=None,
            )
            if df.shape[1] >= 6:
                return _format_histdata_df(df)
        except Exception:
            continue
    raise ValueError("Cannot parse HistData file — check format")


def _format_histdata_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw parsed df to OHLCV with DatetimeIndex."""
    # Column 0 may be 'YYYYMMDD' or 'YYYYMMDD HHMMSS'
    col0 = df.iloc[:, 0].astype(str).str.strip()
    col1 = df.iloc[:, 1].astype(str).str.strip()

    if ' ' in col0.iloc[0]:
        # Format 2: date+time combined in col0
        ts_str = col0
        o_col, h_col, l_col, c_col = 1, 2, 3, 4
        v_col  = 5 if df.shape[1] > 5 else None
    else:
        # Format 1: date in col0, time in col1
        ts_str = col0 + ' ' + col1
        o_col, h_col, l_col, c_col = 2, 3, 4, 5
        v_col  = 6 if df.shape[1] > 6 else None

    # Parse timestamps
    timestamps = pd.to_datetime(ts_str, format='mixed', dayfirst=False, errors='coerce')
    mask = timestamps.notna()
    timestamps = timestamps[mask]

    ohlc = df.iloc[mask.values, [o_col, h_col, l_col, c_col]].copy()
    ohlc.columns = ['open','high','low','close']
    ohlc = ohlc.apply(pd.to_numeric, errors='coerce')

    vol = pd.Series(1000, index=timestamps)
    if v_col is not None and df.shape[1] > v_col:
        try:
            vol = pd.to_numeric(df.iloc[mask.values, v_col], errors='coerce').fillna(1000)
            vol.index = timestamps
        except Exception:
            pass

    result = pd.DataFrame(
        {'open': ohlc['open'].values,
         'high': ohlc['high'].values,
         'low':  ohlc['low'].values,
         'close':ohlc['close'].values,
         'volume': vol.values},
        index=timestamps
    )
    result = result.sort_index()
    result = result[result[['open','high','low','close']].notna().all(axis=1)]
    result = result[result['close'] > 0]
    return result


def load_histdata(path: str) -> pd.DataFrame:
    """
    Load HistData.com file (CSV or ZIP) for any symbol.
    Supports single file or zip containing one CSV.
    """
    path = os.path.expanduser(path)

    if path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as zf:
            csv_files = [f for f in zf.namelist()
                         if f.lower().endswith('.csv') or f.lower().endswith('.txt')]
            if not csv_files:
                raise FileNotFoundError(f"No CSV/TXT in zip: {path}")
            # Use the largest file if multiple
            csv_files.sort(key=lambda f: zf.getinfo(f).file_size, reverse=True)
            content = zf.read(csv_files[0])
        return _parse_histdata_bytes(content)

    elif path.lower().endswith(('.csv', '.txt')):
        with open(path, 'rb') as f:
            content = f.read()
        return _parse_histdata_bytes(content)

    else:
        raise ValueError(f"Unsupported file type: {path}")


def load_histdata_folder(folder: str,
                         pattern: str = '*.zip',
                         max_files: int = 50) -> pd.DataFrame:
    """
    Load all HistData files from a folder (multiple months/years).
    Concatenates and deduplicates.
    """
    folder = os.path.expanduser(folder)
    files  = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        # Try CSV too
        files = sorted(glob.glob(os.path.join(folder, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No files found in {folder} with pattern {pattern}")

    files = files[:max_files]
    dfs   = []
    for f in files:
        try:
            df = load_histdata(f)
            dfs.append(df)
            print(f"  ✅ {os.path.basename(f)}: {len(df):,} bars")
        except Exception as e:
            print(f"  ⚠️  {os.path.basename(f)}: {e}")

    if not dfs:
        raise ValueError("No files loaded successfully")

    combined = pd.concat(dfs).sort_index()
    # Remove duplicates
    combined = combined[~combined.index.duplicated(keep='first')]
    print(f"\n✅ Total: {len(combined):,} M1 bars "
          f"[{combined.index[0].date()} → {combined.index[-1].date()}]")
    return combined


# ─────────────────────────────────────────────────────────────────────
# RESAMPLING
# ─────────────────────────────────────────────────────────────────────

def resample_to_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resample M1 data to target timeframe.
    tf: '5min', '15min', '1h', '4h', '1D' etc.
    Aliases: '5m' → '5min', '15m' → '15min', '1h' → '1h'
    """
    # Alias handling
    aliases = {'5m': '5min', '15m': '15min', '30m': '30min',
               '1h': '1h', '4h': '4h', '1d': '1D', '1D': '1D'}
    tf = aliases.get(tf, tf)

    ohlcv = df.resample(tf).agg(
        open  =('open',  'first'),
        high  =('high',  'max'),
        low   =('low',   'min'),
        close =('close', 'last'),
        volume=('volume','sum'),
    ).dropna(subset=['open', 'close'])

    # Filter market hours for intraday (Sunday gaps etc.)
    ohlcv = ohlcv[ohlcv['close'] > 0]
    return ohlcv


# ─────────────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────────────

def load_or_cache(source: str,
                  cache_path: str,
                  tf: str = '5min',
                  force_reload: bool = False) -> pd.DataFrame:
    """
    Load from cache pkl if exists, otherwise load from source and cache.
    source: path to CSV/ZIP/folder
    cache_path: path to save/load .pkl cache
    """
    import pickle

    if not force_reload and os.path.exists(cache_path):
        print(f"📦 Cache yükleniyor: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    print(f"📥 HistData yükleniyor: {source}")
    if os.path.isdir(source):
        df_m1 = load_histdata_folder(source)
    else:
        df_m1 = load_histdata(source)

    if tf != '1min' and tf != '1m':
        df    = resample_to_tf(df_m1, tf)
    else:
        df    = df_m1

    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(df, f)
    print(f"✅ Cache kaydedildi: {cache_path}")
    return df


# ─────────────────────────────────────────────────────────────────────
# YFINANCE FALLBACK (if no HistData)
# ─────────────────────────────────────────────────────────────────────

def load_yfinance_silver(tf: str = '5m', lookback_days: int = 59) -> pd.DataFrame:
    """
    Fallback: fetch XAG=F (Silver futures) from Yahoo Finance.
    Limited to 60 days for intraday data.
    """
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        end   = datetime.now()
        start = end - timedelta(days=lookback_days)
        df = yf.download('SI=F', start=start, end=end, interval=tf,
                         progress=False, auto_adjust=True)
        if df.empty:
            print("⚠️  XAG=F boş, SLV ETF deneniyor...")
            df = yf.download('SLV', start=start, end=end, interval=tf,
                             progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = df.columns.str.lower()
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        if 'volume' not in df.columns:
            df['volume'] = 1000
        df = df[df['close'] > 0].dropna(subset=['open','high','low','close'])
        print(f"✅ YFinance SLV/SI=F: {len(df):,} bars "
              f"[{df.index[0].date()} → {df.index[-1].date()}]")
        return df
    except Exception as e:
        print(f"❌ YFinance fallback başarısız: {e}")
        return pd.DataFrame()
