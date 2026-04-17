"""
Alpha Engine — Universal Data Loader
=====================================
Sources:
  - Gold / Silver : HistData.com M1 CSV  (header 3 lines skip)
  - Gold fallback : yfinance GC=F
  - Silver        : yfinance SI=F
  - NQ            : yfinance NQ=F / MNQ=F
  - Spectral      : optional meta_bias + regime injection
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from typing import Optional, List

_ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_ENGINE_DIR)

HISTDATA_COLS = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']

INSTRUMENTS = {
    'XAUUSD': {'yf': 'GC=F',  'pv': 100.0, 'tf': '1h'},
    'XAGUSD': {'yf': 'SI=F',  'pv': 50.0,  'tf': '1h'},
    'NQ':     {'yf': 'NQ=F',  'pv': 20.0,  'tf': '5m'},
    'MNQ':    {'yf': 'MNQ=F', 'pv': 2.0,   'tf': '5m'},
}


# ─────────────────────────────────────────────────────────────────────
# LOW-LEVEL LOADERS
# ─────────────────────────────────────────────────────────────────────

def load_histdata_m1(filepath: str) -> pd.DataFrame:
    """
    Load HistData.com M1 CSV.
    Format: 3-header lines, then DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOL
    Date format: YYYY.MM.DD  Time: HH:MM
    """
    df = pd.read_csv(
        filepath,
        skiprows=3,
        header=None,
        names=HISTDATA_COLS,
        dtype={'date': str, 'time': str},
    )
    df.dropna(subset=['date', 'open'], inplace=True)

    # Build datetime index
    dt_str = df['date'].str.strip() + ' ' + df['time'].str.strip()
    df.index = pd.to_datetime(dt_str, format='%Y.%m.%d %H:%M')
    df.index.name = 'datetime'
    df.drop(columns=['date', 'time'], inplace=True)
    df = df.astype(float)
    df.sort_index(inplace=True)
    return df


def load_gold_years(root_dir: str = _ROOT, years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load multiple XAUUSD M1 HistData CSVs and concatenate.
    Looks for DAT_MT_XAUUSD_M1_{YEAR}.csv or .txt in root_dir.
    """
    if years is None:
        years = list(range(2019, 2026))

    frames = []
    for yr in years:
        candidates = [
            os.path.join(root_dir, f'DAT_MT_XAUUSD_M1_{yr}.csv'),
            os.path.join(root_dir, f'DAT_MT_XAUUSD_M1_{yr}.txt'),
        ]
        loaded = False
        for path in candidates:
            if os.path.exists(path):
                try:
                    df_yr = load_histdata_m1(path)
                    frames.append(df_yr)
                    print(f"  Gold {yr}: {len(df_yr):,} M1 bars  [{path}]")
                    loaded = True
                    break
                except Exception as e:
                    print(f"  [WARN]  {yr} load failed: {e}")
        if not loaded:
            print(f"  [WARN]  No Gold file found for {yr}")

    if not frames:
        raise FileNotFoundError(f"No Gold M1 files found in {root_dir}")

    combined = pd.concat(frames).sort_index()
    combined = combined[~combined.index.duplicated(keep='last')]
    return combined


def load_silver_histdata(root_dir: str = _ROOT, years: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Load XAGUSD HistData M1 CSVs.
    Searches: root_dir/, root_dir/XAG_DATA/, root_dir/data/silver/
    Handles yearly files (2019-2025) and monthly files (202601, 202602 ...).
    """
    if years is None:
        years = list(range(2019, 2027))

    # Search directories in priority order
    search_dirs = [
        root_dir,
        os.path.join(root_dir, 'XAG_DATA'),
        os.path.join(root_dir, 'data', 'silver'),
    ]

    frames = []
    for yr in years:
        loaded = False
        for sdir in search_dirs:
            candidates = [
                os.path.join(sdir, f'DAT_MT_XAGUSD_M1_{yr}.csv'),
                os.path.join(sdir, f'DAT_MT_XAGUSD_M1_{yr}.txt'),
            ]
            for path in candidates:
                if os.path.exists(path):
                    try:
                        df_yr = load_histdata_m1(path)
                        frames.append(df_yr)
                        print(f"  Silver {yr}: {len(df_yr):,} M1 bars  [{path}]")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"  [WARN]  Silver {yr} load failed: {e}")
            if loaded:
                break

    # Also scan for monthly files (e.g. DAT_MT_XAGUSD_M1_202601.csv)
    for sdir in search_dirs:
        if not os.path.isdir(sdir):
            continue
        for fname in sorted(os.listdir(sdir)):
            if fname.startswith('DAT_MT_XAGUSD_M1_') and len(fname) > 22 and fname.endswith('.csv'):
                # monthly file like DAT_MT_XAGUSD_M1_202601.csv
                path = os.path.join(sdir, fname)
                try:
                    df_m = load_histdata_m1(path)
                    frames.append(df_m)
                    print(f"  Silver {fname}: {len(df_m):,} M1 bars")
                except Exception:
                    pass

    if frames:
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep='last')]
        return out
    return pd.DataFrame()


def load_yfinance(symbol: str, period: str = 'max',
                  interval: str = '1h',
                  start: Optional[str] = None,
                  end: Optional[str] = None) -> pd.DataFrame:
    """Download data from yfinance, strip timezone."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    kwargs: dict = {'progress': False}
    if start:
        kwargs['start'] = start
        kwargs['end']   = end or '2026-01-01'
    else:
        kwargs['period'] = period

    df = yf.download(symbol, interval=interval, **kwargs)
    if df.empty:
        return df

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    if df.index.tz is not None:
        # Convert to US Eastern so session-hour filters work correctly on NQ/equity data.
        # Gold/Silver are 24h markets so the conversion is harmless for them.
        df.index = df.index.tz_convert('America/New_York').tz_localize(None)

    if 'volume' not in df.columns:
        df['volume'] = 0
    df.sort_index(inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# RESAMPLE
# ─────────────────────────────────────────────────────────────────────

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample M1/M5 data to a coarser timeframe."""
    out = df.resample(rule).agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).dropna(subset=['open', 'close'])
    return out


# ─────────────────────────────────────────────────────────────────────
# SPECTRAL INJECTION
# ─────────────────────────────────────────────────────────────────────

def add_spectral(df: pd.DataFrame,
                 fft_window: int = 120,
                 regime_lookback: int = 1500,
                 add_hmm: bool = False) -> pd.DataFrame:
    """
    Inject SpectralBias (meta_bias) and optionally HMM Regime columns.

    add_hmm=False (default): only meta_bias from FFT is added.
        Backtests and DEAP optimization should use add_hmm=False.
        HMM regime hurt OOS performance in testing and is handled
        separately by the MiroFish intelligence layer at portfolio level.

    add_hmm=True: also compute HMM regime — use only for MiroFish pipeline
        or live portfolio decision-making where regime is consumed externally.
    """
    spec_dir = os.path.join(_ROOT, 'spectral_bias_engine')
    sys.path.insert(0, spec_dir)
    try:
        from fft_bias import add_spectral_features
        df = add_spectral_features(df, window_size=fft_window)
        # fft_bias adds 'spectral_bias' — rename to 'meta_bias' for strategy compatibility
        if 'spectral_bias' in df.columns and 'meta_bias' not in df.columns:
            df = df.rename(columns={'spectral_bias': 'meta_bias'})
        print("  SpectralBias: OK")
    except Exception as e:
        print(f"  SpectralBias skipped: {e}")
        if 'meta_bias' not in df.columns:
            df = df.copy()
            df['meta_bias'] = 0.0

    if add_hmm:
        try:
            from hmm_regime import add_regime_features
            df = add_regime_features(df, lookback=regime_lookback)
            print("  HMM Regime:   OK")
        except Exception as e:
            print(f"  HMM Regime skipped: {e}")
            if 'regime' not in df.columns:
                df = df.copy()
                df['regime'] = 0
    else:
        if 'regime' not in df.columns:
            df = df.copy()
            df['regime'] = 0

    return df


# ─────────────────────────────────────────────────────────────────────
# HIGH-LEVEL LOADERS
# ─────────────────────────────────────────────────────────────────────

def get_gold_1h(years: Optional[List[int]] = None,
                root_dir: str = _ROOT,
                add_meta: bool = True) -> pd.DataFrame:
    """
    Load Gold M1, resample to 1H, optionally inject spectral features.
    Returns 1H OHLCV + meta_bias + regime.
    """
    print("Loading Gold (XAUUSD)...")
    try:
        m1 = load_gold_years(root_dir, years)
    except FileNotFoundError:
        print("  Falling back to yfinance GC=F...")
        m1 = load_yfinance('GC=F', period='max', interval='1h')
        if m1.empty:
            raise

    if _infer_tf(m1) not in ('1h',):
        print("  Resampling M1 -> 1H...")
        df = resample_ohlcv(m1, '1h')
    else:
        df = m1.copy()

    print(f"  1H bars: {len(df):,}  [{df.index[0].date()} -> {df.index[-1].date()}]")
    if add_meta:
        df = add_spectral(df, add_hmm=False)  # HMM handled by MiroFish separately
    return df


def get_gold_m1(years: Optional[List[int]] = None,
                root_dir: str = _ROOT,
                add_meta: bool = True) -> pd.DataFrame:
    """
    Load Gold raw M1 data for scalping backtests.

    SpectralBias strategy: compute FFT meta_bias on 1H bars (fast, 23K bars),
    then forward-fill back to M1 index. This gives the spectral trend context
    without running FFT on 1.4M M1 bars (which would take minutes).
    """
    print("Loading Gold M1 (XAUUSD)...")
    m1 = load_gold_years(root_dir, years)
    print(f"  M1 bars: {len(m1):,}  [{m1.index[0].date()} -> {m1.index[-1].date()}]")

    if add_meta:
        # Step 1: resample to 1H for fast spectral computation
        h1 = resample_ohlcv(m1, '1h')
        h1 = add_spectral(h1, fft_window=120, add_hmm=False)
        print(f"  SpectralBias on 1H ({len(h1):,} bars) -> forward-filled to M1")

        # Step 2: forward-fill meta_bias and regime from 1H back to M1 index
        m1 = m1.copy()
        for col, default in [('meta_bias', 0.0), ('regime', 0)]:
            if col in h1.columns:
                filled = h1[col].reindex(m1.index, method='ffill')
                m1[col] = filled.fillna(default).values
            else:
                m1[col] = default
    else:
        m1['meta_bias'] = 0.0
        m1['regime']    = 0

    return m1


def get_silver_1h(years: Optional[List[int]] = None,
                  root_dir: str = _ROOT,
                  add_meta: bool = True) -> pd.DataFrame:
    """
    Load Silver, resample to 1H.
    Tries HistData CSVs first, falls back to yfinance SI=F.
    """
    print("Loading Silver (XAGUSD)...")
    m1 = load_silver_histdata(root_dir, years)
    if m1.empty:
        print("  No HistData files. Falling back to yfinance SI=F...")
        m1 = load_yfinance('SI=F', period='max', interval='1h')
        if m1.empty:
            raise FileNotFoundError("No Silver data available")
        df = m1
    else:
        df = resample_ohlcv(m1, '1h')

    print(f"  1H bars: {len(df):,}  [{df.index[0].date()} -> {df.index[-1].date()}]")
    if add_meta:
        df = add_spectral(df, add_hmm=False)  # HMM handled by MiroFish separately
    return df


def get_nq_5m(lookback_days: int = 60,
              add_meta: bool = True,
              cache_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load NQ Futures 5m data.
    Uses cache if available, else yfinance NQ=F.
    """
    print("Loading NQ Futures (NQ=F 5m)...")
    df = pd.DataFrame()

    # Try cache
    if cache_path and os.path.exists(cache_path):
        try:
            import pickle
            with open(cache_path, 'rb') as f:
                obj = pickle.load(f)
            df = obj if isinstance(obj, pd.DataFrame) else obj.get('df', pd.DataFrame())
            df.columns = df.columns.str.lower()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            print(f"  Cache: {len(df):,} bars [{cache_path}]")
        except Exception as e:
            print(f"  [WARN]  Cache load failed: {e}")
            df = pd.DataFrame()

    # Auto-detect root cache
    if df.empty:
        for fname in ['cache_qqq_5m.pkl', 'cache_nq_5m.pkl']:
            path = os.path.join(_ROOT, fname)
            if os.path.exists(path):
                try:
                    import pickle
                    with open(path, 'rb') as f:
                        obj = pickle.load(f)
                    df = obj if isinstance(obj, pd.DataFrame) else obj.get('df', pd.DataFrame())
                    df.columns = df.columns.str.lower()
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    print(f"  Cache: {len(df):,} bars [{fname}]")
                    break
                except Exception:
                    pass

    # yfinance fallback — try NQ futures symbols in order
    if df.empty:
        period = f"{min(lookback_days, 60)}d"
        for sym in ['NQ=F', 'NQ1', 'MNQ=F', 'ES=F', 'QQQ']:
            df = load_yfinance(sym, period=period, interval='5m')
            if not df.empty:
                print(f"  yfinance: loaded {sym} ({len(df):,} bars)")
                break
        if df.empty:
            raise FileNotFoundError("No NQ data available (tried NQ=F, NQ1, MNQ=F, ES=F, QQQ)")

    if 'volume' not in df.columns:
        df['volume'] = 0

    print(f"  5m bars: {len(df):,}  [{df.index[0].date()} -> {df.index[-1].date()}]")
    if add_meta:
        df = add_spectral(df, fft_window=60, add_hmm=False)  # HMM handled by MiroFish separately
    return df


# ─────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────

def _infer_tf(df: pd.DataFrame) -> str:
    if len(df) < 3:
        return 'unknown'
    secs = (df.index[1] - df.index[0]).total_seconds()
    for s, lbl in [(60, '1m'), (300, '5m'), (900, '15m'),
                   (3600, '1h'), (14400, '4h'), (86400, '1d')]:
        if abs(secs - s) < s * 0.5:
            return lbl
    return 'unknown'


def date_slice(df: pd.DataFrame,
               start: Optional[str] = None,
               end: Optional[str] = None) -> pd.DataFrame:
    """Slice a dataframe between start and end date strings."""
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index < pd.Timestamp(end)]
    return df
