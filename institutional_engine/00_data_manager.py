"""
=============================================================
DATA MANAGER — Cache Sistemi
=============================================================
M1 veriyi bir kere yükler, farklı timeframe'lere resample eder
ve CSV olarak kaydeder. Bir sonraki çalıştırmada direkt okur.
=============================================================
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


CACHE_DIR    = "cache"
TIMEFRAMES   = ["15min", "1h", "4h", "1D"]


def _load_single_m1(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath, header=None,
        names=['date','time','open','high','low','close','volume'],
        encoding='utf-8-sig',
    )
    df['date'] = df['date'].astype(str).str.strip()
    df['time'] = df['time'].astype(str).str.strip()
    df['timestamp'] = pd.to_datetime(
        df['date'] + ' ' + df['time'],
        format='%Y.%m.%d %H:%M', errors='coerce'
    )
    df = df.set_index('timestamp').drop(columns=['date','time'])
    df = df[~df.index.isna()]
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').replace(0, 1.0).fillna(1.0)
    for c in ['open','high','low','close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['open','high','low','close'])
    df = df[(df['close'] > 0) & (df['close'] < 10_000)]
    df = df[df['high'] >= df['low']]
    return df


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    r = df.resample(tf).agg(
        open=('open','first'), high=('high','max'),
        low=('low','min'),     close=('close','last'),
        volume=('volume','sum'),
    ).dropna(subset=['open','close'])
    return r[r['open'] > 0]


def _cache_path(tf: str) -> str:
    return os.path.join(CACHE_DIR, f"XAUUSD_{tf}.csv")


def _is_cached(tf: str) -> bool:
    return os.path.exists(_cache_path(tf))


def _save_cache(df: pd.DataFrame, tf: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(tf)
    df.to_csv(path)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"   💾 {tf:5s} → {path}  ({size_mb:.1f} MB, {len(df):,} bar)")


def _load_cache(tf: str) -> pd.DataFrame:
    df = pd.read_csv(_cache_path(tf), index_col=0, parse_dates=True)
    return df


def build_cache(
    data_folder: str,
    start_year:  int = 2019,
    end_year:    int = 2025,
    force:       bool = False,    # True = cache'i yenile
) -> dict:
    """
    Tüm timeframe'leri hazırlar.
    Cache varsa direkt okur, yoksa M1'den üretir.

    Returns: {'15min': df, '1h': df, '4h': df, '1D': df}
    """

    # Tüm cache'ler hazır mı?
    all_cached = all(_is_cached(tf) for tf in TIMEFRAMES)

    if all_cached and not force:
        print("=" * 55)
        print("⚡ Cache bulundu — direkt yükleniyor...")
        print("=" * 55)
        result = {}
        for tf in TIMEFRAMES:
            df = _load_cache(tf)
            result[tf] = df
            print(f"   ✅ {tf:5s}: {len(df):>8,} bar  "
                  f"{df.index[0].strftime('%Y-%m-%d')} → "
                  f"{df.index[-1].strftime('%Y-%m-%d')}")
        print("=" * 55 + "\n")
        return result

    # Cache yok veya force=True → M1'den yükle
    print("=" * 55)
    print("📂 M1 Veri Yükleniyor (ilk seferlik)...")
    print("=" * 55)

    all_frames = []
    for year in range(start_year, end_year + 1):
        for ext in ['.csv', '.dat', '.txt']:
            fp = os.path.join(data_folder, f"DAT_MT_XAUUSD_M1_{year}{ext}")
            if os.path.exists(fp):
                try:
                    dy = _load_single_m1(fp)
                    all_frames.append(dy)
                    print(f"   ✅ {year}: {len(dy):>8,} bar")
                    break
                except Exception as e:
                    print(f"   ❌ {year}: {e}")
                    break
        else:
            print(f"   ⚠️  {year}: Dosya yok")

    if not all_frames:
        raise FileNotFoundError(f"Hiç M1 dosyası bulunamadı: {data_folder}")

    df_m1 = pd.concat(all_frames).sort_index()
    df_m1 = df_m1[~df_m1.index.duplicated(keep='first')]

    print(f"\n   📊 Toplam M1: {len(df_m1):,} bar")
    print(f"   💰 Fiyat    : {df_m1['close'].min():.2f}$ → {df_m1['close'].max():.2f}$")
    print(f"\n💾 Cache oluşturuluyor...")

    result = {}
    for tf in TIMEFRAMES:
        df_tf = _resample(df_m1, tf)
        _save_cache(df_tf, tf)
        result[tf] = df_tf

    print(f"\n✅ Cache hazır! Bir daha M1 yüklenmeyecek.\n")
    return result


def load_tf(tf: str) -> pd.DataFrame:
    """Tek bir timeframe yükler (cache'den)"""
    if not _is_cached(tf):
        raise FileNotFoundError(
            f"Cache yok: {_cache_path(tf)}\n"
            f"Önce build_cache() çalıştır."
        )
    return _load_cache(tf)


def cache_info():
    """Cache durumunu gösterir"""
    print("\n📁 CACHE DURUMU")
    print("-" * 45)
    if not os.path.exists(CACHE_DIR):
        print("   ❌ Cache klasörü yok")
        return
    for tf in TIMEFRAMES:
        path = _cache_path(tf)
        if os.path.exists(path):
            df   = _load_cache(path) if False else None
            size = os.path.getsize(path) / 1024 / 1024
            mod  = datetime.fromtimestamp(os.path.getmtime(path))
            print(f"   ✅ {tf:5s}: {size:.1f} MB  (güncelleme: {mod.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"   ❌ {tf:5s}: Yok")
    print()
