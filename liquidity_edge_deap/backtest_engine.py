"""
============================================================
LiquidityEdge XAU/USD — Backtest Engine
============================================================
Safi Liquidity (Likiditasyon) odaklı strateji:

Konsept:
  Kurumsal oyuncular (Smart Money) fiyatı likit bölgelere
  sweep eder, stop hunt yapar, sonra gerçek yönde hareket
  eder. Bu sistem tam o anı yakalar.

Feature Stack (DEAP tarafından optimize edilir):
  1. Order Block (OB)   — Kurumsal ilgi bölgeleri
  2. Liquidity Sweep    — Stop hunt + geri dönüş (LTF + HTF)
  3. Fair Value Gap (FVG) — İmbalans bölgeleri
  4. EMA Stack          — Trend filtresi (21/50/200)
  5. RSI Divergence     — Momentum zayıflığı
  6. ATR Volatility     — Pozisyon boyutu + SL/TP
  7. Session Filter     — London/New York oturumu
  8. VWAP               — Kurumsal fiyat referansı
  9. Displacement       — Güçlü fiyat hareketi teyidi
 10. XGBoost Filter     — ML sinyal kalite filtresi (opsiyonel)

Parametreler (DEAP genome):
   [0] ob_lookback        4-20    Order block arama penceresi
   [1] sweep_margin       0.1-2.0 Sweep eşiği (ATR katsayısı)
   [2] fvg_min_size       0.3-1.5 Min FVG boyutu (ATR katsayısı)
   [3] ema_trend_period   8-50    Ana EMA trendi
   [4] rsi_period         7-21    RSI periyodu
   [5] rsi_ob_level       65-80   RSI aşırı alım eşiği
   [6] rsi_os_level       20-35   RSI aşırı satım eşiği
   [7] atr_period         7-21    ATR periyodu
   [8] tp_atr_mult        2.0-8.0 TP = ATR × bu
   [9] sl_atr_mult        0.5-2.0 SL = ATR × bu
   [10] lot_size          0.01-0.15 Başlangıç lot
   [11] growth_factor     0.10-0.50 Compounding
   [12] session_filter    0-1     Oturum filtresi (0=off, 1=on)
   [13] displacement_mult 1.5-4.0 Displacement eşiği (ATR katsayısı)
   [14] ob_body_pct       0.5-1.0 OB geçerlilik (body yüzdesi)
============================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────
# GENE BOUNDS
# ──────────────────────────────────────────────────────────
GENE_BOUNDS = [
    (4,    20),    # [0]  ob_lookback
    (0.1,  2.0),   # [1]  sweep_margin (ATR mult)
    (0.3,  1.5),   # [2]  fvg_min_size (ATR mult)
    (8,    50),    # [3]  ema_trend_period
    (7,    21),    # [4]  rsi_period
    (65.0, 80.0),  # [5]  rsi_ob_level
    (20.0, 35.0),  # [6]  rsi_os_level
    (7,    21),    # [7]  atr_period
    (2.0,  8.0),   # [8]  tp_atr_mult
    (0.5,  2.0),   # [9]  sl_atr_mult
    (0.01, 0.15),  # [10] lot_size
    (0.10, 0.50),  # [11] growth_factor
    (0.0,  1.0),   # [12] session_filter
    (1.5,  4.0),   # [13] displacement_mult
    (0.5,  1.0),   # [14] ob_body_pct
    (0.0,  0.8),   # [15] meta_bias_threshold
]
GENOME_SIZE = len(GENE_BOUNDS)
GENOME_NAMES = [
    'ob_lookback', 'sweep_margin', 'fvg_min_size', 'ema_trend_period',
    'rsi_period', 'rsi_ob_level', 'rsi_os_level', 'atr_period',
    'tp_atr_mult', 'sl_atr_mult', 'lot_size', 'growth_factor',
    'session_filter', 'displacement_mult', 'ob_body_pct', 'meta_bias_threshold'
]

# Varsayılan "makul başlangıç" parametreler (kıyaslama için)
DEFAULT_PARAMS = {
    'ob_lookback': 10, 'sweep_margin': 0.5, 'fvg_min_size': 0.5,
    'ema_trend_period': 21, 'rsi_period': 14, 'rsi_ob_level': 70.0,
    'rsi_os_level': 30.0, 'atr_period': 14, 'tp_atr_mult': 4.0,
    'sl_atr_mult': 1.0, 'lot_size': 0.05, 'growth_factor': 0.25,
    'session_filter': 1.0, 'displacement_mult': 2.5, 'ob_body_pct': 0.7,
    'meta_bias_threshold': 0.0
}

POINT_VALUE = 100.0   # XAU/USD: 1 lot = 100 oz = $100/puan


def decode(genome: list) -> dict:
    """Genome [0,1] vektörünü gerçek parametre değerlerine dönüştür."""
    p = {}
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        raw  = float(genome[i]) if i < len(genome) else 0.5
        p[i] = lo + abs(raw % 1.0) * (hi - lo)
    return {
        'ob_lookback':       int(np.clip(p[0], 4, 20)),
        'sweep_margin':      float(np.clip(p[1], 0.1, 2.0)),
        'fvg_min_size':      float(np.clip(p[2], 0.3, 1.5)),
        'ema_trend_period':  int(np.clip(p[3], 8, 50)),
        'rsi_period':        int(np.clip(p[4], 7, 21)),
        'rsi_ob_level':      float(np.clip(p[5], 65.0, 80.0)),
        'rsi_os_level':      float(np.clip(p[6], 20.0, 35.0)),
        'atr_period':        int(np.clip(p[7], 7, 21)),
        'tp_atr_mult':       float(np.clip(p[8], 2.0, 8.0)),
        'sl_atr_mult':       float(np.clip(p[9], 0.5, 2.0)),
        'lot_size':          float(np.clip(p[10], 0.01, 0.15)),
        'growth_factor':     float(np.clip(p[11], 0.10, 0.50)),
        'session_filter':    float(p[12]) > 0.5,
        'displacement_mult': float(np.clip(p[13], 1.5, 4.0)),
        'ob_body_pct':       float(np.clip(p[14], 0.5, 1.0)),
        'meta_bias_threshold':float(np.clip(p[15], 0.0, 0.8)),
    }


# ──────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-10))


def compute_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def add_liquidity_features(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """
    Strateji için tüm feature'ları hesapla.
    Returns yeni kolonlar eklenmiş DataFrame.
    """
    df = df.copy()
    n  = p['ob_lookback']

    # ── ATR ──────────────────────────────────────────────
    df['atr'] = compute_atr(df, p['atr_period'])
    atr = df['atr']

    # ── EMA Stack ────────────────────────────────────────
    df['ema_fast']  = compute_ema(df['close'], p['ema_trend_period'])
    df['ema_50']    = compute_ema(df['close'], 50)
    df['ema_200']   = compute_ema(df['close'], 200)
    df['trend_up']  = (df['ema_fast'] > df['ema_50']) & (df['ema_50'] > df['ema_200'])
    df['trend_dn']  = (df['ema_fast'] < df['ema_50']) & (df['ema_50'] < df['ema_200'])

    # ── RSI ──────────────────────────────────────────────
    df['rsi'] = compute_rsi(df['close'], p['rsi_period'])

    # ── VWAP (rolling, oturum proxy) ─────────────────────
    tp = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume'].replace(0, 1)
    df['vwap'] = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()
    df['above_vwap'] = df['close'] > df['vwap']
    df['below_vwap'] = df['close'] < df['vwap']

    # ── Order Block Tespiti ───────────────────────────────
    # Bullish OB: son n bar içindeki bearish mum → sonra güçlü yukarı hareket
    # Bearish OB: son n bar içindeki bullish mum → sonra güçlü aşağı hareket
    is_bearish = df['close'] < df['open']
    is_bullish = df['close'] > df['open']
    body_size  = (df['close'] - df['open']).abs()
    bar_range  = (df['high'] - df['low']).abs().replace(0, 1e-10)
    body_pct   = body_size / bar_range

    # Displacement: bir sonraki mumda güçlü hareket (impuls mum)
    disp_up   = (df['close'] - df['open']) > (atr * p['displacement_mult'])
    disp_dn   = (df['open']  - df['close']) > (atr * p['displacement_mult'])

    ob_bull = pd.Series(False, index=df.index)
    ob_bear = pd.Series(False, index=df.index)
    for lag in range(1, min(n, 10) + 1):
        # Bullish OB: (lag) bar önce bearish, body yeterli, ardından displacement up
        cond_bull = (
            is_bearish.shift(lag) &
            (body_pct.shift(lag) >= p['ob_body_pct']) &
            disp_up.shift(lag - 1)
        )
        ob_bull = ob_bull | cond_bull
        # Bearish OB: (lag) bar önce bullish, body yeterli, ardından displacement dn
        cond_bear = (
            is_bullish.shift(lag) &
            (body_pct.shift(lag) >= p['ob_body_pct']) &
            disp_dn.shift(lag - 1)
        )
        ob_bear = ob_bear | cond_bear
    df['ob_bull'] = ob_bull
    df['ob_bear'] = ob_bear

    # ── Liquidity Sweep ───────────────────────────────────
    # Fiyat son n barın swing high/low'unu geçip geri dönerse → sweep
    swing_high = df['high'].rolling(n).max().shift(1)
    swing_low  = df['low'].rolling(n).min().shift(1)
    sweep_th   = atr * p['sweep_margin']

    df['sweep_low_bull'] = (
        (df['low']  < swing_low - sweep_th) &     # sweep altına indi
        (df['close'] > swing_low)                  # kapanış tekrar üstünde
    )
    df['sweep_high_bear'] = (
        (df['high'] > swing_high + sweep_th) &    # sweep üstüne çıktı
        (df['close'] < swing_high)                  # kapanış tekrar altında
    )

    # ── Fair Value Gap (FVG) ─────────────────────────────
    # 3-mum imbalans: gap[i] low > high[i-2] (bullish) veya tersi
    fvg_min = atr * p['fvg_min_size']
    df['fvg_bull'] = (
        (df['low'] > df['high'].shift(2)) &
        ((df['low'] - df['high'].shift(2)) >= fvg_min)
    )
    df['fvg_bear'] = (
        (df['high'] < df['low'].shift(2)) &
        ((df['low'].shift(2) - df['high']) >= fvg_min)
    )

    # ── RSI Divergence (basit) ────────────────────────────
    # Fiyat higher high ama RSI lower high → bearish div
    # Fiyat lower low  ama RSI higher low → bullish div
    price_hh  = df['high'] > df['high'].shift(n // 2)
    rsi_lh    = df['rsi']  < df['rsi'].shift(n // 2)
    price_ll  = df['low']  < df['low'].shift(n // 2)
    rsi_hl    = df['rsi']  > df['rsi'].shift(n // 2)
    df['rsi_bear_div'] = price_hh & rsi_lh & (df['rsi'] > 55)
    df['rsi_bull_div'] = price_ll & rsi_hl & (df['rsi'] < 45)

    # ── London/New York Oturumu (UTC+3 varsayımı) ─────────
    # London: 10:00-19:00 yerel (07-16 UTC)
    # NY: 16:30-23:00 yerel (13:30-20 UTC)
    if hasattr(df.index, 'hour'):
        hour_utc3 = (df.index.hour + 3) % 24
        df['in_session'] = (
            ((hour_utc3 >= 10) & (hour_utc3 < 19)) |   # London
            ((hour_utc3 >= 16) & (hour_utc3 < 23))     # New York
        )
    else:
        df['in_session'] = True

    # ── Pivot Noktaları (Daily + Weekly) ─────────────────
    # Kurumsal oyuncuların en çok saygı duyduğu fiyat seviyeleri:
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L,  R2 = PP + (H - L)
    # S1 = 2*PP - H,  S2 = PP - (H - L)
    # Daily pivot: önceki günün OHLC → bugünün pivot
    # Weekly pivot: önceki haftanın OHLC → bu haftanın pivot
    try:
        if hasattr(df.index, 'date'):
            # Günlük gruplama
            daily = df.resample('1D').agg({'high':'max','low':'min','close':'last','open':'first'})
            daily = daily.dropna()
            daily['pp']   = (daily['high'] + daily['low'] + daily['close']) / 3
            daily['r1']   = 2 * daily['pp'] - daily['low']
            daily['r2']   = daily['pp'] + (daily['high'] - daily['low'])
            daily['s1']   = 2 * daily['pp'] - daily['high']
            daily['s2']   = daily['pp'] - (daily['high'] - daily['low'])

            # Bir önceki günün pivot değerleri bugüne çekilir (forward-fill)
            for col in ['pp','r1','r2','s1','s2']:
                daily[f'd_{col}'] = daily[col].shift(1)
            daily_piv = daily[[f'd_{c}' for c in ['pp','r1','r2','s1','s2']]]
            daily_piv.columns = ['d_pp','d_r1','d_r2','d_s1','d_s2']

            # Haftalık gruplama
            weekly = df.resample('1W').agg({'high':'max','low':'min','close':'last','open':'first'})
            weekly = weekly.dropna()
            weekly['pp_w'] = (weekly['high'] + weekly['low'] + weekly['close']) / 3
            weekly['r1_w'] = 2 * weekly['pp_w'] - weekly['low']
            weekly['s1_w'] = 2 * weekly['pp_w'] - weekly['high']
            for col in ['pp_w','r1_w','s1_w']:
                weekly[f'd_{col}'] = weekly[col].shift(1)
            weekly_piv = weekly[['d_pp_w','d_r1_w','d_s1_w']]

            # 1H frame'e yeniden index'le (forward-fill)
            df = df.join(daily_piv.resample('1h').ffill(), how='left')
            df = df.join(weekly_piv.resample('1h').ffill(), how='left')
            df[['d_pp','d_r1','d_r2','d_s1','d_s2',
                'd_pp_w','d_r1_w','d_s1_w']] = df[[
                'd_pp','d_r1','d_r2','d_s1','d_s2',
                'd_pp_w','d_r1_w','d_s1_w']].ffill()

            # Pivot Proximity: fiyat pivot ±0.5 ATR içinde mi?
            prox = atr * 0.5
            df['near_support']    = (
                (df['close'] - df['d_s1']).abs() < prox |
                (df['close'] - df['d_s2']).abs() < prox |
                (df['close'] - df['d_pp']).abs()  < prox |
                (df['close'] - df['d_s1_w']).abs() < prox
            )
            df['near_resistance'] = (
                (df['close'] - df['d_r1']).abs() < prox |
                (df['close'] - df['d_r2']).abs() < prox |
                (df['close'] - df['d_pp']).abs()  < prox |
                (df['close'] - df['d_r1_w']).abs() < prox
            )
            # Pivot bounce: sweep + pivot seviyesi = çok güçlü
            df['pivot_bull'] = df['sweep_low_bull'] & df['near_support']
            df['pivot_bear'] = df['sweep_high_bear'] & df['near_resistance']
        else:
            df['near_support']    = False
            df['near_resistance'] = False
            df['pivot_bull']      = False
            df['pivot_bear']      = False
    except Exception:
        df['near_support']    = False
        df['near_resistance'] = False
        df['pivot_bull']      = False
        df['pivot_bear']      = False

    return df



# ──────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ──────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    total_return:  float = 0.0
    cagr:          float = 0.0
    max_drawdown:  float = 0.0
    sharpe_ratio:  float = 0.0
    win_rate:      float = 0.0
    profit_factor: float = 0.0
    total_trades:  int   = 0
    final_equity:  float = 0.0
    avg_rr:        float = 0.0   # Ortalama R:R

    def fitness(self) -> float:
        """DEAP için composite fitness skoru."""
        if self.total_trades < 10:
            return -999.0
        if self.total_return <= 0:
            return float(np.clip(self.total_return * 5, -50, -0.01))
        if self.profit_factor < 1.0:
            return float(np.clip((self.profit_factor - 1) * 10, -20, -0.01))
        if self.max_drawdown > 0.75:
            return -50.0

        pf     = float(np.clip(self.profit_factor, 1.0, 15))
        wr     = float(np.clip(self.win_rate, 0, 1))
        cagr   = float(np.clip(self.cagr, 0, 20))
        sharpe = float(np.clip(self.sharpe_ratio, 0, 10))
        rr     = float(np.clip(self.avg_rr, 0, 10))
        dd_pen = max(0.1, 1 - max(0, self.max_drawdown - 0.15) * 3)

        # RR'ı da ödüllendir — kaliteli sistemler düşük işlem sayısıyla iyi RR yapar
        score = (pf * 0.30 + wr * 0.15 + cagr * 0.30 + sharpe * 0.15 + rr * 0.10) * dd_pen
        return float(np.clip(score, 0.0, 100.0))


def run_backtest(df: pd.DataFrame, p: dict,
                 initial_capital: float = 1_000.0,
                 commission: float = 0.0001,
                 slippage: float = 0.0002,
                 verbose: bool = False) -> BacktestResult:
    """
    LiquidityEdge backtest motoru.
    Giriş koşulları:
      LONG:  sweep_low_bull & ob_bull & trend_up & rsi < rsi_os_level [& in_session]
      SHORT: sweep_high_bear & ob_bear & trend_dn & rsi > rsi_ob_level [& in_session]
    Çıkış: SL/TP (ATR tabanlı)
    """
    result = BacktestResult(final_equity=initial_capital)

    try:
        df2 = add_liquidity_features(df, p)
        if len(df2) < 200:
            return result

        balance   = initial_capital
        peak_eq   = balance
        eq_curve  = [balance]
        trades    = []
        open_pos  = None

        for i in range(210, len(df2)):
            row   = df2.iloc[i]
            price = float(row['close'])
            atr   = float(row['atr']) if float(row['atr']) > 0 else price * 0.005

            # ── Oturum filtresi ──────────────────────────
            if p['session_filter'] and not bool(row.get('in_session', True)):
                if open_pos is None:
                    eq_curve.append(balance)
                    continue

            # ── Açık pozisyonu yönet ─────────────────────
            if open_pos is not None:
                entry = open_pos['entry']
                size  = open_pos['size']
                dir_  = open_pos['dir']
                moved = (price - entry) * dir_

                # Commission + slippage
                pnl = moved * size * POINT_VALUE
                pnl -= abs(price * size * POINT_VALUE) * commission

                closed = False
                if moved <= -open_pos['sl_pts']:
                    pnl    = -open_pos['sl_pts'] * size * POINT_VALUE
                    pnl   -= abs(pnl) * commission
                    closed = True
                    reason = 'SL'
                elif moved >= open_pos['tp_pts']:
                    pnl    = open_pos['tp_pts'] * size * POINT_VALUE
                    pnl   -= abs(pnl) * commission
                    closed = True
                    reason = 'TP'

                if closed:
                    balance += pnl
                    trades.append({
                        'pnl': pnl,
                        'reason': reason,
                        'rr': open_pos['tp_pts'] / (open_pos['sl_pts'] + 1e-10)
                    })
                    open_pos = None
                    if verbose:
                        print(f"  [{df2.index[i]}] {reason}: ${pnl:.2f} | Bal: ${balance:.2f}")

            # ── Yeni giriş ───────────────────────────────
            if open_pos is None:
                # Dinamik lot (compounding)
                profit_blocks = max(0, int((balance - initial_capital) // 1000))
                lot = round(p['lot_size'] * (1.0 + profit_blocks * p['growth_factor']), 3)
                lot = min(lot, 2.0)  # Max lot cap

                sl_pts = atr * p['sl_atr_mult']
                tp_pts = atr * p['tp_atr_mult']
                if sl_pts < 1: sl_pts = 1.0
                if tp_pts < sl_pts * 1.5: tp_pts = sl_pts * 2.0

                # ── SIGNAL SCORING (strict AND yerine) ───────────────
                # Her sinyal 1 puan. Eşik: 3+ puan → gir.
                # Bu hem daha fazla trade hem daha az DD demek.
                long_score = 0
                short_score = 0

                sweep_bull = bool(row.get('sweep_low_bull', False))
                sweep_bear = bool(row.get('sweep_high_bear', False))
                ob_bull    = bool(row.get('ob_bull', False))
                ob_bear    = bool(row.get('ob_bear', False))
                trend_up   = bool(row.get('trend_up', False))
                trend_dn   = bool(row.get('trend_dn', False))
                fvg_bull   = bool(row.get('fvg_bull', False))
                fvg_bear   = bool(row.get('fvg_bear', False))
                rsi_val    = float(row.get('rsi', 50))
                above_vwap = bool(row.get('above_vwap', False))
                below_vwap = bool(row.get('below_vwap', False))
                bull_div   = bool(row.get('rsi_bull_div', False))
                bear_div   = bool(row.get('rsi_bear_div', False))
                pivot_bull = bool(row.get('pivot_bull', False))
                pivot_bear = bool(row.get('pivot_bear', False))

                # LONG puanlama
                if sweep_bull:   long_score += 2   # En güçlü sinyal — 2 puan
                if ob_bull:      long_score += 2   # Kurumsal OB — 2 puan
                if pivot_bull:   long_score += 2   # Pivot destek bounce — 2 puan
                if trend_up:     long_score += 1
                if fvg_bull:     long_score += 1
                if rsi_val < p['rsi_os_level'] + 15: long_score += 1
                if above_vwap:   long_score += 1
                if bull_div:     long_score += 1

                # SHORT puanlama
                if sweep_bear:   short_score += 2
                if ob_bear:      short_score += 2
                if pivot_bear:   short_score += 2  # Pivot direnç bounce — 2 puan
                if trend_dn:     short_score += 1
                if fvg_bear:     short_score += 1
                if rsi_val > p['rsi_ob_level'] - 15: short_score += 1
                if below_vwap:   short_score += 1
                if bear_div:     short_score += 1


                # Minimum eşik: 4 puan (sweep+ob = 4 puan zaten → her ikisi lazım)
                # Veya sweep+ob+trend/fvg/rsi = 4+ puan
                MIN_SCORE = 4

                mb_th = p['meta_bias_threshold']
                meta  = float(row.get('meta_bias', 0.0))
                allow_trade = True

                if long_score >= MIN_SCORE and long_score > short_score:
                    if mb_th >= 0.05 and meta < -mb_th:
                        allow_trade = False
                    
                    if allow_trade:
                        entry = price * (1 + slippage)
                        open_pos = {
                            'entry': entry, 'dir': 1,
                            'size': lot, 'sl_pts': sl_pts, 'tp_pts': tp_pts
                        }
                        if verbose:
                            print(f"  [{df2.index[i]}] LONG @ {entry:.2f} | "
                                  f"Score:{long_score} | SL:{sl_pts:.1f} TP:{tp_pts:.1f}")

                elif short_score >= MIN_SCORE and short_score > long_score:
                    if mb_th >= 0.05 and meta > mb_th:
                        allow_trade = False
                        
                    if allow_trade:
                        entry = price * (1 - slippage)
                        open_pos = {
                            'entry': entry, 'dir': -1,
                            'size': lot, 'sl_pts': sl_pts, 'tp_pts': tp_pts
                        }
                        if verbose:
                            print(f"  [{df2.index[i]}] SHORT @ {entry:.2f} | "
                                  f"Score:{short_score} | SL:{sl_pts:.1f} TP:{tp_pts:.1f}")


            eq_curve.append(balance)

        # Açık pozisyonu kapat
        if open_pos:
            fp    = float(df2['close'].iloc[-1])
            moved = (fp - open_pos['entry']) * open_pos['dir']
            pnl   = moved * open_pos['size'] * POINT_VALUE
            balance += pnl
            trades.append({'pnl': pnl, 'reason': 'EOD', 'rr': 0})
            eq_curve[-1] = balance

        # ── Metrikler ────────────────────────────────────
        eq  = np.array(eq_curve)
        ret = np.diff(eq) / (eq[:-1] + 1e-10)

        result.total_return = (eq[-1] - initial_capital) / initial_capital
        result.final_equity = float(eq[-1])

        peak = np.maximum.accumulate(eq)
        result.max_drawdown = float(abs(((eq - peak) / (peak + 1e-10)).min()))

        # Timeframe periyodları/yıl
        try:
            d = (df2.index[1] - df2.index[0]).total_seconds()
        except Exception:
            d = 3600
        if d <= 900:    ppy = 365 * 96
        elif d <= 3600: ppy = 365 * 24
        elif d <= 14400: ppy = 365 * 6
        else:           ppy = 365

        if len(ret) > 1 and ret.std() > 1e-10:
            result.sharpe_ratio = float(np.clip(
                ret.mean() / ret.std() * np.sqrt(ppy), -10, 10))

        ny = len(eq) / ppy
        if ny > 0 and eq[-1] > 0:
            result.cagr = float(np.clip(
                (eq[-1] / initial_capital) ** (1 / max(ny, 0.1)) - 1, -1, 20))

        if trades:
            result.total_trades = len(trades)
            wins   = [t for t in trades if t['pnl'] > 0]
            losses = [t for t in trades if t['pnl'] <= 0]
            result.win_rate = len(wins) / len(trades)
            gp = sum(t['pnl'] for t in wins)
            gl = abs(sum(t['pnl'] for t in losses)) if losses else 1e-10
            result.profit_factor = float(np.clip(gp / (gl + 1e-10), 0, 20))
            rr_vals = [t['rr'] for t in trades if t.get('rr', 0) > 0]
            result.avg_rr = float(np.mean(rr_vals)) if rr_vals else 0.0

    except Exception as e:
        if verbose:
            import traceback; traceback.print_exc()

    return result


# ──────────────────────────────────────────────────────────
# XGBoost Sinyal Kalite Filtresi (Opsiyonel)
# ──────────────────────────────────────────────────────────

class LiquidityXGBFilter:
    """
    XGBoost ile sinyal öncesi kalite kontrolü.
    Strateji sinyali ürettiğinde, bu model 'geç/alma' kararı verir.
    Yüksek precision → false signal sayısını azaltır.
    """
    def __init__(self):
        self.model   = None
        self.scaler  = None
        self.trained = False
        self.feature_names = []

    def build_features(self, df: pd.DataFrame, p: dict) -> pd.DataFrame:
        """Backtest feature'larından ML özelliklerini oluştur."""
        df2 = add_liquidity_features(df, p)
        feats = pd.DataFrame(index=df2.index)

        # Fiyat dinamikleri
        feats['ret_1']  = df2['close'].pct_change(1)
        feats['ret_5']  = df2['close'].pct_change(5)
        feats['ret_20'] = df2['close'].pct_change(20)

        # Volatilite
        feats['atr_norm'] = df2['atr'] / df2['close']
        feats['vol_5']    = df2['close'].pct_change().rolling(5).std()

        # Teknik
        feats['rsi']         = df2['rsi']
        feats['ema_dist']    = (df2['close'] - df2['ema_fast']) / df2['close']
        feats['vwap_dist']   = (df2['close'] - df2['vwap']) / df2['close']

        # Liquidity sinyalleri
        feats['ob_bull']     = df2['ob_bull'].astype(float)
        feats['ob_bear']     = df2['ob_bear'].astype(float)
        feats['sweep_bull']  = df2['sweep_low_bull'].astype(float)
        feats['sweep_bear']  = df2['sweep_high_bear'].astype(float)
        feats['fvg_bull']    = df2['fvg_bull'].astype(float)
        feats['fvg_bear']    = df2['fvg_bear'].astype(float)
        feats['trend_up']    = df2['trend_up'].astype(float)
        feats['trend_dn']    = df2['trend_dn'].astype(float)
        feats['rsi_bull_div']= df2['rsi_bull_div'].astype(float)
        feats['rsi_bear_div']= df2['rsi_bear_div'].astype(float)

        # Candle şekli
        body   = df2['close'] - df2['open']
        bar_rng = (df2['high'] - df2['low']).replace(0, 1e-10)
        feats['body_ratio']  = body / bar_rng
        feats['upper_wick']  = (df2['high'] - df2[['close','open']].max(axis=1)) / bar_rng
        feats['lower_wick']  = (df2[['close','open']].min(axis=1) - df2['low']) / bar_rng

        self.feature_names = list(feats.columns)
        return feats

    def build_labels(self, df: pd.DataFrame, forward: int = 10,
                     threshold: float = 0.003) -> pd.Series:
        """Forward-return tabanlı label: 2=LONG fırsat, 0=SHORT fırsat, 1=Nötr."""
        fwd = df['close'].shift(-forward) / df['close'] - 1
        labels = pd.Series(1, index=df.index, dtype=int)
        labels[fwd > threshold]  = 2   # LONG
        labels[fwd < -threshold] = 0   # SHORT
        return labels

    def train(self, df: pd.DataFrame, p: dict,
              forward: int = 10, threshold: float = 0.003):
        """XGBoost modelini eğit — walk-forward ile overfitting önle."""
        try:
            from xgboost import XGBClassifier
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import TimeSeriesSplit
        except ImportError:
            print("⚠️  XGBoost/sklearn kurulu değil. pip install xgboost scikit-learn")
            return None

        feats  = self.build_features(df, p)
        labels = self.build_labels(df, forward, threshold)
        data   = pd.concat([feats, labels.rename('y')], axis=1).replace(
            [np.inf, -np.inf], np.nan).dropna()

        X = data[self.feature_names].values
        y = data['y'].values

        # Walk-forward val
        tscv = TimeSeriesSplit(n_splits=5)
        fold_accs = []
        scaler = StandardScaler()

        for train_idx, val_idx in tscv.split(X):
            Xtr, Xv = X[train_idx], X[val_idx]
            ytr, yv = y[train_idx], y[val_idx]
            Xtr = scaler.fit_transform(Xtr)
            Xv  = scaler.transform(Xv)
            clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                objective='multi:softprob', num_class=3,
                                eval_metric='mlogloss', random_state=42,
                                verbosity=0)
            clf.fit(Xtr, ytr)
            acc = (clf.predict(Xv) == yv).mean()
            fold_accs.append(acc)

        print(f"   XGB Walk-Forward Acc: {np.mean(fold_accs):.1%} ± {np.std(fold_accs):.1%}")

        # Son model: tüm veri ile eğit
        X_all = scaler.fit_transform(X)
        clf_final = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.8,
                                   objective='multi:softprob', num_class=3,
                                   eval_metric='mlogloss', random_state=42,
                                   verbosity=0)
        clf_final.fit(X_all, y)
        self.model   = clf_final
        self.scaler  = scaler
        self.trained = True

        # Feature importance
        imp = dict(sorted(
            zip(self.feature_names, clf_final.feature_importances_),
            key=lambda x: x[1], reverse=True))
        print("   Top-5 Features:")
        for k, v in list(imp.items())[:5]:
            print(f"     {k:<20}: {v:.4f}")
        return np.mean(fold_accs)

    def predict(self, row_features: np.ndarray) -> Tuple[int, float]:
        """0=SHORT, 1=NEUTRAL, 2=LONG. Confidence de döner."""
        if not self.trained:
            return 1, 0.34
        X = self.scaler.transform(row_features.reshape(1, -1))
        proba = self.model.predict_proba(X)[0]
        pred  = int(np.argmax(proba))
        return pred, float(proba[pred])


# ── Eğer direkt çalıştırılırsa ───────────────────────────
if __name__ == '__main__':
    print("LiquidityEdge Backtest Engine v1.0 — Doğrudan çalıştırma için")
    print("run_optimizer.py'yi kullanın.")
