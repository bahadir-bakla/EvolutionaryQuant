"""
REGIME DETECTOR + INSTITUTIONAL FLOW — Vectorized v2
Session logic düzeltildi, sinyal eşiği ayarlandı
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegimeState:
    is_trending:  bool  = False
    is_choppy:    bool  = False
    hurst:        float = 0.5
    chop:         float = 50.0
    kalman_vel:   float = 0.0
    kalman_acc:   float = 0.0
    trend_dir:    int   = 0


@dataclass
class InstitutionalSignal:
    direction:       int   = 0
    score:           float = 0.0
    ob_signal:       int   = 0
    fvg_signal:      int   = 0
    liquidity_swept: bool  = False
    reasons:         list  = field(default_factory=list)


class DataPrecomputer:
    """Tüm indikatörleri tek seferde hesaplar"""

    def __init__(self, df: pd.DataFrame,
                 hurst_window: int = 100,
                 chop_period:  int = 14,
                 ob_lookback:  int = 10,
                 rsi_period:   int = 14,
                 liq_period:   int = 20,
                 vwap_period:  int = 20,
                 **kwargs):
        self.df = df
        self.n  = len(df)

        closes = df['close'].values
        highs  = df['high'].values
        lows   = df['low'].values
        opens  = df['open'].values
        vols   = df['volume'].values

        self._kalman_vel, self._kalman_acc = self._kalman(closes)
        self._hurst  = self._hurst_er(closes, hurst_window)
        self._chop   = self._chop(highs, lows, closes, chop_period)
        self._rsi    = self._rsi(closes, rsi_period)
        self._vwap_d = self._vwap_dist(highs, lows, closes, vols, vwap_period)
        self._liq    = self._liquidity(highs, lows, closes, liq_period)
        self._ob     = self._order_block(opens, highs, lows, closes, ob_lookback)
        self._fvg    = self._fvg(highs, lows, closes)
        self._sess   = self._session_bias(df)

    # ── Kalman ────────────────────────────────────────────
    def _kalman(self, prices):
        n   = len(prices)
        kf  = np.zeros(n); vel = np.zeros(n); acc = np.zeros(n)
        kf[0] = prices[0]; err = 1.0
        for i in range(1, n):
            ep   = err + 0.01
            kg   = ep / (ep + 1.5)
            prev = kf[i-1]
            kf[i] = prev + kg * (prices[i] - prev)
            err   = (1 - kg) * ep
            vel[i] = kf[i] - kf[i-1]
        acc[2:] = vel[2:] - vel[1:-1]
        return vel, acc

    # ── Hurst (Efficiency Ratio) ───────────────────────────
    def _hurst_er(self, prices, window):
        n = len(prices)
        h = np.full(n, 0.5)
        for i in range(window, n):
            seg = prices[i-window:i+1]
            d   = abs(seg[-1] - seg[0])
            v   = np.abs(np.diff(seg)).sum() + 1e-10
            h[i] = d / v
        return np.clip(h, 0, 1)

    # ── Choppiness ────────────────────────────────────────
    def _chop(self, highs, lows, closes, period):
        n    = len(closes)
        chop = np.full(n, 50.0)
        tr   = np.zeros(n)
        tr[1:] = np.maximum(highs[1:]-lows[1:],
                 np.maximum(np.abs(highs[1:]-closes[:-1]),
                            np.abs(lows[1:]-closes[:-1])))
        for i in range(period, n):
            s = tr[i-period+1:i+1].sum()
            r = highs[i-period+1:i+1].max() - lows[i-period+1:i+1].min()
            if r > 0:
                chop[i] = 100 * np.log10(max(s,1e-10)/r) / np.log10(period)
        return np.clip(chop, 0, 100)

    # ── RSI ───────────────────────────────────────────────
    def _rsi(self, prices, period):
        d     = np.diff(prices, prepend=prices[0])
        gain  = pd.Series(np.where(d > 0, d, 0.0)).ewm(span=period, adjust=False).mean().values
        loss  = pd.Series(np.where(d < 0, -d, 0.0)).ewm(span=period, adjust=False).mean().values
        rs    = gain / (loss + 1e-10)
        return 100 - 100 / (1 + rs)

    # ── VWAP Distance ─────────────────────────────────────
    def _vwap_dist(self, highs, lows, closes, vols, period):
        tp   = (highs + lows + closes) / 3
        tpv  = pd.Series(tp * vols).rolling(period).sum()
        vol_ = pd.Series(vols).rolling(period).sum()
        vwap = (tpv / (vol_ + 1e-10)).values
        return (closes - vwap) / (vwap + 1e-10)

    # ── Liquidity Sweep ───────────────────────────────────
    def _liquidity(self, highs, lows, closes, period):
        n   = len(closes)
        sig = np.zeros(n, dtype=int)
        for i in range(period, n):
            sh = highs[i-period:i].max()
            sl = lows[i-period:i].min()
            if highs[i] > sh and closes[i] < sh:
                sig[i] = -1
            elif lows[i] < sl and closes[i] > sl:
                sig[i] = 1
        return sig

    # ── Order Block ───────────────────────────────────────
    def _order_block(self, opens, highs, lows, closes, lookback):
        n    = len(closes)
        sig  = np.zeros(n, dtype=int)
        body = np.abs(closes - opens)
        rng  = highs - lows + 1e-10
        br   = body / rng
        bull = closes > opens
        for i in range(lookback+2, n):
            c = closes[i]
            for j in range(max(0,i-lookback-2), i-1):
                if br[j] < 0.55: continue
                if lows[j] <= c <= highs[j]:
                    sig[i] = 1 if bull[j] else -1
                    break
        return sig

    # ── FVG ───────────────────────────────────────────────
    def _fvg(self, highs, lows, closes):
        n   = len(closes)
        sig = np.zeros(n, dtype=int)
        for i in range(2, n):
            c = closes[i]
            if highs[i-2] < lows[i] and highs[i-2] <= c <= lows[i]:
                sig[i] = 1
            elif lows[i-2] > highs[i] and highs[i] <= c <= lows[i-2]:
                sig[i] = -1
        return sig

    # ── Session Bias ──────────────────────────────────────
    def _session_bias(self, df):
        """
        Asya+Londra yönü → Amerika ters
        Her gün için hesaplanır, Amerika saatlerinde uygulanır
        """
        n   = len(df)
        sig = np.zeros(n, dtype=int)

        try:
            # Günlük session dönüş değerlerini önceden hesapla
            df2 = df.copy()
            df2['hour'] = df2.index.hour
            df2['date'] = df2.index.date

            for date, day_df in df2.groupby('date'):
                asia_bars   = day_df[day_df['hour'].between(0, 6)]
                london_bars = day_df[day_df['hour'].between(7, 12)]
                ny_bars     = day_df[day_df['hour'].between(13, 19)]

                if len(asia_bars) < 2 or len(london_bars) < 2 or len(ny_bars) < 1:
                    continue

                asia_ret   = float(asia_bars['close'].iloc[-1]   - asia_bars['open'].iloc[0])
                london_ret = float(london_bars['close'].iloc[-1] - london_bars['open'].iloc[0])

                # Her ikisi de yukarı → Amerika short
                if asia_ret > 0 and london_ret > 0:
                    bias = 1
                # Her ikisi de aşağı → Amerika long
                elif asia_ret < 0 and london_ret < 0:
                    bias = -1
                else:
                    continue

                # NY barlarına uygula
                ny_idx = ny_bars.index
                mask   = df.index.isin(ny_idx)
                sig[mask] = bias

        except Exception:
            pass

        return sig

    # ── Index Erişimi ─────────────────────────────────────
    def regime_at(self, idx):
        s = RegimeState()
        s.kalman_vel  = float(self._kalman_vel[idx])
        s.kalman_acc  = float(self._kalman_acc[idx])
        s.hurst       = float(self._hurst[idx])
        s.chop        = float(self._chop[idx])
        s.is_trending = s.hurst > 0.30 and s.chop < 65.0
        s.is_choppy   = s.chop > 70.0
        if s.kalman_vel > 0.03:
            s.trend_dir = 1
        elif s.kalman_vel < -0.03:
            s.trend_dir = -1
        return s

    def signal_at(self, idx, score_threshold=3.0):
        sig = InstitutionalSignal()
        ls  = 0.0
        ss  = 0.0
        r   = sig.reasons

        # Session (en güçlü — %90 win rate)
        sb = int(self._sess[idx])
        if sb == -1:
            ls += 3.0; r.append("Sess:L")
        elif sb == 1:
            ss += 3.0; r.append("Sess:S")

        # Liquidity sweep
        lq = int(self._liq[idx])
        sig.liquidity_swept = lq != 0
        if lq == 1:
            ls += 2.5; r.append("Liq:L")
        elif lq == -1:
            ss += 2.5; r.append("Liq:S")

        # Order block
        ob = int(self._ob[idx])
        sig.ob_signal = ob
        if ob == 1:
            ls += 2.0; r.append("OB:L")
        elif ob == -1:
            ss += 2.0; r.append("OB:S")

        # FVG
        fvg = int(self._fvg[idx])
        sig.fvg_signal = fvg
        if fvg == 1:
            ls += 1.5; r.append("FVG:L")
        elif fvg == -1:
            ss += 1.5; r.append("FVG:S")

        # VWAP
        vd = float(self._vwap_d[idx])
        if vd < -0.001:
            ls += 1.0
        elif vd > 0.001:
            ss += 1.0

        # RSI
        rsi = float(self._rsi[idx])
        if rsi < 40:
            ls += 1.0; r.append(f"RSI:{rsi:.0f}")
        elif rsi > 60:
            ss += 1.0; r.append(f"RSI:{rsi:.0f}")

        # Kalman momentum
        kv = float(self._kalman_vel[idx])
        if kv > 0.05:
            ls += 0.5
        elif kv < -0.05:
            ss += 0.5

        sig.score = max(ls, ss)
        if ls >= score_threshold and ls > ss:
            sig.direction = 1
        elif ss >= score_threshold and ss > ls:
            sig.direction = -1

        return sig


# Geriye uyumluluk
class RegimeDetector:
    def __init__(self, hurst_window=100, chop_period=14):
        self.hurst_window = hurst_window
        self.chop_period  = chop_period

class InstitutionalFlowDetector:
    def __init__(self, ob_lookback=10, resistance_lookback=50, score_threshold=3.0):
        self.ob_lookback         = ob_lookback
        self.resistance_lookback = resistance_lookback
        self.score_threshold     = score_threshold