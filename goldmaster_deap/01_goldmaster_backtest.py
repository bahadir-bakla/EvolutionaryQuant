"""
GoldMaster Backtest Engine
Triple-Tap + FVG + Momentum kırılımı
Parametreler DEAP tarafından optimize edilir
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


def add_features(df: pd.DataFrame, htf_window: int = 48) -> pd.DataFrame:
    df = df.copy()

    # HTF destek/direnç
    df['htf_support']    = df['low'].rolling(htf_window).min().shift(1)
    df['htf_resistance'] = df['high'].rolling(htf_window).max().shift(1)

    # FVG
    df['fvg_bullish'] = df['low'] > df['high'].shift(2)
    df['fvg_bearish'] = df['high'] < df['low'].shift(2)

    # Momentum
    df['momentum_roc'] = df['close'].pct_change(10) * 100

    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low']  - df['close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # RSI
    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['rsi'] = 100 - 100 / (1 + gain / (loss + 1e-10))

    # VWAP (rolling)
    tp  = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).rolling(20).sum() / (df['volume'].rolling(20).sum() + 1e-10)

    return df.dropna()


@dataclass
class GMParams:
    """DEAP tarafından optimize edilecek parametreler"""
    # Giriş kriterleri
    min_taps:        int   = 3       # Triple-tap için kaç dokunuş lazım
    tap_atr_mult:    float = 0.5     # Dokunuş eşiği: ATR * bu
    momentum_thresh: float = 0.15   # Kırılım için minimum momentum
    fvg_required:    bool  = True    # FVG konfirmasyon şart mı?

    # Risk yönetimi
    target_atr_mult: float = 3.0    # TP = ATR * bu
    stop_atr_mult:   float = 1.0    # SL = ATR * bu
    lot_size:        float = 0.05   # Başlangıç lot

    # Para yönetimi
    position_pct:    float = 0.10   # Sermayenin yüzdesi
    growth_factor:   float = 0.30   # Her $1k karda lot artışı

    # HTF penceresi
    htf_window:      int   = 48     # 48 bar = 4h on 1h data
    
    # Spectral Bias
    meta_bias_threshold: float = 0.0


@dataclass
class BacktestResult:
    total_return:  float = 0.0
    cagr:          float = 0.0
    max_drawdown:  float = 0.0
    win_rate:      float = 0.0
    profit_factor: float = 0.0
    total_trades:  int   = 0
    sharpe_ratio:  float = 0.0
    final_equity:  float = 0.0

    def fitness(self) -> float:
        if self.total_trades < 5:
            return -999.0
        if self.total_return <= 0:
            return float(np.clip(self.total_return * 5, -50, -0.01))
        if self.max_drawdown > 0.60:
            return -50.0

        pf     = float(np.clip(self.profit_factor, 0, 10))
        wr     = float(np.clip(self.win_rate, 0, 1))
        cagr   = float(np.clip(self.cagr, 0, 10))
        sharpe = float(np.clip(self.sharpe_ratio, -5, 10))

        score = pf * 0.35 + wr * 0.20 + cagr * 0.30 + sharpe * 0.15

        if self.max_drawdown > 0.20:
            score *= max(0.1, 1 - (self.max_drawdown - 0.20) * 2)

        return float(np.clip(score, 0.01, 50.0))


class GoldMasterBacktester:

    POINT_VALUE = 100.0  # 1 altın noktası = $1

    def __init__(self, initial_capital: float = 1_000.0,
                 commission: float = 0.0001,
                 slippage: float = 0.0002):
        self.initial_capital = initial_capital
        self.commission      = commission
        self.slippage        = slippage

    def run(self, df: pd.DataFrame, p: GMParams,
            verbose: bool = False) -> BacktestResult:

        result = BacktestResult()
        result.final_equity = self.initial_capital

        try:
            df2 = add_features(df, p.htf_window)
            if len(df2) < 100:
                return result

            balance     = self.initial_capital
            eq_curve    = [balance]
            trades      = []
            open_pos    = None

            # Bölge hafızası
            support_taps    = 0
            resistance_taps = 0
            last_support    = None
            last_resistance = None

            for i in range(50, len(df2)):
                row   = df2.iloc[i]
                price = row['close']
                atr   = row['atr'] if row['atr'] > 0 else price * 0.005

                support    = row['htf_support']
                resistance = row['htf_resistance']

                # Bölge resetle
                if last_support != support:
                    support_taps = 0
                    last_support = support
                if last_resistance != resistance:
                    resistance_taps = 0
                    last_resistance = resistance

                # Dokunuş say
                tap_thresh = atr * p.tap_atr_mult
                if support and abs(row['low'] - support) <= tap_thresh:
                    support_taps += 1
                if resistance and abs(row['high'] - resistance) <= tap_thresh:
                    resistance_taps += 1

                # ── Açık pozisyonu yönet ──────────────────
                if open_pos:
                    moved = (price - open_pos['entry']) * open_pos['dir']
                    pnl   = moved * open_pos['size'] * self.POINT_VALUE
                    pnl  -= abs(pnl) * self.commission

                    closed = False
                    if moved <= -open_pos['sl_pts']:
                        pnl    = -open_pos['sl_pts'] * open_pos['size'] * self.POINT_VALUE
                        pnl   -= abs(pnl) * self.commission
                        closed = True; reason = 'SL'
                    elif moved >= open_pos['tp_pts']:
                        pnl    = open_pos['tp_pts'] * open_pos['size'] * self.POINT_VALUE
                        pnl   -= abs(pnl) * self.commission
                        closed = True; reason = 'TP'

                    if closed:
                        balance += pnl
                        trades.append({'pnl': pnl, 'reason': reason,
                                       'dir': open_pos['dir'],
                                       'entry_ts': open_pos.get('ts'),
                                       'exit_ts': df2.index[i]})
                        open_pos = None
                        if verbose:
                            print(f"  [{df2.index[i]}] {reason}: ${pnl:.2f} | Bal: ${balance:.2f}")

                # ── Yeni giriş ────────────────────────────
                if open_pos is None:

                    meta  = float(row.get('meta_bias', 0.0))
                    mb_th = p.meta_bias_threshold

                    # Dinamik lot (compounding)
                    profit_blocks = max(0, int((balance - self.initial_capital) // 1000))
                    mult = 1.0 + profit_blocks * p.growth_factor
                    lot  = round(p.lot_size * mult, 3)

                    sl_pts = atr * p.stop_atr_mult
                    tp_pts = atr * p.target_atr_mult

                    # Giriş A: Triple-tap destek + FVG → LONG
                    fvg_ok = row['fvg_bullish'] if p.fvg_required else True
                    if (support_taps >= p.min_taps and
                            row['close'] > support and fvg_ok):
                        if mb_th < 0.05 or meta >= -mb_th:
                            entry = price * (1 + self.slippage)
                            open_pos = {'entry': entry, 'dir': 1, 'ts': df2.index[i],
                                        'size': lot, 'sl_pts': sl_pts, 'tp_pts': tp_pts}
                            support_taps = 0
                            if verbose:
                                print(f"  [{df2.index[i]}] LONG: {entry:.2f} | SL:{sl_pts:.1f} TP:{tp_pts:.1f}")

                    # Giriş B: Triple-tap direnç + FVG → SHORT
                    elif (resistance_taps >= p.min_taps and
                              row['close'] < resistance and
                              (row['fvg_bearish'] if p.fvg_required else True)):
                        if mb_th < 0.05 or meta <= mb_th:
                            entry = price * (1 - self.slippage)
                            open_pos = {'entry': entry, 'dir': -1, 'ts': df2.index[i],
                                        'size': lot, 'sl_pts': sl_pts, 'tp_pts': tp_pts}
                            resistance_taps = 0
                            if verbose:
                                print(f"  [{df2.index[i]}] SHORT: {entry:.2f}")

                    # Giriş C: Momentum kırılımı → LONG
                    elif (resistance_taps >= p.min_taps - 1 and
                              row['close'] > resistance and
                              row['momentum_roc'] > p.momentum_thresh):
                        if mb_th < 0.05 or meta >= -mb_th:
                            entry = price * (1 + self.slippage)
                            open_pos = {'entry': entry, 'dir': 1, 'ts': df2.index[i],
                                        'size': lot, 'sl_pts': sl_pts, 'tp_pts': tp_pts}
                            resistance_taps = 0

                    # Giriş D: Momentum çöküşü → SHORT
                    elif (support_taps >= p.min_taps - 1 and
                              row['close'] < support and
                              row['momentum_roc'] < -p.momentum_thresh):
                        if mb_th < 0.05 or meta <= mb_th:
                            entry = price * (1 - self.slippage)
                            open_pos = {'entry': entry, 'dir': -1, 'ts': df2.index[i],
                                        'size': lot, 'sl_pts': sl_pts, 'tp_pts': tp_pts}
                            support_taps = 0

                eq_curve.append(balance)

            # Açık pozisyonu kapat
            if open_pos:
                fp    = df2['close'].iloc[-1]
                moved = (fp - open_pos['entry']) * open_pos['dir']
                pnl   = moved * open_pos['size'] * self.POINT_VALUE
                balance += pnl
                trades.append({'pnl': pnl, 'reason': 'EOD',
                               'dir': open_pos['dir'],
                               'entry_ts': open_pos.get('ts'),
                               'exit_ts': df2.index[-1]})
                eq_curve[-1] = balance

            # ── Metrikler ─────────────────────────────────
            eq  = np.array(eq_curve)
            ret = np.diff(eq) / (eq[:-1] + 1e-10)

            result.total_return = (eq[-1] - self.initial_capital) / self.initial_capital
            result.final_equity = float(eq[-1])

            peak = np.maximum.accumulate(eq)
            result.max_drawdown = float(abs(((eq - peak) / (peak + 1e-10)).min()))

            ppy = self._timeframe(df2)
            if len(ret) > 1 and ret.std() > 1e-10:
                result.sharpe_ratio = float(np.clip(
                    ret.mean() / ret.std() * np.sqrt(ppy), -10, 10))

            ny = len(df2) / ppy
            if ny > 0 and eq[-1] > 0:
                result.cagr = float(np.clip(
                    (eq[-1] / self.initial_capital) ** (1 / max(ny, 0.1)) - 1,
                    -1, 10))

            if trades:
                result.total_trades = len(trades)
                wins   = [t for t in trades if t['pnl'] > 0]
                losses = [t for t in trades if t['pnl'] <= 0]
                result.win_rate = len(wins) / len(trades)
                gp = sum(t['pnl'] for t in wins)
                gl = abs(sum(t['pnl'] for t in losses)) if losses else 1e-10
                result.profit_factor = float(np.clip(gp / (gl + 1e-10), 0, 20))

        except Exception as e:
            if verbose:
                import traceback; traceback.print_exc()

        return result

    def _timeframe(self, df):
        try:
            d = (df.index[1] - df.index[0]).total_seconds()
            if d <= 900:   return 365 * 96
            if d <= 3600:  return 365 * 24
            if d <= 14400: return 365 * 6
            if d <= 86400: return 365
            return 52
        except:
            return 252
