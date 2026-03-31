"""
Gold Sniper Only — Hızlı Test (düzeltilmiş kolon isimleri)
============================================================
Holy Trinity'nin GOLD motoru, 6 yıllık XAUUSD verisi üzerinde.

Kullanım:
    python gold_sniper_test.py
"""
import sys, os, json
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from nq_core.gold_sniper_strategy import add_gold_sniper_features

CACHE_DIR = os.path.join(PARENT_DIR, 'institutional_engine', 'cache')
CSV_PATH  = os.path.join(CACHE_DIR, 'XAUUSD_1h.csv')
CAPITAL   = 1_000.0

print("=" * 64)
print("🏆 GOLD SNIPER — HIZLI TEST (NQ bağımsız)")
print("=" * 64)

df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
df.columns = df.columns.str.lower()
if 'volume' not in df.columns:
    df['volume'] = 1000
df.dropna(subset=['open','high','low','close'], inplace=True)

# add_gold_sniper_features'in ürettiği colonları görelim
df = add_gold_sniper_features(df)
print(f"\n  Veri: {len(df):,} bar  {df.index[0].date()} → {df.index[-1].date()}")
print(f"  Mevcut kolonlar: {list(df.columns)}\n")

# Compute daily bias: EMA50 eğimi (trend yönü)
ema50 = df['close'].ewm(span=50, adjust=False).mean()
df['daily_bias'] = np.where(ema50 > ema50.shift(5), 1,
                   np.where(ema50 < ema50.shift(5), -1, 0))

# VWAP proxy (oturum bazlı yoksa rolling kullan)
if 'vwap' not in df.columns:
    tp  = (df['high'] + df['low'] + df['close']) / 3
    vol = df['volume'].replace(0, 1)
    df['vwap'] = (tp * vol).rolling(20).sum() / vol.rolling(20).sum()

def run_gold_sniper_backtest(df: pd.DataFrame, params: dict,
                             initial_capital: float = 1_000.0) -> dict:
    gs_lot         = params.get('gs_lot', 0.02)
    gs_target_pts  = params.get('gs_target_pts', 140)
    gs_stop_pts    = params.get('gs_stop_pts', 12)
    gs_hard_stop   = params.get('gs_hard_stop', 500)
    gs_stale_hours = params.get('gs_stale_hours', 72)
    gs_point_value = params.get('gs_point_value', 100)
    growth_factor  = params.get('growth_factor', 0.20)

    balance   = initial_capital
    eq_curve  = [balance]
    trades    = []
    open_pos  = None
    POINT_VAL = gs_point_value

    for i in range(50, len(df)):
        row   = df.iloc[i]
        price = float(row['close'])

        # Açık pozisyon yönetimi
        if open_pos is not None:
            entry = open_pos['entry']
            dir_  = open_pos['dir']
            size  = open_pos['size']
            moved = (price - entry) * dir_
            pnl_live  = moved * size * POINT_VAL
            bars_held = i - open_pos['bar_idx']

            closed, reason = False, ''
            if moved >= gs_target_pts:
                pnl = gs_target_pts * size * POINT_VAL
                closed, reason = True, 'TP'
            elif moved <= -gs_stop_pts:
                pnl = -gs_stop_pts * size * POINT_VAL
                closed, reason = True, 'SL'
            elif pnl_live <= -gs_hard_stop:
                pnl = -gs_hard_stop
                closed, reason = True, 'HARD_STOP'
            elif bars_held >= gs_stale_hours:
                pnl = pnl_live
                closed, reason = True, 'STALE'

            if closed:
                balance += pnl
                trades.append({'pnl': pnl, 'reason': reason,
                               'rr': gs_target_pts / max(gs_stop_pts, 1)})
                open_pos = None

        # Yeni giriş — add_gold_sniper_features kolonları kullanılıyor
        if open_pos is None:
            profit_blocks = max(0, int((balance - initial_capital) // 1000))
            lot = round(gs_lot * (1.0 + profit_blocks * growth_factor), 3)
            lot = min(lot, 0.50)

            daily_bias   = int(row.get('daily_bias', 0))
            # Gerçek kolon adları (add_gold_sniper_features'dan)
            bull_ob      = bool(row.get('ob_tap_bullish', False))
            bear_ob      = bool(row.get('ob_tap_bearish', False))
            # Minor sweep: fiyat minor high/low'u sweep edip geri döndü mü?
            sweep_bull   = bool(row.get('sweep_minor_low', False))
            sweep_bear   = bool(row.get('sweep_minor_high', False))
            # 4H rejection (HTF onay)
            h4_bull      = bool(row.get('h4_reject_up', False))
            h4_bear      = bool(row.get('h4_reject_down', False))
            # Daily bias flag
            bias_bull    = bool(row.get('daily_bias_bullish', daily_bias > 0))
            bias_bear    = bool(row.get('daily_bias_bearish', daily_bias < 0))
            above_vwap   = price > float(row.get('vwap', price))
            below_vwap   = price < float(row.get('vwap', price))

            # LONG: Bullish bias + sweep low (stop hunt) + OB oluşmuş
            long_score  = 0
            short_score = 0
            # LONG puanlama (gerçek Gold Sniper mantığı)
            if bias_bull:   long_score += 2   # Daily bias yukarı
            if sweep_bull:  long_score += 2   # Minor low sweep (stop hunt)
            if h4_bull:     long_score += 1   # 4H rejection teyidi
            if bull_ob:     long_score += 1   # OB tap
            if above_vwap:  long_score += 1   # VWAP üstü

            # SHORT puanlama
            if bias_bear:   short_score += 2
            if sweep_bear:  short_score += 2
            if h4_bear:     short_score += 1
            if bear_ob:     short_score += 1
            if below_vwap:  short_score += 1

            # Giriş eşiği: bias + sweep = 4 puan minimum
            if long_score >= 4 and long_score > short_score:
                open_pos = {'entry': price, 'dir': 1, 'size': lot, 'bar_idx': i}
            elif short_score >= 4 and short_score > long_score:
                open_pos = {'entry': price, 'dir': -1, 'size': lot, 'bar_idx': i}

        eq_curve.append(balance)

    # Açık pozisyonu kapat
    if open_pos:
        fp    = float(df['close'].iloc[-1])
        moved = (fp - open_pos['entry']) * open_pos['dir']
        pnl   = moved * open_pos['size'] * POINT_VAL
        balance += pnl
        trades.append({'pnl': pnl, 'reason': 'EOD', 'rr': 0})
        eq_curve[-1] = balance

    # Metrikler
    eq  = np.array(eq_curve)
    ret = np.diff(eq) / (eq[:-1] + 1e-10)
    total_return = (eq[-1] - initial_capital) / initial_capital

    peak = np.maximum.accumulate(eq)
    max_dd = float(abs(((eq - peak) / (peak + 1e-10)).min()))

    ppy = 365 * 24
    sharpe = 0.0
    if len(ret) > 1 and ret.std() > 1e-10:
        sharpe = float(np.clip(ret.mean() / ret.std() * np.sqrt(ppy), -10, 10))

    ny = len(eq) / ppy
    cagr = 0.0
    if ny > 0 and eq[-1] > 0:
        cagr = float(np.clip((eq[-1] / initial_capital) ** (1/max(ny, 0.1)) - 1, -1, 20))

    n_trades = len(trades)
    win_rate = pf = avg_rr = 0.0
    if n_trades > 0:
        wins   = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
        win_rate = len(wins) / n_trades
        gp = sum(wins)
        gl = abs(sum(losses)) if losses else 1e-10
        pf = float(np.clip(gp / (gl + 1e-10), 0, 20))
        rrs = [t['rr'] for t in trades if t.get('rr', 0) > 0]
        avg_rr = float(np.mean(rrs)) if rrs else 0.0

    return {
        'cagr': cagr, 'total_return': total_return, 'max_drawdown': max_dd,
        'sharpe_ratio': sharpe, 'win_rate': win_rate, 'profit_factor': pf,
        'avg_rr': avg_rr, 'total_trades': n_trades, 'final_equity': float(eq[-1]),
    }


def print_result(res, label):
    pf_flag = "✅" if res['profit_factor'] > 1.0 else "❌"
    print(f"\n  {label}")
    print(f"  CAGR           : %{res['cagr']*100:+.2f}")
    print(f"  Toplam Getiri  : %{res['total_return']*100:+.2f}")
    print(f"  Max Drawdown   : %{res['max_drawdown']*100:.2f}")
    print(f"  Sharpe         : {res['sharpe_ratio']:.3f}")
    print(f"  Win Rate       : %{res['win_rate']*100:.1f}")
    print(f"  Profit Factor  : {res['profit_factor']:.2f}  {pf_flag}")
    print(f"  Avg R:R        : {res['avg_rr']:.2f}")
    print(f"  Toplam İşlem   : {res['total_trades']}")
    print(f"  Son Bakiye     : ${res['final_equity']:,.2f}")


v7_default = {
    'gs_lot': 0.02, 'gs_hard_stop': 500, 'gs_target_pts': 140,
    'gs_stop_pts': 12, 'gs_stale_hours': 72, 'growth_factor': 0.40,
    'gs_point_value': 100,
}
v7_opt = {
    'gs_lot': 0.0133, 'gs_hard_stop': 843.45, 'gs_target_pts': 149.6,
    'gs_stop_pts': 15.5, 'gs_stale_hours': 24.0, 'growth_factor': 0.20,
    'gs_point_value': 106.76,
}

r_def = run_gold_sniper_backtest(df, v7_default, CAPITAL)
r_opt = run_gold_sniper_backtest(df, v7_opt,     CAPITAL)
print_result(r_def, "DEFAULT PARAMS")
print_result(r_opt, "OPTIMUM PARAMS (V7)")

# Walk-forward
print(f"\n  Walk-Forward (4 split):")
n_splits = 4
sz = len(df) // n_splits
pos_count = 0
for i in range(n_splits):
    df_s = df.iloc[i*sz:(i+1)*sz]
    r = run_gold_sniper_backtest(df_s, v7_opt, CAPITAL)
    flag = "✅" if r['total_return'] > 0 else "❌"
    if r['total_return'] > 0: pos_count += 1
    print(f"  Split {i+1} ({df_s.index[0].date()} → {df_s.index[-1].date()}): "
          f"Getiri=%{r['total_return']*100:+.1f} | "
          f"PF={r['profit_factor']:.2f} | "
          f"Trades={r['total_trades']} {flag}")

print(f"\n  Pozitif split: {pos_count}/{n_splits}")
status = "✅ Gold Sniper PORTFOLYO HAZIR!" if pos_count >= 3 else "⚠️  DEAP optimizasyonu gerekiyor"
print(f"  {status}")
print("=" * 64)
