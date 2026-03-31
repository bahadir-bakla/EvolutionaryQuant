"""
Bulunan en iyi parametreleri direkt test et
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib
import pandas as pd

# Modülleri yükle
dm  = importlib.import_module("00_data_manager")
bt  = importlib.import_module("03_backtest_engine")

build_cache             = dm.build_cache
InstitutionalBacktester = bt.InstitutionalBacktester

# En iyi parametreler
best_params = {
    "score_threshold":      4.538568119938975,
    "hurst_window":         119,
    "chop_period":          29,
    "ob_lookback":          17,
    "resistance_lookback":  84,
    "dca_step_pct":         0.019071604976239262,
    "target_profit_pct":    0.02085648875608801,
    "stop_loss_pct":        0.03343851317590369,
    "position_pct":         0.08963511820154674,
    "max_layers":           2,
    "use_session_filter":   True,
}

# Veri yükle (cache varsa hızlı)
DATA_FOLDER = "C:/Users/9bakl/OneDrive/Masaüstü/kalman"
datasets = build_cache(DATA_FOLDER, 2019, 2025, force=False)

LEVERAGE = 30.0
CAPITAL  = 1_000

print("\n" + "═"*62)
print("🔬 BEST PARAMS TEST — Tüm Timeframe'ler")
print("═"*62)

for tf, df_tf in datasets.items():
    backtester = InstitutionalBacktester(
        initial_capital = CAPITAL,
        commission      = 0.0001,
        slippage        = 0.0002,
        **best_params,
    )
    r = backtester.run(df_tf, verbose=False)
    lret = r.total_return * LEVERAGE
    feq  = CAPITAL * (1 + lret)
    flag = "✅" if lret > 0 else "❌"

    print(f"\n{flag} {tf}")
    print(f"   Getiri (1:30)  : %{lret*100:.1f}  →  ${feq:,.0f}")
    print(f"   CAGR           : %{r.cagr*100:.1f}")
    print(f"   Max Drawdown   : %{r.max_drawdown*100:.1f}")
    print(f"   Win Rate       : %{r.win_rate*100:.0f}")
    print(f"   Profit Factor  : {r.profit_factor:.2f}")
    print(f"   Toplam İşlem   : {r.total_trades}")
    print(f"   Fitness        : {r.fitness():.4f}")

print("\n" + "═"*62)