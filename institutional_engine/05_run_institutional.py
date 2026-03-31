"""
=============================================================
ANA ÇALIŞTIRICI — INSTITUTIONAL ENGINE v4
XM | 1:30 Kaldıraç | Cache | Dynamic Kelly
~3-4 saat versiyonu
=============================================================
"""

import pandas as pd
import numpy as np
import json, os, multiprocessing
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import importlib

data_manager           = importlib.import_module("00_data_manager")
backtest_mod           = importlib.import_module("03_backtest_engine")
optimizer_mod          = importlib.import_module("04_deap_optimizer")

build_cache              = data_manager.build_cache
InstitutionalBacktester  = backtest_mod.InstitutionalBacktester
InstitutionalDEAPOptimizer = optimizer_mod.InstitutionalDEAPOptimizer
decode_genome            = optimizer_mod.decode_genome


def full_report(df, params, initial_capital, leverage):
    bt = InstitutionalBacktester(
        initial_capital=initial_capital,
        commission=0.0001,
        slippage=0.0002,
        **params,
    )
    r = bt.run(df, verbose=False)

    leveraged_return = r.total_return * leverage
    leveraged_cagr   = ((1 + r.cagr) ** leverage - 1) if r.cagr > -1 else -1
    final_equity     = initial_capital * (1 + leveraged_return)

    print("\n" + "═" * 62)
    print("📈 XM HESABI — BACKTEST RAPORU (1:30 Kaldıraçlı)")
    print("═" * 62)
    print(f"  Başlangıç         : ${initial_capital:>10,.0f}")
    print(f"  Kaldıraç          : 1:{leverage:.0f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  CAGR (kaldıraçlı) : %{leveraged_cagr*100:>10.2f}")
    print(f"  Toplam Getiri     : %{leveraged_return*100:>10.2f}")
    print(f"  Max Drawdown      : %{r.max_drawdown*100:>10.2f}")
    print(f"  Sharpe Ratio      : {r.sharpe_ratio:>11.3f}")
    print(f"  Win Rate          : %{r.win_rate*100:>10.1f}")
    print(f"  Profit Factor     : {r.profit_factor:>11.2f}")
    print(f"  Toplam İşlem      : {r.total_trades:>11}")
    print(f"  Fitness Skoru     : {r.fitness():>11.4f}")
    print(f"  ─────────────────────────────────────────")

    if final_equity > initial_capital:
        mult  = final_equity / initial_capital
        print(f"  💰 ${initial_capital:,.0f} → ${final_equity:,.0f}  ({mult:.1f}x)")
        if r.cagr > 0:
            daily_r = (1 + leveraged_cagr) ** (1/365) - 1
            eq_60   = initial_capital * (1 + daily_r) ** 60
            print(f"  📅 60 gün proj.   : ${eq_60:,.0f}")
        flag = "✅ ULAŞILDI!" if mult >= 10 else f"hedef 10x, şu an {mult:.1f}x"
        print(f"  🎯 10x hedef      : {flag}")
    else:
        print(f"  ❌ ${initial_capital:,.0f} → ${final_equity:,.0f}  "
              f"(Zarar: %{abs(leveraged_return)*100:.1f})")
    print("═" * 62)
    return r


def mtf_report(datasets, params, initial_capital, leverage):
    print("\n🔀 MULTI-TIMEFRAME DOĞRULAMA")
    print("-" * 62)
    for tf, df_tf in datasets.items():
        try:
            bt = InstitutionalBacktester(
                initial_capital=initial_capital,
                commission=0.0001,
                slippage=0.0002,
                **params,
            )
            r    = bt.run(df_tf)
            lret = r.total_return * leverage
            flag = "✅" if lret > 0.5 else ("⚠️ " if lret > 0 else "❌")
            print(f"  {flag} {tf:5s}: Getiri=%{lret*100:6.1f} | "
                  f"DD=%{r.max_drawdown*100:5.1f} | "
                  f"PF={r.profit_factor:.2f} | "
                  f"WR=%{r.win_rate*100:.0f} | "
                  f"Trades={r.total_trades}")
        except Exception as e:
            print(f"  ❌ {tf}: {e}")


def plot_and_save(history, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    stats = history['stats']
    if not stats:
        return
    gens = [s['gen'] for s in stats]
    maxf = [s['max_fitness'] for s in stats]
    avgf = [s['avg_fitness'] for s in stats]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Institutional Engine v4', fontsize=14, fontweight='bold')
    axes[0].plot(gens, maxf, 'g-', linewidth=2, label='En İyi')
    axes[0].plot(gens, avgf, 'b--', linewidth=1.5, label='Ortalama', alpha=0.7)
    axes[0].fill_between(gens, avgf, maxf, alpha=0.1, color='green')
    axes[0].set_xlabel('Nesil'); axes[0].set_ylabel('Fitness')
    axes[0].set_title('Evrim'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    hof_f = sorted([ind.fitness.values[0] for ind in history['hof']], reverse=True)
    axes[1].bar(range(len(hof_f)), hof_f,
                color=['gold','silver','#CD7F32']+['steelblue']*max(0,len(hof_f)-3))
    axes[1].set_title('Hall of Fame'); axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(output_dir, 'evolution_v4.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Grafik: {path}")


def save_params(params, fitness, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    data = {'timestamp': ts, 'fitness': fitness, 'params': params}
    path = os.path.join(output_dir, f'best_params_v4_{ts}.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Parametreler: {path}")
    return path


def main():
    print("🏛️  INSTITUTIONAL ENGINE v4 — XM | 1:30 | Cache\n")

    # ══════════════════════════════════════════════════════
    # ⚙️  AYARLAR
    # ══════════════════════════════════════════════════════
    DATA_FOLDER  = "C:/Users/9bakl/OneDrive/Masaüstü/kalman"
    START_YEAR   = 2019
    END_YEAR     = 2025
    OPT_TF       = "1h"      # ← 1h = ~3-4 saat
                              #   15min = ~30 saat (çok yavaş)
    FORCE_CACHE  = False

    OWN_CAPITAL  = 1_000
    LEVERAGE     = 30.0
    # ══════════════════════════════════════════════════════

    # 1. Cache
    datasets = build_cache(
        data_folder=DATA_FOLDER,
        start_year=START_YEAR,
        end_year=END_YEAR,
        force=FORCE_CACHE,
    )

    df_opt = datasets[OPT_TF]
    print(f"🎯 Optimizasyon: {OPT_TF} → {len(df_opt):,} bar\n")

    # 2. Config — hız/kalite dengesi
    config = {
        'population_size':   50,    # 60 → 50
        'n_generations':     150,
        'hall_of_fame_size': 15,
        'crossover_prob':    0.70,
        'mutation_prob':     0.25,
        'tournament_size':   5,
        'initial_capital':   OWN_CAPITAL,
        'commission':        0.0001,
        'slippage':          0.0002,
        'train_pct':         0.70,
        'n_splits':          4,     # 5 → 4 (hız için)
        'stagnation_limit':  15,
        'n_jobs':            11,
        'verbose':           True,
        'log_every':         5,
    }

    # 3. Optimize
    optimizer = InstitutionalDEAPOptimizer(df_opt, config)
    best_params, history = optimizer.run()

    # 4. Raporlar
    full_report(df_opt, best_params, OWN_CAPITAL, LEVERAGE)
    mtf_report(datasets, best_params, OWN_CAPITAL, LEVERAGE)

    # 5. Kaydet
    plot_and_save(history, 'outputs')
    save_params(best_params, history['hof'][0].fitness.values[0], 'outputs')

    # 6. Top 5
    print("\n🏆 TOP 5 PARAMETRE SETİ")
    print("-" * 62)
    for i, ind in enumerate(list(history['hof'])[:5]):
        p = decode_genome(list(ind))
        print(f"#{i+1}  Fit:{ind.fitness.values[0]:.4f}  |  "
              f"Score≥{p['score_threshold']:.1f}  "
              f"Hurst={p['hurst_window']}  "
              f"OB={p['ob_lookback']}  "
              f"DCA={p['dca_step_pct']*100:.1f}%  "
              f"TP={p['target_profit_pct']*100:.1f}%  "
              f"SL={p['stop_loss_pct']*100:.1f}%  "
              f"Pos={p['position_pct']*100:.0f}%  "
              f"Sess={'✅' if p['use_session_filter'] else '❌'}")

    return best_params, history


if __name__ == '__main__':
    multiprocessing.freeze_support()
    best_params, history = main()