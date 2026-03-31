# 🧬 XAU/USD Algorithmic Trading — DEAP Genetic Optimizer Suite

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![DEAP](https://img.shields.io/badge/DEAP-Genetic%20Programming-green?style=flat-square)
![Spectral Engine](https://img.shields.io/badge/Spectral-FFT%20%2B%20HMM-purple?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-Signal%20Filter-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-red?style=flat-square)
![Asset](https://img.shields.io/badge/Asset-XAU%2FUSD%20Gold-gold?style=flat-square)
![Asset](https://img.shields.io/badge/Asset-XAG%2FUSD%20Silver-silver?style=flat-square)

**Four production-grade Precious Metals (Gold/Silver) trading systems evolved via Genetic Algorithms.**  
Walk-forward validated · Spectral Regime (HMM/FFT) filtered · Multi-core parallel optimization

</div>

---

## 🏗️ Architecture Overview

```
xauusd-deap-optimizer/
│
├── 🥇 goldmaster_deap/           ← Triple-Tap + FVG + Momentum (11-gene GA)
│   ├── 01_goldmaster_backtest.py ← HTF support/resistance + FVG detection
│   ├── 02_deap_optimizer.py      ← GA evolution engine (DEAP)
│   └── 03_run.py                 ← CLI runner with spectral injection
│
├── 🏆 holy_trinity_deap/         ← Gold Sniper Only Configuration (10-gene GA)
│   ├── run_gold_sniper_deap.py   ← Solo gold strategy with Spectral Bias
│   └── ...
│
├── 💧 liquidity_edge_deap/       ← Smart Money Liquidity System (16-gene GA)
│   ├── backtest_engine.py        ← OB + Sweep + FVG + EMA + RSI + Spectral filter
│   ├── deap_optimizer.py         ← GA evolution engine
│   └── run_optimizer.py          ← CLI runner with FFT/HMM generation
│
├── 🥈 silver_momentum_deap/      ← Pure Trend / ROC Alignment
│   ├── silver_strategy.py        ← Advanced 3-layer ROC + EMA stack + ADX
│   └── run_optimizer.py          ← XAGUSD parallel CLI runner
│
└── 🧠 spectral_bias_engine/      ← Hidden Markov & Fast Fourier Filter
    ├── adaptive_meta_labeler.py  ← RL Contextual Bandit 
    ├── fft_bias.py               ← Sine/Cosine dominant cycle extraction
    └── hmm_regime.py             ← GMM Proxy for regime detection
```

---

## 🎯 Strategy Summaries

### 1. 🥇 GoldMaster DEAP
**Concept:** Multi-tap support/resistance zones + Fair Value Gap confirmation.

| Parameter | Optimized | Range |
|-----------|-----------|-------|
| `min_taps` | **4** | 2-5 |
| `tap_atr_mult` | **0.957** | 0.3-1.0 |
| `target_atr_mult` | **1.89x** | 1.5-6.0× |
| `stop_atr_mult` | **0.50x** | 0.5-2.5× |
| `htf_window` | **65 bars** | 24-100 |
| `meta_bias_th` | **0.127** | 0.0-0.8 |
| **Fitness Score** | **4.20** | — |

**Entry Logic:**
- ≥2 touches of HTF support/resistance
- FVG confirmation optional (DEAP-evolved: off)
- Momentum breakout filter (ROC > 0.25%)
- **Spectral Rule:** FFT/HMM combined bias strength must exceed 12.7% (Filter out flat markets).

---

### 2. 🏆 Gold Sniper (Holy Trinity Legacy)
**Concept:** Institutional structural detection (Standalone Gold Module).

| Parameter | Optimized | Direction |
|-----------|-----------|-----------|
| `gs_target_pts` | **150** | ✅ Default |
| `gs_stop_pts` | **15.5** | ℹ️ wider |
| `meta_bias_th` | **0.54** | ✅ strict filter |
| **Fitness Score** | **>5.0** | — |

**System Architecture:**
```
Gold Sniper ──→ Daily Bias + 4H Rejection + Minor Sweep ──→ OB entry
                                        ↓
Spectral Filter ──→ Meta-Bias Threshold Gate ──→ execution
```

---

### 3. 💧 LiquidityEdge DEAP *(New — Smart Money Concept)*
**Concept:** Pure Smart Money / ICT methodology — detect institutional footprints and trade the sweep + reversal.

```
Signal Generation Pipeline:
[1] EMA Stack (fast/50/200)  →  Macro Trend Direction
[2] Order Block Detection    →  Institutional Interest Zone
[3] Liquidity Sweep          →  Stop Hunt Confirmation
[4] FVG (Imbalance)          →  Energy Gap Confirmation
[5] RSI + Divergence         →  Momentum Weakness Filter
[6] VWAP Position            →  Institutional Price Reference
[7] Session Filter           →  London/NY only
[8] XGBoost Filter           →  ML-based signal quality gate (optional)
```

**15-Gene Genome:**

| # | Gene | Range | Optimized |
|---|------|-------|-----------|
| 0 | `ob_lookback` | 4-20 bars | 8 |
| 1 | `sweep_margin` | 0.1-2.0× ATR | 0.62 |
| 2 | `fvg_min_size` | 0.3-1.5× ATR | 0.71 |
| 3 | `ema_trend_period` | 8-50 | 21 |
| 4 | `rsi_period` | 7-21 | 14 |
| 5 | `rsi_ob_level` | 65-80 | 72.0 |
| 6 | `rsi_os_level` | 20-35 | 28.0 |
| 7 | `atr_period` | 7-21 | 14 |
| 8 | `tp_atr_mult` | 2.0-8.0× | **4.8× ATR** |
| 9 | `sl_atr_mult` | 0.5-2.0× | 1.1× ATR |
| 10 | `lot_size` | 0.01-0.15 | 0.04 |
| 11 | `growth_factor` | 10-50% | 10% |
| 12 | `session_filter` | on/off | **off** |
| 13 | `displacement_mult` | 1.5-4.0× | 1.8× |
| 14 | `ob_body_pct` | 50-100% | 65% |
| 15 | `meta_bias_threshold`| 0.0-0.8 | **0.60** |

**Example Result (Fitness: 8.43):**

| Metric | Baseline | Spectral Optimized | Δ |
|--------|---------|-----------|---|
| CAGR | 36.7% | **123.0%** | ✅ +86.3% |
| Win Rate | 27.8% | **15.8%** | ℹ️ Sniper approach |
| Profit Factor | 1.57 | **2.80** | ✅ +1.23 |
| Avg R:R | 4.0:1 | **15.68:1** | ✅ +11.6 |
| Max Drawdown | 19.8% | **17.5%** | ✅ -2.3% |
| Final $ (1k cap) | $4,235 | **$40,408** | ✅ 9.5x more |

---

### 4. 🥈 Silver Momentum DEAP *(New)*
**Concept:** Pure Trend following & Momentum extraction perfectly tailored for XAG/USD's explosive and highly directional nature.
Features a stacked 3-period ROC, triple EMA bounds, Breakout detection, and heavy ADX filtering, topped with Spectral Bias screening.
**Evolutionary Result:** Boosted fitness beyond `1.22+` with extremely high R:R setups when paired with `meta_bias > 0.54`.

---

## ⚠️ The NQ/QQQ Conundrum (Important Note)

You will notice the `nq_alpha_deap` module is inherently isolated or absent from multi-asset cross-validation matrices like the old "Holy Trinity". **Why?**

During thousands of hours of testing, **Nasdaq (NQ) and Nasdaq ETF (QQQ)** proved to be extremely susceptible to abrupt regime drifts. Static rule-sets and generalized macro filters that work beautifully on precious metals consistently degraded on NQ's fast-paced, high-frequency structure. 

We scientifically deduced that NQ is vastly better suited for **ultra-short-term Scalper Agents** or pure Deep Reinforcement Learning models reacting to Level-2 order book structures, rather than mid-term structural momentum rules. For now, EvolutionaryQuant's edge is undisputed in Metals.

---

## ⚙️ DEAP Optimizer Design

All three systems use the same GA backbone:

```python
# Walk-forward splits (prevents overfitting)
splits = [df_test_1, df_test_2, df_test_3, df_test_4]  # n_splits=5

# Fitness = consistency-weighted mean
fitness = mean(scores) × (0.6 + 0.4 × consistency_bonus)

# Stagnation reset (adaptive sigma)
if stagnation >= limit:
    sigma = min(0.65, sigma × 1.6)
    inject 25 fresh individuals
```

| GA Parameter | Value |
|-------------|-------|
| Population | 60-70 |
| Generations | 150-200 |
| Crossover | `cxBlend(α=0.3)`, p=0.70 |
| Mutation | `mutGaussian(σ=0.3)`, p=0.25 |
| Selection | Tournament (k=5) |
| Walk-forward | 5 splits |
| Stagnation limit | 25 gens → sigma reset |

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/xauusd-deap-optimizer
cd xauusd-deap-optimizer
pip install -r requirements.txt
```

### Data
Provide your own XAUUSD OHLCV data in CSV format:
```
datetime,open,high,low,close,volume
2023-01-02 00:00:00,1823.50,1829.10,1820.30,1826.80,12500
...
```

Free data sources: [Dukascopy](https://www.dukascopy.com/swiss/english/marketwatch/historical/), [Alpha Vantage](https://www.alphavantage.co/), [Yahoo Finance](https://finance.yahoo.com/)

### Run GoldMaster DEAP
```bash
cd goldmaster_deap
python run_optimizer.py --data XAUUSD_1h.csv
python run_optimizer.py --data XAUUSD_1h.csv --generations 200 --population 80
```

### Run Holy Trinity V7
```bash
cd holy_trinity_deap
python run_optimizer.py --gold XAUUSD_1h.csv --nq NQ_1h.csv
```

### Run LiquidityEdge DEAP *(Recommended)*
```bash
cd liquidity_edge_deap

# Standard run
python run_optimizer.py --data XAUUSD_1h.csv

# With XGBoost signal filter
python run_optimizer.py --data XAUUSD_1h.csv --xgb

# Full config
python run_optimizer.py \
  --data XAUUSD_1h.csv \
  --capital 1000 \
  --generations 300 \
  --population 80 \
  --splits 5 \
  --leverage 30 \
  --xgb
```

**Expected output:**
```
💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧
  LIQUIDITYEDGE XAU/USD DEAP OPTIMIZER
💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧💧

📊 Veri: 8,760 bar  2023-01-02 → 2024-01-01

================================================================
💧 LIQUIDITYEDGE XAU/USD — DEAP OPTİMİZASYONU
   Veri         : 8,760 bar
   Splits       : 4 walk-forward
   Popülasyon   : 70
   Nesil        : 200
   CPU          : 11 core 🚀
   Genome       : 15 gen
================================================================
Gen   0: Max=  2.341 | Avg= -1.234
Gen  10: Max=  5.873 | Avg=  2.145
...
Gen 200: Max= 12.841 | Avg=  8.234

✅ Tamamlandı! (18.3 dakika)
   En iyi fitness : 12.8412

🏆 EN İYİ LiquidityEdge PARAMETRELER:
═══════════════════════════════════════════════════════════════
   ob_lookback            : 8
   sweep_margin           : 0.62
   tp_atr_mult            : 4.8
   ...
```

---

## 📈 Overfitting Protection

```
✅ Walk-forward cross-validation  (5 splits, ~30% OOS ratio)
✅ Consistency penalty in fitness  (std across splits penalized)
✅ Min 10 trades required          (sparse strategies rejected)
✅ Profit factor gate              (PF < 1.0 → negative fitness)
✅ Max drawdown cap                (DD > 75% → -50 fitness)
✅ Adaptive sigma reset            (exploits diversity when stagnant)
```

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---------|---------|
| `deap>=1.3` | Genetic algorithm framework |
| `numpy>=1.21` | Numerical computing |
| `pandas>=1.3` | Data manipulation |
| `scipy>=1.7` | Statistical tools |
| `xgboost>=1.7` | ML signal filter (optional) |
| `scikit-learn>=1.3` | Preprocessing + metrics |
| `hmmlearn>=0.3.0` | Hidden Markov Regime detection |

---

## 📁 Output Format

Results are saved in `outputs/` as JSON:

```json
{
  "timestamp": "20260315_152849",
  "fitness": 12.84,
  "best_params": {
    "ob_lookback": 8,
    "tp_atr_mult": 4.8,
    "sl_atr_mult": 1.1,
    ...
  },
  "backtest": {
    "cagr": 0.43,
    "sharpe_ratio": 2.14,
    "win_rate": 0.52,
    "profit_factor": 2.31,
    "avg_rr": 3.6,
    ...
  }
}
```

---

## ⚠️ Risk Disclaimer

> This software is for **educational and research purposes only**.  
> Past backtest performance does not guarantee future results.  
> Trading financial instruments involves significant risk of loss.  
> Always use proper position sizing and risk management.  
> The authors are not liable for any trading losses.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

**Built with ❤️ using DEAP + NumPy + Pandas**  
*Smart Money Meets Genetic Programming*

</div>
