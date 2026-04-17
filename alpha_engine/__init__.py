"""
Alpha Engine — Unified Multi-Strategy Backtest & DEAP Optimizer
================================================================
Strategies:
  - Gold Sniper  (XAUUSD 1H)
  - Gold Master  (XAUUSD 1H)
  - Silver Reversion (XAGUSD 1H)
  - NQ Alpha     (NQ=F  5m/15m)
  - NQ Liquidity Sweep (NQ=F 5m)

All strategies:
  - SpectralBias (FFT) + HMM Regime filter
  - Kelly half-position sizing
  - DEAP genetic optimization
  - Walk-forward validation (18m train / 6m test)
"""
