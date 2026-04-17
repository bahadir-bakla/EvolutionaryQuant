"""
Gold Master Backtest — DEAP-Optimizable Adapter
=================================================
Delegates to the original, battle-tested GoldMasterBacktester
from goldmaster_deap/01_goldmaster_backtest.py.

GENOME (10 genes):
  [0] min_taps           — zone tap count before entry        (2  – 5)
  [1] tap_atr_mult       — tap detection threshold (× ATR)    (0.2 – 1.5)
  [2] momentum_thresh    — min |ROC%| for momentum entries     (0.05 – 1.0)
  [3] fvg_required       — FVG confirmation (0=off, ≥0.5=on)  (0.0 – 1.0)
  [4] target_atr_mult    — TP = X × ATR                        (2.0 – 15.0)
  [5] stop_atr_mult      — SL = X × ATR                        (0.3 – 3.0)
  [6] lot_size           — base lot size                        (0.01 – 0.20)
  [7] growth_factor      — lot compounding per $1k profit       (0.0 – 1.0)
  [8] htf_window         — HTF support/resistance window (bars) (12 – 96)
  [9] meta_bias_threshold — SpectralBias filter threshold       (0.0 – 0.7)

Proven params (goldmaster_params_20260330, fitness=4.124):
  min_taps=2, tap_atr_mult=0.843, momentum_thresh=0.350, fvg_required=False,
  target_atr_mult=5.585, stop_atr_mult=0.553, lot_size=0.051,
  growth_factor=0.1, htf_window=30, meta_bias_threshold=0.0
  -> Full 2019-2024: +3628% / 1778 trades / DD 17.9%
"""

import os
import sys
import importlib
import numpy as np

_ENGINE = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_ENGINE)
_GM_DIR = os.path.join(_ROOT, 'goldmaster_deap')

# Add goldmaster_deap to path so importlib can find "01_goldmaster_backtest"
if _GM_DIR not in sys.path:
    sys.path.insert(0, _GM_DIR)

# Load the module (filename starts with digit, must use importlib)
_gm = importlib.import_module('01_goldmaster_backtest')
GoldMasterBacktester = _gm.GoldMasterBacktester
GMParams             = _gm.GMParams
BacktestResult       = _gm.BacktestResult


# ─────────────────────────────────────────────────────────────────────
# DEAP INTERFACE
# ─────────────────────────────────────────────────────────────────────

GENE_BOUNDS = [
    (2,     5),    # [0] min_taps           int
    (0.20,  1.50), # [1] tap_atr_mult
    (0.05,  1.00), # [2] momentum_thresh
    (0.0,   1.0),  # [3] fvg_required       (0=False, >=0.5=True)
    (2.0,  15.0),  # [4] target_atr_mult
    (0.3,   3.0),  # [5] stop_atr_mult
    (0.01,  0.20), # [6] lot_size
    (0.0,   1.0),  # [7] growth_factor
    (12,   96),    # [8] htf_window         int
    (0.0,   0.70), # [9] meta_bias_threshold
]

GENE_NAMES = [
    'min_taps', 'tap_atr_mult', 'momentum_thresh', 'fvg_required',
    'target_atr_mult', 'stop_atr_mult', 'lot_size', 'growth_factor',
    'htf_window', 'meta_bias_threshold',
]

GENOME_SIZE = len(GENE_BOUNDS)


def decode_genome(genome: list) -> 'GMParams':
    """Map a float genome vector -> GMParams."""
    p = GMParams()
    for i, (lo, hi) in enumerate(GENE_BOUNDS):
        raw = float(genome[i]) if i < len(genome) else 0.5
        val = lo + abs(raw % 1.0) * (hi - lo)
        setattr(p, GENE_NAMES[i], val)

    # Discrete casts
    p.min_taps     = int(np.clip(round(p.min_taps),   2, 5))
    p.htf_window   = int(np.clip(round(p.htf_window), 12, 96))
    p.fvg_required = bool(p.fvg_required >= 0.5)

    return p


def backtest_from_raw(df_raw, params, initial_capital: float = 1_000.0) -> 'BacktestResult':
    """Run original GoldMasterBacktester on raw OHLCV data."""
    try:
        bt = GoldMasterBacktester(initial_capital=initial_capital)
        return bt.run(df_raw, params)
    except Exception:
        import traceback; traceback.print_exc()
        return BacktestResult(final_equity=initial_capital)


DEFAULT_PARAMS   = GMParams()
GoldMasterParams = GMParams   # alias for portfolio_bt compatibility
