"""LiquidityEdge XAU/USD DEAP Optimizer — Package"""
from .backtest_engine import (
    run_backtest, add_liquidity_features, decode,
    GENE_BOUNDS, GENOME_SIZE, GENOME_NAMES, DEFAULT_PARAMS,
    BacktestResult, LiquidityXGBFilter
)

__version__ = "1.0.0"
__all__ = [
    'run_backtest', 'add_liquidity_features', 'decode',
    'GENE_BOUNDS', 'GENOME_SIZE', 'GENOME_NAMES', 'DEFAULT_PARAMS',
    'BacktestResult', 'LiquidityXGBFilter',
]
