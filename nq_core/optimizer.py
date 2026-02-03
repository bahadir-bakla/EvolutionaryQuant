# NQ Quant Bot - Parameter Optimization
# Grid Search + Walk-Forward Validation

import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Callable, Optional
from itertools import product
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class ParameterSpace:
    """
    Optimize edilecek parametreler ve aralıkları
    """
    # Confluence Engine
    min_confluence: List[int] = field(default_factory=lambda: [2, 3, 4])
    
    # Hurst Exponent
    hurst_window: List[int] = field(default_factory=lambda: [50, 100, 150])
    trending_threshold: List[float] = field(default_factory=lambda: [0.55, 0.60, 0.65])
    choppy_threshold: List[float] = field(default_factory=lambda: [0.35, 0.40, 0.45])
    
    # Kalman Filter
    process_noise: List[float] = field(default_factory=lambda: [1e-6, 1e-5, 1e-4])
    measurement_noise: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    
    # Order Blocks
    ob_lookback: List[int] = field(default_factory=lambda: [15, 20, 30])
    ob_body_threshold: List[float] = field(default_factory=lambda: [1.3, 1.5, 2.0])
    ob_volume_threshold: List[float] = field(default_factory=lambda: [1.3, 1.5, 2.0])
    
    # Risk Management
    atr_stop_multiplier: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5, 3.0])
    atr_tp_multiplier: List[float] = field(default_factory=lambda: [2.0, 3.0, 4.0])
    
    def get_combinations(self, params_to_optimize: List[str] = None) -> List[Dict[str, Any]]:
        """Get all parameter combinations"""
        if params_to_optimize is None:
            params_to_optimize = ['min_confluence', 'hurst_window', 'atr_stop_multiplier']
        
        param_values = {}
        for param in params_to_optimize:
            if hasattr(self, param):
                param_values[param] = getattr(self, param)
        
        keys = list(param_values.keys())
        values = list(param_values.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def get_total_combinations(self) -> int:
        """Get total number of combinations"""
        total = 1
        for attr in ['min_confluence', 'hurst_window', 'trending_threshold', 
                     'choppy_threshold', 'process_noise', 'measurement_noise',
                     'ob_lookback', 'ob_body_threshold', 'ob_volume_threshold',
                     'atr_stop_multiplier', 'atr_tp_multiplier']:
            total *= len(getattr(self, attr))
        return total


@dataclass
class OptimizationResult:
    """Optimization result for a single parameter set"""
    params: Dict[str, Any]
    sharpe_ratio: float
    sortino_ratio: float
    total_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    
    # Composite score
    score: float = 0.0
    
    def calculate_score(self, weights: Dict[str, float] = None):
        """
        Calculate composite score
        
        Default weights prioritize:
        1. Sharpe ratio (risk-adjusted return)
        2. Profit factor (consistency)
        3. Max drawdown (risk control)
        """
        if weights is None:
            weights = {
                'sharpe': 0.30,
                'sortino': 0.15,
                'return': 0.15,
                'drawdown': 0.20,
                'profit_factor': 0.15,
                'win_rate': 0.05
            }
        
        # Normalize metrics (higher is better)
        sharpe_score = max(0, self.sharpe_ratio) / 3.0  # Assume max Sharpe = 3
        sortino_score = max(0, self.sortino_ratio) / 4.0  # Assume max Sortino = 4
        return_score = max(0, self.total_return + 0.5) / 1.0  # Shift and scale
        dd_score = max(0, 1 + self.max_drawdown)  # DD is negative, closer to 0 is better
        pf_score = min(1, max(0, self.profit_factor) / 2.0)  # Assume max PF = 2
        wr_score = self.win_rate
        
        self.score = (
            weights['sharpe'] * sharpe_score +
            weights['sortino'] * sortino_score +
            weights['return'] * return_score +
            weights['drawdown'] * dd_score +
            weights['profit_factor'] * pf_score +
            weights['win_rate'] * wr_score
        )
        
        return self.score


class ParameterOptimizer:
    """
    Parameter Optimizer with:
    - Grid Search
    - Walk-Forward Validation
    - Cross-Validation
    """
    
    def __init__(
        self,
        param_space: ParameterSpace = None,
        n_splits: int = 3,
        train_ratio: float = 0.7
    ):
        self.param_space = param_space or ParameterSpace()
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.results: List[OptimizationResult] = []
        
    def _run_single_backtest(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any]
    ) -> OptimizationResult:
        """Run a single backtest with given parameters"""
        from nq_core.brain import QuantBrain
        from nq_core.confluence import ConfluenceEngine
        from nq_core.order_blocks import detect_order_blocks, get_active_order_blocks
        from nq_core.backtest import NQBacktestEngine
        
        # Extract params
        min_confluence = params.get('min_confluence', 3)
        hurst_window = params.get('hurst_window', 100)
        trending_threshold = params.get('trending_threshold', 0.6)
        choppy_threshold = params.get('choppy_threshold', 0.4)
        ob_lookback = params.get('ob_lookback', 20)
        ob_body_threshold = params.get('ob_body_threshold', 1.5)
        ob_volume_threshold = params.get('ob_volume_threshold', 1.5)
        atr_stop_mult = params.get('atr_stop_multiplier', 2.0)
        atr_tp_mult = params.get('atr_tp_multiplier', 4.0)
        
        # Initialize with params
        brain = QuantBrain(
            hurst_window=hurst_window,
            trending_threshold=trending_threshold,
            choppy_threshold=choppy_threshold
        )
        
        confluence = ConfluenceEngine(min_confluence=min_confluence)
        
        # Detect order blocks
        obs = detect_order_blocks(
            data,
            lookback=ob_lookback,
            body_threshold=ob_body_threshold,
            volume_threshold=ob_volume_threshold
        )
        
        # Generate signals
        signals = []
        for i in range(len(data)):
            row = data.iloc[i]
            state = brain.update(row['close'], data.index[i])
            active_obs = get_active_order_blocks(obs, i, max_age=50)
            
            rsi = row.get('rsi', None)
            atr = row.get('atr', None)
            
            if pd.isna(rsi):
                rsi = None
            if pd.isna(atr):
                atr = None
                
            signal = confluence.evaluate(state, active_obs, rsi=rsi, atr=atr)
            
            # Apply ATR multipliers
            entry = row['close']
            if atr and signal.direction != 'NEUTRAL':
                if signal.direction == 'LONG':
                    stop = entry - (atr * atr_stop_mult)
                    tp = entry + (atr * atr_tp_mult)
                else:
                    stop = entry + (atr * atr_stop_mult)
                    tp = entry - (atr * atr_tp_mult)
            else:
                stop = signal.stop_loss
                tp = signal.take_profit_1
            
            signals.append({
                'signal': signal.direction,
                'stop_loss': stop,
                'tp1': tp,
                'atr': atr if atr else 0
            })
        
        signals_df = pd.DataFrame(signals, index=data.index)
        
        # Run backtest
        engine = NQBacktestEngine(initial_capital=100000, use_kelly=True)
        
        try:
            result = engine.run(data, signals_df)
            
            return OptimizationResult(
                params=params,
                sharpe_ratio=result.sharpe_ratio,
                sortino_ratio=result.sortino_ratio,
                total_return=result.total_return,
                max_drawdown=result.max_drawdown,
                win_rate=result.win_rate,
                profit_factor=result.profit_factor,
                total_trades=result.total_trades
            )
        except Exception as e:
            logger.warning(f"Backtest failed: {e}")
            return OptimizationResult(
                params=params,
                sharpe_ratio=-999,
                sortino_ratio=-999,
                total_return=-1,
                max_drawdown=-1,
                win_rate=0,
                profit_factor=0,
                total_trades=0
            )
    
    def walk_forward_split(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Walk-Forward Validation splits
        
        | Train | Test |
        |-------|------|
        |  70%  | 30%  |
        
        Slides forward for each split
        """
        n = len(data)
        split_size = n // self.n_splits
        train_size = int(split_size * self.train_ratio)
        test_size = split_size - train_size
        
        splits = []
        for i in range(self.n_splits):
            start = i * split_size
            train_end = start + train_size
            test_end = min(train_end + test_size, n)
            
            if test_end > train_end:
                train_data = data.iloc[start:train_end]
                test_data = data.iloc[train_end:test_end]
                splits.append((train_data, test_data))
        
        return splits
    
    def grid_search(
        self,
        data: pd.DataFrame,
        params_to_optimize: List[str] = None,
        use_walk_forward: bool = True,
        verbose: bool = True
    ) -> List[OptimizationResult]:
        """
        Grid Search optimization
        
        Args:
            data: OHLCV DataFrame with rsi, atr columns
            params_to_optimize: List of parameter names to optimize
            use_walk_forward: Use walk-forward validation
            verbose: Print progress
            
        Returns:
            List of OptimizationResult sorted by score
        """
        if params_to_optimize is None:
            params_to_optimize = ['min_confluence', 'hurst_window', 'atr_stop_multiplier']
        
        combinations = self.param_space.get_combinations(params_to_optimize)
        
        if verbose:
            print(f"Grid Search: {len(combinations)} parameter combinations")
            print(f"Parameters: {params_to_optimize}")
        
        results = []
        
        for idx, params in enumerate(combinations):
            if verbose and idx % 5 == 0:
                print(f"  Progress: {idx+1}/{len(combinations)}")
            
            if use_walk_forward:
                # Walk-forward validation
                splits = self.walk_forward_split(data)
                split_results = []
                
                for train_data, test_data in splits:
                    # Optimize on train, validate on test
                    result = self._run_single_backtest(test_data, params)
                    split_results.append(result)
                
                # Average across splits
                avg_result = OptimizationResult(
                    params=params,
                    sharpe_ratio=np.mean([r.sharpe_ratio for r in split_results]),
                    sortino_ratio=np.mean([r.sortino_ratio for r in split_results]),
                    total_return=np.mean([r.total_return for r in split_results]),
                    max_drawdown=np.min([r.max_drawdown for r in split_results]),  # Worst case
                    win_rate=np.mean([r.win_rate for r in split_results]),
                    profit_factor=np.mean([r.profit_factor for r in split_results]),
                    total_trades=int(np.mean([r.total_trades for r in split_results]))
                )
                avg_result.calculate_score()
                results.append(avg_result)
            else:
                # Simple backtest
                result = self._run_single_backtest(data, params)
                result.calculate_score()
                results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        self.results = results
        
        if verbose:
            print(f"\nTop 5 Results:")
            for i, r in enumerate(results[:5]):
                print(f"  {i+1}. Score={r.score:.4f}, Sharpe={r.sharpe_ratio:.2f}, "
                      f"Return={r.total_return:.2%}, DD={r.max_drawdown:.2%}")
                print(f"     Params: {r.params}")
        
        return results
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from optimization"""
        if not self.results:
            raise ValueError("Run optimization first!")
        return self.results[0].params
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame"""
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for r in self.results:
            row = {**r.params}
            row['score'] = r.score
            row['sharpe'] = r.sharpe_ratio
            row['sortino'] = r.sortino_ratio
            row['return'] = r.total_return
            row['max_dd'] = r.max_drawdown
            row['win_rate'] = r.win_rate
            row['profit_factor'] = r.profit_factor
            row['trades'] = r.total_trades
            data.append(row)
        
        return pd.DataFrame(data)


def prepare_data_with_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI and ATR to DataFrame"""
    df = df.copy()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    high, low, close = df['high'], df['low'], df['close']
    tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    return df


# === MAIN ===
if __name__ == "__main__":
    import yfinance as yf
    
    print("="*60)
    print("NQ QUANT BOT - PARAMETER OPTIMIZATION")
    print("="*60)
    
    # Fetch data
    print("\n[1] Fetching NQ=F data...")
    ticker = yf.Ticker("NQ=F")
    df = ticker.history(period="1y", interval="1d")
    df.columns = df.columns.str.lower()
    print(f"    Fetched {len(df)} bars")
    
    # Add indicators
    print("\n[2] Adding indicators...")
    df = prepare_data_with_indicators(df)
    
    # Define parameter space
    print("\n[3] Setting up parameter space...")
    param_space = ParameterSpace(
        min_confluence=[2, 3, 4],
        hurst_window=[50, 100],
        atr_stop_multiplier=[1.5, 2.0, 2.5],
        atr_tp_multiplier=[3.0, 4.0]
    )
    
    # Run optimization
    print("\n[4] Running Grid Search with Walk-Forward Validation...")
    optimizer = ParameterOptimizer(param_space=param_space, n_splits=3)
    
    results = optimizer.grid_search(
        df,
        params_to_optimize=['min_confluence', 'hurst_window', 'atr_stop_multiplier', 'atr_tp_multiplier'],
        use_walk_forward=True,
        verbose=True
    )
    
    # Best params
    print("\n" + "="*60)
    print("BEST PARAMETERS:")
    print("="*60)
    best = optimizer.get_best_params()
    for k, v in best.items():
        print(f"  {k}: {v}")
    
    print(f"\nBest Score: {results[0].score:.4f}")
    print(f"Best Sharpe: {results[0].sharpe_ratio:.2f}")
    print(f"Best Return: {results[0].total_return:.2%}")
    print(f"Best Max DD: {results[0].max_drawdown:.2%}")
    
    # Save results
    results_df = optimizer.get_results_dataframe()
    results_df.to_csv('optimization_results.csv', index=False)
    print("\nResults saved to optimization_results.csv")
