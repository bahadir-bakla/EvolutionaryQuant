"""
Alpha-Forge Walk-Forward Analysis & Monte Carlo Validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-Forward Analysis for strategy robustness testing."""

    def __init__(
        self,
        train_bars: int = 500,
        test_bars: int = 100,
        step_bars: int = 50,
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars

    def generate_windows(self, total_bars: int) -> List[Tuple[int, int, int, int]]:
        windows = []
        start = 0
        while start + self.train_bars + self.test_bars <= total_bars:
            windows.append(
                (start, start + self.train_bars, start + self.train_bars, start + self.train_bars + self.test_bars)
            )
            start += self.step_bars
        return windows

    def run(
        self,
        data: pd.DataFrame,
        optimize_func: Callable,
        backtest_func: Callable,
    ) -> Dict:
        windows = self.generate_windows(len(data))
        logger.info(f"Walk-Forward: {len(windows)} windows")

        results = []
        equity_curve = [1000.0]
        all_trades = []

        for i, (ts, te, tes, tee) in enumerate(windows):
            train_data = data.iloc[ts:te]
            test_data = data.iloc[tes:tee]

            config = optimize_func(train_data)
            balance, trades = backtest_func(test_data, config)

            wr = (
                sum(1 for t in trades if t["pnl"] > 0) / len(trades) * 100
                if trades
                else 0
            )

            results.append(
                {
                    "window": i + 1,
                    "balance": balance,
                    "return_pct": (balance - 1000) / 1000 * 100,
                    "trades": len(trades),
                    "win_rate": wr,
                }
            )
            equity_curve.append(balance)
            all_trades.extend(trades)

        total_return = (equity_curve[-1] - 1000) / 1000 * 100
        profitable = sum(1 for r in results if r["return_pct"] > 0)

        return {
            "total_windows": len(windows),
            "profitable_windows": profitable,
            "profitability_rate": profitable / len(windows) * 100,
            "total_return": total_return,
            "final_balance": equity_curve[-1],
            "total_trades": len(all_trades),
            "avg_return": np.mean([r["return_pct"] for r in results]),
            "std_return": np.std([r["return_pct"] for r in results]),
            "equity_curve": equity_curve,
            "window_results": results,
        }


class MonteCarloSimulator:
    """Monte Carlo simulation for strategy validation."""

    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations

    def run_permutation_test(self, trades: List[Dict]) -> Dict:
        pnls = np.array([t["pnl"] for t in trades])
        n = self.n_simulations

        final_balances = np.zeros(n)
        max_drawdowns = np.zeros(n)

        for i in range(n):
            shuffled = np.random.permutation(pnls)
            balance = 1000.0
            peak = balance
            max_dd = 0.0

            for pnl in shuffled:
                balance += pnl
                peak = max(peak, balance)
                dd = (peak - balance) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)

            final_balances[i] = balance
            max_drawdowns[i] = max_dd

        return {
            "n_simulations": n,
            "final_balance": {
                "mean": float(np.mean(final_balances)),
                "median": float(np.median(final_balances)),
                "std": float(np.std(final_balances)),
                "p5": float(np.percentile(final_balances, 5)),
                "p95": float(np.percentile(final_balances, 95)),
            },
            "max_drawdown": {
                "mean": float(np.mean(max_drawdowns)),
                "median": float(np.median(max_drawdowns)),
                "p95": float(np.percentile(max_drawdowns, 95)),
            },
            "probability_profit": float(np.mean(final_balances > 1000)),
            "probability_ruin": float(np.mean(final_balances < 500)),
        }

    def run_bootstrap_test(self, trades: List[Dict]) -> Dict:
        pnls = np.array([t["pnl"] for t in trades])
        n = self.n_simulations
        n_trades = len(pnls)

        total_returns = np.zeros(n)
        for i in range(n):
            sampled = np.random.choice(pnls, size=n_trades, replace=True)
            total_returns[i] = np.sum(sampled) / 1000 * 100

        return {
            "n_simulations": n,
            "return": {
                "mean": float(np.mean(total_returns)),
                "median": float(np.median(total_returns)),
                "std": float(np.std(total_returns)),
                "p5": float(np.percentile(total_returns, 5)),
                "p95": float(np.percentile(total_returns, 95)),
            },
            "probability_positive": float(np.mean(total_returns > 0)),
        }
