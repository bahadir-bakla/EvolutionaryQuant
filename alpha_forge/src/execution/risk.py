"""
Alpha-Forge Risk Manager
Position sizing, drawdown limits, daily loss controls
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RiskManager:
    """Production-grade risk management."""

    def __init__(self, config: Dict):
        self.max_daily_loss_pct = config.get("max_daily_loss_pct", 0.05)
        self.max_drawdown_pct = config.get("max_drawdown_pct", 0.15)
        self.max_position_size_pct = config.get("max_position_size_pct", 0.10)
        self.risk_per_trade_pct = config.get("risk_per_trade_pct", 0.03)
        self.daily_pnl = 0.0
        self.peak_balance = 10000.0
        self.current_balance = 10000.0
        self.daily_trades = 0
        self.max_daily_trades = config.get("max_daily_trades", 20)

    def check_risk(self, instrument: str, signal: Dict) -> bool:
        if self.daily_pnl < -self.max_daily_loss_pct * self.current_balance:
            logger.warning(f"Daily loss limit: {self.daily_pnl:.2f}")
            return False

        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        if drawdown > self.max_drawdown_pct:
            logger.warning(f"Max drawdown: {drawdown:.2%}")
            return False

        if self.daily_trades >= self.max_daily_trades:
            logger.warning(f"Daily trade limit: {self.daily_trades}")
            return False

        return True

    def update_balance(self, pnl: float):
        self.current_balance += pnl
        self.peak_balance = max(self.peak_balance, self.current_balance)
        self.daily_pnl += pnl
        self.daily_trades += 1

    def reset_daily(self):
        self.daily_pnl = 0.0
        self.daily_trades = 0
        logger.info("Daily P&L reset")


class PositionSizer:
    """Dynamic position sizing: risk-based, fixed, or Kelly."""

    def __init__(self, config: Dict):
        self.method = config.get("method", "risk_based")
        self.min_lot = config.get("min_lot", 0.01)
        self.max_lot = config.get("max_lot", 5.0)
        self.risk_pct = config.get("risk_per_trade_pct", 0.03)
        self.kelly_fraction = config.get("kelly_fraction", 0.25)

    def calculate(
        self,
        balance: float,
        atr: float,
        stop_atr_mult: float,
        win_rate: float = 0.5,
        avg_win_loss_ratio: float = 1.5,
    ) -> float:
        if self.method == "risk_based":
            return self._risk_based(balance, atr, stop_atr_mult)
        elif self.method == "kelly":
            return self._kelly(balance, win_rate, avg_win_loss_ratio, atr, stop_atr_mult)
        else:
            return self._fixed(balance)

    def _risk_based(self, balance: float, atr: float, stop_mult: float) -> float:
        risk_amount = balance * self.risk_pct
        stop_distance = atr * stop_mult
        if stop_distance == 0:
            return self.min_lot
        lot = risk_amount / (stop_distance * 20.0)
        return max(self.min_lot, min(self.max_lot, lot))

    def _kelly(
        self,
        balance: float,
        win_rate: float,
        avg_win_loss_ratio: float,
        atr: float,
        stop_mult: float,
    ) -> float:
        kelly_pct = win_rate - (1 - win_rate) / avg_win_loss_ratio
        kelly_pct = max(0, kelly_pct) * self.kelly_fraction
        risk_amount = balance * kelly_pct
        stop_distance = atr * stop_mult
        if stop_distance == 0:
            return self.min_lot
        lot = risk_amount / (stop_distance * 20.0)
        return max(self.min_lot, min(self.max_lot, lot))

    def _fixed(self, balance: float) -> float:
        return max(self.min_lot, min(self.max_lot, balance / 10000 * 0.1))
