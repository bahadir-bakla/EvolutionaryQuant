"""
Alpha-Forge Trade Executor
MetaTrader 5 order execution with retry logic and error handling
"""

import MetaTrader5 as mt5
import time
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TradeExecutor:
    """MT5 trade execution with retry logic."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.slippage = self.config.get("slippage_points", 2)
        self.commission_pct = self.config.get("commission_pct", 0.0001)
        self.max_retries = self.config.get("max_retries", 3)
        self.retry_delay_ms = self.config.get("retry_delay_ms", 500)
        self.connected = False

    def connect(self) -> bool:
        if not mt5.initialize():
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False
        self.connected = True
        logger.info(f"MT5 Connected: {mt5.terminal_info().name}")
        return True

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False

    def execute(
        self,
        symbol: str,
        direction: str,
        volume: float,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        comment: str = "AlphaForge",
    ) -> Optional[Dict]:
        """Execute a market order with retry logic."""
        if not self.connected and not self.connect():
            return None

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol {symbol} not found")
            return None

        if not symbol_info.visible:
            logger.warning(f"Symbol {symbol} not visible, enabling")
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}")
                return None

        point = symbol_info.point
        digits = symbol_info.digits
        order_type = mt5.ORDER_TYPE_BUY if direction == "LONG" else mt5.ORDER_TYPE_SELL

        price = mt5.symbol_info_tick(symbol).ask if direction == "LONG" else mt5.symbol_info_tick(symbol).bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": round(stop_loss, digits) if stop_loss > 0 else 0.0,
            "tp": round(take_profit, digits) if take_profit > 0 else 0.0,
            "deviation": self.slippage,
            "magic": 234000,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        for attempt in range(self.max_retries):
            try:
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    logger.info(
                        f"Order executed: {direction} {symbol} "
                        f"vol={volume} @ {price:.5f} "
                        f"ticket={result.order}"
                    )
                    return {
                        "ticket": result.order,
                        "symbol": symbol,
                        "direction": direction,
                        "volume": volume,
                        "price": price,
                        "sl": stop_loss,
                        "tp": take_profit,
                        "time": datetime.now(),
                        "comment": comment,
                        "retcode": result.retcode,
                    }
                else:
                    logger.warning(
                        f"Order failed (attempt {attempt+1}): "
                        f"{result.retcode} - {result.comment}"
                    )
                    time.sleep(self.retry_delay_ms / 1000)
            except Exception as e:
                logger.error(f"Order error (attempt {attempt+1}): {e}")
                time.sleep(self.retry_delay_ms / 1000)

        logger.error(f"Order failed after {self.max_retries} attempts")
        return None

    def close_position(self, ticket: int) -> Optional[Dict]:
        """Close a position by ticket."""
        if not self.connected:
            return None

        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"Position {ticket} not found")
            return None

        pos = positions[0]
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(pos.symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(pos.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self.slippage,
            "magic": 234000,
            "comment": "AlphaForge Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Position {ticket} closed @ {price:.5f}")
            return {"ticket": ticket, "close_price": price, "retcode": result.retcode}
        else:
            logger.error(f"Close failed: {result.retcode} - {result.comment}")
            return None

    def get_open_positions(self) -> list:
        """Get all open positions."""
        if not self.connected:
            return []
        positions = mt5.positions_get()
        if positions is None:
            return []
        return [
            {
                "ticket": p.ticket,
                "symbol": p.symbol,
                "type": "LONG" if p.type == mt5.POSITION_TYPE_BUY else "SHORT",
                "volume": p.volume,
                "open_price": p.price_open,
                "sl": p.sl,
                "tp": p.tp,
                "profit": p.profit,
                "time": datetime.fromtimestamp(p.time),
            }
            for p in positions
        ]
