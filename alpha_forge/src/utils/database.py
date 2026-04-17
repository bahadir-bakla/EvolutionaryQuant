"""
Alpha-Forge Database
Trade logging and analytics with SQLite
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TradeLogger:
    """Persistent trade logging with SQLite."""

    def __init__(self, db_path: str = "data/alphaforgedb.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    volume REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    pnl REAL,
                    balance REAL,
                    reason TEXT,
                    strategy TEXT,
                    duration_seconds REAL,
                    exit_time TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    strength REAL,
                    features TEXT,
                    executed INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    trades INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    total_pnl REAL,
                    max_drawdown REAL,
                    end_balance REAL
                )
            """)
            conn.commit()

    def log_trade(self, trade: Dict):
        """Log a completed trade."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades (
                    timestamp, symbol, direction, volume,
                    entry_price, exit_price, stop_loss, take_profit,
                    pnl, balance, reason, strategy, duration_seconds,
                    exit_time, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.get("time", datetime.now().isoformat()),
                trade.get("symbol", "NQ"),
                trade.get("dir", trade.get("direction", "LONG")),
                trade.get("volume", 0.01),
                trade.get("entry", 0.0),
                trade.get("exit", 0.0),
                trade.get("sl", 0.0),
                trade.get("tp", 0.0),
                trade.get("pnl", 0.0),
                trade.get("bal", 0.0),
                trade.get("reason", ""),
                trade.get("strategy", "AlphaForge"),
                trade.get("duration", 0.0),
                trade.get("exit_time"),
                json.dumps(trade.get("metadata", {})),
            ))
            conn.commit()

    def log_signal(self, signal: Dict):
        """Log a trading signal."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO signals (timestamp, symbol, signal_type, strength, features)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                signal.get("symbol", "NQ"),
                signal.get("type", ""),
                signal.get("strength", 0.0),
                json.dumps(signal.get("features", {})),
            ))
            conn.commit()

    def get_trades(self, limit: int = 100) -> List[Dict]:
        """Get recent trades."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM trades ORDER BY id DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_daily_stats(self) -> List[Dict]:
        """Get daily performance stats."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT DATE(timestamp) as date,
                       COUNT(*) as trades,
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                       SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                       SUM(pnl) as total_pnl,
                       MAX(balance) as peak_balance,
                       MIN(balance) as min_balance
                FROM trades
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_performance_summary(self) -> Dict:
        """Get overall performance summary."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    MAX(pnl) as max_win,
                    MIN(pnl) as max_loss
                FROM trades
            """)
            row = cursor.fetchone()
            if row is None or row[0] == 0:
                return {}

            total, wins, total_pnl, avg_pnl, max_win, max_loss = row
            win_rate = wins / total * 100 if total > 0 else 0

            return {
                "total_trades": total,
                "win_rate": round(win_rate, 1),
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(avg_pnl, 2),
                "max_win": round(max_win, 2),
                "max_loss": round(max_loss, 2),
            }
