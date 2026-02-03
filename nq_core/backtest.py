# NQ Quant Bot - Backtest Engine
# QuantMuse-inspired with Kelly Criterion & ATR Stop Loss

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """Trade record"""
    entry_time: datetime
    exit_time: Optional[datetime]
    direction: TradeDirection
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    stop_loss: float
    take_profit: float
    pnl: float = 0.0
    pnl_percent: float = 0.0
    status: str = "OPEN"  # OPEN, CLOSED, STOPPED
    
    def close(self, exit_price: float, exit_time: datetime, reason: str = "SIGNAL"):
        """Close the trade"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        
        if self.direction == TradeDirection.LONG:
            self.pnl = (exit_price - self.entry_price) * self.quantity
            self.pnl_percent = (exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl = (self.entry_price - exit_price) * self.quantity
            self.pnl_percent = (self.entry_price - exit_price) / self.entry_price
            
        self.status = reason


@dataclass
class BacktestResult:
    """Backtest sonuçları"""
    # Capital
    initial_capital: float
    final_capital: float
    
    # Returns
    total_return: float
    annualized_return: float
    
    # Risk Metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    
    # Trade Stats
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Data
    equity_curve: pd.DataFrame
    trades: List[Trade]
    
    def __str__(self):
        return f"""
============================================================
              BACKTEST RESULTS                            
============================================================
 Initial Capital:     ${self.initial_capital:>15,.2f}      
 Final Capital:       ${self.final_capital:>15,.2f}      
 Total Return:        {self.total_return:>15.2%}      
 Annualized Return:   {self.annualized_return:>15.2%}      
------------------------------------------------------------
 Sharpe Ratio:        {self.sharpe_ratio:>15.2f}      
 Sortino Ratio:       {self.sortino_ratio:>15.2f}      
 Max Drawdown:        {self.max_drawdown:>15.2%}      
------------------------------------------------------------
 Total Trades:        {self.total_trades:>15d}      
 Win Rate:            {self.win_rate:>15.2%}      
 Profit Factor:       {self.profit_factor:>15.2f}      
 Avg Win:             ${self.avg_win:>15,.2f}      
 Avg Loss:            ${self.avg_loss:>15,.2f}      
============================================================
"""


class KellyCriterion:
    """
    Kelly Criterion Position Sizing
    
    f* = (W/A) - ((1-W)/B)
    
    where:
    - W = Win rate
    - A = Average loss ratio
    - B = Average win ratio
    
    Half-Kelly kullanıyoruz (daha konservatif)
    """
    
    def __init__(
        self,
        win_rate: float = 0.5,
        avg_win_ratio: float = 0.02,
        avg_loss_ratio: float = 0.01,
        kelly_fraction: float = 0.5,  # Half-Kelly
        max_position_size: float = 0.2  # Max %20 of capital per trade
    ):
        self.win_rate = win_rate
        self.avg_win_ratio = avg_win_ratio
        self.avg_loss_ratio = avg_loss_ratio
        self.kelly_fraction = kelly_fraction
        self.max_position_size = max_position_size
        
        self._trade_history: List[Tuple[float, bool]] = []  # (pnl_ratio, is_win)
        
    def update_from_trade(self, pnl_ratio: float, is_win: bool):
        """Update stats from closed trade"""
        self._trade_history.append((pnl_ratio, is_win))
        
        if len(self._trade_history) >= 10:
            wins = [t for t in self._trade_history if t[1]]
            losses = [t for t in self._trade_history if not t[1]]
            
            if wins and losses:
                self.win_rate = len(wins) / len(self._trade_history)
                self.avg_win_ratio = np.mean([t[0] for t in wins])
                self.avg_loss_ratio = abs(np.mean([t[0] for t in losses]))
    
    def calculate_kelly(self) -> float:
        """Calculate Kelly fraction"""
        if self.avg_loss_ratio == 0:
            return 0.0
            
        # Kelly formula
        f = (self.win_rate / self.avg_loss_ratio) - ((1 - self.win_rate) / self.avg_win_ratio)
        
        # Apply Kelly fraction (Half-Kelly)
        f *= self.kelly_fraction
        
        # Clamp to max position size
        f = max(0.0, min(f, self.max_position_size))
        
        return f
    
    def get_position_size(self, capital: float, price: float) -> float:
        """Get position size in units"""
        kelly_fraction = self.calculate_kelly()
        position_value = capital * kelly_fraction
        quantity = position_value / price
        return quantity


class NQBacktestEngine:
    """
    NQ-specific Backtest Engine
    
    Features:
    - Kelly Criterion position sizing
    - ATR-based dynamic stop-loss
    - Long/Short support
    - Detailed performance metrics
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_per_contract: float = 2.0,  # NQ futures commission
        slippage_ticks: int = 1,
        tick_value: float = 5.0,  # NQ tick value = $5
        use_kelly: bool = True
    ):
        self.initial_capital = initial_capital
        self.commission = commission_per_contract
        self.slippage_ticks = slippage_ticks
        self.tick_value = tick_value
        self.use_kelly = use_kelly
        
        self.capital = initial_capital
        self.trades: List[Trade] = []
        self.open_trade: Optional[Trade] = None
        self.equity_curve: List[Dict] = []
        
        self.kelly = KellyCriterion() if use_kelly else None
        
    def reset(self):
        """Reset engine"""
        self.capital = self.initial_capital
        self.trades = []
        self.open_trade = None
        self.equity_curve = []
        if self.kelly:
            self.kelly = KellyCriterion()
    
    def run(
        self,
        data: pd.DataFrame,
        signals: pd.DataFrame
    ) -> BacktestResult:
        """
        Run backtest
        
        Args:
            data: OHLCV DataFrame
            signals: Signal DataFrame with columns:
                - signal: LONG, SHORT, NEUTRAL
                - stop_loss: Stop price
                - tp1: Take profit price
                - atr: ATR value
        """
        self.reset()
        
        # Merge data - handle overlapping columns
        df = data.copy()
        signal_cols = ['signal', 'stop_loss', 'tp1']
        # Only add atr from signals if not in data
        if 'atr' not in df.columns:
            signal_cols.append('atr')
        df = df.join(signals[signal_cols], how='left', rsuffix='_sig')
        
        logger.info(f"Starting backtest with {len(df)} bars")
        
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]
            price = row['close']
            signal = row.get('signal', 'NEUTRAL')
            stop_loss = row.get('stop_loss', 0)
            take_profit = row.get('tp1', 0)
            atr = row.get('atr', 0)
            
            # Check open trade for stop/TP
            if self.open_trade:
                self._check_exit(row, timestamp)
            
            # Check for new signal
            if not self.open_trade and signal in ['LONG', 'SHORT']:
                self._open_trade(
                    timestamp=timestamp,
                    direction=TradeDirection.LONG if signal == 'LONG' else TradeDirection.SHORT,
                    price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    atr=atr
                )
            
            # Update equity curve
            self._update_equity(timestamp, price)
        
        # Close any remaining open trade
        if self.open_trade:
            last_row = df.iloc[-1]
            self._close_trade(last_row['close'], df.index[-1], "END")
        
        # Calculate results
        return self._calculate_results()
    
    def _open_trade(
        self,
        timestamp: datetime,
        direction: TradeDirection,
        price: float,
        stop_loss: float,
        take_profit: float,
        atr: float
    ):
        """Open a new trade"""
        # Apply slippage
        slippage = self.slippage_ticks * self.tick_value / 100  # Convert to price
        if direction == TradeDirection.LONG:
            entry_price = price + slippage
        else:
            entry_price = price - slippage
        
        # Calculate position size
        if self.kelly:
            quantity = self.kelly.get_position_size(self.capital, entry_price)
            quantity = max(1, int(quantity))  # Minimum 1 contract
        else:
            # Fixed 1% risk
            risk_per_trade = self.capital * 0.01
            stop_distance = abs(entry_price - stop_loss) if stop_loss else atr * 2
            quantity = risk_per_trade / (stop_distance + 1e-10)
            quantity = max(1, int(quantity))
        
        # Check capital
        required_margin = entry_price * quantity * 0.1  # 10% margin
        if required_margin > self.capital:
            quantity = int(self.capital * 0.1 / entry_price)
            if quantity < 1:
                return
        
        # Use ATR for stop if not provided
        if not stop_loss or stop_loss == 0:
            if direction == TradeDirection.LONG:
                stop_loss = entry_price - (atr * 2 if atr else entry_price * 0.02)
            else:
                stop_loss = entry_price + (atr * 2 if atr else entry_price * 0.02)
        
        if not take_profit or take_profit == 0:
            if direction == TradeDirection.LONG:
                take_profit = entry_price + (atr * 4 if atr else entry_price * 0.04)
            else:
                take_profit = entry_price - (atr * 4 if atr else entry_price * 0.04)
        
        self.open_trade = Trade(
            entry_time=timestamp,
            exit_time=None,
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        # Deduct commission
        self.capital -= self.commission * quantity
        
        logger.debug(f"Opened {direction.value} @ {entry_price:.2f}, qty={quantity}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
    
    def _check_exit(self, row: pd.Series, timestamp: datetime):
        """Check if trade should be closed"""
        trade = self.open_trade
        high = row['high']
        low = row['low']
        close = row['close']
        
        if trade.direction == TradeDirection.LONG:
            # Check stop loss
            if low <= trade.stop_loss:
                self._close_trade(trade.stop_loss, timestamp, "STOPPED")
                return
            # Check take profit
            if high >= trade.take_profit:
                self._close_trade(trade.take_profit, timestamp, "TP_HIT")
                return
        else:  # SHORT
            # Check stop loss
            if high >= trade.stop_loss:
                self._close_trade(trade.stop_loss, timestamp, "STOPPED")
                return
            # Check take profit
            if low <= trade.take_profit:
                self._close_trade(trade.take_profit, timestamp, "TP_HIT")
                return
        
        # Check for signal reversal
        signal = row.get('signal', 'NEUTRAL')
        if (trade.direction == TradeDirection.LONG and signal == 'SHORT') or \
           (trade.direction == TradeDirection.SHORT and signal == 'LONG'):
            self._close_trade(close, timestamp, "REVERSAL")
    
    def _close_trade(self, exit_price: float, exit_time: datetime, reason: str):
        """Close the open trade"""
        trade = self.open_trade
        if not trade:
            return
        
        # Apply slippage
        slippage = self.slippage_ticks * self.tick_value / 100
        if trade.direction == TradeDirection.LONG:
            exit_price -= slippage
        else:
            exit_price += slippage
        
        # Close trade
        trade.close(exit_price, exit_time, reason)
        
        # Update capital
        self.capital += trade.pnl
        self.capital -= self.commission * trade.quantity  # Exit commission
        
        # Update Kelly
        if self.kelly:
            is_win = trade.pnl > 0
            pnl_ratio = trade.pnl_percent
            self.kelly.update_from_trade(pnl_ratio, is_win)
        
        self.trades.append(trade)
        self.open_trade = None
        
        logger.debug(f"Closed {trade.direction.value} @ {exit_price:.2f}, PnL=${trade.pnl:.2f} ({reason})")
    
    def _update_equity(self, timestamp: datetime, current_price: float):
        """Update equity curve"""
        equity = self.capital
        
        # Add unrealized PnL
        if self.open_trade:
            trade = self.open_trade
            if trade.direction == TradeDirection.LONG:
                unrealized = (current_price - trade.entry_price) * trade.quantity
            else:
                unrealized = (trade.entry_price - current_price) * trade.quantity
            equity += unrealized
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': self.capital,
            'open_pnl': equity - self.capital
        })
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate final results"""
        if not self.equity_curve:
            raise ValueError("No equity curve data")
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        final_capital = equity_df['equity'].iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # Annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0.0
        
        # Risk metrics
        returns = equity_df['returns'].dropna()
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0  # Risk-free = 2%
        
        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - 0.02) / downside_std if downside_std > 0 else 0
        
        # Max Drawdown
        peak = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        in_drawdown = (drawdown < 0).astype(int)
        drawdown_periods = in_drawdown.groupby((in_drawdown != in_drawdown.shift()).cumsum())
        max_dd_duration = drawdown_periods.sum().max() if len(drawdown_periods) > 0 else 0
        
        # Trade statistics
        closed_trades = [t for t in self.trades if t.status != "OPEN"]
        total_trades = len(closed_trades)
        
        if total_trades > 0:
            winning_trades = len([t for t in closed_trades if t.pnl > 0])
            losing_trades = len([t for t in closed_trades if t.pnl <= 0])
            win_rate = winning_trades / total_trades
            
            wins = [t.pnl for t in closed_trades if t.pnl > 0]
            losses = [t.pnl for t in closed_trades if t.pnl <= 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            
            total_wins = sum(wins)
            total_losses = abs(sum(losses)) if losses else 1
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=int(max_dd_duration),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            equity_curve=equity_df,
            trades=self.trades
        )


# === TEST ===
if __name__ == "__main__":
    import yfinance as yf
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from nq_core.brain import QuantBrain
    from nq_core.confluence import ConfluenceEngine
    from nq_core.order_blocks import detect_order_blocks, get_active_order_blocks
    
    # Fetch data
    print("Fetching NQ=F data...")
    ticker = yf.Ticker("NQ=F")
    df = ticker.history(period="6mo", interval="1d")
    df.columns = df.columns.str.lower()
    print(f"Fetched {len(df)} bars")
    
    # Calculate indicators
    def calc_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def calc_atr(df, period=14):
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    df['rsi'] = calc_rsi(df['close'])
    df['atr'] = calc_atr(df)
    
    # Generate signals
    print("Generating signals...")
    brain = QuantBrain()
    confluence = ConfluenceEngine(min_confluence=3)
    obs = detect_order_blocks(df)
    
    signals = []
    for i in range(len(df)):
        row = df.iloc[i]
        state = brain.update(row['close'], df.index[i])
        active_obs = get_active_order_blocks(obs, i, max_age=50)
        signal = confluence.evaluate(state, active_obs, rsi=row['rsi'], atr=row['atr'])
        signals.append({
            'signal': signal.direction,
            'stop_loss': signal.stop_loss,
            'tp1': signal.take_profit_1,
            'atr': row['atr']
        })
    
    signals_df = pd.DataFrame(signals, index=df.index)
    
    # Run backtest
    print("Running backtest...")
    engine = NQBacktestEngine(initial_capital=100000, use_kelly=True)
    result = engine.run(df, signals_df)
    
    print(result)
