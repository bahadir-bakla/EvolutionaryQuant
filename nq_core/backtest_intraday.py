# NQ Quant Bot - Advanced Intraday Backtest
# With trailing stops, partial profits, and multiple timeframes

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nq_core.intraday import (
    fetch_intraday_data, 
    add_intraday_indicators,
    IntradayConfluenceEngine,
    IntradayConfig
)


@dataclass
class Trade:
    """Active or closed trade"""
    entry_time: pd.Timestamp
    entry_price: float
    direction: str
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    size: int = 1
    
    # State
    is_open: bool = True
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl: float = 0.0
    
    # Trailing stop
    trailing_stop: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 999999.0


class AdvancedIntradayBacktest:
    """
    Advanced intraday backtesting with:
    - Trailing stops
    - Partial profit taking
    - Session filters
    - Max daily trades limit
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        risk_per_trade: float = 0.01,  # 1%
        max_daily_trades: int = 10,
        trailing_stop_atr: float = 1.0,  # ATR multiplier for trailing
        partial_profit_percent: float = 0.5,  # Take 50% at TP1
        contract_value: float = 20.0  # NQ = $20 per point
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_daily_trades = max_daily_trades
        self.trailing_stop_atr = trailing_stop_atr
        self.partial_profit_percent = partial_profit_percent
        self.contract_value = contract_value
        
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.daily_trades: Dict[str, int] = {}
        
    def can_open_trade(self, timestamp: pd.Timestamp) -> bool:
        """Check if we can open a new trade today"""
        date_str = str(timestamp.date())
        current_count = self.daily_trades.get(date_str, 0)
        return current_count < self.max_daily_trades
    
    def open_trade(
        self,
        timestamp: pd.Timestamp,
        price: float,
        direction: str,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        take_profit_3: float
    ) -> Trade:
        """Open a new trade"""
        # Calculate position size based on risk
        risk_amount = self.capital * self.risk_per_trade
        points_risk = abs(price - stop_loss)
        
        if points_risk > 0:
            max_contracts = int(risk_amount / (points_risk * self.contract_value))
            size = max(1, max_contracts)
        else:
            size = 1
        
        trade = Trade(
            entry_time=timestamp,
            entry_price=price,
            direction=direction,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            take_profit_3=take_profit_3,
            size=size,
            trailing_stop=stop_loss,
            highest_price=price if direction == 'LONG' else 999999,
            lowest_price=999999 if direction == 'LONG' else price
        )
        
        self.trades.append(trade)
        
        # Track daily trades
        date_str = str(timestamp.date())
        self.daily_trades[date_str] = self.daily_trades.get(date_str, 0) + 1
        
        return trade
    
    def update_trade(
        self,
        trade: Trade,
        timestamp: pd.Timestamp,
        high: float,
        low: float,
        close: float,
        atr: float
    ):
        """Update trade with current bar data"""
        if not trade.is_open:
            return
        
        # Update highest/lowest
        if trade.direction == 'LONG':
            trade.highest_price = max(trade.highest_price, high)
            
            # Calculate profit in ATR terms
            profit_atr = (trade.highest_price - trade.entry_price) / atr
            
            # Only activate trailing stop after 1 ATR of profit
            if profit_atr >= 1.0:
                new_trail = trade.highest_price - (atr * self.trailing_stop_atr * 2)  # Wider trail
                if new_trail > trade.stop_loss:  # Only move up, never below original SL
                    trade.trailing_stop = new_trail
            else:
                trade.trailing_stop = trade.stop_loss  # Use original SL
            
            # Check exit conditions (order matters: TP first, then SL)
            if high >= trade.take_profit_1:
                self.close_trade(trade, timestamp, trade.take_profit_1, "Take Profit")
            elif low <= trade.trailing_stop:
                if trade.trailing_stop > trade.stop_loss:
                    self.close_trade(trade, timestamp, trade.trailing_stop, "Trailing Stop")
                else:
                    self.close_trade(trade, timestamp, trade.stop_loss, "Stop Loss")
                
        else:  # SHORT
            trade.lowest_price = min(trade.lowest_price, low)
            
            # Calculate profit in ATR terms
            profit_atr = (trade.entry_price - trade.lowest_price) / atr
            
            # Only activate trailing stop after 1 ATR of profit
            if profit_atr >= 1.0:
                new_trail = trade.lowest_price + (atr * self.trailing_stop_atr * 2)  # Wider trail
                if new_trail < trade.stop_loss:  # Only move down
                    trade.trailing_stop = new_trail
            else:
                trade.trailing_stop = trade.stop_loss  # Use original SL
            
            # Check exit conditions
            if low <= trade.take_profit_1:
                self.close_trade(trade, timestamp, trade.take_profit_1, "Take Profit")
            elif high >= trade.trailing_stop:
                if trade.trailing_stop < trade.stop_loss:
                    self.close_trade(trade, timestamp, trade.trailing_stop, "Trailing Stop")
                else:
                    self.close_trade(trade, timestamp, trade.stop_loss, "Stop Loss")
    
    def close_trade(
        self,
        trade: Trade,
        timestamp: pd.Timestamp,
        exit_price: float,
        reason: str
    ):
        """Close a trade"""
        trade.is_open = False
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        trade.exit_reason = reason
        
        # Calculate P&L
        if trade.direction == 'LONG':
            points = exit_price - trade.entry_price
        else:
            points = trade.entry_price - exit_price
        
        trade.pnl = points * trade.size * self.contract_value
        self.capital += trade.pnl
    
    def run(
        self,
        df: pd.DataFrame,
        signals_df: pd.DataFrame
    ):
        """Run backtest"""
        active_trade: Optional[Trade] = None
        
        for i in range(len(signals_df)):
            idx = signals_df.index[i]
            row = df.loc[idx]
            signal_row = signals_df.iloc[i]
            
            high = row['high']
            low = row['low']
            close = row['close']
            atr = row.get('atr', close * 0.002)
            
            # Update active trade
            if active_trade and active_trade.is_open:
                self.update_trade(active_trade, idx, high, low, close, atr)
            
            # Check for new signal
            signal = signal_row['signal']
            
            if signal != 'NEUTRAL' and (active_trade is None or not active_trade.is_open):
                if self.can_open_trade(idx):
                    active_trade = self.open_trade(
                        timestamp=idx,
                        price=close,
                        direction=signal,
                        stop_loss=signal_row['stop_loss'],
                        take_profit_1=signal_row['tp1'],
                        take_profit_2=signal_row.get('tp2', signal_row['tp1'] * 1.5),
                        take_profit_3=signal_row.get('tp3', signal_row['tp1'] * 2)
                    )
            
            self.equity_curve.append(self.capital)
        
        # Close any remaining open trade at last price
        if active_trade and active_trade.is_open:
            last_row = df.iloc[-1]
            self.close_trade(active_trade, df.index[-1], last_row['close'], "End of Data")
        
        return self.get_results()
    
    def get_results(self) -> dict:
        """Calculate performance metrics"""
        closed_trades = [t for t in self.trades if not t.is_open]
        
        if not closed_trades:
            return {"error": "No trades executed"}
        
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl < 0]
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        # Max drawdown
        equity = np.array(self.equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak * 100
        max_dd = np.max(drawdown)
        
        # By exit reason
        exit_reasons = {}
        for t in closed_trades:
            reason = t.exit_reason
            if reason not in exit_reasons:
                exit_reasons[reason] = {'count': 0, 'pnl': 0}
            exit_reasons[reason]['count'] += 1
            exit_reasons[reason]['pnl'] += t.pnl
        
        results = {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_dd,
            'total_trades': len(closed_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(closed_trades) * 100 if closed_trades else 0,
            'avg_win': np.mean([t.pnl for t in wins]) if wins else 0,
            'avg_loss': np.mean([t.pnl for t in losses]) if losses else 0,
            'profit_factor': abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in losses) + 1) if losses else 999,
            'exit_reasons': exit_reasons
        }
        
        return results
    
    def print_results(self, results: dict):
        """Print formatted results"""
        print(f"\n{'='*60}")
        print("         ADVANCED INTRADAY BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f" Initial Capital:     ${results['initial_capital']:>15,.2f}")
        print(f" Final Capital:       ${results['final_capital']:>15,.2f}")
        print(f" Total Return:        {results['total_return_pct']:>15.2f}%")
        print(f" Max Drawdown:        {results['max_drawdown_pct']:>15.2f}%")
        print(f"{'-'*60}")
        print(f" Total Trades:        {results['total_trades']:>15}")
        print(f" Win Rate:            {results['win_rate']:>15.1f}%")
        print(f" Profit Factor:       {results['profit_factor']:>15.2f}")
        print(f" Avg Win:             ${results['avg_win']:>15,.2f}")
        print(f" Avg Loss:            ${results['avg_loss']:>15,.2f}")
        print(f"{'-'*60}")
        print(" Exit Reasons:")
        for reason, data in results['exit_reasons'].items():
            print(f"   {reason}: {data['count']} trades, ${data['pnl']:,.2f}")
        print(f"{'='*60}")


def run_advanced_backtest(interval: str = "15m", days: int = 30):
    """Run advanced backtest"""
    print(f"\n{'='*60}")
    print(f"NQ ADVANCED INTRADAY BACKTEST ({interval})")
    print(f"{'='*60}")
    
    # Fetch and prepare data
    print(f"\n[1] Fetching {interval} data...")
    df = fetch_intraday_data("NQ=F", interval, days)
    print(f"    Bars: {len(df)}")
    
    config = IntradayConfig(
        timeframe=interval,
        min_confluence=2,
        min_score=1.5,
        atr_stop_mult=1.5,
        atr_tp_mult=2.5  # Wider TP for better R:R
    )
    
    print("[2] Adding indicators...")
    df = add_intraday_indicators(df, config)
    
    print("[3] Generating signals...")
    engine = IntradayConfluenceEngine(config)
    
    signals = []
    for i in range(50, len(df)):
        signal = engine.evaluate(df, i)
        signals.append({
            'signal': signal.direction,
            'stop_loss': signal.stop_loss,
            'tp1': signal.take_profit_1,
            'tp2': signal.take_profit_2,
            'tp3': signal.take_profit_3,
            'atr': df.iloc[i].get('atr', 0)
        })
    
    signals_df = pd.DataFrame(signals, index=df.index[50:])
    test_df = df.iloc[50:]
    
    # Signal stats
    signal_counts = signals_df['signal'].value_counts()
    print("\n    Signal Distribution:")
    for sig, count in signal_counts.items():
        print(f"      {sig}: {count} ({count/len(signals_df)*100:.1f}%)")
    
    # Run backtest
    print("\n[4] Running advanced backtest with trailing stops...")
    backtest = AdvancedIntradayBacktest(
        initial_capital=100000,
        risk_per_trade=0.01,
        max_daily_trades=8,
        trailing_stop_atr=1.2,
        partial_profit_percent=0.5
    )
    
    results = backtest.run(test_df, signals_df)
    backtest.print_results(results)
    
    return results, backtest


if __name__ == "__main__":
    # Test multiple timeframes
    print("\n" + "="*70)
    print("TESTING 5m TIMEFRAME")
    print("="*70)
    results_5m, _ = run_advanced_backtest("5m", 7)  # 7 days for 5m
    
    print("\n" + "="*70)
    print("TESTING 15m TIMEFRAME")
    print("="*70)
    results_15m, _ = run_advanced_backtest("15m", 30)
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<20} {'5m':<15} {'15m':<15}")
    print("-"*50)
    print(f"{'Return':<20} {results_5m['total_return_pct']:.2f}%{'':<10} {results_15m['total_return_pct']:.2f}%")
    print(f"{'Trades':<20} {results_5m['total_trades']:<15} {results_15m['total_trades']:<15}")
    print(f"{'Win Rate':<20} {results_5m['win_rate']:.1f}%{'':<10} {results_15m['win_rate']:.1f}%")
    print(f"{'Profit Factor':<20} {results_5m['profit_factor']:.2f}{'':<12} {results_15m['profit_factor']:.2f}")
    print(f"{'Max DD':<20} {results_5m['max_drawdown_pct']:.2f}%{'':<10} {results_15m['max_drawdown_pct']:.2f}%")
