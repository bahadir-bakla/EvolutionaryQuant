# NQ Quant Bot - Main Entry Point
# OpenBB + QuantMuse + Custom Kalman/Hurst/OrderFlow

import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "QuantMuse"))

# Our modules
from nq_core.brain import QuantBrain, MarketRegime
from nq_core.order_blocks import detect_order_blocks, get_active_order_blocks
from nq_core.confluence import ConfluenceEngine, SignalType
from config.settings import DEFAULT_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_nq_data(symbol: str = "NQ=F", period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    NQ verisi çek - OpenBB veya yfinance kullan
    """
    try:
        # OpenBB dene
        from openbb import obb
        logger.info(f"Fetching {symbol} via OpenBB...")
        data = obb.equity.price.historical(
            symbol=symbol,
            provider="yfinance",
            interval=interval
        ).to_df()
        logger.info(f"OpenBB: {len(data)} bars fetched")
        return data
    except Exception as e:
        logger.warning(f"OpenBB failed: {e}, falling back to yfinance")
        
    try:
        # yfinance fallback
        import yfinance as yf
        logger.info(f"Fetching {symbol} via yfinance...")
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        data.columns = data.columns.str.lower()
        logger.info(f"yfinance: {len(data)} bars fetched")
        return data
    except Exception as e:
        logger.error(f"yfinance also failed: {e}")
        raise


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Simple RSI calculation"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range calculation"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def run_analysis(df: pd.DataFrame, config=DEFAULT_CONFIG) -> pd.DataFrame:
    """
    Full analysis pipeline
    """
    logger.info("Starting analysis pipeline...")
    
    # Initialize components
    brain = QuantBrain(
        hurst_window=config.hurst.window,
        trending_threshold=config.hurst.trending_threshold,
        choppy_threshold=config.hurst.choppy_threshold
    )
    
    confluence = ConfluenceEngine(min_confluence=config.min_confluence_score)
    
    # Calculate indicators
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'])
    df['atr'] = calculate_atr(df)
    
    # Detect Order Blocks
    order_blocks = detect_order_blocks(
        df,
        lookback=config.order_block.lookback,
        body_threshold=config.order_block.body_threshold,
        volume_threshold=config.order_block.volume_threshold
    )
    logger.info(f"Detected {len(order_blocks)} order blocks")
    
    # Process each bar
    results = []
    
    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']
        timestamp = df.index[i] if hasattr(df.index[i], 'strftime') else None
        
        # Update brain
        state = brain.update(price, timestamp)
        
        # Get active OBs
        active_obs = get_active_order_blocks(order_blocks, i, max_age=50)
        
        # Get confluence signal
        rsi = row['rsi'] if pd.notna(row['rsi']) else None
        atr = row['atr'] if pd.notna(row['atr']) else None
        
        signal = confluence.evaluate(state, active_obs, rsi=rsi, atr=atr)
        
        results.append({
            'timestamp': timestamp,
            'price': price,
            'kalman_price': state.kalman_price,
            'velocity': state.kalman_velocity,
            'hurst': state.hurst_value,
            'hurst_regime': state.hurst_regime,
            'z_score': state.z_score,
            'regime': state.regime.value,
            'rsi': rsi,
            'atr': atr,
            'signal': signal.direction,
            'signal_type': signal.signal_type.value,
            'confluence_score': signal.confluence_score,
            'active_factors': signal.active_factors,
            'confidence': signal.confidence,
            'entry': signal.entry_price if signal.direction != "NEUTRAL" else None,
            'stop_loss': signal.stop_loss if signal.direction != "NEUTRAL" else None,
            'tp1': signal.take_profit_1 if signal.direction != "NEUTRAL" else None,
        })
    
    results_df = pd.DataFrame(results)
    logger.info(f"Analysis complete. {len(results_df)} bars processed")
    
    return results_df


def plot_results(df: pd.DataFrame, results: pd.DataFrame):
    """Sonuçları görselleştir"""
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
    
    x = range(len(results))
    
    # 1. Price + Kalman + Signals
    ax1 = axes[0]
    ax1.plot(x, results['price'], label='Price', alpha=0.6)
    ax1.plot(x, results['kalman_price'], label='Kalman', linewidth=2, color='green')
    
    # Signal markers
    longs = results[results['signal'] == 'LONG']
    shorts = results[results['signal'] == 'SHORT']
    
    ax1.scatter(longs.index, longs['price'], marker='^', c='lime', s=100, label='LONG', zorder=5)
    ax1.scatter(shorts.index, shorts['price'], marker='v', c='red', s=100, label='SHORT', zorder=5)
    
    ax1.set_title('NQ Price + Kalman Filter + Signals', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Hurst Exponent
    ax2 = axes[1]
    ax2.plot(x, results['hurst'], color='purple', linewidth=1.5)
    ax2.axhline(0.5, color='black', linestyle='--', alpha=0.5)
    ax2.axhline(0.6, color='green', linestyle='--', alpha=0.3, label='Trend threshold')
    ax2.axhline(0.4, color='red', linestyle='--', alpha=0.3, label='Choppy threshold')
    ax2.fill_between(x, 0.6, 1.0, alpha=0.1, color='green')
    ax2.fill_between(x, 0.0, 0.4, alpha=0.1, color='red')
    ax2.set_title('Hurst Exponent (Market Regime)', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Z-Score
    ax3 = axes[2]
    ax3.plot(x, results['z_score'], color='orange', linewidth=1.5)
    ax3.axhline(2, color='red', linestyle='--', alpha=0.5)
    ax3.axhline(-2, color='green', linestyle='--', alpha=0.5)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax3.fill_between(x, 2, results['z_score'].max(), alpha=0.1, color='red')
    ax3.fill_between(x, results['z_score'].min(), -2, alpha=0.1, color='green')
    ax3.set_title('Z-Score (Deviation from Kalman)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 4. RSI
    ax4 = axes[3]
    ax4.plot(x, results['rsi'], color='blue', linewidth=1.5)
    ax4.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax4.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax4.axhline(50, color='black', linestyle='--', alpha=0.3)
    ax4.fill_between(x, 70, 100, alpha=0.1, color='red')
    ax4.fill_between(x, 0, 30, alpha=0.1, color='green')
    ax4.set_title('RSI (14)', fontsize=12)
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    
    # 5. Confluence Score
    ax5 = axes[4]
    colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in results['confluence_score']]
    ax5.bar(x, results['confluence_score'], color=colors, alpha=0.7)
    ax5.axhline(0.3, color='green', linestyle='--', alpha=0.3)
    ax5.axhline(-0.3, color='red', linestyle='--', alpha=0.3)
    ax5.set_title('Confluence Score', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nq_analysis.png', dpi=150, bbox_inches='tight')
    logger.info("Chart saved to nq_analysis.png")
    plt.show()


def print_summary(results: pd.DataFrame):
    """Analiz özeti yazdır"""
    print("\n" + "="*60)
    print("NQ QUANT BOT - ANALYSIS SUMMARY")
    print("="*60)
    
    # Son durum
    last = results.iloc[-1]
    print(f"\n[*] Current State:")
    print(f"   Price: {last['price']:.2f}")
    print(f"   Kalman: {last['kalman_price']:.2f}")
    print(f"   Hurst: {last['hurst']:.3f} ({last['hurst_regime']})")
    print(f"   RSI: {last['rsi']:.1f}")
    print(f"   Z-Score: {last['z_score']:.2f}")
    
    print(f"\n[>] Current Signal:")
    print(f"   Direction: {last['signal']}")
    print(f"   Type: {last['signal_type']}")
    print(f"   Confluence: {last['confluence_score']:.2f}")
    print(f"   Active Factors: {last['active_factors']}")
    print(f"   Confidence: {last['confidence']:.1%}")
    
    if last['entry']:
        print(f"\n[$] Trade Levels:")
        print(f"   Entry: {last['entry']:.2f}")
        print(f"   Stop Loss: {last['stop_loss']:.2f}")
        print(f"   Take Profit: {last['tp1']:.2f}")
    
    # Sinyal istatistikleri
    print(f"\n[+] Signal Statistics (Last {len(results)} bars):")
    signal_counts = results['signal'].value_counts()
    for sig, count in signal_counts.items():
        pct = count / len(results) * 100
        print(f"   {sig}: {count} ({pct:.1f}%)")
    
    # Rejim dağılımı
    print(f"\n[~] Regime Distribution:")
    regime_counts = results['hurst_regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(results) * 100
        print(f"   {regime}: {count} ({pct:.1f}%)")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='NQ Quant Trading Bot')
    parser.add_argument('--symbol', default='NQ=F', help='Trading symbol')
    parser.add_argument('--period', default='6mo', help='Data period')
    parser.add_argument('--interval', default='1d', help='Data interval')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    print("\n>>> NQ QUANT BOT - Starting...")
    print(f"   Symbol: {args.symbol}")
    print(f"   Period: {args.period}")
    print(f"   Interval: {args.interval}")
    
    # Fetch data
    df = fetch_nq_data(args.symbol, args.period, args.interval)
    
    # Run analysis
    results = run_analysis(df)
    
    # Print summary
    print_summary(results)
    
    # Plot
    if not args.no_plot:
        plot_results(df, results)
    
    # Save results
    results.to_csv('nq_results.csv', index=False)
    logger.info("Results saved to nq_results.csv")
    
    return results


if __name__ == "__main__":
    main()
