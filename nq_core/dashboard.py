# NQ Quant Bot - Real-Time Dashboard
# Streamlit dashboard for NQ and Gold signals

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nq_core.optimized_strategy import OptimizedStrategy, add_optimized_indicators
from nq_core.gold_strategy import GoldStrategy, GoldConfig
import yfinance as yf

st.set_page_config(
    page_title="NQ Quant Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === STYLES ===
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    .bullish { color: #00ff00; font-weight: bold; }
    .bearish { color: #ff0000; font-weight: bold; }
    .neutral { color: #aaaaaa; }
    
    .big-signal {
        font-size: 24px;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 10px;
    }
    .signal-long { background-color: rgba(0, 255, 0, 0.2); color: #00ff00; border: 1px solid #00ff00; }
    .signal-short { background-color: rgba(255, 0, 0, 0.2); color: #ff0000; border: 1px solid #ff0000; }
    .signal-neutral { background-color: rgba(100, 100, 100, 0.2); color: #aaaaaa; border: 1px solid #aaaaaa; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def fetch_data(symbol: str, interval: str = "5m", period: str = "5d"):
    """Fetch latest data"""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    df.columns = df.columns.str.lower()
    return df

def process_market(symbol: str, strategy, interval: str):
    """Process market data and get signal"""
    with st.spinner(f"Analyzing {symbol}..."):
        df = fetch_data(symbol, interval)
        if len(df) < 50:
            st.error(f"Not enough data for {symbol}")
            return None, None
        
        # Add indicators
        df = add_optimized_indicators(df)
        
        # Get latest signal
        latest_idx = len(df) - 1
        signal = strategy.evaluate(df, latest_idx)
        
        return df, signal

def draw_chart(df: pd.DataFrame, signal, symbol: str):
    """Draw candle chart with indicators"""
    # Last 50 bars
    plot_df = df.iloc[-50:]
    
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['open'],
        high=plot_df['high'],
        low=plot_df['low'],
        close=plot_df['close'],
        name='Price'
    ))
    
    # EMAs
    if 'ema_8' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_8'], line=dict(color='cyan', width=1), name='EMA 8'))
    if 'ema_21' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ema_21'], line=dict(color='orange', width=1), name='EMA 21'))
    
    # VWAP
    if 'vwap' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['vwap'], line=dict(color='purple', width=1, dash='dot'), name='VWAP'))
    
    # Entry/Stop levels (if active signal)
    if signal and signal.direction != 'NEUTRAL':
        fig.add_hline(y=signal.entry, line_dash="dash", line_color="white", annotation_text="Entry")
        fig.add_hline(y=signal.stop_loss, line_dash="solid", line_color="red", annotation_text="Stop")
        fig.add_hline(y=signal.take_profit_1, line_dash="solid", line_color="green", annotation_text="TP1")
    
    fig.update_layout(
        title=f"{symbol} Price Action",
        template="plotly_dark",
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


# === SIDEBAR ===
st.sidebar.title("🤖 NQ Quant Bot")
st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox("Auto Refresh (60s)", value=False)
interval = st.sidebar.selectbox("Timeframe", ["5m", "15m", "1h"], index=0)

if auto_refresh:
    time.sleep(60)
    st.rerun()

st.sidebar.markdown("### Active Strategies")
st.sidebar.success("✅ NQ Optimized")
st.sidebar.success("✅ Gold Futures")


# === MAIN CONTENT ===
st.title(f"Live Market Dashboard ({interval})")

col1, col2 = st.columns(2)

# === NQ Futures ===
with col1:
    st.markdown("## 🖥️ Nasdaq (NQ=F)")
    strategy_nq = OptimizedStrategy()
    df_nq, signal_nq = process_market("NQ=F", strategy_nq, interval)
    
    if signal_nq:
        # Signal Box
        sig_color = "signal-long" if signal_nq.direction == 'LONG' else "signal-short" if signal_nq.direction == 'SHORT' else "signal-neutral"
        st.markdown(f"""
        <div class="big-signal {sig_color}">
            {signal_nq.direction} ({signal_nq.confidence*100:.0f}%)
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Entry", f"{signal_nq.entry:.2f}")
        m2.metric("Stop", f"{signal_nq.stop_loss:.2f}", delta=f"-{abs(signal_nq.entry-signal_nq.stop_loss):.2f}" if signal_nq.direction=='LONG' else f"+{abs(signal_nq.entry-signal_nq.stop_loss):.2f}")
        m3.metric("TP1", f"{signal_nq.take_profit_1:.2f}", delta=f"+{abs(signal_nq.take_profit_1-signal_nq.entry):.2f}" if signal_nq.direction=='LONG' else f"-{abs(signal_nq.take_profit_1-signal_nq.entry):.2f}")
        
        # Factors
        st.caption("Confluence Factors:")
        for factor, desc in signal_nq.factors.items():
            st.code(f"{factor.upper()}: {desc}")
            
        # Chart
        draw_chart(df_nq, signal_nq, "NQ=F")


# === Gold Futures ===
with col2:
    st.markdown("## 🥇 Gold (GC=F)")
    strategy_gold = GoldStrategy(GoldConfig())
    df_gold, signal_gold = process_market("GC=F", strategy_gold, interval)
    
    if signal_gold:
        # Signal Box
        sig_color = "signal-long" if signal_gold.direction == 'LONG' else "signal-short" if signal_gold.direction == 'SHORT' else "signal-neutral"
        st.markdown(f"""
        <div class="big-signal {sig_color}">
            {signal_gold.direction} ({signal_gold.confidence*100:.0f}%)
        </div>
        """, unsafe_allow_html=True)
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Entry", f"{signal_gold.entry:.1f}")
        m2.metric("Stop", f"{signal_gold.stop_loss:.1f}", delta=f"-{abs(signal_gold.entry-signal_gold.stop_loss):.1f}" if signal_gold.direction=='LONG' else f"+{abs(signal_gold.entry-signal_gold.stop_loss):.1f}")
        m3.metric("TP1", f"{signal_gold.take_profit_1:.1f}", delta=f"+{abs(signal_gold.take_profit_1-signal_gold.entry):.1f}" if signal_gold.direction=='LONG' else f"-{abs(signal_gold.take_profit_1-signal_gold.entry):.1f}")
        
        # Factors
        st.caption("Confluence Factors:")
        for factor, desc in signal_gold.factors.items():
            st.code(f"{factor.upper()}: {desc}")
            
        # Chart
        draw_chart(df_gold, signal_gold, "GC=F")

st.markdown("---")
st.caption("Last updated: " + time.strftime("%H:%M:%S"))
