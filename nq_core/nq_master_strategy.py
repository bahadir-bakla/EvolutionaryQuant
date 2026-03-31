import pandas as pd
import numpy as np

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def add_nq_master_features(df_1d, df_1h, df_5m, es_1h=None, es_5m=None, use_tick_data=False, df_ticks=None):
    """
    Computes features across standard timeframes and maps them to the 5m execution timeframe.
    """
    # 1. Daily Trend (1D)
    df_1d['ema_20'] = df_1d['close'].ewm(span=20, adjust=False).mean()
    df_1d['ema_50'] = df_1d['close'].ewm(span=50, adjust=False).mean()
    df_1d['daily_trend'] = 'NEUTRAL'
    df_1d.loc[(df_1d['close'] > df_1d['ema_20']) & (df_1d['ema_20'] > df_1d['ema_50']), 'daily_trend'] = 'BULLISH'
    df_1d.loc[(df_1d['close'] < df_1d['ema_20']) & (df_1d['ema_20'] < df_1d['ema_50']), 'daily_trend'] = 'BEARISH'
    
    # Forward fill daily trend to 5m
    df_1d_aligned = df_1d[['daily_trend']].reindex(df_5m.index, method='ffill')
    df_5m['daily_trend'] = df_1d_aligned['daily_trend'].fillna('NEUTRAL')
    
    # 2. HTF Liquidity (1H Swings)
    df_1h['htf_swing_high'] = df_1h['high'].rolling(20, center=True).max().ffill()
    df_1h['htf_swing_low'] = df_1h['low'].rolling(20, center=True).min().ffill()
    
    df_1h_aligned_h = df_1h[['htf_swing_high']].reindex(df_5m.index, method='ffill')
    df_1h_aligned_l = df_1h[['htf_swing_low']].reindex(df_5m.index, method='ffill')
    df_5m['htf_swing_high'] = df_1h_aligned_h['htf_swing_high']
    df_5m['htf_swing_low'] = df_1h_aligned_l['htf_swing_low']
    
    # 2.2 5m Liquidity Sweeps
    df_5m['sweep_high'] = (df_5m['high'] > df_5m['htf_swing_high']) & (df_5m['close'] < df_5m['htf_swing_high'])
    df_5m['sweep_low'] = (df_5m['low'] < df_5m['htf_swing_low']) & (df_5m['close'] > df_5m['htf_swing_low'])
    
    # 2.3 Institutional Session Extension (Asia + London)
    # Futures trade date starts at 18:00 EST. Offset by +6 hours so 18:00 becomes 00:00 of the "trade date"
    df_5m['trade_date'] = (df_5m.index + pd.Timedelta(hours=6)).date
    daily_opens = df_5m.groupby('trade_date')['open'].first()
    df_5m['daily_open'] = df_5m['trade_date'].map(daily_opens)
    df_5m['session_ext_pct'] = (df_5m['close'] - df_5m['daily_open']) / df_5m['daily_open'] * 100
    
    # 2.5 Dynamic Red News Filter (CPI, NFP, FOMC)
    df_5m['is_red_news_day'] = False
    for ts in df_5m.index:
        day = ts.day
        month = ts.month
        weekday = ts.weekday()
        if day in [12, 13]: # Approx CPI
            df_5m.loc[ts, 'is_red_news_day'] = True
        elif day <= 7 and weekday == 4: # NFP First Friday
            df_5m.loc[ts, 'is_red_news_day'] = True
        elif month in [1, 3, 5, 6, 7, 9, 11, 12] and day in [18, 19, 20]: # Approx FOMC
            df_5m.loc[ts, 'is_red_news_day'] = True
            
    # 3. 15m ORB (Opening Range Breakout) 09:30 - 09:45 EST
    df_5m['is_orb_window'] = (df_5m.index.hour == 9) & (df_5m.index.minute >= 30) & (df_5m.index.minute < 45)
    
    # Create daily groups to find ORB high/low
    df_5m['date_str'] = df_5m.index.date
    orb_highs = df_5m[df_5m['is_orb_window']].groupby('date_str')['high'].max()
    orb_lows = df_5m[df_5m['is_orb_window']].groupby('date_str')['low'].min()
    
    df_5m['orb_high'] = df_5m['date_str'].map(orb_highs)
    df_5m['orb_low'] = df_5m['date_str'].map(orb_lows)
    
    # Breakout logic with Volume Confirmation
    df_5m['orb_breakout'] = 'NONE'
    # Limit ORB triggers strictly to the morning volatility window (09:45 to 12:00)
    post_orb = (df_5m.index.time >= pd.to_datetime('09:45').time()) & (df_5m.index.time <= pd.to_datetime('12:00').time())
    
    df_5m['vol_sma_20'] = df_5m['volume'].rolling(20).mean()
    high_vol = df_5m['volume'] > df_5m['vol_sma_20']
    
    df_5m.loc[post_orb & high_vol & (df_5m['close'] > df_5m['orb_high']), 'orb_breakout'] = 'LONG'
    df_5m.loc[post_orb & high_vol & (df_5m['close'] < df_5m['orb_low']), 'orb_breakout'] = 'SHORT'
    
    # VWAP for Reversion targeting
    df_5m['typical'] = (df_5m['high'] + df_5m['low'] + df_5m['close']) / 3
    df_5m['vwap'] = (df_5m['typical'] * df_5m['volume']).groupby(df_5m['date_str']).cumsum() / df_5m['volume'].groupby(df_5m['date_str']).cumsum()
    
    # 5m RSI for precise Reversion Entries
    df_5m['rsi_5m'] = calculate_rsi(df_5m['close'], period=14)
    df_5m['rsi_5m_prev'] = df_5m['rsi_5m'].shift(1)
    
    # 4. SMT Divergence (ES vs NQ Correlation)
    if es_1h is not None and es_5m is not None:
        es_1h_h = es_1h['high'].rolling(20, center=True).max().ffill().reindex(df_5m.index, method='ffill')
        es_1h_l = es_1h['low'].rolling(20, center=True).min().ffill().reindex(df_5m.index, method='ffill')
        es_5m_aligned = es_5m.reindex(df_5m.index, method='ffill')
        
        df_5m['es_sweep_low'] = (es_5m_aligned['low'] < es_1h_l) & (es_5m_aligned['close'] > es_1h_l)
        df_5m['es_sweep_high'] = (es_5m_aligned['high'] > es_1h_h) & (es_5m_aligned['close'] < es_1h_h)
        
        nq_20_low = df_5m['low'].rolling(20).min()
        nq_20_high = df_5m['high'].rolling(20).max()
        es_20_low = es_5m_aligned['low'].rolling(20).min()
        es_20_high = es_5m_aligned['high'].rolling(20).max()
        
        # Bullish Divergence: NQ at recent low, but ES is NOT (ES stronger)
        df_5m['smt_bullish_div'] = (df_5m['low'] <= nq_20_low) & (es_5m_aligned['low'] > es_20_low)
        # Bearish Divergence: NQ at recent high, but ES is NOT (ES weaker)
        df_5m['smt_bearish_div'] = (df_5m['high'] >= nq_20_high) & (es_5m_aligned['high'] < es_20_high)
        
        df_5m['smt_ok_long'] = df_5m['es_sweep_low'] | df_5m['smt_bullish_div']
        df_5m['smt_ok_short'] = df_5m['es_sweep_high'] | df_5m['smt_bearish_div']
    else:
        df_5m['smt_ok_long'] = True
        df_5m['smt_ok_short'] = True
    
    # 5. Order Flow CVD (Cumulative Volume Delta)
    df_5m['cvd_bullish'] = False
    df_5m['cvd_bearish'] = False
    
    if use_tick_data and df_ticks is not None and not df_ticks.empty:
        # Graceful Degradation: Compute Tick Delta if Level-2 data is supplied by live broker
        # df_ticks expects: ['price', 'volume', 'side' (ASK/BID)]
        # We resample ticks into 5m buckets and sum volume where side == ASK (Buying pressure) vs BID
        try:
            buy_vol = df_ticks[df_ticks['side'] == 'ASK'].resample('5min')['volume'].sum()
            sell_vol = df_ticks[df_ticks['side'] == 'BID'].resample('5min')['volume'].sum()
            cvd = (buy_vol - sell_vol).reindex(df_5m.index, fill_value=0)
            
            # Massive buying pressure hidden in the candle
            df_5m['cvd_bullish'] = cvd > (df_5m['volume'] * 0.20) 
            # Massive selling pressure
            df_5m['cvd_bearish'] = cvd < -(df_5m['volume'] * 0.20)
        except Exception:
            pass # Fallback to False
            
    return df_5m


class NQMasterStrategyCore:
    def __init__(self, starting_balance=1000.0):
        self.balance = starting_balance
        self.basket = []
        self.orb_traded_today = False
        self.last_traded_date = None
        self.last_reversion_hour = None

    def manage_trades(self, current_price, time_step):
        realized_pnl = 0.0
        remaining_basket = []
        
        for trade in self.basket:
            # Pnl based on NQ $20/pt multiplier
            pnl_pts = (current_price - trade['entry']) if trade['dir'] == 'LONG' else (trade['entry'] - current_price)
            pnl_usd = pnl_pts * 20.0 * trade['lot']
            
            # Check Stop Loss / Take Profit
            if trade['dir'] == 'LONG':
                trade['highest'] = max(trade['highest'], current_price)
                mfe = trade['highest'] - trade['entry']
            else:
                trade['lowest'] = min(trade['lowest'], current_price)
                mfe = trade['entry'] - trade['lowest']
                
            # Break-Even Logic: If trade goes 50% towards TP, move SL to entry + 2 points
            if mfe >= trade['tp_pts'] * 0.5:
                # sl_pts is distance from entry down. Negative means it's above entry.
                trade['sl_pts'] = min(trade['sl_pts'], -2.0)
                
            # Trailing stop mechanism or hard limits
            if pnl_pts <= -trade['sl_pts'] or pnl_pts >= trade['tp_pts']:
                realized_pnl += pnl_usd
            else:
                remaining_basket.append(trade)
                
        self.basket = remaining_basket
        self.balance += realized_pnl
        return realized_pnl

    def evaluate(self, row, current_price, time_step, lot_size):
        date_str = time_step.date()
        if self.last_traded_date != date_str:
            self.orb_traded_today = False
            self.last_traded_date = date_str
            self.last_reversion_hour = None
            
        signal = None
        reason = ""
        tp_pts = 100
        sl_pts = 50
        
        # 1. ORB Breakout Logic (Only if NOT a Red News Day)
        if not self.orb_traded_today and row['orb_breakout'] != 'NONE' and not row['is_red_news_day']:
            if row['orb_breakout'] == 'LONG' and row['daily_trend'] in ['BULLISH', 'NEUTRAL']:
                signal = 'LONG'
                reason = 'ORB_BREAKOUT_LONG'
                self.orb_traded_today = True
            elif row['orb_breakout'] == 'SHORT' and row['daily_trend'] in ['BEARISH', 'NEUTRAL']:
                signal = 'SHORT'
                reason = 'ORB_BREAKOUT_SHORT'
                self.orb_traded_today = True

        # 2. Institutional Session Extension Reversion (Fade the >1.5% pushes)
        if not row['is_red_news_day'] and self.last_reversion_hour != time_step.hour:
            extension = row['session_ext_pct']
            
            # Massive downside push (Big Boys Sold Asia/London) -> Mean Revert Up
            if extension <= -1.5 and row['sweep_low']:
                dist_to_vwap = row['vwap'] - current_price
                if dist_to_vwap > 40 and row['rsi_5m_prev'] < 30 and row['rsi_5m'] >= 30 and row.get('smt_ok_long', True): 
                    signal = 'LONG'
                    reason = 'SESSION_EXT_REVERSION_LONG'
                    tp_pts = min(200, dist_to_vwap)
                    sl_pts = tp_pts * 0.5
                    self.last_reversion_hour = time_step.hour
                    
            # Massive upside push (Big Boys Bought Asia/London) -> Mean Revert Down
            elif extension >= 1.5 and row['sweep_high']:
                dist_to_vwap = current_price - row['vwap']
                if dist_to_vwap > 40 and row['rsi_5m_prev'] > 70 and row['rsi_5m'] <= 70 and row.get('smt_ok_short', True):
                    signal = 'SHORT'
                    reason = 'SESSION_EXT_REVERSION_SHORT'
                    tp_pts = min(200, dist_to_vwap)
                    sl_pts = tp_pts * 0.5
                    self.last_reversion_hour = time_step.hour

        if signal and len(self.basket) < 2:  # Precision strikes, max 2 concurrent trades
            self.basket.append({
                'dir': signal,
                'entry': current_price,
                'lot': lot_size,
                'time': time_step,
                'reason': reason,
                'sl_pts': sl_pts,
                'tp_pts': tp_pts,
                'highest': current_price,
                'lowest': current_price
            })
            # print(f"[{time_step}] TRIGGER: {signal} | Reason: {reason} | Price: {current_price:.2f}")

