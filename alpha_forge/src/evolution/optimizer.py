"""
Alpha-Forge DEAP Evolution Engine  — v3 Multi-Asset Full Suite
==============================================================
A+ Tier:
  [Gold]  Reversion Scalper  — red-streak + RSI oversold + 50% retrace
  [Gold]  Sniper              — daily bias + 4H OB tap + basket + trail
  [Gold]  Master              — HTF S/R + FVG + momentum breakout
B+ Tier:
  [NQ]    Breakdown           — 4H macro support break short
  [NQ]    Wick Reversal       — US session ORB wick rejection
B Tier:
  [Silver/NQ] Momentum        — ADX + RSI trend momentum

Oracle: TimesFM, MiroFish, Gemma — confluence filter on all strategies.

Genome (25 parameters total):
  NQ Wick   : wick_threshold, stop_atr, take_profit, orb_bars, vol_ratio, min_body
  Gold Rev  : gold_streak, gold_rsi_limit, gold_retrace, gold_stop_mult, gold_risk_pct
  NQ BD     : nq_bd_lookback, nq_bd_sl_mult, nq_bd_tp_mult, nq_bd_risk_pct
  Gold Sniper: gs_sl_points, gs_tp_mult, gs_risk_pct, gs_max_basket
  Gold Master: gm_tp_points, gm_sl_points, gm_risk_pct, gm_momentum_thresh
  Silver Mom : silver_adx_min, silver_risk_pct
  Oracle     : oracle_thresh, oracle_weight
"""

import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from typing import Dict, List, Tuple, Optional
import json
import os
import time
import logging

logger = logging.getLogger(__name__)


class AdvancedFitnessCalculator:
    """Multi-objective fitness: Sortino + Sharpe + Calmar + PF."""

    def __init__(
        self,
        weights: Dict[str, float] = None,
        risk_free_rate: float = 0.02,
        trading_days: int = 252,
    ):
        self.weights = weights or {
            "sortino": 0.35,
            "sharpe": 0.20,
            "calmar": 0.25,
            "profit_factor": 0.10,
            "win_rate": 0.05,
            "trade_count": 0.05,
        }
        self.rf = risk_free_rate
        self.td = trading_days

    def calculate(self, trades: List[Dict]) -> Tuple[float, Dict]:
        if len(trades) < 3:
            return -10.0, {}

        pnls = np.array([t["pnl"] for t in trades])
        returns = pnls / 1000.0

        # Sortino
        excess = returns - self.rf / self.td
        downside = excess[excess < 0]
        if len(downside) == 0:
            sortino = 10.0
        else:
            d_std = np.sqrt(np.mean(downside**2))
            sortino = np.mean(excess) / d_std * np.sqrt(self.td) if d_std > 0 else 10.0

        # Sharpe
        sharpe = (
            np.mean(excess) / returns.std() * np.sqrt(self.td)
            if returns.std() > 0
            else 0.0
        )

        # Max Drawdown
        balance = 1000.0
        peak = balance
        max_dd = 0.0
        for pnl in pnls:
            balance += pnl
            peak = max(peak, balance)
            dd = (peak - balance) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Calmar
        annual_ret = np.mean(returns) * self.td
        calmar = annual_ret / max_dd if max_dd > 0 else 0.0

        # Profit Factor
        wins = pnls[pnls > 0].sum()
        losses = abs(pnls[pnls < 0].sum())
        pf = wins / losses if losses > 0 else (10.0 if wins > 0 else 0.0)

        # Win Rate
        wr = np.mean(pnls > 0)

        # Normalize
        n_sortino = np.clip(sortino / 3.0, 0, 1)
        n_sharpe = np.clip(sharpe / 2.0, 0, 1)
        n_calmar = np.clip(calmar / 2.0, 0, 1)
        n_pf = np.clip(pf / 3.0, 0, 1)
        n_wr = wr
        n_tc = np.clip(len(trades) / 30.0, 0, 1)

        fitness = (
            n_sortino * self.weights["sortino"]
            + n_sharpe * self.weights["sharpe"]
            + n_calmar * self.weights["calmar"]
            + n_pf * self.weights["profit_factor"]
            + n_wr * self.weights["win_rate"]
            + n_tc * self.weights["trade_count"]
        )

        # Penalties
        if max_dd > 0.20:
            fitness *= 0.5
        if max_dd > 0.35:
            fitness *= 0.2
        if len(trades) < 10:
            fitness *= 0.3

        metrics = {
            "sortino": sortino,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "calmar": calmar,
            "profit_factor": pf,
            "win_rate": wr,
            "trade_count": len(trades),
            "total_return": float(np.sum(returns) * 100),
        }

        return fitness, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  GOLD REVERSION BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────
class GoldReversionBacktester:
    """
    DEAP tarafindan optimize edilen Gold Reversion Scalper.
    Parametreler: gold_streak, gold_rsi_limit, gold_retrace,
                  gold_stop_mult, gold_risk_pct
    Oracle filtresi: gemma_regime, timesfm_signal, oracle_thresh, oracle_weight
    """

    POINT_VALUE = 100.0  # XAUUSD: $100 / ounce-point (lot=1 oz)

    def __init__(self, gold_df: pd.DataFrame):
        self.df = gold_df.copy()

    def run(self, params: Dict, start_balance: float = 500.0) -> Tuple[float, List[Dict]]:
        streak_thresh = max(2, int(round(params.get("gold_streak", 5))))
        rsi_limit     = float(params.get("gold_rsi_limit", 32.0))
        retrace_pct   = float(params.get("gold_retrace", 0.45))
        stop_mult     = float(params.get("gold_stop_mult", 1.5))
        risk_pct      = float(params.get("gold_risk_pct", 3.0)) / 100.0
        oracle_thresh  = float(params.get("oracle_thresh", 0.3))
        oracle_weight  = float(params.get("oracle_weight", 0.5))

        # Compound sizing
        compound_step = 300.0
        compound_rate = 0.35

        df = self.df
        balance  = start_balance
        trades: List[Dict] = []
        position = None
        drop_high = drop_low = 0.0

        for i in range(streak_thresh + 20, len(df)):
            row   = df.iloc[i]
            price = float(row["close"])

            # ── Oracle filter ────────────────────────────────────────────────
            if oracle_weight > 0.1:
                gemma_num    = float(row.get("gemma_regime_numeric", 0.0))
                gemma_conf   = float(row.get("gemma_confidence", 0.5))
                tf_signal    = float(row.get("timesfm_signal", 0.0))
                mf_stress    = float(row.get("mirofish_avg_stress", 0.0))
                oracle_ok    = (gemma_conf >= oracle_thresh * oracle_weight
                                and mf_stress < 0.8)
            else:
                oracle_ok    = True

            # ── Position management ──────────────────────────────────────────
            if position is not None:
                lo, hi = float(row["low"]), float(row["high"])
                pdir, entry_p, sl, tp, size = (
                    position["dir"], position["entry"],
                    position["sl"], position["tp"], position["size"]
                )
                if pdir == "LONG":
                    if lo <= sl:
                        pnl = (sl - entry_p) * size * self.POINT_VALUE
                        balance += pnl
                        trades.append({"pnl": pnl, "dir": pdir, "exit": "SL"})
                        position = None
                    elif hi >= tp:
                        pnl = (tp - entry_p) * size * self.POINT_VALUE
                        balance += pnl
                        trades.append({"pnl": pnl, "dir": pdir, "exit": "TP"})
                        # Flip SHORT
                        sl_s  = drop_high
                        tp_s  = drop_low
                        rppu  = abs(price - sl_s) * self.POINT_VALUE
                        if rppu > 0.01:
                            pb    = max(0, int((balance - start_balance) // compound_step))
                            sz    = min((balance * risk_pct * (1 + pb * compound_rate))
                                        / rppu, 2.0)
                            position = {"dir":"SHORT","entry":price,"size":max(0.001,sz),
                                        "sl":sl_s,"tp":tp_s}
                        else:
                            position = None
                else:
                    if hi >= sl:
                        pnl = (entry_p - sl) * size * self.POINT_VALUE
                        balance += pnl
                        trades.append({"pnl": pnl, "dir": pdir, "exit": "SL"})
                        position = None
                    elif lo <= tp:
                        pnl = (entry_p - tp) * size * self.POINT_VALUE
                        balance += pnl
                        trades.append({"pnl": pnl, "dir": pdir, "exit": "TP"})
                        position = None
                if balance <= 0:
                    break
                continue

            # ── Entry signal ─────────────────────────────────────────────────
            if not oracle_ok:
                continue

            red_s = int(row.get("red_streak", 0))
            rsi_v = float(row.get("rsi", 50))

            if red_s >= streak_thresh and rsi_v < rsi_limit:
                ws = max(0, i - streak_thresh)
                wd = df.iloc[ws : i + 1]
                drop_high = float(wd["high"].iloc[0])
                drop_low  = float(wd["low"].iloc[-1])
                rng = drop_high - drop_low
                if rng < 0.1:
                    continue

                sl_l  = drop_low  - rng * stop_mult
                tp_l  = drop_low  + rng * retrace_pct
                rppu  = abs(price - sl_l) * self.POINT_VALUE
                if rppu < 0.01:
                    continue

                pb    = max(0, int((balance - start_balance) // compound_step))
                size  = min((balance * risk_pct * (1 + pb * compound_rate))
                             / rppu, 2.0)
                position = {"dir":"LONG","entry":price,
                            "size":max(0.001,size),"sl":sl_l,"tp":tp_l}

        # Close open at last price
        if position:
            last = float(df.iloc[-1]["close"])
            pnl  = (last - position["entry"]) * position["size"] * self.POINT_VALUE
            if position["dir"] == "SHORT":
                pnl = -pnl
            balance += pnl
            trades.append({"pnl": pnl, "dir": position["dir"], "exit": "EOD"})

        return balance, trades


# ─────────────────────────────────────────────────────────────────────────────
#  NQ BREAKDOWN BACKTESTER
# ─────────────────────────────────────────────────────────────────────────────
class NQBreakdownBacktester:
    """
    NQ Extreme Breakdown — 4H macro support kirilirsa short.
    Lookback, SL/TP ATR mult DEAP parametrelerinden aliniyor.
    """

    def __init__(self, nq_df: pd.DataFrame):
        self.df = nq_df.copy()

    def run(self, params: Dict, start_balance: float = 150.0) -> Tuple[float, List[Dict]]:
        lookback  = max(10, int(round(params.get("nq_bd_lookback", 56))))
        sl_mult   = float(params.get("nq_bd_sl_mult", 6.5))
        tp_mult   = float(params.get("nq_bd_tp_mult", 12.2))
        risk_pct  = float(params.get("nq_bd_risk_pct", 1.35)) / 100.0
        oracle_thresh = float(params.get("oracle_thresh", 0.3))
        oracle_weight = float(params.get("oracle_weight", 0.5))

        df = self.df
        if "macro_low" not in df.columns:
            df["macro_low"] = df["low"].shift(1).rolling(lookback).min()

        balance  = start_balance
        trades:  List[Dict] = []
        position = None

        for i in range(lookback + 5, len(df)):
            row   = df.iloc[i]
            price = float(row["close"])
            atr   = float(row.get("atr", 1.0))
            if np.isnan(atr) or atr == 0:
                continue

            # Oracle: durante RANGE regime skip breakdown shorts
            if oracle_weight > 0.1:
                gemma_num  = float(row.get("gemma_regime_numeric", 0.0))
                gemma_conf = float(row.get("gemma_confidence", 0.5))
                mf_stress  = float(row.get("mirofish_avg_stress", 0.0))
                # Breakdown sadece RANGE disinda calisir (trending / volatile piyasada)
                if gemma_num == 0.0 and gemma_conf >= oracle_thresh * oracle_weight:
                    if position is None:
                        continue

            if position is not None:
                lo, hi = float(row["low"]), float(row["high"])
                if hi >= position["sl"]:
                    pnl = (position["entry"] - position["sl"]) * position["size"] * 20.0
                    balance += pnl; trades.append({"pnl":pnl,"exit":"SL"}); position = None
                elif lo <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) * position["size"] * 20.0
                    balance += pnl; trades.append({"pnl":pnl,"exit":"TP"}); position = None
                if balance <= 0:
                    break
                continue

            macro_low = row.get("macro_low", np.nan)
            if pd.isna(macro_low):
                continue

            is_breakdown = price < macro_low
            is_red       = price < float(row["open"])
            if is_breakdown and is_red:
                sl   = price + atr * sl_mult
                tp   = price - atr * tp_mult
                rppu = abs(price - sl) * 20.0
                if rppu < 0.01:
                    continue
                size = min((balance * risk_pct) / rppu, 2.0)
                position = {"entry":price,"size":max(0.001,size),"sl":sl,"tp":tp}

        if position:
            last = float(df.iloc[-1]["close"])
            pnl  = (position["entry"] - last) * position["size"] * 20.0
            balance += pnl; trades.append({"pnl":pnl,"exit":"EOD"})

        return balance, trades


# ─────────────────────────────────────────────────────────────────────────────
#  GOLD SNIPER BACKTESTER  (A+ tier)
# ─────────────────────────────────────────────────────────────────────────────
class GoldSniperBacktester:
    """
    Gold Sniper — daily bias + 4H OB tap + basket management + trail stop.
    Signals: bullish_bias + (4H_reject_down OR ob_tap_bullish OR sweep_minor_low)
    Exit: trailing stop (BE at 50% target, trail above 100%); basket hard-stop; 72h stale kill.
    Operates on 1H gold data (same oracle_gold as GoldReversionBacktester).
    """
    POINT_VALUE = 100.0

    def __init__(self, gold_df: pd.DataFrame):
        self.df = self._add_features(gold_df.copy())

    @staticmethod
    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        h = df["high"]
        l = df["low"]
        c = df["close"]
        o = df["open"]
        if "atr" not in df.columns:
            tr = pd.concat([h - l,
                            (h - c.shift()).abs(),
                            (l - c.shift()).abs()], axis=1).max(axis=1)
            df["atr"] = tr.rolling(14).mean()
        # Daily bias (24 bars = 1 day on 1H data)
        df["d1_close"] = c.shift(24)
        df["d2_close"] = c.shift(48)
        df["bias_bull"] = (df["d1_close"] > df["d2_close"]).astype(float)
        df["bias_bear"] = (df["d1_close"] < df["d2_close"]).astype(float)
        # 4H context
        df["h4_high"] = h.rolling(4).max().shift(1)
        df["h4_low"]  = l.rolling(4).min().shift(1)
        h4_range = (df["h4_high"] - df["h4_low"]).clip(lower=1e-3)
        df["h4_rej_down"] = ((c - df["h4_low"]) > h4_range * 0.6).astype(float)
        df["h4_rej_up"]   = ((df["h4_high"] - c) > h4_range * 0.6).astype(float)
        # Minor sweeps
        df["minor_high"] = h.rolling(5).max().shift(1)
        df["minor_low"]  = l.rolling(5).min().shift(1)
        df["sweep_low"]  = ((l < df["minor_low"]) & (c > df["minor_low"])).astype(float)
        df["sweep_high"] = ((h > df["minor_high"]) & (c < df["minor_high"])).astype(float)
        # OB tap
        is_down = (c < o).astype(float)
        is_up   = (c > o).astype(float)
        df["ob_tap_bull"] = (is_down.shift(2) * is_up.shift(1) *
                             (l <= l.shift(2)) * (c >= c.shift(1))).clip(0, 1)
        df["ob_tap_bear"] = (is_up.shift(2) * is_down.shift(1) *
                             (h >= h.shift(2)) * (c <= c.shift(1))).clip(0, 1)
        return df

    def run(self, params: Dict, start_balance: float = 250.0) -> Tuple[float, List[Dict]]:
        sl_points    = float(params.get("gs_sl_points", 15.0))
        tp_mult      = float(params.get("gs_tp_mult", 3.0))      # TP = sl * tp_mult
        risk_pct     = float(params.get("gs_risk_pct", 2.0)) / 100.0
        max_basket   = max(1, int(round(params.get("gs_max_basket", 3))))
        oracle_thresh = float(params.get("oracle_thresh", 0.3))
        oracle_weight = float(params.get("oracle_weight", 0.5))

        df = self.df
        balance = start_balance
        trades: List[Dict] = []
        basket: List[Dict] = []   # active bullets
        basket_dir = 0

        for i in range(50, len(df)):
            row   = df.iloc[i]
            price = float(row["close"])
            atr   = float(row.get("atr", 15.0))
            if np.isnan(atr) or atr == 0:
                continue

            target_pts = sl_points * tp_mult

            # ── Manage basket ─────────────────────────────────────────────────
            if basket:
                # Basket hard-dollar stop
                basket_pnl = sum(
                    (price - b["entry"]) * b["dir"] * b["size"] * self.POINT_VALUE
                    for b in basket
                )
                basket_cost = sum(b["size"] for b in basket) * sl_points * self.POINT_VALUE
                if basket_pnl <= -basket_cost:
                    balance += basket_pnl
                    trades.append({"pnl": basket_pnl, "exit": "BASKET_STOP"})
                    basket = []
                    basket_dir = 0
                    if balance <= 0:
                        break
                    continue

                alive = []
                for b in basket:
                    pts_fav = (price - b["entry"]) * b["dir"]
                    b["max_pts"] = max(b.get("max_pts", 0), pts_fav)

                    # Trailing stop logic
                    if b["max_pts"] >= target_pts:
                        dynamic_stop = -(b["max_pts"] - target_pts * 0.5)
                    elif b["max_pts"] >= target_pts * 0.5:
                        dynamic_stop = -target_pts * 0.10  # break-even lock
                    else:
                        dynamic_stop = sl_points

                    # 72h stale kill
                    age_bars = i - b["bar_entry"]
                    if age_bars > 72:
                        pnl = pts_fav * b["size"] * self.POINT_VALUE
                        balance += pnl
                        trades.append({"pnl": pnl, "exit": "STALE_72H"})
                    elif pts_fav <= -dynamic_stop:
                        pnl = pts_fav * b["size"] * self.POINT_VALUE
                        balance += pnl
                        trades.append({"pnl": pnl, "exit": "TRAIL_STOP"})
                    else:
                        alive.append(b)
                basket = alive
                if not basket:
                    basket_dir = 0
                if balance <= 0:
                    break
                continue

            # ── Entry logic ───────────────────────────────────────────────────
            if oracle_weight > 0.1:
                gemma_conf = float(row.get("gemma_confidence", 0.5))
                mf_stress  = float(row.get("mirofish_avg_stress", 0.0))
                if gemma_conf < oracle_thresh * oracle_weight or mf_stress > 0.85:
                    continue

            bias_bull = float(row.get("bias_bull", 0.0))
            bias_bear = float(row.get("bias_bear", 0.0))
            h4_rd     = float(row.get("h4_rej_down", 0.0))
            h4_ru     = float(row.get("h4_rej_up",   0.0))
            ob_bull   = float(row.get("ob_tap_bull",  0.0))
            ob_bear   = float(row.get("ob_tap_bear",  0.0))
            sw_low    = float(row.get("sweep_low",   0.0))
            sw_high   = float(row.get("sweep_high",  0.0))

            long_ok  = bias_bull and (h4_rd or ob_bull or sw_low)
            short_ok = bias_bear and (h4_ru or ob_bear or sw_high)

            if long_ok or short_ok:
                direction = 1 if long_ok else -1
                rppu = sl_points * self.POINT_VALUE
                if rppu < 0.01:
                    continue
                size = min((balance * risk_pct) / rppu, 2.0)
                size = max(0.001, size)
                bullet = {"entry": price, "dir": direction, "size": size,
                          "max_pts": 0.0, "bar_entry": i}
                basket = [bullet]
                basket_dir = direction

        # EOD close
        if basket:
            price = float(df.iloc[-1]["close"])
            for b in basket:
                pnl = (price - b["entry"]) * b["dir"] * b["size"] * self.POINT_VALUE
                balance += pnl
                trades.append({"pnl": pnl, "exit": "EOD"})

        return balance, trades


# ─────────────────────────────────────────────────────────────────────────────
#  GOLD MASTER BACKTESTER  (A+ tier)
# ─────────────────────────────────────────────────────────────────────────────
class GoldMasterBacktester:
    """
    Gold Master — HTF S/R + FVG + momentum breakout.
    Single high-conviction trade, hard SL/TP in points.
    Operates on 1H gold data.
    """
    POINT_VALUE = 100.0

    def __init__(self, gold_df: pd.DataFrame):
        self.df = self._add_features(gold_df.copy())

    @staticmethod
    def _add_features(df: pd.DataFrame) -> pd.DataFrame:
        h = df["high"]
        l = df["low"]
        c = df["close"]
        if "atr" not in df.columns:
            tr = pd.concat([h - l,
                            (h - c.shift()).abs(),
                            (l - c.shift()).abs()], axis=1).max(axis=1)
            df["atr"] = tr.rolling(14).mean()
        # HTF S/R (48-bar rolling = ~2 days on 1H)
        df["htf_support"]    = l.rolling(48).min().shift(1)
        df["htf_resistance"] = h.rolling(48).max().shift(1)
        # FVG
        df["fvg_bull"] = (l > h.shift(2)).astype(float)
        df["fvg_bear"] = (h < l.shift(2)).astype(float)
        # Momentum ROC (10-bar)
        df["mom_roc"] = c.pct_change(10) * 100
        return df

    def run(self, params: Dict, start_balance: float = 200.0) -> Tuple[float, List[Dict]]:
        tp_points  = float(params.get("gm_tp_points", 80.0))
        sl_points  = float(params.get("gm_sl_points", 40.0))
        risk_pct   = float(params.get("gm_risk_pct", 2.0)) / 100.0
        mom_thresh = float(params.get("gm_momentum_thresh", 0.3))
        oracle_thresh = float(params.get("oracle_thresh", 0.3))
        oracle_weight = float(params.get("oracle_weight", 0.5))

        df = self.df
        balance = start_balance
        trades: List[Dict] = []
        position = None

        for i in range(50, len(df)):
            row   = df.iloc[i]
            price = float(row["close"])
            atr   = float(row.get("atr", 15.0))
            if np.isnan(atr) or atr == 0:
                continue

            # ── Manage open position ──────────────────────────────────────────
            if position is not None:
                pts = (price - position["entry"]) * position["dir"]
                if pts <= -sl_points:
                    pnl = -sl_points * position["size"] * self.POINT_VALUE
                    balance += pnl
                    trades.append({"pnl": pnl, "exit": "SL"})
                    position = None
                elif pts >= tp_points:
                    pnl = tp_points * position["size"] * self.POINT_VALUE
                    balance += pnl
                    trades.append({"pnl": pnl, "exit": "TP"})
                    position = None
                if balance <= 0:
                    break
                if position is None:
                    pass
                else:
                    continue

            # ── Oracle filter ─────────────────────────────────────────────────
            if oracle_weight > 0.1:
                gemma_num  = float(row.get("gemma_regime_numeric", 0.0))
                gemma_conf = float(row.get("gemma_confidence", 0.5))
                if gemma_conf < oracle_thresh * oracle_weight:
                    continue

            htf_sup = float(row.get("htf_support", np.nan))
            htf_res = float(row.get("htf_resistance", np.nan))
            fvg_b   = float(row.get("fvg_bull", 0.0))
            fvg_bear= float(row.get("fvg_bear", 0.0))
            roc     = float(row.get("mom_roc", 0.0))
            if np.isnan(roc):
                roc = 0.0

            near_sup = (not np.isnan(htf_sup) and
                        abs(price - htf_sup) < atr * 1.5)
            near_res = (not np.isnan(htf_res) and
                        abs(price - htf_res) < atr * 1.5)

            long_ok  = near_sup and fvg_b  and roc >  mom_thresh
            short_ok = near_res and fvg_bear and roc < -mom_thresh

            if long_ok or short_ok:
                direction = 1 if long_ok else -1
                rppu = sl_points * self.POINT_VALUE
                if rppu < 0.01:
                    continue
                size = min((balance * risk_pct) / rppu, 2.0)
                position = {"entry": price, "dir": direction,
                            "size": max(0.001, size)}

        # EOD close
        if position:
            price = float(df.iloc[-1]["close"])
            pts = (price - position["entry"]) * position["dir"]
            pnl = pts * position["size"] * self.POINT_VALUE
            balance += pnl
            trades.append({"pnl": pnl, "exit": "EOD"})

        return balance, trades


# ─────────────────────────────────────────────────────────────────────────────
#  SILVER MOMENTUM BACKTESTER  (B tier)
# ─────────────────────────────────────────────────────────────────────────────
class SilverMomentumBacktester:
    """
    Silver-style momentum strategy (runs on NQ oracle as proxy).
    ADX trending + RSI directional bias + SMA20 filter.
    B-tier: lower capital allocation, higher selectivity.
    """

    def __init__(self, nq_df: pd.DataFrame):
        self.df = nq_df.copy()

    def run(self, params: Dict, start_balance: float = 70.0) -> Tuple[float, List[Dict]]:
        adx_min   = float(params.get("silver_adx_min", 28.0))
        risk_pct  = float(params.get("silver_risk_pct", 1.0)) / 100.0
        sl_mult   = float(params.get("nq_bd_sl_mult", 2.5))    # reuse NQ BD param
        tp_mult   = float(params.get("nq_bd_tp_mult", 5.0))    # reuse NQ BD param
        oracle_thresh = float(params.get("oracle_thresh", 0.3))
        oracle_weight = float(params.get("oracle_weight", 0.5))

        df = self.df
        balance  = start_balance
        trades: List[Dict] = []
        position = None

        for i in range(30, len(df)):
            row   = df.iloc[i]
            price = float(row["close"])
            atr   = float(row.get("atr", 1.0))
            if np.isnan(atr) or atr == 0:
                continue

            # ── Manage position ───────────────────────────────────────────────
            if position is not None:
                hi, lo = float(row["high"]), float(row["low"])
                if position["dir"] == 1:
                    if lo <= position["sl"]:
                        pnl = (position["sl"] - position["entry"]) * position["size"] * 20.0
                        balance += pnl; trades.append({"pnl": pnl, "exit": "SL"}); position = None
                    elif hi >= position["tp"]:
                        pnl = (position["tp"] - position["entry"]) * position["size"] * 20.0
                        balance += pnl; trades.append({"pnl": pnl, "exit": "TP"}); position = None
                else:
                    if hi >= position["sl"]:
                        pnl = (position["entry"] - position["sl"]) * position["size"] * 20.0
                        balance += pnl; trades.append({"pnl": pnl, "exit": "SL"}); position = None
                    elif lo <= position["tp"]:
                        pnl = (position["entry"] - position["tp"]) * position["size"] * 20.0
                        balance += pnl; trades.append({"pnl": pnl, "exit": "TP"}); position = None
                if balance <= 0:
                    break
                if position is None:
                    pass
                else:
                    continue

            # ── Oracle filter ─────────────────────────────────────────────────
            if oracle_weight > 0.1:
                gemma_num  = float(row.get("gemma_regime_numeric", 0.0))
                gemma_conf = float(row.get("gemma_confidence", 0.5))
                # Momentum strategy: only trade in TREND regime
                if gemma_num == 0.0 or gemma_conf < oracle_thresh * oracle_weight:
                    continue

            adx  = float(row.get("adx", 0.0))
            rsi  = float(row.get("rsi", 50.0))
            sma  = float(row.get("sma_20", price))
            if np.isnan(adx) or np.isnan(rsi) or np.isnan(sma):
                continue

            if adx < adx_min:
                continue  # no trend, no trade

            long_ok  = (rsi > 55.0) and (price > sma)
            short_ok = (rsi < 45.0) and (price < sma)

            if long_ok or short_ok:
                direction = 1 if long_ok else -1
                sl = price - direction * atr * sl_mult
                tp = price + direction * atr * tp_mult
                rppu = abs(price - sl) * 20.0
                if rppu < 0.01:
                    continue
                size = min((balance * risk_pct) / rppu, 1.0)
                position = {"entry": price, "dir": direction,
                            "sl": sl, "tp": tp, "size": max(0.001, size)}

        if position:
            price = float(df.iloc[-1]["close"])
            pts   = (price - position["entry"]) * position["dir"]
            pnl   = pts * position["size"] * 20.0
            balance += pnl
            trades.append({"pnl": pnl, "exit": "EOD"})

        return balance, trades


# ─────────────────────────────────────────────────────────────────────────────
#  MULTI-ASSET BACKTESTER (Gold + NQ birlikte)
# ─────────────────────────────────────────────────────────────────────────────
class MultiAssetBacktester:
    """
    Full strategy suite, tiered capital allocation:
      A+ Gold  : Reversion (300) + Sniper (250) + Master (200)
      B+ NQ    : Breakdown (130) + Wick (50)
      B  Silver: Momentum proxy on NQ (70)
    Total deployed: $1000
    """

    def __init__(self, gold_oracle_df: pd.DataFrame, nq_oracle_df: pd.DataFrame):
        # A+ Gold strategies
        self.gold_rev    = GoldReversionBacktester(gold_oracle_df)
        self.gold_sniper = GoldSniperBacktester(gold_oracle_df)
        self.gold_master = GoldMasterBacktester(gold_oracle_df)
        # B+ NQ strategies
        self.nq_bd       = NQBreakdownBacktester(nq_oracle_df)
        self.nq_wick     = StrategyBacktesterLegacy(nq_oracle_df)
        # B Silver proxy
        self.silver      = SilverMomentumBacktester(nq_oracle_df)

    def run(self, params: Dict) -> Tuple[float, List[Dict]]:
        # A+ Gold tier
        rev_bal,    rev_t    = self.gold_rev.run(params,    start_balance=300.0)
        sniper_bal, sniper_t = self.gold_sniper.run(params, start_balance=250.0)
        master_bal, master_t = self.gold_master.run(params, start_balance=200.0)
        # B+ NQ tier
        bd_bal,     bd_t     = self.nq_bd.run(params,       start_balance=130.0)
        wick_bal,   wick_t   = self.nq_wick.run(params)
        wick_contrib = (wick_bal - 1000.0) * 0.05  # 5% of NQ Wick profit
        # B Silver tier
        silver_bal, silver_t = self.silver.run(params,      start_balance=70.0)

        total_bal = (rev_bal + sniper_bal + master_bal +
                     bd_bal + wick_contrib + silver_bal)
        total_trades = (rev_t + sniper_t + master_t +
                        bd_t + wick_t + silver_t)
        return total_bal, total_trades


# ─────────────────────────────────────────────────────────────────────────────
#  NQ WICK REVERSAL (mevcut backtester, isim degisti)
# ─────────────────────────────────────────────────────────────────────────────
class StrategyBacktesterLegacy:
    """Fast backtester for DEAP evaluation (NQ Wick Reversal)."""

    def __init__(self, oracle_df: pd.DataFrame):
        self.df = oracle_df

    def run(self, params: Dict) -> Tuple[float, List[Dict]]:
        wick_t = params.get("wick_threshold", 0.3)
        stop_atr = params.get("stop_atr", 1.5)
        tp = params.get("take_profit", 20.0)
        orb_bars = int(params.get("orb_bars", 3))
        vol_r = params.get("vol_ratio", 0.8)
        min_body = params.get("min_body", 2.0)
        base_lot = params.get("base_lot", 0.01)
        max_lot = params.get("max_lot", 0.05)

        df = self.df
        atr = df["atr"].values
        body = (df["close"] - df["open"]).abs().values
        upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)).values
        lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]).values
        is_bull = (df["close"] > df["open"]).values
        is_bear = (df["close"] < df["open"]).values

        body_s = np.where(body > 0, body, 1e-10)
        uw_r = upper_wick / body_s
        lw_r = lower_wick / body_s

        vol_ratio = df["volume_ratio"].values

        h = df.index.hour
        m = df.index.minute
        sess_min = h * 60 + m
        orb_start = 9 * 60 + 30
        orb_end = orb_start + orb_bars * 5
        is_orb = (sess_min >= orb_start) & (sess_min < orb_end)
        is_trade = sess_min >= orb_end

        close_v = df["close"].values
        high_v = df["high"].values
        low_v = df["low"].values

        balance = 1000.0
        trades = []
        last_date = None
        orb_h = None
        orb_l = None
        orb_est = False
        orb_cnt = 0
        traded = False

        for i in range(1, len(df)):
            cd = df.index[i].date()
            if last_date != cd:
                orb_h = None
                orb_l = None
                orb_est = False
                orb_cnt = 0
                traded = False
                last_date = cd

            if is_orb[i]:
                if orb_h is None:
                    orb_h = high_v[i]
                    orb_l = low_v[i]
                    orb_cnt = 1
                else:
                    orb_h = max(orb_h, high_v[i])
                    orb_l = min(orb_l, low_v[i])
                    orb_cnt += 1
                if orb_cnt >= orb_bars:
                    orb_est = True
                continue

            if not orb_est or not is_trade[i] or traded:
                continue

            if np.isnan(atr[i]) or atr[i] == 0 or body[i] == 0:
                continue

            vol_ok = vol_ratio[i] >= vol_r
            sig_dir = 0
            entry = 0.0
            sl = 0.0
            tp_p = 0.0
            lot = base_lot

            if (
                low_v[i] < orb_l
                and lw_r[i - 1] > wick_t
                and is_bull[i - 1]
                and body[i - 1] > min_body
                and vol_ok
            ):
                sig_dir = 1
                entry = close_v[i - 1]
                sl = orb_l - atr[i] * stop_atr
                tp_p = entry + tp
                vf = min(2.0, atr[i] / 15.0)
                lot = min(max_lot, base_lot * (1.0 + vf))

            elif (
                high_v[i] > orb_h
                and uw_r[i - 1] > wick_t
                and is_bear[i - 1]
                and body[i - 1] > min_body
                and vol_ok
            ):
                sig_dir = -1
                entry = close_v[i - 1]
                sl = orb_h + atr[i] * stop_atr
                tp_p = entry - tp
                vf = min(2.0, atr[i] / 15.0)
                lot = min(max_lot, base_lot * (1.0 + vf))

            if sig_dir != 0:
                traded = True
                # Vectorized exit check (much faster)
                if sig_dir == 1:
                    hits_sl = low_v[i+1:min(i+101, len(df))] <= sl
                    hits_tp = high_v[i+1:min(i+101, len(df))] >= tp_p
                else:
                    hits_sl = high_v[i+1:min(i+101, len(df))] >= sl
                    hits_tp = low_v[i+1:min(i+101, len(df))] <= tp_p

                sl_idx = np.argmax(hits_sl) if np.any(hits_sl) else -1
                tp_idx = np.argmax(hits_tp) if np.any(hits_tp) else -1

                if sl_idx == 0 and tp_idx == 0 and not np.any(hits_sl) and not np.any(hits_tp):
                    exit_p = close_v[-1]
                elif sl_idx >= 0 and tp_idx >= 0:
                    if hits_sl[sl_idx] and (not np.any(hits_tp) or sl_idx <= tp_idx):
                        exit_p = sl
                    else:
                        exit_p = tp_p
                elif sl_idx >= 0 and hits_sl[sl_idx]:
                    exit_p = sl
                elif tp_idx >= 0 and hits_tp[tp_idx]:
                    exit_p = tp_p
                else:
                    exit_p = close_v[-1]

                pnl = sig_dir * (exit_p - entry) * 20.0 * lot
                balance += pnl
                trades.append(
                    {
                        "time": str(df.index[i]),
                        "dir": "LONG" if sig_dir == 1 else "SHORT",
                        "entry": entry,
                        "exit": exit_p,
                        "pnl": pnl,
                        "bal": balance,
                    }
                )

        return balance, trades


class DEAPEvolutionEngine:
    """DEAP-based multi-asset strategy evolution with AI oracle filtering."""

    PARAM_BOUNDS = {
        # ── NQ Wick Reversal ─────────────────────────────────────────────────
        "wick_threshold": (0.05, 1.0),
        "stop_atr":       (0.3,  5.0),
        "take_profit":    (5.0,  60.0),
        "orb_bars":       (1,    12),
        "vol_ratio":      (0.3,  2.5),
        "min_body":       (0.5,  20.0),
        # ── Gold Reversion (A+) ──────────────────────────────────────────────
        "gold_streak":    (3.0,  8.0),
        "gold_rsi_limit": (20.0, 45.0),
        "gold_retrace":   (0.25, 0.75),
        "gold_stop_mult": (0.5,  3.0),
        "gold_risk_pct":  (1.0,  5.0),
        # ── NQ Breakdown (B+) ────────────────────────────────────────────────
        "nq_bd_lookback": (20.0, 100.0),
        "nq_bd_sl_mult":  (1.5,  6.0),
        "nq_bd_tp_mult":  (3.0,  12.0),
        "nq_bd_risk_pct": (0.5,  3.0),
        # ── Gold Sniper (A+) ─────────────────────────────────────────────────
        "gs_sl_points":   (8.0,  30.0),
        "gs_tp_mult":     (2.0,  6.0),
        "gs_risk_pct":    (1.0,  4.0),
        "gs_max_basket":  (1,    4),
        # ── Gold Master (A+) ─────────────────────────────────────────────────
        "gm_tp_points":   (40.0, 150.0),
        "gm_sl_points":   (20.0, 80.0),
        "gm_risk_pct":    (1.0,  4.0),
        "gm_momentum_thresh": (0.1, 1.5),
        # ── Silver Momentum (B) ──────────────────────────────────────────────
        "silver_adx_min":  (20.0, 40.0),
        "silver_risk_pct": (0.5,  2.0),
        # ── Oracle AI filtresi ────────────────────────────────────────────────
        "oracle_thresh":  (0.0,  0.9),
        "oracle_weight":  (0.0,  1.0),
    }

    INTEGER_PARAMS = {"orb_bars", "gs_max_basket"}

    def __init__(self, oracle_df: pd.DataFrame, config: Dict = None,
                 gold_df: pd.DataFrame = None):
        self.df       = oracle_df          # NQ oracle data
        self.gold_df  = gold_df            # Gold oracle data (opsiyonel)
        self.config   = config or {
            "population_size": 100,
            "n_generations":   50,
            "cxpb":            0.7,
            "mutpb":           0.2,
            "tournsize":       5,
            "hof_size":        10,
            "stagnation_limit":15,
        }

        # DEAP hizlandirma: son eval_days gun ile calis, tam validasyon icin full data kullan
        eval_days = self.config.get("eval_days", 90)
        if eval_days and len(oracle_df) > 0:
            cutoff     = oracle_df.index[-1] - pd.Timedelta(days=eval_days)
            nq_trim    = oracle_df[oracle_df.index >= cutoff]
            gold_trim  = (gold_df[gold_df.index  >= cutoff]
                          if gold_df is not None else None)
            logger.info(f"DEAP trim ({eval_days}d): NQ {len(nq_trim):,} | "
                        f"Gold {len(gold_trim) if gold_trim is not None else 0:,}")
        else:
            nq_trim, gold_trim = oracle_df, gold_df

        # Backtester secimi
        if gold_trim is not None:
            self.backtester = MultiAssetBacktester(gold_trim, nq_trim)
            logger.info("MultiAssetBacktester: Gold Rev+Sniper+Master (A+) | NQ BD+Wick (B+) | Silver Momentum (B)")
        else:
            self.backtester = StrategyBacktesterLegacy(nq_trim)
            logger.info("Fallback: NQ Wick Reversal only")

        self.fitness_calc = AdvancedFitnessCalculator()
        self.param_names  = list(self.PARAM_BOUNDS.keys())
        self.n_params     = len(self.param_names)

        self._setup_deap()

    def _setup_deap(self):
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        def create_ind():
            ind = []
            for name, (lo, hi) in self.PARAM_BOUNDS.items():
                if name in self.INTEGER_PARAMS:
                    ind.append(random.randint(int(lo), int(hi)))
                else:
                    ind.append(random.uniform(lo, hi))
            return ind

        self.toolbox.register("individual", tools.initIterate, creator.Individual, create_ind)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register(
            "mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2
        )
        self.toolbox.register("select", tools.selTournament, tournsize=self.config["tournsize"])
        self.toolbox.register("evaluate", self._evaluate)

    def _evaluate(self, individual):
        params = dict(zip(self.param_names, individual))

        # Clamp
        for name, (lo, hi) in self.PARAM_BOUNDS.items():
            if name in self.INTEGER_PARAMS:
                params[name] = int(max(lo, min(hi, params[name])))
            else:
                params[name] = max(lo, min(hi, params[name]))

        try:
            balance, trades = self.backtester.run(params)
        except Exception:
            return (-10.0,)

        fitness, _ = self.fitness_calc.calculate(trades)
        return (fitness,)

    def run(self) -> Dict:
        logger.info(
            f"DEAP Evolution: pop={self.config['population_size']}, "
            f"gen={self.config['n_generations']}"
        )

        pop = self.toolbox.population(n=self.config["population_size"])
        hof = tools.HallOfFame(self.config["hof_size"])
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        t0 = time.time()
        pop, log = algorithms.eaSimple(
            pop,
            self.toolbox,
            cxpb=self.config["cxpb"],
            mutpb=self.config["mutpb"],
            ngen=self.config["n_generations"],
            stats=stats,
            halloffame=hof,
            verbose=True,
        )
        elapsed = time.time() - t0
        logger.info(f"Evolution complete in {elapsed:.1f}s")

        # Extract best
        best = hof[0]
        best_params = dict(zip(self.param_names, list(best)))
        for name, (lo, hi) in self.PARAM_BOUNDS.items():
            if name == "orb_bars":
                best_params[name] = int(max(lo, min(hi, best_params[name])))
            else:
                best_params[name] = max(lo, min(hi, best_params[name]))

        # Full backtest with best params
        balance, trades = self.backtester.run(best_params)
        fitness, metrics = self.fitness_calc.calculate(trades)

        results = {
            "best_params": best_params,
            "fitness": fitness,
            "metrics": metrics,
            "balance": balance,
            "trades": trades,
            "log": log,
            "elapsed_seconds": elapsed,
        }

        # Save
        os.makedirs("outputs", exist_ok=True)
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path = f"outputs/deap_results_{ts}.json"
        save_data = {
            "best_params": best_params,
            "fitness": fitness,
            "metrics": metrics,
            "balance": balance,
            "elapsed_seconds": elapsed,
        }
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        logger.info(f"Results saved: {path}")

        return results
