"""
Decision Engine — 3 Zekanın Birleşimi + Kelly Criterion
========================================================

Katmanlar:
  1. DEAP Algo Signal  — fiyat/teknik (ağırlık: %50)
  2. MiroFish Score    — crowd simulation (ağırlık: %35)
  3. News Sentiment    — haber skoru (ağırlık: %15)

Conviction skoruna göre Kelly ile lot boyutu belirlenir:
  conviction > 0.7  → full Kelly lot
  conviction > 0.5  → half Kelly lot
  conviction < 0.3  → geç / FLAT

Çıktı: TradeDecision (instrument, action, lot, conviction, reasoning)
"""

import sys
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from alpha_forge.src.intelligence.mirofish_bridge import MiroFishBridge, MiroFishResult
from alpha_forge.src.intelligence.news_feed import NewsFeed

# Kelly Criterion
KELLY_PATH = os.path.join(os.path.dirname(__file__), "../../../kelly_garch_sizer")
sys.path.insert(0, KELLY_PATH)
try:
    from kelly_criterion import kelly_lot, compute_kelly
    KELLY_OK = True
except ImportError:
    KELLY_OK = False

logger = logging.getLogger(__name__)


@dataclass
class AlgoSignal:
    """DEAP algo'dan gelen ham sinyal."""
    instrument:  str
    action:      str    = "FLAT"   # "LONG" / "SHORT" / "FLAT"
    confidence:  float  = 0.5      # 0-1
    entry_price: float  = 0.0
    sl_points:   float  = 20.0
    tp_points:   float  = 60.0
    recent_trades: List[Dict] = field(default_factory=list)


@dataclass
class TradeDecision:
    """Decision Engine'in nihai kararı."""
    instrument:   str
    action:       str    = "FLAT"
    lot:          float  = 0.0
    conviction:   float  = 0.0
    entry_price:  float  = 0.0
    sl_points:    float  = 0.0
    tp_points:    float  = 0.0

    # Bileşen skorları
    algo_score:   float  = 0.0
    mirofish_score: float = 0.0
    news_score:   float  = 0.0

    reasoning:    str    = ""
    kelly_fraction: float = 0.0


class DecisionEngine:
    """
    3 Zeka + Kelly → Nihai karar.

    Kullanım:
        engine = DecisionEngine(balance=1000.0)
        signal = AlgoSignal(
            instrument="XAUUSD",
            action="LONG",
            confidence=0.75,
            entry_price=2380.0,
            sl_points=20.0,
            tp_points=60.0,
            recent_trades=[{"pnl": 50}, {"pnl": -20}, ...]
        )
        decision = engine.decide(signal)
    """

    # Katman ağırlıkları
    WEIGHTS = {
        "algo":      0.50,
        "mirofish":  0.35,
        "news":      0.15,
    }

    # Conviction eşikleri
    CONVICTION_TRADE  = 0.45   # bu eşiğin altında FLAT
    CONVICTION_FULL   = 0.70   # bu eşiğin üstünde full Kelly

    # Kelly parametreleri
    KELLY_FRACTION = 0.5       # half-Kelly
    KELLY_MAX      = 0.20      # max sermayenin %20'si

    def __init__(
        self,
        balance: float = 1000.0,
        mirofish_url: str = "http://localhost:5001",
        sim_rounds: int = 20,
        min_lot: float = 0.01,
        max_lot: float = 2.0,
        point_value: float = 100.0,
    ):
        self.balance     = balance
        self.sim_rounds  = sim_rounds
        self.min_lot     = min_lot
        self.max_lot     = max_lot
        self.point_value = point_value

        self.mirofish = MiroFishBridge(base_url=mirofish_url)
        self.news     = NewsFeed()

    # ── Ana karar metodu ──────────────────────────────────────────────────────

    def decide(self, signal: AlgoSignal) -> TradeDecision:
        """
        DEAP sinyalini alır, MiroFish + News ile zenginleştirir,
        Kelly ile lot hesaplar, nihai kararı döndürür.
        """
        inst = signal.instrument

        # ── 1. DEAP algo skoru ────────────────────────────────────────────────
        algo_score = self._algo_to_score(signal)

        # ── 2. MiroFish (crowd simulation) ───────────────────────────────────
        headlines   = self.news.get_headlines(inst, max_items=15)
        market_ctx  = {
            "instrument": inst,
            "price":      signal.entry_price,
            "trend":      signal.action.lower() if signal.action != "FLAT" else "neutral",
        }
        mf_result   = self.mirofish.query(headlines, market_ctx, self.sim_rounds)
        mf_score    = mf_result.score

        # ── 3. News sentiment skoru ───────────────────────────────────────────
        news_score  = self.news.get_market_sentiment_score(inst)

        # ── 4. Ağırlıklı conviction ───────────────────────────────────────────
        raw_conviction = (
            algo_score  * self.WEIGHTS["algo"] +
            mf_score    * self.WEIGHTS["mirofish"] +
            news_score  * self.WEIGHTS["news"]
        )
        conviction = float(np.clip(abs(raw_conviction), 0.0, 1.0))

        # Yön: raw_conviction'ın işareti
        direction = "LONG" if raw_conviction > 0 else "SHORT"

        # DEAP ile MiroFish zıt yöndeyse conviction cez
        if signal.action != "FLAT" and mf_result.action != "FLAT":
            if signal.action != mf_result.action:
                conviction *= 0.6
                logger.info(f"[{inst}] DEAP vs MiroFish divergence — conviction -40%")

        # Conviction eşiği kontrolü
        if conviction < self.CONVICTION_TRADE or signal.action == "FLAT":
            return TradeDecision(
                instrument=inst, action="FLAT",
                conviction=conviction,
                algo_score=algo_score, mirofish_score=mf_score, news_score=news_score,
                reasoning=f"Conviction {conviction:.2f} < {self.CONVICTION_TRADE} eşiği"
            )

        # ── 5. Kelly Criterion lot hesabı ─────────────────────────────────────
        lot = self._kelly_lot(
            signal.recent_trades,
            signal.sl_points,
            conviction,
        )

        # ── 6. Karar ─────────────────────────────────────────────────────────
        reasoning = (
            f"Algo:{algo_score:+.2f}({signal.action}) | "
            f"MiroFish:{mf_score:+.2f}({mf_result.action}) | "
            f"News:{news_score:+.2f} | "
            f"Conviction:{conviction:.2f} | "
            f"Lot:{lot:.3f}"
        )
        logger.info(f"[{inst}] DECISION: {direction} | {reasoning}")

        return TradeDecision(
            instrument    = inst,
            action        = direction,
            lot           = lot,
            conviction    = round(conviction, 3),
            entry_price   = signal.entry_price,
            sl_points     = signal.sl_points,
            tp_points     = signal.tp_points,
            algo_score    = round(algo_score, 3),
            mirofish_score= round(mf_score, 3),
            news_score    = round(news_score, 3),
            reasoning     = reasoning,
        )

    # ── Yardımcı metotlar ─────────────────────────────────────────────────────

    @staticmethod
    def _algo_to_score(signal: AlgoSignal) -> float:
        """DEAP sinyal → [-1, +1] skor."""
        if signal.action == "FLAT":
            return 0.0
        direction = 1.0 if signal.action == "LONG" else -1.0
        return direction * float(np.clip(signal.confidence, 0.0, 1.0))

    def _kelly_lot(
        self,
        recent_trades: List[Dict],
        sl_points: float,
        conviction: float,
    ) -> float:
        """Kelly Criterion ile lot boyutu hesapla."""
        if not KELLY_OK or not recent_trades or sl_points <= 0:
            # Fallback: conviction × max_lot × 0.5
            return float(np.clip(
                conviction * self.max_lot * 0.5,
                self.min_lot, self.max_lot
            ))

        # Conviction bazlı Kelly fraksiyonu
        kelly_frac = self.KELLY_FRACTION
        if conviction >= self.CONVICTION_FULL:
            kelly_frac = self.KELLY_FRACTION        # full half-Kelly
        else:
            kelly_frac = self.KELLY_FRACTION * (conviction / self.CONVICTION_FULL)

        lot = kelly_lot(
            capital       = self.balance,
            sl_pts        = sl_points,
            point_value   = self.point_value,
            trades        = recent_trades,
            fraction      = kelly_frac,
            max_fraction  = self.KELLY_MAX,
            min_lot       = self.min_lot,
            max_lot       = self.max_lot,
        )

        # Conviction < 0.5 ise lotı yarıya indir
        if conviction < 0.5:
            lot *= 0.5

        return round(float(np.clip(lot, self.min_lot, self.max_lot)), 3)

    def update_balance(self, new_balance: float):
        """Her trade sonrası balance güncelle (Kelly dinamik çalışsın)."""
        self.balance = new_balance
