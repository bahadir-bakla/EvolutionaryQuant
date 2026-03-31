"""
=============================================================
KATMAN 4+5: POZİSYON YÖNETİCİSİ
=============================================================
- Kelly Criterion (Hurst + Kalman feedback ile)
- Compounding (sermaye büyüdükçe lot büyür)
- Basket / DCA (sadece trend'de, chop'ta yasak)
- Feedback loop (her işlem sonrası Kelly güncellenir)
=============================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


# ─────────────────────────────────────────────────────────
# KELLY CRITERION
# ─────────────────────────────────────────────────────────

class InstitutionalKelly:
    """
    Dinamik Kelly — Hurst + Kalman ile feedback.

    f* = W - (1-W) / R
    Yarım Kelly kullanılır (güvenlik için)
    Hurst ve Kalman rejim durumuna göre ölçeklenir.
    """

    def __init__(
        self,
        safe_fraction:   float = 0.10,   # Yeterli geçmiş yokken %10
        max_fraction:    float = 0.25,   # Maksimum %25 pozisyon
        min_history:     int   = 15,     # En az 15 işlem gerekli
    ):
        self.safe_fraction = safe_fraction
        self.max_fraction  = max_fraction
        self.min_history   = min_history

        # Trade geçmişi
        self._wins         = 0
        self._losses       = 0
        self._total_profit = 0.0
        self._total_loss   = 0.0

    def record(self, pnl: float, risk: float):
        """Her işlem sonrası Kelly'i güncelle"""
        if pnl > 0:
            self._wins         += 1
            self._total_profit += pnl
        else:
            self._losses       += 1
            self._total_loss   += abs(pnl) if pnl < 0 else risk

    def fraction(self, hurst: float = 0.5, kalman_acc: float = 0.0) -> float:
        """
        Optimal pozisyon fraksiyonu döndürür.
        Hurst ve Kalman ile ölçeklenir.
        """
        total = self._wins + self._losses
        if total < self.min_history:
            return self.safe_fraction

        win_rate = self._wins / total
        avg_win  = self._total_profit / self._wins   if self._wins   > 0 else 0
        avg_loss = self._total_loss   / self._losses if self._losses > 0 else 1

        if avg_loss < 1e-10:
            return self.max_fraction

        r_ratio  = avg_win / avg_loss
        if r_ratio <= 0:
            return 0.01

        # Kelly formülü
        kelly    = win_rate - (1 - win_rate) / r_ratio
        half_k   = kelly / 2.0   # Half Kelly

        if half_k <= 0:
            return 0.01

        # ── Regime feedback ─────────────────────────────
        # Hurst > 0.6: güçlü trend → agresif
        # Hurst < 0.4: chop → çok küçük
        if hurst > 0.65:
            regime_mult = 1.4
        elif hurst > 0.55:
            regime_mult = 1.1
        elif hurst > 0.45:
            regime_mult = 0.7
        else:
            regime_mult = 0.2   # Chop → neredeyse dur

        # Kalman ivmesi pozitifse biraz daha agresif
        if kalman_acc > 0.02:
            mom_mult = 1.2
        elif kalman_acc > 0:
            mom_mult = 1.0
        else:
            mom_mult = 0.8

        final = half_k * regime_mult * mom_mult
        return float(np.clip(final, 0.01, self.max_fraction))

    def lot_size(
        self,
        equity:     float,
        hurst:      float = 0.5,
        kalman_acc: float = 0.0,
        base_equity: float = 1000.0,
    ) -> float:
        """
        Compounding dahil lot büyüklüğü.
        Sermaye büyüdükçe lot da büyür.
        """
        frac   = self.fraction(hurst, kalman_acc)
        # Global ölçek: başlangıç sermayesine göre büyüme
        scale  = max(0.5, equity / base_equity)
        return round(equity * frac, 2)

    @property
    def stats(self) -> dict:
        total = self._wins + self._losses
        wr    = self._wins / total if total > 0 else 0
        return {
            'trades':   total,
            'win_rate': wr,
            'wins':     self._wins,
            'losses':   self._losses,
        }


# ─────────────────────────────────────────────────────────
# BASKET / DCA YÖNETİCİSİ
# ─────────────────────────────────────────────────────────

@dataclass
class BasketTrade:
    entry_price: float
    lot:         float
    direction:   int     # 1=long, -1=short
    layer:       int     # 1=ilk giriş, 2=DCA1, 3=DCA2...
    reason:      str = ""


class BasketManager:
    """
    Institutional sepetleme / DCA yönetimi.

    KURALLAR (senin dediklerin):
    - SADECE trend'de açılır (Hurst > 0.55 + chop < 61.8)
    - Max 3 katman (4. katman = hesap patlar)
    - Her katman önceki * 1.0 lot (eşit lot, martingale değil)
    - DCA sadece OB/FVG bölgesinde eklenir
    - Haber öncesi yeni katman açılmaz
    - Asya saatlerinde açılmaz
    """

    def __init__(
        self,
        max_layers:       int   = 3,
        dca_step_pct:     float = 0.008,  # %0.8 ters gidince DCA
        target_profit_pct:float = 0.015,  # %1.5 kâr hedefi (sepet)
        stop_loss_pct:    float = 0.025,  # %2.5 maksimum zarar (tüm sepet)
    ):
        self.max_layers        = max_layers
        self.dca_step_pct      = dca_step_pct
        self.target_profit_pct = target_profit_pct
        self.stop_loss_pct     = stop_loss_pct

        self.basket:    List[BasketTrade] = []
        self.direction: int               = 0
        self.is_open:   bool              = False

    def open(
        self,
        price:  float,
        lot:    float,
        direction: int,
        reason: str = "INITIAL",
    ) -> bool:
        """İlk giriş"""
        if self.is_open:
            return False
        self.basket    = [BasketTrade(price, lot, direction, 1, reason)]
        self.direction = direction
        self.is_open   = True
        return True

    def should_dca(
        self,
        current_price: float,
        is_trending:   bool,
        ob_or_fvg:     bool,
    ) -> bool:
        """DCA koşulları: trend var + OB/FVG bölgesi + yeterli mesafe"""
        if not self.is_open or len(self.basket) >= self.max_layers:
            return False
        if not is_trending:
            return False   # Chop'ta DCA yasak
        if not ob_or_fvg:
            return False   # Sadece yapısal bölgelerde

        last_entry = self.basket[-1].entry_price
        move_pct   = (current_price - last_entry) / last_entry

        if self.direction == 1:   # Long'duk, düştü mü?
            return move_pct <= -self.dca_step_pct
        else:                     # Short'tuk, yükseldi mi?
            return move_pct >= self.dca_step_pct

    def add_layer(self, price: float, lot: float, reason: str = "DCA") -> bool:
        """DCA katmanı ekle"""
        if len(self.basket) >= self.max_layers:
            return False
        layer = len(self.basket) + 1
        self.basket.append(BasketTrade(price, lot, self.direction, layer, reason))
        return True

    def avg_entry(self) -> float:
        """Ağırlıklı ortalama giriş fiyatı"""
        if not self.basket:
            return 0.0
        total_lot   = sum(t.lot for t in self.basket)
        total_value = sum(t.lot * t.entry_price for t in self.basket)
        return total_value / (total_lot + 1e-10)

    def total_lot(self) -> float:
        return sum(t.lot for t in self.basket)

    def unrealized_pnl(self, current_price: float) -> float:
        """Anlık kâr/zarar"""
        pnl = 0.0
        for t in self.basket:
            if t.direction == 1:
                pnl += t.lot * (current_price - t.entry_price)
            else:
                pnl += t.lot * (t.entry_price - current_price)
        return pnl

    def should_close(self, current_price: float, equity: float) -> Optional[str]:
        """
        Sepeti kapat mı?
        Returns: 'TARGET' | 'STOP' | 'TRAIL' | None
        """
        if not self.is_open:
            return None

        avg    = self.avg_entry()
        pnl    = self.unrealized_pnl(current_price)
        pnl_pct = pnl / (equity + 1e-10)

        # Kâr hedefi
        if pnl_pct >= self.target_profit_pct:
            return 'TARGET'

        # Stop loss
        if pnl_pct <= -self.stop_loss_pct:
            return 'STOP'

        return None

    def close(self) -> List[BasketTrade]:
        """Sepeti kapat, işlemleri döndür"""
        closed     = self.basket.copy()
        self.basket    = []
        self.direction = 0
        self.is_open   = False
        return closed
