# Order Block Detection
# Kurumsal emirlerin bıraktığı hacim + body size bölgelerini tespit

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderBlock:
    """Order Block bilgisi"""
    index: int                  # Bar index
    low: float                  # OB alt sınır
    high: float                 # OB üst sınır
    direction: str              # BULLISH, BEARISH
    strength: float             # 0-1 arası güç
    volume_ratio: float         # Normal hacme göre oran
    body_ratio: float           # Normal body'ye göre oran
    touched: bool = False       # Fiyat OB'ye dokundu mu?
    broken: bool = False        # OB kırıldı mı?


def detect_order_blocks(
    df: pd.DataFrame,
    lookback: int = 20,
    body_threshold: float = 1.5,
    volume_threshold: float = 1.5,
    min_strength: float = 0.5
) -> List[OrderBlock]:
    """
    Order Block Detection Algorithm
    
    Order Block Kriterleri:
    1. Büyük body (ortalamadan X kat büyük)
    2. Yüksek hacim (ortalamadan X kat yüksek)
    3. Güçlü momentum (önceki barlardan farklı yön)
    
    Args:
        df: OHLCV DataFrame (open, high, low, close, volume)
        lookback: Geriye bakış periyodu
        body_threshold: Body size threshold
        volume_threshold: Volume threshold
        min_strength: Minimum OB strength
        
    Returns:
        List[OrderBlock]: Tespit edilen order blocklar
    """
    
    if len(df) < lookback + 1:
        return []
    
    order_blocks: List[OrderBlock] = []
    
    # Normalize column names
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    # Body size hesapla
    df['body'] = abs(df['close'] - df['open'])
    
    # Yön belirle
    df['bullish'] = df['close'] > df['open']
    
    for i in range(lookback, len(df)):
        # Mevcut bar
        current = df.iloc[i]
        
        # Lookback penceresi
        window = df.iloc[i-lookback:i]
        
        # Ortalama body ve volume
        avg_body = window['body'].mean()
        avg_volume = window['volume'].mean() if 'volume' in df.columns else 1
        
        # Body ve volume oranları
        body_ratio = current['body'] / max(avg_body, 1e-10)
        volume_ratio = (current['volume'] / max(avg_volume, 1e-10)) if 'volume' in df.columns else 1.0
        
        # Order Block kriterleri
        is_large_body = body_ratio > body_threshold
        is_high_volume = volume_ratio > volume_threshold
        
        if is_large_body and is_high_volume:
            # Yön belirleme
            if current['bullish']:
                direction = "BULLISH"
                ob_low = min(current['open'], current['close'])
                ob_high = max(current['open'], current['close'])
            else:
                direction = "BEARISH"
                ob_low = min(current['open'], current['close'])
                ob_high = max(current['open'], current['close'])
            
            # Güç hesaplama (body + volume kombinasyonu)
            strength = min(1.0, (body_ratio / body_threshold + volume_ratio / volume_threshold) / 4)
            
            if strength >= min_strength:
                order_blocks.append(OrderBlock(
                    index=i,
                    low=ob_low,
                    high=ob_high,
                    direction=direction,
                    strength=strength,
                    volume_ratio=volume_ratio,
                    body_ratio=body_ratio
                ))
    
    return order_blocks


def check_ob_interaction(
    price: float,
    order_blocks: List[OrderBlock],
    touch_tolerance: float = 0.001
) -> Tuple[Optional[OrderBlock], str]:
    """
    Fiyatın Order Block ile etkileşimini kontrol et
    
    Args:
        price: Mevcut fiyat
        order_blocks: Aktif order blocklar
        touch_tolerance: Tolerance for touch detection
        
    Returns:
        Tuple[OrderBlock | None, interaction_type:str]
        interaction_type: "INSIDE", "TOUCHED", "BROKEN", "NONE"
    """
    for ob in order_blocks:
        if ob.broken:
            continue
            
        # Fiyat OB içinde mi?
        if ob.low <= price <= ob.high:
            ob.touched = True
            return ob, "INSIDE"
        
        # Fiyat OB'ye dokunuyor mu? (tolerance ile)
        ob_range = ob.high - ob.low
        touch_zone_low = ob.low - (ob_range * touch_tolerance)
        touch_zone_high = ob.high + (ob_range * touch_tolerance)
        
        if touch_zone_low <= price <= touch_zone_high:
            ob.touched = True
            return ob, "TOUCHED"
        
        # OB kırıldı mı?
        # Bullish OB kırılması: Fiyat OB altına düştü
        # Bearish OB kırılması: Fiyat OB üstüne çıktı
        if ob.direction == "BULLISH" and price < ob.low:
            ob.broken = True
            return ob, "BROKEN"
        elif ob.direction == "BEARISH" and price > ob.high:
            ob.broken = True
            return ob, "BROKEN"
    
    return None, "NONE"


def get_active_order_blocks(
    order_blocks: List[OrderBlock],
    current_index: int,
    max_age: int = 50
) -> List[OrderBlock]:
    """
    Aktif (kırılmamış ve yaşlı olmayan) order blockları getir
    
    Args:
        order_blocks: Tüm order blocklar
        current_index: Mevcut bar index
        max_age: Maximum OB yaşı
        
    Returns:
        List[OrderBlock]: Aktif OB'ler
    """
    active = []
    for ob in order_blocks:
        if not ob.broken and (current_index - ob.index) <= max_age:
            active.append(ob)
    return active


def calculate_ob_confluence(
    price: float,
    order_blocks: List[OrderBlock],
    direction_filter: Optional[str] = None
) -> float:
    """
    Fiyatın etrafındaki Order Block yoğunluğunu hesapla
    
    Birden fazla OB aynı bölgede = Güçlü destek/direnç
    
    Args:
        price: Mevcut fiyat
        order_blocks: Order blocklar
        direction_filter: Sadece belirli yöndeki OB'leri say
        
    Returns:
        float: Confluence skoru (0-1)
    """
    if not order_blocks:
        return 0.0
    
    nearby_strength = 0.0
    price_range = price * 0.02  # %2 etrafında
    
    for ob in order_blocks:
        if ob.broken:
            continue
            
        if direction_filter and ob.direction != direction_filter:
            continue
        
        # OB fiyatın yakınında mı?
        ob_mid = (ob.high + ob.low) / 2
        distance = abs(price - ob_mid)
        
        if distance <= price_range:
            # Yakın OB -> Confluence'a katkı
            proximity_factor = 1 - (distance / price_range)
            nearby_strength += ob.strength * proximity_factor
    
    return min(1.0, nearby_strength)


# === TEST ===
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Simüle OHLCV data
    np.random.seed(42)
    n = 200
    
    # Base price
    base = 100 + np.cumsum(np.random.randn(n) * 0.5)
    
    # OHLCV oluştur
    data = {
        'open': base + np.random.randn(n) * 0.3,
        'close': base + np.random.randn(n) * 0.3,
        'volume': np.abs(np.random.randn(n) * 100 + 500)
    }
    
    # Bazı barlara büyük hareketler ekle (Order Block oluşturmak için)
    for i in [50, 100, 150]:
        data['close'][i] = data['open'][i] + 3  # Büyük bullish
        data['volume'][i] *= 3
    for i in [75, 125, 175]:
        data['close'][i] = data['open'][i] - 3  # Büyük bearish  
        data['volume'][i] *= 3
    
    data['high'] = np.maximum(data['open'], data['close']) + np.random.rand(n) * 0.5
    data['low'] = np.minimum(data['open'], data['close']) - np.random.rand(n) * 0.5
    
    df = pd.DataFrame(data)
    
    # Order Blockları tespit et
    obs = detect_order_blocks(df, lookback=20, body_threshold=1.5, volume_threshold=1.5)
    
    print(f"Tespit edilen Order Block sayısı: {len(obs)}")
    for ob in obs:
        print(f"  Index: {ob.index}, Dir: {ob.direction}, "
              f"Range: {ob.low:.2f}-{ob.high:.2f}, Strength: {ob.strength:.2f}")
    
    # Plot
    plt.figure(figsize=(14, 8))
    plt.plot(df['close'], label='Close Price', alpha=0.7)
    
    # Order Blockları çiz
    for ob in obs:
        color = 'green' if ob.direction == 'BULLISH' else 'red'
        plt.axhspan(ob.low, ob.high, xmin=ob.index/len(df), xmax=1, 
                   color=color, alpha=0.2)
        plt.scatter(ob.index, df['close'].iloc[ob.index], 
                   color=color, s=100, marker='^' if ob.direction == 'BULLISH' else 'v')
    
    plt.title('Order Block Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
