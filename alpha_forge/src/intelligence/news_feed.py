"""
News Feed — Piyasa haberleri toplayıcısı
=========================================
MiroFish'e seed materyal olarak gönderilecek haberleri toplar.

Kaynaklar (hepsi ücretsiz):
  1. RSS Feeds     — Reuters, Bloomberg, FT, Investing.com
  2. Alpha Vantage — haber + sentiment (ücretsiz tier: 25 req/gün)
  3. NewsAPI       — ücretsiz tier: 100 req/gün

Filtreler:
  - Gold/XAUUSD haberler için
  - NQ/Nasdaq haberler için
  - Makro (Fed, CPI, GDP) haberleri için
"""

import os
import time
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, field

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    import feedparser
    FEEDPARSER_OK = True
except ImportError:
    FEEDPARSER_OK = False

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "../../../data/news_cache")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
AV_KEY      = os.getenv("ALPHA_VANTAGE_KEY", "")


@dataclass
class NewsItem:
    title:     str
    source:    str
    published: str
    sentiment: float = 0.0   # -1 bearish → +1 bullish (AV sağlarsa)
    url:       str   = ""


# RSS Feed'ler
RSS_FEEDS = {
    "macro": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.bbci.co.uk/news/business/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
    ],
    "gold": [
        "https://www.kitco.com/rss/kitconewsrss.xml",
        "https://feeds.reuters.com/reuters/commoditiesNews",
        "https://www.investing.com/rss/news_25.rss",  # Gold
    ],
    "nq": [
        "https://feeds.reuters.com/reuters/technologyNews",
        "https://www.investing.com/rss/news_14.rss",  # Tech stocks
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=QQQ&region=US&lang=en-US",
    ],
}

# Alpha Vantage News sorgu tickers
AV_TICKERS = {
    "XAUUSD": "GOLD",
    "NQ":     "QQQ,MSFT,NVDA",
    "XAGUSD": "SLV",
}


class NewsFeed:
    """
    Piyasa haber toplayıcısı.

    Kullanım:
        feed = NewsFeed()
        headlines = feed.get_headlines("XAUUSD", max_items=15)
        # → ["Gold surges on Fed pivot hopes", "Dollar weakens ahead of CPI", ...]
    """

    CACHE_TTL = 1800  # 30 dk

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

    def get_headlines(self, instrument: str, max_items: int = 15) -> List[str]:
        """
        instrument için son haberleri döndür.
        MiroFish seed materyali olarak kullanılır.
        """
        items = self.get_news_items(instrument, max_items)
        return [item.title for item in items]

    def get_news_items(self, instrument: str, max_items: int = 15) -> List[NewsItem]:
        """Ham NewsItem listesi döndür (başlık + sentiment + kaynak)."""
        cached = self._load_cache(instrument)
        if cached:
            return [NewsItem(**i) for i in cached[:max_items]]

        items: List[NewsItem] = []

        # 1. Alpha Vantage (sentiment dahil)
        av_items = self._fetch_alpha_vantage(instrument)
        items.extend(av_items)

        # 2. RSS Feeds
        rss_key = "gold" if instrument in ("XAUUSD", "XAGUSD") else (
                  "nq"   if instrument == "NQ" else "macro")
        rss_items = self._fetch_rss(rss_key)
        items.extend(rss_items)

        # 3. Makro haberler her zaman ekle
        macro_items = self._fetch_rss("macro")
        items.extend(macro_items[:5])

        # Dedupe ve sırala
        seen = set()
        unique = []
        for it in items:
            key = it.title[:60].lower()
            if key not in seen:
                seen.add(key)
                unique.append(it)

        unique = unique[:max_items]

        # Cache'e yaz
        self._save_cache(instrument, [
            {"title": i.title, "source": i.source,
             "published": i.published, "sentiment": i.sentiment, "url": i.url}
            for i in unique
        ])

        return unique

    def get_market_sentiment_score(self, instrument: str) -> float:
        """
        Haberlerin ortalama sentiment skorunu döndür.
        Alpha Vantage varsa gerçek skor, yoksa keyword-based.
        """
        items = self.get_news_items(instrument, max_items=20)
        if not items:
            return 0.0

        scores = [i.sentiment for i in items if i.sentiment != 0.0]
        if scores:
            return float(sum(scores) / len(scores))

        # Keyword fallback
        all_text = " ".join(i.title.lower() for i in items)
        bull_kw = ["rise", "gain", "surge", "rally", "bullish", "buy", "strong", "high"]
        bear_kw = ["fall", "drop", "plunge", "bearish", "sell", "weak", "low", "crash"]
        b = sum(1 for kw in bull_kw if kw in all_text)
        s = sum(1 for kw in bear_kw if kw in all_text)
        total = b + s
        return float((b - s) / total) if total > 0 else 0.0

    # ── Alpha Vantage ─────────────────────────────────────────────────────────

    def _fetch_alpha_vantage(self, instrument: str) -> List[NewsItem]:
        if not REQUESTS_OK or not AV_KEY:
            return []

        ticker = AV_TICKERS.get(instrument, "")
        if not ticker:
            return []

        try:
            r = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "NEWS_SENTIMENT",
                    "tickers": ticker,
                    "limit": 20,
                    "apikey": AV_KEY,
                },
                timeout=10,
            )
            if r.status_code != 200:
                return []

            data = r.json()
            feed = data.get("feed", [])
            items = []
            for art in feed:
                # AV sentiment: -1..+1
                ts = art.get("ticker_sentiment", [])
                sent = 0.0
                for t in ts:
                    if t.get("ticker") in ticker.split(","):
                        try:
                            sent = float(t.get("ticker_sentiment_score", 0))
                        except Exception:
                            pass
                items.append(NewsItem(
                    title     = art.get("title", ""),
                    source    = art.get("source", "AV"),
                    published = art.get("time_published", ""),
                    sentiment = sent,
                    url       = art.get("url", ""),
                ))
            logger.info(f"Alpha Vantage: {len(items)} news for {instrument}")
            return items
        except Exception as e:
            logger.debug(f"AV news fetch failed: {e}")
            return []

    # ── RSS ───────────────────────────────────────────────────────────────────

    def _fetch_rss(self, category: str) -> List[NewsItem]:
        if not FEEDPARSER_OK or not REQUESTS_OK:
            return []

        feeds = RSS_FEEDS.get(category, [])
        items = []
        for url in feeds:
            try:
                parsed = feedparser.parse(url)
                for entry in parsed.entries[:8]:
                    title = entry.get("title", "").strip()
                    if not title:
                        continue
                    published = entry.get("published", str(datetime.utcnow()))
                    items.append(NewsItem(
                        title=title,
                        source=parsed.feed.get("title", url),
                        published=published,
                    ))
            except Exception as e:
                logger.debug(f"RSS {url} failed: {e}")

        return items

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _cache_path(self, instrument: str) -> str:
        return os.path.join(CACHE_DIR, f"news_{instrument.lower()}.json")

    def _load_cache(self, instrument: str) -> Optional[List]:
        path = self._cache_path(instrument)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                d = json.load(f)
            if time.time() - d.get("_ts", 0) > self.CACHE_TTL:
                return None
            return d.get("items", [])
        except Exception:
            return None

    def _save_cache(self, instrument: str, items: List[Dict]):
        path = self._cache_path(instrument)
        try:
            with open(path, "w") as f:
                json.dump({"_ts": time.time(), "items": items}, f)
        except Exception as e:
            logger.debug(f"News cache save failed: {e}")
