"""
MiroFish Bridge — alpha_forge ↔ MiroFish API
=============================================
MiroFish gerçek API akışına göre entegrasyon.

Gerçek API akışı (5001 portu):
  1. POST /api/graph/ontology/generate  → project_id (metin dosyası yükle)
  2. POST /api/graph/build              → task_id (graph inşa et, bekle)
  3. POST /api/simulation/create        → simulation_id
  4. POST /api/simulation/prepare       → task_id (agent profilleri üret, bekle)
  5. POST /api/simulation/start         → simülasyonu başlat
  6. GET  /api/simulation/<id>/run-status → tamamlanana kadar bekle
  7. POST /api/report/generate          → task_id (rapor üret, bekle)
  8. GET  /api/report/<report_id>       → sonuç raporu al

Dönen skor:
  +1.0  → crowd çok bullish
   0.0  → nötr
  -1.0  → crowd çok bearish
"""

import os
import time
import json
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

logger = logging.getLogger(__name__)

MIROFISH_BASE = os.getenv("MIROFISH_URL", "http://localhost:5001")
CACHE_DIR = os.path.join(os.path.dirname(__file__), "../../../data/mirofish_cache")


@dataclass
class MiroFishResult:
    score:       float = 0.0       # -1.0 bearish → +1.0 bullish
    conviction:  float = 0.0       # 0 = no data, 1 = high
    action:      str   = "FLAT"    # "LONG" / "SHORT" / "FLAT"
    summary:     str   = ""        # MiroFish rapor özeti
    sim_rounds:  int   = 0
    cached:      bool  = False
    timestamp:   str   = field(default_factory=lambda: datetime.utcnow().isoformat())


class MiroFishBridge:
    """
    alpha_forge ile MiroFish arasındaki köprü.

    Kullanım:
        bridge = MiroFishBridge()
        news   = ["Gold hits ATH as dollar weakens", "Fed holds rates steady"]
        market = {"instrument": "XAUUSD", "price": 2380, "trend": "bullish"}
        result = bridge.query(news, market, sim_rounds=20)
    """

    CACHE_TTL        = 3600 * 4  # 4 saat
    POLL_INTERVAL    = 3         # saniye
    MAX_WAIT_GRAPH   = 180       # graph build için maks bekleme (sn)
    MAX_WAIT_PREPARE = 300       # prepare için maks bekleme (sn)
    MAX_WAIT_RUN     = 300       # simülasyon çalışması için maks bekleme (sn)
    MAX_WAIT_REPORT  = 120       # rapor üretimi için maks bekleme (sn)

    def __init__(self, base_url: str = MIROFISH_BASE, cache_dir: str = CACHE_DIR):
        self.base = base_url.rstrip("/")
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    # ── Ana metot ─────────────────────────────────────────────────────────────

    def query(
        self,
        headlines: List[str],
        market_ctx: Dict,
        sim_rounds: int = 20,
    ) -> MiroFishResult:
        """
        headlines : son N haber başlığı listesi
        market_ctx: {"instrument": "XAUUSD", "price": 2380, "trend": "up", ...}
        sim_rounds: kaç round simüle edilsin (max_rounds olarak geçilir)
        """
        if not REQUESTS_OK:
            logger.warning("requests modülü yok, MiroFish devre dışı")
            return MiroFishResult()

        # Cache kontrolü
        cache_key = self._cache_key(headlines, market_ctx)
        cached = self._load_cache(cache_key)
        if cached:
            cached.pop("_ts", None)
            return MiroFishResult(**cached, cached=True)

        # MiroFish ayakta mı?
        if not self._health_check():
            logger.warning("MiroFish offline — FLAT döndürülüyor")
            return MiroFishResult()

        instrument = market_ctx.get("instrument", "MARKET")
        seed_text  = self._build_seed(headlines, market_ctx)

        # Adım 1: Ontology + Project oluştur
        project_id = self._step1_create_project(seed_text, instrument, market_ctx)
        if not project_id:
            return MiroFishResult()

        # Adım 2: Graph inşa et ve bekle
        if not self._step2_build_graph(project_id):
            return MiroFishResult()

        # Adım 3: Simülasyon oluştur
        simulation_id = self._step3_create_simulation(project_id)
        if not simulation_id:
            return MiroFishResult()

        # Adım 4: Hazırlan ve bekle
        if not self._step4_prepare_simulation(simulation_id):
            return MiroFishResult()

        # Adım 5: Simülasyonu başlat ve bekle
        if not self._step5_run_simulation(simulation_id, sim_rounds):
            return MiroFishResult()

        # Adım 6: Rapor üret ve al
        result = self._step6_generate_report(simulation_id)

        # Cache'e yaz
        if result.conviction > 0:
            self._save_cache(cache_key, {
                "score":      result.score,
                "conviction": result.conviction,
                "action":     result.action,
                "summary":    result.summary,
                "sim_rounds": result.sim_rounds,
                "timestamp":  result.timestamp,
            })

        return result

    # ── Adım 1: Proje ve Ontology ─────────────────────────────────────────────

    def _step1_create_project(self, seed_text: str, instrument: str, ctx: Dict) -> Optional[str]:
        """Seed metni TXT olarak yükle, ontology oluştur → project_id döndür."""
        try:
            import io
            trend = ctx.get("trend", "neutral")
            sim_req = (
                f"Analyze how diverse market participants — retail traders, hedge fund managers, "
                f"commodity analysts, news traders, algorithmic systems — would react to this "
                f"{instrument} market situation. Determine crowd sentiment: bullish or bearish? "
                f"What is the likely price direction in next 24-48 hours?"
            )

            txt_bytes = seed_text.encode("utf-8")
            files = {
                "files": (f"{instrument}_briefing.txt", io.BytesIO(txt_bytes), "text/plain"),
            }
            data = {
                "simulation_requirement": sim_req,
                "project_name": f"{instrument} Market Analysis {datetime.utcnow().strftime('%Y%m%d_%H%M')}",
            }

            r = requests.post(
                f"{self.base}/api/graph/ontology/generate",
                files=files,
                data=data,
                timeout=120,
            )
            if r.status_code == 200:
                resp = r.json()
                if resp.get("success"):
                    pid = resp["data"]["project_id"]
                    logger.info(f"MiroFish project created: {pid}")
                    return pid
            logger.warning(f"MiroFish ontology/generate failed: {r.status_code} {r.text[:300]}")
            return None
        except Exception as e:
            logger.warning(f"MiroFish step1 error: {e}")
            return None

    # ── Adım 2: Graph Build ───────────────────────────────────────────────────

    def _step2_build_graph(self, project_id: str) -> bool:
        """Graph inşa et, task tamamlanana kadar bekle."""
        try:
            r = requests.post(
                f"{self.base}/api/graph/build",
                json={"project_id": project_id},
                timeout=30,
            )
            if r.status_code != 200:
                logger.warning(f"MiroFish graph/build failed: {r.status_code}")
                return False
            resp = r.json()
            if not resp.get("success"):
                logger.warning(f"MiroFish graph/build error: {resp.get('error')}")
                return False

            task_id = resp["data"].get("task_id")
            if not task_id:
                logger.warning("MiroFish graph/build: no task_id returned")
                return False

            logger.info(f"MiroFish graph build started: task={task_id}")
            return self._poll_task(f"{self.base}/api/graph/task/{task_id}", self.MAX_WAIT_GRAPH, "graph build")

        except Exception as e:
            logger.warning(f"MiroFish step2 error: {e}")
            return False

    # ── Adım 3: Simulation Create ─────────────────────────────────────────────

    def _step3_create_simulation(self, project_id: str) -> Optional[str]:
        """Simulation oluştur → simulation_id döndür."""
        try:
            r = requests.post(
                f"{self.base}/api/simulation/create",
                json={"project_id": project_id},
                timeout=30,
            )
            if r.status_code == 200:
                resp = r.json()
                if resp.get("success"):
                    sid = resp["data"]["simulation_id"]
                    logger.info(f"MiroFish simulation created: {sid}")
                    return sid
            logger.warning(f"MiroFish simulation/create failed: {r.status_code} {r.text[:200]}")
            return None
        except Exception as e:
            logger.warning(f"MiroFish step3 error: {e}")
            return None

    # ── Adım 4: Simulation Prepare ────────────────────────────────────────────

    def _step4_prepare_simulation(self, simulation_id: str) -> bool:
        """Agent profilleri üret, hazırlanana kadar bekle."""
        try:
            r = requests.post(
                f"{self.base}/api/simulation/prepare",
                json={"simulation_id": simulation_id},
                timeout=30,
            )
            if r.status_code != 200:
                logger.warning(f"MiroFish simulation/prepare failed: {r.status_code}")
                return False
            resp = r.json()
            if not resp.get("success"):
                logger.warning(f"MiroFish simulation/prepare error: {resp.get('error')}")
                return False

            # Zaten hazır mı?
            if resp["data"].get("already_prepared") or resp["data"].get("status") == "ready":
                logger.info("MiroFish simulation already prepared")
                return True

            task_id = resp["data"].get("task_id")
            if not task_id:
                # task_id yoksa poll simulation status
                return self._poll_simulation_ready(simulation_id, self.MAX_WAIT_PREPARE)

            logger.info(f"MiroFish prepare started: task={task_id}")
            return self._poll_task(
                f"{self.base}/api/simulation/prepare/status",
                self.MAX_WAIT_PREPARE,
                "prepare",
                post_body={"task_id": task_id, "simulation_id": simulation_id},
            )

        except Exception as e:
            logger.warning(f"MiroFish step4 error: {e}")
            return False

    # ── Adım 5: Simulation Run ────────────────────────────────────────────────

    def _step5_run_simulation(self, simulation_id: str, max_rounds: int) -> bool:
        """Simülasyonu başlat, tamamlanana kadar bekle."""
        try:
            r = requests.post(
                f"{self.base}/api/simulation/start",
                json={
                    "simulation_id": simulation_id,
                    "platform": "parallel",
                    "max_rounds": max_rounds,
                },
                timeout=30,
            )
            if r.status_code != 200:
                logger.warning(f"MiroFish simulation/start failed: {r.status_code} {r.text[:200]}")
                return False
            resp = r.json()
            if not resp.get("success"):
                logger.warning(f"MiroFish simulation/start error: {resp.get('error')}")
                return False

            logger.info(f"MiroFish simulation running: {simulation_id}")
            return self._poll_simulation_done(simulation_id, self.MAX_WAIT_RUN)

        except Exception as e:
            logger.warning(f"MiroFish step5 error: {e}")
            return False

    # ── Adım 6: Report Generate ───────────────────────────────────────────────

    def _step6_generate_report(self, simulation_id: str) -> MiroFishResult:
        """Rapor üret, bekle, skora çevir."""
        try:
            r = requests.post(
                f"{self.base}/api/report/generate",
                json={"simulation_id": simulation_id},
                timeout=30,
            )
            if r.status_code != 200:
                logger.warning(f"MiroFish report/generate failed: {r.status_code}")
                return MiroFishResult()
            resp = r.json()
            if not resp.get("success"):
                return MiroFishResult()

            # Zaten var mı?
            if resp["data"].get("already_generated") or resp["data"].get("status") == "completed":
                report_id = resp["data"].get("report_id")
                if report_id:
                    return self._fetch_report(report_id)

            task_id = resp["data"].get("task_id")
            if not task_id:
                return MiroFishResult()

            # Rapor tamamlanana kadar bekle
            deadline = time.time() + self.MAX_WAIT_REPORT
            while time.time() < deadline:
                sr = requests.post(
                    f"{self.base}/api/report/generate/status",
                    json={"task_id": task_id},
                    timeout=10,
                )
                if sr.status_code == 200:
                    sd = sr.json().get("data", {})
                    status = sd.get("status", "")
                    if status in ("completed", "success"):
                        report_id = sd.get("report_id")
                        if report_id:
                            return self._fetch_report(report_id)
                        break
                    elif status in ("failed", "error"):
                        logger.warning(f"MiroFish report generation failed")
                        break
                time.sleep(self.POLL_INTERVAL)

            # Fallback: simulation_id üzerinden raporu bul
            cr = requests.get(
                f"{self.base}/api/report/check/{simulation_id}",
                timeout=10,
            )
            if cr.status_code == 200:
                cd = cr.json().get("data", {})
                report_id = cd.get("report_id")
                if report_id:
                    return self._fetch_report(report_id)

            return MiroFishResult()

        except Exception as e:
            logger.warning(f"MiroFish step6 error: {e}")
            return MiroFishResult()

    def _fetch_report(self, report_id: str) -> MiroFishResult:
        """Raporu al ve skora çevir."""
        try:
            r = requests.get(f"{self.base}/api/report/{report_id}", timeout=30)
            if r.status_code != 200:
                return MiroFishResult()
            data = r.json().get("data", {})

            # Outline summary — en güvenilir kısa özet (LLM token hatası olmaz)
            outline = data.get("outline") or {}
            summary = outline.get("summary", "")

            # Markdown içeriği (rapor başarıyla üretilmişse)
            if not summary:
                summary = data.get("markdown_content", "")

            # Sections içinden topla
            if not summary:
                for sec in (outline.get("sections") or data.get("sections", [])):
                    if isinstance(sec, dict):
                        summary += sec.get("content", "") + " "
                summary = summary.strip()

            # Son çare: diğer alanlar
            if not summary:
                summary = (
                    data.get("summary") or
                    data.get("conclusion") or
                    data.get("report") or ""
                )

            score = self._parse_score_from_report(data, summary)
            action = "LONG" if score > 0.15 else ("SHORT" if score < -0.15 else "FLAT")

            logger.info(f"MiroFish report parsed: score={score:.3f} action={action}")
            return MiroFishResult(
                score=round(score, 3),
                conviction=min(0.9, abs(score) + 0.3),
                action=action,
                summary=summary[:500],
                sim_rounds=data.get("sim_rounds", 0),
            )
        except Exception as e:
            logger.warning(f"MiroFish fetch_report error: {e}")
            return MiroFishResult()

    # ── Poll yardımcıları ─────────────────────────────────────────────────────

    def _poll_task(
        self,
        url: str,
        max_wait: int,
        label: str,
        post_body: Optional[Dict] = None,
    ) -> bool:
        """GET veya POST ile task tamamlanana kadar bekle."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                if post_body:
                    r = requests.post(url, json=post_body, timeout=10)
                else:
                    r = requests.get(url, timeout=10)

                if r.status_code == 200:
                    d = r.json().get("data", {})
                    status = d.get("status", "")
                    if status in ("completed", "success", "ready", "done"):
                        logger.info(f"MiroFish {label} completed")
                        return True
                    elif status in ("failed", "error"):
                        logger.warning(f"MiroFish {label} failed: {d.get('error', '')}")
                        return False
            except Exception:
                pass
            time.sleep(self.POLL_INTERVAL)

        logger.warning(f"MiroFish {label} timeout after {max_wait}s")
        return False

    def _poll_simulation_ready(self, simulation_id: str, max_wait: int) -> bool:
        """Simulation READY durumuna gelene kadar bekle."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                r = requests.get(
                    f"{self.base}/api/simulation/{simulation_id}",
                    timeout=10,
                )
                if r.status_code == 200:
                    status = r.json().get("data", {}).get("status", "")
                    if status == "ready":
                        return True
                    elif status in ("failed", "error"):
                        return False
            except Exception:
                pass
            time.sleep(self.POLL_INTERVAL)
        return False

    def _poll_simulation_done(self, simulation_id: str, max_wait: int) -> bool:
        """Simülasyon bitene kadar bekle."""
        deadline = time.time() + max_wait
        while time.time() < deadline:
            try:
                r = requests.get(
                    f"{self.base}/api/simulation/{simulation_id}/run-status",
                    timeout=10,
                )
                if r.status_code == 200:
                    d = r.json().get("data", {})
                    runner = d.get("runner_status", d.get("status", ""))
                    if runner in ("completed", "finished", "stopped", "done"):
                        logger.info(f"MiroFish simulation done: {simulation_id}")
                        return True
                    elif runner in ("failed", "error", "crashed"):
                        logger.warning(f"MiroFish simulation crashed: {simulation_id}")
                        return False
            except Exception:
                pass
            time.sleep(self.POLL_INTERVAL)

        # Timeout sonrası yine de rapor deneyelim (kısmen çalışmış olabilir)
        logger.warning(f"MiroFish simulation timeout — attempting report anyway")
        return True

    # ── Health check ──────────────────────────────────────────────────────────

    def _health_check(self) -> bool:
        try:
            r = requests.get(f"{self.base}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            try:
                r = requests.get(self.base, timeout=5)
                return r.status_code < 500
            except Exception:
                return False

    # ── Seed materyal builder ─────────────────────────────────────────────────

    @staticmethod
    def _build_seed(headlines: List[str], ctx: Dict) -> str:
        instrument = ctx.get("instrument", "Gold")
        price      = ctx.get("price", "N/A")
        trend      = ctx.get("trend", "neutral")
        rsi        = ctx.get("rsi", "N/A")
        atr        = ctx.get("atr", "N/A")

        news_block = "\n".join(f"- {h}" for h in headlines[:15])

        return f"""MARKET INTELLIGENCE BRIEFING
=============================
Instrument  : {instrument}
Current Price: {price}
Short-term Trend: {trend}
RSI: {rsi} | ATR: {atr}
Timestamp   : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

LATEST NEWS HEADLINES:
{news_block}

ANALYSIS OBJECTIVE:
Simulate how a diverse crowd of market participants — retail traders,
hedge fund managers, commodity analysts, news traders, algorithmic systems —
would interpret this information and position themselves in {instrument}.
Focus on: crowd sentiment, likely price action in next 24-48 hours,
institutional vs retail divergence, and key risk events.
""".strip()

    # ── Score parser ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_score_from_report(data: Dict, summary: str) -> float:
        """MiroFish raporundan sayısal skor çıkar."""
        for key in ("sentiment_score", "score", "bullish_score", "prediction_score",
                    "bullish_ratio", "sentiment"):
            val = data.get(key)
            if val is not None:
                try:
                    fval = float(val)
                    return fval / 100.0 if abs(fval) > 1 else fval
                except Exception:
                    pass

        text = (summary + " " + str(data)).lower()
        bullish_kw = [
            "bullish", "upward", "buying", "long", "rise", "positive",
            "strong demand", "buy pressure", "rally", "gain", "上涨", "看涨",
            "多头", "买入", "上升", "增长", "涨势",
        ]
        bearish_kw = [
            "bearish", "downward", "selling", "short", "fall", "negative",
            "weak", "sell pressure", "drop", "decline", "下跌", "看跌",
            "空头", "卖出", "下降", "回调", "跌势",
        ]

        b_count = sum(1 for kw in bullish_kw if kw in text)
        s_count = sum(1 for kw in bearish_kw if kw in text)
        total   = b_count + s_count

        if total == 0:
            return 0.0
        return float((b_count - s_count) / total)

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _cache_key(self, headlines: List[str], ctx: Dict) -> str:
        content = json.dumps({"h": headlines[:5], "i": ctx.get("instrument")}, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _load_cache(self, key: str) -> Optional[Dict]:
        path = os.path.join(self.cache_dir, f"mf_{key}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                d = json.load(f)
            if time.time() - d.get("_ts", 0) > self.CACHE_TTL:
                return None
            return d
        except Exception:
            return None

    def _save_cache(self, key: str, data: Dict):
        path = os.path.join(self.cache_dir, f"mf_{key}.json")
        try:
            data["_ts"] = time.time()
            with open(path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Cache save failed: {e}")
