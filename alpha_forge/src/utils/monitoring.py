"""
Alpha-Forge System Monitoring
GPU, memory, CPU, and trading health monitoring
"""

import psutil
import time
import logging
from typing import Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class SystemMonitor:
    """System resource monitoring."""

    def __init__(self):
        self.start_time = time.time()
        self.last_check = time.time()
        self.check_interval = 60  # seconds

    def get_system_stats(self) -> Dict:
        """Get current system resource usage."""
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        stats = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "cpu_percent": cpu,
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent": memory.percent,
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "used_gb": round(disk.used / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent": disk.percent,
            },
        }

        # GPU stats (if available)
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                stats["gpu"] = {
                    "temp_c": int(parts[0].strip()),
                    "utilization_pct": float(parts[1].strip().replace("%", "")),
                    "memory_used_mb": int(parts[2].strip().replace("MiB", "")),
                    "memory_total_mb": int(parts[3].strip().replace("MiB", "")),
                }
        except Exception:
            pass

        return stats

    def check_health(self) -> Dict:
        """Check system health and return alerts."""
        stats = self.get_system_stats()
        alerts = []

        if stats["cpu_percent"] > 90:
            alerts.append(f"High CPU: {stats['cpu_percent']}%")

        if stats["memory"]["percent"] > 90:
            alerts.append(f"High Memory: {stats['memory']['percent']}%")

        if stats["disk"]["percent"] > 90:
            alerts.append(f"High Disk: {stats['disk']['percent']}%")

        if "gpu" in stats:
            if stats["gpu"]["temp_c"] > 85:
                alerts.append(f"High GPU Temp: {stats['gpu']['temp_c']}C")

        return {
            "status": "healthy" if not alerts else "warning",
            "alerts": alerts,
            "stats": stats,
        }

    def update(self):
        """Periodic health check."""
        now = time.time()
        if now - self.last_check >= self.check_interval:
            health = self.check_health()
            if health["alerts"]:
                for alert in health["alerts"]:
                    logger.warning(f"System Alert: {alert}")
            self.last_check = now
