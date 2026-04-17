"""
ML Scalper Dashboard — FastAPI Backend
=======================================
Kullanım:
  pip install fastapi uvicorn jinja2
  python dashboard/app.py          # http://localhost:8080
  python dashboard/app.py --port 8080 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import uvicorn

BASE_DIR      = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / 'templates'
STATIC_DIR    = BASE_DIR / 'static'
STATE_FILE    = BASE_DIR.parent / 'logs' / 'ml_scalper' / 'live_state.json'
NEWS_LOG      = BASE_DIR.parent / 'logs' / 'ml_scalper' / 'news_feed.json'

app = FastAPI(title='ML Scalper Dashboard', docs_url=None, redoc_url=None)

if STATIC_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {
        'updated_at': None,
        'mode': 'offline',
        'symbol': 'XAUUSD',
        'equity': 0,
        'open_position': {'active': False},
        'stats': {
            'total_trades': 0, 'wins': 0, 'losses': 0,
            'win_rate': 0, 'profit_factor': 0, 'total_pnl': 0,
        },
        'trades': [],
        'params': {},
    }


def _load_news() -> list:
    if NEWS_LOG.exists():
        try:
            return json.loads(NEWS_LOG.read_text())[-20:]
        except Exception:
            pass
    return []


@app.get('/', response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})


@app.get('/api/state')
async def api_state():
    return JSONResponse(_load_state())


@app.get('/api/equity_curve')
async def api_equity_curve():
    state = _load_state()
    trades = state.get('trades', [])
    initial = 1000.0

    points = [{'t': state.get('updated_at', ''), 'v': initial}]
    running = initial
    for t in trades:
        running += t.get('pnl', 0)
        points.append({'t': t.get('close_ts', ''), 'v': round(running, 2)})

    return JSONResponse(points)


@app.get('/api/weekly_pnl')
async def api_weekly_pnl():
    import pandas as pd
    state  = _load_state()
    trades = state.get('trades', [])
    if not trades:
        return JSONResponse([])

    rows = []
    for t in trades:
        ts = t.get('close_ts')
        if ts:
            rows.append({'ts': pd.to_datetime(ts), 'pnl': t.get('pnl', 0)})

    if not rows:
        return JSONResponse([])

    df = pd.DataFrame(rows).set_index('ts')
    weekly = df['pnl'].resample('W').sum().tail(12)
    return JSONResponse([
        {'week': str(idx.date()), 'pnl': round(v, 2)}
        for idx, v in weekly.items()
    ])


@app.get('/api/news')
async def api_news():
    return JSONResponse(_load_news())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    print(f"Dashboard -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
