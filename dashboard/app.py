"""
Trading Dashboard — FastAPI Backend
=====================================
Serves state from both live engines:
  - ML Scalper    (logs/ml_scalper/live_state.json)
  - Algo Director (logs/algo_director/algo_state.json)

Usage:
  pip install fastapi uvicorn jinja2 pandas
  python dashboard/app.py              # http://localhost:8080
  python dashboard/app.py --port 8080 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

BASE_DIR      = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / 'templates'
STATIC_DIR    = BASE_DIR / 'static'

ML_STATE_FILE   = BASE_DIR.parent / 'logs' / 'ml_scalper'   / 'live_state.json'
ALGO_STATE_FILE = BASE_DIR.parent / 'logs' / 'algo_director' / 'algo_state.json'
NEWS_LOG        = BASE_DIR.parent / 'logs' / 'ml_scalper'   / 'news_feed.json'

app = FastAPI(title='Trading Dashboard', docs_url=None, redoc_url=None)

if STATIC_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ─── Loaders ─────────────────────────────────────────────────────────────────

def _load_ml_state() -> dict:
    if ML_STATE_FILE.exists():
        try:
            return json.loads(ML_STATE_FILE.read_text())
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


def _load_algo_state() -> dict:
    if ALGO_STATE_FILE.exists():
        try:
            return json.loads(ALGO_STATE_FILE.read_text())
        except Exception:
            pass
    return {
        'updated_at': None,
        'mode': 'offline',
        'sentiment': {'score': 0, 'action': 'FLAT', 'summary': '', 'age_min': 0},
        'gold_scalper':  {'stats': {}, 'open_position': {'active': False}, 'trades': []},
        'nq_liquidity': {'stats': {}, 'open_position': {'active': False}, 'trades': []},
    }


def _load_news() -> list:
    if NEWS_LOG.exists():
        try:
            return json.loads(NEWS_LOG.read_text())[-20:]
        except Exception:
            pass
    return []


def _build_equity_curve(trades: list, initial: float = 1000.0) -> list:
    points = [{'t': '', 'v': initial}]
    running = initial
    for t in trades:
        running += t.get('pnl', 0)
        points.append({'t': t.get('close_ts', ''), 'v': round(running, 2)})
    return points


def _weekly_pnl(trades: list) -> list:
    try:
        import pandas as pd
        rows = [
            {'ts': pd.to_datetime(t['close_ts']), 'pnl': t.get('pnl', 0)}
            for t in trades if t.get('close_ts')
        ]
        if not rows:
            return []
        df = pd.DataFrame(rows).set_index('ts')
        weekly = df['pnl'].resample('W').sum().tail(12)
        return [{'week': str(idx.date()), 'pnl': round(v, 2)} for idx, v in weekly.items()]
    except Exception:
        return []


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get('/', response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, 'index.html')


# ── ML Scalper endpoints ──────────────────────────────────────────────────────

@app.get('/api/state')
async def api_ml_state():
    return JSONResponse(_load_ml_state())


@app.get('/api/equity_curve')
async def api_ml_equity_curve():
    state = _load_ml_state()
    return JSONResponse(_build_equity_curve(state.get('trades', [])))


@app.get('/api/weekly_pnl')
async def api_ml_weekly_pnl():
    state = _load_ml_state()
    return JSONResponse(_weekly_pnl(state.get('trades', [])))


@app.get('/api/news')
async def api_news():
    return JSONResponse(_load_news())


# ── Algo Director endpoints ───────────────────────────────────────────────────

@app.get('/api/algo/state')
async def api_algo_state():
    return JSONResponse(_load_algo_state())


@app.get('/api/algo/equity_curve')
async def api_algo_equity_curve():
    state = _load_algo_state()
    gold_trades = state.get('gold_scalper', {}).get('trades', [])
    nq_trades   = state.get('nq_liquidity', {}).get('trades', [])
    # Merge and sort by close_ts
    all_trades = sorted(
        gold_trades + nq_trades,
        key=lambda t: t.get('close_ts', ''),
    )
    return JSONResponse(_build_equity_curve(all_trades))


@app.get('/api/algo/gold/equity_curve')
async def api_algo_gold_equity_curve():
    state = _load_algo_state()
    trades = state.get('gold_scalper', {}).get('trades', [])
    return JSONResponse(_build_equity_curve(trades))


@app.get('/api/algo/nq/equity_curve')
async def api_algo_nq_equity_curve():
    state = _load_algo_state()
    trades = state.get('nq_liquidity', {}).get('trades', [])
    return JSONResponse(_build_equity_curve(trades))


@app.get('/api/algo/weekly_pnl')
async def api_algo_weekly_pnl():
    state = _load_algo_state()
    gold_trades = state.get('gold_scalper', {}).get('trades', [])
    nq_trades   = state.get('nq_liquidity', {}).get('trades', [])
    all_trades  = gold_trades + nq_trades
    return JSONResponse(_weekly_pnl(all_trades))


@app.get('/api/algo/sentiment')
async def api_sentiment():
    state = _load_algo_state()
    return JSONResponse(state.get('sentiment', {}))


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8081)
    args = parser.parse_args()
    print(f"Dashboard -> http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level='warning')
