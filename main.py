# main.py - Simplified FastAPI application
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

# Import utility functions
from functions.data_loader import load_data
from functions.redis_tracker import RedisTrainTracker
from functions.route_calculator import (
    precalculate_train_routes,
    precalculate_station_distances,
    precalculate_two_train_routes
)

# Import URL handlers
from urls import github, data, positions, routes, reports, live


# Global variables
DATA, CURRENT_REVISION = load_data()

import hashlib
import json
import os
from urls.data import get_all_trains as fetch_all_trains

DATA_HASHES = {}

def precalculate_data_hashes():
    print("Precalculating data hashes...")

    # Base hash (for versions < 28)
    base_data = fetch_all_trains(DATA, version=0)
    DATA_HASHES[0] = hashlib.sha256(json.dumps(base_data, sort_keys=True).encode("utf-8")).hexdigest()

    # Precalculate version 28 specifically as it's the fallback
    v28_data = fetch_all_trains(DATA, version=28)
    DATA_HASHES[28] = hashlib.sha256(json.dumps(v28_data, sort_keys=True).encode("utf-8")).hexdigest()

    # Version hashes based on directories
    if os.path.exists("train_routes"):
        for folder in os.listdir("train_routes"):
            if folder.startswith("version"):
                try:
                    version_num = int(folder.replace("version", ""))
                    if version_num not in DATA_HASHES:
                        v_data = fetch_all_trains(DATA, version=version_num)
                        DATA_HASHES[version_num] = hashlib.sha256(json.dumps(v_data, sort_keys=True).encode("utf-8")).hexdigest()
                except ValueError:
                    pass
    print(f"Hashes precalculated for versions: {list(DATA_HASHES.keys())}")

precalculate_data_hashes()

TWO_TRAIN_ROUTES = {}
TRAIN_ROUTES = {}
STATION_DISTANCES = {}

# Redis-based train tracker (replaces AsyncTimedStack)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
tracker = RedisTrainTracker(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, ttl_seconds=600)
tracker.set_train_data(DATA)  # Provide train schedule data for scheduled position calculation

# 30-day journey history (delay statistics + ETA prediction)
from functions.history_store import HistoryStore
history = HistoryStore()
tracker.history = history

# Live position WebSocket hub (/ws/live)
from functions.live_hub import LiveHub
live_hub = LiveHub(tracker, redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB)

from functions.track_geometry import shared_track

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global TWO_TRAIN_ROUTES, TRAIN_ROUTES, STATION_DISTANCES
    
    print("Starting Find My BR Train FastAPI Server...")
    print("API Base URL: findmytrain.freeddns.org")
    print("Health Check: findmytrain.freeddns.org/health")
    print("\nPress Ctrl+C to stop the server\n")
    
    # Check Redis connection
    if tracker.health_check():
        print("✓ Redis connection established")
    else:
        print("⚠ Warning: Redis connection failed - position tracking won't work!")
    
    # Precalculate routes and distances
    TRAIN_ROUTES = precalculate_train_routes(DATA)
    STATION_DISTANCES = precalculate_station_distances(DATA)
    
    print("\n" + "="*60)
    print("INITIALIZING TWO-TRAIN ROUTE PRECALCULATION")
    print("="*60)
    TWO_TRAIN_ROUTES = precalculate_two_train_routes(DATA, CURRENT_REVISION)
    print("="*60 + "\n")

    shared_track().segment_count  # load track geometry up front, not on the first ping
    await live_hub.start()
    print("✓ Live WebSocket hub started (/ws/live)")

    yield

    print("Shutting down FastAPI Server...")
    await live_hub.stop()
    history.close()


app = FastAPI(
    title="Find My BR Train Server",
    description="API server for Find My BR Train app",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware. This API has no cookie/credentialed auth, so it's safe
# to allow every origin as long as allow_credentials stays False (the wildcard
# + credentials combination makes browsers trust every origin for credentialed
# requests, which we don't want).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Moesif middleware
from moesifasgi import MoesifMiddleware

def identify_user(request, response):
    # Extract User ID from header
    user_id = request.headers.get("x-user-id") or request.headers.get("X-User-Id")
    if user_id:
        return str(user_id)

    # Extract User ID from Authorization header. Never forward the raw
    # credential to a third party - hash it so requests from the same
    # caller still group together without leaking the token itself.
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth:
        return "auth:" + hashlib.sha256(auth.encode("utf-8")).hexdigest()[:16]

    # Extract User ID from query parameters
    try:
        user_id = request.query_params.get("user_id") or request.query_params.get("user")
        if user_id:
            return str(user_id)
    except Exception:
        pass

    # Extract User ID from cached body (e.g. for /sendupdate)
    try:
        if hasattr(request, "_body") and request._body:
            import json
            body_json = json.loads(request._body.decode("utf-8"))
            uid = body_json.get("user_id") or body_json.get("id")
            if uid:
                return str(uid)
    except Exception:
        pass

    return None

moesif_settings = {
    'APPLICATION_ID': os.getenv("MOESIF_APPLICATION_ID", ""),
    # Request/response bodies (GPS coordinates, user IDs) are not sent to the
    # third-party analytics service - only metadata (route, status, timing).
    'LOG_BODY': False,
    'IDENTIFY_USER': identify_user,
}

app.add_middleware(MoesifMiddleware, settings=moesif_settings)

# Register WebSocket Chat endpoint
from functions.chat_manager import register_chat_endpoint
register_chat_endpoint(app, tracker)

# Register Chat User Report endpoints
from functions.user_report_manager import register_user_report_endpoints
register_user_report_endpoints(app)


# Mount static files
import os
if os.path.exists("train_routes"):
    app.mount("/train_routes", StaticFiles(directory="train_routes"), name="train_routes")
if os.path.exists(".well-known"):
    app.mount("/.well-known", StaticFiles(directory=".well-known"), name="well-known")
if os.path.exists("ads"):
    app.mount("/ads", StaticFiles(directory="ads"), name="ads")


# ============= ROUTES =============

# GitHub webhook
@app.post("/payload")
async def github_webhook_handler(request: Request):
    return await github.github_webhook(request)


# Data endpoints
from typing import Optional

@app.get("/initrevision")
async def get_revision(version: Optional[int] = None):
    return data.get_revision(CURRENT_REVISION, DATA_HASHES, version)


@app.get("/alltrains")
async def get_all_trains(request: Request, version: int = 0, is_sql: bool = False):
    from fastapi.responses import Response, FileResponse
    import gzip
    import json
    import os

    # Serve SQLite database if requested
    if is_sql:
        resolved_version = 0
        if version >= 29 and os.path.exists("train_routes"):
            available_versions = []
            for folder in os.listdir("train_routes"):
                if folder.startswith("version"):
                    try:
                        v = int(folder.replace("version", ""))
                        if v >= 29:
                            available_versions.append(v)
                    except ValueError:
                        pass
            valid_versions = [v for v in available_versions if v <= version]
            if valid_versions:
                resolved_version = max(valid_versions)

        db_dir = "sqlite_db"
        os.makedirs(db_dir, exist_ok=True)
        db_path = os.path.join(db_dir, f"version{resolved_version}.db")
        
        # Generate or check database validity
        from sqlite_generator import get_or_create_sqlite_db
        get_or_create_sqlite_db(DATA, CURRENT_REVISION, DATA_HASHES, resolved_version, db_path)

        if "gzip" in request.headers.get("accept-encoding", ""):
            with open(db_path, "rb") as f:
                db_data = f.read()
            compressed_data = gzip.compress(db_data)
            return Response(
                content=compressed_data,
                media_type="application/vnd.sqlite3",
                headers={
                    "Content-Encoding": "gzip",
                    "Content-Disposition": f'attachment; filename="alltrains_v{resolved_version}.db"'
                }
            )
            
        return FileResponse(
            db_path,
            media_type="application/vnd.sqlite3",
            filename=f"alltrains_v{resolved_version}.db"
        )

    result = data.get_all_trains(DATA, version)

    if version >= 29:
        # Inject hash using same max-Y <= version logic
        valid_versions = [v for v in DATA_HASHES.keys() if v <= version and v != 0]
        if valid_versions:
            result["hash"] = DATA_HASHES[max(valid_versions)]
        elif 0 in DATA_HASHES:
            result["hash"] = DATA_HASHES[0]

        if "gzip" in request.headers.get("accept-encoding", ""):
            compressed_data = gzip.compress(json.dumps(result, separators=(",", ":")).encode("utf-8"))
            return Response(
                content=compressed_data,
                media_type="application/json",
                headers={"Content-Encoding": "gzip"}
            )

    return result


# Position endpoints
@app.get("/current/{train_ids}")
async def get_current_positions_handler(train_ids: str):
    return positions.get_current_positions(train_ids, tracker)


@app.get("/bounds/{train_id}")
async def get_train_bounds_handler(train_id: str):
    """Debug endpoint: the reference position new pings for this train are
    currently being teleport-checked against"""
    return positions.get_train_bounds(train_id, tracker)


@app.post("/sendupdate")
async def receive_update_handler(update: positions.LocationUpdate):
    result = positions.receive_update(update, tracker)
    live_hub.mark_dirty(update.train_id or update.id)
    return result


# Live positions over one WebSocket per app (see functions/live_hub.py)
from fastapi import WebSocket


@app.websocket("/ws/live")
async def live_socket_handler(websocket: WebSocket):
    await live_hub.handle(websocket)


# History & prediction endpoints
from fastapi import HTTPException


def _station_times_or_404(train_id: str):
    station_times = tracker._get_station_times(train_id)
    if not station_times or len(station_times) < 2:
        raise HTTPException(404, f"No timetable for train {train_id}")
    return station_times


@app.get("/history/{train_id}/delays")
async def get_delay_history_handler(train_id: str, days: int = 7):
    """Per-run delay over the last `days` days (max 30), plus per-stop medians."""
    _station_times_or_404(train_id)
    from fastapi.concurrency import run_in_threadpool
    return await run_in_threadpool(history.delay_summary, train_id, days)


@app.get("/predict/{train_id}")
async def get_prediction_handler(train_id: str):
    """
    Predicted delay and ETA at every scheduled stop ahead of the train's
    current position, learned from up to 30 days of past runs. Returns
    "stops": [] when there's no recent position to predict from.
    """
    import time as _time
    from fastapi.concurrency import run_in_threadpool

    station_times = _station_times_or_404(train_id)
    info = tracker.get_positions([train_id]).get(train_id)
    if not info or _time.time() - info.get("timestamp", 0) > 3 * 3600:
        return {"train_id": train_id, "stops": [], "history_runs": 0, "based_on": None}
    result = await run_in_threadpool(
        history.predict, train_id, station_times, tracker._stop_count(train_id),
        float(info["position"]), float(info["timestamp"]),
    )
    result["speed_kmh"] = info.get("speed_kmh")
    result["is_live"] = info.get("is_live", False)
    return result


# Route endpoints
@app.get("/two-train-routes/{from_station}/{to_station}")
async def get_two_train_routes_handler(from_station: str, to_station: str):
    return routes.get_two_train_routes(from_station, to_station, TWO_TRAIN_ROUTES, DATA)


@app.get("/two-train-routes-all")
async def get_all_two_train_routes_handler():
    return routes.get_all_two_train_routes(TWO_TRAIN_ROUTES, DATA)


@app.post("/nearbyroute")
async def find_nearby_routes_handler(request: routes.NearbyRouteRequest):
    return await routes.find_nearby_routes(request, DATA, TRAIN_ROUTES, STATION_DISTANCES)


# Report endpoints
@app.post("/fix")
async def report_issue_handler(report: reports.IssueReport):
    return await reports.report_issue_post(report)


@app.get("/report")
async def view_reports_handler():
    return await reports.view_reports()


# Live endpoints
@app.get("/health")
async def health_check_handler():
    result = live.health_check(CURRENT_REVISION, tracker)
    result["live_socket"] = live_hub.summary()
    result["track_segments"] = shared_track().segment_count
    return result


@app.get("/live")
async def view_live_trains_handler():
    return live.view_live_trains(tracker, DATA)


@app.get("/activetrains")
async def get_active_trains_handler():
    return live.get_active_trains_details(tracker)


@app.get("/recent-conversations")
async def get_recent_conversations():
    import time
    import json
    redis_client = tracker.redis
    try:
        keys = list(redis_client.scan_iter(match="train:*:chat", count=200))
    except Exception as e:
        print(f"Error fetching keys from Redis: {e}")
        return []
        
    conversations = []
    current_time = int(time.time())
    train_names = DATA.get("train_names", {})
    
    for key in keys:
        if isinstance(key, bytes):
            key_str = key.decode("utf-8")
        else:
            key_str = key
            
        parts = key_str.split(":")
        if len(parts) < 3:
            continue
        train_id = parts[1]
        
        try:
            latest_items = redis_client.zrange(key, -1, -1, withscores=True)
            if not latest_items:
                continue
                
            msg_data_raw, score = latest_items[0]
            if isinstance(msg_data_raw, bytes):
                msg_data_str = msg_data_raw.decode("utf-8")
            else:
                msg_data_str = msg_data_raw
                
            msg_data = json.loads(msg_data_str)
            timestamp = int(score)
            
            names = train_names.get(train_id)
            if not names:
                train_name_en = f"Train {train_id}"
                train_name_bn = f"ট্রেন {train_id}"
            else:
                train_name_en = names[0]
                train_name_bn = names[1]
                
            conversations.append({
                "train_id": train_id,
                "train_name_en": train_name_en,
                "train_name_bn": train_name_bn,
                "latest_message": msg_data.get("text", ""),
                "sender": msg_data.get("sender", ""),
                "timestamp": timestamp,
                "elapsed_seconds": max(0, current_time - timestamp)
            })
        except Exception as e:
            print(f"Error parsing conversation for {key_str}: {e}")
            
    conversations.sort(key=lambda x: x["timestamp"], reverse=True)
    return conversations


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Find My BR Train FastAPI Server",
        "version": "1.0.0",
        "framework": "FastAPI",
        "endpoints": {
            "/initrevision": "GET - Check data revision",
            "/alltrains": "GET - Download complete database",
            "/activetrains": "GET - Get active trains details",
            "/current/{train_ids}": "GET - Get current train positions",
            "/sendupdate": "POST - Submit location update",
            "/ws/live": "WebSocket - live positions (subscribe) and rider updates",
            "/history/{train_id}/delays?days=7": "GET - Delay history (up to 30 days)",
            "/predict/{train_id}": "GET - Predicted delay/ETA for upcoming stops",
            "/fix": "POST - Report incorrect information",
            "/report": "GET - View all issue reports",
            "/nearbyroute": "POST - Find alternative routes",
            "/health": "GET - Server health check",
            "/live": "GET - View live trains",
            "/recent-conversations": "GET - Get trains sorted by latest chat messages",
            "/ads/promo_banners.json": "GET - House banner ad fallback config",
            "/docs": "GET - Interactive API documentation",
        },
        "github": "https://github.com/jisangain/find-my-br-train",
        "contribute": "Visit our GitHub repository to contribute!"
    }


if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        host='0.0.0.0',
        port=8000,
        reload=False,
        log_level="info"
    )
