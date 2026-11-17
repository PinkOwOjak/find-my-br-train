# scratch/test_live_features.py
#
# Tests for the live-tracking redesign: track geometry, speed estimation,
# 30-day history + ETA prediction, and the /ws/live WebSocket hub.
#
# Run:  python3 -m pytest scratch/test_live_features.py -q
#
# The hub test needs a local Redis. It uses db 15 (never the production db 0)
# and a synthetic train id, and wipes only its own keys.

import datetime
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from functions.history_store import (
    BD_TZ,
    HistoryStore,
    service_date_for,
    stop_crossings,
)
from functions.motion import combine_device_speeds, update_motion
from functions.position_filter import build_cumulative_km_from_legs
from functions.track_geometry import (
    TrackGeometry,
    decode_polyline,
    encode_polyline,
    route_leg_lengths_km,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Track geometry
# ---------------------------------------------------------------------------

def test_polyline_round_trip():
    pts = [(23.73425, 90.42642), (23.85207, 90.40807), (24.0, 90.0), (21.4, 92.1)]
    decoded = decode_polyline(encode_polyline(pts))
    assert len(decoded) == len(pts)
    for (a, b), (c, d) in zip(pts, decoded):
        assert abs(a - c) < 1e-5 and abs(b - d) < 1e-5


def test_track_db_lengths_beat_straight_lines():
    track = TrackGeometry(os.path.join(ROOT, "track", "track.db"))
    if track.segment_count == 0:
        pytest.skip("track/track.db not built")
    import json
    with open(os.path.join(ROOT, "data.json"), encoding="utf-8") as f:
        data = json.load(f)
    stations = data["tid_to_stations"]["702"]  # Dhaka -> Chattogram
    locs = {**data.get("xsid_to_sloc", {}), **data["sid_to_sloc"]}
    legs_track = route_leg_lengths_km(stations, locs, track)
    legs_straight = route_leg_lengths_km(stations, locs, TrackGeometry("/nonexistent.db"))
    total_track, total_straight = sum(legs_track), sum(legs_straight)
    # Dhaka-Chattogram by rail is ~320 km. Straight lines between our
    # (slightly off-track) station points land close to that too; the track
    # version must agree within a few percent, not wildly differ.
    assert 300 < total_track < 340, total_track
    assert abs(total_track - total_straight) / total_straight < 0.05
    cum = build_cumulative_km_from_legs(stations, legs_track)
    assert cum[0] == 0 and all(b > a for a, b in zip(cum, cum[1:]))
    assert abs(cum[-1] - total_track) < 1e-6


# ---------------------------------------------------------------------------
# Speed
# ---------------------------------------------------------------------------

def test_speed_from_track_progress():
    m = update_motion(None, 1.0, 10.0, 1000)
    assert m["speed"] is None
    m = update_motion(m, 1.1, 11.0, 1060)          # 1 km in 60 s = 60 km/h
    assert m["speed"] == pytest.approx(60, abs=0.5)
    assert m["moving"]


def test_speed_is_smoothed_and_backwards_is_zero():
    m = {"pos": 1.0, "km": 10.0, "ts": 1000, "speed": 60.0, "moving": True}
    m2 = update_motion(m, 1.0, 9.95, 1010)         # GPS jitter backwards, 10 s
    assert 0 < m2["speed"] < 60                     # decays towards 0, doesn't jump
    stopped = m2
    for i in range(30):
        stopped = update_motion(stopped, 1.0, 9.95, 1010 + (i + 1) * 20)
    assert stopped["speed"] < 1 and not stopped["moving"]


def test_speed_ignores_tiny_intervals_and_caps():
    m = {"pos": 1.0, "km": 10.0, "ts": 1000, "speed": 50.0, "moving": True}
    m2 = update_motion(m, 1.5, 15.0, 1002)          # 5 km in 2 s: noise, too short
    assert m2["ts"] == 1000 and m2["speed"] == 50.0
    m3 = update_motion(m, 3.0, 40.0, 1060)          # 30 km in 60 s -> capped
    assert m3["speed"] <= 140


def test_device_speed_blend():
    assert combine_device_speeds([50, 52, 300]) == 52  # median, 300 clamped to 140
    m = {"pos": 1.0, "km": 10.0, "ts": 1000, "speed": None, "moving": False}
    m2 = update_motion(m, 1.1, 11.0, 1060, device_speed_kmh=70.0)
    assert 60 < m2["speed"] < 70


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def _ts(y, mo, d, h, mi):
    return datetime.datetime(y, mo, d, h, mi, tzinfo=BD_TZ).timestamp()


def test_stop_crossings_interpolate():
    out = stop_crossings(0.5, 1000, 2.5, 3000, 5)
    assert [k for k, _ in out] == [1, 2]
    assert out[0][1] == pytest.approx(1500)
    assert stop_crossings(2.5, 1000, 2.4, 2000, 5) == []        # backwards
    assert stop_crossings(0.0, 1000, 0.2, 1100, 5)[0][0] == 0   # departure from origin


def test_service_date_for_overnight_run():
    times = [(0, 23 * 60 + 30), (1, 2 * 60), (2, 6 * 60)]   # 23:30 -> 06:00
    assert service_date_for(times, _ts(2026, 11, 10, 23, 45)) == "2026-11-10"
    assert service_date_for(times, _ts(2026, 11, 11, 5, 0)) == "2026-11-10"


def test_history_delay_summary_and_prediction():
    with tempfile.TemporaryDirectory() as tmp:
        store = HistoryStore(os.path.join(tmp, "h.db"))
        times = [(0, 8 * 60), (1, 9 * 60), (2, 10 * 60), (3, 11 * 60)]  # 08:00 .. 11:00
        now = _ts(2026, 11, 17, 8, 0)
        # Five past runs: 10 min late at stop 1, always loses another 20 min
        # by stop 2 (a habitual crossing wait), recovers 5 by stop 3.
        for day in range(10, 15):
            prev = None
            fixes = [(0.0, 8, 0), (0.5, 8, 35), (1.0, 9, 10), (1.5, 9, 50), (2.0, 10, 30), (2.6, 11, 0), (3.0, 11, 25)]
            for pos, h, mi in fixes:
                ts = int(_ts(2026, 11, day, h, mi))
                store.record("T1", times, 4, prev, pos, ts, None, 50.0, 1)
                prev = {"pos": pos, "ts": ts}
        summary = store.delay_summary("T1", days=14, now=now)
        assert summary["runs"] == 5
        assert summary["daily"][0]["delay_min"] == 25          # 11:25 vs 11:00 at the last stop
        assert summary["max_delay_min"] == 25 and summary["min_delay_min"] == 25

        # Today's run: at stop 1 (position 1.0) at 09:40 -> 40 min late.
        pred = store.predict("T1", times, 4, 1.0, _ts(2026, 11, 17, 9, 40), now=now)
        by_stop = {s["stop"]: s for s in pred["stops"]}
        assert pred["history_runs"] == 5
        assert by_stop[2]["source"] == "history"
        assert by_stop[2]["predicted_delay_min"] == 40 + 20
        assert by_stop[3]["predicted_delay_min"] == 40 + 15
        eta3 = datetime.datetime.fromtimestamp(by_stop[3]["eta"], tz=BD_TZ)
        assert (eta3.hour, eta3.minute) == (11, 55)

        # No history for another train -> delay carried forward unchanged.
        pred2 = store.predict("T2", times, 4, 1.0, _ts(2026, 11, 17, 9, 40), now=now)
        assert all(s["source"] == "carry" and s["predicted_delay_min"] == 40 for s in pred2["stops"])
        store.close()


# ---------------------------------------------------------------------------
# WebSocket hub (needs local Redis; uses db 15)
# ---------------------------------------------------------------------------

def _redis_available():
    try:
        import redis
        redis.Redis(db=15).ping()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _redis_available(), reason="local Redis not running")
def test_live_socket_end_to_end():
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, WebSocket
    from fastapi.testclient import TestClient

    from functions.live_hub import LiveHub
    from functions.redis_tracker import RedisTrainTracker

    tid = "test_live_701"
    now = datetime.datetime.now(tz=BD_TZ)
    start = now - datetime.timedelta(minutes=30)
    times = [start + datetime.timedelta(minutes=60 * i) for i in range(4)]
    data = {
        "tid_to_stations": {tid: [
            ["Dhaka", 1, times[0].strftime("%H:%M")],
            ["Biman_Bandar", 1, times[1].strftime("%H:%M")],
            ["Tongi", 1, times[2].strftime("%H:%M")],
            ["Joydebpur", 1, times[3].strftime("%H:%M")],
        ]},
        "sid_to_sloc": {
            "Dhaka": [23.73424, 90.42642], "Biman_Bandar": [23.85206, 90.40806],
            "Tongi": [23.89553, 90.40463], "Joydebpur": [23.99902, 90.42129],
        },
    }
    tracker = RedisTrainTracker(db=15)
    tracker.redis.delete(*([k for k in tracker.redis.scan_iter(f"train:{tid}:*")] or ["_none_"]))
    tracker.set_train_data(data)
    with tempfile.TemporaryDirectory() as tmp:
        tracker.history = HistoryStore(os.path.join(tmp, "h.db"))
        from functions import gps_log
        gps_log.DB_PATH, gps_log._conn = os.path.join(tmp, "gps.db"), None
        hub = LiveHub(tracker, redis_db=15)

        @asynccontextmanager
        async def lifespan(app):
            await hub.start()
            yield
            await hub.stop()

        app = FastAPI(lifespan=lifespan)

        @app.websocket("/ws/live")
        async def ws_endpoint(websocket: WebSocket):
            await hub.handle(websocket)

        with TestClient(app) as client:
            with client.websocket_connect("/ws/live") as viewer, \
                    client.websocket_connect("/ws/live") as rider:
                viewer.send_json({"t": "hello", "user_id": "viewer_1"})
                viewer.send_json({"t": "sub", "trains": [tid]})
                snap = viewer.receive_json()
                assert snap["t"] == "snap" and snap["trains"][tid] is None  # no data yet

                rider.send_json({"t": "hello", "user_id": "rider_1"})
                t0 = int(time.time()) - 120
                rider.send_json({"t": "pos", "ref": 1, "train_id": tid, "position": 0.40,
                                 "time": t0 * 1000, "speed_kmh": 48.0})
                ack = rider.receive_json()
                assert ack == {"t": "ack", "ref": 1, "ok": True, "msg": "Position updated"}

                upd = viewer.receive_json()
                assert upd["t"] == "upd"
                state = upd["trains"][tid]
                assert state["is_live"] and state["position"] == pytest.approx(0.40)
                assert state["active_user"] == 1

                # Second report 60 s later, further along: speed appears.
                time.sleep(1.6)  # per-connection rate limit
                rider.send_json({"t": "pos", "ref": 2, "train_id": tid, "position": 0.48,
                                 "time": (t0 + 60) * 1000, "speed_kmh": 52.0})
                assert rider.receive_json()["ok"] is True
                upd2 = viewer.receive_json()
                state2 = upd2["trains"][tid]
                assert state2["position"] == pytest.approx(0.48)
                assert state2["speed_kmh"] is not None and 20 < state2["speed_kmh"] < 80
                assert state2["moving"] is True

                # Throttled rider + ping/pong
                rider.send_json({"t": "pos", "ref": 3, "train_id": tid, "position": 0.5, "time": t0 * 1000})
                assert rider.receive_json()["ok"] is False
                viewer.send_json({"t": "ping"})
                assert viewer.receive_json()["t"] == "pong"

                # sub_all snapshot includes the live train
                viewer.send_json({"t": "sub_all"})
                snap_all = viewer.receive_json()
                assert snap_all["scope"] == "all" and tid in snap_all["trains"]
        tracker.history.close()
    tracker.redis.delete(*([k for k in tracker.redis.scan_iter(f"train:{tid}:*")] or ["_none_"]))
    tracker.redis.srem("active_trains", tid)
    tracker.redis.srem("all_trains_with_history", tid)
