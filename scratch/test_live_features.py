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
        summary = store.delay_summary("T1", days=14, now=now, station_times=times, stop_count=4)
        assert summary["runs"] == 5
        with_data = [d for d in summary["daily"] if d["delay_min"] is not None]
        assert with_data[0]["delay_min"] == 25 and with_data[0]["source"] == "observed"  # 11:25 vs 11:00
        assert summary["max_delay_min"] == 25 and summary["min_delay_min"] == 25
        # Every day of the window is listed; the 8 days without data are null
        # (today, 17th, has nothing recorded yet and is left out, not "missing").
        assert [d["date"] for d in summary["daily"]][0] == "2026-11-04"
        assert summary["missing_days"] == 8 and len(summary["daily"]) == 13

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
        from functions.speed_profile import SpeedProfileModel
        tracker.speed_model = SpeedProfileModel(os.path.join(tmp, "h.db"), tracker)
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
                # 10-minute motion plan: starts at the fix, never goes backwards
                plan = state2["plan"]
                assert plan and plan[0] == [pytest.approx(0.48, abs=1e-4), 0]
                assert plan[-1][1] == 600
                assert all(b[0] >= a[0] and b[1] > a[1] for a, b in zip(plan, plan[1:]))
                assert plan[-1][0] > plan[0][0]   # timetable pace moves it forward

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


# ---------------------------------------------------------------------------
# Speed profiles / 10-minute motion plan
# ---------------------------------------------------------------------------

from functions.speed_profile import (  # noqa: E402
    CELL_KM, SpeedProfileModel, TrainProfile, km_to_position, run_cell_times,
)
from functions.position_filter import position_to_km as _p2km  # noqa: E402


def test_km_position_round_trip():
    cum = [0.0, 12.0, 30.5, 31.0, 80.0]
    for pos in (0.0, 0.25, 1.0, 1.7, 2.5, 3.999, 4.0):
        assert km_to_position(cum, _p2km(cum, pos)) == pytest.approx(pos, abs=1e-9)


def test_run_cell_times_capture_dwell_and_skip_gaps():
    # 60 km/h (1 km/min) to km 1.0, stand 180 s, continue at 60 km/h.
    pts = [(0, 0.0), (30, 0.5), (60, 1.0), (240, 1.0), (270, 1.5), (300, 2.0)]
    cells = run_cell_times(pts)
    assert 0 not in cells                           # start boundary never crossed - not measurable
    assert cells[1] == pytest.approx(12)            # 0.2 km at 1 km/min
    assert cells[4] == pytest.approx(12)            # last cell before the stop
    assert cells[5] == pytest.approx(180 + 12)      # cell starting at the stop holds the dwell
    # a 40-minute hole in the data must not be learned as one very slow stretch
    cells2 = run_cell_times([(0, 0.0), (60, 1.0), (60 + 2400, 3.0), (2520, 4.0)])
    assert 2 in cells2 and 16 in cells2             # measured inside each unbroken chain
    assert all(c not in cells2 for c in range(5, 16))   # nothing learned across the hole


class _FakeTracker:
    """Just the timetable helpers SpeedProfileModel needs."""
    def __init__(self, cum_km, stop_sids, times):
        self.cum_km, self.times = cum_km, times
        self.train_data = {"tid_to_stations": {"T": [[sid, 1, "00:00"] for sid in stop_sids]}}
    def _get_cumulative_km(self, tid):
        return self.cum_km
    def _get_station_times(self, tid):
        return self.times


def _seed_history(path, runs):
    store = HistoryStore(path)
    db = store._db()
    for date, pts in runs:
        day = int(date[-2:]) * 86400
        for ts, pos in pts:
            db.execute("INSERT INTO run_point VALUES ('T', ?, ?, ?, NULL, NULL, NULL, 1)", (date, day + ts, pos))
    db.commit()
    store.close()


def test_plan_follows_history_and_dwells_at_station():
    cum = [0.0, 5.0, 10.0]                         # stops A (km 0), B (km 5), C (km 10)
    tr = _FakeTracker(cum, ["A", "B", "C"], [(0, 600), (1, 610), (2, 620)])
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "h.db")
        # Every past run: 60 km/h, a 4-minute stop at B, 60 km/h again.
        runs = []
        for d in range(1, 6):
            date = f"2026-11-0{d}"
            pts, t = [], 0
            for i in range(0, 51):                  # km 0 -> 5 in 300 s
                pts.append((t + i * 6, i * 0.1 / 5))
            t = 300 + 240                           # dwell at B
            for i in range(0, 51):
                pts.append((t + i * 6, 1 + i * 0.1 / 5))
            runs.append((date, pts))
        _seed_history(path, runs)
        model = SpeedProfileModel(path, tr, refresh_s=1e9)
        model.build(now=datetime.datetime(2026, 11, 10, tzinfo=BD_TZ).timestamp(), since_date="2026-10-01")
        prof = model._profile_for("T")
        assert prof.runs == 5 and len(prof.medians) >= 45

        # From km 3.0 at speed 60: reaches B (km 5) after ~120 s, stands
        # ~4 min, then continues.
        plan = model.plan("T", km_to_position(cum, 3.0), 60.0)
        km = {t: _p2km(cum, p) for p, t in plan}
        assert km[90] == pytest.approx(4.5, abs=0.3)
        assert 4.7 < km[180] <= 5.25 and 4.7 < km[300] <= 5.25    # standing at B
        assert km[600] > 7.0                                       # left B and kept going

        # A train reported stopped holds for a minute before following history.
        stopped = model.plan("T", km_to_position(cum, 2.0), 0.0)
        assert _p2km(cum, stopped[2][0]) == pytest.approx(2.0, abs=1e-6)   # t = 60 s
        assert _p2km(cum, stopped[4][0]) > 2.3                              # t = 120 s


def test_plan_falls_back_to_timetable():
    cum = [0.0, 20.0, 40.0]
    # 20 km legs scheduled at 25 min each -> ~(1500-60)/20 = 72 s/km running pace
    tr = _FakeTracker(cum, ["A", "B", "C"], [(0, 600), (1, 625), (2, 650)])
    with tempfile.TemporaryDirectory() as tmp:
        model = SpeedProfileModel(os.path.join(tmp, "missing.db"), tr, refresh_s=1e9)
        plan = model.plan("T", 0.0, None)
        km10 = _p2km(cum, plan[-1][0])
        assert 600 / 72 * 0.9 < km10 < 600 / 72 * 1.1
        secs, source = model.cell_seconds(model._profile_for("T"), 10)
        assert source == "timetable" and secs == pytest.approx(72 * CELL_KM, rel=0.01)


# ---------------------------------------------------------------------------
# Per-station delay history
# ---------------------------------------------------------------------------

def test_window_dates_skip_offdays():
    now = _ts(2026, 11, 18, 12, 0)                     # Wednesday
    dates, skipped = HistoryStore.window_dates(7, now, ["Fri"])
    assert dates[-1] == "2026-11-18" and "2026-11-13" not in dates and skipped == ["2026-11-13"]
    assert len(dates) == 6


def test_stop_delay_history_observed_interpolated_estimated_missing_and_offday():
    times = [(0, 8 * 60), (1, 9 * 60), (2, 10 * 60), (3, 11 * 60), (4, 12 * 60)]
    now = _ts(2026, 11, 18, 20, 0)                     # Wed evening; window Thu 12th .. Wed 18th
    with tempfile.TemporaryDirectory() as tmp:
        store = HistoryStore(os.path.join(tmp, "h.db"))
        db = store._db()

        def point(date, h, mi, pos):
            db.execute("INSERT INTO run_point VALUES ('S', ?, ?, ?, NULL, NULL, NULL, 1)",
                       (date, int(_ts(*map(int, date.split("-")), h, mi)), pos))

        # 12th: dense data, crossing of stop 2 recorded live at 10:20 -> observed 20 min.
        store.record("S", times, 5, {"pos": 1.9, "ts": int(_ts(2026, 11, 12, 10, 15))}, 2.1,
                     int(_ts(2026, 11, 12, 10, 25)), None, 50.0, 1)
        # 14th: a 2-hour hole around stop 2: seen at 1.5 at 09:40, next at 2.5 at 11:10
        # -> passing 2.0 interpolated at 10:25 -> 25 min late.
        point("2026-11-14", 9, 40, 1.5)
        point("2026-11-14", 11, 10, 2.5)
        # 15th: data stops at 1.2 (scheduled 09:12) seen at 09:42 - 30 min late there; within
        # 1.5 stops of stop 2 -> its delay is used as an estimate: 30.
        point("2026-11-15", 9, 42, 1.2)
        # 16th: data only near the origin (0.1) - too far from stop 2 -> missing.
        point("2026-11-16", 8, 5, 0.1)
        # 13th is Friday = off day -> not listed at all. 17th: nothing -> missing.
        # 18th (today): nothing recorded yet -> left out, not missing.
        db.commit()

        h = store.stop_delay_history("S", times, 2, days=7, offdays=["Fri"], now=now)
        by_date = {d["date"]: d for d in h["daily"]}
        assert h["offdays_skipped"] == ["2026-11-13"] and "2026-11-13" not in by_date
        assert by_date["2026-11-12"] == {"date": "2026-11-12", "delay_min": 20, "source": "observed"}
        assert by_date["2026-11-14"]["delay_min"] == 25 and by_date["2026-11-14"]["source"] == "interpolated"
        assert by_date["2026-11-15"]["delay_min"] == 30 and by_date["2026-11-15"]["source"] == "estimated"
        assert by_date["2026-11-16"]["delay_min"] is None
        assert by_date["2026-11-17"]["delay_min"] is None
        assert "2026-11-18" not in by_date
        assert h["runs"] == 3 and h["missing_days"] == 2
        assert h["avg_delay_min"] == 25 and h["max_delay_min"] == 30 and h["min_delay_min"] == 20

        # A run in progress that hasn't reached the stop yet isn't "missing".
        point("2026-11-18", 19, 30, 0.4)
        db.commit()
        h2 = store.stop_delay_history("S", times, 4, days=1, now=now)
        assert h2["daily"] == []
        store.close()


def test_implausibly_early_arrivals_are_ignored():
    times = [(0, 8 * 60), (1, 9 * 60), (2, 10 * 60)]
    now = _ts(2026, 11, 18, 20, 0)
    with tempfile.TemporaryDirectory() as tmp:
        store = HistoryStore(os.path.join(tmp, "h.db"))
        db = store._db()
        # 16th: "arrived" at stop 1 at 07:50 - 70 min early: a bad report.
        db.execute("INSERT INTO station_event VALUES ('E', '2026-11-16', 1, ?, -70)", (int(_ts(2026, 11, 16, 7, 50)),))
        # 17th: 5 min early - plausible, kept.
        db.execute("INSERT INTO station_event VALUES ('E', '2026-11-17', 1, ?, -5)", (int(_ts(2026, 11, 17, 8, 55)),))
        db.commit()
        h = store.stop_delay_history("E", times, 1, days=3, now=now)
        by_date = {d["date"]: d["delay_min"] for d in h["daily"]}
        assert by_date["2026-11-16"] is None and by_date["2026-11-17"] == -5
        assert h["min_delay_min"] == -5
        store.close()
