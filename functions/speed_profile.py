# speed_profile.py - "Where will this train be in the next 10 minutes?"
#
# Most trains report a position only every few minutes (the collector runs
# roughly every 9 min; riders are rare). Between reports the app has to
# guess. Carrying the last speed forward works for a minute or two, but it
# can't know that the train is about to slow down for a station, stand there
# for a few minutes, or crawl through a habitual crossing loop.
#
# This learns each train's own *pace profile* along its route from up to 30
# days of history (history_store.run_point): the route is cut into
# CELL_KM-long cells and, for every past run, we measure how long the train
# took to get through each cell. A cell where it normally stops (a station,
# a crossing wait) simply takes longer to cross, so the profile slows and
# "dwells" there by itself - no separate stop model needed. Where riders
# shared dense GPS, real dwell times show up; where only the ~9-minute
# collector data exists, the dwell is spread over the cells around it.
#
# Predicting from a fresh fix: start at the fix and move through the cells
# ahead at this train's historical pace. The live speed is only used to tell
# "stopped right now" apart (then the plan holds still for STOPPED_HOLD_S
# before following history). The result is a short plan of
# (position, seconds-after-fix) knots that the app follows for up to
# HORIZON_S, so every viewer sees the same, history-informed motion.
#
# Why not blend in the live speed? It was the original idea, and
# tools/backtest_motion.py measured it on 7 days of real journeys (~41k
# fixes, profiles learned only from earlier days). Median position error:
#
#                       3 min    5 min   10 min
#   app v45 (2 min const)  866     1897     5060 m
#   const 10 min, stop cap 736     1124     1933 m
#   blend, tau 90 s        655      947     1322 m
#   history only (this)    472      727     1107 m
#
# Blending (any tau) and scaling history by today's recent pace both made it
# worse: the speed the server knows is mostly an average over the
# collector's ~9-minute gaps, while the profile knows where this train speeds
# up and slows down. (The app still eases from its own short-term speed into
# the plan over the first ~30 s, which matters only for riders' 5-second
# updates.) BLEND_TAU_S/pace_ratio stay as parameters for re-testing later.
#
# Fallbacks for cells with too little of this train's own history, in order:
#   1. other trains between the same two consecutive scheduled stops
#      (only trains that stop at both ends - an express that runs through
#      isn't comparable to a local that stops),
#   2. the timetable: the leg's scheduled minutes minus a default dwell,
#      spread over its length, with that dwell added at the stop,
#   3. a flat default speed.

import math
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

from .history_store import BD_TZ, RETENTION_DAYS
from .position_filter import position_to_km

CELL_KM = 0.2
HORIZON_S = 600
PLAN_STEP_S = 30
SIM_STEP_S = 5
BLEND_TAU_S = 0.0           # 0 = history pace only (best in backtest; see header)
STOPPED_KMH = 3.0
STOPPED_HOLD_S = 60
MAX_GAP_S = 25 * 60          # don't learn from stretches between fixes further apart than this
RECENCY_HALF_LIFE_DAYS = 10.0
MIN_SAMPLES = 2
DEFAULT_DWELL_S = 60
DEFAULT_SPEED_KMH = 45.0
MAX_SPEED_KMH = 130.0
MAX_CELL_S = 15 * 60         # a single 200 m cell can't plausibly take longer than this
REFRESH_S = 6 * 3600         # a full rebuild is ~30 s of CPU; runs change once a day anyway
SHARED_BUCKETS = 20          # fraction-of-leg buckets for cross-train sharing


def km_to_position(cum_km: Sequence[float], km: float) -> float:
    """Inverse of position_filter.position_to_km (linear within each stop leg)."""
    n = len(cum_km)
    if n < 2 or km <= cum_km[0]:
        return 0.0
    if km >= cum_km[-1]:
        return float(n - 1)
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cum_km[mid] <= km:
            lo = mid
        else:
            hi = mid
    span = cum_km[lo + 1] - cum_km[lo]
    return lo + ((km - cum_km[lo]) / span if span > 0 else 0.0)


def weighted_median(samples: List[Tuple[float, float]]) -> Optional[float]:
    """samples: [(value, weight)]"""
    if not samples:
        return None
    ordered = sorted(samples)
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, w in ordered:
        acc += w
        if acc >= total / 2:
            return value
    return ordered[-1][0]


def run_cell_times(points: Sequence[Tuple[float, float]], cell_km: float = CELL_KM,
                   max_gap_s: float = MAX_GAP_S) -> Dict[int, float]:
    """
    One run's (ts, km) fixes -> {cell index: seconds spent crossing it}.
    Km is made non-decreasing (GPS jitter backwards is ignored). A cell is
    only measured when both of its boundaries were crossed inside a chain of
    fixes no further apart than max_gap_s.
    """
    out: Dict[int, float] = {}
    if len(points) < 2:
        return out
    chain: Dict[int, float] = {}   # boundary index -> crossing time, current unbroken chain

    def flush():
        for b, t in chain.items():
            if b + 1 in chain:
                out[b] = chain[b + 1] - t
        chain.clear()

    prev_t, prev_km = points[0]
    for t, km in points[1:]:
        km = max(km, prev_km)
        if t - prev_t > max_gap_s:
            flush()
        elif km > prev_km:
            first = math.floor(prev_km / cell_km) + 1
            last = math.floor(km / cell_km)
            for b in range(first, last + 1):
                x = b * cell_km
                chain[b] = prev_t + (x - prev_km) / (km - prev_km) * (t - prev_t)
        prev_t, prev_km = t, km
    flush()
    return out


def compute_profiles(db_path: str, static: Dict[str, Tuple[List[float], List[str], Dict[int, int]]],
                     now: float, since_date: str, cutoff_date: Optional[str] = None):
    """
    The CPU-heavy part of SpeedProfileModel.build, as a plain function so it
    can run in another process. static: {train_id: (cum_km, stop_sids,
    stop_minutes)}. Returns (medians {train_id: {cell: seconds}},
    shared {(stop a, stop b, bucket): s/km}, runs {train_id: count}, rows).
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10)
    except sqlite3.Error as e:
        print(f"SpeedProfileModel: cannot open {db_path}: {e}")
        return {}, {}, {}, 0
    try:
        query = ("SELECT train_id, service_date, ts, position FROM run_point "
                 "WHERE service_date >= ?" + (" AND service_date < ?" if cutoff_date else "") +
                 " ORDER BY train_id, service_date, ts")
        params = (since_date, cutoff_date) if cutoff_date else (since_date,)
        rows = conn.execute(query, params).fetchall()
    except sqlite3.Error as e:
        print(f"SpeedProfileModel: history query failed: {e}")
        rows = []
    finally:
        conn.close()

    profiles: Dict[str, TrainProfile] = {}
    shared: Dict[Tuple[str, str, int], List[Tuple[float, float]]] = {}
    today = datetime.fromtimestamp(now, tz=BD_TZ).date()

    def finish_run(profile: "TrainProfile", date: str, pts: List[Tuple[float, float]]):
        if len(pts) < 2:
            return
        age = max(0, (today - datetime.strptime(date, "%Y-%m-%d").date()).days)
        weight = 0.5 ** (age / RECENCY_HALF_LIFE_DAYS)
        cells = run_cell_times(pts)
        if not cells:
            return
        profile.runs += 1
        for j, secs in cells.items():
            secs = min(secs, MAX_CELL_S)
            profile.samples.setdefault(j, []).append((secs, weight))
            mid_km = (j + 0.5) * CELL_KM
            if mid_km >= profile.total_km:
                continue
            k = profile.leg_of(mid_km)
            leg_len = profile.cum_km[k + 1] - profile.cum_km[k]
            if leg_len <= 0:
                continue
            bucket = min(SHARED_BUCKETS - 1, int((mid_km - profile.cum_km[k]) / leg_len * SHARED_BUCKETS))
            key = (profile.stop_sids[k], profile.stop_sids[k + 1], bucket)
            shared.setdefault(key, []).append((secs / CELL_KM, weight))  # pace, s/km

    current_train, current_date, profile, pts = None, None, None, []
    for train_id, date, ts, position in rows:
        if train_id != current_train:
            if profile is not None:
                finish_run(profile, current_date, pts)
            current_train, current_date, pts = train_id, date, []
            profile = None
            if train_id in static:
                cum_km, stop_sids, stop_minutes = static[train_id]
                profile = profiles[train_id] = TrainProfile(train_id, cum_km, stop_sids, stop_minutes)
        if profile is None:
            continue
        if date != current_date:
            finish_run(profile, current_date, pts)
            current_date, pts = date, []
        pts.append((float(ts), position_to_km(profile.cum_km, float(position))))
    if profile is not None:
        finish_run(profile, current_date, pts)

    medians = {
        tid: {j: weighted_median(v) for j, v in p.samples.items() if len(v) >= MIN_SAMPLES}
        for tid, p in profiles.items()
    }
    shared_medians = {k: weighted_median(v) for k, v in shared.items() if len(v) >= MIN_SAMPLES}
    return medians, shared_medians, {tid: p.runs for tid, p in profiles.items()}, len(rows)


class TrainProfile:
    """Pace knowledge for one train: per-cell crossing time in seconds."""

    def __init__(self, train_id: str, cum_km: List[float], stop_sids: List[str],
                 stop_minutes: Dict[int, int]):
        self.train_id = train_id
        self.cum_km = cum_km
        self.stop_sids = stop_sids
        self.stop_minutes = stop_minutes      # stop index -> scheduled minute-of-day
        self.samples: Dict[int, List[Tuple[float, float]]] = {}   # only while building
        self.medians: Dict[int, float] = {}                       # cell -> seconds (>= MIN_SAMPLES runs)
        self.runs = 0

    @property
    def total_km(self) -> float:
        return self.cum_km[-1] if self.cum_km else 0.0

    def leg_of(self, km: float) -> int:
        """Index k of the stop leg [k, k+1] containing km."""
        pos = km_to_position(self.cum_km, km)
        return min(int(pos), max(0, len(self.cum_km) - 2))


class SpeedProfileModel:
    """
    Builds TrainProfiles for every train from history.db and turns a fresh
    fix into a short motion plan. Rebuilt in a background thread every
    REFRESH_S so that the request path never waits on the database.
    """

    def __init__(self, history_db_path: str, tracker, refresh_s: float = REFRESH_S):
        self.history_db_path = history_db_path
        self.tracker = tracker
        self.refresh_s = refresh_s
        self._profiles: Dict[str, TrainProfile] = {}
        self._shared: Dict[Tuple[str, str, int], float] = {}   # (stop a, stop b, bucket) -> s/km
        self._built_at = 0.0
        self._lock = threading.Lock()
        self._building = False

    # -- building -------------------------------------------------------------------

    def _train_static(self, train_id: str):
        cum_km = self.tracker._get_cumulative_km(train_id)
        stations = (self.tracker.train_data or {}).get("tid_to_stations", {}).get(str(train_id)) or []
        stop_sids = [s[0] for s in stations if len(s) > 1 and s[1] == 1]
        times = self.tracker._get_station_times(train_id) or []
        return cum_km, stop_sids, {k: m for k, m in times}

    def _all_static(self) -> Dict[str, Tuple[List[float], List[str], Dict[int, int]]]:
        out = {}
        for tid in ((self.tracker.train_data or {}).get("tid_to_stations") or {}):
            cum_km, stop_sids, stop_minutes = self._train_static(tid)
            if cum_km and len(stop_sids) == len(cum_km):
                out[str(tid)] = (cum_km, stop_sids, stop_minutes)
        return out

    def build(self, now: Optional[float] = None, cutoff_date: Optional[str] = None,
              since_date: Optional[str] = None, in_subprocess: bool = False) -> None:
        """
        (Re)build all profiles. cutoff_date (exclusive) / since_date
        (inclusive) restrict which service dates are learned from - used by
        the backtest to avoid learning from the runs it then predicts.

        in_subprocess: do the CPU-heavy part (~30 s) in a separate process.
        In the server this matters - a rebuild in a thread holds Python's GIL
        and made API requests take seconds while it ran.
        """
        now = now or time.time()
        since_date = since_date or (
            datetime.fromtimestamp(now, tz=BD_TZ) - timedelta(days=RETENTION_DAYS)
        ).strftime("%Y-%m-%d")
        static = self._all_static()
        args = (self.history_db_path, static, now, since_date, cutoff_date)
        if in_subprocess:
            import concurrent.futures
            import multiprocessing
            ctx = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
                medians, shared_medians, runs, n_rows = pool.submit(compute_profiles, *args).result(timeout=900)
        else:
            medians, shared_medians, runs, n_rows = compute_profiles(*args)

        profiles: Dict[str, TrainProfile] = {}
        for tid, cells in medians.items():
            cum_km, stop_sids, stop_minutes = static[tid]
            p = TrainProfile(tid, cum_km, stop_sids, stop_minutes)
            p.medians = cells
            p.runs = runs.get(tid, 0)
            profiles[tid] = p
        with self._lock:
            self._profiles = profiles
            self._shared = shared_medians
            self._built_at = time.time()
        print(f"SpeedProfileModel: built {len(profiles)} train profiles from {n_rows} history points")

    def _ensure_fresh(self):
        if time.time() - self._built_at < self.refresh_s or self._building:
            return
        self._building = True

        def run():
            try:
                self.build(in_subprocess=True)
            except Exception as e:
                print(f"SpeedProfileModel: build failed: {e}")
            finally:
                self._building = False

        threading.Thread(target=run, name="speed-profile-build", daemon=True).start()

    # -- prediction -----------------------------------------------------------------

    def _profile_for(self, train_id: str) -> Optional[TrainProfile]:
        with self._lock:
            p = self._profiles.get(str(train_id))
        if p is not None:
            return p
        cum_km, stop_sids, stop_minutes = self._train_static(train_id)
        if not cum_km or len(stop_sids) != len(cum_km):
            return None
        return TrainProfile(str(train_id), cum_km, stop_sids, stop_minutes)

    def cell_seconds(self, profile: TrainProfile, j: int) -> Tuple[float, str]:
        """Seconds to cross cell j and which source said so."""
        own = profile.medians.get(j)
        if own is not None:
            return max(own, CELL_KM / MAX_SPEED_KMH * 3600), "own"

        mid_km = (j + 0.5) * CELL_KM
        k = profile.leg_of(mid_km)
        if k + 1 < len(profile.cum_km):
            leg_start, leg_end = profile.cum_km[k], profile.cum_km[k + 1]
            leg_len = leg_end - leg_start
            if leg_len > 0:
                bucket = min(SHARED_BUCKETS - 1, int((mid_km - leg_start) / leg_len * SHARED_BUCKETS))
                with self._lock:
                    shared = self._shared.get((profile.stop_sids[k], profile.stop_sids[k + 1], bucket))
                if shared is not None:
                    return max(shared * CELL_KM, CELL_KM / MAX_SPEED_KMH * 3600), "shared"

                # Timetable: scheduled leg time minus a dwell, which is put
                # back as a hold in the cell holding the next stop.
                a, b = profile.stop_minutes.get(k), profile.stop_minutes.get(k + 1)
                if a is not None and b is not None:
                    leg_s = ((b - a) % 1440) * 60
                    is_last = k + 1 == len(profile.cum_km) - 1
                    dwell = 0 if is_last else DEFAULT_DWELL_S
                    run_s = max(leg_s - dwell, leg_len / MAX_SPEED_KMH * 3600)
                    secs = run_s * CELL_KM / leg_len
                    if j == int(leg_end / CELL_KM) and not is_last:
                        secs += dwell
                    return secs, "timetable"
        return CELL_KM / DEFAULT_SPEED_KMH * 3600, "default"

    def plan(self, train_id: str, position: float, speed_kmh: Optional[float],
             horizon_s: int = HORIZON_S, profile: Optional[TrainProfile] = None,
             blend_tau_s: float = BLEND_TAU_S, pace_ratio: float = 1.0,
             stopped_hold: bool = True) -> List[List[float]]:
        """
        Motion plan from a fix at `position`: [[position, seconds_after_fix], ...]
        every PLAN_STEP_S up to horizon_s. Empty if the train has no route data.
        """
        self._ensure_fresh()
        profile = profile or self._profile_for(train_id)
        if profile is None or len(profile.cum_km) < 2:
            return []
        km_end = profile.total_km
        km = min(position_to_km(profile.cum_km, float(position)), km_end)
        v_now = None if speed_kmh is None else max(0.0, float(speed_kmh)) / 3600.0  # km/s
        stopped = stopped_hold and speed_kmh is not None and speed_kmh < STOPPED_KMH

        out = [[round(km_to_position(profile.cum_km, km), 5), 0]]
        t = 0.0
        hold_until = float(STOPPED_HOLD_S) if stopped else 0.0
        for knot_t in range(PLAN_STEP_S, horizon_s + 1, PLAN_STEP_S):
            # Advance from t to knot_t exactly cell by cell: each cell is
            # crossed at its own pace, so a slow "station" cell is never
            # partly skipped by stepping over its start.
            while t < knot_t - 1e-9 and km < km_end:
                if t < hold_until:
                    t = min(hold_until, float(knot_t))
                    continue
                j = int(km / CELL_KM + 1e-9)
                secs, _ = self.cell_seconds(profile, j)
                v = CELL_KM / max(secs, 1.0) * pace_ratio             # km/s
                dt_max = knot_t - t
                if v_now is not None and not stopped and blend_tau_s > 0:
                    w = math.exp(-t / blend_tau_s)
                    if w > 0.01:
                        v = w * v_now + (1 - w) * v
                        dt_max = min(dt_max, SIM_STEP_S)
                cell_end = min((j + 1) * CELL_KM, km_end)
                need = (cell_end - km) / v if v > 0 else float("inf")
                if need <= dt_max:
                    km, t = cell_end, t + need
                else:
                    km, t = km + v * dt_max, t + dt_max
            t = float(knot_t)
            out.append([round(km_to_position(profile.cum_km, km), 5), knot_t])
        return out

    def summary(self) -> Dict:
        with self._lock:
            return {
                "trains": len(self._profiles),
                "cells": sum(len(p.medians) for p in self._profiles.values()),
                "shared_keys": len(self._shared),
                "built_at": int(self._built_at),
            }
