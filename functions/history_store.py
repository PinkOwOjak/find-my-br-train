# history_store.py - 30 days of per-train journey history, and predictions
# built from it.
#
# Two kinds of rows, both derived from the *aggregated* train position (the
# same value the app shows), never from raw individual pings:
#
#   run_point      one sample per train every ~30 s: position, km, speed,
#                  delay, reporter count. Compact trace for analysis/replay.
#   station_event  the moment the train reached each scheduled stop
#                  (interpolated between the two fixes either side of it)
#                  and its delay there. This is what predictions use.
#
# A "run" is one journey: train_id + service_date, where service_date is the
# Bangladesh calendar date of that run's scheduled first departure. Runs that
# cross midnight keep the date they *started* on, so an overnight train's
# 23:30 departure and its 06:00 arrival belong to the same run.
#
# Privacy: only train-level aggregates are stored. No user ids, no
# individual GPS traces.
#
# Storage is a local SQLite file (WAL) - no extra server infrastructure.
# Rows older than RETENTION_DAYS are pruned automatically.

import math
import os
import sqlite3
import statistics
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from .position_filter import position_to_delay_minutes

BD_TZ = ZoneInfo("Asia/Dhaka")
DB_PATH = os.getenv("HISTORY_DB_PATH", "history.db")

RETENTION_DAYS = 30
POINT_INTERVAL_S = 30
MAX_CROSSING_GAP_S = 45 * 60     # don't interpolate a stop crossing across a gap longer than this
MIN_RUNS_FOR_HISTORY = 2         # need at least this many past runs to trust a learned delay change
PRUNE_INTERVAL_S = 3600
# Per-station delay when the train wasn't seen exactly at the station:
STOP_INTERPOLATE_MAX_GAP_S = 4 * 3600   # interpolate between fixes either side up to this far apart
STOP_ESTIMATE_MAX_STOPS = 1.5           # else borrow the delay of a fix at most this far away (stop units)
PENDING_WINDOW_S = 3 * 3600             # a run seen this recently hasn't necessarily reached the station yet
# Bangladesh Railway trains don't leave stations ahead of time, so a train
# "arriving" more than this early is a bad report (e.g. someone sharing the
# opposite-direction train, or a ping filed under the wrong run) - ~2.7% of
# recorded arrivals. Such values are ignored when reading history (kept in
# the database, so the threshold can be changed later).
MIN_PLAUSIBLE_DELAY_MIN = -30

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_point (
    train_id     TEXT    NOT NULL,
    service_date TEXT    NOT NULL,
    ts           INTEGER NOT NULL,
    position     REAL    NOT NULL,
    km           REAL,
    speed_kmh    REAL,
    delay_min    REAL,
    users        INTEGER,
    PRIMARY KEY (train_id, ts)
);
CREATE TABLE IF NOT EXISTS station_event (
    train_id     TEXT    NOT NULL,
    service_date TEXT    NOT NULL,
    stop_idx     INTEGER NOT NULL,   -- index among the train's scheduled (type-1) stops
    ts           INTEGER NOT NULL,   -- when the train reached it (unix seconds)
    delay_min    REAL    NOT NULL,   -- + late / - early vs the timetable
    PRIMARY KEY (train_id, service_date, stop_idx)
);
CREATE INDEX IF NOT EXISTS idx_run_point_run ON run_point(train_id, service_date);
CREATE INDEX IF NOT EXISTS idx_station_event_date ON station_event(service_date);
"""


# ---------------------------------------------------------------------------
# Timetable helpers (pure)
# ---------------------------------------------------------------------------

def minute_of_day(ts: float) -> float:
    dt = datetime.fromtimestamp(ts, tz=BD_TZ)
    return dt.hour * 60 + dt.minute + dt.second / 60.0


def shifted_schedule(station_times: Sequence[Tuple[int, int]]) -> List[float]:
    """Minutes since first departure for each timed stop (wrapping midnight)."""
    out = [0.0]
    prev = station_times[0][1]
    for _, m in station_times[1:]:
        out.append(out[-1] + (m - prev) % 1440)
        prev = m
    return out


def run_start_ts(station_times: Sequence[Tuple[int, int]], ts: float) -> float:
    """
    Unix time of the scheduled first departure of the run that `ts` belongs
    to - the most recent scheduled departure at or before `ts` (same
    "minutes since the most recent departure" rule the tracker's journey
    window uses, so both agree on which run a ping is part of).
    """
    first_dep = station_times[0][1]
    since = (minute_of_day(ts) - first_dep) % 1440
    return ts - since * 60.0


def service_date_for(station_times: Sequence[Tuple[int, int]], ts: float) -> str:
    start = run_start_ts(station_times, ts)
    return datetime.fromtimestamp(start + 1, tz=BD_TZ).strftime("%Y-%m-%d")


def stop_crossings(prev_pos: float, prev_ts: float, pos: float, ts: float,
                   stop_count: int) -> List[Tuple[int, float]]:
    """
    Scheduled stops passed between two consecutive fixes, with interpolated
    times. A stop k counts as reached when the position first gets to k
    (for the origin, k=0, when it first leaves it - its departure).
    """
    if ts <= prev_ts or pos <= prev_pos or ts - prev_ts > MAX_CROSSING_GAP_S:
        return []
    out = []
    for k in range(stop_count):
        threshold = 0.005 if k == 0 else float(k)
        if prev_pos < threshold <= pos:
            frac = (threshold - prev_pos) / (pos - prev_pos)
            out.append((k, prev_ts + frac * (ts - prev_ts)))
    return out


class HistoryStore:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._last_point_ts: Dict[str, float] = {}
        self._last_prune = 0.0

    # -- connection -----------------------------------------------------------

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(_SCHEMA)
            conn.commit()
            self._conn = conn
        return self._conn

    # -- writing --------------------------------------------------------------

    def record(self, train_id: str, station_times: Optional[Sequence[Tuple[int, int]]],
               stop_count: int, prev: Optional[Dict], position: float, ts: int,
               km: Optional[float], speed_kmh: Optional[float], users: int) -> None:
        """
        Record one aggregated fix. `prev` is the previous aggregated fix
        ({"pos", "ts"}) from the motion state, used to detect stop crossings.
        Best-effort: never raises.
        """
        try:
            if not station_times or len(station_times) < 2:
                return
            service_date = service_date_for(station_times, ts)
            delay = position_to_delay_minutes(station_times, position, minute_of_day(ts))
            events = []
            if prev and prev.get("pos") is not None and prev.get("ts") is not None:
                if service_date_for(station_times, prev["ts"]) == service_date:
                    for k, t in stop_crossings(prev["pos"], prev["ts"], position, ts, stop_count):
                        d = position_to_delay_minutes(station_times, float(k), minute_of_day(t))
                        if d is not None:
                            events.append((train_id, service_date, k, int(t), round(d, 1)))

            write_point = ts - self._last_point_ts.get(train_id, 0) >= POINT_INTERVAL_S
            with self._lock:
                db = self._db()
                if write_point:
                    db.execute(
                        "INSERT OR IGNORE INTO run_point VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (train_id, service_date, int(ts), round(position, 4),
                         None if km is None else round(km, 3), speed_kmh,
                         None if delay is None else round(delay, 1), users),
                    )
                    self._last_point_ts[train_id] = ts
                if events:
                    # First crossing wins - a later GPS wobble back and forth
                    # across the same stop mustn't overwrite the real time.
                    db.executemany("INSERT OR IGNORE INTO station_event VALUES (?, ?, ?, ?, ?)", events)
                db.commit()
            self._maybe_prune()
        except Exception as e:
            print(f"Warning: history record failed for train {train_id}: {e}")

    def _maybe_prune(self):
        now = time.time()
        if now - self._last_prune < PRUNE_INTERVAL_S:
            return
        self._last_prune = now
        cutoff_ts = int(now - RETENTION_DAYS * 86400)
        cutoff_date = (datetime.now(tz=BD_TZ) - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
        with self._lock:
            db = self._db()
            db.execute("DELETE FROM run_point WHERE ts < ?", (cutoff_ts,))
            db.execute("DELETE FROM station_event WHERE service_date < ?", (cutoff_date,))
            db.commit()

    # -- reading --------------------------------------------------------------

    def _runs(self, train_id: str, since_date: str, exclude_date: Optional[str] = None) -> Dict[str, Dict[int, float]]:
        with self._lock:
            rows = self._db().execute(
                "SELECT service_date, stop_idx, delay_min FROM station_event "
                "WHERE train_id = ? AND service_date >= ? ORDER BY service_date, stop_idx",
                (train_id, since_date),
            ).fetchall()
        runs: Dict[str, Dict[int, float]] = {}
        for date, k, d in rows:
            if date != exclude_date and d >= MIN_PLAUSIBLE_DELAY_MIN:
                runs.setdefault(date, {})[k] = d
        return runs

    # -- per-day helpers -------------------------------------------------------------

    @staticmethod
    def window_dates(days: int, now: float, offdays: Optional[Sequence[str]] = None) -> Tuple[List[str], List[str]]:
        """
        The last `days` service dates up to today (oldest first), minus the
        train's weekly off days. Returns (dates, skipped_offday_dates).
        """
        off = {str(d).strip().lower()[:3] for d in (offdays or [])}
        today = datetime.fromtimestamp(now, tz=BD_TZ).date()
        dates, skipped = [], []
        for i in range(days - 1, -1, -1):
            d = today - timedelta(days=i)
            (skipped if d.strftime("%a").lower() in off else dates).append(d.strftime("%Y-%m-%d"))
        return dates, skipped

    def _run_points(self, train_id: str, since_date: str) -> Dict[str, List[Tuple[float, float]]]:
        """{service_date: [(ts, position), ...]} with position made non-decreasing."""
        with self._lock:
            rows = self._db().execute(
                "SELECT service_date, ts, position FROM run_point WHERE train_id = ? AND service_date >= ? "
                "ORDER BY service_date, ts", (train_id, since_date),
            ).fetchall()
        out: Dict[str, List[Tuple[float, float]]] = {}
        for date, ts, pos in rows:
            pts = out.setdefault(date, [])
            pts.append((float(ts), max(float(pos), pts[-1][1] if pts else float(pos))))
        return out

    @staticmethod
    def stop_delay_from_points(points: Sequence[Tuple[float, float]], station_times: Sequence[Tuple[int, int]],
                               stop_idx: int) -> Optional[Tuple[float, str]]:
        """
        Delay at scheduled stop `stop_idx` from a run's position fixes when
        no exact crossing was recorded:
          interpolated - fixes on both sides of the station (<= 4 h apart):
                         the passing time is interpolated between them;
          estimated    - fixes on one side only, within 1.5 stops: that
                         fix's own delay is assumed to hold at the station.
        """
        if not points:
            return None
        before = next((p for p in reversed(points) if p[1] <= stop_idx), None)
        after = next((p for p in points if p[1] >= stop_idx), None)
        if before and after and after[0] - before[0] <= STOP_INTERPOLATE_MAX_GAP_S:
            (tb, pb), (ta, pa) = before, after
            t = tb if pa <= pb else tb + (stop_idx - pb) / (pa - pb) * (ta - tb)
            d = position_to_delay_minutes(station_times, float(stop_idx), minute_of_day(t))
            return (d, "interpolated") if d is not None and d >= MIN_PLAUSIBLE_DELAY_MIN else None
        candidates = [p for p in (before, after) if p and abs(p[1] - stop_idx) <= STOP_ESTIMATE_MAX_STOPS]
        if not candidates:
            return None
        ts, pos = min(candidates, key=lambda p: abs(p[1] - stop_idx))
        d = position_to_delay_minutes(station_times, pos, minute_of_day(ts))
        return (d, "estimated") if d is not None and d >= MIN_PLAUSIBLE_DELAY_MIN else None

    def _stop_value(self, events: Dict[int, float], points, station_times, stop_idx):
        if stop_idx in events:
            return events[stop_idx], "observed"
        if station_times:
            return self.stop_delay_from_points(points or [], station_times, stop_idx)
        return None

    @staticmethod
    def _pending(points, stop_idx: int, now: float) -> bool:
        """A run still in progress that hasn't reached this stop yet."""
        return bool(points) and now - points[-1][0] < PENDING_WINDOW_S and points[-1][1] < stop_idx

    @staticmethod
    def _summary(rows: List[Dict]) -> Dict:
        values = [r["delay_min"] for r in rows if r["delay_min"] is not None]
        return {
            "runs": len(values),
            "missing_days": sum(1 for r in rows if r["delay_min"] is None),
            "avg_delay_min": round(statistics.mean(values)) if values else None,
            "max_delay_min": max(values) if values else None,
            "min_delay_min": min(values) if values else None,
        }

    def stop_delay_history(self, train_id: str, station_times: Sequence[Tuple[int, int]], stop_idx: int,
                           days: int = 7, offdays: Optional[Sequence[str]] = None,
                           now: Optional[float] = None) -> Dict:
        """
        Delay at one scheduled stop for each of the last `days` days. Every
        running day is listed - delay_min is null on days with no usable
        data - and the train's off days are left out entirely.
        """
        days = max(1, min(RETENTION_DAYS, int(days)))
        now = now or time.time()
        dates, skipped = self.window_dates(days, now, offdays)
        runs = self._runs(train_id, dates[0]) if dates else {}
        points = self._run_points(train_id, dates[0]) if dates else {}
        rows = []
        for date in dates:
            value = self._stop_value(runs.get(date, {}), points.get(date), station_times, stop_idx)
            if value is None and self._pending(points.get(date), stop_idx, now):
                continue                          # today's run just hasn't got there yet
            if value is None and date == dates[-1] and date not in points and date not in runs:
                continue                          # today, nothing recorded yet - not "missing"
            rows.append({
                "date": date,
                "delay_min": None if value is None else round(value[0]),
                "source": None if value is None else value[1],
            })
        return {"train_id": train_id, "stop": stop_idx, "days": days,
                "offdays_skipped": skipped, "daily": rows, **self._summary(rows)}

    def delay_summary(self, train_id: str, days: int = 7, now: Optional[float] = None,
                      station_times: Optional[Sequence[Tuple[int, int]]] = None,
                      stop_count: Optional[int] = None, offdays: Optional[Sequence[str]] = None) -> Dict:
        """
        Per-run delay over the last `days` days: the delay at the destination
        (observed, interpolated or estimated as in stop_delay_history), else
        at the furthest stop we have ("partial"). Every running day is
        listed (delay_min null when there's no data); off days are skipped.
        """
        days = max(1, min(RETENTION_DAYS, int(days)))
        now = now or time.time()
        dates, skipped = self.window_dates(days, now, offdays)
        runs = self._runs(train_id, dates[0]) if dates else {}
        points = self._run_points(train_id, dates[0]) if dates else {}
        dest = (stop_count - 1) if stop_count else None
        day_rows = []
        for date in dates:
            stops = runs.get(date, {})
            value = self._stop_value(stops, points.get(date), station_times, dest) if dest is not None else None
            if value is None and stops:
                last = max(stops)
                value = (stops[last], "observed" if last == dest else "partial")
            if value is None and dest is not None and self._pending(points.get(date), dest, now):
                continue
            if value is None and date == dates[-1] and date not in points and not stops:
                continue
            day_rows.append({
                "date": date,
                "delay_min": None if value is None else round(value[0]),
                "source": None if value is None else value[1],
                "last_stop": max(stops) if stops else None,
                "stops_recorded": len(stops),
            })
        per_stop: Dict[int, List[float]] = {}
        for date, stops in runs.items():
            if date in dates:
                for k, d in stops.items():
                    per_stop.setdefault(k, []).append(d)
        return {
            "train_id": train_id,
            "days": days,
            "offdays_skipped": skipped,
            **self._summary(day_rows),
            "daily": day_rows,
            "per_stop": [
                {"stop": k, "median_delay_min": round(statistics.median(v)), "samples": len(v)}
                for k, v in sorted(per_stop.items())
            ],
        }

    def predict(self, train_id: str, station_times: Sequence[Tuple[int, int]], stop_count: int,
                position: float, fix_ts: float, now: Optional[float] = None) -> Dict:
        """
        Predicted delay and ETA for every scheduled stop still ahead of
        `position`. Starts from the delay the train has right now and adds
        how that delay has typically *changed* between here and each later
        stop on past runs (median over up to 30 days) - recovery time in the
        timetable, habitual crossing waits on single-line sections, slow
        approaches into Dhaka, and so on. With too little history the delay
        is simply carried forward unchanged (the old behaviour).
        """
        now = now or time.time()
        current_delay = position_to_delay_minutes(station_times, position, minute_of_day(fix_ts))
        if current_delay is None:
            return {"train_id": train_id, "stops": [], "history_runs": 0}

        current_date = service_date_for(station_times, fix_ts)
        since = (datetime.fromtimestamp(now, tz=BD_TZ) - timedelta(days=RETENTION_DAYS)).strftime("%Y-%m-%d")
        runs = self._runs(train_id, since, exclude_date=current_date)

        last_passed = max(0, min(stop_count - 1, int(math.floor(position + 1e-9))))
        start = run_start_ts(station_times, fix_ts)
        shifted = shifted_schedule(station_times)
        timed_index = {k: i for i, (k, _) in enumerate(station_times)}

        stops = []
        for k in range(last_passed + 1, stop_count):
            if k not in timed_index:
                continue
            changes = []
            for stops_r in runs.values():
                if k not in stops_r:
                    continue
                ref = [j for j in stops_r if j <= last_passed]
                if ref:
                    changes.append(stops_r[k] - stops_r[max(ref)])
            if len(changes) >= MIN_RUNS_FOR_HISTORY:
                predicted = current_delay + statistics.median(changes)
                source = "history"
            else:
                predicted = current_delay
                source = "carry"
            # Trains wait at stops rather than run far ahead of the timetable.
            predicted = max(predicted, min(current_delay, 0.0) - 5.0)
            eta = start + shifted[timed_index[k]] * 60.0 + predicted * 60.0
            stops.append({
                "stop": k,
                "predicted_delay_min": round(predicted),
                "eta": int(eta),
                "samples": len(changes),
                "source": source,
            })
        return {
            "train_id": train_id,
            "based_on": {"position": position, "timestamp": int(fix_ts), "delay_min": round(current_delay)},
            "history_runs": len(runs),
            "stops": stops,
        }

    def close(self):
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
