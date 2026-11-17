#!/usr/bin/env python3
"""
backfill_history.py - Seed history.db from the existing GPS log
(gps_updates.db, written by functions/gps_log.py since well before the
history store existed), so delay statistics and ETA prediction have data on
day one instead of after 30 days of fresh collection.

Replays every *accepted* ping from the last N days, per train in time order,
through the same HistoryStore.record() the live server uses. Individual
pings are noisier than the server's aggregated position, but stop events
keep only the first crossing of each stop per run, which is robust to that.

Safe to re-run: rows are INSERT OR IGNORE.

Usage (on the server, from the repo root):
    python3 tools/backfill_history.py [--gps gps_updates.db] [--history history.db] [--days 30]
"""

import argparse
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions.history_store import RETENTION_DAYS, HistoryStore  # noqa: E402
from functions.redis_tracker import RedisTrainTracker  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gps", default=os.getenv("GPS_LOG_DB_PATH", "gps_updates.db"))
    p.add_argument("--history", default=os.getenv("HISTORY_DB_PATH", "history.db"))
    p.add_argument("--data", default="data.json")
    p.add_argument("--days", type=int, default=RETENTION_DAYS)
    args = p.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    # Only the timetable helpers are used - no Redis traffic.
    tracker = RedisTrainTracker()
    tracker.set_train_data(data)

    since = int(time.time()) - args.days * 86400
    src = sqlite3.connect(f"file:{args.gps}?mode=ro", uri=True)
    rows = src.execute(
        "SELECT train_id, timestamp, position, position_km FROM gps_updates "
        "WHERE accepted = 1 AND timestamp >= ? ORDER BY train_id, timestamp",
        (since,),
    )
    store = HistoryStore(args.history)
    store._last_prune = time.time()  # don't prune mid-backfill

    count, trains = 0, set()
    current, prev, station_times, stop_count = None, None, None, 0
    for train_id, ts, position, km in rows:
        if train_id != current:
            current, prev = train_id, None
            station_times = tracker._get_station_times(train_id)
            stop_count = tracker._stop_count(train_id)
            trains.add(train_id)
        if not station_times:
            continue
        if prev is not None and ts <= prev["ts"]:
            continue
        store.record(train_id, station_times, stop_count, prev, float(position), int(ts), km, None, 1)
        prev = {"pos": float(position), "ts": int(ts)}
        count += 1
        if count % 20000 == 0:
            print(f"  ... {count} pings")

    events = store._db().execute("SELECT COUNT(*) FROM station_event").fetchone()[0]
    runs = store._db().execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT train_id, service_date FROM station_event)"
    ).fetchone()[0]
    store.close()
    print(f"Replayed {count} pings for {len(trains)} trains -> {events} stop events across {runs} runs "
          f"in {args.history}")


if __name__ == "__main__":
    main()
