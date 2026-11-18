#!/usr/bin/env python3
"""
backtest_motion.py - How well do we guess where a train is between reports?

Replays recorded journeys from history.db. For every real position fix it
predicts where the train will be 3, 5 and 10 minutes later and compares
with where the recorded data says it actually was. Profiles are learned only
from days *before* the test window, so no run is predicted with knowledge of
itself.

Methods compared:
  hold          marker frozen at the last fix
  app_2min      constant speed for 2 min, never past the next stop (app v45 behaviour)
  const_10min   constant speed for the whole horizon, never past the next stop
  profile       functions/speed_profile.py plan (live speed blended into history)

Usage:
    python3 tools/backtest_motion.py --history history.db [--test-days 7] [--train-days 23]
"""

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from functions.history_store import BD_TZ  # noqa: E402
from functions.position_filter import position_to_km  # noqa: E402
from functions.redis_tracker import RedisTrainTracker  # noqa: E402
from functions.speed_profile import SpeedProfileModel  # noqa: E402

HORIZONS = (180, 300, 600)
APP_EXTRAPOLATION_S = 120
MIN_ORIGIN_SPACING_S = 120
MAX_TRUTH_GAP_S = 600
SPEED_WINDOW_S = 60
MAX_SPEED_GAP_S = 15 * 60


def interp(points, t):
    """km at time t from (ts, km) points, if bracketed by fixes <= MAX_TRUTH_GAP_S apart."""
    lo, hi = 0, len(points) - 1
    if t < points[0][0] or t > points[-1][0]:
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if points[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    (t0, k0), (t1, k1) = points[lo], points[hi]
    if t1 - t0 > MAX_TRUTH_GAP_S:
        return None
    if t1 == t0:
        return k0
    return k0 + (k1 - k0) * (t - t0) / (t1 - t0)


def plan_km_at(plan, cum_km, h):
    for (p0, s0), (p1, s1) in zip(plan, plan[1:]):
        if s0 <= h <= s1:
            k0, k1 = position_to_km(cum_km, p0), position_to_km(cum_km, p1)
            return k0 + (k1 - k0) * ((h - s0) / (s1 - s0) if s1 > s0 else 0)
    return position_to_km(cum_km, plan[-1][0])


def pct(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default=os.path.join(root, "history.db"))
    ap.add_argument("--data", default=os.path.join(root, "data.json"))
    ap.add_argument("--test-days", type=int, default=7)
    ap.add_argument("--train-days", type=int, default=23)
    ap.add_argument("--json", default=None, help="also write results here")
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        data = json.load(f)
    tracker = RedisTrainTracker()          # timetable helpers only - no Redis traffic
    tracker.set_train_data(data)

    now = time.time()
    today = datetime.fromtimestamp(now, tz=BD_TZ)
    test_start = (today - timedelta(days=args.test_days)).strftime("%Y-%m-%d")
    train_start = (today - timedelta(days=args.test_days + args.train_days)).strftime("%Y-%m-%d")

    model = SpeedProfileModel(args.history, tracker, refresh_s=10 ** 9)
    t0 = time.time()
    model.build(now=now, cutoff_date=test_start, since_date=train_start)
    print(f"Learned from {train_start} .. <{test_start} in {time.time() - t0:.1f}s: {model.summary()}")

    import sqlite3
    conn = sqlite3.connect(f"file:{args.history}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT train_id, service_date, ts, position FROM run_point WHERE service_date >= ? "
        "ORDER BY train_id, service_date, ts", (test_start,)).fetchall()
    conn.close()

    runs = defaultdict(list)
    for tid, date, ts, pos in rows:
        runs[(tid, date)].append((float(ts), float(pos)))

    errors = defaultdict(list)          # (method, horizon, density) -> [metres]
    sources = defaultdict(int)
    origins = 0
    t0 = time.time()
    for (tid, date), pts in runs.items():
        cum_km = tracker._get_cumulative_km(tid)
        if not cum_km or len(pts) < 3:
            continue
        profile = model._profile_for(tid)
        if profile is None:
            continue
        kms, best = [], 0.0
        for ts, pos in pts:
            best = max(best, position_to_km(cum_km, pos))
            kms.append((ts, best))
        gaps = [b[0] - a[0] for a, b in zip(kms, kms[1:])]
        density = "dense (riders)" if statistics.median(gaps) <= 60 else "sparse (collector)"

        last_origin = -1e18
        for i in range(1, len(kms)):
            t_i, km_i = kms[i]
            if t_i - last_origin < MIN_ORIGIN_SPACING_S:
                continue
            # speed as the server would know it: progress over the last minute or more
            j = i - 1
            while j > 0 and t_i - kms[j][0] < SPEED_WINDOW_S:
                j -= 1
            dt = t_i - kms[j][0]
            if dt <= 0 or dt > MAX_SPEED_GAP_S:
                continue
            v = (km_i - kms[j][1]) / dt          # km/s
            truths = {h: interp(kms, t_i + h) for h in HORIZONS}
            if all(x is None for x in truths.values()):
                continue
            last_origin = t_i
            origins += 1
            next_stop = next((c for c in cum_km if c > km_i + 0.02), cum_km[-1])
            plan = model.plan(tid, position=_km_to_pos(cum_km, km_i), speed_kmh=v * 3600, profile=profile)
            for c in range(int(km_i / 0.2), int(km_i / 0.2) + 5):
                sources[model.cell_seconds(profile, c)[1]] += 1
            for h, truth in truths.items():
                if truth is None:
                    continue
                preds = {
                    "hold": km_i,
                    "app_2min": min(km_i + v * min(h, APP_EXTRAPOLATION_S), next_stop),
                    "const_10min": min(km_i + v * h, next_stop),
                    "profile": plan_km_at(plan, cum_km, h),
                }
                for method, km_pred in preds.items():
                    err = abs(km_pred - truth) * 1000
                    errors[(method, h, density)].append(err)
                    errors[(method, h, "all")].append(err)

    print(f"Predicted from {origins} fixes in {time.time() - t0:.1f}s "
          f"(cells ahead came from: {dict(sources)})\n")
    methods = ("hold", "app_2min", "const_10min", "profile")
    results = {}
    for density in ("all", "dense (riders)", "sparse (collector)"):
        print(f"== {density}: median / p90 error in metres (n = fixes)")
        print(f"{'horizon':>8}  " + "  ".join(f"{m:>18}" for m in methods))
        for h in HORIZONS:
            cells = []
            for m in methods:
                e = errors.get((m, h, density), [])
                cells.append(f"{pct(e, 0.5):7.0f} / {pct(e, 0.9):6.0f}" if e else f"{'-':>18}")
                results[f"{density}|{h}|{m}"] = {"n": len(e), "median_m": pct(e, 0.5), "p90_m": pct(e, 0.9)}
            n = len(errors.get(("hold", h, density), []))
            print(f"{h // 60:>5} min  " + "  ".join(f"{c:>18}" for c in cells) + f"   n={n}")
        print()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)


def _km_to_pos(cum_km, km):
    from functions.speed_profile import km_to_position
    return km_to_position(cum_km, km)


if __name__ == "__main__":
    main()
