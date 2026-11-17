# redis_tracker.py - Redis-based train position tracking

import json
import time
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import redis

from . import gps_log
from .history_store import HistoryStore
from .motion import combine_device_speeds, update_motion
from .track_geometry import route_leg_lengths_km
from .position_filter import (
    build_cumulative_km,
    build_cumulative_km_from_legs,
    position_to_km,
    check_teleport,
    cluster_pings_by_km,
    position_to_delay_minutes,
    welford_update,
    welford_stddev,
)

# Bangladesh timezone - train schedules are in this timezone
BD_TZ = ZoneInfo("Asia/Dhaka")


class RedisTrainTracker:
    """
    Redis-based train position tracker.
    - Stores last ping from each user per train
    - Auto-expires user data after 10 minutes (TTL)
    - Pre-calculates and caches a position when new data arrives, using
      distance-clustered consensus when several reports disagree
    - Stores last known position for 10 hours (fallback when no active users)
    - Every ping - bot or user - is run through the same plausibility filter
      (functions/position_filter.py): journey-active window, then a real-km
      teleport/speed check against the train's last known position. An
      authenticated "bot" caller (see urls/positions.py) is a trusted
      *identity*, not a trusted *position* - it gets no bypass. (A "can't be
      ahead of the timetable" ceiling was tried and dropped as a hard gate -
      real schedules pad different legs very unevenly, so an on-time train
      can legitimately look tens of percent "ahead" in km on a fast
      non-stop stretch. position_km/scheduled_km are still logged per ping,
      so the same "how far ahead of schedule" signal is available to a
      future ML model - it's just not used as a reject condition here.)
    - The teleport check can't tell a good reference from a bad one - it only
      knows "consistent with what's already believed" or not - so a single
      wrong report (e.g. someone selects the wrong train by mistake) could
      otherwise become the reference and then get a *correct* report from
      someone else rejected instead. It's only enforced against a different
      reporter once the reference has held (unsuperseded) for
      confirmation_grace_seconds; a lone, freshly-established one gives way
      to a disagreeing new reporter. Always enforced against the *same*
      reporter regardless of confidence - that case is a single source
      contradicting only itself, exactly what the check is for.
    - Calculates scheduled position automatically from train data
    - Persists across server restarts (stored in Redis)
    - Data validity: positions older than 10 hours are considered invalid
    - Logs every accepted/rejected ping to a local SQLite DB (gps_log.py) as
      training data for a future ML-based fake-GPS classifier
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0,
                 ttl_seconds: int = 600, last_known_ttl: int = 36000):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl = ttl_seconds            # 10 minutes for active user data
        self.last_known_ttl = last_known_ttl  # 10 hours for last known position/bot data
        self.max_valid_age = 36000        # 10 hours in seconds - data older than this is invalid
        self.pre_departure_window = 30    # Minutes before scheduled departure that pings are accepted
        self.max_delay_allowance = 480    # Minutes past scheduled arrival that pings are still accepted

        # Plausibility-filter tuning (see functions/position_filter.py). These
        # are deliberately generous: false-accepts get cleaned up by the next
        # real report, false-rejects lock a real passenger out of the feature.
        self.max_train_speed_kmh = 130.0      # Generous ceiling for BR intercity speeds
        self.position_slack_km = 1.0          # GPS jitter + multi-bogie spread + interpolation error
        self.cluster_tolerance_km = 1.5       # Reports within this of each other count as agreeing
        # How long a reference must hold (unsuperseded) before it's enforced
        # against a *different* reporter - see push()/_update_cached_position.
        # Short enough that a real, prompt correction (a genuine passenger
        # reporting soon after someone else's mistake) still gets through;
        # long enough that an established, still-uncontested trend isn't
        # casually knocked over by one new disagreeing claim.
        self.confirmation_grace_seconds = 240
        # Sustained-backward-drift detection (see push()): a single backward
        # step this small is ordinary GPS jitter; several of them in a row
        # from the same source is a real, sustained trend a real train
        # wouldn't produce along its own scheduled direction.
        self.backward_drift_epsilon_km = 0.3
        self.backward_drift_streak_limit = 3

        # Escape valve for a *stuck* reference (see _check_shadow_consensus):
        # once confirmed, a reference can never be dislodged by elapsed time
        # alone if it was simply wrong to begin with - the "impossible jump"
        # only grows. A rolling buffer of recent teleport-rejections lets a
        # run of mutually-consistent ones (an internally coherent trend that
        # just happens to disagree with the stuck reference) out-vote it.
        # Diversity requirement (2+ reporters, or the run's own time span)
        # stops one identity from forcing a re-anchor by spamming a handful
        # of fake values in quick succession.
        self.shadow_buffer_size = 5
        self.shadow_min_consistent_size = 3
        self.shadow_min_diversity_seconds = 600
        # Someone standing at a station with "I'm inside" switched on (not
        # actually riding - the app already down-weights this in the
        # multi-reporter average below) reports the same near-station
        # position indefinitely. Trivially "continuing" its own last value
        # (nothing moved, so nothing contradicted it either) would otherwise
        # let it accumulate confirmation_grace_seconds confidence just like
        # a real, progressing trend does - then it blocks the train's real
        # movement once someone actually boards. See _update_cached_position.
        self.near_station_confidence_epsilon = 0.02

        # Per-train historical baselines (Welford running mean/stddev over
        # every *accepted, genuinely plausible* ping for that train - see
        # _update_train_stats), replacing flat universal constants with
        # numbers earned from that specific train's own history. Only ever
        # WIDENS the speed ceiling above max_train_speed_kmh, never narrows
        # it - a train we don't have enough history on yet still gets the
        # same protection as before, and a noisy/short baseline can only
        # make the check more permissive, not stricter. Needs a minimum
        # sample size before it's trusted at all, since a baseline built
        # from a handful of points has a wide, unreliable stddev estimate.
        self.min_speed_baseline_samples = 20
        self.speed_baseline_stddev_multiplier = 4.0
        # Delay baseline is logged only for now (delay_zscore below), not
        # enforced - the schedule-ceiling check above already taught us that
        # a time/schedule-based hard reject can go wrong in ways that only
        # show up against real traffic. Once there's evidence a per-train
        # delay baseline actually behaves well, it can graduate to a check.
        self.min_delay_baseline_samples = 10

        self.train_data = None            # Reference to train schedule data
        self._cum_km_cache: Dict[str, Optional[List[float]]] = {}  # train_id -> cumulative km per stop

        # Reporters' own GPS speed is only blended into the train's speed
        # while it's this fresh (see _update_cached_position).
        self.device_speed_max_age = 60
        # 30-day journey history (functions/history_store.py) - optional,
        # best-effort; position tracking never depends on it.
        self.history: Optional[HistoryStore] = None
        # Redis pub/sub channel announcing "this train's position changed",
        # so every worker's LiveHub can push it to its socket subscribers.
        self.updates_channel = "live:train_updates"

    def set_train_data(self, data: Dict):
        """Set reference to train schedule data for scheduled position calculations"""
        self.train_data = data
        self._cum_km_cache = {}  # Route/schedule may have changed - drop cached distances
    
    def _get_station_times(self, train_id: str) -> Optional[list]:
        """
        Parse scheduled times for a train's type-1 (stoppage) stations.
        Returns a list of (type1_index, minutes_since_midnight) tuples,
        or None if the train or its schedule data is unavailable.
        """
        if self.train_data is None:
            return None

        tid_to_stations = self.train_data.get("tid_to_stations", {})
        stations = tid_to_stations.get(str(train_id))

        if not stations:
            return None

        station_times = []  # List of (type1_index, raw_minutes)

        type1_idx = 0
        for station in stations:
            is_type1 = (len(station) > 1 and station[1] == 1)

            if is_type1 and len(station) >= 3 and station[2] is not None:
                time_str = station[2]
                if isinstance(time_str, str) and time_str != "--:--":
                    try:
                        parts = time_str.split(':')
                        station_minutes = int(parts[0]) * 60 + int(parts[1])
                        station_times.append((type1_idx, station_minutes))
                    except (ValueError, IndexError, AttributeError):
                        pass

            if is_type1:
                type1_idx += 1

        return station_times if station_times else None

    def _current_bd_minutes(self, timestamp: int = None) -> int:
        """Minute-of-day in Bangladesh timezone for the given (or current) time."""
        if timestamp:
            now = datetime.fromtimestamp(timestamp, tz=BD_TZ)
        else:
            now = datetime.now(tz=BD_TZ)
        return now.hour * 60 + now.minute

    def _is_journey_active(self, train_id: str, timestamp: int = None) -> Tuple[bool, str]:
        """
        Check whether the train's scheduled journey window contains the given time.
        The window opens `pre_departure_window` minutes before the first scheduled
        departure and stays open for the scheduled journey duration plus
        `max_delay_allowance` (so heavily delayed trains remain trackable).

        Uses minute-of-day arithmetic on a rolling 24h basis, which also handles
        midnight-crossing schedules. Trains without schedule data are never gated.

        Returns: (active: bool, reason: str)
        """
        station_times = self._get_station_times(train_id)
        if not station_times:
            return True, ""  # No schedule data - cannot gate, accept the update

        first_departure = station_times[0][1]
        last_arrival = station_times[-1][1]
        journey_duration = (last_arrival - first_departure) % 1440

        current_minutes = self._current_bd_minutes(timestamp)
        since_departure = (current_minutes - first_departure) % 1440

        if since_departure <= journey_duration + self.max_delay_allowance:
            return True, ""

        minutes_until_departure = (first_departure - current_minutes) % 1440
        if minutes_until_departure <= self.pre_departure_window:
            return True, ""

        return False, (
            f"Journey not active for train {train_id}: scheduled departure is in "
            f"{minutes_until_departure} minutes (updates accepted from "
            f"{self.pre_departure_window} minutes before departure)"
        )

    def _calculate_scheduled_position(self, train_id: str, timestamp: int = None) -> Optional[float]:
        """
        Calculate scheduled position based on current time and train schedule.
        Uses interpolation between stations based on scheduled times.

        Works entirely in "minutes since the most recent scheduled departure"
        (via `% 1440`, exactly like _is_journey_active's own since_departure),
        instead of trying to detect whether *this schedule's clock times*
        cross midnight and special-casing that. That per-schedule heuristic
        breaks for a same-day schedule (e.g. 12:25 -> 20:30) once `timestamp`
        itself has rolled into the next calendar day while still inside the
        delay-allowance grace period _is_journey_active correctly keeps open -
        it looks like "before today's first station" and returns 0.0, when
        the train is actually deep into (or already past) that run, just very
        delayed. Only called once _is_journey_active has confirmed we're in
        an active window, so `since_departure` below is guaranteed to fall
        within journey_duration + max_delay_allowance (well under 1440 for
        every real schedule here).

        Returns position as float (0 = first station, N-1 = last station)
        Returns None if train not found or no schedule data.
        """
        station_times = self._get_station_times(train_id)
        if not station_times:
            return None

        first_departure = station_times[0][1]

        # Minutes-since-departure for each stop, built by walking the
        # schedule forward and always taking the smallest non-negative step
        # to the next stop's clock time (wrapping past midnight as needed).
        # This handles any schedule - same-day, overnight, or (in principle)
        # multiple midnight crossings - uniformly.
        indices = [idx for idx, _ in station_times]
        elapsed_at = [0]
        prev_clock = first_departure
        for _, minute in station_times[1:]:
            delta = (minute - prev_clock) % 1440
            elapsed_at.append(elapsed_at[-1] + delta)
            prev_clock = minute

        current_minutes = self._current_bd_minutes(timestamp)
        since_departure = (current_minutes - first_departure) % 1440

        # since_departure is a minute-of-day-only quantity (`% 1440`), so it
        # can't itself distinguish "hasn't departed yet today" from "long
        # past a previous day's already-finished run" - both wrap to the
        # same value. Bound it the same way _is_journey_active does: beyond
        # a plausible in-progress-or-just-finished window, treat it as "not
        # departed yet" rather than wrapping into "past the last station".
        # In the normal call path (after _is_journey_active already passed)
        # this never trips - it only matters for callers that invoke this
        # standalone, outside an active window.
        if since_departure > elapsed_at[-1] + self.max_delay_allowance:
            return 0.0

        # since_departure >= 0 == elapsed_at[0], so we always have a
        # bracketing "previous" stop; only "next" can be missing (past the
        # last scheduled stop, within the delay-allowance grace period).
        previous_i = 0
        next_i = None
        for i, elapsed in enumerate(elapsed_at):
            if elapsed <= since_departure:
                previous_i = i
            else:
                next_i = i
                break

        if next_i is not None:
            total = elapsed_at[next_i] - elapsed_at[previous_i]
            progress = (since_departure - elapsed_at[previous_i]) / total if total > 0 else 0.0
            return indices[previous_i] + progress * (indices[next_i] - indices[previous_i])

        return float(indices[previous_i])  # past the last scheduled stop

    _EMPTY_STATS = {
        "speed_n": 0, "speed_mean": 0.0, "speed_m2": 0.0,
        "delay_n": 0, "delay_mean": 0.0, "delay_m2": 0.0,
    }

    def _get_train_stats(self, train_id: str) -> Dict:
        """
        This train's running speed/delay baseline (Welford accumulators),
        or a fresh/empty one if it hasn't got any history yet. No expiry -
        unlike live position data, a historical baseline is meant to persist
        and keep accumulating indefinitely (small enough - 255 trains, one
        tiny JSON blob each - that this isn't a real storage concern).
        """
        raw = self.redis.get(f"train:{train_id}:stats")
        return json.loads(raw) if raw else dict(self._EMPTY_STATS)

    def _update_train_stats(self, train_id: str, speed_value: Optional[float],
                             delay_value: Optional[float]):
        """
        Feed one more accepted, genuinely plausible data point (see push() -
        only pings whose teleport check was actually satisfied, not ones let
        through via the confirmation-grace relaxation, contribute here) into
        this train's running baseline. Uses WATCH/MULTI so concurrent
        updates for the same train can't lose an update to each other.
        """
        if speed_value is None and delay_value is None:
            return
        stats_key = f"train:{train_id}:stats"

        def _apply(pipe):
            existing = pipe.get(stats_key)
            stats = json.loads(existing) if existing else dict(self._EMPTY_STATS)
            if speed_value is not None:
                stats["speed_n"], stats["speed_mean"], stats["speed_m2"] = welford_update(
                    stats["speed_n"], stats["speed_mean"], stats["speed_m2"], speed_value
                )
            if delay_value is not None:
                stats["delay_n"], stats["delay_mean"], stats["delay_m2"] = welford_update(
                    stats["delay_n"], stats["delay_mean"], stats["delay_m2"], delay_value
                )
            pipe.multi()
            pipe.set(stats_key, json.dumps(stats))

        self.redis.transaction(_apply, stats_key)

    def _get_cumulative_km(self, train_id: str) -> Optional[List[float]]:
        """
        Cumulative real-world distance (km) at each of this train's type-1
        stops, used to convert the abstract "position" unit into a physically
        meaningful scale for the teleport/speed check. Cached per train_id
        until the next set_train_data() call (route/schedule reload).
        """
        train_id = str(train_id)
        if train_id in self._cum_km_cache:
            return self._cum_km_cache[train_id]

        cum_km = None
        if self.train_data:
            stations = self.train_data.get("tid_to_stations", {}).get(train_id)
            sid_to_sloc = dict(self.train_data.get("xsid_to_sloc", {}) or {})
            sid_to_sloc.update(self.train_data.get("sid_to_sloc", {}) or {})
            if stations:
                # Real along-track distance where track geometry knows the
                # leg (track/track.db), straight line elsewhere.
                try:
                    cum_km = build_cumulative_km_from_legs(
                        stations, route_leg_lengths_km(stations, sid_to_sloc)
                    )
                except Exception as e:
                    print(f"Warning: track-based km failed for train {train_id}: {e}")
                    cum_km = None
                if cum_km is None:
                    cum_km = build_cumulative_km(stations, sid_to_sloc)

        self._cum_km_cache[train_id] = cum_km
        return cum_km

    def _get_reference_position(self, train_id: str, now: int) -> Optional[Dict]:
        """
        The train's best currently-known reference, used as the anchor for
        the teleport/speed check on the *next* incoming ping. Tries the
        pre-calculated live cache first, then the last-known fallback.
        Returns None if there's nothing usable (or it's aged past validity) -
        i.e. this is the first ping of a fresh journey, so there's nothing to
        compare against yet.

        Returns a dict: {"position", "timestamp", "established_at",
        "last_user_id"}. `established_at` is when the *current* value was
        first reached (carried forward across rounds that just continue the
        same trend, reset when the aggregate jumps to a different candidate)
        - see _update_cached_position - and is what push() uses to decide
        whether this reference is "confirmed" enough to gatekeep a
        *different* reporter, or still too fresh to enforce against anyone
        but the same source repeating itself. Missing on data written before
        this field existed - defaults keep old cached data working, just
        with confidence starting from scratch.
        """
        live_data = self.redis.get(f"train:{train_id}:cached_live")
        if live_data:
            cached = json.loads(live_data)
            if now - cached["timestamp"] <= self.max_valid_age:
                return {
                    "position": cached["position"],
                    "timestamp": cached["timestamp"],
                    "established_at": cached.get("established_at", cached["timestamp"]),
                    "last_user_id": cached.get("last_user_id"),
                }

        last_known = self._get_last_known_position(train_id)
        if last_known and now - last_known["timestamp"] <= self.max_valid_age:
            return {
                "position": last_known["position"],
                "timestamp": last_known["timestamp"],
                "established_at": last_known.get("established_at", last_known["timestamp"]),
                "last_user_id": last_known.get("last_user_id"),
            }

        return None

    def _check_shadow_consensus(self, train_id: str, position_km: float, timestamp: int,
                                 user_id: str, effective_max_speed_kmh: float) -> bool:
        """
        Called only when a ping just failed the *enforced* teleport check
        (see push()). Records this attempt in a small rolling buffer of
        recent rejections for this train, then checks whether the most
        recent run of them - walking backward from this one - tells one
        mutually-consistent story on its own (each consecutive step
        physically plausible relative to the one before it), regardless of
        how far that story has drifted from the stuck official reference.

        Requires shadow_min_consistent_size or more such entries, spanning
        either 2+ distinct reporters or shadow_min_diversity_seconds on
        their own - a single identity can't force a re-anchor by quickly
        submitting a handful of fake-but-mutually-agreeing values.

        Returns True if this ping should be accepted and the reference
        cleared so it restarts fresh (confidence resets to zero - the new
        anchor has to earn its own confirmation before it can lock out the
        next challenger too, same as any other reference).
        """
        key = f"train:{train_id}:shadow_rejects"
        self.redis.lpush(key, json.dumps({"km": position_km, "ts": timestamp, "uid": user_id}))
        self.redis.ltrim(key, 0, self.shadow_buffer_size - 1)
        self.redis.expire(key, self.max_valid_age)

        buffer = sorted(
            (json.loads(raw) for raw in self.redis.lrange(key, 0, -1)),
            key=lambda e: e["ts"],
        )

        # Walk backward from the newest entry, extending the tail only
        # while each next-older entry is plausible relative to the current
        # start of the tail - an old, unrelated bout of noise naturally
        # breaks the chain instead of being folded in.
        tail = [buffer[-1]]
        for entry in reversed(buffer[:-1]):
            ok, _, _ = check_teleport(
                tail[0]["km"], entry["km"], abs(tail[0]["ts"] - entry["ts"]),
                effective_max_speed_kmh, self.position_slack_km,
            )
            if not ok:
                break
            tail.insert(0, entry)

        if len(tail) < self.shadow_min_consistent_size:
            return False

        distinct_reporters = len({e["uid"] for e in tail})
        time_span_seconds = tail[-1]["ts"] - tail[0]["ts"]
        if distinct_reporters < 2 and time_span_seconds < self.shadow_min_diversity_seconds:
            return False

        # Promoted: clear the stuck reference (this ping, once stored
        # normally below, becomes a fresh - not yet confirmed - one) and
        # this buffer, so a repeat challenge has to earn confirmation again.
        self.redis.delete(f"train:{train_id}:cached_live", f"train:{train_id}:last_known", key)
        return True

    def push(self, train_id: str, user_id: str, position: float, timestamp: int,
             is_bot: bool = False, speed_kmh: Optional[float] = None) -> Tuple[bool, str]:
        """
        Store a user's latest position update for a train.
        Only keeps the last update per user (overwrites previous).
        Auto-expires after TTL.

        `is_bot` must be determined by the caller from an authenticated source
        (a shared bot secret), never from the client-supplied user_id - that
        string is fully attacker-controlled and must not grant trust on its
        own. It only affects storage TTL and the SQLite log tag below - it is
        NOT a validation bypass. Bot pings run through exactly the same
        journey-active window and teleport/speed check as user pings (see
        functions/position_filter.py). Scheduled position is calculated
        automatically from train data.

        Every ping - accepted or rejected - is logged to a local SQLite DB
        (functions/gps_log.py) as training data for a future ML classifier.

        Returns: (success: bool, message: str)
        """
        received_at = int(time.time())

        # Convert timestamp from milliseconds to seconds if needed
        if timestamp > 2500000000:
            timestamp = int(timestamp / 1000)

        def _log(accepted: bool, reason: str = "", **extra):
            try:
                gps_log.log_update(
                    received_at=received_at, train_id=str(train_id), user_id=user_id,
                    is_bot=is_bot, position=position, timestamp=timestamp,
                    accepted=accepted, reject_reason=reason, **extra,
                )
            except Exception as e:
                print(f"Warning: GPS log call failed: {e}")

        # Reject future timestamps (allow up to 60 seconds tolerance for clock drift)
        current_time = received_at
        max_future_tolerance = 60  # seconds
        if timestamp > current_time + max_future_tolerance:
            future_diff = timestamp - current_time
            reason = f"Timestamp rejected: {future_diff}s in the future (clock drift?)"
            _log(False, reason)
            return False, reason

        # Reject illegal position values exceeding maximum possible type-1 index
        if self.train_data:
            tid_to_stations = self.train_data.get("tid_to_stations", {})
            stations = tid_to_stations.get(str(train_id))
            if stations:
                type1_count = sum(1 for s in stations if len(s) > 1 and s[1] == 1)
                max_pos = max(0, type1_count - 1)
                if position > max_pos + 0.99:
                    reason = f"Illegal position {position} exceeds maximum valid position {max_pos} for train {train_id}"
                    _log(False, reason)
                    return False, reason

        # Reject pings outside the scheduled journey window. Without this,
        # passengers waiting at the origin station publish "live at position
        # 0" hours before the train actually departs. Applies to bots too.
        active, reason = self._is_journey_active(train_id, timestamp)
        if not active:
            _log(False, reason)
            return False, reason

        scheduled_position = self._calculate_scheduled_position(train_id, timestamp)
        cum_km = self._get_cumulative_km(train_id)
        position_km = position_to_km(cum_km, position) if cum_km else None
        scheduled_km = (
            position_to_km(cum_km, scheduled_position)
            if (cum_km and scheduled_position is not None) else None
        )

        reference = self._get_reference_position(train_id, current_time)
        reference_km = None
        reference_age_seconds = None
        enforce_teleport = True
        if reference and cum_km:
            reference_km = position_to_km(cum_km, reference["position"])
            # Absolute difference, not a directional one: a late-arriving ping
            # can legitimately describe a moment *before* the current
            # reference (out-of-order delivery), in which case what matters
            # is still "how much real time separates these two observations".
            reference_age_seconds = abs(timestamp - reference["timestamp"])

            # Don't let a lone, freshly-established reference silently lock
            # out a *different* reporter - e.g. someone selects the wrong
            # train by mistake, and a real passenger on the actual train
            # reports moments later; without this, the mistake becomes the
            # reference and the real report looks like the implausible one.
            # A reference is only enforced against a different reporter once
            # it's held for confirmation_grace_seconds without being
            # superseded (see _update_cached_position) - carried forward
            # across rounds that just continue the same trend, so a
            # long-running, self-consistent source (even a lone one) is
            # still fully protected once it's had time to prove itself.
            # Against the *same* reporter, always enforced regardless of
            # confidence - that's still a single source contradicting only
            # itself, which is exactly what this check exists to catch.
            is_same_source = reference.get("last_user_id") == user_id
            confidence_age = abs(timestamp - reference["established_at"])
            enforce_teleport = is_same_source or confidence_age >= self.confirmation_grace_seconds

        delay_minutes = None
        station_times = self._get_station_times(train_id)
        if station_times:
            delay_minutes = position_to_delay_minutes(
                station_times, position, self._current_bd_minutes(timestamp)
            )

        # This train's own historical baseline (Welford running mean/stddev -
        # see _update_train_stats). Only ever widens the speed ceiling above
        # the flat default, never narrows it, and only once there's enough
        # history to trust it - a train we haven't seen much of yet gets
        # exactly the same protection as before. delay_zscore is logged only
        # for now, not enforced (see the schedule-ceiling note below).
        stats = self._get_train_stats(train_id)
        effective_max_speed_kmh = self.max_train_speed_kmh
        if stats["speed_n"] >= self.min_speed_baseline_samples:
            speed_stddev = welford_stddev(stats["speed_n"], stats["speed_m2"])
            if speed_stddev is not None:
                baseline_ceiling = stats["speed_mean"] + self.speed_baseline_stddev_multiplier * speed_stddev
                effective_max_speed_kmh = max(effective_max_speed_kmh, baseline_ceiling)

        delay_zscore = None
        if delay_minutes is not None and stats["delay_n"] >= self.min_delay_baseline_samples:
            delay_stddev = welford_stddev(stats["delay_n"], stats["delay_m2"])
            if delay_stddev:  # exclude 0 too - a degenerate/all-identical baseline isn't a useful scale
                delay_zscore = (delay_minutes - stats["delay_mean"]) / delay_stddev

        log_extra = dict(
            scheduled_position=scheduled_position, position_km=position_km,
            scheduled_km=scheduled_km, reference_km=reference_km,
            reference_age_seconds=reference_age_seconds, delay_minutes=delay_minutes,
            delay_zscore=delay_zscore,
        )

        # Raw plausibility verdict, independent of whether it's actually
        # enforced this round - used below to decide whether this ping is
        # "clean" enough to feed the baseline (see _update_train_stats call).
        teleport_ok = True
        implied_speed = None

        if position_km is not None:
            # Teleport/speed check: is this plausible given how far the train
            # could physically have travelled since it was last seen? No-op
            # (always passes) if there's no usable reference yet, or if the
            # reference isn't confirmed enough yet to enforce against a
            # different reporter (see enforce_teleport above).
            teleport_ok, implied_speed, teleport_reason = check_teleport(
                position_km, reference_km, reference_age_seconds,
                effective_max_speed_kmh, self.position_slack_km,
            )
            log_extra["implied_speed_kmh"] = implied_speed
            log_extra["teleport_enforced"] = enforce_teleport
            log_extra["shadow_promoted"] = False
            if not teleport_ok and enforce_teleport:
                if self._check_shadow_consensus(
                    train_id, position_km, timestamp, user_id, effective_max_speed_kmh
                ):
                    # A run of mutually-consistent, sufficiently independent
                    # reports has out-voted the stuck reference - accept
                    # this one instead of rejecting it (see
                    # _check_shadow_consensus for why elapsed time alone
                    # can never recover from a reference that was simply
                    # wrong to begin with).
                    log_extra["teleport_enforced"] = False
                    log_extra["shadow_promoted"] = True
                else:
                    _log(False, teleport_reason, **log_extra)
                    return False, teleport_reason

            # NOTE: an earlier version also hard-rejected here if a ping was
            # further along than "scheduled_position + a few km" (can't be
            # ahead of the timetable). Real schedules turned out to be paced
            # far too unevenly per leg for that to hold as a hard gate - a
            # long fast non-stop night segment can legitimately run tens of
            # percent of the whole route "ahead" of a timetable that pads
            # other legs with dwell/recovery time, with zero funny business
            # involved. position_km/scheduled_km (logged below) still carry
            # the same "how far ahead of schedule" signal for a future ML
            # model - it's just not enforced as a reject condition here.
        else:
            # No route/coordinate data for this train - can't convert to km,
            # so the physical checks above can't run. Fall through and accept
            # (matches the existing "no schedule data - can't gate" stance
            # taken elsewhere for trains we don't have full data for).
            log_extra["implied_speed_kmh"] = None
            log_extra["teleport_enforced"] = None
            log_extra["shadow_promoted"] = None

        user_key = f"train:{train_id}:user:{user_id}:last"
        # Use longer TTL for bot users
        user_ttl = self.last_known_ttl if is_bot else self.ttl

        # Sustained-backward-drift check: "position" is progress along THIS
        # train_id's own scheduled stop order, which a real train doesn't
        # sustain moving backward through. Genuine GPS jitter is roughly
        # zero-mean (a touch forward one reading, a touch back the next), so
        # several CONSECUTIVE backward-trending reports in a row from the
        # SAME source is a much stronger signal than any single step - this
        # is what catches someone genuinely riding the opposite-direction
        # train on a shared track (e.g. an up/down pair sharing rails), whose
        # own reports smoothly and repeatedly go backward relative to the
        # train they selected, each step individually far too small/slow to
        # trip the teleport check above. Bounded by the same per-user ping
        # TTL as everything else here, so a long gap before this source's
        # next report naturally starts the streak fresh rather than
        # comparing against a stale prior position.
        backward_streak_key = f"train:{train_id}:user:{user_id}:backward_streak"
        log_extra["backward_streak"] = None
        if position_km is not None:
            own_prev_raw = self.redis.get(user_key)
            if own_prev_raw:
                own_prev_km = position_to_km(cum_km, json.loads(own_prev_raw)["pos"])
                if position_km < own_prev_km - self.backward_drift_epsilon_km:
                    streak = int(self.redis.get(backward_streak_key) or 0) + 1
                    self.redis.set(backward_streak_key, streak, ex=user_ttl)
                    log_extra["backward_streak"] = streak
                    if streak >= self.backward_drift_streak_limit:
                        reason = (
                            f"Sustained backward movement: {streak} consecutive reports "
                            f"trending backward ({own_prev_km - position_km:.2f}km behind the "
                            f"prior one) - possibly reporting for the wrong-direction train"
                        )
                        _log(False, reason, **log_extra)
                        return False, reason
                else:
                    self.redis.delete(backward_streak_key)
                    log_extra["backward_streak"] = 0

        ping = {
            "pos": position,
            "ts": timestamp
        }
        # The phone's own GPS speed (Doppler), if it sent one - blended into
        # the train's speed estimate in _update_cached_position.
        if speed_kmh is not None and 0 <= speed_kmh <= 200:
            ping["spd"] = round(float(speed_kmh), 1)

        active_users_key = f"train:{train_id}:active_users"
        active_trains_key = "active_trains"
        all_trains_key = "all_trains_with_history"

        # Pipeline for atomic operations (single network roundtrip)
        pipe = self.redis.pipeline()
        pipe.set(user_key, json.dumps(ping), ex=user_ttl)  # Store ping with appropriate TTL
        pipe.sadd(active_users_key, user_id)               # Mark user active
        pipe.expire(active_users_key, self.ttl)            # Auto-clean user list (10 min)
        pipe.sadd(active_trains_key, train_id)             # Track active trains
        pipe.expire(active_trains_key, self.ttl)           # Auto-clean train list
        pipe.sadd(all_trains_key, train_id)                # Track all trains with history
        pipe.expire(all_trains_key, self.last_known_ttl)   # 10 hour expiry
        pipe.execute()

        # Pre-calculate and cache the current position for this train
        self._update_cached_position(train_id)

        # Feed this train's historical baseline - but only from a ping whose
        # teleport check was genuinely satisfied on its own merits, not one
        # that only got through via the confirmation-grace relaxation (that
        # data is exactly the kind we don't want normalizing the baseline).
        if teleport_ok:
            self._update_train_stats(
                train_id,
                speed_value=implied_speed,
                delay_value=delay_minutes,
            )

        _log(True, **log_extra)
        return True, "Position updated"


    def _update_cached_position(self, train_id: str):
        """
        Pre-calculate and cache the current position for a train.
        Called after each push to keep it fresh - position is served directly from cache.
        Stores both live position (from active users) and last known position.
        """
        active_users_key = f"train:{train_id}:active_users"
        user_ids = list(self.redis.smembers(active_users_key))
        
        if not user_ids:
            return
        
        # Batch fetch all user pings
        keys = [f"train:{train_id}:user:{uid}:last" for uid in user_ids]
        raw = self.redis.mget(keys)
        
        pings = []
        expired_users = []
        current_time = int(time.time())
        
        for uid, item in zip(user_ids, raw):
            if item is None:
                expired_users.append(uid)
                continue
            ping = json.loads(item)
            # Validate data age - skip if older than max_valid_age (10 hours)
            if current_time - ping["ts"] > self.max_valid_age:
                expired_users.append(uid)
                continue
            ping["uid"] = uid  # in-memory only, for last_user_id below
            pings.append(ping)
        
        # Cleanup expired/invalid users from set
        if expired_users:
            self.redis.srem(active_users_key, *expired_users)
        
        if not pings:
            return

        # How many people are actually reporting right now - reported to the
        # app as-is, regardless of whether some of them get excluded as
        # outliers below (that's about which position to trust, not about
        # under-reporting how many people are engaged).
        total_active_users = len(pings)

        # When several reports disagree, trust the largest mutually-agreeing
        # cluster (by real-world km) and discard the rest as outliers, rather
        # than averaging a lone fake/mistaken report in with genuine ones.
        # With 0-1 pings (the common case for this app) there's nothing to
        # cluster and this is a no-op.
        cum_km = self._get_cumulative_km(train_id)
        if cum_km and len(pings) > 1:
            kms = [position_to_km(cum_km, p["pos"]) for p in pings]
            clusters = cluster_pings_by_km(kms, self.cluster_tolerance_km)
            if len(clusters) > 1:
                best = max(clusters, key=lambda c: (len(c), max(pings[i]["ts"] for i in c)))
                pings = [pings[i] for i in best]

        # Calculate final position using weighted priority based on update age
        sum_pos_weight = 0.0
        sum_weights = 0.0

        for p in pings:
            time_diff_min = (current_time - p["ts"]) / 60.0
            
            # Check proximity to a type 1 station (integer index)
            dist_to_station = abs(p["pos"] - round(p["pos"]))
            if dist_to_station <= 0.01:
                weight = 0.05
            elif dist_to_station <= 0.02:
                weight = 0.10
            else:
                # Determine priority weight based on age of the update
                if time_diff_min < 1.0:
                    weight = 0.75
                elif time_diff_min < 2.0:
                    weight = 0.40
                elif time_diff_min < 3.0:
                    weight = 0.25
                elif time_diff_min < 5.0:
                    weight = 0.15
                elif time_diff_min < 7.0:
                    weight = 0.10
                elif time_diff_min <= 10.0:
                    weight = 0.05  # Decaying priority
                else:
                    weight = 0.0
                
            sum_pos_weight += p["pos"] * weight
            sum_weights += weight
            
        if sum_weights > 0.0:
            final_position = sum_pos_weight / sum_weights
        else:
            final_position = statistics.median([p["pos"] for p in pings])

        max_timestamp = max(p["ts"] for p in pings)
        # Whoever contributed the most-recent ping in the winning set - used
        # by push() to tell "the same source repeating itself" (always
        # checked for self-consistency) from "a different reporter" (only
        # gatekept once the reference has held for confirmation_grace_seconds).
        last_user_id = next(p["uid"] for p in pings if p["ts"] == max_timestamp)

        # established_at tracks how long the *current* value has held,
        # carried forward across rounds that just continue the same trend
        # and reset when the aggregate jumps to a genuinely different
        # candidate - see push()'s confidence_age/enforce_teleport. Compare
        # against the OLD cached value (before this round overwrites it).
        established_at = max_timestamp  # default: nothing to compare against yet
        old_live = self.redis.get(f"train:{train_id}:cached_live")
        old_cached = json.loads(old_live) if old_live else None
        if old_cached:
            old_established_at = old_cached.get("established_at", old_cached["timestamp"])
            if cum_km:
                old_km = position_to_km(cum_km, old_cached["position"])
                new_km = position_to_km(cum_km, final_position)
                elapsed = abs(max_timestamp - old_cached["timestamp"])
                continuing, _, _ = check_teleport(
                    new_km, old_km, elapsed, self.max_train_speed_kmh, self.position_slack_km
                )
            else:
                # No route data to judge continuity in km - don't penalize
                # confidence for a train we can't reason about physically.
                continuing = True

            # A position sitting at/near an exact station index never gets
            # to "carry forward" confidence, even if it trivially continues
            # itself (delta ~0) - see near_station_confidence_epsilon above.
            # It still gets displayed and stored normally; it just can't age
            # into a protected reference the way genuine, progressing
            # movement does, so a later real rider isn't locked out by a
            # bystander who never actually moved.
            at_station = abs(final_position - round(final_position)) <= self.near_station_confidence_epsilon
            if continuing and not at_station:
                established_at = old_established_at

        # Speed: how far the aggregate moved along the real track since the
        # previous aggregate, blended with the reporters' own GPS speed.
        device_speeds = [
            p["spd"] for p in pings
            if p.get("spd") is not None and max_timestamp - p["ts"] <= self.device_speed_max_age
        ]
        motion_key = f"train:{train_id}:motion"
        prev_motion_raw = self.redis.get(motion_key)
        prev_motion = json.loads(prev_motion_raw) if prev_motion_raw else None
        final_km = position_to_km(cum_km, final_position) if cum_km else None
        if prev_motion and max_timestamp < prev_motion.get("ts", 0):
            motion = prev_motion  # an out-of-order (older) aggregate - don't rewind
        else:
            motion = update_motion(
                prev_motion, final_position, final_km, max_timestamp,
                combine_device_speeds(device_speeds),
            )

        # Store as cached live position (short TTL, refreshed on each update)
        live_cache_key = f"train:{train_id}:cached_live"
        live_data = {
            "position": final_position,
            "timestamp": max_timestamp,
            "active_user": total_active_users,
            "cached_at": current_time,
            "established_at": established_at,
            "last_user_id": last_user_id,
            "speed_kmh": motion.get("speed"),
            "moving": motion.get("moving", False),
        }

        # Store as last known position with 10-hour TTL (fallback)
        last_known_key = f"train:{train_id}:last_known"
        last_known_data = {
            "position": final_position,
            "timestamp": max_timestamp,
            "established_at": established_at,
            "last_user_id": last_user_id,
        }

        # Pipeline for atomic operations
        pipe = self.redis.pipeline()
        pipe.set(live_cache_key, json.dumps(live_data), ex=self.ttl)
        pipe.set(last_known_key, json.dumps(last_known_data), ex=self.last_known_ttl)
        pipe.set(motion_key, json.dumps(motion), ex=self.last_known_ttl)
        pipe.execute()

        if self.history is not None and motion is not prev_motion:
            station_times = self._get_station_times(train_id)
            stop_count = self._stop_count(train_id)
            self.history.record(
                train_id, station_times, stop_count, prev_motion, final_position,
                max_timestamp, final_km, motion.get("speed"), total_active_users,
            )

        # Tell every worker's LiveHub this train has news for its subscribers.
        try:
            self.redis.publish(self.updates_channel, train_id)
        except Exception as e:
            print(f"Warning: live update publish failed for train {train_id}: {e}")

    def _stop_count(self, train_id: str) -> int:
        """Number of scheduled (type-1) stops - the position scale's range."""
        if not self.train_data:
            return 0
        stations = self.train_data.get("tid_to_stations", {}).get(str(train_id)) or []
        return sum(1 for s in stations if len(s) > 1 and s[1] == 1)

    def get_motion(self, train_id: str) -> Optional[Dict]:
        raw = self.redis.get(f"train:{train_id}:motion")
        return json.loads(raw) if raw else None
    
    def get_train_bounds(self, train_id: str) -> Optional[Dict]:
        """
        Debug/diagnostic endpoint data: the reference position and age that
        the next incoming ping for this train would be teleport-checked
        against (see _get_reference_position / push). Not used by the app;
        kept for the same `/bounds/{train_id}` URL for compatibility with any
        external tooling.

        This replaces the old per-bot "bounds" mechanism (a lower bound only
        a bot could raise, an upper bound pinned to the static schedule) with
        the same reference the real filter now uses, which - unlike the old
        one - is grounded in actual recent reports rather than the timetable,
        and isn't bot-specific.
        """
        reference = self._get_reference_position(train_id, int(time.time()))
        if reference is None:
            return None
        return {
            "reference_position": reference["position"],
            "reference_timestamp": reference["timestamp"],
            "established_at": reference["established_at"],
        }

    def get_train_position(self, train_id: str) -> Optional[Dict]:
        """
        Get position for a train - serves directly from pre-calculated cache.
        1. First tries cached live position (calculated when data arrives)
        2. Falls back to last known position (up to 10 hours old)
        3. Validates data age - returns None if data is too old (>10 hours)
        
        Returns format compatible with both old and new app:
        {
            "position": ..., "timestamp": ..., "user_count": ..., "is_live": ...,
            "unconfirmed": {"position": ..., "timestamp": ...}  # For old app compatibility
        }
        """
        current_time = int(time.time())
        
        # Try to get cached live position (pre-calculated on push)
        live_cache_key = f"train:{train_id}:cached_live"
        live_data = self.redis.get(live_cache_key)
        
        if live_data:
            cached = json.loads(live_data)
            # Validate data age
            if current_time - cached["timestamp"] <= self.max_valid_age:
                result = {
                    "position": cached["position"],
                    "timestamp": cached["timestamp"],
                    "active_user": cached["active_user"],
                    "is_live": True,
                    "speed_kmh": cached.get("speed_kmh"),
                    "moving": cached.get("moving", False),
                }
                result["unconfirmed"] = {
                    "position": cached["position"],
                    "timestamp": cached["timestamp"]
                }
                return result
        
        # Fall back to last known position
        last_known = self._get_last_known_position(train_id)
        if last_known:
            # Validate data age
            if current_time - last_known["timestamp"] > self.max_valid_age:
                return None  # Data too old, invalid
            result = {**last_known, "is_live": False, "active_user": 0}
            result["unconfirmed"] = {
                "position": last_known["position"],
                "timestamp": last_known["timestamp"]
            }
            return result
        
        return None
    
    def _get_last_known_position(self, train_id: str) -> Optional[Dict]:
        """
        Get last known position (stored for up to 10 hours).
        Used as fallback when no active users.
        """
        last_known_key = f"train:{train_id}:last_known"
        data = self.redis.get(last_known_key)
        
        if data:
            return json.loads(data)
        return None
    
    def get_positions(self, train_ids: List[str]) -> Dict[str, Dict]:
        """
        Get positions for multiple trains efficiently using batch fetch.
        Serves directly from pre-calculated cache - no recalculation.
        Returns dict with train_id as key.
        """
        if not train_ids:
            return {}
        
        current_time = int(time.time())
        positions = {}
        
        # Batch fetch cached live positions (single Redis call)
        live_keys = [f"train:{tid}:cached_live" for tid in train_ids]
        live_data = self.redis.mget(live_keys)
        
        # Track which trains need fallback to last_known
        need_fallback = []
        
        for tid, data in zip(train_ids, live_data):
            if data:
                cached = json.loads(data)
                # Validate data age
                if current_time - cached["timestamp"] <= self.max_valid_age:
                    positions[tid] = {
                        "position": cached["position"],
                        "timestamp": cached["timestamp"],
                        "active_user": cached["active_user"],
                        "is_live": True,
                        "speed_kmh": cached.get("speed_kmh"),
                        "moving": cached.get("moving", False),
                        "unconfirmed": {
                            "position": cached["position"],
                            "timestamp": cached["timestamp"]
                        }
                    }
                    continue
            need_fallback.append(tid)
        
        # Batch fetch last_known for trains without live data
        if need_fallback:
            last_known_keys = [f"train:{tid}:last_known" for tid in need_fallback]
            last_known_data = self.redis.mget(last_known_keys)
            
            for tid, data in zip(need_fallback, last_known_data):
                if data:
                    last_known = json.loads(data)
                    # Validate data age
                    if current_time - last_known["timestamp"] <= self.max_valid_age:
                        positions[tid] = {
                            "position": last_known["position"],
                            "timestamp": last_known["timestamp"],
                            "is_live": False,
                            "active_user": 0,
                            "unconfirmed": {
                                "position": last_known["position"],
                                "timestamp": last_known["timestamp"]
                            }
                        }
        
        return positions
    
    def get_all_active_trains(self) -> List[str]:
        """Get list of all trains with active user data (last 10 min)"""
        active_trains_key = "active_trains"
        return list(self.redis.smembers(active_trains_key))
    
    def get_all_trains_with_history(self) -> List[str]:
        """Get list of all trains with any position history (up to 10 hours)"""
        all_trains_key = "all_trains_with_history"
        return list(self.redis.smembers(all_trains_key))
    
    def get_active_train_count(self) -> int:
        """Get count of active trains"""
        return len(self.get_all_active_trains())
    
    def get_user_count_for_train(self, train_id: str) -> int:
        """Get number of active users for a train"""
        active_users_key = f"train:{train_id}:active_users"
        return self.redis.scard(active_users_key)
    
    def health_check(self) -> bool:
        """Check if Redis connection is healthy"""
        try:
            self.redis.ping()
            return True
        except:
            return False
