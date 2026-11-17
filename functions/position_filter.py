# position_filter.py - Physics-based plausibility checks for crowdsourced
# "I'm inside this train" GPS updates.
#
# Context: most trains get 0 reporters at any given time, occasionally 1,
# rarely 2, and very rarely 3. That means we usually can't rely on consensus
# between multiple simultaneous users to spot a fake/mistaken report - most
# of the defense has to work for a single, lone ping. The checks here are
# built around that constraint:
#
#   - Convert the abstract "position" unit (fractional type-1 station index,
#     e.g. 3.35 = 35% of the way from station 3 to station 4) to real
#     kilometres, using the same haversine great-circle distance already used
#     elsewhere in this codebase (route_calculator.calculate_distance). This
#     matters because a fixed tolerance expressed in "station index units"
#     (the old bounds system) is not physically meaningful: 0.5 station-index
#     can be 300m on a commuter route with closely-spaced stops, or 40km on an
#     intercity route - so the same constant was simultaneously too loose and
#     too strict depending on the train.
#   - Judge a new ping by comparing it to where the train was last known to
#     be and how much real time has passed, using a generous max train speed
#     plus a flat slack distance. The slack absorbs: GPS jitter (a few
#     metres), and multiple simultaneous valid reports from different
#     bogies of the same train (a rake is at most a couple hundred metres
#     long, but we leave real margin here).
#   - Never assume the train is anywhere near its timetabled position - BR
#     trains are frequently 4-5+ hours late, so "expected position right now"
#     from the static timetable is a weak, generous ceiling at best, not a
#     reliable estimate of where the train actually is. The real anchor is
#     "where a real report last put this train, and how long ago was that".
#
# None of this is bot-specific. The caller must run every one of these checks
# for bot-sourced pings exactly like user-sourced ones - an authenticated bot
# is a trusted *identity*, not a trusted *position*.

import math
from typing import Dict, List, Optional, Tuple

from .route_calculator import calculate_distance


def build_cumulative_km(stations: list, sid_to_sloc: Dict[str, list]) -> Optional[List[float]]:
    """
    Build a monotonically increasing list of cumulative distance (km) at each
    of a train's type-1 (scheduled stop) stations, in schedule order, using
    straight-line (haversine) distance between consecutive stops.

    This under-estimates true track distance (rail lines curve; great-circle
    distance doesn't), so it's a conservative lower bound - if anything it
    makes the teleport check slightly more permissive on curvy segments,
    never less.

    Returns None if there are fewer than 2 type-1 stations (nothing to anchor
    a distance scale to).
    """
    type1_names = [s[0] for s in stations if len(s) > 1 and s[1] == 1]
    if len(type1_names) < 2:
        return None

    def _coords(name) -> Optional[Tuple[float, float]]:
        loc = sid_to_sloc.get(name)
        if not loc or len(loc) < 2:
            return None
        lat, lon = loc[0], loc[1]
        if lat == 0.0 and lon == 0.0:
            return None
        return (lat, lon)

    cum_km = [0.0]
    prev_coords = _coords(type1_names[0])
    for name in type1_names[1:]:
        cur_coords = _coords(name)
        if prev_coords is None or cur_coords is None:
            # Missing/placeholder coordinates for one of the two stations -
            # we can't compute a real distance for this leg. Fall back to a
            # rough typical inter-station spacing rather than collapsing the
            # scale to zero, so the cumulative array stays monotonic and
            # later legs aren't thrown off.
            step = 5.0
        else:
            step = calculate_distance(prev_coords[0], prev_coords[1], cur_coords[0], cur_coords[1])
            if step <= 0:
                step = 0.1  # guard against duplicate/degenerate coordinates
        cum_km.append(cum_km[-1] + step)
        prev_coords = cur_coords if cur_coords is not None else prev_coords

    return cum_km


def build_cumulative_km_from_legs(stations: list, legs_km: List[Optional[float]]) -> Optional[List[float]]:
    """
    Like build_cumulative_km, but from per-leg distances over the train's
    *full* station list (stops, pass-through stations and via points - see
    track_geometry.route_leg_lengths_km), so curves and every intermediate
    point count. Legs with no distance fall back to the same rough 5 km
    spacing build_cumulative_km uses for missing coordinates, spread over
    the stop-to-stop leg they belong to.
    """
    stop_indices = [i for i, s in enumerate(stations) if len(s) > 1 and s[1] == 1]
    if len(stop_indices) < 2 or len(legs_km) != len(stations) - 1:
        return None
    cum_km = [0.0]
    for a, b in zip(stop_indices, stop_indices[1:]):
        legs = legs_km[a:b]
        if any(leg is None for leg in legs):
            step = 5.0
        else:
            step = sum(legs)
        cum_km.append(cum_km[-1] + max(step, 0.1))
    return cum_km


def position_to_km(cum_km: List[float], position: float) -> float:
    """Interpolate a fractional station-index position into cumulative km."""
    n = len(cum_km)
    if n == 0:
        return 0.0
    if n == 1:
        return cum_km[0]
    idx = int(math.floor(position))
    frac = position - idx
    if idx < 0:
        return cum_km[0]
    if idx >= n - 1:
        return cum_km[-1]
    return cum_km[idx] + frac * (cum_km[idx + 1] - cum_km[idx])


def check_teleport(
    position_km: float,
    reference_km: Optional[float],
    reference_age_seconds: Optional[float],
    max_speed_kmh: float,
    slack_km: float,
) -> Tuple[bool, Optional[float], str]:
    """
    Check a new position against the train's last known real-time position.

    If there's no usable reference (first ping of a journey, or the last one
    aged out), there's nothing to compare against - this always passes, and
    other checks (schedule ceiling, journey-active window) carry the load.

    Returns (ok, implied_speed_kmh, reason). implied_speed_kmh is None when
    it couldn't be computed (no reference, or ~zero elapsed time).
    """
    if reference_km is None or reference_age_seconds is None:
        return True, None, ""

    elapsed_hours = max(reference_age_seconds, 0.0) / 3600.0
    max_km = slack_km + max_speed_kmh * elapsed_hours
    actual_km = abs(position_km - reference_km)
    speed = (actual_km / elapsed_hours) if elapsed_hours > 1e-6 else None

    if actual_km <= max_km:
        return True, speed, ""

    return False, speed, (
        f"Implausible jump: {actual_km:.2f}km from last known position in "
        f"{elapsed_hours * 60:.1f} min (max allowed {max_km:.2f}km at "
        f"{max_speed_kmh:.0f}km/h + {slack_km:.2f}km slack)"
    )


def check_schedule_ceiling(
    position_km: float,
    scheduled_km: Optional[float],
    slack_km: float,
) -> Tuple[bool, str]:
    """
    NOT currently wired into RedisTrainTracker.push()'s reject path - kept
    here (tested, still called out in redis_tracker.py's comments) because
    it documents a real design decision. It was tried as a hard "can't be
    ahead of the timetable" gate and dropped: real BR schedules pad
    different legs very unevenly (dwell time, slower urban approach vs. a
    long fast non-stop night stretch), so a genuinely on-time train can
    legitimately look tens of percent of the whole route "ahead" in km with
    nothing suspicious going on - a bug this surfaced against live bot data
    (train 790 was 48% of its route "ahead" while running normally). No
    fixed or percentage slack_km can be both loose enough for that and tight
    enough to mean anything. check_teleport (anchored on real prior reports,
    not the timetable) is the check that actually carries this load.
    """
    if scheduled_km is None:
        return True, ""
    ceiling = scheduled_km + slack_km
    if position_km > ceiling:
        return False, (
            f"Position {position_km:.2f}km is ahead of scheduled position "
            f"{scheduled_km:.2f}km (ceiling {ceiling:.2f}km)"
        )
    return True, ""


def cluster_pings_by_km(values_km: List[float], tolerance_km: float) -> List[List[int]]:
    """
    Group ping indices into clusters by proximity in km, using a chained
    sweep: sort by km, then a point joins the current cluster if it's within
    `tolerance_km` of the *previous* point already in that cluster (not the
    cluster's first point). This lets a spread-out but mutually-consistent
    group (e.g. several bogies of the same long train) stay one cluster,
    while a lone far-away outlier still splits off into its own cluster.

    Returns a list of clusters (each a list of original indices), unsorted
    relative to each other.
    """
    if not values_km:
        return []
    order = sorted(range(len(values_km)), key=lambda i: values_km[i])
    clusters: List[List[int]] = []
    current = [order[0]]
    for idx in order[1:]:
        if values_km[idx] - values_km[current[-1]] <= tolerance_km:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)
    return clusters


def position_to_delay_minutes(
    station_times: List[Tuple[int, int]],
    position: float,
    current_minutes: int,
) -> Optional[float]:
    """
    Estimate how many minutes late (positive) or early (negative) a train is,
    given it's currently reporting `position` at `current_minutes` (minute-of-
    day, Bangladesh time). This is the inverse of "scheduled position at time
    T" - here we ask "what time was this position scheduled for, and how far
    is that from now" - and mirrors the delay calculation already shown to
    users client-side (DataService.calculateTrainDelay in the Flutter app),
    so logged values line up with what people actually see in the app.

    Not used for filtering (any reported position implies *some* delay figure
    by construction, so it can't independently corroborate that position) -
    this is purely a feature for the SQLite log / future ML dataset.

    Returns None if there isn't enough schedule data to interpolate from.
    """
    if not station_times or len(station_times) < 2:
        return None

    station_positions = [idx for idx, _ in station_times]
    schedule_minutes = [m for _, m in station_times]

    shift_minute = schedule_minutes[0]
    shifted_minutes = []
    previous = None
    for m in schedule_minutes:
        shifted = (m - shift_minute + 1440) % 1440
        if previous is not None and shifted < previous:
            shifted += 1440
        shifted_minutes.append(shifted)
        previous = shifted

    expected_shifted_minute = shifted_minutes[-1]
    if position <= station_positions[0]:
        expected_shifted_minute = shifted_minutes[0]
    elif position >= station_positions[-1]:
        expected_shifted_minute = shifted_minutes[-1]
    else:
        for i in range(len(station_positions) - 1):
            idx0, idx1 = station_positions[i], station_positions[i + 1]
            if position <= idx1:
                span = idx1 - idx0
                if span > 0:
                    fraction = (position - idx0) / span
                    expected_shifted_minute = (
                        shifted_minutes[i] + fraction * (shifted_minutes[i + 1] - shifted_minutes[i])
                    )
                else:
                    expected_shifted_minute = shifted_minutes[i]
                break

    shifted_reference_minute = (current_minutes - shift_minute) % 1440
    if shifted_reference_minute < 0:
        shifted_reference_minute += 1440

    raw_delay = shifted_reference_minute - expected_shifted_minute
    normalized = raw_delay % 1440
    if normalized > 720:
        normalized -= 1440
    elif normalized < -720:
        normalized += 1440

    return normalized


def welford_update(n: int, mean: float, m2: float, value: float) -> Tuple[int, float, float]:
    """
    One step of Welford's online algorithm: updates running (count, mean,
    sum-of-squared-deviations) with a new value, without storing history.
    Used to build per-train speed/delay baselines (see redis_tracker.py's
    _update_train_stats) incrementally, one accepted ping at a time, instead
    of re-scanning gps_updates.db on every request.
    """
    n += 1
    delta = value - mean
    mean += delta / n
    delta2 = value - mean
    m2 += delta * delta2
    return n, mean, m2


def welford_stddev(n: int, m2: float) -> Optional[float]:
    """Sample standard deviation from Welford accumulators. None if n < 2."""
    if n < 2:
        return None
    return math.sqrt(m2 / (n - 1))
