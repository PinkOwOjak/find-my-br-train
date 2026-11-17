# motion.py - Per-train speed estimation from the aggregated position stream.
#
# Every accepted ping updates the train's aggregated position (see
# RedisTrainTracker._update_cached_position). Speed is derived from how far
# that aggregate moved along the real track (km, via track geometry) over the
# time between two aggregates, optionally blended with the phones' own GPS
# speed (Doppler-based, usually accurate to ~1 km/h when the fix is good).
#
# The result is smoothed with an exponential moving average whose weight
# depends on the time gap, so a burst of pings a second apart can't make the
# speed jump around, while a fresh reading after a long gap still counts in
# full. Viewers use speed to keep the marker moving smoothly between updates
# (dead reckoning), so a stable value matters more than an instant one.

import math
from typing import Dict, List, Optional

MAX_SPEED_KMH = 140.0          # nothing in Bangladesh runs faster
MIN_INTERVAL_S = 4             # gaps shorter than this are too noisy to derive speed from
RESET_INTERVAL_S = 900         # after 15 min of silence, start fresh
SMOOTHING_TAU_S = 40.0         # EMA time constant
DEVICE_SPEED_WEIGHT = 0.65     # trust phone GPS speed over position deltas when both exist
MOVING_THRESHOLD_KMH = 3.0


def _clamp_speed(v: Optional[float]) -> Optional[float]:
    if v is None or not math.isfinite(v):
        return None
    return max(0.0, min(MAX_SPEED_KMH, v))


def combine_device_speeds(speeds: List[float]) -> Optional[float]:
    """Median of the reporters' own GPS speeds (robust to one bad phone)."""
    clean = sorted(s for s in (_clamp_speed(x) for x in speeds) if s is not None)
    if not clean:
        return None
    mid = len(clean) // 2
    return clean[mid] if len(clean) % 2 else (clean[mid - 1] + clean[mid]) / 2.0


def update_motion(prev: Optional[Dict], position: float, km: Optional[float], ts: int,
                  device_speed_kmh: Optional[float] = None) -> Dict:
    """
    Fold one new aggregated fix into the train's motion state.

    prev/return shape: {"pos": float, "km": float|None, "ts": int,
                        "speed": float|None, "moving": bool}
    """
    device_speed_kmh = _clamp_speed(device_speed_kmh)

    if prev is None or prev.get("ts") is None or ts - prev["ts"] > RESET_INTERVAL_S:
        speed = device_speed_kmh
        return _state(position, km, ts, speed)

    dt = ts - prev["ts"]
    if dt < MIN_INTERVAL_S:
        # Too close to the previous aggregate for a position delta to mean
        # anything. Keep the old anchor (so the next delta spans a usable
        # interval) but let a fresh device reading nudge the speed.
        speed = prev.get("speed")
        if device_speed_kmh is not None and dt >= 0:
            speed = device_speed_kmh if speed is None else speed + 0.2 * (device_speed_kmh - speed)
        return {**prev, "speed": _round(speed), "moving": _is_moving(speed)}

    measured = None
    if km is not None and prev.get("km") is not None:
        # Backward movement along the route is GPS jitter around a stop
        # (a real train doesn't reverse along its own schedule) - read as 0.
        measured = _clamp_speed(max(0.0, km - prev["km"]) / dt * 3600.0)
    if device_speed_kmh is not None:
        measured = device_speed_kmh if measured is None else (
            DEVICE_SPEED_WEIGHT * device_speed_kmh + (1 - DEVICE_SPEED_WEIGHT) * measured
        )

    old = prev.get("speed")
    if measured is None:
        speed = old
    elif old is None:
        speed = measured
    else:
        alpha = 1.0 - math.exp(-dt / SMOOTHING_TAU_S)
        speed = old + alpha * (measured - old)
    return _state(position, km, ts, speed)


def _round(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 1)


def _is_moving(speed: Optional[float]) -> bool:
    return speed is not None and speed >= MOVING_THRESHOLD_KMH


def _state(position, km, ts, speed) -> Dict:
    speed = _clamp_speed(speed)
    return {"pos": position, "km": km, "ts": ts, "speed": _round(speed), "moving": _is_moving(speed)}
