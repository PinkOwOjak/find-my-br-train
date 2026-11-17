# track_geometry.py - Real rail-track geometry between adjacent stations.
#
# Built offline by tools/build_track.py from an OpenStreetMap export and
# stored in track/track.db (the same file is bundled into the Flutter app as
# assets/track/track.db). Each row is the actual track polyline between two
# adjacent stations, keyed by the station-id pair in canonical (sorted) order.
#
# The server only needs the along-track *length* of each pair, to turn the
# abstract "position" unit (fractional stop index) into real kilometres for
# speed estimation and the teleport check. Rails curve; straight-line
# (haversine) distance between stations under-reads the true distance, so
# speeds derived from it read low and the km scale is uneven between legs.
#
# Track data is optional: any pair that isn't in the file (or a missing file
# altogether) falls back to straight-line distance, which is exactly what
# the code did before this module existed.

import math
import os
import sqlite3
import threading
from typing import Dict, Iterable, List, Optional, Tuple

TRACK_DB_PATH = os.getenv(
    "TRACK_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "track", "track.db"),
)

EARTH_RADIUS_M = 6371000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def pair_key(a: str, b: str) -> Tuple[str, str]:
    """Canonical (sorted) key for an unordered station pair."""
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Google encoded-polyline format (precision 1e5 ~= 1.1 m). Compact, and
# trivially decodable in Dart without any dependency.
# ---------------------------------------------------------------------------

def _encode_value(v: int) -> str:
    v = ~(v << 1) if v < 0 else (v << 1)
    out = []
    while v >= 0x20:
        out.append(chr((0x20 | (v & 0x1F)) + 63))
        v >>= 5
    out.append(chr(v + 63))
    return "".join(out)


def encode_polyline(points: Iterable[Tuple[float, float]]) -> str:
    out = []
    prev_lat = prev_lon = 0
    for lat, lon in points:
        ilat, ilon = round(lat * 1e5), round(lon * 1e5)
        out.append(_encode_value(ilat - prev_lat))
        out.append(_encode_value(ilon - prev_lon))
        prev_lat, prev_lon = ilat, ilon
    return "".join(out)


def decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    points = []
    index = lat = lon = 0
    n = len(encoded)
    while index < n:
        for which in (0, 1):
            shift = result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta = ~(result >> 1) if (result & 1) else (result >> 1)
            if which == 0:
                lat += delta
            else:
                lon += delta
        points.append((lat / 1e5, lon / 1e5))
    return points


class TrackGeometry:
    """
    Read-only, in-memory view of track/track.db: along-track length per
    adjacent-station pair. Loaded once (the whole file is ~100 KB) and
    shared; safe to call from multiple threads.
    """

    def __init__(self, db_path: str = TRACK_DB_PATH):
        self.db_path = db_path
        self._lengths_m: Dict[Tuple[str, str], float] = {}
        self._meta: Dict[str, str] = {}
        self._loaded = False
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            if os.path.exists(self.db_path):
                try:
                    conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
                    try:
                        for a, b, length_m in conn.execute(
                            "SELECT from_sid, to_sid, length_m FROM track_segment"
                        ):
                            self._lengths_m[pair_key(a, b)] = float(length_m)
                        self._meta = dict(conn.execute("SELECT key, value FROM meta"))
                    finally:
                        conn.close()
                    print(f"✓ Track geometry loaded: {len(self._lengths_m)} segments from {self.db_path}")
                except Exception as e:
                    print(f"⚠ Could not load track geometry from {self.db_path}: {e}")
            else:
                print(f"ℹ No track geometry at {self.db_path} - using straight-line distances")
            self._loaded = True

    @property
    def meta(self) -> Dict[str, str]:
        self._ensure_loaded()
        return dict(self._meta)

    @property
    def segment_count(self) -> int:
        self._ensure_loaded()
        return len(self._lengths_m)

    def pair_length_m(self, a: str, b: str) -> Optional[float]:
        """Along-track length between two adjacent stations, or None."""
        self._ensure_loaded()
        return self._lengths_m.get(pair_key(a, b))


_shared: Optional[TrackGeometry] = None


def shared_track() -> TrackGeometry:
    global _shared
    if _shared is None:
        _shared = TrackGeometry()
    return _shared


def route_leg_lengths_km(
    stations: list,
    sid_to_sloc: Dict[str, list],
    track: Optional[TrackGeometry] = None,
) -> List[Optional[float]]:
    """
    Distance (km) of each consecutive leg of a train's *full* station list
    (every stop/pass/via point, in schedule order): the real track length
    where track data has that pair, else straight-line distance, else None
    when either end has no usable coordinates.
    """
    track = track if track is not None else shared_track()

    def coords(sid):
        loc = sid_to_sloc.get(sid)
        if not loc or len(loc) < 2:
            return None
        if loc[0] == 0.0 and loc[1] == 0.0:
            return None
        return float(loc[0]), float(loc[1])

    legs: List[Optional[float]] = []
    for i in range(len(stations) - 1):
        a, b = stations[i][0], stations[i + 1][0]
        length_m = track.pair_length_m(a, b)
        if length_m is not None:
            legs.append(length_m / 1000.0)
            continue
        ca, cb = coords(a), coords(b)
        legs.append(haversine_m(ca[0], ca[1], cb[0], cb[1]) / 1000.0 if ca and cb else None)
    return legs
