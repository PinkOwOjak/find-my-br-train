#!/usr/bin/env python3
"""
build_track.py - Build track/track.db (real rail geometry between adjacent
stations) from an OpenStreetMap export.

Input
-----
1. osm/bd_rail.json - Overpass "raw OSM data" export of Bangladesh rail
   lines. Re-create it at https://overpass-turbo.eu with:

       [out:json][timeout:180];
       area["ISO3166-1"="BD"][admin_level=2]->.bangladesh;
       (
         way["railway"~"^(rail|narrow_gauge)$"](area.bangladesh);
       );
       out geom;

   (`out geom` keeps both node ids - needed for junction connectivity - and
   coordinates.)
2. data.json (station coordinates + every train's station list).
3. Optional osm/stations.json - Overpass export of railway=station|halt
   (`nwr[...] ; out center tags;`), used only to cross-check station
   coordinates in the report.

Output
------
track/track.db (SQLite):
    track_segment(from_sid, to_sid, polyline, length_m)   canonical sorted pair
    station_snap(sid, lat, lng, snap_m)                    station snapped onto track
    meta(key, value)                                       schema/osm date/build id
track/track_report.txt  - stations far from any track, pairs that had to fall
                          back to straight lines, and why.

Copy track/track.db into the Flutter app as assets/track/track.db (pass
--app-asset to do it automatically). Rebuild whenever stations or routes
change; pairs missing from the file just draw as straight lines in the app.

Usage:
    python3 tools/build_track.py [--osm osm/bd_rail.json] [--data data.json]
                                 [--out track/track.db] [--app-asset PATH]
                                 [--preview TRAIN_ID ...]
"""

import argparse
import heapq
import json
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from functions.track_geometry import encode_polyline, haversine_m, pair_key  # noqa: E402

SCHEMA_VERSION = "1"

# Cost multipliers per OSM `service=` value. Yards/sidings are needed for
# connectivity at a few junctions, but a through route should only use them
# when there is genuinely no main-line path.
SERVICE_COST = {None: 1.0, "crossover": 2.0, "link": 1.5, "yard": 8.0, "siding": 8.0, "spur": 8.0}
DEFAULT_SERVICE_COST = 3.0

MAX_SNAP_M = 1500.0         # a station further than this from any track isn't trusted
MAIN_LINE_SNAP_M = 600.0    # prefer a main-line node over a closer yard node within this
MIN_COMPONENT_NODES = 150   # ignore tiny isolated yard fragments when snapping
MAX_DETOUR_RATIO = 1.8      # path/straight above this AND ...
MAX_DETOUR_EXTRA_M = 3000.0 # ... more than this many extra metres -> reject pair
SIMPLIFY_TOLERANCE_M = 3.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _local_xy(lat: float, lon: float, lat0: float) -> Tuple[float, float]:
    return (lon * 111320.0 * math.cos(math.radians(lat0)), lat * 110540.0)


def simplify(points: List[Tuple[float, float]], tolerance_m: float) -> List[Tuple[float, float]]:
    """Douglas-Peucker in a local metric projection."""
    if len(points) < 3:
        return list(points)
    lat0 = points[0][0]
    xy = [_local_xy(lat, lon, lat0) for lat, lon in points]
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        ax, ay = xy[a]
        bx, by = xy[b]
        dx, dy = bx - ax, by - ay
        seg_len = math.hypot(dx, dy)
        best, best_i = -1.0, -1
        for i in range(a + 1, b):
            px, py = xy[i]
            if seg_len < 1e-9:
                d = math.hypot(px - ax, py - ay)
            else:
                d = abs(dy * px - dx * py + bx * ay - by * ax) / seg_len
            if d > best:
                best, best_i = d, i
        if best > tolerance_m:
            keep[best_i] = True
            stack.append((a, best_i))
            stack.append((best_i, b))
    return [p for p, k in zip(points, keep) if k]


def _fmt_m(d: float) -> str:
    return f"{d:.0f} m" if math.isfinite(d) else "more than 2 km"


def polyline_length_m(points: List[Tuple[float, float]]) -> float:
    return sum(haversine_m(*points[i], *points[i + 1]) for i in range(len(points) - 1))


# ---------------------------------------------------------------------------
# Rail network graph
# ---------------------------------------------------------------------------

class RailGraph:
    def __init__(self, osm: dict):
        self.coord: Dict[int, Tuple[float, float]] = {}
        self.adj: Dict[int, List[Tuple[int, float, float]]] = defaultdict(list)  # node -> [(nbr, length_m, cost)]
        self.main_line_nodes = set()
        ways = [e for e in osm.get("elements", []) if e.get("type") == "way"]
        for way in ways:
            nodes, geom = way.get("nodes") or [], way.get("geometry") or []
            if len(nodes) != len(geom) or len(nodes) < 2:
                continue
            service = (way.get("tags") or {}).get("service")
            factor = SERVICE_COST.get(service, DEFAULT_SERVICE_COST)
            for nid, g in zip(nodes, geom):
                self.coord[nid] = (g["lat"], g["lon"])
                if service is None:
                    self.main_line_nodes.add(nid)
            for a, b in zip(nodes, nodes[1:]):
                length = haversine_m(*self.coord[a], *self.coord[b])
                self.adj[a].append((b, length, length * factor))
                self.adj[b].append((a, length, length * factor))
        self.component_size = self._component_sizes()
        self._grid: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for nid, (lat, lon) in self.coord.items():
            if self.component_size.get(nid, 0) >= MIN_COMPONENT_NODES:
                self._grid[(int(lat * 100), int(lon * 100))].append(nid)
        self.way_count = len(ways)

    def _component_sizes(self) -> Dict[int, int]:
        size: Dict[int, int] = {}
        seen = set()
        for start in self.adj:
            if start in seen:
                continue
            comp, stack = [start], [start]
            seen.add(start)
            while stack:
                x = stack.pop()
                for y, _, _ in self.adj[x]:
                    if y not in seen:
                        seen.add(y)
                        comp.append(y)
                        stack.append(y)
            for n in comp:
                size[n] = len(comp)
        return size

    def snap(self, lat: float, lon: float) -> Tuple[Optional[int], float]:
        """Nearest usable node; a main-line node wins over a closer yard node
        if it's within MAIN_LINE_SNAP_M."""
        gx, gy = int(lat * 100), int(lon * 100)
        best_any, best_any_d = None, float("inf")
        best_main, best_main_d = None, float("inf")
        for i in range(gx - 2, gx + 3):  # ~1.1 km cells -> ~2 km search radius
            for j in range(gy - 2, gy + 3):
                for nid in self._grid.get((i, j), ()):
                    d = haversine_m(lat, lon, *self.coord[nid])
                    if d < best_any_d:
                        best_any, best_any_d = nid, d
                    if nid in self.main_line_nodes and d < best_main_d:
                        best_main, best_main_d = nid, d
        if best_main is not None and best_main_d <= MAIN_LINE_SNAP_M:
            return best_main, best_main_d
        return best_any, best_any_d

    def snap_candidates(self, lat: float, lon: float, spread_m: float = 150.0) -> Dict[int, float]:
        """
        Every usable node within `spread_m` of the best snap, with its
        distance. Double-track sections are mapped in OSM as two parallel
        lines that only meet at crossovers, so the single nearest node of
        station A and of station B can sit on *different* tracks - forcing a
        long detour to the nearest crossover. Routing from all nearby
        candidates to all nearby candidates lets the search pick the track
        that actually connects them directly.
        """
        best, best_d = self.snap(lat, lon)
        if best is None:
            return {}
        main_only = best in self.main_line_nodes
        limit = best_d + spread_m
        gx, gy = int(lat * 100), int(lon * 100)
        out: Dict[int, float] = {}
        for i in range(gx - 2, gx + 3):
            for j in range(gy - 2, gy + 3):
                for nid in self._grid.get((i, j), ()):
                    if main_only and nid not in self.main_line_nodes:
                        continue
                    d = haversine_m(lat, lon, *self.coord[nid])
                    if d <= limit:
                        out[nid] = d
        out[best] = best_d
        return out

    def shortest_path(self, sources: Dict[int, float], targets: Dict[int, float],
                      max_cost: float) -> Optional[Tuple[List[int], float]]:
        """
        Multi-source / multi-target Dijkstra on weighted cost. Each source
        starts at its snap distance; reaching a target adds its snap
        distance. Returns (node path, real along-track length_m).
        """
        dist: Dict[int, float] = {}
        real: Dict[int, float] = {}
        prev: Dict[int, int] = {}
        heap = []
        for nid, d in sources.items():
            dist[nid] = d
            real[nid] = 0.0
            heapq.heappush(heap, (d, nid))
        best_total, best_target = float("inf"), None
        while heap:
            cost, node = heapq.heappop(heap)
            if cost >= best_total or cost > max_cost:
                break
            if cost > dist.get(node, float("inf")):
                continue
            if node in targets:
                total = cost + targets[node]
                if total < best_total:
                    best_total, best_target = total, node
            for nbr, length, edge_cost in self.adj[node]:
                nc = cost + edge_cost
                if nc < dist.get(nbr, float("inf")):
                    dist[nbr] = nc
                    real[nbr] = real[node] + length
                    prev[nbr] = node
                    heapq.heappush(heap, (nc, nbr))
        if best_target is None:
            return None
        path = [best_target]
        while path[-1] in prev:
            path.append(prev[path[-1]])
        return path[::-1], real[best_target]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def load_station_coords(data: dict) -> Dict[str, Tuple[float, float]]:
    coords = {}
    for source in ("sid_to_sloc", "xsid_to_sloc"):
        for sid, loc in (data.get(source) or {}).items():
            if loc and len(loc) >= 2 and not (loc[0] == 0 and loc[1] == 0):
                coords[sid] = (float(loc[0]), float(loc[1]))
    return coords


def collect_pairs(data: dict) -> Dict[Tuple[str, str], List[str]]:
    """Every adjacent station pair used by any train -> train ids using it."""
    pairs: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for tid, stations in (data.get("tid_to_stations") or {}).items():
        for a, b in zip(stations, stations[1:]):
            if a[0] != b[0]:
                pairs[pair_key(a[0], b[0])].append(tid)
    return pairs


def build(osm_path: str, data_path: str, out_path: str, stations_path: Optional[str] = None):
    t0 = time.time()
    with open(osm_path, encoding="utf-8") as f:
        osm = json.load(f)
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    graph = RailGraph(osm)
    print(f"Graph: {graph.way_count} ways, {len(graph.coord)} nodes, "
          f"{sum(len(v) for v in graph.adj.values()) // 2} edges")

    station_coords = load_station_coords(data)
    pairs = collect_pairs(data)
    used_stations = {s for pair in pairs for s in pair}

    snaps: Dict[str, Tuple[Optional[int], float]] = {}
    for sid in sorted(used_stations):
        if sid in station_coords:
            snaps[sid] = graph.snap(*station_coords[sid])

    segments = {}
    rejected: List[Tuple[str, str, str]] = []
    for (a, b), trains in sorted(pairs.items()):
        if a not in station_coords or b not in station_coords:
            rejected.append((a, b, "station has no coordinates"))
            continue
        na, da = snaps.get(a, (None, float("inf")))
        nb, db = snaps.get(b, (None, float("inf")))
        if na is None or da > MAX_SNAP_M:
            rejected.append((a, b, f"{a} is {_fmt_m(da)} from any track"))
            continue
        if nb is None or db > MAX_SNAP_M:
            rejected.append((a, b, f"{b} is {_fmt_m(db)} from any track"))
            continue
        straight = haversine_m(*graph.coord[na], *graph.coord[nb])
        if straight < 30:
            rejected.append((a, b, "stations snap to the same point"))
            continue
        result = graph.shortest_path(
            graph.snap_candidates(*station_coords[a]),
            graph.snap_candidates(*station_coords[b]),
            max_cost=straight * 8 + 20000,
        )
        if result is None:
            rejected.append((a, b, "no connected track path"))
            continue
        path, length = result
        straight = haversine_m(*graph.coord[path[0]], *graph.coord[path[-1]])
        if length / straight > MAX_DETOUR_RATIO and length - straight > MAX_DETOUR_EXTRA_M:
            rejected.append((a, b, f"track path {length / 1000:.1f} km vs {straight / 1000:.1f} km "
                                   f"straight - likely wrong branch"))
            continue
        points = [graph.coord[n] for n in path]
        # Store in canonical a->b order (pair key is already sorted).
        simplified = simplify(points, SIMPLIFY_TOLERANCE_M)
        segments[(a, b)] = (encode_polyline(simplified), length, len(simplified))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    conn = sqlite3.connect(tmp_path)
    conn.executescript("""
        CREATE TABLE track_segment (
            from_sid TEXT NOT NULL,
            to_sid   TEXT NOT NULL,
            polyline TEXT NOT NULL,   -- Google encoded polyline, precision 1e5, from_sid -> to_sid
            length_m REAL NOT NULL,   -- along-track length (unsimplified path)
            PRIMARY KEY (from_sid, to_sid)
        );
        CREATE TABLE station_snap (
            sid    TEXT PRIMARY KEY,
            lat    REAL NOT NULL,     -- station position snapped onto the track
            lng    REAL NOT NULL,
            snap_m REAL NOT NULL      -- distance moved by snapping
        );
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    conn.executemany(
        "INSERT INTO track_segment VALUES (?, ?, ?, ?)",
        [(a, b, enc, round(length, 1)) for (a, b), (enc, length, _) in segments.items()],
    )
    conn.executemany(
        "INSERT INTO station_snap VALUES (?, ?, ?, ?)",
        [
            (sid, round(graph.coord[n][0], 6), round(graph.coord[n][1], 6), round(d, 1))
            for sid, (n, d) in snaps.items()
            if n is not None and d <= MAX_SNAP_M
        ],
    )
    osm_ts = (osm.get("osm3s") or {}).get("timestamp_osm_base", "")
    build_id = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    conn.executemany("INSERT INTO meta VALUES (?, ?)", [
        ("schema_version", SCHEMA_VERSION),
        ("build_id", build_id),
        ("osm_timestamp", osm_ts),
        ("data_revision", str(data.get("Revision", ""))),
        ("segment_count", str(len(segments))),
        ("license", "Track geometry (c) OpenStreetMap contributors, ODbL 1.0"),
    ])
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    os.replace(tmp_path, out_path)

    total_points = sum(n for _, _, n in segments.values())
    print(f"Wrote {out_path}: {len(segments)}/{len(pairs)} pairs, {total_points} points, "
          f"{os.path.getsize(out_path) / 1024:.0f} KB in {time.time() - t0:.1f}s")

    write_report(out_path, data, graph, station_coords, snaps, pairs, segments, rejected, stations_path)
    return segments


def write_report(out_path, data, graph, station_coords, snaps, pairs, segments, rejected, stations_path):
    names = data.get("sid_to_sname") or {}

    def label(sid):
        n = names.get(sid)
        return f"{sid} ({n[0]})" if isinstance(n, list) and n and n[0] != sid else sid

    lines = [
        "Track build report",
        "==================",
        f"Pairs with real track: {len(segments)} / {len(pairs)}",
        f"Pairs drawn as straight lines: {len(rejected)}",
        "",
        "Stations more than 1 km from the nearest track (check these coordinates):",
    ]
    far = sorted(((d, sid) for sid, (_, d) in snaps.items() if d > 1000), reverse=True)
    for d, sid in far:
        lat, lon = station_coords[sid]
        dist = f"{d / 1000:6.2f} km" if math.isfinite(d) else "  > 2 km"
        lines.append(f"  {dist}  {label(sid)}  at {lat:.5f},{lon:.5f}")
    if not far:
        lines.append("  (none)")

    missing = sorted(sid for pair in pairs for sid in pair if sid not in station_coords)
    if missing:
        lines += ["", "Stations used by routes but missing coordinates:"]
        lines += [f"  {label(s)}" for s in sorted(set(missing))]

    if stations_path and os.path.exists(stations_path):
        with open(stations_path, encoding="utf-8") as f:
            osm_st = json.load(f)
        pts = []
        for e in osm_st.get("elements", []):
            c = e.get("center") or ({"lat": e["lat"], "lon": e["lon"]} if "lat" in e else None)
            if c:
                tags = e.get("tags") or {}
                pts.append((c["lat"], c["lon"], tags.get("name:en") or tags.get("name") or "?"))
        lines += ["", "Nearest OSM station to each of our stations (only those > 1 km apart):"]
        for sid, (lat, lon) in sorted(station_coords.items()):
            if not pts:
                break
            d, name = min((haversine_m(lat, lon, p[0], p[1]), p[2]) for p in pts)
            if d > 1000:
                lines.append(f"  {d / 1000:6.2f} km  {label(sid)}  nearest OSM station: {name}")

    lines += ["", "Pairs drawn as straight lines (reason):"]
    for a, b, why in rejected:
        lines.append(f"  {label(a)} <-> {label(b)}: {why}  [trains: {', '.join(sorted(pairs[pair_key(a, b)])[:6])}]")

    report_path = os.path.join(os.path.dirname(os.path.abspath(out_path)), "track_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report: {report_path}")


def preview(out_path: str, data_path: str, train_ids: List[str], png_path: str):
    """Render the given trains' old straight-line path vs the new track path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from functions.track_geometry import decode_polyline

    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)
    coords = load_station_coords(data)
    conn = sqlite3.connect(out_path)
    seg = {(a, b): p for a, b, p in conn.execute("SELECT from_sid, to_sid, polyline FROM track_segment")}
    fig, axes = plt.subplots(1, len(train_ids), figsize=(7 * len(train_ids), 9))
    if len(train_ids) == 1:
        axes = [axes]
    for ax, tid in zip(axes, train_ids):
        stations = data["tid_to_stations"][tid]
        old = [coords[s[0]] for s in stations if s[0] in coords]
        ax.plot([p[1] for p in old], [p[0] for p in old], "-", color="#e57373", lw=1.2, label="old (straight)")
        for a, b in zip(stations, stations[1:]):
            key = pair_key(a[0], b[0])
            if key in seg:
                pts = decode_polyline(seg[key])
                ax.plot([p[1] for p in pts], [p[0] for p in pts], "-", color="#0b6e4f", lw=1.6)
        stops = [coords[s[0]] for s in stations if s[1] == 1 and s[0] in coords]
        ax.plot([p[1] for p in stops], [p[0] for p in stops], "o", ms=3, color="black")
        ax.set_title(f"Train {tid}")
        ax.set_aspect(1 / math.cos(math.radians(old[0][0])))
        ax.legend(["old (straight)", "new (track)"], loc="best")
    plt.tight_layout()
    plt.savefig(png_path, dpi=110)
    print(f"Preview: {png_path}")


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--osm", default=os.path.join(root, "osm", "bd_rail.json"))
    p.add_argument("--data", default=os.path.join(root, "data.json"))
    p.add_argument("--stations", default=os.path.join(root, "osm", "stations.json"))
    p.add_argument("--out", default=os.path.join(root, "track", "track.db"))
    p.add_argument("--app-asset", default=None, help="also copy the result here (e.g. ../assets/track/track.db)")
    p.add_argument("--preview", nargs="*", default=None, help="train ids to render into track/preview.png")
    args = p.parse_args()

    build(args.osm, args.data, args.out, args.stations)
    if args.app_asset:
        import shutil
        os.makedirs(os.path.dirname(os.path.abspath(args.app_asset)), exist_ok=True)
        shutil.copyfile(args.out, args.app_asset)
        print(f"Copied to {args.app_asset}")
    if args.preview:
        preview(args.out, args.data, args.preview,
                os.path.join(os.path.dirname(os.path.abspath(args.out)), "preview.png"))


if __name__ == "__main__":
    main()
