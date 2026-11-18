#!/usr/bin/env python3
"""
Nightly backup of every project's data on this server, driven by a JSON
config - add a project (e.g. EduShare) by adding an entry, no code change.

Per project, any mix of:
    "sqlite":   SQLite files, copied with SQLite's online backup API (a plain
                cp of a live WAL database can be inconsistent), then checked
    "mysql":    MySQL/MariaDB databases, dumped with mysqldump
                --single-transaction (consistent for InnoDB while it runs)
    "files":    plain files or directories, copied as they are
    "commands": any dump command whose output is the backup, e.g.
                {"name": "edushare.sql", "run": ["sudo", "-n", "-u", "postgres", "pg_dump", "edushare"]}

Each project becomes one verified .tar.xz, kept locally (last keep_local) and
uploaded to Oracle Object Storage through a write-only pre-authenticated
request URL (read from upload_link_file, never logged). That URL can upload
but not list or delete, so retention works by reusing a fixed set of names -
each upload replaces the oldest copy:

    <project>/daily/Mon.tar.xz ... Sun     every run       (7)
    <project>/weekly/w0.tar.xz ... w3      Fridays         (4)
    <project>/monthly/01.tar.xz ... 12     1st of month    (12)

i.e. at most 23 objects per project. Keep object versioning OFF on the
bucket, or replaced copies pile up as hidden versions.

A failing project doesn't stop the others. The outcome goes to status.json
next to the config (shown in Find My Train's /health) and to stdout.
Standard library only - runs on the server's system Python 3.8.

Example config (~/server-backup/backup.json):
    {
      "upload_link_file": "~/.fmt_backup_par",
      "keep_local": 7,
      "projects": {
        "find-my-train": {
          "sqlite": ["~/find-my-br-train/history.db", "~/find-my-br-train/gps_updates.db"],
          "files": ["~/find-my-br-train/issue_reports.log"]
        },
        "clash-of-empires": {"mysql": ["clash_db"]}
      }
    }

Cron (04:00 Bangladesh time = 22:00 UTC):
    0 22 * * * /usr/bin/python3 /home/ubuntu/server-backup/server_backup.py >> /home/ubuntu/server-backup/backup.log 2>&1

Restore: download e.g. find-my-train/daily/Mon.tar.xz in the OCI console,
`tar -xJf Mon.tar.xz`, stop the app, put the files back (paths inside the
archive are relative to the home directory; MySQL dumps restore with
`sudo mysql < mysql/clash_db.sql`), start the app.

Flags: --all-slots (also fill the weekly/monthly slots now - for the first
run), --no-upload (local only), --only NAME (one project), --config PATH.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

HOME = os.path.expanduser("~")
BASE_DIR = os.getenv("SERVER_BACKUP_DIR", os.path.join(HOME, "server-backup"))
DEFAULT_CONFIG = os.path.join(BASE_DIR, "backup.json")

XZ_PRESET = 6
UPLOAD_TRIES = 3
WEEKLY_ON = 4  # Friday (datetime.weekday())
DEFAULT_MYSQLDUMP = ["sudo", "-n", "mysqldump"]
BD_TZ = timezone(timedelta(hours=6))


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def arcname(path: str) -> str:
    """Path inside the archive: relative to the home directory."""
    rel = os.path.relpath(path, HOME)
    return rel if not rel.startswith("..") else path.lstrip("/")


def write_json(path: str, data: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Collecting one project's data into a work directory
# ---------------------------------------------------------------------------

def collect(project: dict, work: str, mysqldump: list) -> tuple:
    """-> (entries {arcname: (source path, kind)}, sizes {arcname: bytes}, missing [paths])."""
    entries, sizes, missing = {}, {}, []

    for path in map(expand, project.get("sqlite", [])):
        if not os.path.exists(path):
            missing.append(path)
            continue
        name = arcname(path)
        dst_path = os.path.join(work, name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
        dst = sqlite3.connect(dst_path)
        try:
            src.backup(dst)
            check = dst.execute("PRAGMA quick_check").fetchone()[0]
        finally:
            dst.close()
            src.close()
        if check != "ok":
            raise RuntimeError(f"{name}: integrity check failed on the copy: {check}")
        entries[name] = (dst_path, "sqlite")
        sizes[name] = os.path.getsize(dst_path)

    for db in project.get("mysql", []):
        name = f"mysql/{db}.sql"
        dst_path = os.path.join(work, name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cmd = mysqldump + ["--single-transaction", "--quick", "--routines", "--triggers", "--events",
                           "--databases", db]
        with open(dst_path, "wb") as out:
            r = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"mysqldump {db} failed: {r.stderr.decode(errors='replace').strip()[:300]}")
        entries[name] = (dst_path, "mysql")
        sizes[name] = os.path.getsize(dst_path)

    for spec in project.get("commands", []):
        name = f"commands/{spec['name']}"
        dst_path = os.path.join(work, name)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with open(dst_path, "wb") as out:
            r = subprocess.run(spec["run"], stdout=out, stderr=subprocess.PIPE, timeout=3600)
        if r.returncode != 0 or os.path.getsize(dst_path) == 0:
            raise RuntimeError(f"command for {spec['name']} failed (exit {r.returncode}): "
                               f"{r.stderr.decode(errors='replace').strip()[:300]}")
        entries[name] = (dst_path, "command")
        sizes[name] = os.path.getsize(dst_path)

    for path in map(expand, project.get("files", [])):
        if not os.path.exists(path):
            missing.append(path)
            continue
        name = arcname(path)
        entries[name] = (path, "file")
        if os.path.isdir(path):
            sizes[name] = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
        else:
            sizes[name] = os.path.getsize(path)

    if not entries:
        raise RuntimeError("nothing to back up (all sources missing?)")
    return entries, sizes, missing


def make_archive(entries: dict, archive: str) -> None:
    with tarfile.open(archive, "w:xz", preset=XZ_PRESET) as tar:
        for name in sorted(entries):
            tar.add(entries[name][0], arcname=name)


def verify_archive(archive: str, entries: dict, scratch: str) -> None:
    """Re-open the archive and check every database in it."""
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        with tarfile.open(archive, "r:xz") as tar:
            names = set(tar.getnames())
            missing = set(entries) - names
            if missing:
                raise RuntimeError(f"archive is missing {sorted(missing)}")
            for name, (_, kind) in entries.items():
                if kind == "sqlite":
                    tar.extract(name, tmp)
                    conn = sqlite3.connect(os.path.join(tmp, name))
                    try:
                        check = conn.execute("PRAGMA quick_check").fetchone()[0]
                    finally:
                        conn.close()
                    if check != "ok":
                        raise RuntimeError(f"{name} in the archive fails its check: {check}")
                elif kind == "mysql":
                    f = tar.extractfile(name)
                    f.seek(0, os.SEEK_END)
                    f.seek(max(0, f.tell() - 400))
                    if b"Dump completed" not in f.read():
                        raise RuntimeError(f"{name} in the archive is incomplete")


def rotate_local(folder: str, keep: int) -> None:
    archives = sorted(
        (os.path.join(folder, n) for n in os.listdir(folder) if n.endswith(".tar.xz")),
        key=os.path.getmtime,
    )
    for old in archives[:-keep]:
        os.remove(old)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def slots_for(project: str, day: datetime, all_slots: bool) -> list:
    slots = [f"{project}/daily/{day.strftime('%a')}.tar.xz"]
    if all_slots or day.weekday() == WEEKLY_ON:
        slots.append(f"{project}/weekly/w{day.isocalendar()[1] % 4}.tar.xz")
    if all_slots or day.day == 1:
        slots.append(f"{project}/monthly/{day.strftime('%m')}.tar.xz")
    return slots


def upload(par_url: str, archive: str, name: str) -> None:
    url = par_url.rstrip("/") + "/" + name
    size = os.path.getsize(archive)
    last = None
    for attempt in range(1, UPLOAD_TRIES + 1):
        try:
            with open(archive, "rb") as f:
                req = urllib.request.Request(
                    url, data=f, method="PUT",
                    headers={"Content-Type": "application/x-xz", "Content-Length": str(size)},
                )
                with urllib.request.urlopen(req, timeout=600) as resp:
                    if resp.status in (200, 201):
                        return
                    last = f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (401, 403, 404):  # link expired or deleted - retrying won't help
                break
        except (urllib.error.URLError, OSError) as e:
            last = type(e).__name__
        if attempt < UPLOAD_TRIES:
            time.sleep(30 * attempt)
    raise RuntimeError(f"upload of {name} failed ({last}) - is the upload link expired or deleted?")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def backup_project(name: str, project: dict, cfg: dict, day: datetime, par_url, all_slots: bool) -> dict:
    status = {"ok": False, "archive": None, "archive_bytes": None, "contents": {}, "missing": [],
              "uploaded": [], "error": None}
    folder = os.path.join(BASE_DIR, "local", name)
    os.makedirs(folder, exist_ok=True)
    archive = os.path.join(folder, f"{name}-{day.strftime('%Y%m%d-%H%M')}.tar.xz")
    try:
        with tempfile.TemporaryDirectory(dir=BASE_DIR) as work:
            entries, status["contents"], status["missing"] = collect(
                project, work, cfg.get("mysqldump_command", DEFAULT_MYSQLDUMP))
            make_archive(entries, archive + ".part")
            os.replace(archive + ".part", archive)
            verify_archive(archive, entries, BASE_DIR)
        status["archive"] = os.path.basename(archive)
        status["archive_bytes"] = os.path.getsize(archive)
        rotate_local(folder, int(cfg.get("keep_local", 7)))
        log(f"[{name}] local backup ok: {status['archive']} ({status['archive_bytes'] // 1024} KB) from "
            + ", ".join(f"{k} {v // 1024} KB" for k, v in status["contents"].items())
            + (f"; MISSING {status['missing']}" if status["missing"] else ""))
        if par_url is not None:
            for slot in slots_for(name, day, all_slots):
                upload(par_url, archive, slot)
                status["uploaded"].append(slot)
            log(f"[{name}] uploaded as {', '.join(status['uploaded'])}")
        status["ok"] = True
    except Exception as e:  # noqa: BLE001 - every failure must end up in status.json
        status["error"] = f"{type(e).__name__}: {e}"
        log(f"[{name}] BACKUP FAILED: {status['error']}")
        if os.path.exists(archive + ".part"):
            os.remove(archive + ".part")
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description="Back up every project's data (see the module docstring).")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--all-slots", action="store_true", help="also fill the weekly and monthly slots now")
    ap.add_argument("--no-upload", action="store_true", help="local backup only")
    ap.add_argument("--only", help="back up just this project")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    projects = cfg.get("projects") or {}
    if args.only:
        projects = {args.only: projects[args.only]}
    os.makedirs(BASE_DIR, exist_ok=True)

    lock = open(os.path.join(BASE_DIR, ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another backup is still running - skipping")
        return 1
    try:
        os.nice(10)
    except OSError:
        pass

    started = time.time()
    day = datetime.now(BD_TZ)
    par_url, link_error = None, None
    if not args.no_upload:
        link_file = expand(cfg.get("upload_link_file", "~/.fmt_backup_par"))
        if os.path.exists(link_file):
            par_url = open(link_file).read().strip()
        else:
            link_error = f"no upload link in {link_file}"
            log(link_error)

    results = {}
    for name, project in projects.items():
        results[name] = backup_project(name, project, cfg, day, par_url, args.all_slots)
        if link_error and results[name]["ok"]:
            results[name].update(ok=False, error=link_error)

    status = {
        "ok": bool(results) and all(r["ok"] for r in results.values()),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.time() - started, 1),
        "projects": results,
    }
    status_path = os.path.join(os.path.dirname(expand(args.config)), "status.json")
    if args.only and os.path.exists(status_path):  # keep the other projects' last results
        try:
            with open(status_path) as f:
                previous = json.load(f)
            merged = dict(previous.get("projects") or {})
            merged.update(results)
            status["projects"] = merged
            status["ok"] = all(r.get("ok") for r in merged.values())
        except (OSError, ValueError):
            pass
    write_json(status_path, status)
    log(f"done in {status['duration_s']} s: " + ", ".join(
        f"{n} {'ok' if r['ok'] else 'FAILED'}" for n, r in results.items()))
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
