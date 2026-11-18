# backup_status.py - the last nightly server backup, for /health.
#
# tools/server_backup.py (cron, 04:00 BD time) writes its outcome to
# ~/server-backup/status.json. /health is public, so this only says whether
# the backup is OK and fresh - not which other projects are on the server or
# why something failed (that's in status.json / backup.log on the server).

import json
import os
from datetime import datetime, timezone
from typing import Optional

STATUS_PATH = os.path.expanduser(os.getenv("SERVER_BACKUP_STATUS", "~/server-backup/status.json"))
# Runs nightly; older than this means a run was missed.
MAX_AGE_HOURS = 30


def summary(path: Optional[str] = None, now: Optional[datetime] = None) -> dict:
    try:
        with open(path or STATUS_PATH) as f:
            status = json.load(f)
    except (OSError, ValueError):
        return {"ok": None, "note": "no backup has run yet"}
    now = now or datetime.now(timezone.utc)
    age_hours = None
    try:
        finished = datetime.fromisoformat(status["finished_at"])
        age_hours = round((now - finished).total_seconds() / 3600, 1)
    except (KeyError, TypeError, ValueError):
        pass
    projects = status.get("projects") or {}
    good = sum(1 for p in projects.values() if p.get("ok"))
    return {
        "ok": bool(status.get("ok")) and age_hours is not None and age_hours <= MAX_AGE_HOURS,
        "last_finished_at": status.get("finished_at"),
        "age_hours": age_hours,
        "projects_ok": f"{good}/{len(projects)}",
    }
