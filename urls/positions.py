# positions.py - Train position endpoints

import hmac
import os
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Tuple

# Shared secret that authenticates a "bot" (trusted GPS source) update. The
# client-supplied user_id string must never grant bot trust on its own - it's
# fully attacker-controlled. If this isn't configured, no request can be
# treated as a bot (fail closed).
BOT_API_KEY = os.getenv("BOT_API_KEY", "")


class LocationUpdate(BaseModel):
    train_id: Optional[str] = None
    id: Optional[str] = None  # For backward compatibility
    user_id: Optional[str] = "unknown"
    time: int
    position: float
    bot_token: Optional[str] = None  # Shared secret for trusted bot updates
    speed_kmh: Optional[float] = None  # Phone's own GPS speed, if known (app v46+)
    # Note: scheduled_position is calculated automatically from train data


def _is_authenticated_bot(bot_token: Optional[str]) -> bool:
    if not BOT_API_KEY or not bot_token:
        return False
    return hmac.compare_digest(bot_token, BOT_API_KEY)


def get_current_positions(train_ids: str, tracker) -> Dict:
    """Return current positions for specified trains"""
    ids = train_ids.split(',')
    positions = tracker.get_positions(ids)
    print(f"Current positions for trains {ids}: {positions}")
    return positions


def get_train_bounds(train_id: str, tracker) -> Dict:
    """Debug endpoint: the reference position new pings for this train are
    currently being teleport-checked against (see RedisTrainTracker.push)."""
    bounds = tracker.get_train_bounds(train_id)
    if bounds:
        return {"train_id": train_id, "bounds": bounds}
    return {"train_id": train_id, "bounds": None, "message": "No reference position yet"}


def process_update(tracker, train_id: Optional[str], user_id: Optional[str], timestamp, position,
                   is_bot: bool = False, speed_kmh: Optional[float] = None) -> Tuple[bool, str]:
    """
    Validate and store one location update. Shared by the HTTP endpoint
    (/sendupdate) and the live WebSocket (functions/live_hub.py) so both
    paths run exactly the same checks. Returns (accepted, message).
    """
    user_id = user_id or 'unknown'
    if train_id is None or timestamp is None or position is None:
        return False, "Missing required fields"
    try:
        position = float(position)
        timestamp = int(timestamp)
    except (TypeError, ValueError):
        return False, "Invalid field types"
    if not (0 <= position <= 150):
        return False, "Invalid position value"
    if speed_kmh is not None:
        try:
            speed_kmh = float(speed_kmh)
        except (TypeError, ValueError):
            speed_kmh = None

    print(f"{'[BOT]' if is_bot else '[USER]'} train={train_id} pos={position} user={user_id} ts={timestamp}")
    success, message = tracker.push(str(train_id), str(user_id), position, timestamp,
                                    is_bot=is_bot, speed_kmh=speed_kmh)
    if not success:
        print(f"Position rejected: {message}")
    return success, message


def receive_update(update: LocationUpdate, tracker):
    """Receive location update from user"""
    # Support both old and new formats for compatibility
    train_id = update.train_id or update.id
    if not (0 <= update.position <= 150):
        raise HTTPException(400, "Invalid position value")
    if train_id is None or update.time is None or update.position is None:
        raise HTTPException(status_code=400, detail="Missing required fields")

    success, message = process_update(
        tracker, train_id, update.user_id, update.time, update.position,
        is_bot=_is_authenticated_bot(update.bot_token), speed_kmh=update.speed_kmh,
    )
    if not success:
        raise HTTPException(400, f"Position rejected: {message}")
    return {"status": "success", "message": message}
