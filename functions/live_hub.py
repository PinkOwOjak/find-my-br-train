# live_hub.py - One WebSocket per app for live train positions.
#
# Replaces polling /current and /activetrains every 10 s: the app opens a
# single connection to /ws/live, subscribes to the trains it's showing (or to
# "all" for the map), and the server pushes each train's state as soon as it
# changes - at most once every MIN_TRAIN_INTERVAL_S per train, so a busy
# train with several reporters can't flood viewers. Riders can also send
# their location over the same connection instead of POSTing /sendupdate;
# it goes through exactly the same validation (urls/positions.process_update).
#
# Multi-worker safe: RedisTrainTracker publishes every accepted position on
# a Redis pub/sub channel and every worker's hub listens to it, so a rider
# connected to worker A reaches viewers connected to worker B. With a single
# worker the pub/sub round-trip is just a little redundant.
#
# Protocol (JSON text frames, "t" = type):
#   client -> server
#     {"t":"hello","user_id":"user_123456","app":"46"}
#     {"t":"sub","trains":["701","702"]}      {"t":"unsub","trains":["701"]}
#     {"t":"sub_all"}                          {"t":"unsub_all"}
#     {"t":"pos","ref":7,"train_id":"701","position":3.42,"time":<ms|s>,"speed_kmh":54.1}
#     {"t":"ping"}
#   server -> client
#     {"t":"snap","scope":"trains"|"all","trains":{"701":{...state...}, ...}}
#     {"t":"upd","trains":{"701":{...state...}|null}}    (null = no data any more)
#     {"t":"ack","ref":7,"ok":true,"msg":"Position updated"}
#     {"t":"pong","server_time":1763412345}
#   state = {"position","timestamp","active_user","is_live","speed_kmh","moving",
#            "plan": [[position, seconds_after_timestamp], ...] | null}

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

from urls.positions import process_update

MIN_TRAIN_INTERVAL_S = 2.0     # per-train broadcast throttle
FLUSH_INTERVAL_S = 0.5
ALL_SNAPSHOT_INTERVAL_S = 30.0 # full refresh for "all trains" subscribers (catches expiries)
IDLE_TIMEOUT_S = 90.0          # client pings every ~25 s
POS_MIN_INTERVAL_S = 1.5       # per-connection rider update rate limit
MAX_TRAIN_SUBS = 30
MAX_MESSAGE_BYTES = 4096
MAX_CLIENTS = 20000
SEND_TIMEOUT_S = 5.0

STATE_KEYS = ("position", "timestamp", "active_user", "is_live", "speed_kmh", "moving", "plan")


def public_state(info: Optional[Dict]) -> Optional[Dict]:
    if not info:
        return None
    return {k: info.get(k) for k in STATE_KEYS}


@dataclass(eq=False)
class _Client:
    ws: WebSocket
    user_id: str = "unknown"
    app_version: str = ""
    trains: Set[str] = field(default_factory=set)
    all_trains: bool = False
    last_pos_at: float = 0.0
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class LiveHub:
    def __init__(self, tracker, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
        self.tracker = tracker
        self._redis_args = dict(host=redis_host, port=redis_port, db=redis_db)
        self.clients: Set[_Client] = set()
        self.train_subs: Dict[str, Set[_Client]] = {}
        self.all_subs: Set[_Client] = set()
        self._dirty: Set[str] = set()
        self._last_sent: Dict[str, float] = {}
        self._tasks: List[asyncio.Task] = []
        self.stats = {"connections_total": 0, "messages_out": 0, "positions_in": 0}

    # -- lifecycle --------------------------------------------------------------

    async def start(self):
        if self._tasks:
            return
        self._tasks = [
            asyncio.create_task(self._pubsub_loop(), name="livehub-pubsub"),
            asyncio.create_task(self._flush_loop(), name="livehub-flush"),
            asyncio.create_task(self._all_snapshot_loop(), name="livehub-snapshot"),
        ]

    async def stop(self):
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []

    def mark_dirty(self, train_id: str):
        self._dirty.add(str(train_id))

    def summary(self) -> Dict:
        return {
            "clients": len(self.clients),
            "all_subscribers": len(self.all_subs),
            "trains_watched": len(self.train_subs),
            **self.stats,
        }

    # -- background loops ---------------------------------------------------------

    async def _pubsub_loop(self):
        import redis.asyncio as aioredis

        while True:
            client = None
            try:
                client = aioredis.Redis(decode_responses=True, **self._redis_args)
                pubsub = client.pubsub()
                await pubsub.subscribe(self.tracker.updates_channel)
                async for message in pubsub.listen():
                    if message.get("type") == "message" and message.get("data"):
                        self._dirty.add(str(message["data"]))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"LiveHub pub/sub error (retrying in 3s): {e}")
                await asyncio.sleep(3)
            finally:
                if client is not None:
                    try:
                        await client.aclose()
                    except Exception:
                        pass

    async def _flush_loop(self):
        while True:
            await asyncio.sleep(FLUSH_INTERVAL_S)
            try:
                await self._flush()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"LiveHub flush error: {e}")

    async def _flush(self):
        if not self._dirty:
            return
        now = time.monotonic()
        ready = [t for t in self._dirty if now - self._last_sent.get(t, 0.0) >= MIN_TRAIN_INTERVAL_S]
        if not ready:
            return
        for t in ready:
            self._dirty.discard(t)
        wanted = [t for t in ready if t in self.train_subs or self.all_subs]
        if not wanted:
            return
        positions = await asyncio.to_thread(self.tracker.get_positions, wanted)
        for t in wanted:
            self._last_sent[t] = now
        states = {t: public_state(positions.get(t)) for t in wanted}

        recipients: Dict[_Client, Dict[str, Optional[Dict]]] = {}
        for t, state in states.items():
            for c in self.train_subs.get(t, ()):
                recipients.setdefault(c, {})[t] = state
            if state and state.get("is_live"):
                for c in self.all_subs:
                    recipients.setdefault(c, {})[t] = state
        await asyncio.gather(*(
            self._send(c, {"t": "upd", "trains": trains}) for c, trains in recipients.items()
        ))

    async def _all_snapshot_loop(self):
        while True:
            await asyncio.sleep(ALL_SNAPSHOT_INTERVAL_S)
            if not self.all_subs:
                continue
            try:
                snapshot = await asyncio.to_thread(self._live_snapshot)
                msg = {"t": "snap", "scope": "all", "trains": snapshot}
                await asyncio.gather(*(self._send(c, msg) for c in list(self.all_subs)))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"LiveHub snapshot error: {e}")

    def _live_snapshot(self) -> Dict[str, Dict]:
        active = self.tracker.get_all_active_trains()
        positions = self.tracker.get_positions(active)
        return {t: public_state(info) for t, info in positions.items() if info.get("is_live")}

    # -- sending ------------------------------------------------------------------

    async def _send(self, client: _Client, message: Dict) -> bool:
        payload = json.dumps(message, separators=(",", ":"))
        try:
            async with client.send_lock:
                await asyncio.wait_for(client.ws.send_text(payload), timeout=SEND_TIMEOUT_S)
            self.stats["messages_out"] += 1
            return True
        except Exception:
            self._drop(client)
            return False

    # -- connections --------------------------------------------------------------

    def _drop(self, client: _Client):
        self.clients.discard(client)
        self.all_subs.discard(client)
        for t in list(client.trains):
            subs = self.train_subs.get(t)
            if subs is not None:
                subs.discard(client)
                if not subs:
                    del self.train_subs[t]
        client.trains.clear()

    def _subscribe(self, client: _Client, train_ids: Iterable[str]) -> List[str]:
        added = []
        for t in train_ids:
            t = str(t).strip()[:16]
            if not t or t in client.trains or len(client.trains) >= MAX_TRAIN_SUBS:
                continue
            client.trains.add(t)
            self.train_subs.setdefault(t, set()).add(client)
            added.append(t)
        return added

    def _unsubscribe(self, client: _Client, train_ids: Iterable[str]):
        for t in train_ids:
            t = str(t)
            client.trains.discard(t)
            subs = self.train_subs.get(t)
            if subs is not None:
                subs.discard(client)
                if not subs:
                    del self.train_subs[t]

    async def handle(self, ws: WebSocket):
        if len(self.clients) >= MAX_CLIENTS:
            await ws.close(code=1013)
            return
        await ws.accept()
        client = _Client(ws=ws)
        self.clients.add(client)
        self.stats["connections_total"] += 1
        try:
            while True:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=IDLE_TIMEOUT_S)
                if len(raw) > MAX_MESSAGE_BYTES:
                    continue
                try:
                    msg = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if isinstance(msg, dict):
                    await self._dispatch(client, msg)
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        except Exception as e:
            print(f"LiveHub connection error: {e}")
        finally:
            self._drop(client)
            try:
                await ws.close()
            except Exception:
                pass

    async def _dispatch(self, client: _Client, msg: Dict):
        kind = msg.get("t")
        if kind == "ping":
            await self._send(client, {"t": "pong", "server_time": int(time.time())})
        elif kind == "hello":
            uid = str(msg.get("user_id") or "unknown")[:64]
            client.user_id = uid
            client.app_version = str(msg.get("app") or "")[:16]
        elif kind == "sub":
            trains = msg.get("trains") or []
            if isinstance(trains, list):
                added = self._subscribe(client, trains[:MAX_TRAIN_SUBS])
                if added:
                    positions = await asyncio.to_thread(self.tracker.get_positions, added)
                    await self._send(client, {
                        "t": "snap", "scope": "trains",
                        "trains": {t: public_state(positions.get(t)) for t in added},
                    })
        elif kind == "unsub":
            trains = msg.get("trains") or []
            if isinstance(trains, list):
                self._unsubscribe(client, trains)
        elif kind == "sub_all":
            self.all_subs.add(client)
            snapshot = await asyncio.to_thread(self._live_snapshot)
            await self._send(client, {"t": "snap", "scope": "all", "trains": snapshot})
        elif kind == "unsub_all":
            self.all_subs.discard(client)
        elif kind == "pos":
            await self._handle_position(client, msg)

    async def _handle_position(self, client: _Client, msg: Dict):
        ref = msg.get("ref")
        now = time.monotonic()
        if now - client.last_pos_at < POS_MIN_INTERVAL_S:
            await self._send(client, {"t": "ack", "ref": ref, "ok": False, "msg": "Too many updates"})
            return
        client.last_pos_at = now
        self.stats["positions_in"] += 1
        train_id = msg.get("train_id")
        user_id = client.user_id if client.user_id != "unknown" else msg.get("user_id")
        ok, text = await asyncio.to_thread(
            process_update, self.tracker, train_id, user_id, msg.get("time"),
            msg.get("position"), False, msg.get("speed_kmh"),
        )
        if ok and train_id is not None:
            self.mark_dirty(str(train_id))
        await self._send(client, {"t": "ack", "ref": ref, "ok": ok, "msg": text})
