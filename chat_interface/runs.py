"""In-process registry of running agent generations.

A generation runs in a background thread and appends SSE-ready frames to a
RunHandle. Any number of HTTP responses can subscribe to a handle: the original
POST, or a later /reconnect after a browser refresh. Each subscriber replays the
buffered frames, then tails live ones until the run finishes. The background
thread owns the run lifecycle, so the assistant message is persisted (by the
event generator it consumes) even if every client has disconnected.

This is intentionally a single-process, in-memory design — it fits the Django
dev server / a single Gunicorn worker. A multi-worker deployment would need a
shared backing store (Redis pub/sub) instead.
"""
import threading

from django.db import connection

_REGISTRY: dict[str, "RunHandle"] = {}
_REGISTRY_LOCK = threading.Lock()


class RunHandle:
    """A growing, replayable buffer of SSE frame strings for one generation."""

    def __init__(self):
        self._frames: list[str] = []
        self._cv = threading.Condition()
        self._done = False

    def emit(self, frame: str) -> None:
        with self._cv:
            self._frames.append(frame)
            self._cv.notify_all()

    def finish(self) -> None:
        with self._cv:
            self._done = True
            self._cv.notify_all()

    def subscribe(self):
        """Yield every frame — replayed from the start, then live — until done."""
        idx = 0
        while True:
            with self._cv:
                while idx >= len(self._frames) and not self._done:
                    self._cv.wait()
                new = self._frames[idx:]
                idx = len(self._frames)
                finished = self._done and idx >= len(self._frames)
            for frame in new:
                yield frame
            if finished:
                return


def get_run(session_id) -> "RunHandle | None":
    with _REGISTRY_LOCK:
        return _REGISTRY.get(str(session_id))


def start_run(session_id, frame_iterable_factory) -> "RunHandle":
    """Start a background generation unless one is already active for this session.

    frame_iterable_factory() must return an iterator of SSE frame strings (e.g.
    _stream_agent_events(...)). If a run is already active the factory is NOT
    called and the existing handle is returned, which both de-duplicates double
    submits and lets a reconnect attach to the in-flight run.
    """
    key = str(session_id)
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(key)
        if existing is not None:
            return existing
        handle = RunHandle()
        _REGISTRY[key] = handle

    def worker():
        try:
            for frame in frame_iterable_factory():
                handle.emit(frame)
        finally:
            handle.finish()
            with _REGISTRY_LOCK:
                if _REGISTRY.get(key) is handle:
                    del _REGISTRY[key]
            # Background threads get their own DB connection; close it so it
            # isn't leaked past the run.
            connection.close()

    threading.Thread(target=worker, daemon=True, name=f"agent-run-{key}").start()
    return handle
