"""Small per-profile catalog cache for the temporary fallback_models property bridge."""
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Callable


@dataclass
class _Entry:
    models: tuple[str, ...] | None = None
    rows: tuple[dict, ...] = ()
    retry_at: float = 0
    fetching: bool = False


class CatalogCache:
    """Keep successful catalogs for five minutes; failed probes back off for one minute."""

    def __init__(self):
        self._entries: OrderedDict[tuple[str, str, str], _Entry] = OrderedDict()
        self._lock = Lock()

    def _entry(self, key: tuple[str, str, str]) -> _Entry:
        entry = self._entries.setdefault(key, _Entry())
        self._entries.move_to_end(key)
        # Do not evict in-flight entries: another reader must still see their fetch marker.
        for old_key in list(self._entries):
            if len(self._entries) <= 32:
                break
            if old_key != key and not self._entries[old_key].fetching:
                del self._entries[old_key]
        return entry

    def remember(self, key: tuple[str, str, str], models: list[str] | None, *, rows=None) -> None:
        with self._lock:
            entry = self._entry(key)
            if models is not None:
                entry.models = tuple(models)
                entry.rows = tuple(dict(row) for row in (rows or []))
            entry.retry_at = monotonic() + (300 if models is not None else 60)

    def read(self, key: tuple[str, str, str], fetch: Callable[[], list[str] | None]) -> tuple[str, ...] | None:
        with self._lock:
            entry = self._entry(key)
            if entry.fetching or monotonic() < entry.retry_at:
                return entry.models
            entry.fetching = True
        try:
            models = fetch()
            self.remember(key, models, rows=self.rows(key))
        finally:
            with self._lock:
                entry.fetching = False
        return entry.models

    def rows(self, key: tuple[str, str, str]) -> list[dict]:
        with self._lock:
            entry = self._entries.get(key)
            return [dict(row) for row in entry.rows] if entry else []
