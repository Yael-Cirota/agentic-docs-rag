"""
watcher.py – Watch .md/.mdx files for changes and rebuild the FAISS index.

How it works
------------
  1. A watchdog Observer monitors each project root recursively.
  2. Any create / modify / delete / rename on a .md or .mdx file schedules
     a debounced rebuild (default: 2 seconds after the last event).
  3. After a successful rebuild the optional `on_rebuilt(index)` callback is
     called – app.py uses this to hot-swap the live query engine without a
     server restart.

Usage (standalone – keeps index fresh while you edit files)
-----------------------------------------------------------
    python watcher.py ./sample_project           # watch + rebuild forever
    python watcher.py ./proj1 ./proj2            # watch multiple roots

Usage (programmatic – called from app.py)
-----------------------------------------
    from watcher import start_watcher

    observer = start_watcher(
        project_roots=["./sample_project"],
        on_rebuilt=lambda index: reload_engine(index),
    )
    # … later …
    observer.stop()
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

_MD_SUFFIXES = {".md", ".mdx"}


# ─────────────────────────────────────────────────────────────────────────────
# Debounced rebuilder
# ─────────────────────────────────────────────────────────────────────────────

class _IndexRebuilder:
    """
    Collects file-change events and triggers a full index rebuild after
    DEBOUNCE seconds of quiet (so rapid saves don't cause many rebuilds).
    """

    DEBOUNCE: float = 2.0

    def __init__(self, project_roots: list[str | Path], on_rebuilt=None):
        self._roots = [str(r) for r in project_roots]
        self._on_rebuilt = on_rebuilt          # optional callback(index)
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._pending_paths: set[str] = set()

    def schedule(self, path: str) -> None:
        """Called from the watchdog thread each time a file event fires."""
        with self._lock:
            self._pending_paths.add(path)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.DEBOUNCE, self._rebuild)
            self._timer.daemon = True
            self._timer.start()

    def _rebuild(self) -> None:
        with self._lock:
            changed = sorted(self._pending_paths)
            self._pending_paths.clear()
            self._timer = None

        logger.info(
            "Detected %d changed file(s): %s – rebuilding index …",
            len(changed),
            ", ".join(Path(p).name for p in changed),
        )

        try:
            from pipeline import build_index  # imported here to avoid circular imports
            index = build_index(project_roots=self._roots)
            logger.info("✅ Index rebuilt successfully.")
            if self._on_rebuilt is not None:
                self._on_rebuilt(index)
        except Exception as exc:
            logger.error("❌ Index rebuild failed: %s", exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# Watchdog event handler
# ─────────────────────────────────────────────────────────────────────────────

class _MDFileHandler(FileSystemEventHandler):
    """Forwards relevant file-system events to the rebuilder."""

    def __init__(self, rebuilder: _IndexRebuilder):
        super().__init__()
        self._rebuilder = rebuilder

    @staticmethod
    def _is_md(path: str) -> bool:
        return Path(path).suffix.lower() in _MD_SUFFIXES

    def on_modified(self, event) -> None:
        if not event.is_directory and self._is_md(event.src_path):
            self._rebuilder.schedule(event.src_path)

    def on_created(self, event) -> None:
        if not event.is_directory and self._is_md(event.src_path):
            self._rebuilder.schedule(event.src_path)

    def on_deleted(self, event) -> None:
        if not event.is_directory and self._is_md(event.src_path):
            self._rebuilder.schedule(event.src_path)

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        if self._is_md(event.src_path) or self._is_md(event.dest_path):
            self._rebuilder.schedule(event.dest_path)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def start_watcher(
    project_roots: list[str | Path],
    on_rebuilt=None,
) -> Observer:
    """
    Start a background Observer that watches *project_roots* for .md changes.

    Parameters
    ----------
    project_roots : list of paths
        Folders to watch recursively.
    on_rebuilt : callable(index) | None
        Called on the rebuilder thread after every successful rebuild.
        Use it to hot-swap the live query engine (see app.py).

    Returns
    -------
    Observer
        Call `observer.stop(); observer.join()` to shut down cleanly.
    """
    rebuilder = _IndexRebuilder(project_roots=project_roots, on_rebuilt=on_rebuilt)
    handler = _MDFileHandler(rebuilder)

    observer = Observer()
    for root in project_roots:
        path = str(Path(root).resolve())
        observer.schedule(handler, path=path, recursive=True)
        logger.info("👁  Watching: %s", path)

    observer.start()
    logger.info(
        "File watcher started. Monitoring %d folder(s) for .md changes.",
        len(project_roots),
    )
    return observer


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point  (python watcher.py ./sample_project)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python watcher.py <project_root> [<project_root> ...]")
        sys.exit(1)

    roots = sys.argv[1:]
    observer = start_watcher(project_roots=roots)

    print("Watching for .md changes. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping watcher …")
        observer.stop()
        observer.join()
        print("Done.")
