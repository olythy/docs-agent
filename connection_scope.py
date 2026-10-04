"""Shared-or-short-lived database connection handling, used by composition.

A store can be handed an existing connection (to reuse it across several
operations), can open one for the duration of a ``with`` block, or -- used
bare -- can open a short-lived connection per call. That logic is the same for
every store, so it lives here once instead of being copied into each of them.

Key exports:
    ConnectionScope -- Owns that logic; a store holds one and delegates to it.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from db import get_connection


class ConnectionScope:
    """Decides which connection a store's operations run on.

    Args:
        conn: Optional active connection. If given, the caller owns it and
            this class never closes it.
        connect: Factory used to open a connection when none was given.
            Defaults to :func:`db.get_connection`; injectable so a store (or a
            test) can substitute its own.
    """

    def __init__(
        self, conn: Any = None, connect: Callable[[], Any] = get_connection
    ) -> None:
        self._conn = conn
        self._connect = connect
        self._managed_conn: Any = None
        self._depth = 0

    def enter(self) -> None:
        """Enter a ``with`` scope, opening one reusable connection if none exists."""
        if self._conn is None:
            self._managed_conn = self._connect()
            self._conn = self._managed_conn
        self._depth += 1

    def exit(self) -> None:
        """Leave a ``with`` scope; the outermost exit closes a connection this class opened."""
        self._depth -= 1
        if self._depth <= 0:
            self._depth = 0
            if self._managed_conn is not None:
                try:
                    self._managed_conn.close()
                finally:
                    self._conn = None
                    self._managed_conn = None

    @contextmanager
    def connection(self) -> Iterator[Any]:
        """Yield the active connection, or a fresh one that is closed afterwards."""
        if self._conn is not None:
            yield self._conn
        else:
            conn = self._connect()
            try:
                yield conn
            finally:
                conn.close()
