"""Tests for connection_scope.ConnectionScope, with a fake connection factory."""

from unittest.mock import MagicMock

from connection_scope import ConnectionScope


def _factory():
    opened = []

    def connect():
        conn = MagicMock()
        opened.append(conn)
        return conn

    return connect, opened


def test_a_bare_call_opens_and_closes_its_own_connection():
    connect, opened = _factory()
    scope = ConnectionScope(connect=connect)

    with scope.connection() as conn:
        assert conn is opened[0]

    opened[0].close.assert_called_once()


def test_a_given_connection_is_used_and_never_closed():
    given = MagicMock()
    scope = ConnectionScope(given, connect=lambda: MagicMock())

    with scope.connection() as conn:
        assert conn is given
    scope.enter()
    scope.exit()

    given.close.assert_not_called()


def test_a_with_scope_reuses_one_connection_and_closes_it_at_the_outermost_exit():
    connect, opened = _factory()
    scope = ConnectionScope(connect=connect)

    scope.enter()
    scope.enter()  # nested
    with scope.connection() as first:
        pass
    with scope.connection() as second:
        pass
    scope.exit()
    opened[0].close.assert_not_called()  # inner exit: still open
    scope.exit()

    assert len(opened) == 1
    assert first is second is opened[0]
    opened[0].close.assert_called_once()


def test_after_the_scope_ends_the_next_call_opens_a_fresh_connection():
    connect, opened = _factory()
    scope = ConnectionScope(connect=connect)
    scope.enter()
    scope.exit()

    with scope.connection() as conn:
        pass

    assert len(opened) == 2
    assert conn is opened[1]
