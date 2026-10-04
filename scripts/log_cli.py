"""Structured logging and telemetry CLI for docs-agent.

Provides live monitoring, tailing, statistics, and log management for
the structured JSONL event log (logs/log.jsonl):
- Real-time formatted event watcher (replaces raw jsonl `tail -f`)
- Action-specific rich visualizations (FTS keyword filtering, cross-encoder reranking)
- High-level telemetry stats and stopword analysis
- Quick log file clearing

Usage:
    uv run python scripts/log_cli.py [command] [options]

Commands:
    watch [options]       Follow log file in real-time (default command).
                          Options:
                            --lines, -n <N>    Show last N events on startup (default: 10).
                            --all, -a          Show entire log before following.
                            --action <name>    Filter by action (e.g. fts_query_filtered).
                            --test             Watch logs/log-test.jsonl instead of log.jsonl.
                            --no-color         Disable ANSI terminal colors.
                            --checkpoint-every <N>
                                               Print a permanent running-total line every N
                                               matched events (default: 20, 0 disables). Between
                                               events, a self-overwriting heartbeat line shows
                                               elapsed time and counts so far -- useful during a
                                               long-running ingest, without needing to scroll back.
    tail [options]        Print recent formatted events and exit (same options as watch).
    stats [options]       Summarize telemetry events, top dropped stopwords, and rerank ratios.
                          Options:
                            --test             Analyze test telemetry log.
    clear [options]       Clear (truncate) the target log file.
                          Options:
                            --test             Clear test telemetry log.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from logger import LogAction
from scripts.utils import truncate

# ANSI Color Codes
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_DIM = "\033[2m"
COLOR_CYAN = "\033[36m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_MAGENTA = "\033[35m"
COLOR_BLUE = "\033[34m"
COLOR_RED = "\033[31m"


def resolve_log_path(is_test: bool = False, explicit_path: str | None = None) -> Path:
    """Resolve the target log file path based on flags and settings."""
    if explicit_path:
        return Path(explicit_path)
    if is_test:
        return PROJECT_ROOT / "logs" / "log-test.jsonl"
    return PROJECT_ROOT / Path(settings.LOG_FILE)


def format_timestamp(iso_str: str) -> str:
    """Convert an ISO-8601 timestamp string to a human-readable HH:MM:SS format."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%H:%M:%S")
    except (ValueError, TypeError):
        return iso_str[:8] if iso_str else "??:??:??"


def format_event(entry: dict[str, Any], color: bool = True) -> str:
    """Format a single JSON structured log entry into a human-readable string.

    Args:
        entry: Decoded JSON dictionary from the log file.
        color: Whether to include ANSI color escape codes.

    Returns:
        Formatted multi-line representation of the log event.
    """
    ts = format_timestamp(entry.get("timestamp", ""))
    action = str(entry.get("action", "unknown"))
    data = entry.get("data", {})

    def _c(col: str, txt: str) -> str:
        return f"{col}{txt}{COLOR_RESET}" if color else txt

    dim_ts = _c(COLOR_DIM, f"[{ts}]")

    if action == LogAction.FTS_QUERY_FILTERED:
        badge = _c(COLOR_CYAN + COLOR_BOLD, "[FTS FILTER]")
        query = data.get("original_query", "")
        kept = data.get("kept_terms", [])
        dropped = data.get("dropped_terms", [])

        kept_str = _c(COLOR_GREEN, f"{kept}") if kept else _c(COLOR_DIM, "[]")
        dropped_str = (
            _c(COLOR_YELLOW, f"{dropped}") if dropped else _c(COLOR_DIM, "(none)")
        )

        lines = [
            f"{dim_ts} {badge} Query: {_c(COLOR_BOLD, f'{query!r}')}",
            f"       Kept:    {kept_str}",
            f"       Dropped: {dropped_str}",
        ]
        return "\n".join(lines)

    if action == LogAction.RERANK_APPLIED:
        badge = _c(COLOR_MAGENTA + COLOR_BOLD, "[RERANK]")
        question = data.get("question", "")
        model = data.get("reranker_model", "")
        cand_count = data.get("candidates_count", 0)
        acc_count = data.get("accepted_count", 0)
        threshold = data.get("threshold", 0.0)
        top_score = data.get("top_score")

        pct = (acc_count / cand_count * 100) if cand_count else 0.0
        acc_col = COLOR_GREEN if acc_count > 0 else COLOR_YELLOW
        acc_str = _c(acc_col + COLOR_BOLD, f"{acc_count}/{cand_count} ({pct:.0f}%)")

        score_str = f"{top_score:.2f}" if isinstance(top_score, (int, float)) else "N/A"
        lines = [
            f"{dim_ts} {badge} Question: {_c(COLOR_BOLD, f'{truncate(question, 60)!r}')}",
            f"       Accepted: {acc_str} | Top Score: {_c(COLOR_BOLD, score_str)} (Threshold: {threshold})",
            f"       Model:    {_c(COLOR_DIM, model)}",
        ]
        return "\n".join(lines)

    if action == LogAction.DOCUMENT_INGESTED:
        badge = _c(COLOR_BLUE + COLOR_BOLD, "[INGEST]")
        doc = data.get("path", data.get("source", "unknown"))
        chunks = data.get("chunks_count", data.get("chunks", "?"))
        return (
            f"{dim_ts} {badge} Document: {_c(COLOR_BOLD, str(doc))} ({chunks} chunks)"
        )

    if action == LogAction.RELEVANCE_GATE_CHECKED:
        badge = _c(COLOR_YELLOW + COLOR_BOLD, "[GATE]")
        passed = data.get("passed", False)
        pass_str = (
            _c(COLOR_GREEN, "PASSED") if passed else _c(COLOR_RED, "FALLBACK (EMPTY)")
        )
        return f"{dim_ts} {badge} Status: {pass_str} | Data: {data}"

    if action == LogAction.ANSWER_GENERATED:
        badge = _c(COLOR_GREEN + COLOR_BOLD, "[ANSWER]")
        latency = data.get("latency_seconds", "?")
        return f"{dim_ts} {badge} Generated in {latency}s | Data: {data}"

    # Generic fallback for custom/future actions
    badge = _c(COLOR_YELLOW, f"[{action.upper()}]")
    data_str = json.dumps(data, ensure_ascii=False) if data else ""
    return f"{dim_ts} {badge} {data_str}"


def read_recent_lines(file_path: Path, max_lines: int) -> list[str]:
    """Read the last ``max_lines`` lines from a file without loading the entire file."""
    if not file_path.is_file() or max_lines <= 0:
        return []

    try:
        with file_path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return lines[-max_lines:]
    except OSError:
        return []


def follow_log_file(
    file_path: Path,
    initial_lines: int = 10,
    read_all: bool = False,
    poll_interval: float = 0.5,
) -> Generator[str | None, None, None]:
    """Yield lines from ``file_path``, streaming new lines as they are appended.

    Also yields ``None`` once per idle poll (no new line since the last
    check) -- callers that want a live "still watching, nothing new yet"
    heartbeat (see ``cmd_watch``) need this tick; callers that don't can
    just skip ``None`` values, so this stays backward compatible with
    simple "format every line" consumption.

    Args:
        file_path: Path to the log file.
        initial_lines: How many recent lines to yield before live-following.
        read_all: If True, read the entire file from start before following.
        poll_interval: Sleep interval in seconds between EOF checks.
    """
    # Wait for file to exist
    while not file_path.exists():
        time.sleep(poll_interval)

    with file_path.open("r", encoding="utf-8", errors="replace") as f:
        if read_all:
            pass  # start from beginning
        elif initial_lines > 0:
            all_lines = f.readlines()
            for line in all_lines[-initial_lines:]:
                yield line
        else:
            f.seek(0, os.SEEK_END)

        last_pos = f.tell()

        while True:
            line = f.readline()
            if line:
                yield line
                last_pos = f.tell()
            else:
                # Check for file truncation/rotation
                try:
                    current_size = file_path.stat().st_size
                    if current_size < last_pos:
                        f.seek(0, os.SEEK_SET)
                        last_pos = 0
                        continue
                except OSError:
                    pass

                time.sleep(poll_interval)
                yield None


def cmd_watch(argv: list[str]) -> int:
    """Follow and format structured log events in real-time."""
    parser = argparse.ArgumentParser(
        prog="log_cli.py watch",
        description="Follow structured log events in real time with formatted output.",
    )
    parser.add_argument(
        "--lines",
        "-n",
        type=int,
        default=10,
        help="Number of initial lines to show (default: 10, use 0 for new only).",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Read all previous logs before following.",
    )
    parser.add_argument(
        "--action",
        type=str,
        default=None,
        help="Filter events by action name (e.g. fts_query_filtered, rerank_applied).",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Watch logs/log-test.jsonl instead of default logs/log.jsonl.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Custom log file path to follow.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color codes.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Print a permanent running-total line every N matched events (default: 20, 0 disables).",
    )
    args = parser.parse_args(argv)

    log_path = resolve_log_path(is_test=args.test, explicit_path=args.path)
    use_color = not args.no_color and sys.stdout.isatty()

    rel_path = (
        log_path.relative_to(PROJECT_ROOT)
        if log_path.is_relative_to(PROJECT_ROOT)
        else log_path
    )
    header = f"=== Watching {rel_path} (Ctrl+C to stop) ==="
    if use_color:
        header = f"{COLOR_BOLD}{COLOR_CYAN}{header}{COLOR_RESET}"
    print(header)

    start_time = time.time()
    counts: collections.Counter[str] = collections.Counter()
    heartbeat_shown = False

    def _heartbeat_line() -> str:
        elapsed = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)
        total = sum(counts.values())
        breakdown = ", ".join(f"{act}={n}" for act, n in counts.most_common(3))
        suffix = f" ({breakdown})" if breakdown else ""
        text = f"⏳ watching... {mins:02d}:{secs:02d} elapsed | {total} event(s){suffix}"
        return f"{COLOR_DIM}{text}{COLOR_RESET}" if use_color else text

    try:
        for raw_line in follow_log_file(
            log_path,
            initial_lines=args.lines,
            read_all=args.all,
        ):
            if raw_line is None:
                # Idle tick: redraw the self-overwriting heartbeat line in
                # place, same "stays put while nothing's happening" feel
                # as a fixed header, without needing a curses screen.
                sys.stdout.write("\r\033[K" + _heartbeat_line())
                sys.stdout.flush()
                heartbeat_shown = True
                continue

            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                if heartbeat_shown:
                    sys.stdout.write("\r\033[K")
                    heartbeat_shown = False
                print(f"[raw] {line}")
                continue

            if args.action and entry.get("action") != args.action:
                continue

            if heartbeat_shown:
                sys.stdout.write("\r\033[K")
                heartbeat_shown = False

            action = str(entry.get("action", "unknown"))
            counts[action] += 1

            print(format_event(entry, color=use_color))
            print()

            if args.checkpoint_every > 0 and sum(counts.values()) % args.checkpoint_every == 0:
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                breakdown = ", ".join(f"{act}={n}" for act, n in counts.most_common())
                checkpoint = f"--- {sum(counts.values())} event(s) in {mins:02d}:{secs:02d} ({breakdown}) ---"
                if use_color:
                    checkpoint = f"{COLOR_BOLD}{COLOR_YELLOW}{checkpoint}{COLOR_RESET}"
                print(checkpoint)
                print()
    except KeyboardInterrupt:
        if heartbeat_shown:
            sys.stdout.write("\r\033[K")
        print("\nWatcher stopped.")
        return 0

    return 0


def cmd_tail(argv: list[str]) -> int:
    """Print the last N formatted log events and exit without following."""
    parser = argparse.ArgumentParser(
        prog="log_cli.py tail",
        description="Print recent formatted log events and exit.",
    )
    parser.add_argument(
        "--lines",
        "-n",
        type=int,
        default=20,
        help="Number of lines to display (default: 20).",
    )
    parser.add_argument(
        "--action",
        type=str,
        default=None,
        help="Filter events by action name.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Tail logs/log-test.jsonl.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Custom log file path.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color codes.",
    )
    args = parser.parse_args(argv)

    log_path = resolve_log_path(is_test=args.test, explicit_path=args.path)
    use_color = not args.no_color and sys.stdout.isatty()

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return 0

    lines = read_recent_lines(log_path, args.lines)
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            print(f"[raw] {line}")
            continue

        if args.action and entry.get("action") != args.action:
            continue

        print(format_event(entry, color=use_color))
        print()

    return 0


def cmd_stats(argv: list[str]) -> int:
    """Compute aggregate statistics from structured log records."""
    parser = argparse.ArgumentParser(
        prog="log_cli.py stats",
        description="Compute summary statistics from structured event logs.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Analyze logs/log-test.jsonl instead of log.jsonl.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Custom log file path.",
    )
    args = parser.parse_args(argv)

    log_path = resolve_log_path(is_test=args.test, explicit_path=args.path)
    if not log_path.is_file():
        print(f"Log file does not exist: {log_path}")
        return 0

    total_records = 0
    actions_counter: collections.Counter[str] = collections.Counter()
    dropped_stopwords: collections.Counter[str] = collections.Counter()
    fts_queries = 0
    rerank_total = 0
    rerank_candidates = 0
    rerank_accepted = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_records += 1
            action = str(entry.get("action", "unknown"))
            actions_counter[action] += 1
            data = entry.get("data", {})

            if action == LogAction.FTS_QUERY_FILTERED:
                fts_queries += 1
                for term in data.get("dropped_terms", []):
                    dropped_stopwords[term] += 1

            elif action == LogAction.RERANK_APPLIED:
                rerank_total += 1
                rerank_candidates += int(data.get("candidates_count", 0))
                rerank_accepted += int(data.get("accepted_count", 0))

    rel_path = (
        log_path.relative_to(PROJECT_ROOT)
        if log_path.is_relative_to(PROJECT_ROOT)
        else log_path
    )
    print("=" * 60)
    print(f"TELEMETRY LOG SUMMARY: {rel_path}")
    print("=" * 60)
    print(f"Total structured entries: {total_records}")
    print("\nBreakdown by action:")
    for act, count in actions_counter.most_common():
        pct = (count / total_records * 100) if total_records else 0.0
        print(f"  - {act:<25} {count:>6}  ({pct:>5.1f}%)")

    if fts_queries > 0:
        print(f"\nFTS Query Filtering ({fts_queries} queries):")
        print(f"  Total dropped stopword instances: {sum(dropped_stopwords.values())}")
        if dropped_stopwords:
            top_10 = dropped_stopwords.most_common(10)
            print("  Top 10 dropped stopwords:")
            for word, freq in top_10:
                print(f"    - {word!r:<15} {freq:>4} times")

    if rerank_total > 0:
        acc_pct = (
            (rerank_accepted / rerank_candidates * 100) if rerank_candidates else 0.0
        )
        print(f"\nCross-Encoder Reranking ({rerank_total} invocations):")
        print(f"  Total candidates evaluated: {rerank_candidates}")
        print(f"  Total accepted by threshold: {rerank_accepted} ({acc_pct:.1f}%)")

    print("=" * 60)
    return 0


def cmd_clear(argv: list[str]) -> int:
    """Clear (truncate) the target log file."""
    parser = argparse.ArgumentParser(
        prog="log_cli.py clear",
        description="Clear (truncate) the structured event log.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Clear logs/log-test.jsonl instead of log.jsonl.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help="Custom log file path to clear.",
    )
    args = parser.parse_args(argv)

    log_path = resolve_log_path(is_test=args.test, explicit_path=args.path)
    if not log_path.exists():
        print(f"Log file does not exist: {log_path} (nothing to clear)")
        return 0

    try:
        with log_path.open("w", encoding="utf-8") as f:
            f.truncate(0)
        print(f"Log file cleared: {log_path}")
        return 0
    except OSError as e:
        print(f"ERROR clearing log file: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    """CLI dispatcher for logging tools."""
    args = argv if argv is not None else sys.argv[1:]

    # Default to "watch" if no command provided
    if not args or args[0] in {"-h", "--help"}:
        if not args:
            return cmd_watch([])
        print(__doc__)
        return 0

    cmd = args[0]
    sub_args = args[1:]

    if cmd in {"watch", "follow"}:
        return cmd_watch(sub_args)
    if cmd in {"tail", "recent"}:
        return cmd_tail(sub_args)
    if cmd in {"stats", "summary"}:
        return cmd_stats(sub_args)
    if cmd in {"clear", "flush"}:
        return cmd_clear(sub_args)

    # If first argument starts with a flag (e.g. `log_cli.py --test`), treat as `watch`
    if cmd.startswith("-"):
        return cmd_watch(args)

    print(f"ERROR: Unknown command '{cmd}'. See 'log_cli.py --help'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
