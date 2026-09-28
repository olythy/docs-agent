"""Shared utilities for CLI scripts (formatting, subprocess runner, path helpers).

Consolidates common CLI infrastructure:
- Safe project root bootstrapping
- Clean subprocess execution with inherited output and standardized error reporting
- Consistent terminal text and paragraph wrapping/truncation
- Target document path resolution from CLI args or .env configuration
"""

import subprocess
import sys
import textwrap
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WIDTH = 78


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    silent: bool = False,
) -> int:
    """Execute a system command in the project root with inherited stdout/stderr.

    Args:
        cmd: Command and arguments list.
        cwd: Working directory (defaults to PROJECT_ROOT).
        env: Optional environment variables dictionary.
        silent: If True, suppress stdout and stderr.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    stdout = subprocess.DEVNULL if silent else None
    stderr = subprocess.DEVNULL if silent else None
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, env=env, stdout=stdout, stderr=stderr, check=False
        )
        return proc.returncode
    except FileNotFoundError:
        if not silent:
            print(f"ERROR: Command not found: {cmd[0]}")
        return 127
    except (OSError, subprocess.SubprocessError) as e:
        if not silent:
            print(f"ERROR executing {' '.join(cmd)}: {e}")
        return 1


def truncate(text: str, max_len: int = DEFAULT_WIDTH) -> str:
    """Shorten ``text`` to at most ``max_len`` characters, on a word boundary."""
    return textwrap.shorten(text, width=max_len, placeholder="...")


def wrap(text: str, width: int = DEFAULT_WIDTH) -> str:
    """Wrap ``text`` across multiple lines of at most ``width`` characters."""
    return textwrap.fill(text, width=width)


def format_paragraphs(text: str, indent: str = "  ", width: int = 76) -> str:
    """Format and word-wrap multi-paragraph text with consistent indentation."""
    paragraphs = text.split("\n")
    formatted: list[str] = []
    wrap_width = max(width - len(indent), 20)
    for p in paragraphs:
        stripped = p.strip()
        if not stripped:
            formatted.append("")
        else:
            lines = textwrap.wrap(stripped, width=wrap_width)
            formatted.append("\n".join(f"{indent}{line}" for line in lines))
    return "\n".join(formatted)


def resolve_doc_path(path_arg: str | None = None) -> Path:
    """Resolve target document from CLI argument or TEST_DOC_PATH in .env.

    Raises:
        SystemExit: If no path was provided or the target file does not exist.
    """
    from config import settings

    if path_arg:
        p = Path(path_arg)
    elif settings.TEST_DOC_PATH:
        p = Path(settings.TEST_DOC_PATH)
    else:
        print("ERROR: No document path provided.")
        print("  Provide a file path argument or set TEST_DOC_PATH in your .env file.")
        sys.exit(1)

    if not p.exists():
        print(f"ERROR: File not found: {p}")
        sys.exit(1)

    return p
