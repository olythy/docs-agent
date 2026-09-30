"""Development environment and infrastructure CLI.

Manages local container infrastructure, one-shot project setup,
code quality checks (linting/formatting), and environment health checks (doctor).

Usage:
    uv run python scripts/dev_cli.py [command]

Commands:
    docker-up       Start local Postgres container (dev + test) and wait until healthy.
    docker-down     Stop the local Postgres container (keeps data volume).
    docker-clean    Stop the container AND delete its data volume (full reset).
    setup           One-shot onboarding: start container + migrate dev & test databases.
    doctor          Verify environment, Docker container status, and database connections.
    lint            Run ruff linter check.
    lint-fix        Auto-fix lint errors and format code (alias: fix).
    format          Format code with ruff format.
"""

import os
import shutil
import sys
from pathlib import Path

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import run_cmd


def cmd_docker_up() -> int:
    """Start local Postgres container and wait until healthy."""
    if not shutil.which("docker"):
        print("ERROR: 'docker' command not found. Please install Docker.")
        return 1

    print("Starting local Postgres (dev + test) container...")
    code = run_cmd(["docker", "compose", "up", "-d", "--wait"])
    if code != 0:
        print(
            "ERROR: Failed to start Docker container. Ensure Docker daemon is running."
        )
        return code
    print("Postgres container is up and healthy.")
    return 0


def cmd_docker_down() -> int:
    """Stop local Postgres container, keeping its data volume."""
    print("Stopping Postgres container (keeping volume)...")
    return run_cmd(["docker", "compose", "down"])


def cmd_docker_clean() -> int:
    """Stop local Postgres container and delete its data volume (full reset)."""
    print("Stopping Postgres container and deleting data volume...")
    return run_cmd(["docker", "compose", "down", "-v"])


def cmd_setup() -> int:
    """One-shot onboarding: start Docker and migrate both dev and test databases."""
    print("=== Step 1/3: Starting Postgres container ===")
    code = cmd_docker_up()
    if code != 0:
        return code

    db_cli_path = PROJECT_ROOT / "scripts" / "db_cli.py"

    print("\n=== Step 2/3: Migrating development database ===")
    dev_code = run_cmd([sys.executable, str(db_cli_path), "up"])
    if dev_code != 0:
        print("ERROR: Failed to migrate development database.")
        return dev_code

    print("\n=== Step 3/3: Migrating test database ===")
    test_env = {**os.environ, "AGENT_ENV": "test"}
    test_code = run_cmd([sys.executable, str(db_cli_path), "up"], env=test_env)
    if test_code != 0:
        print("ERROR: Failed to migrate test database.")
        return test_code

    print("\nReady — dev and test databases are both migrated.")
    return 0


def cmd_doctor() -> int:
    """Check configuration files, Docker status, and database connectivity."""
    print("Running system doctor check...\n")
    all_ok = True

    # 1. Check environment files
    env_file = PROJECT_ROOT / ".env"
    env_test_file = PROJECT_ROOT / ".env.test"

    if env_file.exists():
        print("  [✓] .env file exists")
    else:
        print("  [✗] .env file is missing (copy from .env.example)")
        all_ok = False

    if env_test_file.exists():
        print("  [✓] .env.test file exists")
    else:
        print(
            "  [!] .env.test file is missing (needed for test suite, copy from .env.test.example)"
        )

    # 2. Check Docker CLI & Daemon
    if not shutil.which("docker"):
        print("  [✗] 'docker' binary not found on PATH")
        all_ok = False
    else:
        docker_ping_code = run_cmd(["docker", "info"], silent=True)
        if docker_ping_code == 0:
            print("  [✓] Docker daemon is running")
        else:
            print("  [✗] Docker daemon is not responding (is Docker Desktop running?)")
            all_ok = False

    # 3. Check DB connectivity (dev)
    try:
        from db import get_connection

        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            ext = cur.fetchone()
        conn.close()
        if ext:
            print("  [✓] Dev database connection OK (pgvector extension enabled)")
        else:
            print("  [!] Dev database connection OK, but pgvector extension is missing")
    except (RuntimeError, OSError) as e:
        print(f"  [✗] Dev database connection failed: {e}")
        all_ok = False

    print(
        "\n"
        + (
            "All critical checks passed!"
            if all_ok
            else "Some checks failed. See details above."
        )
    )
    return 0 if all_ok else 1


def cmd_lint() -> int:
    """Run ruff check, then pyright."""
    print("Running ruff check...")
    ruff_code = run_cmd([sys.executable, "-m", "ruff", "check", "."])
    print("Running pyright...")
    pyright_code = run_cmd([sys.executable, "-m", "pyright"])
    return ruff_code if ruff_code != 0 else pyright_code


def cmd_typecheck() -> int:
    """Run pyright only."""
    print("Running pyright...")
    return run_cmd([sys.executable, "-m", "pyright"])


def cmd_format() -> int:
    """Format code with ruff format."""
    print("Running ruff format...")
    return run_cmd([sys.executable, "-m", "ruff", "format", "."])


def cmd_lint_fix() -> int:
    """Auto-fix lint errors and reformat code."""
    print("Running ruff check --fix...")
    fix_code = run_cmd([sys.executable, "-m", "ruff", "check", "--fix", "."])
    print("Running ruff format...")
    fmt_code = run_cmd([sys.executable, "-m", "ruff", "format", "."])
    return fix_code if fix_code != 0 else fmt_code


COMMANDS = {
    "docker-up": cmd_docker_up,
    "up": cmd_docker_up,
    "docker-down": cmd_docker_down,
    "down": cmd_docker_down,
    "docker-clean": cmd_docker_clean,
    "clean": cmd_docker_clean,
    "setup": cmd_setup,
    "doctor": cmd_doctor,
    "lint": cmd_lint,
    "lint-fix": cmd_lint_fix,
    "fix": cmd_lint_fix,
    "format": cmd_format,
    "typecheck": cmd_typecheck,
}


def print_help() -> None:
    """Print command usage and descriptions."""
    print((__doc__ or "").strip())


def main() -> None:
    """CLI entry point for development environment commands."""
    if len(sys.argv) > 1 and sys.argv[1] in {"--help", "-h", "help"}:
        print_help()
        sys.exit(0)

    command = sys.argv[1] if len(sys.argv) > 1 else "doctor"
    if command not in COMMANDS:
        print(f"Unknown command: '{command}'")
        print(f"Available commands: {', '.join(sorted(COMMANDS.keys()))}")
        sys.exit(1)

    exit_code = COMMANDS[command]()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
