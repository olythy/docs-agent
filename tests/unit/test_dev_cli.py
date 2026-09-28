"""Unit tests for scripts.dev_cli (pure logic, mocked subprocess)."""

from unittest.mock import MagicMock, patch

from scripts.dev_cli import (
    cmd_docker_clean,
    cmd_docker_down,
    cmd_docker_up,
    cmd_doctor,
    cmd_format,
    cmd_lint,
    cmd_lint_fix,
    cmd_setup,
)


@patch("scripts.dev_cli.shutil.which", return_value="/usr/local/bin/docker")
@patch("scripts.dev_cli.run_cmd", return_value=0)
def test_docker_up_success(mock_run, mock_which):
    assert cmd_docker_up() == 0
    mock_run.assert_called_once_with(["docker", "compose", "up", "-d", "--wait"])


@patch("scripts.dev_cli.shutil.which", return_value=None)
def test_docker_up_missing_binary(mock_which):
    assert cmd_docker_up() == 1


@patch("scripts.dev_cli.run_cmd", return_value=0)
def test_docker_down(mock_run):
    assert cmd_docker_down() == 0
    mock_run.assert_called_once_with(["docker", "compose", "down"])


@patch("scripts.dev_cli.run_cmd", return_value=0)
def test_docker_clean(mock_run):
    assert cmd_docker_clean() == 0
    mock_run.assert_called_once_with(["docker", "compose", "down", "-v"])


@patch("scripts.dev_cli.cmd_docker_up", return_value=0)
@patch("scripts.dev_cli.run_cmd", return_value=0)
def test_setup_success(mock_run, mock_up):
    assert cmd_setup() == 0
    assert mock_up.called
    assert mock_run.call_count == 2
    # Second call should have AGENT_ENV=test in env
    _, kwargs = mock_run.call_args_list[1]
    assert kwargs.get("env", {}).get("AGENT_ENV") == "test"


@patch("scripts.dev_cli.Path.exists", return_value=True)
@patch("scripts.dev_cli.shutil.which", return_value="/usr/local/bin/docker")
@patch("scripts.dev_cli.run_cmd", return_value=0)
@patch("db.get_connection")
def test_doctor_all_ok(mock_get_conn, mock_run, mock_which, mock_exists):
    fake_conn = MagicMock()
    mock_get_conn.return_value = fake_conn
    fake_conn.cursor.return_value.__enter__.return_value.fetchone.return_value = (
        "vector",
    )

    assert cmd_doctor() == 0


@patch("scripts.dev_cli.run_cmd", return_value=0)
def test_lint(mock_run):
    assert cmd_lint() == 0
    assert mock_run.called


@patch("scripts.dev_cli.run_cmd", return_value=0)
def test_format(mock_run):
    assert cmd_format() == 0
    assert mock_run.called


@patch("scripts.dev_cli.run_cmd", return_value=0)
def test_lint_fix(mock_run):
    assert cmd_lint_fix() == 0
    assert mock_run.call_count == 2
