"""Fresh-interpreter checks for durable mode: settings sources and an unchanged source tree."""

import contextlib
import os
import socket
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.pipeline_sources import CALC_YAML, ORDINARY_WRAPPER, write_tree


def _run(script: str, *, cwd: Path, env: dict[str, str] | None = None) -> str:
    # Nothing may redirect or disable bytecode writing on the interpreter's behalf.
    base = {k: v for k, v in os.environ.items() if not k.startswith(("HAYHOOKS_", "PYTHONDONTWRITE", "PYTHONPYCACHE"))}
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=cwd,
        env={**base, "PYTHONPATH": str(Path(__file__).parents[1]), **(env or {})},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _snapshot(*trees: Path) -> dict[Path, bytes | None]:
    """Map every path, directories included, to its bytes."""
    return {path: None if path.is_dir() else path.read_bytes() for tree in trees for path in tree.rglob("*")}


@pytest.mark.parametrize("source", ["environment", "dotenv"])
@pytest.mark.parametrize("enabled", ["true", "false"])
def test_setting_selects_the_mode_of_python_and_cli_apps(tmp_path: Path, source: str, enabled: str) -> None:
    pipelines_dir = write_tree(tmp_path / "pipelines", {"calc.yml": CALC_YAML})
    variables = {"HAYHOOKS_DURABLE_MODE": enabled, "HAYHOOKS_PIPELINES_DIR": str(pipelines_dir)}
    if source == "dotenv":
        (tmp_path / ".env").write_text("".join(f"{key}={value}\n" for key, value in variables.items()))
    script = """
        from hayhooks.cli.base import get_app
        from hayhooks.server.app import create_app

        for app in (create_app(), get_app()):
            print(app.state.durable_mode, "/deploy_files" in app.openapi()["paths"])
    """

    output = _run(script, cwd=tmp_path, env=variables if source == "environment" else None)

    durable = enabled == "true"
    assert output.splitlines() == [f"{durable} {not durable}"] * 2


def test_durable_mode_leaves_the_source_tree_unchanged(tmp_path: Path) -> None:
    good = write_tree(
        tmp_path / "good",
        {
            "calc.yml": CALC_YAML,
            "double/__init__.py": "from .helpers import FACTOR\n",
            "double/helpers.py": "FACTOR = 2\n",
            "double/deferred.py": "def double(value):\n    return value * 2\n",
            "double/pipeline_wrapper.py": ORDINARY_WRAPPER.replace(
                "        return value * 2", "        from .deferred import double\n\n        return double(value)"
            ),
        },
    )
    broken = write_tree(
        tmp_path / "broken",
        {"ok/pipeline_wrapper.py": ORDINARY_WRAPPER, "zz/pipeline_wrapper.py": "import missing_module\n"},
    )
    trees = _snapshot(good, broken)
    script = f"""
        import sys

        from fastapi.testclient import TestClient

        from hayhooks.server.app import create_app
        from hayhooks.settings import settings

        assert not sys.dont_write_bytecode
        settings.durable_mode = True
        settings.pipelines_dir = {str(broken)!r}
        try:
            create_app()
        except Exception:
            pass
        else:
            raise AssertionError("the broken tree must fail startup")

        settings.pipelines_dir = {str(good)!r}
        with TestClient(create_app()) as client:
            assert client.post("/double/run", json={{"value": 21}}).json() == {{"result": 42}}
        print("ok")
    """

    assert _run(script, cwd=tmp_path) == "ok"

    # Paths and bytes, not access times: no bytecode, backups, or rewritten sources.
    assert _snapshot(good, broken) == trees


CLI = [sys.executable, "-c", "from hayhooks.cli import hayhooks_cli; hayhooks_cli()"]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def servers(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[bool, tuple[int, Path]]]:
    """One mutable and one durable-mode server, each logging its requests to a file."""
    started = {}
    try:
        for durable in (False, True):
            root = tmp_path_factory.mktemp(f"server-{durable}")
            port = _free_port()
            log = root / "server.log"
            env = {
                **os.environ,
                "HAYHOOKS_DURABLE_MODE": str(durable).lower(),
                "HAYHOOKS_PIPELINES_DIR": str(write_tree(root / "pipelines", {"calc.yml": CALC_YAML})),
            }
            with log.open("w") as output:
                process = subprocess.Popen(  # noqa: S603
                    [*CLI, "run", "--host", "127.0.0.1", "--port", str(port)], env=env, stdout=output, stderr=output
                )
            started[durable] = (process, port, log)
        for durable, (_process, port, log) in started.items():
            for _ in range(200):
                with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
                time.sleep(0.05)
            else:
                pytest.fail(f"server (durable={durable}) did not start:\n{log.read_text()}")
        yield {durable: (port, log) for durable, (_process, port, log) in started.items()}
    finally:
        for process, _port, _log in started.values():
            process.terminate()
            process.wait(10)


def _cli(*args: str, durable: bool, port: int = 1416) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HAYHOOKS_DURABLE_MODE": str(durable).lower(),
        "HAYHOOKS_HOST": "127.0.0.1",
        "HAYHOOKS_PORT": str(port),
        "COLUMNS": "200",
    }
    return subprocess.run([*CLI, *args], env=env, capture_output=True, text=True, timeout=60, check=False)  # noqa: S603


@pytest.mark.integration
@pytest.mark.parametrize("local_durable", [False, True], ids=["local-default", "local-durable"])
def test_remote_cli_commands_are_governed_by_the_target_server(
    servers: dict[bool, tuple[int, Path]], tmp_path: Path, local_durable: bool
) -> None:
    help_output = {durable: _cli("pipeline", "--help", durable=durable).stdout for durable in (False, True)}
    assert help_output[local_durable] == help_output[not local_durable]
    assert all(command in help_output[local_durable] for command in ("deploy-files", "deploy-yaml", "undeploy"))
    pipeline_dir = write_tree(tmp_path / f"doubled_{local_durable}", {"pipeline_wrapper.py": ORDINARY_WRAPPER})

    for target_durable, (port, log) in servers.items():
        requests_before = log.read_text().count('HTTP/1.1"')
        deployed = _cli(
            "pipeline",
            "deploy-files",
            "-n",
            pipeline_dir.name,
            str(pipeline_dir),
            "-s",
            durable=local_durable,
            port=port,
        )
        if target_durable:
            assert deployed.returncode != 0
            assert "Not Found" in deployed.stdout
        else:
            assert deployed.returncode == 0, deployed.stdout + deployed.stderr
        # One deployment request, with no mode preflight; the access log may lag the response.
        for _ in range(100):
            if (requests := log.read_text().count('HTTP/1.1"') - requests_before) >= 1:
                break
            time.sleep(0.02)
        time.sleep(0.2)
        assert log.read_text().count('HTTP/1.1"') - requests_before == requests == 1
