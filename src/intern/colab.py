"""Execute commands on an existing Colab runtime with explicit remote exit status."""

from __future__ import annotations

import math
import shlex
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from textwrap import dedent
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, StrictInt, ValidationError


class _RemoteStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_id: str
    returncode: StrictInt


def execute_remote(
    experiment_dir: Path,
    command: str,
    *,
    session: str,
    state_file: Path,
    remote_dir: str = "/content/my-little-ml-intern",
    timeout_s: float = 10800,
    log_name: str | None = None,
) -> int:
    """Return the remote command's exit code, or 2 when the transport cannot prove its outcome."""
    try:
        argv = shlex.split(command)
        if not argv or any("\0" in arg for arg in argv):
            raise ValueError("The command must contain an executable")
        if not session.strip() or session.startswith("-") or any(char in session for char in "\0\r\n"):
            raise ValueError("Set the Colab session name")
        remote_path = PurePosixPath(remote_dir)
        if (
            not remote_path.is_absolute()
            or remote_path == PurePosixPath("/")
            or ".." in remote_path.parts
            or any(char in remote_dir for char in "\0\r\n")
        ):
            raise ValueError("Use an absolute remote directory below the filesystem root, without '..'")
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("The timeout must be finite and positive")
        if log_name is not None and (
            not log_name
            or PurePosixPath(log_name).name != log_name
            or log_name in (".", "..", "stderr.log")
            or any(char in log_name for char in "\0\r\n")
        ):
            raise ValueError("Use a log filename other than stderr.log")
    except ValueError as err:
        logger.error("{}", err)
        return 2

    execution_id = uuid4().hex
    remote_status = f"/content/intern-status-{execution_id}.json"
    remote_log = (
        str(PurePosixPath(remote_dir) / "experiments" / experiment_dir.name / "logs" / log_name) if log_name else None
    )
    code = dedent(f"""\
        def _intern_execute():
            import json
            import os
            import subprocess
            import sys
            from pathlib import Path

            status = {{"execution_id": {execution_id!r}, "returncode": 2}}
            log = None
            log_path = {remote_log!r}
            try:
                if log_path is not None:
                    path = Path(log_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    log = path.open("w", encoding="utf-8")
                    print("Remote log: " + str(path), flush=True)
                result = subprocess.run(
                    {argv!r}, cwd={remote_dir!r},
                    env={{**os.environ, "PROJECT_ROOT": {remote_dir!r}, "PYTHONUNBUFFERED": "1"}},
                    stdout=log, stderr=subprocess.STDOUT if log else None,
                    timeout={timeout_s!r}, check=False,
                )
                status["returncode"] = result.returncode if result.returncode >= 0 else 128 - result.returncode
            except subprocess.TimeoutExpired:
                status["returncode"] = 124
                print("The remote command exceeded its timeout", file=sys.stderr)
            except OSError as err:
                print(str(err), file=sys.stderr)
            finally:
                if log is not None:
                    log.close()
                Path({remote_status!r}).write_text(json.dumps(status), encoding="utf-8")

        _intern_execute()
        del _intern_execute
        """)
    cli = ["colab", "--auth", "adc", "--config", str(state_file.resolve())]
    try:
        subprocess.run(
            [*cli, "exec", "-s", session, "--timeout", str(timeout_s + 60)],
            input=code,
            text=True,
            timeout=timeout_s + 120,
            check=True,
        )
        with tempfile.TemporaryDirectory(prefix="intern-colab-") as directory:
            local_status = Path(directory) / "status.json"
            subprocess.run(
                [*cli, "download", "-s", session, remote_status, str(local_status)],
                timeout=120,
                check=True,
            )
            status = _RemoteStatus.model_validate_json(local_status.read_text(encoding="utf-8"))
            if status.execution_id != execution_id or not 0 <= status.returncode <= 255:
                raise ValueError("The remote status does not match this execution")
    except (OSError, subprocess.SubprocessError, ValidationError, ValueError) as err:
        logger.error(
            "Colab execution has no confirmed outcome ({}). Inspect the runtime before a retry", type(err).__name__
        )
        return 2
    logger.info("Colab command exit {}", status.returncode)
    return status.returncode
