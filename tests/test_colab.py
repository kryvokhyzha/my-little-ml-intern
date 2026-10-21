"""Exercise the Colab transport with real subprocesses in a local remote directory."""

import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import rootutils
from hydra import compose, initialize_config_dir

from intern.colab import execute_remote


ROOT = rootutils.find_root(__file__, indicator=".project-root")
REAL_RUN = subprocess.run


class FakeColab:
    def __init__(self, status_dir):
        self.status_dir = status_dir
        self.calls = []
        self.executions = []
        self.status_payload = None
        self.exec_failure = None
        self.download_failure = False
        self.skip_download = False

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if "exec" in argv:
            if self.exec_failure is not None:
                if isinstance(self.exec_failure, Exception):
                    raise self.exec_failure
                return self._complete(argv, self.exec_failure, kwargs)
            code = kwargs["input"].replace("/content/intern-status-", str(self.status_dir / "status-"))
            execution = REAL_RUN(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.executions.append(execution)
            return self._complete(argv, 0, kwargs)
        assert "download" in argv
        if self.download_failure:
            return self._complete(argv, 1, kwargs)
        if not self.skip_download:
            source, target = self.status_dir / Path(argv[-2]).name.replace("intern-status-", "status-"), Path(argv[-1])
            content = source.read_text()
            if self.status_payload is not None:
                content = self.status_payload(content)
            target.write_text(content)
        return self._complete(argv, 0, kwargs)

    @staticmethod
    def _complete(argv, returncode, kwargs):
        if returncode and kwargs.get("check"):
            raise subprocess.CalledProcessError(returncode, argv)
        return subprocess.CompletedProcess(argv, returncode)


@pytest.fixture
def remote(tmp_path, monkeypatch):
    experiment = tmp_path / "local" / "experiments" / "001-demo"
    experiment.mkdir(parents=True)
    remote_dir = tmp_path / "remote project"
    remote_dir.mkdir()
    transport = FakeColab(tmp_path)
    monkeypatch.setattr(subprocess, "run", transport)
    return {
        "experiment_dir": experiment,
        "session": "intern-001",
        "state_file": tmp_path / "sessions.json",
        "remote_dir": str(remote_dir),
        "timeout_s": 3,
    }, transport


def _python(code, *args):
    return shlex.join([sys.executable, "-c", code, *args])


@pytest.mark.parametrize("returncode", [0, 1, 7])
def test_remote_returncode_overrides_successful_colab_exec(remote, returncode):
    options, transport = remote

    result = execute_remote(command=_python(f"raise SystemExit({returncode})"), **options)

    assert result == returncode
    assert len(transport.calls) == 2
    assert "exec" in transport.calls[0][0]
    assert "download" in transport.calls[1][0]


def test_transport_uses_explicit_auth_state_session_and_timeouts(remote, tmp_path, monkeypatch):
    options, transport = remote
    monkeypatch.chdir(tmp_path)
    options["state_file"] = Path("relative-sessions.json")

    assert execute_remote(command=_python("pass"), **options) == 0

    for argv, kwargs in transport.calls:
        assert argv[:5] == ["colab", "--auth", "adc", "--config", str(tmp_path / "relative-sessions.json")]
        assert argv[argv.index("-s") + 1] == options["session"]
        assert not kwargs.get("shell", False)
        assert not kwargs.get("capture_output", False)
        assert kwargs.get("stdout") is None
        assert kwargs.get("stderr") is None
    argv, kwargs = transport.calls[0]
    assert float(argv[argv.index("--timeout") + 1]) == options["timeout_s"] + 60
    assert kwargs["timeout"] == options["timeout_s"] + 120
    assert kwargs["text"] is True


def test_remote_command_preserves_quoted_arguments_and_environment(remote, monkeypatch):
    options, _ = remote
    monkeypatch.setenv("COLAB_TEST_INHERITED", "present")
    remote_dir = Path(options["remote_dir"])
    probe = (
        "import json, os, sys; from pathlib import Path; "
        "Path('observed.json').write_text(json.dumps({"
        "'argv': sys.argv[1:], 'cwd': os.getcwd(), "
        "'root': os.environ['PROJECT_ROOT'], "
        "'unbuffered': os.environ['PYTHONUNBUFFERED'], "
        "'inherited': os.environ['COLAB_TEST_INHERITED']}))"
    )
    arguments = ["two words", "$(touch injected)", "a; touch injected", "quote'and\"double"]

    assert execute_remote(command=_python(probe, *arguments), **options) == 0

    observed = json.loads((remote_dir / "observed.json").read_text())
    assert observed == {
        "argv": arguments,
        "cwd": str(remote_dir),
        "root": str(remote_dir),
        "unbuffered": "1",
        "inherited": "present",
    }
    assert not (remote_dir / "injected").exists()


def test_missing_remote_executable_returns_usage_error(remote):
    options, _ = remote

    assert execute_remote(command="/definitely-missing-colab-executable", **options) == 2


def test_remote_timeout_returns_124(remote):
    options, _ = remote
    options["timeout_s"] = 1

    assert execute_remote(command=_python("import time; time.sleep(30)"), **options) == 124


def test_remote_signal_returns_shell_exit_code(remote):
    options, _ = remote
    command = _python("import os, signal; os.kill(os.getpid(), signal.SIGTERM)")

    assert execute_remote(command=command, **options) == 143


def test_remote_log_preserves_stdout_stderr_and_adapter_stderr(remote):
    options, _ = remote
    logs = Path(options["remote_dir"]) / "experiments" / options["experiment_dir"].name / "logs"
    logs.mkdir(parents=True)
    (logs / "stderr.log").write_text("adapter owns this file\n")
    (logs / "train.log").write_text("old output\n")
    command = _python("import sys; print('output'); print('diagnostic', file=sys.stderr); raise SystemExit(5)")

    assert execute_remote(command=command, log_name="train.log", **options) == 5

    log = (logs / "train.log").read_text()
    assert "output" in log
    assert "diagnostic" in log
    assert "old output" not in log
    assert (logs / "stderr.log").read_text() == "adapter owns this file\n"


def test_remote_output_inherits_local_streams(remote):
    options, transport = remote
    command = _python("import sys; print('output'); print('diagnostic', file=sys.stderr)")

    assert execute_remote(command=command, **options) == 0

    assert "output" in transport.executions[0].stdout
    assert "diagnostic" in transport.executions[0].stderr


def test_remote_log_wrapper_emits_no_syntax_warning(remote):
    options, transport = remote

    assert execute_remote(command=_python("print('output')"), log_name="train.log", **options) == 0

    assert "SyntaxWarning" not in transport.executions[0].stderr


@pytest.mark.parametrize(
    "payload",
    [
        lambda _: "{",
        lambda _: "{}",
        lambda _: "[]",
        lambda original: json.dumps(json.loads(original) | {"execution_id": "wrong-execution"}),
        lambda original: json.dumps(json.loads(original) | {"returncode": "0"}),
        lambda original: json.dumps(json.loads(original) | {"returncode": False}),
        lambda original: json.dumps(json.loads(original) | {"returncode": -1}),
        lambda original: json.dumps(json.loads(original) | {"returncode": 256}),
        lambda original: json.dumps(json.loads(original) | {"unexpected": "field"}),
    ],
    ids=[
        "malformed",
        "missing-fields",
        "wrong-shape",
        "stale",
        "string-code",
        "boolean-code",
        "negative-code",
        "large-code",
        "extra-field",
    ],
)
def test_invalid_status_refuses_success(remote, payload):
    options, transport = remote
    transport.status_payload = payload

    assert execute_remote(command=_python("pass"), **options) == 2


def test_missing_downloaded_status_refuses_success(remote):
    options, transport = remote
    transport.skip_download = True

    assert execute_remote(command=_python("pass"), **options) == 2


def test_download_failure_refuses_success(remote):
    options, transport = remote
    transport.download_failure = True

    assert execute_remote(command=_python("pass"), **options) == 2


@pytest.mark.parametrize(
    "failure",
    [17, FileNotFoundError("colab"), subprocess.TimeoutExpired("colab", 123)],
    ids=["nonzero-exit", "missing-cli", "timeout"],
)
def test_transport_failure_returns_usage_error(remote, failure):
    options, transport = remote
    transport.exec_failure = failure

    assert execute_remote(command=_python("pass"), **options) == 2
    assert len(transport.calls) == 1


def test_each_execution_uses_unique_status_and_removes_local_copy(remote):
    options, transport = remote

    assert execute_remote(command=_python("pass"), **options) == 0
    assert execute_remote(command=_python("pass"), **options) == 0

    downloads = [argv for argv, _ in transport.calls if "download" in argv]
    assert downloads[0][-2] != downloads[1][-2]
    for argv in downloads:
        status_path = transport.status_dir / Path(argv[-2]).name.replace("intern-status-", "status-")
        status = json.loads(status_path.read_text())
        assert set(status) == {"execution_id", "returncode"}
        assert isinstance(status["execution_id"], str)
        assert status["execution_id"]
        assert status["returncode"] == 0
        assert not Path(argv[-1]).exists()


@pytest.mark.parametrize("command", ["", "   ", "python 'unterminated", "python\x00bad"])
def test_invalid_command_refuses_transport(remote, command):
    options, transport = remote

    assert execute_remote(command=command, **options) == 2
    assert transport.calls == []


@pytest.mark.parametrize("session", ["", "   ", "-sneaky", "line\nbreak"])
def test_invalid_session_refuses_transport(remote, session):
    options, transport = remote
    options["session"] = session

    assert execute_remote(command=_python("pass"), **options) == 2
    assert transport.calls == []


@pytest.mark.parametrize("remote_dir", ["relative/repo", "/", "/content/../other", "/content/line\nbreak"])
def test_invalid_remote_directory_refuses_transport(remote, remote_dir):
    options, transport = remote
    options["remote_dir"] = remote_dir

    assert execute_remote(command=_python("pass"), **options) == 2
    assert transport.calls == []


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout_refuses_transport(remote, timeout):
    options, transport = remote
    options["timeout_s"] = timeout

    assert execute_remote(command=_python("pass"), **options) == 2
    assert transport.calls == []


@pytest.mark.parametrize("log_name", ["", "stderr.log", "../escape.log", "/tmp/escape.log", "nested/train.log"])
def test_invalid_log_name_refuses_transport(remote, log_name):
    options, transport = remote

    assert execute_remote(command=_python("pass"), log_name=log_name, **options) == 2
    assert transport.calls == []


def test_hydra_selects_colab_compute_lane():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(config_name="000-tiny-sft-smoke", overrides=["compute=colab"])

    assert cfg.compute.kind == "colab"
    assert cfg.compute.gpu == "T4"
    assert cfg.compute.high_mem is False
    assert cfg.compute.remote_dir == "/content/my-little-ml-intern"
    assert cfg.compute.timeout_s == 10800


def test_hydra_supports_colab_cpu_and_high_ram():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="000-tiny-sft-smoke",
            overrides=["compute=colab", "compute.gpu=null", "compute.high_mem=true"],
        )

    assert cfg.compute.gpu is None
    assert cfg.compute.high_mem is True
