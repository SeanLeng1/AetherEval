import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from aethereval.progress import Progress


_STARTUP_TIMEOUT_SECONDS = 1800
_REQUEST_TIMEOUT_SECONDS = 24 * 60 * 60
_PORT_START_ATTEMPTS = 3
_URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_HARMONY_ENCODING_ENV = "TIKTOKEN_ENCODINGS_BASE"
_HARMONY_ENCODING_FILENAME = "o200k_base.tiktoken"
_HARMONY_ENCODING_SHA256 = (
    "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
)
_BUNDLED_HARMONY_ENCODING_DIR = Path(__file__).with_name("encodings")
_UNSUPPORTED_SERVER_ARGS = {
    "grpc_http_sidecar_port",
    "log_level_http",
}
_CONTROLLED_SERVER_ARGS = {
    "grpc_mode",
    "host",
    "log_level",
    "model",
    "model_path",
    "nccl_port",
    "port",
    "router_log_level",
    "tensor_parallel_size",
    "tp_size",
}
_ROUTER_MODEL_ARGS = {
    "chat_template": "--chat-template",
    "reasoning_parser": "--reasoning-parser",
    "tool_call_parser": "--tool-call-parser",
}


def _resolve_harmony_encoding_dir() -> Path:
    configured_dir = os.environ.get(_HARMONY_ENCODING_ENV)
    encoding_dir = (
        Path(configured_dir).expanduser()
        if configured_dir
        else _BUNDLED_HARMONY_ENCODING_DIR
    )
    vocab_path = encoding_dir / _HARMONY_ENCODING_FILENAME
    if not vocab_path.is_file():
        source = (
            f"{_HARMONY_ENCODING_ENV}={configured_dir!r}"
            if configured_dir
            else "the bundled AetherEval assets"
        )
        raise RuntimeError(
            f"Missing Harmony encoding {vocab_path} from {source}. "
            "Reinstall AetherEval with package data included, or set "
            f"{_HARMONY_ENCODING_ENV} to a directory containing "
            f"{_HARMONY_ENCODING_FILENAME}."
        )

    digest = hashlib.sha256(vocab_path.read_bytes()).hexdigest()
    if digest != _HARMONY_ENCODING_SHA256:
        raise RuntimeError(
            f"Invalid Harmony encoding at {vocab_path}: SHA-256 is {digest}, "
            f"expected {_HARMONY_ENCODING_SHA256}."
        )
    return encoding_dir.resolve()


def _free_port(max_port: int = 65535) -> int:
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = int(sock.getsockname()[1])
        if port <= max_port:
            return port


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("", port))
        except OSError:
            return False
    return True


def _guarded_command(command: list[str]) -> list[str]:
    return [
        sys.executable,
        "-m",
        "aethereval.backends.sglang.process_guard",
        str(os.getpid()),
        *command,
    ]


def _server_cli_args(model_kwargs: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for raw_key, value in model_kwargs.items():
        key = str(raw_key)
        if key in _UNSUPPORTED_SERVER_ARGS:
            raise ValueError(
                f"SGLang argument {key!r} is not supported: gRPC workers "
                "do not expose an HTTP sidecar"
            )
        if key in _CONTROLLED_SERVER_ARGS:
            continue
        if value is None or value is False:
            continue
        if key == "tokenizer":
            key = "tokenizer_path"
        flag = "--" + key.replace("_", "-")
        if value is True:
            args.append(flag)
        elif isinstance(value, dict):
            args.extend((flag, json.dumps(value, separators=(",", ":"))))
        elif isinstance(value, (list, tuple)):
            args.append(flag)
            args.extend(str(item) for item in value)
        else:
            args.extend((flag, str(value)))
    return args


def _router_model_cli_args(
    model: str,
    model_kwargs: dict[str, Any],
) -> list[str]:
    args = ["--model-path", model]
    tokenizer_path = model_kwargs.get(
        "tokenizer_path",
        model_kwargs.get("tokenizer"),
    )
    if tokenizer_path is not None:
        args.extend(("--tokenizer-path", str(tokenizer_path)))
    for key, flag in _ROUTER_MODEL_ARGS.items():
        value = model_kwargs.get(key)
        if value is not None and value is not False:
            args.extend((flag, str(value)))
    return args


def _check_url(url: str, timeout: float = 5.0) -> None:
    with _URL_OPENER.open(url, timeout=timeout) as response:
        response.read()


def _post_json(url: str, payload: dict[str, Any]) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _URL_OPENER.open(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"SGLang request failed with HTTP {exc.code}: {error_body}"
        ) from exc


def _wait_until_ready(
    base_url: str,
    process: subprocess.Popen[Any] | None,
    endpoint: str = "/health",
    timeout: float = _STARTUP_TIMEOUT_SECONDS,
    tokenizer_model: str | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"SGLang process exited during startup with code {process.returncode}"
            )
        try:
            _check_url(f"{base_url}{endpoint}", timeout=2.0)
            # SMG marks workers healthy before asynchronous tokenizer registration finishes.
            if tokenizer_model is not None:
                with _URL_OPENER.open(f"{base_url}/v1/tokenizers", timeout=2.0) as response:
                    tokenizers = json.loads(response.read())["tokenizers"]
                if not any(item["name"] == tokenizer_model for item in tokenizers):
                    raise RuntimeError(f"Tokenizer for {tokenizer_model!r} is not ready")
            return
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(
        f"Timed out waiting for SGLang service at {base_url}: {last_error}"
    )


def _wait_for_port(
    host: str,
    port: int,
    process: subprocess.Popen[Any] | None,
    timeout: float = _STARTUP_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"SGLang process exited during startup with code {process.returncode}"
            )
        try:
            with socket.create_connection((host, port), timeout=2.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(
        f"Timed out waiting for SGLang gRPC worker at {host}:{port}: {last_error}"
    )


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process_group_id: int,
    timeout: float,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group_id):
            return True
        time.sleep(0.1)
    return not _process_group_exists(process_group_id)


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None:
        return

    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass

    if process.poll() is None:
        try:
            process.wait(timeout=20)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass

    if not _wait_for_process_group_exit(process_group_id, timeout=5):
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait(timeout=10)


class _SGLangServerActor:
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int,
        model_kwargs: dict[str, Any],
    ) -> None:
        import ray

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        warning_filters = env.get("PYTHONWARNINGS", "")
        quiet_filters = "ignore::SyntaxWarning,ignore::FutureWarning"
        env["PYTHONWARNINGS"] = ",".join(
            value for value in (warning_filters, quiet_filters) if value
        )
        env["TORCH_CPP_LOG_LEVEL"] = "ERROR"
        env["TQDM_DISABLE"] = "1"
        # smg-grpc-servicer defaults to the legacy list[int] scheduler
        # contract. SGLang 0.5.15 uses array("q") for generated token IDs, so
        # request IDs must use the same container type.
        env["SGLANG_GRPC_TOKEN_ID_ARRAY"] = "1"
        node_ip = ray.util.get_node_ip_address()
        self._process: subprocess.Popen[Any] | None = None

        for attempt in range(_PORT_START_ATTEMPTS):
            port = _free_port(max_port=55535)
            nccl_port = _free_port()
            while nccl_port == port:
                nccl_port = _free_port()
            command = _guarded_command(
                [
                    sys.executable,
                    "-m",
                    "aethereval.backends.sglang.grpc_worker",
                    "serve",
                    "--model-path",
                    model,
                    "--tp-size",
                    str(tensor_parallel_size),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(port),
                    "--nccl-port",
                    str(nccl_port),
                    "--grpc-mode",
                    "--log-level",
                    "error",
                    *_server_cli_args(model_kwargs),
                ]
            )
            process = subprocess.Popen(
                command,
                env=env,
                start_new_session=True,
            )
            self._process = process
            try:
                _wait_for_port("127.0.0.1", port, process)
            except BaseException:
                port_collision = process.poll() is not None and (
                    not _port_is_available(port) or not _port_is_available(nccl_port)
                )
                _stop_process(process)
                self._process = None
                if port_collision and attempt + 1 < _PORT_START_ATTEMPTS:
                    print(
                        "[aethereval] SGLang worker startup port collision; "
                        f"retrying with a new port ({attempt + 2}/"
                        f"{_PORT_START_ATTEMPTS})"
                    )
                    continue
                raise
            self._url = f"grpc://{node_ip}:{port}"
            break

    def url(self) -> str:
        return self._url

    def close(self) -> None:
        _stop_process(self._process)
        self._process = None


class SGLangService:
    """Ray-managed SGLang replicas behind one SMG router.

    Each Ray actor owns one tensor-parallel server. Ray places the actors across
    the attached cluster, while SMG routes every request independently across
    those replicas. No per-node SGLang setup is required.
    """

    def __init__(
        self,
        *,
        model: str,
        dp_size: int,
        tensor_parallel_size: int,
        model_kwargs: dict[str, Any] | None = None,
        router_policy: str | None = None,
    ) -> None:
        if int(dp_size) < 1:
            raise ValueError(f"dp_size must be >= 1, got {dp_size}")
        if int(tensor_parallel_size) < 1:
            raise ValueError(
                f"tensor_parallel_size must be >= 1, got {tensor_parallel_size}"
            )

        harmony_encoding_dir = _resolve_harmony_encoding_dir()

        try:
            import ray
        except ImportError as exc:
            raise RuntimeError(
                "ray is required for the SGLang service backend"
            ) from exc

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        self._ray = ray
        self.model = str(model)
        self.model_kwargs = dict(model_kwargs or {})
        if not self.model_kwargs.get("tokenizer_path") and not self.model_kwargs.get(
            "tokenizer"
        ):
            self.model_kwargs["tokenizer_path"] = self.model
        self.dp_size = int(dp_size)
        self.router_policy = router_policy
        self.router_log_level = str(self.model_kwargs.get("router_log_level", "warn"))
        self._harmony_encoding_dir = harmony_encoding_dir
        self._workers: list[Any] = []
        self._router: subprocess.Popen[Any] | None = None
        self._closed = False

        actor_cls = ray.remote(
            num_cpus=1,
            num_gpus=int(tensor_parallel_size),
        )(_SGLangServerActor)
        try:
            self._workers = [
                actor_cls.remote(
                    self.model,
                    int(tensor_parallel_size),
                    dict(self.model_kwargs),
                )
                for _ in range(self.dp_size)
            ]
            worker_urls = ray.get([worker.url.remote() for worker in self._workers])
            self.base_url = self._start_router([str(url) for url in worker_urls])
        except BaseException:
            self.close()
            raise

    def _start_router(self, worker_urls: list[str]) -> str:
        for attempt in range(_PORT_START_ATTEMPTS):
            port = _free_port()
            prometheus_port = _free_port()
            while prometheus_port == port:
                prometheus_port = _free_port()
            command = _guarded_command(
                [
                    sys.executable,
                    "-m",
                    "sglang_router.launch_router",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--worker-urls",
                    *worker_urls,
                    "--log-level",
                    self.router_log_level,
                    "--prometheus-host",
                    "127.0.0.1",
                    "--prometheus-port",
                    str(prometheus_port),
                    *_router_model_cli_args(self.model, self.model_kwargs),
                ]
            )
            if self.router_policy is not None:
                command.extend(("--policy", self.router_policy))
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["TQDM_DISABLE"] = "1"
            env[_HARMONY_ENCODING_ENV] = str(self._harmony_encoding_dir)
            process = subprocess.Popen(
                command,
                env=env,
                start_new_session=True,
            )
            self._router = process
            base_url = f"http://127.0.0.1:{port}"
            try:
                _wait_until_ready(
                    base_url,
                    process,
                    endpoint="/readiness",
                    tokenizer_model=self.model,
                )
            except BaseException:
                port_collision = process.poll() is not None and (
                    not _port_is_available(port)
                    or not _port_is_available(prometheus_port)
                )
                _stop_process(process)
                self._router = None
                if port_collision and attempt + 1 < _PORT_START_ATTEMPTS:
                    print(
                        "[aethereval] SMG router startup port collision; "
                        f"retrying with new ports ({attempt + 2}/"
                        f"{_PORT_START_ATTEMPTS})"
                    )
                    continue
                raise
            return base_url
        raise AssertionError("unreachable")

    def request_many(
        self,
        path: str,
        payloads: list[dict[str, Any]],
        *,
        show_progress: bool,
        progress_desc: str,
        progress_unit: str,
    ) -> list[Any]:
        if self._closed:
            raise RuntimeError("SGLang service is closed")
        if not payloads:
            return []
        results: list[Any] = [None] * len(payloads)
        max_workers = min(
            len(payloads),
            max(64, self.dp_size * 64),
        )
        progress = Progress(len(payloads), progress_desc, progress_unit, show_progress)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {}
        pending = iter(enumerate(payloads))

        def submit_next():
            item = next(pending, None)
            if item is not None:
                index, payload = item
                future = executor.submit(
                    _post_json,
                    f"{self.base_url}{path}",
                    {"model": self.model, **payload},
                )
                futures[future] = index

        try:
            for _ in range(max_workers):
                submit_next()
            while futures:
                completed, _ = wait(futures, timeout=10, return_when=FIRST_COMPLETED)
                progress.refresh()
                for future in completed:
                    results[futures.pop(future)] = future.result()
                    progress.update()
                    submit_next()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
            progress.close()
        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _stop_process(self._router)
        self._router = None
        if self._workers:
            try:
                self._ray.get([worker.close.remote() for worker in self._workers])
            except Exception:
                pass
            finally:
                for worker in self._workers:
                    try:
                        self._ray.kill(worker, no_restart=True)
                    except Exception:
                        pass
        self._workers = []


__all__ = ["SGLangService"]
