import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm.auto import tqdm


_STARTUP_TIMEOUT_SECONDS = 1800
_REQUEST_TIMEOUT_SECONDS = 24 * 60 * 60
_URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
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


def _free_port(max_port: int = 65535) -> int:
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            port = int(sock.getsockname()[1])
        if port <= max_port:
            return port


def _allocate_worker_ports(count: int) -> list[int]:
    used: set[int] = set()
    ports: list[int] = []
    for _ in range(count):
        port = _free_port(max_port=55535)
        while port in used:
            port = _free_port(max_port=55535)
        used.add(port)
        ports.append(port)
    return ports


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
        with _URL_OPENER.open(
            request, timeout=_REQUEST_TIMEOUT_SECONDS
        ) as response:
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


def _stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)


class _SGLangServerActor:
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int,
        model_kwargs: dict[str, Any],
        port: int,
    ) -> None:
        import ray

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
                "--grpc-mode",
                "--log-level",
                "error",
                *_server_cli_args(model_kwargs),
            ]
        )
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
        self._process = subprocess.Popen(
            command,
            env=env,
            start_new_session=True,
        )
        node_ip = ray.util.get_node_ip_address()
        self._url = f"grpc://{node_ip}:{port}"
        try:
            _wait_for_port("127.0.0.1", port, self._process)
        except BaseException:
            _stop_process(self._process)
            raise

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
                "tensor_parallel_size must be >= 1, "
                f"got {tensor_parallel_size}"
            )

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
        self.dp_size = int(dp_size)
        self.router_policy = router_policy
        self.router_log_level = str(
            self.model_kwargs.get("router_log_level", "warn")
        )
        self._workers: list[Any] = []
        self._router: subprocess.Popen[Any] | None = None
        self._closed = False

        actor_cls = ray.remote(
            num_cpus=1,
            num_gpus=int(tensor_parallel_size),
        )(_SGLangServerActor)
        try:
            worker_ports = _allocate_worker_ports(self.dp_size)
            self._workers = [
                actor_cls.remote(
                    self.model,
                    int(tensor_parallel_size),
                    dict(self.model_kwargs),
                    port,
                )
                for port in worker_ports
            ]
            worker_urls = ray.get([worker.url.remote() for worker in self._workers])
            self.base_url = self._start_router([str(url) for url in worker_urls])
        except BaseException:
            self.close()
            raise

    def _start_router(self, worker_urls: list[str]) -> str:
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
        self._router = subprocess.Popen(
            command,
            env=env,
            start_new_session=True,
        )
        base_url = f"http://127.0.0.1:{port}"
        _wait_until_ready(base_url, self._router, endpoint="/readiness")
        return base_url

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
        request_payloads = [
            {"model": self.model, **payload}
            for payload in payloads
        ]

        progress = None
        if show_progress:
            progress = tqdm(
                total=len(request_payloads),
                desc=progress_desc,
                unit=progress_unit,
                dynamic_ncols=True,
                mininterval=1.0,
            )

        results: list[Any] = [None] * len(request_payloads)
        max_workers = min(
            len(request_payloads),
            max(64, self.dp_size * 64),
        )
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _post_json,
                        f"{self.base_url}{path}",
                        payload,
                    ): index
                    for index, payload in enumerate(request_payloads)
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    if progress is not None:
                        progress.update(1)
        finally:
            if progress is not None:
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
