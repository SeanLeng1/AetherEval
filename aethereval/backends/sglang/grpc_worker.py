"""SGLang worker entrypoint: apply AetherEval's worker patches, then `sglang serve`.

Supported stack (enforced by service._check_stack_versions): SGLang >= 0.5.18 and
smg-grpc-servicer >= 0.8.0. Those servicers build complete SGLang request objects
and send array("q") token ids, so no request shims are needed. Reward-model
(embedding) replicas serve plain HTTP and never reach SMG's gRPC embedding path.
"""

import os
from typing import Any


def disable_smg_http_sidecar(server: Any) -> None:
    # SGLang's legacy SMG entrypoint (sglang.srt.entrypoints.grpc_server) imports
    # smg_grpc_servicer.sglang.server.serve_grpc lazily and starts an HTTP sidecar
    # on port + 1 only when its signature accepts on_request_manager_ready.
    # Hiding that hook keeps each worker on its single allocated port.
    original_serve_grpc = server.serve_grpc

    async def grpc_only(server_args: Any, model_info: Any = None) -> Any:
        return await original_serve_grpc(server_args, model_info)

    server.serve_grpc = grpc_only


def main() -> None:
    if os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE") == "aethereval.backends.sglang.models":
        from aethereval.backends.sglang.models.gpt2_context import install_context_patch

        install_context_patch()

    import smg_grpc_servicer.sglang.server as server
    from sglang.cli.main import main as sglang_main

    disable_smg_http_sidecar(server)
    sglang_main()


if __name__ == "__main__":
    main()
