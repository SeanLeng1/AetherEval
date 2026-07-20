import inspect
import struct
from numbers import Real
from typing import Any


_OPTIONAL_REQUEST_FIELDS = ("input_embeds", "token_type_ids")


def _encode_varint(value: int) -> bytes:
    encoded = bytearray()
    while value > 0x7F:
        encoded.append((value & 0x7F) | 0x80)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _serialize_router_embed_response(response: Any) -> bytes:
    embeddings = [float(value) for value in response.embedding]
    packed_embeddings = struct.pack(
        f"<{len(embeddings)}f",
        *embeddings,
    )
    complete = bytearray()
    if packed_embeddings:
        complete.extend(b"\x0a")
        complete.extend(_encode_varint(len(packed_embeddings)))
        complete.extend(packed_embeddings)
    prompt_tokens = int(response.prompt_tokens)
    if prompt_tokens:
        complete.extend(b"\x10")
        complete.extend(_encode_varint(prompt_tokens))
    embedding_dim = int(response.embedding_dim or len(embeddings))
    if embedding_dim:
        complete.extend(b"\x20")
        complete.extend(_encode_varint(embedding_dim))

    return b"\x12" + _encode_varint(len(complete)) + bytes(complete)


def patch_smg_request_type(servicer: Any) -> tuple[str, ...]:
    request_type = servicer.TokenizedGenerateReqInput
    parameters = inspect.signature(request_type).parameters
    fields = tuple(
        name
        for name in _OPTIONAL_REQUEST_FIELDS
        if name in parameters
        and parameters[name].default is inspect.Parameter.empty
    )
    if not fields:
        return ()

    def compatible_request(*args: Any, **kwargs: Any) -> Any:
        for name in fields:
            kwargs.setdefault(name, None)
        return request_type(*args, **kwargs)

    servicer.TokenizedGenerateReqInput = compatible_request
    return fields


def patch_smg_embedding_request_type(servicer: Any) -> bool:
    request_type = servicer.TokenizedEmbeddingReqInput
    parameters = inspect.signature(request_type).parameters
    if "mm_inputs" not in parameters or "image_inputs" in parameters:
        return False

    def compatible_request(*args: Any, **kwargs: Any) -> Any:
        if "image_inputs" in kwargs:
            kwargs.setdefault("mm_inputs", kwargs.pop("image_inputs"))
        else:
            kwargs.setdefault("mm_inputs", None)
        return request_type(*args, **kwargs)

    servicer.TokenizedEmbeddingReqInput = compatible_request
    return True


def patch_smg_scalar_embedding_output(request_manager: Any) -> None:
    manager_type = request_manager.GrpcRequestManager
    original_handler = manager_type._handle_embedding_output

    async def compatible_handler(self: Any, batch_out: Any) -> Any:
        batch_out.embeddings = [
            [float(embedding)] if isinstance(embedding, Real) else embedding
            for embedding in batch_out.embeddings
        ]
        return await original_handler(self, batch_out)

    manager_type._handle_embedding_output = compatible_handler


def patch_smg_embedding_response_wire(pb2_grpc: Any) -> bool:
    pb2 = getattr(pb2_grpc, "sglang_scheduler_pb2", None)
    if pb2 is None:
        pb2 = pb2_grpc.sglang__scheduler__pb2
    response_type = pb2.EmbedResponse
    field_numbers = {
        field.number for field in response_type.DESCRIPTOR.fields
    }
    if 2 in field_numbers or 3 in field_numbers:
        return False

    original_add = pb2_grpc.add_SglangSchedulerServicer_to_server

    def compatible_add(servicer: Any, server: Any) -> Any:
        original_factory = pb2_grpc.grpc.unary_unary_rpc_method_handler

        def compatible_factory(
            behavior: Any,
            request_deserializer: Any = None,
            response_serializer: Any = None,
        ) -> Any:
            if getattr(behavior, "__name__", None) == "Embed":
                response_serializer = _serialize_router_embed_response
            return original_factory(
                behavior,
                request_deserializer=request_deserializer,
                response_serializer=response_serializer,
            )

        pb2_grpc.grpc.unary_unary_rpc_method_handler = compatible_factory
        try:
            return original_add(servicer, server)
        finally:
            pb2_grpc.grpc.unary_unary_rpc_method_handler = original_factory

    pb2_grpc.add_SglangSchedulerServicer_to_server = compatible_add
    return True


def disable_smg_http_sidecar(server: Any) -> None:
    original_serve_grpc = server.serve_grpc

    async def grpc_only(server_args: Any, model_info: Any = None) -> Any:
        return await original_serve_grpc(server_args, model_info)

    server.serve_grpc = grpc_only


def main() -> None:
    import smg_grpc_servicer.sglang.request_manager as request_manager
    import smg_grpc_servicer.sglang.server as server
    import smg_grpc_servicer.sglang.servicer as servicer
    from sglang.cli.main import main as sglang_main

    patch_smg_request_type(servicer)
    patch_smg_embedding_request_type(servicer)
    patch_smg_scalar_embedding_output(request_manager)
    patch_smg_embedding_response_wire(
        servicer.sglang_scheduler_pb2_grpc
    )
    disable_smg_http_sidecar(server)
    sglang_main()


if __name__ == "__main__":
    main()
