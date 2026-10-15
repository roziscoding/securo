from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import math
from typing import Any, AsyncIterator, Literal, Optional, cast

import traceback as traceback_module


# Provider-independent stream bounds.  Individual adapters may impose tighter
# limits, but callers must remain safe even for third-party implementations.
# Count every provider chunk, including empty and unknown chunks, so byte-free
# streams cannot occupy a worker indefinitely.
MAX_STREAM_CHUNKS = 100_000
MAX_STREAM_TEXT_BYTES = 4_000_000
MAX_STREAM_TOOL_ARGUMENT_BYTES = 1_000_000
MAX_STREAM_ALL_TOOL_ARGUMENT_BYTES = 4_000_000
MAX_STREAM_TOOL_CALLS = 128
MAX_STREAM_TOOL_METADATA_BYTES = 1_000_000
MAX_STREAM_ALL_TOOL_METADATA_BYTES = 4_000_000
MAX_STREAM_USAGE_TOKENS = 2_147_483_647
_MAX_TOOL_ARGUMENT_NODES = 20_000
_MAX_TOOL_ARGUMENT_DEPTH = 32

# In-process providers are trusted project code and must cooperate with
# cancellation.  Remote transports have their own shorter connect/read
# deadlines; this bounds the complete provider operation without moving a
# request-scoped session or ContextVar state into another task.
LLM_OPERATION_TIMEOUT_SECONDS = 300.0
LLM_CLEANUP_TIMEOUT_SECONDS = 5.0


def _unique_json_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def _validate_tool_arguments(value: dict[str, Any]) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_TOOL_ARGUMENT_NODES or depth > _MAX_TOOL_ARGUMENT_DEPTH:
            raise ValueError("tool arguments are too complex")
        if isinstance(item, str):
            item.encode()
            if "\x00" in item:
                raise ValueError("invalid JSON string")
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("invalid JSON key")
                key.encode()
                if "\x00" in key:
                    raise ValueError("invalid JSON key")
                stack.append((child, depth + 1))
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("non-finite JSON number")
        elif item is not None and not isinstance(item, (bool, int, float)):
            raise ValueError("invalid JSON value")


# --- Errors ---------------------------------------------------------------

class LLMError(Exception):
    """Sanitized provider failure with structured diagnostic metadata."""

    code: str = "unknown"
    retryable: bool
    status: Optional[int]
    provider_code: Optional[str]
    provider_param: Optional[str]
    request_id: Optional[str]
    body_category: Optional[str]

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status: Optional[int] = None,
        provider_code: Optional[str] = None,
        provider_param: Optional[str] = None,
        request_id: Optional[str] = None,
        body_category: Optional[str] = None,
    ):
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.provider_code = provider_code
        self.provider_param = provider_param
        self.request_id = request_id
        self.body_category = body_category


class LLMAuthError(LLMError):
    code = "auth"


class LLMRateLimitError(LLMError):
    code = "rate_limit"

    def __init__(self, message: str, *, status: Optional[int] = None, provider_code: Optional[str] = None, provider_param: Optional[str] = None, request_id: Optional[str] = None, body_category: Optional[str] = None):
        super().__init__(message, retryable=True, status=status, provider_code=provider_code, provider_param=provider_param, request_id=request_id, body_category=body_category)


class LLMUnavailableError(LLMError):
    code = "unavailable"

    def __init__(self, message: str, *, status: Optional[int] = None, provider_code: Optional[str] = None, provider_param: Optional[str] = None, request_id: Optional[str] = None, body_category: Optional[str] = None):
        super().__init__(message, retryable=True, status=status, provider_code=provider_code, provider_param=provider_param, request_id=request_id, body_category=body_category)


class LLMNotSupportedError(LLMError):
    code = "not_supported"


def discard_untrusted_exception(failure: BaseException) -> None:
    """Best-effort removal of traceback links without rendering the error."""
    try:
        traceback = BaseException.__dict__["__traceback__"].__get__(
            failure, type(failure)
        )
        if traceback is not None:
            traceback_module.clear_frames(traceback)
    except BaseException:
        pass
    for name in ("__traceback__", "__cause__", "__context__"):
        try:
            BaseException.__dict__[name].__set__(failure, None)
        except BaseException:
            pass
    try:
        BaseException.__dict__["args"].__set__(failure, ())
    except BaseException:
        pass


class OperationCancellationScope:
    """Track cancellation introduced after a trusted pre-operation checkpoint."""

    def __init__(self) -> None:
        self._task = asyncio.current_task()
        self._baseline = self._task.cancelling() if self._task is not None else 0

    async def checkpoint(self) -> None:
        await asyncio.sleep(0)

    def requested(self) -> bool:
        try:
            return self._task is not None and self._task.cancelling() > self._baseline
        except BaseException:
            return False


@asynccontextmanager
async def _owned_async_iterator(
    stream: AsyncIterator[Any], *, cleanup_error: str
) -> AsyncGenerator[AsyncIterator[Any], None]:
    """Close one trusted iterator owner without replacing a primary failure.

    The cleanup deadline is cooperative by design: Python cannot forcibly stop
    a coroutine that suppresses cancellation. Provider implementations are
    trusted project code; untrusted remote stalls are bounded at transports and
    by ``bounded_async_iterator``'s whole-operation deadline.
    """
    try:
        yield stream
    except BaseException as primary_failure:
        try:
            close = getattr(stream, "aclose", None)
            if close is not None:
                async with asyncio.timeout(LLM_CLEANUP_TIMEOUT_SECONDS):
                    await close()
        except asyncio.CancelledError:
            if isinstance(primary_failure, GeneratorExit):
                raise
        except BaseException as cleanup_failure:
            discard_untrusted_exception(cleanup_failure)
        raise
    else:
        cancellation = OperationCancellationScope()
        try:
            await cancellation.checkpoint()
            close = getattr(stream, "aclose", None)
            if close is not None:
                async with asyncio.timeout(LLM_CLEANUP_TIMEOUT_SECONDS):
                    await close()
        except asyncio.CancelledError as cleanup_failure:
            if cancellation.requested():
                raise
            discard_untrusted_exception(cleanup_failure)
            raise LLMUnavailableError(cleanup_error) from None
        except BaseException as cleanup_failure:
            discard_untrusted_exception(cleanup_failure)
            raise LLMUnavailableError(cleanup_error) from None


async def bounded_async_iterator(
    stream: AsyncIterator[Any],
    *,
    timeout_seconds: float = LLM_OPERATION_TIMEOUT_SECONDS,
    timeout_error: str = "LLM operation timed out",
    cleanup_error: str = "LLM stream cleanup failed",
) -> AsyncGenerator[Any, None]:
    """Iterate demand-first in the caller task with one cooperative owner."""
    try:
        async with asyncio.timeout(timeout_seconds):
            async with _owned_async_iterator(stream, cleanup_error=cleanup_error):
                async for item in stream:
                    yield item
    except TimeoutError:
        raise LLMUnavailableError(timeout_error) from None


@asynccontextmanager
async def owned_async_context(
    manager: Any, *, cleanup_error: str
) -> AsyncGenerator[Any, None]:
    """Enter/exit one trusted async context without losing a primary failure."""
    value = await manager.__aenter__()
    try:
        yield value
    except BaseException as primary_failure:
        try:
            primary_traceback = BaseException.__dict__["__traceback__"].__get__(
                primary_failure, type(primary_failure)
            )
        except BaseException:
            primary_traceback = None
        try:
            async with asyncio.timeout(LLM_CLEANUP_TIMEOUT_SECONDS):
                await manager.__aexit__(
                    type(primary_failure),
                    primary_failure,
                    primary_traceback,
                )
        except asyncio.CancelledError:
            if isinstance(primary_failure, GeneratorExit):
                raise
        except BaseException as cleanup_failure:
            discard_untrusted_exception(cleanup_failure)
        raise
    else:
        cancellation = OperationCancellationScope()
        try:
            await cancellation.checkpoint()
            async with asyncio.timeout(LLM_CLEANUP_TIMEOUT_SECONDS):
                await manager.__aexit__(None, None, None)
        except asyncio.CancelledError as cleanup_failure:
            if cancellation.requested():
                raise
            discard_untrusted_exception(cleanup_failure)
            raise LLMUnavailableError(cleanup_error) from None
        except BaseException as cleanup_failure:
            discard_untrusted_exception(cleanup_failure)
            raise LLMUnavailableError(cleanup_error) from None


# --- Messages, tools, chunks ---------------------------------------------

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Opaque Gemini signature; replay unchanged, never pass to the tool itself.
    thought_signature: Optional[str] = None


@dataclass
class ChatMessage:
    role: Role
    content: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # For role="tool":
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


@dataclass
class ToolDefinition:
    """Provider-agnostic tool spec. Matches MCP/JSON-Schema shape."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments object


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ChatChunk:
    """Streaming event. Exactly one of these fields is populated per chunk."""
    type: Literal["text_delta", "tool_call_start", "tool_call_args_delta", "tool_call_end", "finish", "usage"]
    text: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    args_delta: Optional[str] = None  # incremental JSON args while streaming
    finish_reason: Optional[str] = None
    usage: Optional[Usage] = None
    # Delivered on tool_call_end, after all streaming deltas have arrived.
    thought_signature: Optional[str] = None


_CHAT_CHUNK_FIELDS = (
    "text",
    "tool_call_id",
    "tool_name",
    "args_delta",
    "finish_reason",
    "usage",
    "thought_signature",
)
_ALL_CHAT_CHUNK_FIELDS = ("type", *_CHAT_CHUNK_FIELDS)
_CHAT_CHUNK_TYPES = frozenset(
    {
        "text_delta",
        "tool_call_start",
        "tool_call_args_delta",
        "tool_call_end",
        "finish",
        "usage",
    }
)


def _chunk_string(
    value: object,
    *,
    error: str,
    maximum_bytes: int,
    allow_empty: bool = True,
    size_error: Optional[str] = None,
) -> str:
    """Copy a string without invoking subclass hooks, then validate it."""
    value_type = type(value)
    try:
        is_string = any(
            base is str for base in type.__getattribute__(value_type, "__mro__")
        )
    except TypeError:
        is_string = False
    if not is_string:
        raise LLMUnavailableError(error)
    normalized = str.__str__(value)
    if (not allow_empty and not normalized) or "\x00" in normalized:
        raise LLMUnavailableError(error)
    try:
        encoded = str.encode(normalized)
    except UnicodeEncodeError:
        raise LLMUnavailableError(error) from None
    if len(encoded) > maximum_bytes:
        raise LLMUnavailableError(size_error or error)
    return normalized


def _snapshot_exact_fields(
    value: object,
    expected_type: type,
    fields: tuple[str, ...],
    *,
    error: str,
) -> dict[str, object]:
    """Copy required dataclass fields without descriptor fallback."""
    if type(value) is not expected_type:
        raise LLMUnavailableError(error)
    try:
        dictionary = object.__getattribute__(value, "__dict__")
        if type(dictionary) is not dict or any(
            not dict.__contains__(dictionary, name) for name in fields
        ):
            raise LLMUnavailableError(error)
        return {name: dict.__getitem__(dictionary, name) for name in fields}
    except LLMUnavailableError:
        raise
    except BaseException:  # malformed boundary objects must not leak access faults
        raise LLMUnavailableError(error) from None


def _reject_populated_chunk_fields(chunk: ChatChunk, *, allowed: frozenset[str]) -> None:
    for field_name in _CHAT_CHUNK_FIELDS:
        if field_name not in allowed and object.__getattribute__(chunk, field_name) is not None:
            raise LLMUnavailableError("LLM returned an invalid chunk")


def validate_chat_chunk(
    chunk: object,
    *,
    max_text_bytes: int = MAX_STREAM_TEXT_BYTES,
    max_tool_argument_bytes: int = MAX_STREAM_TOOL_ARGUMENT_BYTES,
    max_tool_metadata_bytes: int = MAX_STREAM_TOOL_METADATA_BYTES,
    max_usage_tokens: int = MAX_STREAM_USAGE_TOKENS,
) -> ChatChunk:
    """Return a hook-free, fully validated snapshot of one provider chunk."""
    snapshot = _snapshot_exact_fields(
        chunk,
        ChatChunk,
        _ALL_CHAT_CHUNK_FIELDS,
        error="LLM returned an invalid chunk",
    )
    chunk = ChatChunk(**cast(Any, snapshot))

    chunk_type = _chunk_string(
        object.__getattribute__(chunk, "type"),
        error="LLM returned an invalid chunk",
        maximum_bytes=64,
        allow_empty=False,
    )
    if chunk_type not in _CHAT_CHUNK_TYPES:
        raise LLMUnavailableError("LLM returned an unknown chunk type")

    if chunk_type == "text_delta":
        _reject_populated_chunk_fields(chunk, allowed=frozenset({"text"}))
        text = _chunk_string(
            object.__getattribute__(chunk, "text"),
            error="LLM returned an invalid text delta",
            maximum_bytes=max_text_bytes,
            size_error="LLM response text exceeded the size limit",
        )
        return ChatChunk(type="text_delta", text=text)

    if chunk_type == "tool_call_start":
        _reject_populated_chunk_fields(
            chunk, allowed=frozenset({"tool_call_id", "tool_name"})
        )
        call_id = _chunk_string(
            object.__getattribute__(chunk, "tool_call_id"),
            error="LLM returned invalid tool metadata",
            maximum_bytes=max_tool_metadata_bytes,
            allow_empty=False,
            size_error="LLM tool metadata exceeded the size limit",
        )
        name = _chunk_string(
            object.__getattribute__(chunk, "tool_name"),
            error="LLM returned invalid tool metadata",
            maximum_bytes=max_tool_metadata_bytes,
            allow_empty=False,
            size_error="LLM tool metadata exceeded the size limit",
        )
        return ChatChunk(type="tool_call_start", tool_call_id=call_id, tool_name=name)

    if chunk_type == "tool_call_args_delta":
        _reject_populated_chunk_fields(
            chunk, allowed=frozenset({"tool_call_id", "args_delta"})
        )
        call_id = _chunk_string(
            object.__getattribute__(chunk, "tool_call_id"),
            error="LLM returned invalid tool metadata",
            maximum_bytes=max_tool_metadata_bytes,
            allow_empty=False,
            size_error="LLM tool metadata exceeded the size limit",
        )
        delta = _chunk_string(
            object.__getattribute__(chunk, "args_delta"),
            error="LLM returned invalid tool arguments",
            maximum_bytes=max_tool_argument_bytes,
            size_error="LLM tool arguments exceeded the size limit",
        )
        return ChatChunk(
            type="tool_call_args_delta", tool_call_id=call_id, args_delta=delta
        )

    if chunk_type == "tool_call_end":
        _reject_populated_chunk_fields(
            chunk, allowed=frozenset({"tool_call_id", "thought_signature"})
        )
        call_id = _chunk_string(
            object.__getattribute__(chunk, "tool_call_id"),
            error="LLM returned invalid tool metadata",
            maximum_bytes=max_tool_metadata_bytes,
            allow_empty=False,
            size_error="LLM tool metadata exceeded the size limit",
        )
        raw_signature = object.__getattribute__(chunk, "thought_signature")
        signature = None
        if raw_signature is not None:
            signature = _chunk_string(
                raw_signature,
                error="LLM returned invalid tool metadata",
                maximum_bytes=max_tool_metadata_bytes,
                size_error="LLM tool metadata exceeded the size limit",
            )
        return ChatChunk(
            type="tool_call_end",
            tool_call_id=call_id,
            thought_signature=signature,
        )

    if chunk_type == "usage":
        _reject_populated_chunk_fields(chunk, allowed=frozenset({"usage"}))
        raw_usage = object.__getattribute__(chunk, "usage")
        usage_snapshot = _snapshot_exact_fields(
            raw_usage,
            Usage,
            ("input_tokens", "output_tokens"),
            error="LLM returned invalid usage",
        )
        input_tokens = usage_snapshot["input_tokens"]
        output_tokens = usage_snapshot["output_tokens"]
        if (
            type(input_tokens) is not int
            or not 0 <= input_tokens <= max_usage_tokens
            or type(output_tokens) is not int
            or not 0 <= output_tokens <= max_usage_tokens
        ):
            raise LLMUnavailableError("LLM returned invalid usage")
        return ChatChunk(
            type="usage",
            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        )

    _reject_populated_chunk_fields(chunk, allowed=frozenset({"finish_reason"}))
    finish_reason = _chunk_string(
        object.__getattribute__(chunk, "finish_reason"),
        error="LLM returned an invalid finish reason",
        maximum_bytes=max_tool_metadata_bytes,
        allow_empty=False,
    )
    return ChatChunk(type="finish", finish_reason=finish_reason)


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)


# --- Provider interface ---------------------------------------------------

class LLMProvider(ABC):
    name: str
    supports_native_tools: bool = True
    # Embedding capability — Anthropic returns False, others True.
    supports_embeddings: bool = True

    def __init__(self, *, api_key: str = "", base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key
        self.base_url: str = base_url or ""
        self.default_model = model

    @abstractmethod
    def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatChunk]:
        """Yield streaming chunks. Implementations are async generators."""
        ...

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
    ) -> ChatResponse:
        """Convenience: collect a streaming response into one ChatResponse."""
        text_parts: list[str] = []
        text_bytes = 0
        tool_calls: dict[str, dict] = {}
        all_args_bytes = 0
        all_metadata_bytes = 0
        finish: str = "stop"
        finish_seen = False
        usage = Usage()
        usage_seen = False
        chunk_count = 0
        stream = bounded_async_iterator(
            self.chat_stream(
                messages,
                model=model,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            timeout_seconds=LLM_OPERATION_TIMEOUT_SECONDS,
        )
        try:
            async for raw_chunk in stream:
                chunk_count += 1
                if chunk_count > MAX_STREAM_CHUNKS:
                    raise LLMUnavailableError("LLM stream returned too many chunks")
                chunk = validate_chat_chunk(
                    raw_chunk,
                    max_text_bytes=MAX_STREAM_TEXT_BYTES,
                    max_tool_argument_bytes=MAX_STREAM_TOOL_ARGUMENT_BYTES,
                    max_tool_metadata_bytes=MAX_STREAM_TOOL_METADATA_BYTES,
                    max_usage_tokens=MAX_STREAM_USAGE_TOKENS,
                )
                if finish_seen:
                    raise LLMUnavailableError("LLM stream continued after completion")
                if usage_seen:
                    if chunk.type == "usage":
                        raise LLMUnavailableError("LLM returned duplicate usage")
                    if chunk.type != "finish":
                        raise LLMUnavailableError("LLM stream continued after usage")
                if chunk.type == "text_delta":
                    if not isinstance(chunk.text, str):
                        raise LLMUnavailableError("LLM returned an invalid text delta")
                    if "\x00" in chunk.text:
                        raise LLMUnavailableError("LLM returned an invalid text delta")
                    try:
                        delta_bytes = len(chunk.text.encode())
                    except UnicodeEncodeError:
                        raise LLMUnavailableError("LLM returned invalid Unicode") from None
                    if delta_bytes > MAX_STREAM_TEXT_BYTES - text_bytes:
                        raise LLMUnavailableError("LLM response text exceeded the size limit")
                    if not chunk.text:
                        continue
                    text_bytes += delta_bytes
                    text_parts.append(chunk.text)
                elif chunk.type == "tool_call_start":
                    if (
                        not isinstance(chunk.tool_call_id, str)
                        or not chunk.tool_call_id
                        or not isinstance(chunk.tool_name, str)
                        or "\x00" in chunk.tool_call_id
                        or "\x00" in chunk.tool_name
                    ):
                        raise LLMUnavailableError("LLM returned invalid tool metadata")
                    try:
                        id_bytes = len(chunk.tool_call_id.encode())
                        name_bytes = len(chunk.tool_name.encode())
                    except UnicodeEncodeError:
                        raise LLMUnavailableError("LLM returned invalid tool metadata") from None
                    metadata_bytes = id_bytes + name_bytes
                    if (
                        id_bytes > MAX_STREAM_TOOL_METADATA_BYTES
                        or name_bytes > MAX_STREAM_TOOL_METADATA_BYTES
                        or metadata_bytes
                        > MAX_STREAM_ALL_TOOL_METADATA_BYTES - all_metadata_bytes
                    ):
                        raise LLMUnavailableError("LLM tool metadata exceeded the size limit")
                    if chunk.tool_call_id in tool_calls:
                        raise LLMUnavailableError("LLM returned a duplicate tool call")
                    if len(tool_calls) >= MAX_STREAM_TOOL_CALLS:
                        raise LLMUnavailableError("LLM returned too many tool calls")
                    all_metadata_bytes += metadata_bytes
                    tool_calls[chunk.tool_call_id] = {
                        "id": chunk.tool_call_id,
                        "name": chunk.tool_name or "",
                        "args_parts": [],
                        "args_bytes": 0,
                        "ended": False,
                    }
                elif chunk.type == "tool_call_args_delta":
                    call_id = cast(str, chunk.tool_call_id)
                    delta = cast(str, chunk.args_delta)
                    delta_bytes = len(delta.encode())
                    tc = tool_calls.get(call_id)
                    if tc is None:
                        raise LLMUnavailableError("LLM returned tool arguments before the tool call")
                    if tc["ended"]:
                        raise LLMUnavailableError("LLM stream continued after completion")
                    if not delta:
                        continue
                    args_bytes = cast(int, tc["args_bytes"])
                    if (
                        delta_bytes > MAX_STREAM_TOOL_ARGUMENT_BYTES - args_bytes
                        or delta_bytes > MAX_STREAM_ALL_TOOL_ARGUMENT_BYTES - all_args_bytes
                    ):
                        raise LLMUnavailableError("LLM tool arguments exceeded the size limit")
                    tc["args_bytes"] = args_bytes + delta_bytes
                    all_args_bytes += delta_bytes
                    cast(list[str], tc["args_parts"]).append(delta)
                elif chunk.type == "tool_call_end":
                    tc = tool_calls.get(cast(str, chunk.tool_call_id))
                    if tc is None or tc["ended"]:
                        raise LLMUnavailableError("LLM stream was incomplete")
                    signature = chunk.thought_signature
                    if signature is not None:
                        if not isinstance(signature, str) or "\x00" in signature:
                            raise LLMUnavailableError("LLM returned invalid tool metadata")
                        try:
                            signature_bytes = len(signature.encode())
                        except UnicodeEncodeError:
                            raise LLMUnavailableError("LLM returned invalid tool metadata") from None
                        if (
                            signature_bytes > MAX_STREAM_TOOL_METADATA_BYTES
                            or signature_bytes
                            > MAX_STREAM_ALL_TOOL_METADATA_BYTES - all_metadata_bytes
                        ):
                            raise LLMUnavailableError("LLM tool metadata exceeded the size limit")
                        all_metadata_bytes += signature_bytes
                    tc["thought_signature"] = chunk.thought_signature
                    tc["ended"] = True
                elif chunk.type == "finish":
                    if any(not tc["ended"] for tc in tool_calls.values()):
                        raise LLMUnavailableError("LLM stream was incomplete")
                    finish = cast(str, chunk.finish_reason)
                    finish_seen = True
                elif chunk.type == "usage":
                    if any(not tc["ended"] for tc in tool_calls.values()):
                        raise LLMUnavailableError("LLM stream was incomplete")
                    usage = cast(Usage, chunk.usage)
                    usage_seen = True
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
        if not finish_seen or any(not tc["ended"] for tc in tool_calls.values()):
            raise LLMUnavailableError("LLM stream was incomplete")
        import json
        parsed_calls = []
        for tc in tool_calls.values():
            args_buf = "".join(cast(list[str], tc["args_parts"]))
            try:
                args = (
                    json.loads(
                        args_buf,
                        object_pairs_hook=_unique_json_members,
                        parse_constant=lambda _value: (_ for _ in ()).throw(
                            ValueError("invalid JSON constant")
                        ),
                    )
                    if args_buf
                    else {}
                )
            except (json.JSONDecodeError, ValueError, RecursionError):
                raise LLMUnavailableError("LLM returned malformed tool arguments") from None
            if not isinstance(args, dict):
                raise LLMUnavailableError("LLM returned non-object tool arguments")
            try:
                _validate_tool_arguments(args)
            except (UnicodeEncodeError, ValueError):
                raise LLMUnavailableError("LLM returned invalid tool arguments") from None
            parsed_calls.append(ToolCall(
                id=tc["id"], name=tc["name"], arguments=args,
                thought_signature=tc.get("thought_signature"),
            ))
        return ChatResponse(content="".join(text_parts), tool_calls=parsed_calls, finish_reason=finish, usage=usage)

    @abstractmethod
    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        ...
