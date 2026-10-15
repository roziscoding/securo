"""OpenAI ChatGPT-plan provider for the public Responses API."""
from __future__ import annotations

import asyncio
import json
import math
import ssl
import traceback as traceback_module
import types
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TypedDict, cast

import httpx

from app.agents.config import MAX_OAUTH_TIMEOUT_SECONDS, get_agent_settings
from app.agents.providers.base import (
    ChatChunk,
    ChatMessage,
    LLMAuthError,
    LLMError,
    LLMNotSupportedError,
    LLMProvider,
    LLMRateLimitError,
    LLMUnavailableError,
    ToolDefinition,
    Usage,
    bounded_async_iterator,
    owned_async_context,
)

_RESPONSES_URL = "https://api.openai.com/v1/responses"
_TOOL_NAMESPACE = "securo"
_MAX_REQUEST_BYTES = 4_000_000
_MAX_STREAM_BYTES = 8_000_000
_MAX_EVENT_BYTES = 1_000_000
_MAX_EVENTS = 100_000
_MAX_TEXT_BYTES = 4_000_000
_MAX_TOOL_ARGUMENT_BYTES = 1_000_000
_MAX_ALL_TOOL_ARGUMENT_BYTES = 4_000_000
_MAX_ERROR_BYTES = 65_536
_MAX_USAGE_TOKENS = 2_147_483_647
_MAX_JSON_INTEGER = 2**63 - 1
_MAX_JSON_DEPTH = 32
_MAX_JSON_MEMBERS = 20_000
_MAX_FIELD_BYTES = 1_000_000


@dataclass(frozen=True)
class AccessTokenSnapshot:
    access_token: str = field(repr=False)
    credential_version: int


AccessTokenContext = Callable[..., AbstractAsyncContextManager[AccessTokenSnapshot]]


class _StreamingResponse(Protocol):
    status_code: int

    def aiter_raw(self) -> AsyncIterator[bytes]: ...


class _VersionRefreshRequired(Exception):
    """Private control signal for one pre-output version-bound refresh."""


class _SecretBoundaryFailure(Exception):
    """Internal marker for an unexpected failure after bearer acquisition."""


_EXCEPTION_ARGS = BaseException.__dict__["args"]
_EXCEPTION_CAUSE = BaseException.__dict__["__cause__"]
_EXCEPTION_CONTEXT = BaseException.__dict__["__context__"]
_EXCEPTION_TRACEBACK = BaseException.__dict__["__traceback__"]
_EXCEPTION_DICTIONARY = BaseException.__dict__["__dict__"]


def _safe_mro(value: object) -> tuple[type, ...]:
    try:
        return type.__getattribute__(type(value), "__mro__")
    except BaseException:
        return ()


def _is_instance_without_hooks(value: object, expected: type) -> bool:
    try:
        return any(base is expected for base in _safe_mro(value))
    except BaseException:
        return False


def _exception_links(error: BaseException) -> tuple[BaseException, ...]:
    """Read native exception links without invoking subclass accessors."""
    links: list[BaseException] = []
    try:
        cause = _EXCEPTION_CAUSE.__get__(error, type(error))
        if _is_instance_without_hooks(cause, BaseException):
            links.append(cause)
    except BaseException:
        pass
    try:
        context = _EXCEPTION_CONTEXT.__get__(error, type(error))
        if _is_instance_without_hooks(context, BaseException):
            links.append(context)
    except BaseException:
        pass
    return tuple(links)


def _exception_contains_cancellation(error: BaseException) -> bool:
    """Hook-free search for an actual CancelledError in the exception chain."""
    try:
        pending = [error]
        seen: set[int] = set()
        inspected = 0
        while pending and inspected < 50_000:
            current = pending.pop()
            identity = id(current)
            if identity in seen:
                continue
            seen.add(identity)
            inspected += 1
            if _is_instance_without_hooks(current, asyncio.CancelledError):
                return True
            pending.extend(_exception_links(current))
        if pending:
            return True
    except BaseException:
        return False
    return False


class _OperationCancellation:
    """Recognize cancellation introduced during one same-task operation."""

    def __init__(self) -> None:
        self._task = asyncio.current_task()
        self._baseline = self._task.cancelling() if self._task is not None else 0

    async def checkpoint(self) -> None:
        await asyncio.sleep(0)

    def contains(self, error: BaseException) -> bool:
        if _exception_contains_cancellation(error):
            return True
        try:
            return self._task is not None and self._task.cancelling() > self._baseline
        except BaseException:
            return False


def _discard_exception_secrets_safely(error: BaseException) -> None:
    """Best-effort graph/frame scrubber; hostile objects can never escape it."""
    try:
        pending: list[object] = [error]
        seen: set[int] = set()
        inspected = 0
        while pending and inspected < 50_000:
            value = pending.pop()
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            inspected += 1
            mro = _safe_mro(value)

            if BaseException in mro:
                current = cast(BaseException, value)
                pending.extend(_exception_links(current))
                try:
                    arguments = _EXCEPTION_ARGS.__get__(current, type(current))
                    if type(arguments) is tuple:
                        pending.extend(tuple.__iter__(arguments))
                except BaseException:
                    pass
                try:
                    dictionary = _EXCEPTION_DICTIONARY.__get__(
                        current, type(current)
                    )
                    if type(dictionary) is dict:
                        pending.extend(dict.values(dictionary))
                        dict.clear(dictionary)
                except BaseException:
                    pass
                try:
                    traceback = _EXCEPTION_TRACEBACK.__get__(
                        current, type(current)
                    )
                    if traceback is not None:
                        traceback_module.clear_frames(traceback)
                except BaseException:
                    pass
                for cls in mro:
                    try:
                        descriptors = type.__getattribute__(cls, "__dict__").values()
                    except BaseException:
                        continue
                    for descriptor in descriptors:
                        if not isinstance(descriptor, types.MemberDescriptorType):
                            continue
                        try:
                            item = descriptor.__get__(current, type(current))
                        except BaseException:
                            continue
                        pending.append(item)
                        try:
                            descriptor.__set__(current, None)
                        except BaseException:
                            pass
                for descriptor, replacement in (
                    (_EXCEPTION_ARGS, ()),
                    (_EXCEPTION_TRACEBACK, None),
                    (_EXCEPTION_CAUSE, None),
                    (_EXCEPTION_CONTEXT, None),
                ):
                    try:
                        descriptor.__set__(current, replacement)
                    except BaseException:
                        pass
                continue

            if bytearray in mro:
                try:
                    bytearray.clear(cast(bytearray, value))
                except BaseException:
                    pass
                continue
            if dict in mro:
                try:
                    mapping = cast(dict[object, object], value)
                    for key, item in dict.items(mapping):
                        pending.append(key)
                        pending.append(item)
                    dict.clear(mapping)
                except BaseException:
                    pass
                continue
            if list in mro:
                try:
                    sequence = cast(list[object], value)
                    pending.extend(list.__iter__(sequence))
                    list.clear(sequence)
                except BaseException:
                    pass
                continue
            if set in mro:
                try:
                    collection = cast(set[object], value)
                    pending.extend(set.__iter__(collection))
                    set.clear(collection)
                except BaseException:
                    pass
                continue
            if tuple in mro:
                try:
                    pending.extend(tuple.__iter__(cast(tuple[object, ...], value)))
                except BaseException:
                    pass
                continue
            if frozenset in mro:
                try:
                    pending.extend(frozenset.__iter__(cast(frozenset[object], value)))
                except BaseException:
                    pass
                continue
            if httpx.Request in mro or httpx.Response in mro:
                try:
                    from app.agents.services.openai_oauth import (
                        _scrub_request,
                        _scrub_response,
                    )

                    if httpx.Request in mro:
                        _scrub_request(cast(httpx.Request, value))
                    else:
                        _scrub_response(cast(httpx.Response, value))
                    continue
                except BaseException:
                    # Fall through to native dictionary/slot erasure.
                    pass
            if value is None or any(
                base in {str, bytes, int, float, bool, type, types.ModuleType,
                         types.FunctionType, types.CodeType}
                for base in mro
            ):
                continue

            for cls in mro:
                try:
                    descriptors = type.__getattribute__(cls, "__dict__").values()
                except BaseException:
                    continue
                for descriptor in descriptors:
                    if isinstance(descriptor, types.GetSetDescriptorType):
                        try:
                            dictionary = descriptor.__get__(value, type(value))
                        except BaseException:
                            continue
                        if type(dictionary) is dict:
                            pending.extend(dict.values(dictionary))
                            try:
                                dict.clear(dictionary)
                            except BaseException:
                                pass
                    elif isinstance(descriptor, types.MemberDescriptorType):
                        try:
                            item = descriptor.__get__(value, type(value))
                        except BaseException:
                            continue
                        pending.append(item)
                        try:
                            descriptor.__set__(value, None)
                        except BaseException:
                            pass
    except BaseException:
        pass


def _safe_exception_text(error: BaseException) -> str | None:
    try:
        arguments = _EXCEPTION_ARGS.__get__(error, type(error))
        if type(arguments) is not tuple or not arguments:
            return ""
        value = tuple.__getitem__(arguments, 0)
        if type(value) is not str:
            return None
        normalized = value
        str.encode(normalized)
        return normalized
    except BaseException:
        return None


def _safe_llm_failure(
    error: BaseException,
) -> tuple[type[LLMError], str, int | None, _ErrorMetadata]:
    """Snapshot an LLMError without calling hostile exception hooks."""
    fallback = (
        LLMUnavailableError,
        "ChatGPT request failed safely",
        None,
        _EMPTY_ERROR_METADATA.copy(),
    )
    try:
        actual_type = type(error)
        mro = _safe_mro(error)
        if actual_type is LLMRateLimitError:
            error_type: type[LLMError] = LLMRateLimitError
        elif actual_type is LLMUnavailableError:
            error_type = LLMUnavailableError
        elif actual_type is LLMAuthError:
            error_type = LLMAuthError
        elif actual_type is LLMNotSupportedError:
            error_type = LLMNotSupportedError
        elif LLMError in mro:
            return fallback
        else:
            return fallback
        message = _safe_exception_text(error)
        dictionary = _EXCEPTION_DICTIONARY.__get__(error, type(error))
        if message is None or type(dictionary) is not dict:
            return fallback
        status = dict.get(dictionary, "status")
        if status is not None and type(status) is not int:
            return fallback
        metadata_values: dict[str, str | None] = {
            "provider_code": None,
            "provider_param": None,
            "request_id": None,
            "body_category": None,
        }
        for name in metadata_values:
            value = dict.get(dictionary, name)
            if value is not None:
                if type(value) is not str:
                    return fallback
                str.encode(value)
            metadata_values[name] = value
        return error_type, message, status, cast(_ErrorMetadata, metadata_values)
    except BaseException:
        return fallback


def _validate_json_graph(value: Any, *, label: str, allow_nul: bool = False) -> None:
    remaining = _MAX_JSON_MEMBERS

    def visit(item: Any, depth: int) -> None:
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise LLMNotSupportedError(f"ChatGPT {label} has too many JSON values")
        if depth > _MAX_JSON_DEPTH:
            raise LLMNotSupportedError(f"ChatGPT {label} is too deeply nested")
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, str):
            if not allow_nul and "\x00" in item:
                raise LLMNotSupportedError(f"ChatGPT {label} contains an invalid string")
            try:
                item.encode()
            except UnicodeEncodeError:
                raise LLMNotSupportedError(f"ChatGPT {label} contains invalid Unicode") from None
            return
        if isinstance(item, int):
            if abs(item) > _MAX_JSON_INTEGER:
                raise LLMNotSupportedError(f"ChatGPT {label} contains an oversized integer")
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise LLMNotSupportedError(f"ChatGPT {label} contains a non-finite number")
            return
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise LLMNotSupportedError(f"ChatGPT {label} contains a non-string key")
                if not allow_nul and "\x00" in key:
                    raise LLMNotSupportedError(f"ChatGPT {label} contains an invalid key")
                try:
                    key.encode()
                except UnicodeEncodeError:
                    raise LLMNotSupportedError(
                        f"ChatGPT {label} contains invalid Unicode"
                    ) from None
                visit(child, depth + 1)
            return
        raise LLMNotSupportedError(f"ChatGPT {label} contains a non-JSON value")

    visit(value, 0)


def _bounded_utf8_size(value: str, *, label: str, maximum: int) -> int:
    if isinstance(value, str) and "\x00" in value:
        raise LLMNotSupportedError(f"ChatGPT {label} contains an invalid string")
    try:
        size = len(value.encode()) if isinstance(value, str) else maximum + 1
    except UnicodeEncodeError:
        raise LLMNotSupportedError(f"ChatGPT {label} contains invalid Unicode") from None
    if size > maximum:
        raise LLMNotSupportedError(f"ChatGPT {label} exceeded the size limit")
    return size


def _json_size_upper_bound(value: Any, *, label: str, limit: int) -> int:
    """Bound encoded size without materializing an encoded aggregate."""
    total = 0
    remaining = _MAX_JSON_MEMBERS
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        remaining -= 1
        if remaining < 0:
            raise LLMNotSupportedError(f"ChatGPT {label} has too many JSON values")
        if depth > _MAX_JSON_DEPTH:
            raise LLMNotSupportedError(f"ChatGPT {label} is too deeply nested")
        if item is None:
            total += 4
        elif isinstance(item, bool):
            total += 5
        elif isinstance(item, str):
            total += 2 + 6 * _bounded_utf8_size(item, label=label, maximum=limit)
        elif isinstance(item, int):
            if abs(item) > _MAX_JSON_INTEGER:
                raise LLMNotSupportedError(f"ChatGPT {label} contains an oversized integer")
            total += 20
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise LLMNotSupportedError(f"ChatGPT {label} contains a non-finite number")
            total += 32
        elif isinstance(item, list):
            total += 2 + len(item)
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, dict):
            total += 2 + len(item)
            for key, child in item.items():
                if not isinstance(key, str):
                    raise LLMNotSupportedError(f"ChatGPT {label} contains a non-string key")
                total += 3 + 6 * _bounded_utf8_size(key, label=label, maximum=limit)
                stack.append((child, depth + 1))
        else:
            raise LLMNotSupportedError(f"ChatGPT {label} contains a non-JSON value")
        if total > limit:
            raise LLMNotSupportedError(f"ChatGPT {label} exceeded the size limit")
    return total


def _bounded_identifier(value: Any, *, label: str, maximum: int = 512) -> str:
    try:
        size = len(value.encode()) if isinstance(value, str) else 0
    except UnicodeEncodeError:
        size = 0
    if (
        not isinstance(value, str)
        or not value
        or not size
        or size > maximum
        or "\x00" in value
    ):
        raise LLMNotSupportedError(f"ChatGPT {label} is invalid")
    return value


def _serialize_input(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
    instructions: list[str] = []
    input_items: list[dict] = []
    function_calls = 0
    call_ids: set[str] = set()
    pending_outputs: set[str] = set()
    completed_outputs: set[str] = set()
    request_bytes = 0
    if not messages:
        raise LLMNotSupportedError("ChatGPT request has no messages")
    if len(messages) > 512:
        raise LLMNotSupportedError("ChatGPT request has too many messages")
    saw_non_system = False
    for message in messages:
        if message.role not in {"system", "user", "assistant", "tool"}:
            raise LLMNotSupportedError("ChatGPT request contains an invalid message role")
        if pending_outputs and message.role != "tool":
            raise LLMNotSupportedError("ChatGPT function call history is incomplete")
        if message.content is not None and not isinstance(message.content, str):
            raise LLMNotSupportedError("ChatGPT request contains invalid message content")
        if message.content is not None:
            request_bytes += _bounded_utf8_size(
                message.content, label="message content", maximum=_MAX_FIELD_BYTES
            )
            if request_bytes > _MAX_REQUEST_BYTES:
                raise LLMNotSupportedError("ChatGPT request exceeded the size limit")
        if message.role != "assistant" and message.tool_calls:
            raise LLMNotSupportedError("ChatGPT request contains misplaced function calls")
        if message.role != "tool" and message.tool_call_id is not None:
            raise LLMNotSupportedError("ChatGPT request contains a misplaced function call ID")
        if message.role == "user" and message.content is None:
            raise LLMNotSupportedError("ChatGPT user message is missing content")
        if message.role == "assistant" and message.content is None and not message.tool_calls:
            raise LLMNotSupportedError("ChatGPT assistant message is empty")
        if message.role == "system":
            if saw_non_system:
                raise LLMNotSupportedError("ChatGPT system instructions must precede messages")
            if message.content:
                instructions.append(message.content)
            continue
        saw_non_system = True
        if message.role in {"user", "assistant"} and message.content is not None:
            if message.role == "user" and pending_outputs:
                raise LLMNotSupportedError("ChatGPT function call history is incomplete")
            input_items.append({"role": message.role, "content": message.content})
        if message.role == "assistant":
            function_calls += len(message.tool_calls)
            if function_calls > 128:
                raise LLMNotSupportedError("ChatGPT request has too many function calls")
            for call in message.tool_calls:
                call_id = _bounded_identifier(call.id, label="function call ID")
                name = _bounded_identifier(call.name, label="function name", maximum=256)
                if call_id in call_ids:
                    raise LLMNotSupportedError("ChatGPT request contains duplicate function call IDs")
                if not isinstance(call.arguments, dict):
                    raise LLMNotSupportedError("ChatGPT function arguments must be a JSON object")
                _validate_json_graph(call.arguments, label="function arguments")
                request_bytes += _json_size_upper_bound(
                    call.arguments,
                    label="function arguments",
                    limit=_MAX_REQUEST_BYTES - request_bytes,
                )
                try:
                    arguments = json.dumps(
                        call.arguments,
                        separators=(",", ":"),
                        sort_keys=True,
                        allow_nan=False,
                    )
                except (TypeError, ValueError, RecursionError):
                    raise LLMNotSupportedError("ChatGPT function arguments are not valid JSON") from None
                call_ids.add(call_id)
                pending_outputs.add(call_id)
                input_items.append(
                    {
                        "type": "function_call",
                        "namespace": _TOOL_NAMESPACE,
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                    }
                )
        elif message.role == "tool":
            call_id = message.tool_call_id
            if not call_id:
                raise LLMNotSupportedError("ChatGPT tool output is missing a call ID")
            if call_id not in pending_outputs or call_id in completed_outputs:
                raise LLMNotSupportedError("ChatGPT tool output has no unique function call")
            pending_outputs.remove(call_id)
            completed_outputs.add(call_id)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": message.content or "",
                }
            )
        if len(input_items) > 1024:
            raise LLMNotSupportedError("ChatGPT request has too many input items")
    if pending_outputs:
        raise LLMNotSupportedError("ChatGPT function call history is incomplete")
    return ("\n\n".join(instructions) or None, input_items)


def _serialize_tools(tools: Optional[list[ToolDefinition]]) -> list[dict] | None:
    if not tools:
        return None
    if len(tools) > 128:
        raise LLMNotSupportedError("ChatGPT request has too many tools")
    serialized = []
    names: set[str] = set()
    aggregate_bytes = 0
    for tool in tools:
        name = _bounded_identifier(tool.name, label="tool name", maximum=256)
        if name in names:
            raise LLMNotSupportedError("ChatGPT request contains duplicate tool names")
        try:
            description_size = len(tool.description.encode()) if isinstance(tool.description, str) else -1
        except UnicodeEncodeError:
            description_size = -1
        if (
            description_size < 0
            or description_size > 32_768
            or "\x00" in tool.description
        ):
            raise LLMNotSupportedError("ChatGPT tool description is invalid")
        if not isinstance(tool.parameters, dict):
            raise LLMNotSupportedError("ChatGPT tool parameters must be a JSON schema object")
        _validate_json_graph(tool.parameters, label="tool schema")
        aggregate_bytes += len(name.encode()) + description_size
        aggregate_bytes += _json_size_upper_bound(
            tool.parameters,
            label="tool schema",
            limit=_MAX_REQUEST_BYTES - aggregate_bytes,
        )
        if aggregate_bytes > _MAX_REQUEST_BYTES:
            raise LLMNotSupportedError("ChatGPT tools exceeded the size limit")
        names.add(name)
        serialized.append(
            {
                "type": "function",
                "name": name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        )
    return [
        {
            "type": "namespace",
            "name": _TOOL_NAMESPACE,
            "description": "Securo client-side tools",
            "tools": serialized,
        }
    ]


def _unique_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def _strict_json(data: bytes) -> Any:
    return json.loads(
        data,
        object_pairs_hook=_unique_members,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}")
        ),
    )


def _new_responses_transport() -> httpx.AsyncBaseTransport:
    settings = get_agent_settings()
    ssl_context = ssl.create_default_context(
        cafile=settings.siwc_oauth_ca_bundle or None
    )
    return httpx.AsyncHTTPTransport(
        verify=ssl_context,
        proxy=settings.siwc_oauth_proxy_url or None,
        trust_env=False,
    )


async def _iter_sse(response: _StreamingResponse) -> AsyncGenerator[dict[str, Any], None]:
    buffer = bytearray()
    total = 0
    count = 0
    event_start = 0
    line_start = 0
    last_line_end = 0
    scan_from = 0

    def extract_blocks(*, eof: bool) -> list[bytes]:
        nonlocal event_start, line_start, last_line_end, scan_from
        blocks: list[bytes] = []
        while scan_from < len(buffer):
            marker = buffer[scan_from]
            if marker not in {10, 13}:
                scan_from += 1
                continue
            if marker == 13 and scan_from + 1 == len(buffer) and not eof:
                break
            terminator_end = scan_from + 1
            if marker == 13 and terminator_end < len(buffer) and buffer[terminator_end] == 10:
                terminator_end += 1
            if scan_from == line_start:
                blocks.append(bytes(buffer[event_start:last_line_end]))
                event_start = terminator_end
                line_start = terminator_end
                last_line_end = terminator_end
            else:
                last_line_end = scan_from
                line_start = terminator_end
            scan_from = terminator_end
        return blocks

    def decode_block(block: bytes) -> dict[str, Any] | None:
        nonlocal count
        if len(block) > _MAX_EVENT_BYTES:
            raise LLMUnavailableError("ChatGPT response event exceeded the size limit")
        data_lines = []
        for line in block.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n"):
            if line.startswith(b"data:"):
                data_lines.append(line[5:].lstrip(b" "))
        if not data_lines:
            return None
        count += 1
        if count > _MAX_EVENTS:
            raise LLMUnavailableError("ChatGPT response had too many events")
        try:
            event = _strict_json(b"\n".join(data_lines))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
            raise LLMUnavailableError("ChatGPT returned a malformed stream event") from None
        if not isinstance(event, dict):
            raise LLMUnavailableError("ChatGPT returned a malformed stream event")
        try:
            _validate_json_graph(event, label="stream event", allow_nul=True)
        except LLMNotSupportedError:
            raise LLMUnavailableError("ChatGPT returned an oversized stream event") from None
        return event

    raw_stream = bounded_async_iterator(
        response.aiter_raw(),
        timeout_seconds=get_agent_settings().siwc_oauth_timeout_seconds,
        timeout_error="ChatGPT response timed out",
        cleanup_error="ChatGPT response cleanup failed",
    )
    try:
        async for raw in raw_stream:
            if len(raw) > _MAX_STREAM_BYTES - total:
                raise LLMUnavailableError("ChatGPT response exceeded the stream limit")
            total += len(raw)
            buffer.extend(raw)
            for block in extract_blocks(eof=False):
                event = decode_block(block)
                if event is not None:
                    yield event
            if len(buffer) - event_start > _MAX_EVENT_BYTES:
                raise LLMUnavailableError("ChatGPT response event exceeded the size limit")
        for block in extract_blocks(eof=True):
            event = decode_block(block)
            if event is not None:
                yield event
        if buffer[event_start:].strip():
            raise LLMUnavailableError("ChatGPT stream ended with a partial event")
    finally:
        close = getattr(raw_stream, "aclose", None)
        if close is not None:
            await close()


@dataclass(frozen=True)
class _ProviderError:
    code: str | None = None
    param: str | None = None
    request_id: str | None = None
    category: str | None = None


def _safe_metadata(value: Any, maximum: int = 512) -> str | None:
    try:
        if (
            type(value) is not str
            or type(maximum) is not int
            or maximum < 0
            or not value
            or str.__len__(value) > maximum
            or str.__contains__(value, "\x00")
        ):
            return None
        str.encode(value)
    except BaseException:
        return None
    return value


async def _read_error_body(response: _StreamingResponse) -> _ProviderError:
    size = 0
    body = bytearray()
    headers = getattr(response, "headers", {})
    request_id = _safe_metadata(headers.get("x-request-id"))
    raw_stream = bounded_async_iterator(
        response.aiter_raw(),
        timeout_seconds=get_agent_settings().siwc_oauth_timeout_seconds,
        timeout_error="ChatGPT error response timed out",
        cleanup_error="ChatGPT error response cleanup failed",
    )
    try:
        async for part in raw_stream:
            if len(part) > _MAX_ERROR_BYTES - size:
                raise LLMUnavailableError(
                    f"ChatGPT request failed with HTTP {response.status_code} (body too large)",
                    status=response.status_code,
                    request_id=request_id,
                    body_category="oversized_body",
                )
            size += len(part)
            body.extend(part)
    finally:
        close = getattr(raw_stream, "aclose", None)
        if close is not None:
            await close()
    try:
        payload = _strict_json(bytes(body)) if body else None
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
        return _ProviderError(request_id=request_id, category="malformed_body")
    finally:
        body[:] = b"\0" * len(body)
        body.clear()
    if not isinstance(payload, dict):
        return _ProviderError(request_id=request_id, category="non_object_body")
    error = payload.get("error")
    return _ProviderError(
        code=_safe_metadata(error.get("code"), 256) if isinstance(error, dict) else None,
        param=_safe_metadata(error.get("param")) if isinstance(error, dict) else None,
        request_id=request_id or _safe_metadata(payload.get("request_id")),
        category="error_object" if isinstance(error, dict) else "structured_body",
    )


class _ErrorMetadata(TypedDict):
    provider_code: str | None
    provider_param: str | None
    request_id: str | None
    body_category: str | None


_EMPTY_ERROR_METADATA: _ErrorMetadata = {
    "provider_code": None,
    "provider_param": None,
    "request_id": None,
    "body_category": None,
}


def _error_metadata(detail: _ProviderError) -> _ErrorMetadata:
    return {
        "provider_code": detail.code,
        "provider_param": detail.param,
        "request_id": detail.request_id,
        "body_category": detail.category,
    }


def _raise_status(status: int, detail: _ProviderError | None = None) -> None:
    detail = detail or _ProviderError()
    code = detail.code
    metadata = _error_metadata(detail)
    if code == "subscription_sharing_usage_limit_exceeded":
        raise LLMRateLimitError("ChatGPT plan usage limit was reached", status=429, **metadata)
    if code in {"subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable"}:
        raise LLMUnavailableError("ChatGPT plan is temporarily unavailable", status=503, **metadata)
    if code in {"subscription_sharing_unsupported_capability", "subscription_sharing_route_not_supported"}:
        raise LLMNotSupportedError("ChatGPT plan does not support this request", status=status, **metadata)
    if code == "subscription_sharing_invalid_user":
        raise LLMAuthError("ChatGPT credentials were rejected", status=401, **metadata)
    if status in {401, 403}:
        raise LLMAuthError("ChatGPT credentials were rejected", status=status, **metadata)
    if status == 429:
        raise LLMRateLimitError("ChatGPT plan is rate limited", status=status, **metadata)
    if 400 <= status < 500:
        raise LLMNotSupportedError("ChatGPT rejected the request", status=status, **metadata)
    raise LLMUnavailableError(f"ChatGPT request failed with HTTP {status}", status=status, **metadata)


def _raise_terminal_event(event_type: str, event: dict[str, Any]) -> None:
    response = event.get("response")
    error = response.get("error") if isinstance(response, dict) else event.get("error")
    code = error.get("code") if isinstance(error, dict) else event.get("code")
    detail = _ProviderError(
        code=_safe_metadata(code, 256),
        param=_safe_metadata(error.get("param")) if isinstance(error, dict) else None,
        request_id=_safe_metadata(event.get("request_id")),
        category=event_type,
    )
    metadata = _error_metadata(detail)
    if code == "subscription_sharing_usage_limit_exceeded":
        raise LLMRateLimitError("ChatGPT plan usage limit was reached", status=429, **metadata)
    if code == "subscription_sharing_invalid_user":
        raise _VersionRefreshRequired()
    if code in {"subscription_sharing_user_not_eligible", "chatpass_v2_scope_not_authorized", "chatpass_v2_invalid_authorization_context"}:
        raise LLMAuthError("ChatGPT credentials were rejected", status=403, **metadata)
    if code in {"subscription_sharing_unsupported_capability", "subscription_sharing_route_not_supported"}:
        raise LLMNotSupportedError("ChatGPT plan does not support this request", **metadata)
    if code in {"subscription_sharing_usage_unavailable", "subscription_sharing_user_unavailable"}:
        raise LLMUnavailableError("ChatGPT plan is temporarily unavailable", status=503, **metadata)
    if code:
        raise LLMNotSupportedError("ChatGPT rejected the request", **metadata)
    if event_type == "response.incomplete":
        raise LLMUnavailableError("ChatGPT response was incomplete", **metadata)
    raise LLMUnavailableError("ChatGPT response failed", **metadata)


def _stream_index(value: Any, *, label: str, maximum: int = 127) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 0 <= value <= maximum
    ):
        raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
    return value


def _stream_id(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or "\x00" in value
    ):
        raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
    try:
        value.encode()
    except UnicodeEncodeError:
        raise LLMUnavailableError(f"ChatGPT returned an invalid {label}") from None
    return value


def _validate_output_part(part: Any, *, label: str) -> tuple[str, str]:
    if not isinstance(part, dict):
        raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
    part_type = part.get("type")
    if part_type == "output_text":
        value = part.get("text")
        if (
            not set(part).issubset({"type", "text", "annotations", "logprobs"})
            or not isinstance(value, str)
            or part.get("annotations") != []
            or ("logprobs" in part and not isinstance(part.get("logprobs"), list))
        ):
            raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
        try:
            _validate_json_graph(part.get("logprobs", []), label="text log probabilities")
            _json_size_upper_bound(
                part.get("logprobs", []),
                label="text log probabilities",
                limit=_MAX_EVENT_BYTES,
            )
        except LLMNotSupportedError:
            raise LLMUnavailableError(f"ChatGPT returned an invalid {label}") from None
    elif part_type == "refusal":
        value = part.get("refusal")
        if set(part) != {"type", "refusal"} or not isinstance(value, str):
            raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
    elif part_type == "reasoning_text":
        value = part.get("text")
        if set(part) != {"type", "text"} or not isinstance(value, str):
            raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
    else:
        raise LLMUnavailableError(f"ChatGPT returned an unsupported {label}")
    if "\x00" in value:
        raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
    try:
        value.encode()
    except UnicodeEncodeError:
        raise LLMUnavailableError(f"ChatGPT returned an invalid {label}") from None
    return part_type, value


def _json_values_equal(left: Any, right: Any) -> bool:
    """Compare parsed JSON without Python's bool/int numeric coercion."""
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_values_equal(left_value, right_value)
            for left_value, right_value in zip(left, right)
        )
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _json_values_equal(left[key], right[key]) for key in left
        )
    return bool(left == right)


def _matches_canonical_output_part(part: Any, state: dict[str, Any]) -> bool:
    try:
        part_type, value = _validate_output_part(part, label="final output content")
    except LLMUnavailableError:
        return False
    if part_type != state["type"] or value != state["value"]:
        return False
    if part_type != "output_text":
        return part == {
            "type": part_type,
            "refusal" if part_type == "refusal" else "text": value,
        }
    expected = {"type": "output_text", "text": value, "annotations": []}
    canonical_logprobs = state.get("logprobs", [])
    if canonical_logprobs:
        return _json_values_equal(part, {**expected, "logprobs": canonical_logprobs})
    return _json_values_equal(part, expected) or _json_values_equal(
        part, {**expected, "logprobs": []}
    )


def _validate_message_item(item: Any, *, done: bool) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(item, dict):
        raise LLMUnavailableError("ChatGPT returned an invalid output item")
    item_id = _stream_id(item.get("id"), label="output item ID")
    content = item.get("content")
    valid_statuses = {"completed", "incomplete"} if done else {"in_progress"}
    if (
        item.get("type") != "message"
        or item.get("role") != "assistant"
        or not isinstance(content, list)
        or item.get("status") not in valid_statuses
        or not set(item).issubset(
            {"id", "type", "role", "content", "status", "phase"}
        )
        or item.get("phase") not in {None, "commentary", "final_answer"}
        or (not done and content != [])
    ):
        raise LLMUnavailableError("ChatGPT returned an invalid output item")
    for part in content:
        _validate_output_part(part, label="output item content")
    return item_id, content


def _validate_reasoning_item(
    item: Any, *, done: bool
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]] | None, str | None]:
    if (
        not isinstance(item, dict)
        or item.get("type") != "reasoning"
        or not isinstance(item.get("summary"), list)
        or not set(item).issubset(
            {"id", "type", "summary", "content", "encrypted_content", "status"}
        )
    ):
        raise LLMUnavailableError("ChatGPT returned an invalid output item")
    if any(
        not isinstance(part, dict)
        or part.get("type") != "summary_text"
        or not isinstance(part.get("text"), str)
        for part in item["summary"]
    ):
        raise LLMUnavailableError("ChatGPT returned an invalid output item")
    content = item.get("content")
    if content is not None and (
        not isinstance(content, list)
        or any(
            not isinstance(part, dict)
            or part.get("type") != "reasoning_text"
            or not isinstance(part.get("text"), str)
            for part in content
        )
    ):
        raise LLMUnavailableError("ChatGPT returned an invalid output item")
    encrypted = item.get("encrypted_content")
    if encrypted is not None and not isinstance(encrypted, str):
        raise LLMUnavailableError("ChatGPT returned an invalid output item")
    status = item.get("status")
    valid_statuses = {"completed", "incomplete"} if done else {"in_progress"}
    if status is not None and status not in valid_statuses:
        raise LLMUnavailableError("ChatGPT returned an invalid output item")
    if not done and (item["summary"] != [] or (content is not None and content != [])):
        raise LLMUnavailableError("ChatGPT returned an invalid output item")
    try:
        for part in item["summary"]:
            if "\x00" in part["text"]:
                raise LLMUnavailableError("ChatGPT returned an invalid output item")
            part["text"].encode()
        for part in content or []:
            if "\x00" in part["text"]:
                raise LLMUnavailableError("ChatGPT returned an invalid output item")
            part["text"].encode()
        if encrypted is not None:
            if "\x00" in encrypted:
                raise LLMUnavailableError("ChatGPT returned an invalid output item")
            encrypted.encode()
    except UnicodeEncodeError:
        raise LLMUnavailableError("ChatGPT returned an invalid output item") from None
    return (
        _stream_id(item.get("id"), label="output item ID"),
        item["summary"],
        content,
        encrypted,
    )


# Complete event surface consumed by this adapter.  Lifecycle and container
# events are output-silent but still schema- and state-validated.  Every other
# Responses event (audio, hosted tools, annotations, future additions) fails
# closed because this provider advertises only text and namespaced functions.
_SUPPORTED_RESPONSE_EVENTS = frozenset(
    {
        "response.created",
        "response.queued",
        "response.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.content_part.added",
        "response.content_part.done",
        "response.output_text.delta",
        "response.output_text.done",
        "response.refusal.delta",
        "response.refusal.done",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.completed",
        "response.failed",
        "response.incomplete",
        "error",
    }
)


def _validate_stream_event_shape(event_type: str, event: dict[str, Any]) -> None:
    """Validate official required fields before parser state is changed.

    The SIWC preview omits ``logprobs`` and much of the embedded Response
    object, so those documented preview omissions stay optional.  Fields that
    the adapter actually consumes remain mandatory and strictly typed.
    """
    if event_type not in _SUPPORTED_RESPONSE_EVENTS:
        raise LLMUnavailableError("ChatGPT returned an unknown event type")
    if event_type in {
        "response.created",
        "response.queued",
        "response.in_progress",
        "response.completed",
        "response.failed",
        "response.incomplete",
    }:
        response = event.get("response")
        if not isinstance(response, dict):
            raise LLMUnavailableError("ChatGPT returned an invalid lifecycle event")
        response_id = response.get("id")
        try:
            _stream_id(response_id, label="response ID")
        except LLMUnavailableError:
            if event_type == "response.created":
                label = "creation event"
            elif event_type == "response.completed":
                label = "completion event"
            else:
                label = "terminal lifecycle event"
            raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
        return
    if event_type in {"response.output_item.added", "response.output_item.done"}:
        _stream_index(event.get("output_index"), label="output index")
        if not isinstance(event.get("item"), dict):
            raise LLMUnavailableError("ChatGPT returned an invalid output item")
        return
    if event_type in {"response.content_part.added", "response.content_part.done"}:
        _stream_id(event.get("item_id"), label="output item ID")
        _stream_index(event.get("output_index"), label="output index")
        _stream_index(event.get("content_index"), label="content index")
        if not isinstance(event.get("part"), dict):
            raise LLMUnavailableError("ChatGPT returned an invalid content part")
        return
    if event_type.startswith("response.output_text."):
        value_key = "delta" if event_type.endswith(".delta") else "text"
        label = "text delta" if value_key == "delta" else "text completion"
        if any(
            key not in event
            for key in ("item_id", "output_index", "content_index", value_key)
        ):
            raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
        _stream_id(event.get("item_id"), label="output item ID")
        _stream_index(event.get("output_index"), label="output index")
        _stream_index(event.get("content_index"), label="content index")
        if not isinstance(event.get(value_key), str):
            raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
        if "logprobs" in event and not isinstance(event["logprobs"], list):
            raise LLMUnavailableError("ChatGPT returned invalid text log probabilities")
        return
    if event_type.startswith("response.refusal."):
        _stream_id(event.get("item_id"), label="output item ID")
        _stream_index(event.get("output_index"), label="refusal output index")
        _stream_index(event.get("content_index"), label="refusal content index")
        value_key = "delta" if event_type.endswith(".delta") else "refusal"
        if not isinstance(event.get(value_key), str):
            raise LLMUnavailableError("ChatGPT returned an invalid refusal event")
        return
    if event_type.startswith("response.function_call_arguments."):
        _stream_id(event.get("item_id"), label="output item ID")
        _stream_index(event.get("output_index"), label="output index")
        value_key = "delta" if event_type.endswith(".delta") else "arguments"
        if not isinstance(event.get(value_key), str):
            raise LLMUnavailableError("ChatGPT returned invalid function arguments")
        return
    if event_type.startswith("response.reasoning_summary_part."):
        _stream_id(event.get("item_id"), label="output item ID")
        _stream_index(event.get("output_index"), label="output index")
        _stream_index(event.get("summary_index"), label="summary index")
        part = event.get("part")
        if (
            not isinstance(part, dict)
            or set(part) != {"type", "text"}
            or part.get("type") != "summary_text"
            or not isinstance(part.get("text"), str)
            or (
                event_type.endswith(".done")
                and event.get("status") not in {None, "incomplete"}
            )
        ):
            raise LLMUnavailableError("ChatGPT returned an invalid reasoning summary event")
        return
    if event_type.startswith("response.reasoning_summary_text."):
        _stream_id(event.get("item_id"), label="output item ID")
        _stream_index(event.get("output_index"), label="output index")
        _stream_index(event.get("summary_index"), label="summary index")
        value_key = "delta" if event_type.endswith(".delta") else "text"
        if not isinstance(event.get(value_key), str):
            raise LLMUnavailableError("ChatGPT returned an invalid reasoning summary event")
        return
    if event_type.startswith("response.reasoning_text."):
        _stream_id(event.get("item_id"), label="output item ID")
        _stream_index(event.get("output_index"), label="output index")
        _stream_index(event.get("content_index"), label="content index")
        value_key = "delta" if event_type.endswith(".delta") else "text"
        if not isinstance(event.get(value_key), str):
            raise LLMUnavailableError("ChatGPT returned an invalid reasoning content event")
        return
    if event_type == "error":
        code = event.get("code")
        message = event.get("message")
        param = event.get("param")
        if (
            "code" not in event
            or "message" not in event
            or "param" not in event
            or (code is not None and not isinstance(code, str))
            or not isinstance(message, str)
            or (param is not None and not isinstance(param, str))
        ):
            raise LLMUnavailableError("ChatGPT returned an invalid error event")


class OpenAIChatGPTProvider(LLMProvider):
    """Responses adapter whose bearer credential is leased per request."""

    name = "openai_chatgpt"
    supports_embeddings = False

    def __init__(self, *, access_token: AccessTokenContext, model: str | None = None):
        super().__init__(model=model)
        self._access_token = access_token

    async def chat_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatChunk]:
        failure: tuple[type[LLMError], str, int | None, _ErrorMetadata] | None = None
        failure_kind: str | None = None
        secret_stream: AsyncGenerator[ChatChunk, None] | None = None
        cancellation = _OperationCancellation()
        await cancellation.checkpoint()
        try:
            secret_stream = bounded_async_iterator(
                self._chat_stream_secret(
                    messages,
                    model=model,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ),
                timeout_seconds=get_agent_settings().siwc_oauth_timeout_seconds,
                timeout_error="ChatGPT request timed out",
                cleanup_error="ChatGPT request cleanup failed",
            )
            try:
                async for chunk in secret_stream:
                    yield chunk
            finally:
                await secret_stream.aclose()
        except BaseException as exc:  # noqa: BLE001 - secret boundary classification
            from app.agents.services.openai_oauth import OAuthLifecycleError

            if cancellation.contains(exc):
                failure_kind = "cancelled"
            elif _is_instance_without_hooks(exc, LLMError):
                failure = _safe_llm_failure(exc)
            elif _is_instance_without_hooks(exc, LookupError):
                failure = (
                    LLMAuthError,
                    "ChatGPT OAuth credential is unavailable",
                    None,
                    _EMPTY_ERROR_METADATA.copy(),
                )
            elif _is_instance_without_hooks(exc, OAuthLifecycleError):
                retryable = False
                try:
                    dictionary = _EXCEPTION_DICTIONARY.__get__(
                        exc, type(exc)
                    )
                    retryable = (
                        type(dictionary) is dict
                        and dict.get(dictionary, "retryable") is True
                    )
                except BaseException:
                    pass
                failure = (
                    LLMUnavailableError if retryable else LLMAuthError,
                    "ChatGPT OAuth credential is unavailable",
                    None,
                    _EMPTY_ERROR_METADATA.copy(),
                )
            elif _is_instance_without_hooks(exc, _SecretBoundaryFailure):
                failure = (
                    LLMUnavailableError,
                    "ChatGPT request failed safely",
                    None,
                    _EMPTY_ERROR_METADATA.copy(),
                )
            else:
                raise
            _discard_exception_secrets_safely(exc)
        finally:
            secret_stream = None
        if failure_kind == "cancelled":
            raise asyncio.CancelledError() from None
        if failure is not None:
            error_type, message, status, metadata = failure
            if error_type is LLMRateLimitError:
                raise LLMRateLimitError(message, status=status, **metadata) from None
            if error_type is LLMUnavailableError:
                raise LLMUnavailableError(message, status=status, **metadata) from None
            if error_type is LLMAuthError:
                raise LLMAuthError(message, status=status, **metadata) from None
            raise LLMNotSupportedError(message, status=status, **metadata) from None

    async def _chat_stream_secret(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        tools: Optional[list[ToolDefinition]] = None,
        temperature: float = 0.4,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[ChatChunk, None]:
        instructions, input_items = _serialize_input(messages)
        model = _bounded_identifier(model, label="model", maximum=512)
        payload: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "store": False,
            "stream": True,
        }
        if instructions is not None:
            payload["instructions"] = instructions
        serialized_tools = _serialize_tools(tools)
        if serialized_tools:
            payload["tools"] = serialized_tools
        _validate_json_graph(payload, label="request")
        _json_size_upper_bound(payload, label="request", limit=_MAX_REQUEST_BYTES)
        try:
            encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        except (TypeError, ValueError, RecursionError):
            raise LLMNotSupportedError("ChatGPT request is not valid JSON") from None
        if len(encoded) > _MAX_REQUEST_BYTES:
            raise LLMNotSupportedError("ChatGPT request exceeded the size limit")

        settings = get_agent_settings()
        operation_timeout = settings.siwc_oauth_timeout_seconds
        request_timeout = httpx.Timeout(
            settings.siwc_oauth_timeout_seconds,
            connect=settings.siwc_oauth_connect_timeout_seconds,
        )
        if operation_timeout > MAX_OAUTH_TIMEOUT_SECONDS:  # validated settings invariant
            raise LLMUnavailableError("ChatGPT timeout configuration is invalid")
        from app.agents.services.openai_oauth import OAuthLifecycleError

        bearer_acquired = False
        cancellation = _OperationCancellation()
        await cancellation.checkpoint()
        try:
            async with asyncio.timeout(operation_timeout):
                rejected_version: int | None = None
                for attempt in range(2):
                    force = attempt == 1
                    async with self._access_token(
                        force=force, rejected_version=rejected_version
                    ) as snapshot:
                        bearer_acquired = True
                        headers = {
                            "Authorization": f"Bearer {snapshot.access_token}",
                            "Content-Type": "application/json",
                            "Accept": "text/event-stream",
                            "Accept-Encoding": "identity",
                        }
                        client_manager = httpx.AsyncClient(
                            transport=_new_responses_transport(),
                            timeout=request_timeout,
                            follow_redirects=False,
                            trust_env=False,
                            auth=None,
                            cookies=None,
                            event_hooks={},
                        )
                        async with owned_async_context(
                            client_manager,
                            cleanup_error="ChatGPT client cleanup failed",
                        ) as client:
                            response_manager = client.stream(
                                "POST", _RESPONSES_URL, content=encoded, headers=headers
                            )
                            yielded_output = False
                            try:
                                async with owned_async_context(
                                    response_manager,
                                    cleanup_error="ChatGPT response cleanup failed",
                                ) as response:
                                    if 300 <= response.status_code < 400:
                                        await _read_error_body(response)
                                        raise LLMUnavailableError(
                                            "ChatGPT returned a forbidden redirect",
                                            status=response.status_code,
                                        )
                                    if response.status_code == 401 and attempt == 0:
                                        await _read_error_body(response)
                                        raise _VersionRefreshRequired()
                                    if response.status_code != 200:
                                        detail = await _read_error_body(response)
                                        _raise_status(response.status_code, detail)
                                    content_type = response.headers.get("content-type", "")
                                    media_type = content_type.split(";", 1)[0].strip().lower()
                                    if media_type != "text/event-stream":
                                        raise LLMUnavailableError(
                                            "ChatGPT returned an invalid content type",
                                            status=response.status_code,
                                        )
                                    yielded_output = False
                                    parsed_stream = bounded_async_iterator(
                                        self._parse_stream(response),
                                        timeout_seconds=operation_timeout,
                                        timeout_error="ChatGPT response timed out",
                                        cleanup_error="ChatGPT response cleanup failed",
                                    )
                                    try:
                                        async for chunk in parsed_stream:
                                            yielded_output = True
                                            yield chunk
                                    finally:
                                        await parsed_stream.aclose()
                                    return
                            except _VersionRefreshRequired:
                                if attempt == 0 and not yielded_output:
                                    rejected_version = snapshot.credential_version
                                    continue
                                raise LLMAuthError(
                                    "ChatGPT credentials were rejected", status=401
                                ) from None
        except TimeoutError:
            raise LLMUnavailableError("ChatGPT request timed out") from None
        except httpx.HTTPError as exc:
            _discard_exception_secrets_safely(exc)
            raise LLMUnavailableError("ChatGPT is unreachable") from None
        except GeneratorExit:
            raise
        except (asyncio.CancelledError, LLMError, LookupError, OAuthLifecycleError):
            raise
        except BaseException as exc:  # noqa: BLE001 - bearer may be in frame locals
            if bearer_acquired:
                cancelled = cancellation.contains(exc)
                _discard_exception_secrets_safely(exc)
                if cancelled:
                    raise asyncio.CancelledError() from None
                raise _SecretBoundaryFailure() from None
            raise

    async def _parse_stream(
        self, response: _StreamingResponse
    ) -> AsyncGenerator[ChatChunk, None]:
        response_id: str | None = None
        created = False
        completed = False
        text_bytes = 0
        calls: dict[int, dict[str, Any]] = {}
        output_items: dict[int, dict[str, Any]] = {}
        content_parts: dict[tuple[int, int], dict[str, Any]] = {}
        reasoning_summaries: dict[tuple[int, int], dict[str, Any]] = {}
        compact_content: dict[tuple[int, int], dict[str, Any]] = {}
        item_ids: dict[str, int] = {}
        lifecycle_events: set[str] = set()
        all_args_bytes = 0
        last_sequence = -1
        event_stream = bounded_async_iterator(
            _iter_sse(response),
            timeout_seconds=get_agent_settings().siwc_oauth_timeout_seconds,
            timeout_error="ChatGPT response timed out",
            cleanup_error="ChatGPT response cleanup failed",
        )
        try:
            async for event in event_stream:
                event_type = event.get("type")
                if not isinstance(event_type, str):
                    raise LLMUnavailableError("ChatGPT returned a malformed stream event")
                if event_type not in _SUPPORTED_RESPONSE_EVENTS:
                    raise LLMUnavailableError("ChatGPT returned an unknown event type")
                sequence = event.get("sequence_number")
                # OpenAI's published schema says sequence_number is used to order
                # events but does not guarantee a zero origin or contiguity.  Enforce
                # strict monotonicity/deduplication without rejecting valid gaps.
                if (
                    not isinstance(sequence, int)
                    or isinstance(sequence, bool)
                    or sequence < 0
                    or sequence <= last_sequence
                ):
                    if sequence is None:
                        raise LLMUnavailableError("ChatGPT stream event sequence was missing")
                    raise LLMUnavailableError("ChatGPT stream events were out of order")
                _validate_stream_event_shape(event_type, event)
                last_sequence = sequence
                event_response = event.get("response")
                event_id = (
                    event_response.get("id")
                    if isinstance(event_response, dict)
                    else event.get("response_id")
                )
                if event_id is not None:
                    _stream_id(event_id, label="response ID")
                    if response_id is None:
                        response_id = event_id
                    elif event_id != response_id:
                        raise LLMUnavailableError("ChatGPT stream mixed response IDs")
                if completed:
                    raise LLMUnavailableError("ChatGPT sent events after completion")
                if event_type == "response.created":
                    if (
                        created
                        or not isinstance(event_response, dict)
                        or response_id is None
                        or (
                            event_response.get("status") is not None
                            and event_response.get("status") != "in_progress"
                        )
                    ):
                        raise LLMUnavailableError("ChatGPT returned an invalid creation event")
                    created = True
                    lifecycle_events.add(event_type)
                elif event_type not in {"error", "response.failed", "response.incomplete"} and not created:
                    raise LLMUnavailableError("ChatGPT sent output before response.created")
                elif event_type in {"response.queued", "response.in_progress"}:
                    expected_status = event_type.removeprefix("response.")
                    if (
                        event_type in lifecycle_events
                        or (
                            event_type == "response.queued"
                            and (
                                "response.in_progress" in lifecycle_events
                                or output_items
                                or compact_content
                                or content_parts
                                or reasoning_summaries
                                or calls
                            )
                        )
                        or not isinstance(event_response, dict)
                        or event_response.get("id") != response_id
                        or (
                            event_response.get("status") is not None
                            and event_response.get("status") != expected_status
                        )
                    ):
                        raise LLMUnavailableError(
                            "ChatGPT returned an invalid lifecycle event"
                        )
                    lifecycle_events.add(event_type)
                elif event_type == "response.output_text.delta":
                    index = _stream_index(event.get("output_index"), label="output index")
                    content_index = _stream_index(
                        event.get("content_index"), label="content index"
                    )
                    item_id = _stream_id(event.get("item_id"), label="output item ID")
                    delta = event.get("delta")
                    logprobs = event.get("logprobs")
                    if not isinstance(delta, str) or (
                        logprobs is not None and not isinstance(logprobs, list)
                    ):
                        raise LLMUnavailableError("ChatGPT returned an invalid text delta")
                    key = (index, content_index)
                    part = content_parts.get(key) or compact_content.get(key)
                    if part is None:
                        claimed_index = item_ids.get(item_id)
                        if claimed_index is not None and claimed_index != index:
                            raise LLMUnavailableError("ChatGPT returned a duplicate output item")
                        prior_parts = [
                            state
                            for (output_index, _content_index), state in compact_content.items()
                            if output_index == index
                        ]
                        known_indices = set(output_items) | {
                            output_index for output_index, _content_index in compact_content
                        }
                        if any(output_index > index for output_index in known_indices):
                            raise LLMUnavailableError("ChatGPT output was out of order")
                        if index not in known_indices and index != len(known_indices):
                            raise LLMUnavailableError("ChatGPT output was out of order")
                        if index not in known_indices and (
                            any(not state["done"] for state in output_items.values())
                            or any(not state["done"] for state in compact_content.values())
                        ):
                            raise LLMUnavailableError("ChatGPT output was out of order")
                        if index in output_items or any(
                            output_index == index and state["item_id"] != item_id
                            for (output_index, _content_index), state in compact_content.items()
                        ) or content_index != len(prior_parts) or any(
                            not state["done"] for state in prior_parts
                        ):
                            raise LLMUnavailableError("ChatGPT content was out of order")
                        item_ids[item_id] = index
                        part = {
                            "item_id": item_id,
                            "type": "output_text",
                            "parts": [],
                            "bytes": 0,
                            "done": False,
                            "official": False,
                        }
                        compact_content[key] = part
                    if (
                        part["item_id"] != item_id
                        or part["type"] != "output_text"
                        or part["done"]
                    ):
                        raise LLMUnavailableError("ChatGPT text delta was out of order")
                    if any(
                        output_index < index and not state["done"]
                        for output_index, state in output_items.items()
                    ) or any(
                        output_index < index and not state["done"]
                        for (output_index, _content_index), state in compact_content.items()
                    ):
                        raise LLMUnavailableError("ChatGPT output was out of order")
                    if "\x00" in delta:
                        raise LLMUnavailableError("ChatGPT returned an invalid text delta")
                    try:
                        delta_bytes = len(delta.encode())
                    except UnicodeEncodeError:
                        raise LLMUnavailableError("ChatGPT returned an invalid text delta") from None
                    if delta_bytes > _MAX_TEXT_BYTES - text_bytes:
                        raise LLMUnavailableError("ChatGPT text exceeded the size limit")
                    text_bytes += delta_bytes
                    part["bytes"] = cast(int, part["bytes"]) + delta_bytes
                    cast(list[str], part["parts"]).append(delta)
                    if logprobs is not None:
                        try:
                            _validate_json_graph(logprobs, label="text log probabilities")
                        except LLMNotSupportedError:
                            raise LLMUnavailableError(
                                "ChatGPT returned invalid text log probabilities"
                            ) from None
                        cast(list[Any], part.setdefault("logprobs", [])).extend(logprobs)
                    if delta:
                        yield ChatChunk(type="text_delta", text=delta)
                elif event_type == "response.output_item.added":
                    item = event.get("item")
                    if not isinstance(item, dict):
                        raise LLMUnavailableError("ChatGPT returned an invalid output item")
                    index = _stream_index(event.get("output_index"), label="output index")
                    if index in output_items:
                        raise LLMUnavailableError("ChatGPT returned a duplicate output item")
                    if any(
                        output_index == index
                        for output_index, _content_index in compact_content
                    ):
                        raise LLMUnavailableError("ChatGPT mixed content state formats")
                    known_indices = set(output_items) | {
                        output_index for output_index, _content_index in compact_content
                    }
                    if index != len(known_indices):
                        raise LLMUnavailableError("ChatGPT output was out of order")
                    if item.get("type") == "message":
                        item_id, _content = _validate_message_item(item, done=False)
                        if item_id in item_ids:
                            raise LLMUnavailableError("ChatGPT returned a duplicate output item")
                        item_ids[item_id] = index
                        output_items[index] = {
                            "item_id": item_id,
                            "type": "message",
                            "done": False,
                            "phase_present": "phase" in item,
                            "phase": item.get("phase"),
                        }
                        continue
                    if item.get("type") == "reasoning":
                        item_id, _summary, _content, encrypted = _validate_reasoning_item(
                            item, done=False
                        )
                        if item_id in item_ids:
                            raise LLMUnavailableError("ChatGPT returned a duplicate output item")
                        item_ids[item_id] = index
                        output_items[index] = {
                            "item_id": item_id,
                            "type": "reasoning",
                            "done": False,
                            "content_present": "content" in item,
                            "encrypted_present": "encrypted_content" in item,
                            "encrypted_content": encrypted,
                        }
                        continue
                    if item.get("type") != "function_call":
                        raise LLMUnavailableError("ChatGPT returned an unsupported output item")
                    item_id = _stream_id(item.get("id"), label="output item ID")
                    call_id = _stream_id(item.get("call_id"), label="tool call ID")
                    name = _stream_id(item.get("name"), label="tool name")
                    arguments = item.get("arguments")
                    if (
                        index in calls
                        or len(name) > 256
                        or not set(item).issubset(
                            {"id", "type", "call_id", "name", "arguments", "namespace", "status"}
                        )
                        or item.get("namespace") not in {None, _TOOL_NAMESPACE}
                        or arguments != ""
                        or item.get("status") not in {None, "in_progress"}
                        or any(
                            call["item_id"] == item_id or call["call_id"] == call_id
                            for call in calls.values()
                        )
                    ):
                        raise LLMUnavailableError("ChatGPT returned an invalid function call")
                    if item_id in item_ids:
                        raise LLMUnavailableError("ChatGPT returned a duplicate output item")
                    item_ids[item_id] = index
                    calls[index] = {
                        "item_id": item_id,
                        "call_id": call_id,
                        "name": name,
                        "namespace_present": "namespace" in item,
                        "namespace": item.get("namespace"),
                        "argument_parts": [],
                        "arguments": None,
                        "args_bytes": 0,
                        "args_done": False,
                        "ended": False,
                    }
                    output_items[index] = {
                        "item_id": item_id,
                        "type": "function_call",
                        "done": False,
                    }
                    yield ChatChunk(
                        type="tool_call_start", tool_call_id=call_id, tool_name=name
                    )
                elif event_type == "response.function_call_arguments.delta":
                    index = _stream_index(event.get("output_index"), label="output index")
                    call = calls.get(index)
                    delta = event.get("delta")
                    if (
                        call is None
                        or call["ended"]
                        or call["args_done"]
                        or event.get("item_id") != call["item_id"]
                        or not isinstance(delta, str)
                    ):
                        raise LLMUnavailableError("ChatGPT function arguments were out of order")
                    if "\x00" in delta:
                        raise LLMUnavailableError("ChatGPT returned invalid function arguments")
                    try:
                        delta_bytes = len(delta.encode())
                    except UnicodeEncodeError:
                        raise LLMUnavailableError(
                            "ChatGPT returned invalid function arguments"
                        ) from None
                    if (
                        delta_bytes > _MAX_TOOL_ARGUMENT_BYTES - call["args_bytes"]
                        or delta_bytes > _MAX_ALL_TOOL_ARGUMENT_BYTES - all_args_bytes
                    ):
                        raise LLMUnavailableError("ChatGPT function arguments exceeded the size limit")
                    call["args_bytes"] += delta_bytes
                    all_args_bytes += delta_bytes
                    call["argument_parts"].append(delta)
                    if delta:
                        yield ChatChunk(
                            type="tool_call_args_delta",
                            tool_call_id=call["call_id"],
                            args_delta=delta,
                        )
                elif event_type == "response.function_call_arguments.done":
                    index = _stream_index(event.get("output_index"), label="output index")
                    call = calls.get(index)
                    arguments = event.get("arguments")
                    if (
                        call is None
                        or call["ended"]
                        or call["args_done"]
                        or event.get("item_id") != call["item_id"]
                        or not isinstance(arguments, str)
                    ):
                        raise LLMUnavailableError("ChatGPT function arguments were out of order")
                    assembled_arguments = "".join(call["argument_parts"])
                    if arguments != assembled_arguments:
                        raise LLMUnavailableError("ChatGPT function arguments were out of order")
                    try:
                        parsed_arguments = _strict_json(arguments.encode())
                    except (UnicodeEncodeError, json.JSONDecodeError, ValueError, RecursionError):
                        raise LLMUnavailableError("ChatGPT returned malformed function arguments") from None
                    if not isinstance(parsed_arguments, dict):
                        raise LLMUnavailableError("ChatGPT returned malformed function arguments")
                    try:
                        _validate_json_graph(parsed_arguments, label="function arguments")
                    except LLMNotSupportedError:
                        raise LLMUnavailableError(
                            "ChatGPT returned invalid function arguments"
                        ) from None
                    call["arguments"] = assembled_arguments
                    call["argument_parts"].clear()
                    call["args_done"] = True
                elif event_type == "response.content_part.added":
                    index = _stream_index(event.get("output_index"), label="output index")
                    content_index = _stream_index(
                        event.get("content_index"), label="content index"
                    )
                    item_id = _stream_id(event.get("item_id"), label="output item ID")
                    state = output_items.get(index)
                    part_type, value = _validate_output_part(
                        event.get("part"), label="content part"
                    )
                    added_part = cast(dict[str, Any], event["part"])
                    key = (index, content_index)
                    prior_parts = [
                        part_state
                        for (output_index, _content_index), part_state in content_parts.items()
                        if output_index == index
                    ]
                    if (
                        state is None
                        or state["type"] not in {"message", "reasoning"}
                        or (state["type"] == "message" and part_type == "reasoning_text")
                        or (state["type"] == "reasoning" and part_type != "reasoning_text")
                        or state["done"]
                        or state["item_id"] != item_id
                        or key in content_parts
                        or value != ""
                        or added_part.get("logprobs") not in (None, [])
                        or content_index != len(prior_parts)
                        or any(not part_state["content_done"] for part_state in prior_parts)
                    ):
                        raise LLMUnavailableError("ChatGPT content part was out of order")
                    content_parts[key] = {
                        "item_id": item_id,
                        "type": part_type,
                        "parts": [],
                        "bytes": 0,
                        "done": False,
                        "content_done": False,
                        "official": True,
                        "value": None,
                        "logprobs": [],
                    }
                elif event_type in {"response.output_text.done", "response.refusal.done"}:
                    expected_type = (
                        "output_text"
                        if event_type == "response.output_text.done"
                        else "refusal"
                    )
                    value_key = "text" if expected_type == "output_text" else "refusal"
                    label = "text completion" if expected_type == "output_text" else "refusal"
                    if any(
                        key not in event
                        for key in ("item_id", "output_index", "content_index", value_key)
                    ):
                        raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
                    index = _stream_index(event.get("output_index"), label="output index")
                    content_index = _stream_index(
                        event.get("content_index"), label="content index"
                    )
                    item_id = _stream_id(event.get("item_id"), label="output item ID")
                    value = event.get(value_key)
                    logprobs = event.get("logprobs")
                    part = content_parts.get((index, content_index)) or compact_content.get(
                        (index, content_index)
                    )
                    if (
                        not isinstance(value, str)
                        or (logprobs is not None and not isinstance(logprobs, list))
                        or part is None
                        or part["item_id"] != item_id
                        or part["type"] != expected_type
                        or part["done"]
                        or (
                            expected_type == "output_text"
                            and (
                                (
                                    bool(part.get("logprobs", []))
                                    and "logprobs" not in event
                                )
                                or (
                                    "logprobs" in event
                                    and not _json_values_equal(
                                        logprobs, part.get("logprobs", [])
                                    )
                                )
                            )
                        )
                    ):
                        raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
                    assembled = "".join(cast(list[str], part["parts"]))
                    if value != assembled:
                        raise LLMUnavailableError(f"ChatGPT returned an invalid {label}")
                    part["value"] = value
                    cast(list[str], part["parts"]).clear()
                    part["done"] = True
                elif event_type == "response.refusal.delta":
                    index = _stream_index(event.get("output_index"), label="refusal output index")
                    content_index = _stream_index(
                        event.get("content_index"), label="refusal content index"
                    )
                    item_id = _stream_id(event.get("item_id"), label="output item ID")
                    delta = event.get("delta")
                    part = content_parts.get((index, content_index))
                    if (
                        not isinstance(delta, str)
                        or part is None
                        or part["item_id"] != item_id
                        or part["type"] != "refusal"
                        or part["done"]
                    ):
                        raise LLMUnavailableError("ChatGPT returned an invalid refusal event")
                    if any(
                        output_index < index and not state["done"]
                        for output_index, state in output_items.items()
                    ) or any(
                        output_index < index and not state["done"]
                        for (output_index, _content_index), state in compact_content.items()
                    ):
                        raise LLMUnavailableError("ChatGPT output was out of order")
                    if "\x00" in delta:
                        raise LLMUnavailableError("ChatGPT returned an invalid refusal event")
                    try:
                        delta_bytes = len(delta.encode())
                    except UnicodeEncodeError:
                        raise LLMUnavailableError(
                            "ChatGPT returned an invalid refusal event"
                        ) from None
                    if delta_bytes > _MAX_TEXT_BYTES - text_bytes:
                        raise LLMUnavailableError("ChatGPT text exceeded the size limit")
                    text_bytes += delta_bytes
                    part["bytes"] = cast(int, part["bytes"]) + delta_bytes
                    cast(list[str], part["parts"]).append(delta)
                    if delta:
                        yield ChatChunk(type="text_delta", text=delta)
                elif event_type == "response.reasoning_summary_part.added":
                    index = _stream_index(event.get("output_index"), label="output index")
                    summary_index = _stream_index(
                        event.get("summary_index"), label="summary index"
                    )
                    item_id = _stream_id(event.get("item_id"), label="output item ID")
                    state = output_items.get(index)
                    key = (index, summary_index)
                    prior = [
                        summary
                        for (output_index, _summary_index), summary in reasoning_summaries.items()
                        if output_index == index
                    ]
                    if (
                        state is None
                        or state["type"] != "reasoning"
                        or state["done"]
                        or state["item_id"] != item_id
                        or key in reasoning_summaries
                        or summary_index != len(prior)
                        or any(not summary["part_done"] for summary in prior)
                        or event.get("part") != {"type": "summary_text", "text": ""}
                    ):
                        raise LLMUnavailableError("ChatGPT reasoning summary was out of order")
                    reasoning_summaries[key] = {
                        "item_id": item_id,
                        "parts": [],
                        "bytes": 0,
                        "value": None,
                        "text_done": False,
                        "part_done": False,
                    }
                elif event_type == "response.reasoning_summary_text.delta":
                    index = _stream_index(event.get("output_index"), label="output index")
                    summary_index = _stream_index(
                        event.get("summary_index"), label="summary index"
                    )
                    item_id = _stream_id(event.get("item_id"), label="output item ID")
                    summary = reasoning_summaries.get((index, summary_index))
                    delta = event.get("delta")
                    if (
                        summary is None
                        or summary["item_id"] != item_id
                        or summary["text_done"]
                        or not isinstance(delta, str)
                        or "\x00" in delta
                    ):
                        raise LLMUnavailableError("ChatGPT reasoning summary was out of order")
                    try:
                        delta_bytes = len(delta.encode())
                    except UnicodeEncodeError:
                        raise LLMUnavailableError(
                            "ChatGPT returned an invalid reasoning summary event"
                        ) from None
                    if delta_bytes > _MAX_TEXT_BYTES - text_bytes:
                        raise LLMUnavailableError("ChatGPT text exceeded the size limit")
                    text_bytes += delta_bytes
                    summary["bytes"] += delta_bytes
                    summary["parts"].append(delta)
                elif event_type == "response.reasoning_summary_text.done":
                    index = _stream_index(event.get("output_index"), label="output index")
                    summary_index = _stream_index(
                        event.get("summary_index"), label="summary index"
                    )
                    item_id = _stream_id(event.get("item_id"), label="output item ID")
                    summary = reasoning_summaries.get((index, summary_index))
                    text = event.get("text")
                    if (
                        summary is None
                        or summary["item_id"] != item_id
                        or summary["text_done"]
                        or not isinstance(text, str)
                        or text != "".join(summary["parts"])
                    ):
                        raise LLMUnavailableError("ChatGPT returned an invalid reasoning summary event")
                    summary["parts"].clear()
                    summary["value"] = text
                    summary["text_done"] = True
                elif event_type == "response.reasoning_summary_part.done":
                    index = _stream_index(event.get("output_index"), label="output index")
                    summary_index = _stream_index(
                        event.get("summary_index"), label="summary index"
                    )
                    item_id = _stream_id(event.get("item_id"), label="output item ID")
                    summary = reasoning_summaries.get((index, summary_index))
                    if (
                        summary is None
                        or summary["item_id"] != item_id
                        or not summary["text_done"]
                        or summary["part_done"]
                        or event.get("status") is not None
                        or event.get("part")
                        != {"type": "summary_text", "text": summary["value"]}
                    ):
                        raise LLMUnavailableError("ChatGPT reasoning summary was out of order")
                    summary["part_done"] = True
                elif event_type in {
                    "response.reasoning_text.delta",
                    "response.reasoning_text.done",
                }:
                    index = _stream_index(event.get("output_index"), label="output index")
                    content_index = _stream_index(
                        event.get("content_index"), label="content index"
                    )
                    item_id = _stream_id(event.get("item_id"), label="output item ID")
                    part = content_parts.get((index, content_index))
                    value_key = "delta" if event_type.endswith(".delta") else "text"
                    value = event.get(value_key)
                    if (
                        part is None
                        or part["item_id"] != item_id
                        or part["type"] != "reasoning_text"
                        or part["done"]
                        or not isinstance(value, str)
                        or "\x00" in value
                    ):
                        raise LLMUnavailableError("ChatGPT reasoning content was out of order")
                    if event_type.endswith(".delta"):
                        try:
                            delta_bytes = len(value.encode())
                        except UnicodeEncodeError:
                            raise LLMUnavailableError(
                                "ChatGPT returned an invalid reasoning content event"
                            ) from None
                        if delta_bytes > _MAX_TEXT_BYTES - text_bytes:
                            raise LLMUnavailableError("ChatGPT text exceeded the size limit")
                        text_bytes += delta_bytes
                        part["bytes"] += delta_bytes
                        part["parts"].append(value)
                    else:
                        if value != "".join(part["parts"]):
                            raise LLMUnavailableError(
                                "ChatGPT returned an invalid reasoning content event"
                            )
                        part["parts"].clear()
                        part["value"] = value
                        part["done"] = True
                elif event_type == "response.content_part.done":
                    index = _stream_index(event.get("output_index"), label="output index")
                    content_index = _stream_index(
                        event.get("content_index"), label="content index"
                    )
                    item_id = _stream_id(event.get("item_id"), label="output item ID")
                    part_type, value = _validate_output_part(
                        event.get("part"), label="content part"
                    )
                    part = content_parts.get((index, content_index))
                    if (
                        part is None
                        or part["item_id"] != item_id
                        or part["type"] != part_type
                        or not part["done"]
                        or part["content_done"]
                        or part["value"] != value
                        or not _matches_canonical_output_part(event.get("part"), part)
                    ):
                        raise LLMUnavailableError("ChatGPT content part was out of order")
                    part["content_done"] = True
                    part["final_part"] = event["part"]
                elif event_type == "response.output_item.done":
                    item = event.get("item")
                    index = _stream_index(event.get("output_index"), label="output index")
                    state = output_items.get(index)
                    if not isinstance(item, dict):
                        raise LLMUnavailableError("ChatGPT returned an invalid output item")
                    if item.get("type") == "message":
                        item_id, content = _validate_message_item(item, done=True)
                        item_parts = [
                            part_state
                            for (output_index, _content_index), part_state in sorted(
                                content_parts.items()
                            )
                            if output_index == index
                        ]
                        if (
                            state is None
                            or state["type"] != "message"
                            or state["done"]
                            or state["item_id"] != item_id
                            or state["phase_present"] != ("phase" in item)
                            or state["phase"] != item.get("phase")
                            or any(
                                not part_state["done"] or not part_state["content_done"]
                                for part_state in item_parts
                            )
                            or not _json_values_equal(
                                content,
                                [part_state["final_part"] for part_state in item_parts],
                            )
                        ):
                            raise LLMUnavailableError(
                                "ChatGPT output item completion was invalid"
                            )
                        state["done"] = True
                        state["final_item"] = item
                        continue
                    if item.get("type") == "reasoning":
                        item_id, summary, content, encrypted = _validate_reasoning_item(
                            item, done=True
                        )
                        item_summaries = [
                            summary_state
                            for (output_index, _summary_index), summary_state in sorted(
                                reasoning_summaries.items()
                            )
                            if output_index == index
                        ]
                        item_parts = [
                            part_state
                            for (output_index, _content_index), part_state in sorted(
                                content_parts.items()
                            )
                            if output_index == index
                        ]
                        expected_summary = [
                            {"type": "summary_text", "text": summary_state["value"]}
                            for summary_state in item_summaries
                        ]
                        expected_content = [
                            part_state["final_part"] for part_state in item_parts
                        ]
                        added_encrypted = (
                            state.get("encrypted_content") if state is not None else None
                        )
                        encrypted_valid = (
                            added_encrypted in {None, ""}
                            or encrypted == added_encrypted
                        )
                        content_valid = (
                            content == expected_content
                            if item_parts or (state is not None and state["content_present"])
                            else content is None and "content" not in item
                        )
                        if (
                            state is None
                            or state["type"] != "reasoning"
                            or state["done"]
                            or state["item_id"] != item_id
                            or any(not summary_state["part_done"] for summary_state in item_summaries)
                            or any(
                                not part_state["done"] or not part_state["content_done"]
                                for part_state in item_parts
                            )
                            or summary != expected_summary
                            or not content_valid
                            or not encrypted_valid
                            or (
                                state["encrypted_present"]
                                and "encrypted_content" not in item
                            )
                        ):
                            raise LLMUnavailableError(
                                "ChatGPT output item completion was invalid"
                            )
                        state["done"] = True
                        state["final_item"] = item
                        continue
                    if item.get("type") != "function_call":
                        raise LLMUnavailableError("ChatGPT returned an unsupported output item")
                    call = calls.get(index)
                    if (
                        call is None
                        or state is None
                        or state["type"] != "function_call"
                        or state["done"]
                        or call["ended"]
                        or not call["args_done"]
                        or item.get("id") != call["item_id"]
                        or item.get("call_id") != call["call_id"]
                        or item.get("name") != call["name"]
                        or not set(item).issubset(
                            {"id", "type", "call_id", "name", "arguments", "namespace", "status"}
                        )
                        or ("namespace" in item) != call["namespace_present"]
                        or item.get("namespace") != call["namespace"]
                        or item.get("arguments") != call["arguments"]
                        or item.get("status") not in {None, "completed"}
                    ):
                        raise LLMUnavailableError("ChatGPT function call completion was invalid")
                    call["ended"] = True
                    state["done"] = True
                    state["final_item"] = item
                    yield ChatChunk(type="tool_call_end", tool_call_id=call["call_id"])
                elif event_type == "response.completed":
                    if (
                        not isinstance(event_response, dict)
                        or event_response.get("id") != response_id
                        or event_response.get("status") != "completed"
                    ):
                        raise LLMUnavailableError("ChatGPT returned an invalid completion event")
                    if any(not call["ended"] for call in calls.values()):
                        raise LLMUnavailableError("ChatGPT completed with an unfinished function call")
                    if any(not state["done"] for state in output_items.values()):
                        raise LLMUnavailableError("ChatGPT completed with an unfinished output item")
                    for state in output_items.values():
                        item = state.get("final_item")
                        status = item.get("status") if isinstance(item, dict) else None
                        if (
                            (state["type"] == "message" and status != "completed")
                            or (
                                state["type"] in {"function_call", "reasoning"}
                                and status not in {None, "completed"}
                            )
                        ):
                            raise LLMUnavailableError(
                                "ChatGPT completed with an invalid final output status"
                            )
                    if any(not state["done"] for state in compact_content.values()):
                        raise LLMUnavailableError("ChatGPT completed with unfinished content")
                    output = event_response.get("output")
                    compact_indices = {
                        output_index for output_index, _content_index in compact_content
                    }
                    streamed_indices = set(output_items) | compact_indices
                    valid_output = (
                        type(output) is list
                        and len(output) <= 128
                        and len(output) == len(streamed_indices)
                        and streamed_indices == set(range(len(output)))
                    )
                    if valid_output:
                        for index, item in enumerate(cast(list[Any], output)):
                            if index in output_items:
                                state = output_items[index]
                                if not _json_values_equal(state.get("final_item"), item):
                                    valid_output = False
                                    break
                                status = item.get("status") if isinstance(item, dict) else None
                                if (
                                    (state["type"] == "message" and status != "completed")
                                    or (
                                        state["type"] in {"function_call", "reasoning"}
                                        and status not in {None, "completed"}
                                    )
                                ):
                                    valid_output = False
                                    break
                                continue
                            try:
                                item_id, content = _validate_message_item(item, done=True)
                            except LLMUnavailableError:
                                valid_output = False
                                break
                            parts = [
                                part
                                for (output_index, _content_index), part in sorted(
                                    compact_content.items()
                                )
                                if output_index == index
                            ]
                            if (
                                not parts
                                or any(part["item_id"] != item_id for part in parts)
                                or item.get("status") != "completed"
                                or len(content) != len(parts)
                                or any(
                                    not _matches_canonical_output_part(content_part, part)
                                    for content_part, part in zip(content, parts)
                                )
                            ):
                                valid_output = False
                                break
                    if not valid_output:
                        raise LLMUnavailableError(
                            "ChatGPT final output did not match streamed items"
                        )
                    usage = event_response.get("usage")
                    if usage is not None:
                        if not isinstance(usage, dict):
                            raise LLMUnavailableError("ChatGPT returned invalid usage")
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)
                        total_tokens = usage.get("total_tokens")
                        if (
                            not isinstance(input_tokens, int)
                            or "input_tokens" not in usage
                            or isinstance(input_tokens, bool)
                            or input_tokens < 0
                            or input_tokens > _MAX_USAGE_TOKENS
                            or not isinstance(output_tokens, int)
                            or "output_tokens" not in usage
                            or isinstance(output_tokens, bool)
                            or output_tokens < 0
                            or output_tokens > _MAX_USAGE_TOKENS
                            or (
                                total_tokens is not None
                                and (
                                    not isinstance(total_tokens, int)
                                    or isinstance(total_tokens, bool)
                                    or total_tokens != input_tokens + output_tokens
                                )
                            )
                        ):
                            raise LLMUnavailableError("ChatGPT returned invalid usage")
                        yield ChatChunk(
                            type="usage",
                            usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
                        )
                    completed = True
                    yield ChatChunk(
                        type="finish", finish_reason="tool_calls" if calls else "stop"
                    )
                elif event_type in {"response.failed", "response.incomplete"}:
                    expected_status = event_type.removeprefix("response.")
                    if (
                        not isinstance(event_response, dict)
                        or event_response.get("id") != response_id
                        or event_response.get("status") != expected_status
                    ):
                        raise LLMUnavailableError(
                            "ChatGPT returned an invalid terminal lifecycle event"
                        )
                    _raise_terminal_event(event_type, event)
                elif event_type == "error":
                    code = event.get("code")
                    message = event.get("message")
                    param = event.get("param")
                    if (
                        "code" not in event
                        or "message" not in event
                        or "param" not in event
                        or (code is not None and not isinstance(code, str))
                        or not isinstance(message, str)
                        or (param is not None and not isinstance(param, str))
                    ):
                        raise LLMUnavailableError("ChatGPT returned an invalid error event")
                    _raise_terminal_event(event_type, event)
                else:
                    raise LLMUnavailableError("ChatGPT returned an unknown event type")
            if not completed:
                raise LLMUnavailableError("ChatGPT stream ended before response.completed")
        finally:
            await event_stream.aclose()

    async def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        raise LLMNotSupportedError("ChatGPT plan credentials cannot be used for embeddings")
