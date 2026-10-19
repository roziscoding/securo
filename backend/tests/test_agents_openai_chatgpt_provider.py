"""OpenAI ChatGPT-plan Responses API provider contract tests."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import types
from typing import Any, cast
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import httpx

from app.agents.providers.base import (
    ChatChunk,
    ChatMessage,
    LLMAuthError,
    LLMNotSupportedError,
    LLMRateLimitError,
    LLMUnavailableError,
    ToolCall,
    ToolDefinition,
)
from app.agents.providers.openai_chatgpt import (
    AccessTokenSnapshot,
    OpenAIChatGPTProvider,
    _iter_sse,
    _new_responses_transport,
    _read_error_body,
    _serialize_input,
    _serialize_tools,
)

_EXCEPTION_ARGS = BaseException.__dict__["args"]
_EXCEPTION_CAUSE = BaseException.__dict__["__cause__"]
_EXCEPTION_CONTEXT = BaseException.__dict__["__context__"]
_EXCEPTION_TRACEBACK = BaseException.__dict__["__traceback__"]
_EXCEPTION_DICTIONARY = BaseException.__dict__["__dict__"]


async def test_public_stream_close_joins_secret_stream_cleanup() -> None:
    cleaned = False

    async def secret_stream(*_args, **_kwargs):
        nonlocal cleaned
        try:
            yield ChatChunk(type="text_delta", text="one")
            await asyncio.Event().wait()
        finally:
            cleaned = True

    provider = OpenAIChatGPTProvider(access_token=cast(Any, object()))
    with patch.object(provider, "_chat_stream_secret", new=secret_stream):
        stream = provider.chat_stream([], model="gpt-test")
        assert (await anext(stream)).text == "one"
        await cast(Any, stream).aclose()

    assert cleaned is True


async def test_sse_close_joins_raw_response_iterator_cleanup() -> None:
    cleaned = False

    class Response:
        def aiter_raw(self):
            async def raw_stream():
                nonlocal cleaned
                try:
                    yield b'data: {"type":"test"}\n\n'
                    await asyncio.Event().wait()
                finally:
                    cleaned = True

            return raw_stream()

    stream = _iter_sse(cast(Any, Response()))
    assert (await anext(stream))["type"] == "test"
    await stream.aclose()

    assert cleaned is True


async def test_sse_remote_read_stall_uses_transport_layer_deadline(monkeypatch) -> None:
    from app.agents.providers import openai_chatgpt

    cleaned = False

    class Response:
        def aiter_raw(self):
            async def raw_stream():
                nonlocal cleaned
                try:
                    await asyncio.Event().wait()
                    yield b""  # pragma: no cover
                finally:
                    cleaned = True

            return raw_stream()

    monkeypatch.setattr(
        openai_chatgpt,
        "get_agent_settings",
        lambda: SimpleNamespace(siwc_oauth_timeout_seconds=0.01),
    )

    with pytest.raises(LLMUnavailableError, match="response timed out"):
        _ = [event async for event in _iter_sse(cast(Any, Response()))]
    assert cleaned is True


async def test_error_body_overflow_joins_raw_iterator_cleanup() -> None:
    cleaned = False

    class Response:
        status_code = 500
        headers = {}

        def aiter_raw(self):
            async def raw_stream():
                nonlocal cleaned
                try:
                    yield b"too large"
                finally:
                    cleaned = True

            return raw_stream()

    with (
        patch("app.agents.providers.openai_chatgpt._MAX_ERROR_BYTES", 1),
        pytest.raises(LLMUnavailableError),
    ):
        await _read_error_body(cast(Any, Response()))

    assert cleaned is True


def _retained_provider_secret_text(*roots: object) -> str:
    """Render exception graphs and provider traceback locals for leak assertions."""
    seen: set[int] = set()
    rendered: list[str] = []

    def visit(value: object, depth: int = 0) -> None:
        if value is None or depth > 8 or id(value) in seen:
            return
        seen.add(id(value))
        if type(value) is str:
            rendered.append(str.__str__(value))
            return
        if type(value) is bytes:
            rendered.append(bytes.decode(value, errors="replace"))
            return
        if type(value) is bytearray:
            rendered.append(bytes(value).decode(errors="replace"))
            return
        if isinstance(value, BaseException):
            try:
                visit(_EXCEPTION_ARGS.__get__(value, type(value)), depth + 1)
                visit(_EXCEPTION_CAUSE.__get__(value, type(value)), depth + 1)
                visit(_EXCEPTION_CONTEXT.__get__(value, type(value)), depth + 1)
                traceback = _EXCEPTION_TRACEBACK.__get__(value, type(value))
            except BaseException:
                traceback = None
            while traceback is not None:
                if traceback.tb_frame.f_code.co_filename.endswith("openai_chatgpt.py"):
                    visit(traceback.tb_frame.f_locals, depth + 1)
                traceback = traceback.tb_next
            try:
                visit(_EXCEPTION_DICTIONARY.__get__(value, type(value)), depth + 1)
            except BaseException:
                pass
            for cls in type.__getattribute__(type(value), "__mro__"):
                for descriptor in type.__getattribute__(cls, "__dict__").values():
                    if not isinstance(descriptor, types.MemberDescriptorType):
                        continue
                    try:
                        visit(descriptor.__get__(value, type(value)), depth + 1)
                    except BaseException:
                        pass
            return
        if isinstance(value, dict):
            for key, item in dict.items(value):
                visit(key, depth + 1)
                visit(item, depth + 1)
            return
        if isinstance(value, list):
            for item in list.__iter__(value):
                visit(item, depth + 1)
            return
        if isinstance(value, tuple):
            for item in tuple.__iter__(value):
                visit(item, depth + 1)
            return
        if isinstance(value, set):
            for item in set.__iter__(value):
                visit(item, depth + 1)
            return
        if isinstance(value, frozenset):
            for item in frozenset.__iter__(value):
                visit(item, depth + 1)
            return
        if isinstance(value, types.TracebackType):
            traceback = value
            while traceback is not None:
                if traceback.tb_frame.f_code.co_filename.endswith("openai_chatgpt.py"):
                    visit(traceback.tb_frame.f_locals, depth + 1)
                traceback = traceback.tb_next
            return
        dictionary = getattr(value, "__dict__", None)
        if isinstance(dictionary, dict):
            visit(dictionary, depth + 1)
        for cls in type(value).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in {"__dict__", "__weakref__"}:
                    continue
                try:
                    visit(getattr(value, slot), depth + 1)
                except AttributeError:
                    pass

    for root in roots:
        visit(root)
    return "\n".join(rendered)


def test_responses_transport_uses_only_validated_explicit_egress_settings() -> None:
    settings = SimpleNamespace(
        siwc_oauth_ca_bundle="/validated/ca.pem",
        siwc_oauth_proxy_url="https://proxy.example",
    )
    ssl_context = object()
    transport = object()
    with (
        patch("app.agents.providers.openai_chatgpt.get_agent_settings", return_value=settings),
        patch(
            "app.agents.providers.openai_chatgpt.ssl.create_default_context",
            return_value=ssl_context,
        ) as create_context,
        patch(
            "app.agents.providers.openai_chatgpt.httpx.AsyncHTTPTransport",
            return_value=transport,
        ) as transport_factory,
    ):
        assert _new_responses_transport() is transport

    create_context.assert_called_once_with(cafile="/validated/ca.pem")
    transport_factory.assert_called_once_with(
        verify=ssl_context,
        proxy="https://proxy.example",
        trust_env=False,
    )


def test_responses_serialization_replays_full_tool_history() -> None:
    messages = [
        ChatMessage(role="system", content="first"),
        ChatMessage(role="system", content="second"),
        ChatMessage(role="user", content="question"),
        ChatMessage(
            role="assistant",
            content="checking",
            tool_calls=[ToolCall(id="call_1", name="lookup", arguments={"id": 7})],
        ),
        ChatMessage(role="tool", tool_call_id="call_1", name="lookup", content='{"ok":true}'),
    ]

    instructions, input_items = _serialize_input(messages)

    assert instructions == "first\n\nsecond"
    assert input_items == [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "checking"},
        {
            "type": "function_call",
            "namespace": "securo",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": json.dumps({"id": 7}, separators=(",", ":"), sort_keys=True),
        },
        {"type": "function_call_output", "call_id": "call_1", "output": '{"ok":true}'},
    ]
    assert _serialize_tools(
        [ToolDefinition(name="lookup", description="Look up", parameters={"type": "object"})]
    ) == [
        {
            "type": "namespace",
            "name": "securo",
            "description": "Securo client-side tools",
            "tools": [
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up",
                    "parameters": {"type": "object"},
                }
            ],
        }
    ]


async def test_responses_streams_text_usage_and_completes_only_on_completed() -> None:
    auth_calls: list[tuple[bool, int | None]] = []

    @asynccontextmanager
    async def access(*, force: bool = False, rejected_version: int | None = None):
        auth_calls.append((force, rejected_version))
        yield AccessTokenSnapshot(access_token="access-secret", credential_version=3)

    raw_chunks = [
        b"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\"}}\n\nevent: response.output_text.delta\nda",
        b"ta: {\"type\":\"response.output_text.delta\",\"sequence_number\":1,\"response_id\":\"resp_1\",\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\",\"delta\":\"Hello \"}\n\n",
        b"data: {\"type\":\"response.output_text.delta\",\"sequence_number\":2,\"response_id\":\"resp_1\",\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\",\"delta\":\"world\"}\n\n",
        b"data: {\"type\":\"response.output_text.done\",\"sequence_number\":3,\"response_id\":\"resp_1\",\"output_index\":0,\"content_index\":0,\"item_id\":\"msg_1\",\"text\":\"Hello world\"}\n\n",
        b"data: {\"type\":\"response.completed\",\"sequence_number\":4,\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello world\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":12,\"output_tokens\":4}}}\n\n",
    ]
    captured: dict = {}

    class Response:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_raw(self):
            for chunk in raw_chunks:
                yield chunk

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, method, url, **kwargs):
            captured["request"] = (method, url, kwargs)
            return Response()

    provider = OpenAIChatGPTProvider(access_token=access)
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        chunks = [
            chunk
            async for chunk in provider.chat_stream(
                [ChatMessage(role="user", content="hi")], model="gpt-test", temperature=0.9
            )
        ]

    assert [(chunk.type, chunk.text) for chunk in chunks[:2]] == [
        ("text_delta", "Hello "),
        ("text_delta", "world"),
    ]
    assert chunks[2].type == "usage"
    assert chunks[2].usage and (chunks[2].usage.input_tokens, chunks[2].usage.output_tokens) == (12, 4)
    assert (chunks[3].type, chunks[3].finish_reason) == ("finish", "stop")
    assert auth_calls == [(False, None)]
    method, url, request = captured["request"]
    assert (method, url) == ("POST", "https://api.openai.com/v1/responses")
    assert request["headers"] == {
        "Authorization": "Bearer access-secret",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Accept-Encoding": "identity",
    }
    decoded_request = json.loads(request["content"])
    assert decoded_request == {
        "model": "gpt-test",
        "input": [{"role": "user", "content": "hi"}],
        "store": False,
        "stream": True,
    }
    assert "temperature" not in decoded_request
    assert "max_output_tokens" not in decoded_request
    assert "previous_response_id" not in decoded_request
    assert captured["client"]["follow_redirects"] is False
    assert captured["client"]["trust_env"] is False
    assert captured["client"]["auth"] is None
    assert captured["client"]["cookies"] is None
    assert captured["client"]["event_hooks"] == {}
    assert captured["client"]["timeout"].connect == 5.0
    assert captured["client"]["timeout"].read == 30.0


async def test_responses_streams_multiple_fragmented_function_calls() -> None:
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {"id": "resp_tools"}},
        {
            "type": "response.output_item.added",
            "response_id": "resp_tools",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "namespace": "securo",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "one",
                "arguments": "",
                "status": "in_progress",
            },
        },
        {
            "type": "response.output_item.added",
            "response_id": "resp_tools",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "namespace": "securo",
                "id": "fc_2",
                "call_id": "call_2",
                "name": "two",
                "arguments": "",
                "status": "in_progress",
            },
        },
        {
            "type": "response.function_call_arguments.delta",
            "response_id": "resp_tools",
            "output_index": 1,
            "item_id": "fc_2",
            "delta": '{"b":',
        },
        {
            "type": "response.function_call_arguments.delta",
            "response_id": "resp_tools",
            "output_index": 0,
            "item_id": "fc_1",
            "delta": '{"a":1}',
        },
        {
            "type": "response.function_call_arguments.delta",
            "response_id": "resp_tools",
            "output_index": 1,
            "item_id": "fc_2",
            "delta": "2}",
        },
        {
            "type": "response.function_call_arguments.done",
            "response_id": "resp_tools",
            "output_index": 0,
            "item_id": "fc_1",
            "arguments": '{"a":1}',
        },
        {
            "type": "response.output_item.done",
            "response_id": "resp_tools",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "namespace": "securo",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "one",
                "arguments": '{"a":1}',
                "status": "completed",
            },
        },
        {
            "type": "response.function_call_arguments.done",
            "response_id": "resp_tools",
            "output_index": 1,
            "item_id": "fc_2",
            "arguments": '{"b":2}',
        },
        {
            "type": "response.output_item.done",
            "response_id": "resp_tools",
            "output_index": 1,
            "item": {
                "type": "function_call",
                "namespace": "securo",
                "id": "fc_2",
                "call_id": "call_2",
                "name": "two",
                "arguments": '{"b":2}',
                "status": "completed",
            },
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp_tools",
                "status": "completed",
                "usage": {"input_tokens": 21, "output_tokens": 8},
            },
        },
    ]
    events[-1]["response"]["output"] = [events[7]["item"], events[9]["item"]]
    for sequence, event in enumerate(events):
        event["sequence_number"] = sequence
    wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)

    class Response:
        status_code = 200

        async def aiter_raw(self):
            for offset in range(0, len(wire), 17):
                yield wire[offset : offset + 17]

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    provider = OpenAIChatGPTProvider(access_token=access)
    chunks = [chunk async for chunk in provider._parse_stream(Response())]

    assert [
        (chunk.type, chunk.tool_call_id, chunk.tool_name, chunk.args_delta)
        for chunk in chunks
        if chunk.type.startswith("tool_call")
    ] == [
        ("tool_call_start", "call_1", "one", None),
        ("tool_call_start", "call_2", "two", None),
        ("tool_call_args_delta", "call_2", None, '{"b":'),
        ("tool_call_args_delta", "call_1", None, '{"a":1}'),
        ("tool_call_args_delta", "call_2", None, "2}"),
        ("tool_call_end", "call_1", None, None),
        ("tool_call_end", "call_2", None, None),
    ]
    assert chunks[-1].type == "finish"
    assert chunks[-1].finish_reason == "tool_calls"


async def test_response_failed_usage_limit_maps_to_rate_limit_without_provider_message() -> None:
    event = {
        "type": "response.failed",
        "sequence_number": 0,
        "response": {
            "id": "resp_failed",
            "status": "failed",
            "error": {
                "code": "subscription_sharing_usage_limit_exceeded",
                "message": "provider detail that must not escape",
            },
        },
    }
    wire = b"data: " + json.dumps(event).encode() + b"\n\n"

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield wire

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    provider = OpenAIChatGPTProvider(access_token=access)
    try:
        chunks = [chunk async for chunk in provider._parse_stream(Response())]
    except LLMRateLimitError as exc:
        assert "provider detail" not in str(exc)
        assert exc.status == 429
    else:  # pragma: no cover - documents the required typed failure
        raise AssertionError(f"expected rate limit, got {chunks!r}")


async def test_response_failed_scope_error_preserves_forbidden_semantics() -> None:
    event = {
        "type": "response.failed",
        "sequence_number": 0,
        "response": {
            "id": "resp_scope",
            "status": "failed",
            "error": {"code": "chatpass_v2_scope_not_authorized"},
        },
    }
    wire = b"data: " + json.dumps(event).encode() + b"\n\n"

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield wire

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    provider = OpenAIChatGPTProvider(access_token=access)
    with pytest.raises(LLMAuthError) as exc_info:
        _ = [chunk async for chunk in provider._parse_stream(Response())]
    assert exc_info.value.status == 403


async def test_http_401_forces_exactly_one_version_bound_refresh_and_retry() -> None:
    auth_calls: list[tuple[bool, int | None]] = []

    @asynccontextmanager
    async def access(*, force=False, rejected_version=None):
        auth_calls.append((force, rejected_version))
        version = 3 if not force else 4
        yield AccessTokenSnapshot(access_token=f"secret-{version}", credential_version=version)

    complete = (
        b'data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_retry"}}\n\n'
        b'data: {"type":"response.completed","sequence_number":1,"response":{"id":"resp_retry",'
        b'"status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    )
    responses = [(401, b'{"error":"expired"}'), (200, complete)]
    seen_headers: list[dict[str, str]] = []

    class Response:
        headers = {"content-type": "text/event-stream"}

        def __init__(self, status, body):
            self.status_code = status
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_raw(self):
            if self.body:
                yield self.body

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, _method, _url, **kwargs):
            seen_headers.append(kwargs["headers"])
            return Response(*responses.pop(0))

    provider = OpenAIChatGPTProvider(access_token=access)
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        chunks = [
            chunk
            async for chunk in provider.chat_stream(
                [ChatMessage(role="user", content="retry")], model="gpt-test"
            )
        ]

    assert auth_calls == [(False, None), (True, 3)]
    assert [headers["Authorization"] for headers in seen_headers] == [
        "Bearer secret-3",
        "Bearer secret-4",
    ]
    assert chunks[-1].type == "finish"


async def test_second_http_401_is_terminal_without_another_refresh() -> None:
    auth_calls: list[tuple[bool, int | None]] = []
    responses = [(401, b'{"error":"expired"}'), (401, b'{"error":"rejected"}')]

    @asynccontextmanager
    async def access(*, force=False, rejected_version=None):
        auth_calls.append((force, rejected_version))
        yield AccessTokenSnapshot(
            access_token="secret", credential_version=3 if not force else 4
        )

    class Response:
        headers = {"content-type": "text/event-stream"}

        def __init__(self, status, body):
            self.status_code = status
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_raw(self):
            yield self.body

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response(*responses.pop(0))

    provider = OpenAIChatGPTProvider(access_token=access)
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        with pytest.raises(LLMAuthError):
            _ = [
                chunk
                async for chunk in provider.chat_stream(
                    [ChatMessage(role="user", content="retry once")], model="gpt-test"
                )
            ]

    assert auth_calls == [(False, None), (True, 3)]
    assert responses == []


async def test_structured_401_after_output_is_terminal_without_refresh() -> None:
    auth_calls: list[tuple[bool, int | None]] = []
    wire = (
        b'data: {"type":"response.created","sequence_number":0,"response":{"id":"partial"}}\n\n'
        b'data: {"type":"response.output_text.delta","sequence_number":1,"response_id":"partial",'
        b'"output_index":0,"content_index":0,"item_id":"message","delta":"visible"}\n\n'
        b'data: {"type":"response.failed","sequence_number":2,"response":{"id":"partial",'
        b'"status":"failed","error":{"code":"subscription_sharing_invalid_user"}}}\n\n'
    )

    @asynccontextmanager
    async def access(*, force=False, rejected_version=None):
        auth_calls.append((force, rejected_version))
        yield AccessTokenSnapshot(access_token="secret", credential_version=3)

    class Response:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_raw(self):
            yield wire

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    provider = OpenAIChatGPTProvider(access_token=access)
    chunks = []
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        with pytest.raises(LLMAuthError):
            async for chunk in provider.chat_stream(
                [ChatMessage(role="user", content="no retry after output")],
                model="gpt-test",
            ):
                chunks.append(chunk)

    assert [chunk.text for chunk in chunks] == ["visible"]
    assert auth_calls == [(False, None)]


async def test_structured_invalid_user_forces_one_version_bound_refresh() -> None:
    auth_calls: list[tuple[bool, int | None]] = []
    response_exit_calls = 0

    @asynccontextmanager
    async def access(*, force=False, rejected_version=None):
        auth_calls.append((force, rejected_version))
        yield AccessTokenSnapshot(
            access_token="secret",
            credential_version=7 if not force else 8,
        )

    failed = (
        b'data: {"type":"response.failed","sequence_number":0,"response":{"id":"failed","status":"failed",'
        b'"error":{"code":"subscription_sharing_invalid_user"}}}\n\n'
    )
    complete = (
        b'data: {"type":"response.created","sequence_number":0,"response":{"id":"ok"}}\n\n'
        b'data: {"type":"response.completed","sequence_number":1,"response":{"id":"ok","status":"completed",'
        b'"output":[],"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    )
    bodies = [failed, complete]

    class Response:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, *_args):
            nonlocal response_exit_calls
            response_exit_calls += 1
            if response_exit_calls == 1:
                assert exc_type is not None
                raise RuntimeError("response cleanup sentinel")
            return None

        async def aiter_raw(self):
            yield bodies.pop(0)

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    provider = OpenAIChatGPTProvider(access_token=access)
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        chunks = [
            chunk
            async for chunk in provider.chat_stream(
                [ChatMessage(role="user", content="retry structured auth")],
                model="gpt-test",
            )
        ]

    assert auth_calls == [(False, None), (True, 7)]
    assert response_exit_calls == 2
    assert chunks[-1].type == "finish"


async def test_http_403_does_not_refresh() -> None:
    auth_calls = 0

    @asynccontextmanager
    async def access(**_kwargs):
        nonlocal auth_calls
        auth_calls += 1
        yield AccessTokenSnapshot(access_token="secret", credential_version=3)

    class Response:
        status_code = 403
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_raw(self):
            yield b"denied"

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    provider = OpenAIChatGPTProvider(access_token=access)
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        try:
            _ = [
                chunk
                async for chunk in provider.chat_stream(
                    [ChatMessage(role="user", content="no retry")], model="gpt-test"
                )
            ]
        except LLMAuthError:
            pass
        else:  # pragma: no cover
            raise AssertionError("expected auth error")
    assert auth_calls == 1


@pytest.mark.parametrize(
    "events",
    [
        [{"type": "response.created", "response": {"id": "resp"}}],
        [
            {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 2},
            {
                "type": "response.output_text.delta",
                "response_id": "resp",
                "delta": "x",
                "sequence_number": 2,
            },
        ],
        [
            {"type": "response.created", "response": {"id": "resp_a"}},
            {"type": "response.output_text.delta", "response_id": "resp_b", "delta": "x"},
        ],
        [
            {"type": "response.created", "response": {"id": "resp"}},
            {
                "type": "response.function_call_arguments.delta",
                "response_id": "resp",
                "output_index": 0,
                "item_id": "fc",
                "delta": "{}",
            },
        ],
        [
            {
                "type": "response.completed",
                "response": {"id": "resp", "status": "completed"},
            },
            {
                "type": "response.completed",
                "response": {"id": "resp", "status": "completed"},
            },
        ],
    ],
    ids=["eof", "duplicate-sequence", "mixed-response", "arguments-before-call", "duplicate-complete"],
)
async def test_malformed_duplicate_and_out_of_order_streams_fail(events) -> None:
    wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield wire

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    provider = OpenAIChatGPTProvider(access_token=access)
    with pytest.raises(Exception, match="ChatGPT"):
        _ = [chunk async for chunk in provider._parse_stream(Response())]


async def test_responses_rejects_unsequenced_to_sequenced_mode_switch_before_duplicate_text() -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}},
        {
            "type": "response.output_text.delta",
            "response_id": "resp",
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg",
            "delta": "secret-duplicate",
        },
        {
            "type": "response.output_text.delta",
            "response_id": "resp",
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg",
            "delta": "secret-duplicate",
            "sequence_number": 1,
        },
        {
            "type": "response.completed",
            "response": {"id": "resp", "status": "completed"},
            "sequence_number": 2,
        },
    ]
    wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield wire

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    provider = OpenAIChatGPTProvider(access_token=access)
    with pytest.raises(LLMUnavailableError, match="sequence") as exc_info:
        _ = [chunk async for chunk in provider._parse_stream(Response())]
    assert "secret-duplicate" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("bad_event", "field"),
    [
        (
            {
                "type": "response.output_text.delta",
                "response_id": "resp",
                "output_index": 0,
                "content_index": 0,
                "item_id": "msg",
                "delta": {"secret": "text-delta-secret"},
                "sequence_number": 1,
            },
            "text delta",
        ),
        (
            {
                "type": "response.function_call_arguments.delta",
                "response_id": "resp",
                "output_index": 0,
                "item_id": "fc",
                "delta": {"secret": "argument-delta-secret"},
                "sequence_number": 2,
            },
            "function arguments",
        ),
    ],
    ids=["object-text-delta", "object-arguments-delta"],
)
async def test_responses_rejects_object_deltas_as_sanitized_provider_errors(
    bad_event, field
) -> None:
    events = [
        {
            "type": "response.created",
            "response": {"id": "resp"},
            "sequence_number": 0,
        }
    ]
    if bad_event["type"] == "response.function_call_arguments.delta":
        events.append(
            {
                "type": "response.output_item.added",
                "response_id": "resp",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "namespace": "securo",
                    "id": "fc",
                    "call_id": "call",
                    "name": "lookup",
                    "arguments": "",
                    "status": "in_progress",
                },
                "sequence_number": 1,
            }
        )
    events.append(bad_event)
    wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield wire

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    provider = OpenAIChatGPTProvider(access_token=access)
    with pytest.raises(LLMUnavailableError, match=field) as exc_info:
        _ = [chunk async for chunk in provider._parse_stream(Response())]
    assert "secret" not in str(exc_info.value)


@pytest.mark.parametrize("sequence", [-1, True])
async def test_responses_rejects_negative_and_boolean_sequences(sequence) -> None:
    event = {
        "type": "response.created",
        "response": {"id": "resp"},
        "sequence_number": sequence,
    }
    wire = b"data: " + json.dumps(event).encode() + b"\n\n"

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield wire

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    provider = OpenAIChatGPTProvider(access_token=access)
    with pytest.raises(LLMUnavailableError, match="order"):
        _ = [chunk async for chunk in provider._parse_stream(Response())]


async def test_responses_accepts_official_sequenced_stream_with_sse_comments() -> None:
    events = [
        {
            "type": "response.created",
            "response": {"id": "resp"},
            "sequence_number": 0,
        },
        {
            "type": "response.in_progress",
            "response": {"id": "resp"},
            "sequence_number": 1,
        },
        {
            "type": "response.output_text.delta",
            "response_id": "resp",
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg",
            "delta": "ok",
            "sequence_number": 2,
        },
        {
            "type": "response.output_text.done",
            "response_id": "resp",
            "output_index": 0,
            "content_index": 0,
            "item_id": "msg",
            "text": "ok",
            "sequence_number": 3,
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp",
                "status": "completed",
                "output": [
                    {
                        "id": "msg",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "ok",
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            "sequence_number": 4,
        },
    ]
    wire = b": keep-alive\n\n" + b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield wire

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    provider = OpenAIChatGPTProvider(access_token=access)
    chunks = [chunk async for chunk in provider._parse_stream(Response())]
    assert [(chunk.type, chunk.text) for chunk in chunks] == [
        ("text_delta", "ok"),
        ("usage", None),
        ("finish", None),
    ]


async def _parse_response_events(events: list[dict[str, Any]]):
    wire = b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield wire

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    provider = OpenAIChatGPTProvider(access_token=access)
    return [chunk async for chunk in provider._parse_stream(Response())]


def _official_text_events_with_logprobs() -> list[dict[str, Any]]:
    logprobs = [
        {
            "token": "ok",
            "bytes": [1, 107],
            "logprob": -0.125,
            "top_logprobs": [
                {"token": "okay", "bytes": [111, 107, 97, 121], "logprob": -1.5}
            ],
        }
    ]
    empty_part = {
        "type": "output_text",
        "text": "",
        "annotations": [],
        "logprobs": [],
    }
    done_part = {
        "type": "output_text",
        "text": "ok",
        "annotations": [],
        "logprobs": deepcopy(logprobs),
    }
    done_item = {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "content": [deepcopy(done_part)],
        "status": "completed",
    }
    return [
        {
            "type": "response.created",
            "response": {"id": "resp", "status": "in_progress"},
            "sequence_number": 0,
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "content": [],
                "status": "in_progress",
            },
            "sequence_number": 1,
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": empty_part,
            "sequence_number": 2,
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "ok",
            "logprobs": deepcopy(logprobs),
            "sequence_number": 3,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "text": "ok",
            "logprobs": deepcopy(logprobs),
            "sequence_number": 4,
        },
        {
            "type": "response.content_part.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": deepcopy(done_part),
            "sequence_number": 5,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": deepcopy(done_item),
            "sequence_number": 6,
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp",
                "status": "completed",
                "output": [deepcopy(done_item)],
            },
            "sequence_number": 7,
        },
    ]


async def test_responses_accepts_exact_nonempty_logprobs_through_text_lifecycle() -> None:
    chunks = await _parse_response_events(_official_text_events_with_logprobs())

    assert [(chunk.type, chunk.text, chunk.finish_reason) for chunk in chunks] == [
        ("text_delta", "ok", None),
        ("finish", None, "stop"),
    ]


@pytest.mark.parametrize("mode", ["omitted", "mutated"])
async def test_responses_rejects_nonempty_delta_logprobs_changed_at_text_done(mode) -> None:
    events = _official_text_events_with_logprobs()
    if mode == "omitted":
        events[4].pop("logprobs")
    else:
        events[4]["logprobs"][0]["logprob"] = -9.0

    with pytest.raises(LLMUnavailableError, match="text completion"):
        await _parse_response_events(events)


@pytest.mark.parametrize("boundary", ["text_done", "content_part", "output_item", "completed"])
async def test_responses_rejects_logprob_scalar_type_changes(boundary) -> None:
    events = _official_text_events_with_logprobs()
    if boundary == "text_done":
        logprobs = events[4]["logprobs"]
    elif boundary == "content_part":
        logprobs = events[5]["part"]["logprobs"]
    elif boundary == "output_item":
        logprobs = events[6]["item"]["content"][0]["logprobs"]
    else:
        logprobs = events[7]["response"]["output"][0]["content"][0]["logprobs"]
    logprobs[0]["bytes"][0] = True

    with pytest.raises(LLMUnavailableError):
        await _parse_response_events(events)


@pytest.mark.parametrize("boundary", ["content_part", "output_item", "completed"])
@pytest.mark.parametrize("field", ["logprobs", "annotations"])
@pytest.mark.parametrize("mode", ["omitted", "mutated"])
async def test_responses_rejects_output_text_metadata_changed_at_final_boundaries(
    boundary, field, mode
) -> None:
    events = _official_text_events_with_logprobs()
    if boundary == "content_part":
        part = events[5]["part"]
    elif boundary == "output_item":
        part = events[6]["item"]["content"][0]
    else:
        part = events[7]["response"]["output"][0]["content"][0]
    if mode == "omitted":
        part.pop(field)
    elif field == "logprobs":
        part[field][0]["token"] = "mutated"
    else:
        part[field] = [{"type": "url_citation", "url": "https://invalid.example"}]

    expected_error = {
        "content_part": "content part",
        "output_item": "output item (?:completion|content)",
        "completed": "final output",
    }[boundary]
    with pytest.raises(LLMUnavailableError, match=expected_error):
        await _parse_response_events(events)


async def test_responses_accepts_full_official_text_lifecycle_and_sequence_gaps() -> None:
    # The official schema describes sequence_number only as the value used to
    # order events; it does not promise contiguous values or a zero origin.
    # https://github.com/openai/openai-openapi/blob/master/openapi.yaml
    empty_part = {
        "type": "output_text",
        "text": "",
        "annotations": [],
        "logprobs": [],
    }
    done_part = {**empty_part, "text": "ok"}
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {"id": "resp", "status": "in_progress"}, "sequence_number": 0},
        {"type": "response.in_progress", "response": {"id": "resp", "status": "in_progress"}, "sequence_number": 2},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg", "type": "message", "role": "assistant", "content": [], "status": "in_progress"},
            "sequence_number": 3,
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": empty_part,
            "sequence_number": 4,
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "ok",
            "logprobs": [],
            "sequence_number": 5,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "text": "ok",
            "logprobs": [],
            "sequence_number": 6,
        },
        {
            "type": "response.content_part.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": done_part,
            "sequence_number": 7,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"id": "msg", "type": "message", "role": "assistant", "content": [done_part], "status": "completed"},
            "sequence_number": 8,
        },
        {"type": "response.completed", "response": {"id": "resp", "status": "completed"}, "sequence_number": 9},
    ]
    events[-1]["response"]["output"] = [events[-2]["item"]]

    chunks = await _parse_response_events(events)
    assert [(chunk.type, chunk.text, chunk.finish_reason) for chunk in chunks] == [
        ("text_delta", "ok", None),
        ("finish", None, "stop"),
    ]


async def test_responses_accepts_and_correlates_official_refusal_events() -> None:
    events: list[dict[str, Any]] = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "msg", "type": "message", "role": "assistant", "content": [], "status": "in_progress"},
            "sequence_number": 1,
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "refusal", "refusal": ""},
            "sequence_number": 2,
        },
        {
            "type": "response.refusal.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "no",
            "sequence_number": 3,
        },
        {
            "type": "response.refusal.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "refusal": "no",
            "sequence_number": 4,
        },
        {
            "type": "response.content_part.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "refusal", "refusal": "no"},
            "sequence_number": 5,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"id": "msg", "type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "no"}], "status": "completed"},
            "sequence_number": 6,
        },
        {"type": "response.completed", "response": {"id": "resp", "status": "completed"}, "sequence_number": 7},
    ]
    events[-1]["response"]["output"] = [events[-2]["item"]]

    chunks = await _parse_response_events(events)
    assert [(chunk.type, chunk.text) for chunk in chunks] == [
        ("text_delta", "no"),
        ("finish", None),
    ]


async def test_responses_rejects_nul_refusal_delta_before_emitting_or_retaining() -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "content": [],
                "status": "in_progress",
            },
            "sequence_number": 1,
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "refusal", "refusal": ""},
            "sequence_number": 2,
        },
        {
            "type": "response.refusal.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "no\x00secret",
            "sequence_number": 3,
        },
    ]
    wire = b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield wire

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    stream = OpenAIChatGPTProvider(access_token=access)._parse_stream(Response())
    with pytest.raises(LLMUnavailableError, match="invalid refusal event"):
        await anext(stream)


async def test_responses_rejects_nul_function_arguments_before_emitting_or_retaining() -> None:
    events = _function_call_events()
    events[2]["delta"] = '{"value":"bad\x00argument"}'
    events[3]["arguments"] = events[2]["delta"]
    events[4]["item"]["arguments"] = events[2]["delta"]
    events[5]["response"]["output"] = [events[4]["item"]]
    wire = b"".join(
        b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
    )

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield wire

    provider = OpenAIChatGPTProvider(access_token=cast(Any, None))
    stream = provider._parse_stream(Response())
    first = await anext(stream)
    assert first.type == "tool_call_start"
    with pytest.raises(LLMUnavailableError, match="invalid function arguments"):
        await anext(stream)


@pytest.mark.parametrize(
    "item",
    [
        {
            "id": "reasoning",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "bad\x00summary"}],
            "status": "in_progress",
        },
        {
            "id": "reasoning",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "bad\x00encrypted",
            "status": "in_progress",
        },
    ],
    ids=["summary", "encrypted-content"],
)
async def test_responses_rejects_nul_reasoning_metadata_before_retention(item) -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": item,
            "sequence_number": 1,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="invalid output item"):
        await _parse_response_events(events)


async def test_responses_rejects_nul_response_id_before_retention() -> None:
    with pytest.raises(LLMUnavailableError, match="creation event"):
        await _parse_response_events(
            [
                {
                    "type": "response.created",
                    "response": {"id": "resp\x00secret"},
                    "sequence_number": 0,
                }
            ]
        )


def test_responses_rejects_nul_request_values_and_error_metadata() -> None:
    from app.agents.providers import openai_chatgpt

    with pytest.raises(LLMNotSupportedError, match="invalid string"):
        _serialize_input([ChatMessage(role="user", content="bad\x00message")])
    assert openai_chatgpt._safe_metadata("bad\x00metadata") is None


def test_metadata_sanitizers_reject_hostile_string_subclasses_without_hooks() -> None:
    from app.agents.providers import openai_chatgpt

    class HostileString(str):
        def __len__(self):
            raise RuntimeError("metadata-length-sentinel")

        def __contains__(self, _value):
            raise RuntimeError("metadata-contains-sentinel")

        def encode(self, *_args, **_kwargs):
            raise RuntimeError("metadata-encode-sentinel")

        def __str__(self):
            raise RuntimeError("metadata-string-sentinel")

    hostile = HostileString("metadata-bearer-sentinel")
    assert openai_chatgpt._safe_metadata(hostile) is None
    failure = LLMUnavailableError("safe")
    failure.provider_code = hostile
    snapshot = openai_chatgpt._safe_llm_failure(failure)
    assert snapshot[3]["provider_code"] is None


@pytest.mark.parametrize(
    ("event", "error"),
    [
        ({"type": "response.in_progress", "response": "bad"}, "lifecycle"),
        ({"type": "response.queued", "response": {"id": "resp", "status": "completed"}}, "lifecycle"),
        ({"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}}, "output item"),
        ({"type": "response.content_part.added", "item_id": "msg", "output_index": 0, "content_index": 0}, "content part"),
        ({"type": "response.content_part.done", "item_id": "msg", "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": ""}}, "content part"),
        ({"type": "response.output_text.done", "text": "secret"}, "text completion"),
        ({"type": "response.refusal.delta", "item_id": "msg", "output_index": -1, "content_index": 0, "delta": "secret"}, "refusal"),
        ({"type": "error", "code": "bad", "message": "secret"}, "error event"),
    ],
    ids=[
        "in-progress-fields",
        "queued-status",
        "message-item-fields",
        "content-part-required",
        "content-part-order",
        "output-text-done-fields",
        "refusal-index",
        "error-required",
    ],
)
async def test_responses_rejects_malformed_official_event_schemas(event, error) -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {**event, "sequence_number": 1},
    ]
    with pytest.raises(LLMUnavailableError, match=error) as exc_info:
        await _parse_response_events(events)
    assert "secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    "done_event",
    [
        {"item_id": "other", "output_index": 0, "content_index": 0, "text": "ok", "logprobs": []},
        {"item_id": "msg", "output_index": 0, "content_index": 0, "text": "different", "logprobs": []},
    ],
    ids=["mismatched-item", "mismatched-text"],
)
async def test_responses_rejects_mismatched_output_text_done(done_event) -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "ok",
            "logprobs": [],
            "sequence_number": 1,
        },
        {"type": "response.output_text.done", **done_event, "sequence_number": 2},
    ]
    with pytest.raises(LLMUnavailableError, match="text completion"):
        await _parse_response_events(events)


async def test_responses_rejects_completion_with_unfinished_compact_text() -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "partial",
            "sequence_number": 1,
        },
        {
            "type": "response.completed",
            "response": {"id": "resp", "status": "completed"},
            "sequence_number": 2,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="unfinished content"):
        await _parse_response_events(events)


async def test_responses_rejects_mixed_compact_and_official_content_state() -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "partial",
            "sequence_number": 1,
        },
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "content": [],
                "status": "in_progress",
            },
            "sequence_number": 2,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="mixed content state"):
        await _parse_response_events(events)


def _function_call_events(*, done_status: str = "completed", output=None):
    added = {
        "type": "function_call",
        "namespace": "securo",
        "id": "item",
        "call_id": "call",
        "name": "lookup",
        "arguments": "",
        "status": "in_progress",
    }
    done = {**added, "arguments": "{}", "status": done_status}
    final_output = [done] if output is None else output
    return [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {"type": "response.output_item.added", "output_index": 0, "item": added, "sequence_number": 1},
        {"type": "response.function_call_arguments.delta", "item_id": "item", "output_index": 0, "delta": "{}", "sequence_number": 2},
        {"type": "response.function_call_arguments.done", "item_id": "item", "output_index": 0, "arguments": "{}", "sequence_number": 3},
        {"type": "response.output_item.done", "output_index": 0, "item": done, "sequence_number": 4},
        {"type": "response.completed", "response": {"id": "resp", "status": "completed", "output": final_output}, "sequence_number": 5},
    ]


@pytest.mark.parametrize(
    "events",
    [
        _function_call_events(done_status="in_progress"),
        _function_call_events(output=[]),
    ],
    ids=["nonterminal-item-status", "missing-final-output"],
)
async def test_responses_reconciles_function_items_with_final_output(events) -> None:
    with pytest.raises(LLMUnavailableError, match="function call|final output"):
        await _parse_response_events(events)


async def test_responses_rejects_prepopulated_message_added_item_replacement() -> None:
    initial = {"type": "output_text", "text": "INITIAL", "annotations": []}
    final = {"type": "output_text", "text": "FINAL", "annotations": []}
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "content": [initial],
                "status": "in_progress",
            },
            "sequence_number": 1,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "content": [final],
                "status": "completed",
            },
            "sequence_number": 2,
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp",
                "status": "completed",
                "output": [
                    {
                        "id": "msg",
                        "type": "message",
                        "role": "assistant",
                        "content": [final],
                        "status": "completed",
                    }
                ],
            },
            "sequence_number": 3,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="invalid output item"):
        await _parse_response_events(events)


@pytest.mark.parametrize(
    ("added_overrides", "done_overrides"),
    [
        (
            {"summary": [{"type": "summary_text", "text": "INITIAL"}]},
            {"summary": [{"type": "summary_text", "text": "FINAL"}]},
        ),
        (
            {"content": [{"type": "reasoning_text", "text": "INITIAL"}]},
            {"content": [{"type": "reasoning_text", "text": "FINAL"}]},
        ),
        (
            {"encrypted_content": "prefix"},
            {"encrypted_content": None},
        ),
        (
            {"encrypted_content": "prefix"},
            {"encrypted_content": "different"},
        ),
    ],
    ids=["summary-mutation", "content-mutation", "encrypted-omission", "encrypted-mutation"],
)
async def test_responses_rejects_reasoning_added_item_replacement(
    added_overrides, done_overrides
) -> None:
    added = {
        "id": "reasoning",
        "type": "reasoning",
        "summary": [],
        "status": "in_progress",
        **added_overrides,
    }
    done = {
        "id": "reasoning",
        "type": "reasoning",
        "summary": [],
        "status": "completed",
        **done_overrides,
    }
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {"type": "response.output_item.added", "output_index": 0, "item": added, "sequence_number": 1},
        {"type": "response.output_item.done", "output_index": 0, "item": done, "sequence_number": 2},
        {
            "type": "response.completed",
            "response": {"id": "resp", "status": "completed", "output": [done]},
            "sequence_number": 3,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="invalid output item|completion was invalid"):
        await _parse_response_events(events)


async def test_responses_rejects_reasoning_encrypted_content_prefix_evolution() -> None:
    added = {
        "id": "reasoning",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "INITIAL",
        "status": "in_progress",
    }
    done = {
        **added,
        "encrypted_content": "INITIAL-FINAL",
        "status": "completed",
    }
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {"type": "response.output_item.added", "output_index": 0, "item": added, "sequence_number": 1},
        {"type": "response.output_item.done", "output_index": 0, "item": done, "sequence_number": 2},
        {
            "type": "response.completed",
            "response": {"id": "resp", "status": "completed", "output": [done]},
            "sequence_number": 3,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="completion was invalid"):
        await _parse_response_events(events)


async def test_responses_accepts_reasoning_encrypted_content_after_null_added_default() -> None:
    added = {
        "id": "reasoning",
        "type": "reasoning",
        "summary": [],
        "encrypted_content": None,
        "status": "in_progress",
    }
    done = {
        **added,
        "encrypted_content": "FINAL",
        "status": "completed",
    }
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {"type": "response.output_item.added", "output_index": 0, "item": added, "sequence_number": 1},
        {"type": "response.output_item.done", "output_index": 0, "item": done, "sequence_number": 2},
        {
            "type": "response.completed",
            "response": {"id": "resp", "status": "completed", "output": [done]},
            "sequence_number": 3,
        },
    ]

    chunks = await _parse_response_events(events)
    assert [(chunk.type, chunk.finish_reason) for chunk in chunks] == [("finish", "stop")]


async def test_responses_rejects_nonempty_added_content_part_logprobs() -> None:
    initial_logprobs = [
        {"token": "INITIAL", "bytes": [73], "logprob": -0.1, "top_logprobs": []}
    ]
    final_part = {
        "type": "output_text",
        "text": "ok",
        "annotations": [],
        "logprobs": [],
    }
    done = {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "content": [final_part],
        "status": "completed",
    }
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "content": [],
                "status": "in_progress",
            },
            "sequence_number": 1,
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": initial_logprobs,
            },
            "sequence_number": 2,
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "ok",
            "logprobs": [],
            "sequence_number": 3,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "text": "ok",
            "logprobs": [],
            "sequence_number": 4,
        },
        {
            "type": "response.content_part.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": final_part,
            "sequence_number": 5,
        },
        {"type": "response.output_item.done", "output_index": 0, "item": done, "sequence_number": 6},
        {
            "type": "response.completed",
            "response": {"id": "resp", "status": "completed", "output": [done]},
            "sequence_number": 7,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="content part"):
        await _parse_response_events(events)


async def test_responses_accepts_canonical_reasoning_evolution() -> None:
    done = {
        "id": "reasoning",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "summary"}],
        "content": [{"type": "reasoning_text", "text": "thought"}],
        "encrypted_content": "encrypted",
        "status": "completed",
    }
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "reasoning",
                "type": "reasoning",
                "summary": [],
                "content": [],
                "encrypted_content": "encrypted",
                "status": "in_progress",
            },
            "sequence_number": 1,
        },
        {
            "type": "response.reasoning_summary_part.added",
            "item_id": "reasoning",
            "output_index": 0,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": ""},
            "sequence_number": 2,
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": "reasoning",
            "output_index": 0,
            "summary_index": 0,
            "delta": "summary",
            "sequence_number": 3,
        },
        {
            "type": "response.reasoning_summary_text.done",
            "item_id": "reasoning",
            "output_index": 0,
            "summary_index": 0,
            "text": "summary",
            "sequence_number": 4,
        },
        {
            "type": "response.reasoning_summary_part.done",
            "item_id": "reasoning",
            "output_index": 0,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": "summary"},
            "sequence_number": 5,
        },
        {
            "type": "response.content_part.added",
            "item_id": "reasoning",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "reasoning_text", "text": ""},
            "sequence_number": 6,
        },
        {
            "type": "response.reasoning_text.delta",
            "item_id": "reasoning",
            "output_index": 0,
            "content_index": 0,
            "delta": "thought",
            "sequence_number": 7,
        },
        {
            "type": "response.reasoning_text.done",
            "item_id": "reasoning",
            "output_index": 0,
            "content_index": 0,
            "text": "thought",
            "sequence_number": 8,
        },
        {
            "type": "response.content_part.done",
            "item_id": "reasoning",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "reasoning_text", "text": "thought"},
            "sequence_number": 9,
        },
        {"type": "response.output_item.done", "output_index": 0, "item": done, "sequence_number": 10},
        {
            "type": "response.completed",
            "response": {"id": "resp", "status": "completed", "output": [done]},
            "sequence_number": 11,
        },
    ]

    chunks = await _parse_response_events(events)
    assert [(chunk.type, chunk.finish_reason) for chunk in chunks] == [("finish", "stop")]


async def test_responses_requires_terminal_output_for_completed_function_call() -> None:
    events = _function_call_events()
    events[-1]["response"].pop("output")

    with pytest.raises(
        LLMUnavailableError,
        match="^ChatGPT final output did not match streamed items$",
    ) as exc_info:
        await _parse_response_events(events)

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize("output", [None, {}, "not-a-list", [None] * 129])
async def test_responses_requires_bounded_terminal_output_list(output) -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.completed",
            "response": {"id": "resp", "status": "completed", "output": output},
            "sequence_number": 1,
        },
    ]

    with pytest.raises(
        LLMUnavailableError,
        match="^ChatGPT final output did not match streamed items$",
    ):
        await _parse_response_events(events)


async def test_responses_requires_terminal_output_after_compact_text_and_usage() -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "ok",
            "sequence_number": 1,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "text": "ok",
            "sequence_number": 2,
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp",
                "status": "completed",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            "sequence_number": 3,
        },
    ]

    with pytest.raises(
        LLMUnavailableError,
        match="^ChatGPT final output did not match streamed items$",
    ):
        await _parse_response_events(events)


async def test_responses_accepts_explicit_empty_terminal_output() -> None:
    chunks = await _parse_response_events(
        [
            {
                "type": "response.created",
                "response": {"id": "resp"},
                "sequence_number": 0,
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
                "sequence_number": 1,
            },
        ]
    )

    assert [chunk.type for chunk in chunks] == ["usage", "finish"]


async def test_responses_accepts_optional_function_namespace_and_status() -> None:
    events = _function_call_events()
    for item in (events[1]["item"], events[4]["item"]):
        item.pop("namespace")
        item.pop("status")
    events[5]["response"]["output"] = [events[4]["item"]]

    chunks = await _parse_response_events(events)

    assert [chunk.type for chunk in chunks] == [
        "tool_call_start",
        "tool_call_args_delta",
        "tool_call_end",
        "finish",
    ]


async def test_sse_scanning_resumes_near_new_bytes(monkeypatch) -> None:
    import builtins
    from app.agents.providers import openai_chatgpt

    reads = 0

    class TrackingBuffer(builtins.bytearray):
        def __getitem__(self, key):
            nonlocal reads
            reads += 1
            return super().__getitem__(key)

    class Response:
        status_code = 200

        async def aiter_raw(self):
            for _ in range(100):
                yield b"x"

    monkeypatch.setattr(openai_chatgpt, "bytearray", TrackingBuffer, raising=False)
    with pytest.raises(LLMUnavailableError, match="partial event"):
        _ = [event async for event in openai_chatgpt._iter_sse(Response())]
    assert reads <= 200


async def test_sse_many_events_in_one_chunk_scans_forward_without_front_deletes(
    monkeypatch,
) -> None:
    import builtins
    from app.agents.providers import openai_chatgpt

    reads = 0
    front_deletes = 0

    class TrackingBuffer(builtins.bytearray):
        def __getitem__(self, key):
            nonlocal reads
            reads += 1
            return super().__getitem__(key)

        def __delitem__(self, key):
            nonlocal front_deletes
            front_deletes += 1
            return super().__delitem__(key)

    wire = b"".join(
        b'data: {"type":"probe","value":1}\n\n' for _ in range(500)
    )

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield wire

    monkeypatch.setattr(openai_chatgpt, "bytearray", TrackingBuffer, raising=False)
    events = [event async for event in openai_chatgpt._iter_sse(Response())]

    assert len(events) == 500
    assert front_deletes == 0
    assert reads <= len(wire) * 2


@pytest.mark.parametrize(
    "output",
    [
        [],
        [
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "different",
                        "annotations": [],
                    }
                ],
            }
        ],
        [
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "ok",
                        "annotations": [],
                        "logprobs": [{"token": "mutated"}],
                    }
                ],
            }
        ],
    ],
    ids=["missing", "contradictory", "logprobs-mutation"],
)
async def test_responses_reconciles_compact_text_with_final_output(output) -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "ok",
            "sequence_number": 1,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "text": "ok",
            "sequence_number": 2,
        },
        {
            "type": "response.completed",
            "response": {"id": "resp", "status": "completed", "output": output},
            "sequence_number": 3,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="final output"):
        await _parse_response_events(events)


async def test_responses_rejects_incomplete_message_in_completed_final_output() -> None:
    item = {
        "id": "msg",
        "type": "message",
        "role": "assistant",
        "status": "incomplete",
        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
    }
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "ok",
            "sequence_number": 1,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "text": "ok",
            "sequence_number": 2,
        },
        {
            "type": "response.completed",
            "response": {"id": "resp", "status": "completed", "output": [item]},
            "sequence_number": 3,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="final output"):
        await _parse_response_events(events)


async def test_responses_rejects_incomplete_retained_item_when_final_output_is_omitted() -> None:
    empty_part = {"type": "output_text", "text": "", "annotations": []}
    done_part = {**empty_part, "text": "ok"}
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "content": [],
                "status": "in_progress",
            },
            "sequence_number": 1,
        },
        {
            "type": "response.content_part.added",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": empty_part,
            "sequence_number": 2,
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "ok",
            "sequence_number": 3,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "text": "ok",
            "sequence_number": 4,
        },
        {
            "type": "response.content_part.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "part": done_part,
            "sequence_number": 5,
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "content": [done_part],
                "status": "incomplete",
            },
            "sequence_number": 6,
        },
        {
            "type": "response.completed",
            "response": {"id": "resp", "status": "completed"},
            "sequence_number": 7,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="final output status"):
        await _parse_response_events(events)


@pytest.mark.parametrize("mode", ["compact", "official"])
async def test_responses_rejects_noncontiguous_content_indices_before_output(mode) -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
    ]
    if mode == "official":
        events.extend(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "id": "msg",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "status": "in_progress",
                    },
                    "sequence_number": 1,
                },
                {
                    "type": "response.content_part.added",
                    "item_id": "msg",
                    "output_index": 0,
                    "content_index": 127,
                    "part": {
                        "type": "output_text",
                        "text": "",
                        "annotations": [],
                    },
                    "sequence_number": 2,
                },
            ]
        )
    else:
        events.append(
            {
                "type": "response.output_text.delta",
                "item_id": "msg",
                "output_index": 0,
                "content_index": 127,
                "delta": "must-not-emit",
                "sequence_number": 1,
            }
        )

    with pytest.raises(LLMUnavailableError, match="content.*order"):
        await _parse_response_events(events)


async def test_responses_rejects_later_compact_part_before_prior_part_done() -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "A",
            "sequence_number": 1,
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 1,
            "delta": "must-not-emit",
            "sequence_number": 2,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="content.*order"):
        await _parse_response_events(events)


@pytest.mark.parametrize("mode", ["compact", "official"])
async def test_responses_rejects_output_indices_that_begin_out_of_order(mode) -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
    ]
    if mode == "official":
        events.append(
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "status": "in_progress",
                },
                "sequence_number": 1,
            }
        )
    else:
        events.append(
            {
                "type": "response.output_text.delta",
                "item_id": "msg",
                "output_index": 1,
                "content_index": 0,
                "delta": "must-not-emit",
                "sequence_number": 1,
            }
        )

    with pytest.raises(LLMUnavailableError, match="output.*order"):
        await _parse_response_events(events)


@pytest.mark.parametrize("mode", ["compact", "cross-type"])
async def test_responses_rejects_output_item_id_reuse(mode) -> None:
    message = {
        "id": "duplicate",
        "type": "message",
        "role": "assistant",
        "content": [],
        "status": "in_progress",
    }
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
    ]
    if mode == "cross-type":
        events.extend(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": message,
                    "sequence_number": 1,
                },
                {
                    "type": "response.output_item.added",
                    "output_index": 1,
                    "item": {
                        "id": "duplicate",
                        "type": "function_call",
                        "namespace": "securo",
                        "call_id": "call",
                        "name": "lookup",
                        "arguments": "",
                        "status": "in_progress",
                    },
                    "sequence_number": 2,
                },
            ]
        )
    else:
        events.extend(
            [
                {
                    "type": "response.output_text.delta",
                    "item_id": "duplicate",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "A",
                    "sequence_number": 1,
                },
                {
                    "type": "response.output_text.done",
                    "item_id": "duplicate",
                    "output_index": 0,
                    "content_index": 0,
                    "text": "A",
                    "sequence_number": 2,
                },
                {
                    "type": "response.output_text.delta",
                    "item_id": "duplicate",
                    "output_index": 1,
                    "content_index": 0,
                    "delta": "must-not-emit",
                    "sequence_number": 3,
                },
            ]
        )

    with pytest.raises(LLMUnavailableError, match="duplicate output item"):
        await _parse_response_events(events)


async def test_responses_rejects_reopening_compact_output_after_later_output() -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
    ]
    sequence = 1
    for output_index, content_index, text in [(0, 0, "A"), (1, 0, "B"), (0, 1, "C")]:
        events.extend(
            [
                {
                    "type": "response.output_text.delta",
                    "item_id": f"msg-{output_index}",
                    "output_index": output_index,
                    "content_index": content_index,
                    "delta": text,
                    "sequence_number": sequence,
                },
                {
                    "type": "response.output_text.done",
                    "item_id": f"msg-{output_index}",
                    "output_index": output_index,
                    "content_index": content_index,
                    "text": text,
                    "sequence_number": sequence + 1,
                },
            ]
        )
        sequence += 2

    with pytest.raises(LLMUnavailableError, match="output.*order"):
        await _parse_response_events(events)


async def test_responses_rejects_queued_lifecycle_after_output() -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": "ok",
            "sequence_number": 1,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "text": "ok",
            "sequence_number": 2,
        },
        {
            "type": "response.queued",
            "response": {"id": "resp", "status": "queued"},
            "sequence_number": 3,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="lifecycle"):
        await _parse_response_events(events)


@pytest.mark.parametrize("separator", [b"\r\r", b"\r\n\r\n"])
async def test_sse_accepts_official_carriage_return_framing(separator) -> None:
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.completed",
            "response": {"id": "resp", "status": "completed", "output": []},
            "sequence_number": 1,
        },
    ]
    wire = b"".join(b"data: " + json.dumps(event).encode() + separator for event in events)

    class Response:
        status_code = 200

        async def aiter_raw(self):
            for byte in wire:
                yield bytes([byte])

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    provider = OpenAIChatGPTProvider(access_token=access)
    chunks = [chunk async for chunk in provider._parse_stream(Response())]
    assert [(chunk.type, chunk.finish_reason) for chunk in chunks] == [("finish", "stop")]


@pytest.mark.parametrize("kind", ["nul-text", "usage-overflow"])
async def test_responses_rejects_values_that_cannot_be_persisted(kind) -> None:
    text = "bad\x00text" if kind == "nul-text" else "ok"
    usage = (
        {"input_tokens": 2_147_483_648, "output_tokens": 0, "total_tokens": 2_147_483_648}
        if kind == "usage-overflow"
        else {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    )
    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_text.delta",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
            "sequence_number": 1,
        },
        {
            "type": "response.output_text.done",
            "item_id": "msg",
            "output_index": 0,
            "content_index": 0,
            "text": text,
            "sequence_number": 2,
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp",
                "status": "completed",
                "output": [
                    {
                        "id": "msg",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": text,
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": usage,
            },
            "sequence_number": 3,
        },
    ]

    with pytest.raises(LLMUnavailableError, match="text delta|usage"):
        await _parse_response_events(events)


async def test_partial_sse_and_bounded_event_stream_fail_deterministically(monkeypatch) -> None:
    from app.agents.providers import openai_chatgpt

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield b"data: {\"type\":\"response.output_text.delta\""

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    monkeypatch.setattr(openai_chatgpt, "_MAX_EVENT_BYTES", 16)
    provider = OpenAIChatGPTProvider(access_token=access)
    with pytest.raises(Exception, match="event exceeded"):
        _ = [chunk async for chunk in provider._parse_stream(Response())]


async def test_transport_failure_scrubs_bearer_from_retained_http_objects() -> None:
    secret = "provider-bearer-sentinel"
    retained_request = httpx.Request(
        "POST", "https://api.openai.com/v1/responses", headers={"Authorization": f"Bearer {secret}"}
    )

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token=secret, credential_version=1)

    class Response:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_raw(self):
            raise httpx.ReadError("read failed", request=retained_request)
            yield b""  # pragma: no cover

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    provider = OpenAIChatGPTProvider(access_token=access)
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        with pytest.raises(Exception, match="unreachable") as exc_info:
            _ = [
                chunk
                async for chunk in provider.chat_stream(
                    [ChatMessage(role="user", content="secret test")], model="gpt-test"
                )
            ]

    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    traceback = exc_info.value.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith("openai_chatgpt.py"):
            assert secret not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next
    assert "authorization" not in retained_request.headers


@pytest.mark.parametrize(
    "failure_point",
    [
        "client-constructor",
        "request-dispatch",
        "stream-enter",
        "stream-iteration",
        "response-close",
        "client-close",
        "credential-close",
    ],
)
async def test_unexpected_secret_region_failure_rethrows_fresh_sanitized_error(
    failure_point,
) -> None:
    secret = f"unique-bearer-{failure_point}-sentinel"

    class HostileFailure(RuntimeError):
        __slots__ = ("retained",)

    original = HostileFailure(f"hostile diagnostic contains {secret}")
    original.retained = {"nested": [secret]}

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token=secret, credential_version=1)
        if failure_point == "credential-close":
            raise original

    class Response:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def __aenter__(self):
            if failure_point == "stream-enter":
                raise original
            return self

        async def __aexit__(self, *_args):
            if failure_point == "response-close":
                raise original
            return None

        async def aiter_raw(self):
            if failure_point == "stream-iteration":
                raise original
            yield (
                b'data: {"type":"response.created","sequence_number":0,'
                b'"response":{"id":"resp"}}\n\n'
                b'data: {"type":"response.completed","sequence_number":1,'
                b'"response":{"id":"resp","status":"completed","output":[]}}\n\n'
            )

    class Client:
        def __init__(self, **_kwargs):
            if failure_point == "client-constructor":
                raise original

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            if failure_point == "client-close":
                raise original
            return None

        def stream(self, *_args, **_kwargs):
            if failure_point == "request-dispatch":
                raise original
            return Response()

    provider = OpenAIChatGPTProvider(access_token=access)
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        with pytest.raises(
            LLMUnavailableError, match="failed safely|cleanup failed"
        ) as exc_info:
            _ = [
                chunk
                async for chunk in provider.chat_stream(
                    [ChatMessage(role="user", content="secret boundary")],
                    model="gpt-test",
                )
            ]

    raised = exc_info.value
    assert raised is not original
    assert raised.__cause__ is None
    assert raised.__context__ is None
    assert secret not in str(raised)
    assert secret not in _retained_provider_secret_text(original, raised)


@pytest.mark.parametrize(
    "failure_point",
    ["client-constructor", "stream-iteration", "response-close", "credential-close"],
)
@pytest.mark.parametrize("contains_cancellation", [False, True], ids=["failure", "cancellation"])
async def test_hostile_exception_accessors_cannot_escape_secret_boundary(
    failure_point, contains_cancellation
) -> None:
    secret = f"hostile-boundary-{failure_point}-{contains_cancellation}-bearer"
    sentinel = "hostile-exception-accessor-sentinel"

    class HostileFailure(BaseException):
        __slots__ = ("retained",)

        def __getattribute__(self, name):
            if name in {"__cause__", "__context__", "__traceback__", "args", "__dict__"}:
                raise RuntimeError(sentinel)
            return super().__getattribute__(name)

    original = HostileFailure(f"diagnostic {secret}")
    original.retained = {"nested": [bytearray(secret.encode())]}
    if contains_cancellation:
        _EXCEPTION_CAUSE.__set__(original, asyncio.CancelledError())

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token=secret, credential_version=1)
        if failure_point == "credential-close":
            raise original

    class Response:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            if failure_point == "response-close":
                raise original
            return None

        async def aiter_raw(self):
            if failure_point == "stream-iteration":
                raise original
            yield (
                b'data: {"type":"response.created","sequence_number":0,'
                b'"response":{"id":"resp"}}\n\n'
                b'data: {"type":"response.completed","sequence_number":1,'
                b'"response":{"id":"resp","status":"completed","output":[]}}\n\n'
            )

    class Client:
        def __init__(self, **_kwargs):
            if failure_point == "client-constructor":
                raise original

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    provider = OpenAIChatGPTProvider(access_token=access)
    expected = (
        asyncio.CancelledError
        if contains_cancellation and failure_point != "response-close"
        else LLMUnavailableError
    )
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        with pytest.raises(expected) as exc_info:
            _ = [
                chunk
                async for chunk in provider.chat_stream(
                    [ChatMessage(role="user", content="hostile boundary")],
                    model="gpt-test",
                )
            ]

    raised = exc_info.value
    assert type(raised) is expected
    assert _EXCEPTION_CAUSE.__get__(raised, type(raised)) is None
    assert _EXCEPTION_CONTEXT.__get__(raised, type(raised)) is None
    retained = _retained_provider_secret_text(original, raised)
    assert sentinel not in retained
    assert secret not in retained


async def test_hostile_typed_failure_uses_fixed_snapshot_fallback() -> None:
    secret = "hostile-typed-bearer-sentinel"

    class HostileUnavailable(LLMUnavailableError):
        def __getattribute__(self, _name):
            raise RuntimeError("hostile-typed-accessor-sentinel")

    original = HostileUnavailable(f"typed diagnostic retained {secret}")

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token=secret, credential_version=1)

    class Response:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_raw(self):
            raise original
            yield b""  # pragma: no cover

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    provider = OpenAIChatGPTProvider(access_token=access)
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        with pytest.raises(
            LLMUnavailableError, match="failed safely|cleanup failed"
        ) as exc_info:
            _ = [
                chunk
                async for chunk in provider.chat_stream(
                    [ChatMessage(role="user", content="hostile typed")],
                    model="gpt-test",
                )
            ]

    assert type(exc_info.value) is LLMUnavailableError
    assert secret not in _retained_provider_secret_text(original, exc_info.value)


async def test_cancellation_closes_auth_scope_and_preserves_cancellation() -> None:
    released = False

    @asynccontextmanager
    async def access(**_kwargs):
        nonlocal released
        try:
            yield AccessTokenSnapshot(access_token="cancel-secret", credential_version=1)
        finally:
            released = True

    class Response:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_raw(self):
            raise asyncio.CancelledError()
            yield b""  # pragma: no cover

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    provider = OpenAIChatGPTProvider(access_token=access)
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        with pytest.raises(asyncio.CancelledError):
            _ = [
                chunk
                async for chunk in provider.chat_stream(
                    [ChatMessage(role="user", content="cancel")], model="gpt-test"
                )
            ]
    assert released is True


async def test_cancellation_wins_over_secret_region_cleanup_failure() -> None:
    secret = "cancel-cleanup-bearer-sentinel"
    cleanup_failure = RuntimeError(f"cleanup retained {secret}")

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token=secret, credential_version=1)

    class Response:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            raise cleanup_failure

        async def aiter_raw(self):
            raise asyncio.CancelledError()
            yield b""  # pragma: no cover

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    provider = OpenAIChatGPTProvider(access_token=access)
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        with pytest.raises(asyncio.CancelledError) as exc_info:
            _ = [
                chunk
                async for chunk in provider.chat_stream(
                    [ChatMessage(role="user", content="cancel")], model="gpt-test"
                )
            ]

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in _retained_provider_secret_text(cleanup_failure, exc_info.value)


async def test_deep_and_cyclic_cancellation_chains_survive_bearer_boundary() -> None:
    from app.agents.providers import openai_chatgpt

    sentinel = "deep-cancellation-bearer-sentinel"

    async def run_with(error: BaseException) -> None:
        @asynccontextmanager
        async def access(**_kwargs):
            yield AccessTokenSnapshot(access_token=sentinel, credential_version=1)

        class Response:
            status_code = 200
            headers = {"content-type": "text/event-stream"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def aiter_raw(self):
                raise error
                yield b""  # pragma: no cover

        class Client:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def stream(self, *_args, **_kwargs):
                return Response()

        provider = OpenAIChatGPTProvider(access_token=access)
        with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
            with pytest.raises(asyncio.CancelledError):
                _ = [
                    chunk
                    async for chunk in provider.chat_stream(
                        [ChatMessage(role="user", content="cancel")], model="gpt-test"
                    )
                ]

    deep: BaseException = asyncio.CancelledError()
    for _ in range(350):
        wrapper = RuntimeError(sentinel)
        _EXCEPTION_CAUSE.__set__(wrapper, deep)
        deep = wrapper
    assert openai_chatgpt._exception_contains_cancellation(deep) is True
    await run_with(deep)

    cyclic_outer = RuntimeError(sentinel)
    cyclic_cancel = asyncio.CancelledError()
    _EXCEPTION_CAUSE.__set__(cyclic_outer, cyclic_cancel)
    _EXCEPTION_CAUSE.__set__(cyclic_cancel, cyclic_outer)
    assert openai_chatgpt._exception_contains_cancellation(cyclic_outer) is True
    await run_with(cyclic_outer)


@pytest.mark.parametrize("cancel_count,uncancel_count", [(1, 0), (2, 0), (2, 1)])
async def test_sequential_direct_provider_calls_ignore_stale_cancellation_counts(
    cancel_count: int, uncancel_count: int
) -> None:
    async def failed_secret_stream(*_args, **_kwargs):
        raise LLMUnavailableError("independent provider failure")
        yield  # pragma: no cover

    async def scenario() -> tuple[int, int]:
        task = asyncio.current_task()
        assert task is not None
        for _ in range(cancel_count):
            task.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        for _ in range(uncancel_count):
            task.uncancel()
        baseline = task.cancelling()

        provider = OpenAIChatGPTProvider(access_token=cast(Any, object()))
        with patch.object(provider, "_chat_stream_secret", new=failed_secret_stream):
            for _ in range(2):
                with pytest.raises(LLMUnavailableError, match="independent provider failure"):
                    _ = [
                        chunk
                        async for chunk in provider.chat_stream([], model="gpt-test")
                    ]
        return baseline, task.cancelling()

    baseline, final_count = await asyncio.create_task(scenario())
    assert final_count == baseline == cancel_count - uncancel_count


@pytest.mark.parametrize(
    "messages",
    [
        [ChatMessage(role="tool", tool_call_id="missing", content="orphan")],
        [
            ChatMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(id="duplicate", name="one", arguments={}),
                    ToolCall(id="duplicate", name="two", arguments={}),
                ],
            )
        ],
        [
            ChatMessage(
                role="assistant",
                tool_calls=[ToolCall(id="call", name="one", arguments={})],
            ),
            ChatMessage(role="tool", tool_call_id="call", content="one"),
            ChatMessage(role="tool", tool_call_id="call", content="duplicate"),
        ],
    ],
    ids=["orphan-output", "duplicate-call-id", "duplicate-output"],
)
def test_responses_history_rejects_ambiguous_function_replay(messages) -> None:
    with pytest.raises(LLMNotSupportedError):
        _serialize_input(messages)


def test_responses_serialization_rejects_interleaved_assistant_before_tool_output() -> None:
    with pytest.raises(LLMNotSupportedError):
        _serialize_input(
            [
                ChatMessage(
                    role="assistant",
                    tool_calls=[ToolCall(id="call", name="lookup", arguments={})],
                ),
                ChatMessage(role="assistant", content="premature"),
                ChatMessage(role="tool", tool_call_id="call", content="{}"),
            ]
        )


def test_responses_serialization_rejects_non_object_function_arguments() -> None:
    malformed_call = ToolCall(id="call", name="lookup", arguments=cast(dict[str, Any], []))
    with pytest.raises(LLMNotSupportedError):
        _serialize_input(
            [
                ChatMessage(role="assistant", tool_calls=[malformed_call]),
                ChatMessage(role="tool", tool_call_id="call", content="{}"),
            ]
        )


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [ChatMessage(role="user", content="question"), ChatMessage(role="system", content="late")],
    ],
    ids=["empty", "late-system"],
)
def test_responses_serialization_rejects_ambiguous_message_boundaries(messages) -> None:
    with pytest.raises(LLMNotSupportedError):
        _serialize_input(messages)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_responses_tool_schema_rejects_nonfinite_json(nonfinite) -> None:
    with pytest.raises(LLMNotSupportedError):
        _serialize_tools(
            [
                ToolDefinition(
                    name="tool",
                    description="d",
                    parameters={"type": "object", "minimum": nonfinite},
                )
            ]
        )


def test_responses_tool_schema_rejects_excessive_depth() -> None:
    parameters: dict = {"type": "object"}
    cursor = parameters
    for _ in range(40):
        nested: dict = {}
        cursor["nested"] = nested
        cursor = nested
    with pytest.raises(LLMNotSupportedError):
        _serialize_tools([ToolDefinition(name="tool", description="d", parameters=parameters)])


async def test_responses_requires_exact_success_status_and_event_stream_content_type() -> None:
    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="secret", credential_version=1)

    class Response:
        def __init__(self, status: int, content_type: str):
            self.status_code = status
            self.headers = {"content-type": content_type}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_raw(self):
            yield b'data: {"type":"response.completed","response":{"id":"r","status":"completed"}}\n\n'

    class Client:
        responses = [Response(201, "text/event-stream"), Response(200, "application/json")]

        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return self.responses.pop(0)

    provider = OpenAIChatGPTProvider(access_token=access)
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        for expected in ("HTTP 201", "content type"):
            with pytest.raises(LLMUnavailableError, match=expected):
                _ = [
                    chunk
                    async for chunk in provider.chat_stream(
                        [ChatMessage(role="user", content="strict")], model="gpt-test"
                    )
                ]


@pytest.mark.parametrize(
    "wire,match",
    [
        (
            b'data: {"type":"response.created","response":{"id":"r"}}\r\n\r\n'
            b'data: {"type":"future.unknown","response_id":"r"}\r\n\r\n',
            "unknown event",
        ),
        (
            b'data: {"type":"response.created","response":{"id":"r"}}\n\n'
            b"data: [DONE]\n\n",
            "malformed",
        ),
        (
            b'data: {"type":"response.created","response":{"id":"r"}}\n\n'
            b'data: {"type":"response.completed","response":{"id":"r","status":"completed",'
            b'"output":[],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":99}}}\n\n',
            "usage",
        ),
        (
            b'data: {"type":"response.created","response":{"id":"r"}}\n\n'
            b'data: {"type":"response.output_item.added","response_id":"r","output_index":0,'
            b'"item":{"type":"web_search_call"}}\n\n',
            "unsupported output item",
        ),
        (
            b'data: {"type":"response.created","response":{"id":"r"}}\n\n'
            b'data: {"type":"response.completed","response":{"status":"completed",'
            b'"usage":{"input_tokens":1,"output_tokens":1}}}\n\n',
            "completion",
        ),
        (
            b'data: {"type":"response.created","response":{"id":"r"}}\n\n'
            b'data: {"type":"response.completed","response":{"id":"r","status":"completed",'
            b'"output":[],"usage":{"input_tokens":1}}}\n\n',
            "usage",
        ),
    ],
    ids=[
        "unknown-event-crlf",
        "done-sentinel",
        "inconsistent-usage",
        "unsupported-output-item",
        "completion-without-id",
        "incomplete-usage",
    ],
)
async def test_responses_rejects_unknown_done_and_inconsistent_usage(wire, match) -> None:
    sequenced_blocks = []
    sequence = 0
    for block in wire.replace(b"\r\n", b"\n").split(b"\n\n"):
        if block.startswith(b"data: "):
            try:
                event = json.loads(block[6:])
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(event, dict):
                    event["sequence_number"] = sequence
                    sequence += 1
                    block = b"data: " + json.dumps(event).encode()
        sequenced_blocks.append(block)
    wire = b"\n\n".join(sequenced_blocks)

    class Response:
        status_code = 200

        async def aiter_raw(self):
            yield wire

    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="unused", credential_version=1)

    provider = OpenAIChatGPTProvider(access_token=access)
    with pytest.raises(LLMUnavailableError, match=match):
        _ = [chunk async for chunk in provider._parse_stream(Response())]


@pytest.mark.parametrize(
    "status,code,error_type",
    [
        (400, "client_configuration_error", LLMNotSupportedError),
        (403, "chatpass_v2_scope_not_authorized", LLMAuthError),
        (403, "subscription_sharing_route_not_supported", LLMNotSupportedError),
        (429, "subscription_sharing_usage_limit_exceeded", LLMRateLimitError),
        (503, "subscription_sharing_usage_unavailable", LLMUnavailableError),
    ],
)
async def test_http_error_body_uses_official_code_without_leaking_message(
    status, code, error_type
) -> None:
    @asynccontextmanager
    async def access(**_kwargs):
        yield AccessTokenSnapshot(access_token="secret", credential_version=1)

    body = json.dumps(
        {
            "error": {
                "code": code,
                "param": "tools[0].name",
                "message": "provider secret diagnostic",
            },
            "request_id": "req_body_exact",
        }
    ).encode()

    class Response:
        status_code = status
        headers = {"content-type": "application/json"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_raw(self):
            yield body

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, *_args, **_kwargs):
            return Response()

    provider = OpenAIChatGPTProvider(access_token=access)
    with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", Client):
        with pytest.raises(error_type) as exc_info:
            _ = [
                chunk
                async for chunk in provider.chat_stream(
                    [ChatMessage(role="user", content="error")], model="gpt-test"
                )
            ]
    assert "provider secret diagnostic" not in str(exc_info.value)
    assert exc_info.value.provider_code == code
    assert exc_info.value.provider_param == "tools[0].name"
    assert exc_info.value.request_id == "req_body_exact"
    assert exc_info.value.body_category == "error_object"


def test_oversized_message_is_rejected_before_json_encoding(monkeypatch) -> None:
    from app.agents.providers import openai_chatgpt

    monkeypatch.setattr(openai_chatgpt, "_MAX_REQUEST_BYTES", 32)
    called = False

    def forbidden_dumps(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("oversized input must not be JSON encoded")

    monkeypatch.setattr(openai_chatgpt.json, "dumps", forbidden_dumps)
    with pytest.raises(LLMNotSupportedError, match="size limit"):
        _serialize_input([ChatMessage(role="user", content="x" * 100)])
    assert called is False


@pytest.mark.parametrize(
    "arguments",
    [
        "[]",
        '{"x":NaN}',
        '{"x":1,"x":2}',
        '{"x":9223372036854775808}',
        '{"x":"\\ud800"}',
        "{" + '"x":{' * 34 + '"y":1' + "}" * 34 + "}",
    ],
)
async def test_streamed_function_arguments_require_strict_bounded_object(arguments) -> None:
    events = [
        {"type": "response.created", "sequence_number": 0, "response": {"id": "r"}},
        {
            "type": "response.output_item.added",
            "sequence_number": 1,
            "response_id": "r",
            "output_index": 0,
            "item": {"type": "function_call", "namespace": "securo", "id": "i", "call_id": "c", "name": "t", "arguments": "", "status": "in_progress"},
        },
        {"type": "response.function_call_arguments.delta", "sequence_number": 2, "response_id": "r", "output_index": 0, "item_id": "i", "delta": arguments},
        {"type": "response.function_call_arguments.done", "sequence_number": 3, "response_id": "r", "output_index": 0, "item_id": "i", "arguments": arguments},
        {"type": "response.output_item.done", "sequence_number": 4, "response_id": "r", "output_index": 0, "item": {"type": "function_call", "namespace": "securo", "id": "i", "call_id": "c", "name": "t", "arguments": arguments, "status": "completed"}},
    ]
    wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)

    class Response:
        status_code = 200
        async def aiter_raw(self):
            yield wire

    provider = OpenAIChatGPTProvider(access_token=cast(Any, None))
    with pytest.raises(LLMUnavailableError, match="function arguments"):
        _ = [chunk async for chunk in provider._parse_stream(Response())]


async def test_unsequenced_creation_is_rejected_before_text_output() -> None:
    event = {"type": "response.output_text.delta", "response_id": "r", "delta": "same"}
    wire = (
        b'data: {"type":"response.created","response":{"id":"r"}}\n\n'
        + b"data: " + json.dumps(event).encode() + b"\n\n"
        + b"data: " + json.dumps(event).encode() + b"\n\n"
    )

    class Response:
        status_code = 200
        async def aiter_raw(self):
            yield wire

    provider = OpenAIChatGPTProvider(access_token=cast(Any, None))
    seen = []
    with pytest.raises(LLMUnavailableError, match="sequence"):
        async for chunk in provider._parse_stream(Response()):
            seen.append(chunk.text)
    assert seen == []


async def test_stream_rejects_missing_sequence_after_sequenced_creation() -> None:
    events = [
        {"type": "response.created", "sequence_number": 0, "response": {"id": "r"}},
        {"type": "response.output_text.delta", "response_id": "r", "delta": "x"},
        {"type": "response.completed", "sequence_number": 2, "response": {"id": "r", "status": "completed", "usage": {"input_tokens": 1, "output_tokens": 1}}},
    ]
    wire = b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)

    class Response:
        status_code = 200
        async def aiter_raw(self):
            yield wire

    provider = OpenAIChatGPTProvider(access_token=cast(Any, None))
    with pytest.raises(LLMUnavailableError, match="sequence|order"):
        _ = [chunk async for chunk in provider._parse_stream(Response())]


async def test_response_incomplete_preserves_sanitized_metadata() -> None:
    event = {
        "type": "response.incomplete",
        "sequence_number": 0,
        "response": {
            "id": "r",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "error": {"code": "client_configuration_error", "param": "tools[0]", "message": "email=a@b.test bearer secret"},
        },
        "request_id": "req_exact",
    }
    wire = b"data: " + json.dumps(event).encode() + b"\n\n"

    class Response:
        status_code = 200
        async def aiter_raw(self):
            yield wire

    provider = OpenAIChatGPTProvider(access_token=cast(Any, None))
    with pytest.raises(LLMNotSupportedError) as exc_info:
        _ = [chunk async for chunk in provider._parse_stream(Response())]
    exc = exc_info.value
    assert exc.provider_code == "client_configuration_error"
    assert exc.provider_param == "tools[0]"
    assert exc.request_id == "req_exact"
    assert exc.body_category == "response.incomplete"
    assert "a@b.test" not in str(exc)
