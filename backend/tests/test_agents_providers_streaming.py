"""Coverage for the streaming chat + embed paths in
app/agents/providers/{openai,ollama,anthropic}.py. These providers open
httpx.AsyncClient.stream(...) inline, so we replace AsyncClient with a
fake whose stream() yields a pre-baked SSE/JSONL response.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Iterable, cast

import pytest

import app.agents.providers.anthropic as anthropic_module
import app.agents.providers.base as base_module
import app.agents.providers.ollama as ollama_module
from app.agents.providers.anthropic import AnthropicProvider
from app.agents.providers.base import (
    ChatChunk,
    ChatMessage,
    LLMAuthError,
    LLMNotSupportedError,
    LLMProvider,
    LLMRateLimitError,
    LLMUnavailableError,
    ToolDefinition,
    Usage,
)
from app.agents.providers.ollama import OllamaProvider
from app.agents.providers.openai import OpenAIProvider
from app.agents.providers.openai_compatible import OpenAICompatibleProvider


async def test_base_chat_joins_stream_cleanup_on_validation_failure():
    cleaned = False

    class Provider(LLMProvider):
        def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            async def source():
                nonlocal cleaned
                try:
                    yield cast(Any, object())
                    await asyncio.Event().wait()
                finally:
                    cleaned = True

            return source()

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError):
        await Provider().chat([], model="test")

    assert cleaned is True


async def test_base_chat_has_a_whole_operation_deadline(monkeypatch):
    cleaned = False

    class Provider(LLMProvider):
        def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            async def source():
                nonlocal cleaned
                try:
                    await asyncio.Event().wait()
                    yield  # pragma: no cover
                finally:
                    cleaned = True

            return source()

        async def embed(self, texts, *, model):
            return []

    monkeypatch.setattr(base_module, "LLM_OPERATION_TIMEOUT_SECONDS", 0.01)
    with pytest.raises(LLMUnavailableError, match="timed out"):
        await Provider().chat([], model="test")

    assert cleaned is True


async def test_base_chat_cleanup_failure_does_not_replace_primary_error():
    class Source:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise LLMAuthError("primary auth failure")

        async def aclose(self):
            raise RuntimeError("cleanup sentinel")

    class Provider(LLMProvider):
        def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            return Source()

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMAuthError, match="primary auth failure"):
        await Provider().chat([], model="test")


async def test_base_chat_cleanup_failure_after_success_is_sanitized():
    class Source:
        def __init__(self):
            self.chunks = iter([ChatChunk(type="finish", finish_reason="stop")])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.chunks)
            except StopIteration:
                raise StopAsyncIteration from None

        async def aclose(self):
            raise RuntimeError("cleanup sentinel")

    class Provider(LLMProvider):
        def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            return Source()

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="cleanup failed") as exc_info:
        await Provider().chat([], model="test")

    assert "sentinel" not in str(exc_info.value)


async def test_base_chat_synthetic_cleanup_cancellation_is_availability_error():
    class Source:
        def __init__(self):
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return ChatChunk(type="finish", finish_reason="stop")

        async def aclose(self):
            raise asyncio.CancelledError

    class Provider(LLMProvider):
        def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            return Source()

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="cleanup failed"):
        await Provider().chat([], model="test")


# --------------------------------------------------------------------- httpx fake

class _FakeStreamResponse:
    """Stand-in for the response object yielded from `client.stream(...)`."""

    def __init__(self, *, status_code: int, lines: Iterable[str], body: bytes = b""):
        self.status_code = status_code
        self._lines = list(lines)
        self._body = body

    async def aiter_lines(self):
        for ln in self._lines:
            yield ln

    async def aread(self) -> bytes:
        return self._body


class _FakeResponse:
    """Stand-in for a plain (non-streaming) response."""

    def __init__(self, *, status_code: int, json_body=None, text: str = ""):
        self.status_code = status_code
        self._body = json_body
        self.text = text

    def json(self):
        return self._body


class _FakeAsyncClient:
    queue_stream: list[_FakeStreamResponse] = []
    queue_post: list[_FakeResponse] = []
    posted: list[tuple[str, dict]] = []

    def __init__(self, *_, **__):  # accept (timeout=...) kw
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @asynccontextmanager
    async def stream(self, method, url, *, json=None, headers=None):  # noqa: A002
        type(self).posted.append((url, json or {}))
        yield type(self).queue_stream.pop(0)

    async def post(self, url, *, json=None, headers=None):  # noqa: A002
        type(self).posted.append((url, json or {}))
        return type(self).queue_post.pop(0)


@pytest.fixture(autouse=True)
def _fake_httpx(monkeypatch):
    _FakeAsyncClient.queue_stream = []
    _FakeAsyncClient.queue_post = []
    _FakeAsyncClient.posted = []
    monkeypatch.setattr(ollama_module.httpx, "AsyncClient", _FakeAsyncClient)
    monkeypatch.setattr(anthropic_module.httpx, "AsyncClient", _FakeAsyncClient)
    yield


# --------------------------------------------------------------------- base collector


class _HostileValue:
    @property
    def input_tokens(self):
        raise RuntimeError("model-derived-client-secret-sentinel")

    @property
    def output_tokens(self):
        raise RuntimeError("model-derived-client-secret-sentinel")

    def __str__(self):
        raise RuntimeError("model-derived-client-secret-sentinel")

    def __repr__(self):
        raise RuntimeError("model-derived-client-secret-sentinel")

    def __hash__(self):
        raise RuntimeError("model-derived-client-secret-sentinel")

    def __eq__(self, _other):
        raise RuntimeError("model-derived-client-secret-sentinel")


class _HostileChunk(ChatChunk):
    def __getattribute__(self, _name):
        raise RuntimeError("model-derived-client-secret-sentinel")


def _uninitialized_chunk() -> ChatChunk:
    return object.__new__(ChatChunk)


def _chunk_with_deleted_field() -> ChatChunk:
    chunk = ChatChunk(type="text_delta", text="must-not-escape")
    del chunk.type
    return chunk


def _chunk_with_uninitialized_usage() -> ChatChunk:
    return ChatChunk(type="usage", usage=object.__new__(Usage))


def _chunk_with_deleted_usage_field() -> ChatChunk:
    usage = Usage(input_tokens=1, output_tokens=2)
    del usage.output_tokens
    return ChatChunk(type="usage", usage=usage)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunk",
    [
        _HostileValue(),
        _uninitialized_chunk(),
        _chunk_with_deleted_field(),
        _chunk_with_uninitialized_usage(),
        _chunk_with_deleted_usage_field(),
        ChatChunk(type=cast(Any, _HostileValue())),
        ChatChunk(type=cast(Any, "unknown")),
        ChatChunk(type="text_delta", text=cast(Any, _HostileValue())),
        ChatChunk(type="text_delta", text="ok", tool_name="unexpected"),
        ChatChunk(type="tool_call_start", tool_call_id=cast(Any, _HostileValue()), tool_name="lookup"),
        ChatChunk(type="tool_call_start", tool_call_id="call", tool_name=cast(Any, _HostileValue())),
        ChatChunk(type="tool_call_start", tool_call_id="call", tool_name=""),
        ChatChunk(type="tool_call_start", tool_call_id="call", tool_name="lookup", args_delta="unexpected"),
        ChatChunk(type="tool_call_args_delta", tool_call_id=cast(Any, _HostileValue()), args_delta=""),
        ChatChunk(type="tool_call_args_delta", tool_call_id="call", args_delta=cast(Any, _HostileValue())),
        ChatChunk(type="tool_call_args_delta", tool_call_id="call", args_delta="", tool_name="unexpected"),
        ChatChunk(type="tool_call_end", tool_call_id=cast(Any, _HostileValue())),
        ChatChunk(type="tool_call_end", tool_call_id="call", thought_signature=cast(Any, _HostileValue())),
        ChatChunk(type="tool_call_end", tool_call_id="call", text="unexpected"),
        ChatChunk(type="usage", usage=cast(Any, _HostileValue())),
        ChatChunk(type="usage", usage=cast(Any, {"input_tokens": 1, "output_tokens": 2})),
        ChatChunk(type="usage", usage=Usage(input_tokens=cast(Any, True), output_tokens=0)),
        ChatChunk(type="usage", usage=Usage(input_tokens=-1, output_tokens=0)),
        ChatChunk(type="usage", usage=Usage(input_tokens=0, output_tokens=cast(Any, _HostileValue()))),
        ChatChunk(type="usage", usage=Usage(), text="unexpected"),
        ChatChunk(type="finish", finish_reason=cast(Any, _HostileValue())),
        ChatChunk(type="finish", finish_reason=cast(Any, [])),
        ChatChunk(type="finish", finish_reason=None),
        ChatChunk(type="finish", finish_reason="stop", usage=Usage()),
    ],
    ids=[
        "not-chunk", "uninitialized-chunk", "deleted-chunk-field",
        "uninitialized-usage", "deleted-usage-field", "hostile-type", "unknown-type",
        "hostile-text", "text-extra-field", "hostile-start-id", "hostile-start-name",
        "empty-start-name", "start-extra-field", "hostile-args-id", "hostile-args",
        "args-extra-field", "hostile-end-id", "hostile-signature", "end-extra-field",
        "hostile-usage", "usage-dict", "usage-bool", "usage-negative",
        "hostile-output-tokens", "usage-extra-field", "hostile-finish", "finish-list",
        "finish-none", "finish-extra-field",
    ],
)
async def test_base_chat_rejects_every_malformed_chunk_shape_without_inspecting_values(chunk):
    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            yield chunk

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError) as exc_info:
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")

    assert str(exc_info.value) in {
        "LLM returned an invalid chunk",
        "LLM returned an unknown chunk type",
        "LLM returned an invalid text delta",
        "LLM returned invalid tool metadata",
        "LLM returned invalid tool arguments",
        "LLM returned invalid usage",
        "LLM returned an invalid finish reason",
    }
    assert "sentinel" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_base_chat_rejects_chunk_subclass_before_attribute_access():
    hostile_chunk = object.__new__(_HostileChunk)

    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            yield hostile_chunk

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="invalid chunk"):
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")


@pytest.mark.asyncio
@pytest.mark.parametrize("after_end", [False, True], ids=["orphan", "after-end"])
async def test_base_chat_validates_empty_argument_delta_state_before_suppression(after_end):
    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            if after_end:
                yield ChatChunk(type="tool_call_start", tool_call_id="call", tool_name="lookup")
                yield ChatChunk(type="tool_call_end", tool_call_id="call")
            yield ChatChunk(type="tool_call_args_delta", tool_call_id="call", args_delta="")

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="before the tool call|continued"):
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")


@pytest.mark.asyncio
async def test_base_chat_rejects_duplicate_usage_before_completion():
    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            yield ChatChunk(type="usage", usage=Usage(input_tokens=1, output_tokens=2))
            yield ChatChunk(type="usage", usage=Usage(input_tokens=3, output_tokens=4))
            yield ChatChunk(type="finish", finish_reason="stop")

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="duplicate usage"):
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["open-tool", "payload-after-usage"])
async def test_base_chat_enforces_usage_terminal_order(mode):
    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            if mode == "open-tool":
                yield ChatChunk(type="tool_call_start", tool_call_id="call", tool_name="lookup")
            yield ChatChunk(type="usage", usage=Usage(input_tokens=1, output_tokens=2))
            if mode == "payload-after-usage":
                yield ChatChunk(type="text_delta", text="late")
            yield ChatChunk(type="finish", finish_reason="stop")

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="usage|incomplete"):
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")


@pytest.mark.asyncio
async def test_base_chat_rejects_malformed_tool_arguments_without_raw_fallback():
    secret = "base-collector-secret"

    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            yield ChatChunk(type="tool_call_start", tool_call_id="call", tool_name="lookup")
            yield ChatChunk(
                type="tool_call_args_delta",
                tool_call_id="call",
                args_delta="{" + secret,
            )
            yield ChatChunk(type="tool_call_end", tool_call_id="call")
            yield ChatChunk(type="finish", finish_reason="tool_calls")

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="malformed tool arguments") as exc_info:
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")
    assert secret not in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "error"),
    [
        (
            [
                ChatChunk(type="text_delta", text="éé"),
                ChatChunk(type="text_delta", text="a"),
            ],
            "text exceeded",
        ),
        (
            [
                ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="lookup"),
                ChatChunk(type="tool_call_args_delta", tool_call_id="a", args_delta="é"),
                ChatChunk(type="tool_call_start", tool_call_id="b", tool_name="lookup"),
                ChatChunk(type="tool_call_args_delta", tool_call_id="b", args_delta="abc"),
            ],
            "tool arguments exceeded",
        ),
        (
            [
                ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="lookup"),
                ChatChunk(
                    type="tool_call_args_delta",
                    tool_call_id="a",
                    args_delta=[],  # ty: ignore[invalid-argument-type]
                ),
            ],
            "invalid tool arguments",
        ),
    ],
)
async def test_base_chat_rejects_aggregate_utf8_before_retaining_fragment(
    monkeypatch, chunks, error
):
    monkeypatch.setattr(base_module, "MAX_STREAM_TEXT_BYTES", 4)
    monkeypatch.setattr(base_module, "MAX_STREAM_TOOL_ARGUMENT_BYTES", 4)
    monkeypatch.setattr(base_module, "MAX_STREAM_ALL_TOOL_ARGUMENT_BYTES", 4)

    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            for chunk in chunks:
                yield chunk

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match=error):
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")


@pytest.mark.asyncio
async def test_base_chat_accepts_exact_multibyte_boundaries(monkeypatch):
    monkeypatch.setattr(base_module, "MAX_STREAM_TEXT_BYTES", 4)
    monkeypatch.setattr(base_module, "MAX_STREAM_TOOL_ARGUMENT_BYTES", 4)
    monkeypatch.setattr(base_module, "MAX_STREAM_ALL_TOOL_ARGUMENT_BYTES", 4)

    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            yield ChatChunk(type="text_delta", text="é")
            yield ChatChunk(type="text_delta", text="é")
            yield ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="lookup")
            yield ChatChunk(type="tool_call_args_delta", tool_call_id="a", args_delta="{}")
            yield ChatChunk(type="tool_call_end", tool_call_id="a")
            yield ChatChunk(type="tool_call_start", tool_call_id="b", tool_name="lookup")
            yield ChatChunk(type="tool_call_args_delta", tool_call_id="b", args_delta="{}")
            yield ChatChunk(type="tool_call_end", tool_call_id="b")
            yield ChatChunk(type="finish", finish_reason="tool_calls")

        async def embed(self, texts, *, model):
            return []

    response = await Provider().chat(
        [ChatMessage(role="user", content="call")], model="test"
    )
    assert response.content == "éé"
    assert [call.arguments for call in response.tool_calls] == [{}, {}]


@pytest.mark.asyncio
async def test_base_chat_rejects_tool_call_count_before_retaining_call(monkeypatch):
    monkeypatch.setattr(base_module, "MAX_STREAM_TOOL_CALLS", 1)

    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            yield ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="lookup")
            yield ChatChunk(type="tool_call_start", tool_call_id="b", tool_name="lookup")

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="too many tool calls"):
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")


@pytest.mark.asyncio
async def test_base_chat_does_not_retain_empty_fragments():
    import sys

    class EmptyFragment(str):
        pass

    empty_text = EmptyFragment("")
    empty_args = EmptyFragment("")
    text_refs = sys.getrefcount(empty_text)
    args_refs = sys.getrefcount(empty_args)

    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            yield ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="lookup")
            for _ in range(10_000):
                yield ChatChunk(type="text_delta", text=empty_text)
                yield ChatChunk(type="tool_call_args_delta", tool_call_id="a", args_delta=empty_args)
            assert sys.getrefcount(empty_text) <= text_refs + 2
            assert sys.getrefcount(empty_args) <= args_refs + 2
            yield ChatChunk(type="tool_call_args_delta", tool_call_id="a", args_delta="{}")
            yield ChatChunk(type="tool_call_end", tool_call_id="a")
            yield ChatChunk(type="finish", finish_reason="tool_calls")

        async def embed(self, texts, *, model):
            return []

    response = await Provider().chat(
        [ChatMessage(role="user", content="call")], model="test"
    )

    assert response.content == ""
    assert [call.arguments for call in response.tool_calls] == [{}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("empty_chunks", "should_succeed"),
    [(0, True), (2, True), (3, False)],
    ids=["under-limit", "at-limit", "over-limit"],
)
async def test_base_chat_bounds_every_chunk_before_empty_suppression(
    monkeypatch, empty_chunks, should_succeed
):
    monkeypatch.setattr(base_module, "MAX_STREAM_CHUNKS", 4)

    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            for _ in range(empty_chunks):
                yield ChatChunk(type="text_delta", text="")
            yield ChatChunk(type="text_delta", text="ok")
            yield ChatChunk(type="finish", finish_reason="stop")

        async def embed(self, texts, *, model):
            return []

    response = Provider().chat([ChatMessage(role="user", content="call")], model="test")
    if should_succeed:
        assert (await response).content == "ok"
    else:
        with pytest.raises(LLMUnavailableError, match="too many chunks"):
            await response


@pytest.mark.asyncio
async def test_base_chat_stops_an_unbounded_empty_chunk_generator(monkeypatch):
    monkeypatch.setattr(base_module, "MAX_STREAM_CHUNKS", 3)
    yielded = 0

    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            nonlocal yielded
            while True:
                yielded += 1
                yield ChatChunk(type="text_delta", text="")

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="too many chunks"):
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")
    assert yielded == 4


@pytest.mark.asyncio
async def test_base_chat_rejects_unknown_chunk_after_buffering_state():
    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            yield ChatChunk(type="text_delta", text="buffered-secret")
            yield ChatChunk(type="tool_call_start", tool_call_id="call", tool_name="lookup")
            yield ChatChunk(type="tool_call_args_delta", tool_call_id="call", args_delta="{}")
            yield ChatChunk(type=cast(Any, "bogus"))
            yield ChatChunk(type="tool_call_end", tool_call_id="call")
            yield ChatChunk(type="finish", finish_reason="tool_calls")

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="unknown chunk type") as exc_info:
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")
    assert "buffered-secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_base_chat_counts_unknown_chunks_before_type_inspection(monkeypatch):
    monkeypatch.setattr(base_module, "MAX_STREAM_CHUNKS", 1)

    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            yield ChatChunk(type="text_delta", text="")
            yield ChatChunk(type=cast(Any, "bogus"))

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="too many chunks"):
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks",
    [
        [ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="lookup")],
        [ChatChunk(type="text_delta", text="partial")],
        [
            ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="lookup"),
            ChatChunk(type="tool_call_args_delta", tool_call_id="a", args_delta="{}"),
            ChatChunk(type="finish", finish_reason="tool_calls"),
        ],
        [
            ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="lookup"),
            ChatChunk(type="tool_call_end", tool_call_id="a"),
            ChatChunk(type="tool_call_args_delta", tool_call_id="a", args_delta="{}"),
            ChatChunk(type="finish", finish_reason="tool_calls"),
        ],
        [
            ChatChunk(type="finish", finish_reason="stop"),
            ChatChunk(type="text_delta", text="late"),
        ],
    ],
    ids=["open-tool-eof", "missing-finish", "finish-before-tool-end", "args-after-end", "after-finish"],
)
async def test_base_chat_rejects_incomplete_or_post_terminal_streams(chunks):
    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            for chunk in chunks:
                yield chunk

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="incomplete|continued"):
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks",
    [
        [ChatChunk(type="text_delta", text="bad\x00text"), ChatChunk(type="finish")],
        [
            ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="lookup"),
            ChatChunk(
                type="tool_call_args_delta",
                tool_call_id="a",
                args_delta='{"value":"bad\x00argument"}',
            ),
        ],
        [
            ChatChunk(type="tool_call_start", tool_call_id="abc", tool_name="de"),
            ChatChunk(type="tool_call_end", tool_call_id="abc"),
            ChatChunk(type="finish", finish_reason="tool_calls"),
        ],
        [
            ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="b"),
            ChatChunk(type="tool_call_end", tool_call_id="a", thought_signature="abcde"),
            ChatChunk(type="finish", finish_reason="tool_calls"),
        ],
        [
            ChatChunk(
                type="usage",
                usage=base_module.Usage(input_tokens=2_147_483_648, output_tokens=0),
            ),
            ChatChunk(type="finish"),
        ],
    ],
    ids=["nul-text", "nul-argument", "aggregate-metadata", "signature", "usage"],
)
async def test_base_chat_rejects_unpersistable_text_metadata_and_usage(monkeypatch, chunks):
    monkeypatch.setattr(base_module, "MAX_STREAM_TOOL_METADATA_BYTES", 4)
    monkeypatch.setattr(base_module, "MAX_STREAM_ALL_TOOL_METADATA_BYTES", 4)

    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            for chunk in chunks:
                yield chunk

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="invalid|size limit"):
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")


@pytest.mark.asyncio
async def test_base_chat_rejects_numeric_overflow_arguments():
    class Provider(LLMProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            yield ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="lookup")
            yield ChatChunk(
                type="tool_call_args_delta", tool_call_id="a", args_delta='{"x":1e999}'
            )
            yield ChatChunk(type="tool_call_end", tool_call_id="a")
            yield ChatChunk(type="finish", finish_reason="tool_calls")

        async def embed(self, texts, *, model):
            return []

    with pytest.raises(LLMUnavailableError, match="invalid tool arguments"):
        await Provider().chat([ChatMessage(role="user", content="call")], model="test")


# --------------------------------------------------------------------- OpenAI-compatible: chat_stream

@pytest.mark.asyncio
@pytest.mark.parametrize("provider_class", [OpenAICompatibleProvider, OpenAIProvider])
@pytest.mark.parametrize("signature_timing", ["first", "late", "absent"])
async def test_openai_tool_signatures_survive_multiple_rounds(provider_class, signature_timing):
    """Keep signatures on their own calls across parallel and sequential tools."""
    def response(deltas):
        return _FakeStreamResponse(status_code=200, lines=[
            "data: " + json.dumps({"choices": [{"delta": delta}]}) for delta in deltas
        ] + ['data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}', "data: [DONE]"])

    signature_a = "opaque-A+/=="
    first = {"index": 0, "id": "a", "function": {"name": "lookup", "arguments": '{"city":'}}
    extra_a = {"google": {"thought_signature": signature_a}}
    if signature_timing == "first":
        first["extra_content"] = extra_a
    deltas = [
        {"tool_calls": [first]},
        {"tool_calls": [{"index": 1, "id": "b", "function": {"name": "lookup", "arguments": "{}"}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]},
    ]
    if signature_timing == "late":
        deltas.append({"tool_calls": [{"index": 0, "extra_content": extra_a}]})
    second_call = {"index": 0, "id": "c", "function": {"name": "lookup", "arguments": "{}"}}
    if signature_timing != "absent":
        second_call["extra_content"] = {"google": {"thought_signature": "opaque-B=="}}
    _FakeAsyncClient.queue_stream.extend([
        response(deltas), response([{"tool_calls": [second_call]}]),
        response([{"content": "Done"}]),
    ])
    provider = provider_class(base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    messages = [ChatMessage(role="user", content="Look up cities")]
    for round_index in range(2):
        result = await provider.chat(messages, model="gemini-test")
        if round_index == 0:
            assert [tc.id for tc in result.tool_calls] == ["a", "b"]
            assert result.tool_calls[0].arguments == {"city": "Paris"}
        messages.append(ChatMessage(role="assistant", tool_calls=result.tool_calls))
        messages.extend(ChatMessage(role="tool", tool_call_id=tc.id, content="ok") for tc in result.tool_calls)
    await provider.chat(messages, model="gemini-test")

    assert len(_FakeAsyncClient.posted) == 3
    for url, payload in _FakeAsyncClient.posted:
        assert url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        for message in payload["messages"]:
            for call in message.get("tool_calls", []):
                if signature_timing == "absent" or call["id"] == "b":
                    assert set(call) == {"id", "type", "function"}
                else:
                    expected = signature_a if call["id"] == "a" else "opaque-B=="
                    assert call["extra_content"] == {"google": {"thought_signature": expected}}
    final_calls = [tc for m in _FakeAsyncClient.posted[-1][1]["messages"] for tc in m.get("tool_calls", [])]
    assert [tc["id"] for tc in final_calls] == ["a", "b", "c"]


# --------------------------------------------------------------------- Ollama: chat_stream

@pytest.mark.asyncio
async def test_ollama_chat_stream_yields_text_and_finish():
    """Two JSONL deltas + a final done frame should produce text deltas
    plus usage + finish chunks."""
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=[
        json.dumps({"message": {"content": "Hel"}}),
        json.dumps({"message": {"content": "lo"}}),
        json.dumps({
            "message": {"content": ""},
            "done": True, "done_reason": "stop",
            "prompt_eval_count": 7, "eval_count": 4,
        }),
        "",  # blank line — should be skipped
    ]))

    provider = OllamaProvider(base_url="http://ollama:11434")
    chunks = []
    async for chunk in provider.chat_stream(
        [ChatMessage(role="user", content="hi")], model="llama3", temperature=0.1, max_tokens=64,
    ):
        chunks.append(chunk)

    text = "".join(c.text or "" for c in chunks if c.type == "text_delta")
    assert text == "Hello"
    assert any(c.type == "usage" and c.usage and c.usage.input_tokens == 7 for c in chunks)
    assert any(c.type == "finish" and c.finish_reason == "stop" for c in chunks)

    # Verify the request payload picked up our temperature + num_predict.
    _, payload = _FakeAsyncClient.posted[-1]
    assert payload["options"]["temperature"] == 0.1
    assert payload["options"]["num_predict"] == 64


@pytest.mark.asyncio
async def test_ollama_chat_stream_emits_tool_call_events():
    """A streaming tool_calls payload should produce start/args/end chunks."""
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=[
        json.dumps({
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "list_accounts", "arguments": {"limit": 5}}},
                ],
            },
        }),
        json.dumps({"message": {"content": ""}, "done": True, "done_reason": "tool_calls"}),
    ]))

    provider = OllamaProvider()
    types = []
    async for c in provider.chat_stream([ChatMessage(role="user", content="x")], model="llama3"):
        types.append(c.type)
    assert "tool_call_start" in types
    assert "tool_call_args_delta" in types
    assert "tool_call_end" in types


@pytest.mark.asyncio
async def test_ollama_chat_stream_raises_on_http_error():
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(
        status_code=503, lines=[], body=b"unavailable",
    ))
    provider = OllamaProvider()
    with pytest.raises(LLMUnavailableError):
        async for _ in provider.chat_stream(
            [ChatMessage(role="user", content="x")], model="llama3",
        ):
            pass


@pytest.mark.asyncio
async def test_ollama_chat_stream_includes_tools_in_payload():
    """When tools are passed, they must show up in the request body."""
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=[
        json.dumps({"message": {"content": ""}, "done": True, "done_reason": "stop"}),
    ]))
    provider = OllamaProvider()
    tools = [ToolDefinition(name="t", description="d", parameters={"type": "object"})]
    async for _ in provider.chat_stream(
        [ChatMessage(role="user", content="x")], model="llama3", tools=tools,
    ):
        pass
    _, payload = _FakeAsyncClient.posted[-1]
    assert payload["tools"][0]["function"]["name"] == "t"


# --------------------------------------------------------------------- Ollama: embed

@pytest.mark.asyncio
async def test_ollama_embed_returns_vectors_per_input():
    _FakeAsyncClient.queue_post.append(_FakeResponse(status_code=200, json_body={
        "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    }))
    provider = OllamaProvider()
    out = await provider.embed(["a", "b"], model="nomic-embed-text")
    assert out == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


@pytest.mark.asyncio
async def test_ollama_embed_handles_missing_embeddings_array():
    """If the server returns an unexpected shape, we should yield [] rather than crash."""
    _FakeAsyncClient.queue_post.append(_FakeResponse(status_code=200, json_body={"unexpected": True}))
    provider = OllamaProvider()
    out = await provider.embed(["a"], model="nomic-embed-text")
    assert out == []


@pytest.mark.asyncio
async def test_ollama_embed_raises_on_http_error():
    _FakeAsyncClient.queue_post.append(_FakeResponse(status_code=500, text="boom"))
    provider = OllamaProvider()
    with pytest.raises(LLMUnavailableError):
        await provider.embed(["a"], model="nomic-embed-text")


# --------------------------------------------------------------------- Anthropic: chat_stream

@pytest.mark.asyncio
async def test_anthropic_chat_stream_parses_text_block_sequence():
    sse = [
        f"data: {json.dumps({'type': 'message_start', 'message': {'usage': {'input_tokens': 5, 'output_tokens': 0}}})}",
        f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text'}})}",
        f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hi'}})}",
        f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': '!'}})}",
        f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}",
        f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 2}})}",
        "",  # blank — skipped
        "non-sse line — should be ignored",  # not 'data:' prefixed
    ]
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=sse))

    provider = AnthropicProvider(api_key="sk-test")
    chunks = []
    async for c in provider.chat_stream([ChatMessage(role="user", content="hi")], model="claude-x"):
        chunks.append(c)

    text = "".join(c.text or "" for c in chunks if c.type == "text_delta")
    assert text == "Hi!"
    assert any(c.type == "usage" and c.usage is not None and c.usage.output_tokens == 2 for c in chunks)
    assert any(c.type == "finish" and c.finish_reason == "end_turn" for c in chunks)


@pytest.mark.asyncio
async def test_anthropic_chat_stream_parses_tool_use_block():
    # Build SSE lines via explicit json.dumps payloads — avoids backslash
    # escapes inside f-strings, which Python 3.11 doesn't allow.
    events = [
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "toolu_1", "name": "list_accounts"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '{"limit":'}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": "5}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta",
         "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 1}},
    ]
    sse = ["data: " + json.dumps(e) for e in events]
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=sse))

    provider = AnthropicProvider(api_key="sk-test")
    chunks = []
    async for c in provider.chat_stream([ChatMessage(role="user", content="x")], model="claude-x"):
        chunks.append(c)

    starts = [c for c in chunks if c.type == "tool_call_start"]
    args = [c for c in chunks if c.type == "tool_call_args_delta"]
    ends = [c for c in chunks if c.type == "tool_call_end"]
    assert starts and starts[0].tool_name == "list_accounts"
    assert "".join(c.args_delta or "" for c in args) == '{"limit":5}'
    assert ends and ends[0].tool_call_id == "toolu_1"


@pytest.mark.asyncio
async def test_anthropic_includes_system_and_tools_in_payload():
    sse = [f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {}})}"]
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=sse))
    provider = AnthropicProvider(api_key="sk-test")
    tools = [ToolDefinition(name="t", description="d", parameters={"type": "object"})]
    msgs = [
        ChatMessage(role="system", content="part1"),
        ChatMessage(role="system", content="part2"),
        ChatMessage(role="user", content="hi"),
    ]
    async for _ in provider.chat_stream(msgs, model="claude-x", tools=tools, max_tokens=200):
        pass
    _, payload = _FakeAsyncClient.posted[-1]
    assert payload["system"] == "part1\n\npart2"
    assert payload["max_tokens"] == 200
    assert payload["tools"][0]["name"] == "t"


@pytest.mark.asyncio
async def test_anthropic_payload_omits_temperature():
    sse = [f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {}})}"]
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(status_code=200, lines=sse))
    provider = AnthropicProvider(api_key="sk-test")
    msgs = [ChatMessage(role="user", content="hi")]
    async for _ in provider.chat_stream(msgs, model="claude-x", temperature=0.2):
        pass
    _, payload = _FakeAsyncClient.posted[-1]
    assert "temperature" not in payload


# --------------------------------------------------------------------- Anthropic: HTTP errors → typed exceptions

@pytest.mark.asyncio
@pytest.mark.parametrize("status,exc_type", [
    (401, LLMAuthError),
    (403, LLMAuthError),
    (429, LLMRateLimitError),
    (500, LLMUnavailableError),
    (502, LLMUnavailableError),
    (400, LLMUnavailableError),
])
async def test_anthropic_http_errors_map_to_typed_exceptions(status, exc_type):
    _FakeAsyncClient.queue_stream.append(_FakeStreamResponse(
        status_code=status, lines=[], body=b'{"error":{"message":"x"}}',
    ))
    provider = AnthropicProvider(api_key="sk-test")
    with pytest.raises(exc_type):
        async for _ in provider.chat_stream([ChatMessage(role="user", content="x")], model="claude-x"):
            pass


@pytest.mark.asyncio
async def test_anthropic_embed_raises_not_supported():
    provider = AnthropicProvider(api_key="sk-test")
    with pytest.raises(LLMNotSupportedError):
        await provider.embed(["a"], model="x")
