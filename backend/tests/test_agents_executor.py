"""End-to-end-style executor tests with a scripted LLM and a fake MCP.

These exercise the full agent runtime — turn loop, tool dispatch, message
persistence, usage logging, error classification — without ever hitting a
real LLM or MCP server.

Pattern:
  - `_ScriptedProvider` yields a predetermined sequence of ChatChunks.
  - `_FakeMCP` exposes one synthetic tool and records calls.
  - We patch `_provider_for` and pass our fake MCP into AgentExecutor.
"""
import asyncio
import uuid
from contextvars import ContextVar
from copy import deepcopy
from typing import Any, AsyncIterator, cast
from unittest.mock import patch

import pytest
from sqlalchemy import select

import app.agents.runtime.executor as executor_module
from app.agents.mcp.client import MCPRegistry, ToolHandle
from app.agents.models.agent import Agent
from app.agents.models.conversation import Conversation, Message
from app.agents.models.usage import LlmUsage
from app.agents.providers.base import (
    ChatChunk,
    LLMAuthError,
    LLMError,
    LLMNotSupportedError,
    LLMProvider,
    LLMRateLimitError,
    LLMUnavailableError,
    Usage,
)
from app.agents.runtime.executor import (
    AgentExecutor,
    ExecutorEvent,
    _classify_error,
    _provider_and_model_for,
)
from app.agents.services import conversation_service


pytestmark = pytest.mark.asyncio


# --- Fakes -----------------------------------------------------------------


class _ScriptedProvider(LLMProvider):
    """Yields chunks from a list of "turns". Each turn is a list of chunks
    that simulate one model round. The provider iterates turns across
    successive chat_stream() calls."""

    name = "openai"  # match cost-table key for usage tests

    def __init__(self, turns: list[list[ChatChunk]]):
        super().__init__(api_key="x")
        self._turns = list(turns)
        self.requests = []

    async def chat_stream(  # type: ignore[override]
        self, messages, *, model, tools=None, temperature=0.4, max_tokens=None
    ) -> AsyncIterator[ChatChunk]:
        self.requests.append(deepcopy(messages))
        if not self._turns:
            # No more scripted turns — emit a generic finish.
            yield ChatChunk(type="finish", finish_reason="stop")
            return
        turn = self._turns.pop(0)
        for c in turn:
            yield c

    async def embed(self, texts, *, model):
        return [[0.0] * 4 for _ in texts]


class _FakeMCP(MCPRegistry):
    """In-process MCPRegistry that doesn't open any sockets. Discovers a
    single tool and records every call() invocation."""

    def __init__(self, *, tools: list[ToolHandle], canned_result: dict | None = None):
        # Skip parent __init__ so we don't try to construct real clients.
        self._tools = tools
        self._canned = canned_result or {"ok": True, "data": {"items": [{"hello": "world"}]}, "text": "ok"}
        self.calls: list[tuple[str, dict]] = []

    def server_names(self):
        return ["securo"]

    async def discover(self, *, workspace_id=None, user_id, conversation_id=None, agent_id=None):
        return list(self._tools)

    async def call(self, *, wire_name, arguments, workspace_id=None, user_id, conversation_id=None, agent_id=None):
        self.calls.append((wire_name, arguments))
        return self._canned


# --- Helpers --------------------------------------------------------------


def _patch_provider(p: LLMProvider):
    return patch("app.agents.runtime.executor._provider_for", return_value=p)


async def _drain(executor: AgentExecutor, **kwargs) -> list[ExecutorEvent]:
    return [ev async for ev in executor.run(**kwargs)]


# --- Tests ----------------------------------------------------------------


async def test_explicit_missing_connection_never_falls_back_to_environment(
    session, test_agent
):
    test_agent.connection_id = uuid.uuid4()
    await session.commit()

    with pytest.raises(LLMAuthError, match="configured connection"):
        await _provider_and_model_for(session, test_agent)


async def test_explicit_chatgpt_without_credential_fails_closed_as_auth(
    session, test_user, test_agent, test_conversation
):
    from app.agents.services import connection_service

    connection = await connection_service.create_connection(
        session,
        test_user.id,
        name="ChatGPT",
        kind="openai_chatgpt",
        default_model="gpt-plan",
    )
    test_agent.connection_id = connection.id
    await session.commit()

    with patch(
        "app.agents.runtime.executor._provider_for",
        side_effect=AssertionError("environment fallback must not run"),
    ):
        events = await _drain(
            AgentExecutor(mcp=_FakeMCP(tools=[])),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="hello",
        )

    error = next(event for event in events if event.type == "error")
    assert error.error_code == "auth"
    assert events[-1].type == "done"
    assert events[-1].finish_reason == "error"


async def test_explicit_broken_connection_never_uses_environment_fallback(
    session, test_user, test_agent, test_conversation
):
    from app.agents.services import connection_service

    connection = await connection_service.create_connection(
        session,
        test_user.id,
        name="Broken",
        kind="openai",
        default_model="gpt-test",
    )
    test_agent.connection_id = connection.id
    await session.commit()

    with (
        patch.object(
            connection_service,
            "build_provider_for_connection",
            side_effect=ValueError("broken explicit connection"),
        ),
        patch(
            "app.agents.runtime.executor._provider_for",
            side_effect=AssertionError("environment fallback must not run"),
        ),
    ):
        events = await _drain(
            AgentExecutor(mcp=_FakeMCP(tools=[])),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="hello",
        )

    assert [event.type for event in events[-2:]] == ["error", "done"]
    assert events[-1].finish_reason == "error"


async def test_simple_text_response_no_tools(session, test_user, test_agent: Agent, test_conversation: Conversation):
    """LLM returns a plain answer; no tool calls."""
    provider = _ScriptedProvider([[
        ChatChunk(type="text_delta", text="Hello "),
        ChatChunk(type="text_delta", text="world!"),
        ChatChunk(type="usage", usage=Usage(input_tokens=10, output_tokens=4)),
        ChatChunk(type="finish", finish_reason="stop"),
    ]])

    executor = AgentExecutor(mcp=_FakeMCP(tools=[]))
    with _patch_provider(provider):
        events = await _drain(
            executor,
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="hi",
        )

    text_events = [e for e in events if e.type == "text_delta"]
    assert "".join(e.text or "" for e in text_events) == "Hello world!"
    assert events[-1].type == "done"
    assert events[-1].finish_reason == "stop"

    # Persisted: user msg + assistant msg.
    msgs = (await session.execute(
        select(Message).where(Message.conversation_id == test_conversation.id).order_by(Message.ordinal)
    )).scalars().all()
    roles = [m.role for m in msgs]
    assert roles == ["user", "assistant"]
    assert msgs[1].content == "Hello world!"
    assert msgs[1].input_tokens == 10
    assert msgs[1].output_tokens == 4


async def test_usage_row_recorded(session, test_user, test_agent, test_conversation):
    provider = _ScriptedProvider([[
        ChatChunk(type="text_delta", text="ok"),
        ChatChunk(type="usage", usage=Usage(input_tokens=100, output_tokens=20)),
        ChatChunk(type="finish", finish_reason="stop"),
    ]])
    executor = AgentExecutor(mcp=_FakeMCP(tools=[]))
    with _patch_provider(provider):
        await _drain(
            executor,
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="hi",
        )
    rows = (await session.execute(
        select(LlmUsage).where(LlmUsage.conversation_id == test_conversation.id)
    )).scalars().all()
    assert len(rows) == 1
    assert rows[0].input_tokens == 100
    assert rows[0].output_tokens == 20
    assert rows[0].provider == "openai"
    # gpt-4o-mini rates: (0.15, 0.60) per 1M.
    assert rows[0].cost_usd is not None
    assert float(rows[0].cost_usd) > 0


@pytest.mark.parametrize("signature", [None, "opaque-signature+/=="])
async def test_tool_call_dispatch_and_result_persisted(session, test_user, test_agent, test_conversation, signature):
    """LLM emits a tool call; executor runs it via fake MCP and feeds the
    result back; second turn returns a plain answer."""
    tools = [ToolHandle(server="securo", name="list_accounts", description="d", parameters={"type": "object"})]
    fake_mcp = _FakeMCP(tools=tools, canned_result={
        "ok": True, "data": {"items": [{"id": "a1", "name": "Checking"}]}, "text": "1 account",
    })

    provider = _ScriptedProvider([
        # Turn 1: emit a tool call.
        [
            ChatChunk(type="tool_call_start", tool_call_id="t1", tool_name="securo__list_accounts"),
            ChatChunk(type="tool_call_args_delta", tool_call_id="t1", args_delta="{}"),
            ChatChunk(type="tool_call_end", tool_call_id="t1", thought_signature=signature),
            ChatChunk(type="usage", usage=Usage(input_tokens=20, output_tokens=5)),
            ChatChunk(type="finish", finish_reason="tool_calls"),
        ],
        # Turn 2: respond with the result.
        [
            ChatChunk(type="text_delta", text="You have 1 account."),
            ChatChunk(type="usage", usage=Usage(input_tokens=30, output_tokens=8)),
            ChatChunk(type="finish", finish_reason="stop"),
        ],
    ])

    executor = AgentExecutor(mcp=fake_mcp)
    with _patch_provider(provider):
        events = await _drain(
            executor,
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="how many accounts?",
        )

    # Event stream: tool_call → tool_result → text_delta → done.
    types = [e.type for e in events]
    assert "tool_call" in types
    assert "tool_result" in types
    assert types[-1] == "done"

    # MCP got called with the right wire name.
    assert fake_mcp.calls == [("securo__list_accounts", {})]

    # DB state: user, assistant(turn1, with tool_calls), tool, assistant(turn2, with text).
    msgs = (await session.execute(
        select(Message).where(Message.conversation_id == test_conversation.id).order_by(Message.ordinal)
    )).scalars().all()
    roles = [m.role for m in msgs]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert msgs[1].tool_calls and msgs[1].tool_calls[0]["name"] == "securo__list_accounts"
    assert msgs[3].content == "You have 1 account."

    # Two usage rows, one per provider call.
    usage_rows = (await session.execute(
        select(LlmUsage).where(LlmUsage.conversation_id == test_conversation.id)
    )).scalars().all()
    assert len(usage_rows) == 2

    saved_call = msgs[1].tool_calls[0]
    if signature is None:
        assert "thought_signature" not in saved_call
    else:
        assert saved_call["thought_signature"] == signature
    replayed = [tc for m in provider.requests[1] for tc in m.tool_calls]
    assert replayed[0].thought_signature == signature

    # Start a separate request so history is reconstructed from persisted JSON.
    session.expire(msgs[1])
    with _patch_provider(provider):
        await _drain(
            executor, session=session, agent=test_agent, user_id=test_user.id,
            conversation_id=test_conversation.id, user_message="thanks",
        )
    restored = [tc for m in provider.requests[2] for tc in m.tool_calls]
    assert restored[0].thought_signature == signature


async def test_provider_auth_error_surfaced_as_friendly_message(
    session, test_user, test_agent, test_conversation
):
    """An LLMAuthError becomes an error event with code=auth, not a 500."""

    class _BoomProvider(LLMProvider):
        name = "openai"

        async def chat_stream(self, *args, **kwargs):  # type: ignore[override]
            raise LLMAuthError("missing key")
            yield  # unreachable, makes this a generator

        async def embed(self, texts, *, model):
            return []

    executor = AgentExecutor(mcp=_FakeMCP(tools=[]))
    with _patch_provider(_BoomProvider()):
        events = await _drain(
            executor,
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="hi",
        )

    err = next((e for e in events if e.type == "error"), None)
    assert err is not None
    assert err.error_code == "auth"
    done = next((e for e in events if e.type == "done"), None)
    assert done is not None and done.finish_reason == "error"


async def test_error_classification_is_total_and_never_exposes_exception_text():
    sentinel = "classifier-bearer-sentinel"

    class HostileUnavailable(LLMUnavailableError):
        def __getattribute__(self, _name):
            raise RuntimeError("classifier-accessor-sentinel")

        def __str__(self):
            raise RuntimeError("classifier-string-sentinel")

    assert _classify_error(LLMAuthError(sentinel)) == (
        "auth",
        "LLM provider rejected the credentials.",
    )
    assert _classify_error(LLMRateLimitError(sentinel)) == (
        "rate_limit",
        "LLM provider is rate-limiting.",
    )
    assert _classify_error(LLMUnavailableError(sentinel)) == (
        "unavailable",
        "LLM provider is unavailable.",
    )
    assert _classify_error(LLMNotSupportedError(sentinel)) == (
        "not_supported",
        "LLM provider does not support this request.",
    )
    assert _classify_error(LLMError(sentinel)) == ("unknown", "LLM request failed.")
    assert _classify_error(HostileUnavailable(sentinel)) == (
        "unavailable",
        "LLM provider is unavailable.",
    )
    assert _classify_error(RuntimeError(sentinel)) == ("unknown", "LLM request failed.")


async def test_executor_sanitizes_exact_llm_failure_in_event_log_and_database(
    session, test_user, test_agent, test_conversation, caplog
):
    sentinel = "executor-exact-llm-bearer-sentinel"

    class Provider(_ScriptedProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            raise LLMUnavailableError(sentinel)
            yield  # pragma: no cover

    with _patch_provider(Provider([])):
        events = await _drain(
            AgentExecutor(mcp=_FakeMCP(tools=[])),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="sanitized provider failure",
        )

    assert [(event.type, event.error_code, event.error_message) for event in events] == [
        ("error", "unavailable", "LLM provider is unavailable."),
        ("done", None, None),
    ]
    assert sentinel not in caplog.text
    messages = (
        await session.execute(
            select(Message).where(Message.conversation_id == test_conversation.id)
        )
    ).scalars().all()
    assert sentinel not in repr([(message.role, message.content) for message in messages])


async def test_executor_sanitizes_hostile_llm_subclass_accessors(
    session, test_user, test_agent, test_conversation, caplog
):
    sentinel = "executor-hostile-llm-bearer-sentinel"

    class HostileUnavailable(LLMUnavailableError):
        def __getattribute__(self, _name):
            raise RuntimeError("executor-hostile-accessor-sentinel")

        def __str__(self):
            raise RuntimeError("executor-hostile-string-sentinel")

    class Provider(_ScriptedProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            raise HostileUnavailable(sentinel)
            yield  # pragma: no cover

    with _patch_provider(Provider([])):
        events = await _drain(
            AgentExecutor(mcp=_FakeMCP(tools=[])),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="hostile provider failure",
        )

    assert [(event.type, event.error_code, event.error_message) for event in events] == [
        ("error", "unavailable", "LLM provider is unavailable."),
        ("done", None, None),
    ]
    assert "sentinel" not in caplog.text


async def test_no_model_configured_yields_config_error(session, test_user, test_conversation):
    """An agent without model and without AGENTS_DEFAULT_MODEL should bail
    early with a config error rather than calling the provider."""
    bare_agent = Agent(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="Bare",
        provider="openai",
        model=None,  # the missing piece
    )
    session.add(bare_agent)
    await session.commit()

    # Reuse the existing conversation but switch its agent to the bare one.
    test_conversation.agent_id = bare_agent.id
    await session.commit()

    executor = AgentExecutor(mcp=_FakeMCP(tools=[]))
    # Provider doesn't matter here — should never be called.
    provider = _ScriptedProvider([])
    with _patch_provider(provider), patch.dict("os.environ", {"AGENTS_DEFAULT_MODEL": ""}, clear=False):
        events = await _drain(
            executor,
            session=session,
            agent=bare_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="hi",
        )
    err = next((e for e in events if e.type == "error"), None)
    assert err is not None and err.error_code == "config"


async def test_per_agent_tool_whitelist_filters_discovery(
    session, test_user, test_agent, test_conversation
):
    """If the agent has explicit tool whitelist rows, the executor only
    sends the enabled ones to the LLM."""
    from app.agents.services import agent_service

    # Discover 2 tools, but only enable one.
    tools = [
        ToolHandle(server="securo", name="list_accounts", description="", parameters={"type": "object"}),
        ToolHandle(server="securo", name="list_categories", description="", parameters={"type": "object"}),
    ]
    fake_mcp = _FakeMCP(tools=tools)
    await agent_service.replace_tool_enablement(
        session, test_agent.id,
        [("securo", "list_accounts", True), ("securo", "list_categories", False)],
    )

    captured_tool_names: list[str] = []

    class _Capture(_ScriptedProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            captured_tool_names.extend([t.name for t in (tools or [])])
            async for c in super().chat_stream(messages, model=model, tools=tools, temperature=temperature, max_tokens=max_tokens):
                yield c

    provider = _Capture([[
        ChatChunk(type="text_delta", text="ok"),
        ChatChunk(type="finish", finish_reason="stop"),
    ]])

    executor = AgentExecutor(mcp=fake_mcp)
    with _patch_provider(provider):
        await _drain(
            executor,
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="hi",
        )

    assert captured_tool_names == ["securo__list_accounts"]


@pytest.mark.parametrize(
    ("enabled", "returned_name", "expected_calls"),
    [
        (False, "securo__list_accounts", []),
        (True, "list_accounts", []),
        (True, "other__list_accounts", []),
        (True, "securo__list_accounts", [("securo__list_accounts", {})]),
    ],
)
async def test_model_can_dispatch_only_exact_tool_wire_names_advertised_this_turn(
    session,
    test_user,
    test_agent,
    test_conversation,
    enabled,
    returned_name,
    expected_calls,
):
    from app.agents.services import agent_service

    fake_mcp = _FakeMCP(
        tools=[
            ToolHandle(
                server="securo",
                name="list_accounts",
                description="d",
                parameters={"type": "object"},
            )
        ]
    )
    await agent_service.replace_tool_enablement(
        session, test_agent.id, [("securo", "list_accounts", enabled)]
    )
    provider = _ScriptedProvider(
        [
            [
                ChatChunk(type="tool_call_start", tool_call_id="t1", tool_name=returned_name),
                ChatChunk(type="tool_call_args_delta", tool_call_id="t1", args_delta="{}"),
                ChatChunk(type="tool_call_end", tool_call_id="t1"),
                ChatChunk(type="finish", finish_reason="tool_calls"),
            ],
            [ChatChunk(type="text_delta", text="done"), ChatChunk(type="finish", finish_reason="stop")],
        ]
    )

    with _patch_provider(provider):
        await _drain(
            AgentExecutor(mcp=fake_mcp),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="call it",
        )

    assert fake_mcp.calls == expected_calls


@pytest.mark.parametrize(
    "returned_names",
    [
        ["other__list_accounts"],
        ["list_accounts"],
        ["securo__other__list_accounts"],
        ["securo__list_accounts", "securo__list_accounts"],
        ["securo__list_accounts", "other__list_accounts"],
    ],
    ids=[
        "unadvertised",
        "bare-alias",
        "namespace-confusion",
        "duplicate-wire-name",
        "mixed-authorized-and-unauthorized",
    ],
)
async def test_tool_call_batch_is_authorized_atomically_before_any_model_side_effect(
    session,
    test_user,
    test_agent,
    test_conversation,
    returned_names,
):
    from app.agents.services import agent_service

    fake_mcp = _FakeMCP(
        tools=[
            ToolHandle(
                server="securo",
                name="list_accounts",
                description="d",
                parameters={"type": "object"},
            )
        ]
    )
    await agent_service.replace_tool_enablement(
        session, test_agent.id, [("securo", "list_accounts", True)]
    )
    chunks: list[ChatChunk] = [
        ChatChunk(type="text_delta", text="must stay buffered until authorization"),
    ]
    for index, name in enumerate(returned_names):
        call_id = f"call-{index}"
        chunks.extend(
            [
                ChatChunk(type="tool_call_start", tool_call_id=call_id, tool_name=name),
                ChatChunk(type="tool_call_args_delta", tool_call_id=call_id, args_delta="{}"),
                ChatChunk(type="tool_call_end", tool_call_id=call_id),
            ]
        )
    chunks.extend(
        [
            ChatChunk(type="usage", usage=Usage(input_tokens=10, output_tokens=5)),
            ChatChunk(type="finish", finish_reason="tool_calls"),
        ]
    )

    with _patch_provider(_ScriptedProvider([chunks])):
        events = await _drain(
            AgentExecutor(mcp=fake_mcp),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="call it",
        )

    assert fake_mcp.calls == []
    assert [event.type for event in events] == ["error", "done"]
    assert [(event.type, event.error_code, event.finish_reason) for event in events] == [
        ("error", "unavailable", None),
        ("done", None, "error"),
    ]
    messages = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == test_conversation.id)
            .order_by(Message.ordinal)
        )
    ).scalars().all()
    assert [(message.role, message.tool_calls) for message in messages] == [("user", None)]
    usage_rows = (
        await session.execute(
            select(LlmUsage).where(LlmUsage.conversation_id == test_conversation.id)
        )
    ).scalars().all()
    assert usage_rows == []


@pytest.mark.parametrize(
    ("raw_args", "secret"),
    [
        ("{raw-argument-secret-sentinel", "raw-argument-secret-sentinel"),
        ('{"x":1e999}', "1e999"),
        ('{"\\ud800":1}', "\\ud800"),
    ],
    ids=["malformed", "numeric-overflow", "surrogate-key"],
)
async def test_invalid_advertised_tool_arguments_fail_before_dispatch_or_persistence(
    session, test_user, test_agent, test_conversation, raw_args, secret
):
    fake_mcp = _FakeMCP(
        tools=[
            ToolHandle(
                server="securo",
                name="list_accounts",
                description="d",
                parameters={"type": "object"},
            )
        ]
    )
    provider = _ScriptedProvider(
        [
            [
                ChatChunk(
                    type="tool_call_start",
                    tool_call_id="malformed-call",
                    tool_name="securo__list_accounts",
                ),
                ChatChunk(
                    type="tool_call_args_delta",
                    tool_call_id="malformed-call",
                    args_delta=raw_args,
                ),
                ChatChunk(type="tool_call_end", tool_call_id="malformed-call"),
                ChatChunk(type="finish", finish_reason="tool_calls"),
            ]
        ]
    )

    with _patch_provider(provider):
        events = await _drain(
            AgentExecutor(mcp=fake_mcp),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="call it",
        )

    assert fake_mcp.calls == []
    assert [(event.type, event.error_code, event.finish_reason) for event in events[-2:]] == [
        ("error", "unavailable", None),
        ("done", None, "error"),
    ]
    assert all(secret not in (event.error_message or "") for event in events)
    messages = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == test_conversation.id)
            .order_by(Message.ordinal)
        )
    ).scalars().all()
    assert [(message.role, message.tool_calls) for message in messages] == [("user", None)]


@pytest.mark.parametrize(
    "chunks",
    [
        [
            ChatChunk(
                type="tool_call_start",
                tool_call_id="call",
                tool_name="securo__list_accounts",
            ),
        ],
        [ChatChunk(type="text_delta", text="partial")],
        [
            ChatChunk(
                type="tool_call_start",
                tool_call_id="call",
                tool_name="securo__list_accounts",
            ),
            ChatChunk(type="tool_call_args_delta", tool_call_id="call", args_delta="{}"),
            ChatChunk(type="finish", finish_reason="tool_calls"),
        ],
        [
            ChatChunk(
                type="tool_call_start",
                tool_call_id="call",
                tool_name="securo__list_accounts",
            ),
            ChatChunk(type="tool_call_end", tool_call_id="call"),
            ChatChunk(type="finish", finish_reason="tool_calls"),
            ChatChunk(type="tool_call_args_delta", tool_call_id="call", args_delta="{}"),
        ],
    ],
    ids=["open-tool-eof", "missing-finish", "finish-before-tool-end", "after-finish"],
)
async def test_incomplete_or_post_terminal_stream_fails_before_dispatch_or_persistence(
    session, test_user, test_agent, test_conversation, chunks
):
    fake_mcp = _FakeMCP(
        tools=[
            ToolHandle(
                server="securo",
                name="list_accounts",
                description="d",
                parameters={"type": "object"},
            )
        ]
    )

    with _patch_provider(_ScriptedProvider([chunks])):
        events = await _drain(
            AgentExecutor(mcp=fake_mcp),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="call it",
        )

    assert fake_mcp.calls == []
    assert [(event.type, event.error_code, event.finish_reason) for event in events[-2:]] == [
        ("error", "unavailable", None),
        ("done", None, "error"),
    ]
    messages = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == test_conversation.id)
            .order_by(Message.ordinal)
        )
    ).scalars().all()
    assert [(message.role, message.tool_calls) for message in messages] == [("user", None)]


@pytest.mark.parametrize(
    "kind", ["text", "all-tool-arguments", "tool-count", "non-string-tool-arguments"]
)
async def test_stream_aggregate_limits_fail_before_output_dispatch_or_persistence(
    session, test_user, test_agent, test_conversation, monkeypatch, kind
):
    monkeypatch.setattr(executor_module, "_MAX_LLM_TEXT_BYTES", 4)
    monkeypatch.setattr(executor_module, "_MAX_LLM_TOOL_ARGUMENT_BYTES", 4)
    monkeypatch.setattr(executor_module, "_MAX_LLM_ALL_TOOL_ARGUMENT_BYTES", 4)
    monkeypatch.setattr(executor_module, "_MAX_LLM_TOOL_CALLS", 1)
    tool = ToolHandle(
        server="securo",
        name="list_accounts",
        description="d",
        parameters={"type": "object"},
    )
    fake_mcp = _FakeMCP(tools=[tool])
    if kind == "text":
        chunks = [
            ChatChunk(type="text_delta", text="éé"),
            ChatChunk(type="text_delta", text="overflow"),
        ]
    elif kind == "all-tool-arguments":
        monkeypatch.setattr(executor_module, "_MAX_LLM_TOOL_CALLS", 2)
        chunks = [
            ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="securo__list_accounts"),
            ChatChunk(type="tool_call_args_delta", tool_call_id="a", args_delta="{}"),
            ChatChunk(type="tool_call_start", tool_call_id="b", tool_name="securo__list_accounts"),
            ChatChunk(type="tool_call_args_delta", tool_call_id="b", args_delta="é"),
            ChatChunk(type="tool_call_args_delta", tool_call_id="b", args_delta="x"),
        ]
    elif kind == "tool-count":
        chunks = [
            ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="securo__list_accounts"),
            ChatChunk(type="tool_call_start", tool_call_id="b", tool_name="securo__list_accounts"),
        ]
    else:
        chunks = [
            ChatChunk(type="tool_call_start", tool_call_id="a", tool_name="securo__list_accounts"),
            ChatChunk(
                type="tool_call_args_delta",
                tool_call_id="a",
                args_delta=[],  # ty: ignore[invalid-argument-type]
            ),
            ChatChunk(type="tool_call_end", tool_call_id="a"),
            ChatChunk(type="finish", finish_reason="tool_calls"),
        ]
    provider = _ScriptedProvider([chunks])

    with _patch_provider(provider):
        events = await _drain(
            AgentExecutor(mcp=fake_mcp),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="bounded",
        )

    assert fake_mcp.calls == []
    assert events[-2].type == "error"
    assert events[-2].error_code == "unavailable"
    assert events[-1].finish_reason == "error"
    assert all(event.text != "overflow" for event in events)
    messages = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == test_conversation.id)
            .order_by(Message.ordinal)
        )
    ).scalars().all()
    assert [(message.role, message.tool_calls) for message in messages] == [("user", None)]


async def test_process_chunk_ignores_empty_fragment_floods():
    text_buf: list[str] = []
    open_calls = {
        "a": {
            "id": "a",
            "name": "lookup",
            "args_parts": [],
            "args_bytes": 0,
            "ended": False,
        }
    }
    totals = {"text_bytes": 0, "all_args_bytes": 0}
    events = []

    for _ in range(10_000):
        events.extend(
            [
                event
                async for event in executor_module._process_chunk(
                    ChatChunk(type="text_delta", text=""),
                    text_buf,
                    open_calls,
                    totals,
                )
            ]
        )
        events.extend(
            [
                event
                async for event in executor_module._process_chunk(
                    ChatChunk(
                        type="tool_call_args_delta", tool_call_id="a", args_delta=""
                    ),
                    text_buf,
                    open_calls,
                    totals,
                )
            ]
        )

    assert events == []
    assert text_buf == []
    assert open_calls["a"]["args_parts"] == []
    assert totals == {"text_bytes": 0, "all_args_bytes": 0}


@pytest.mark.parametrize(
    ("empty_chunks", "should_succeed"),
    [(0, True), (2, True), (3, False)],
    ids=["under-limit", "at-limit", "over-limit"],
)
async def test_executor_bounds_every_chunk_before_empty_suppression(
    session,
    test_user,
    test_agent,
    test_conversation,
    monkeypatch,
    empty_chunks,
    should_succeed,
):
    monkeypatch.setattr(executor_module, "MAX_STREAM_CHUNKS", 4)
    chunks = [ChatChunk(type="text_delta", text="") for _ in range(empty_chunks)]
    chunks.extend(
        [ChatChunk(type="text_delta", text="ok"), ChatChunk(type="finish", finish_reason="stop")]
    )
    fake_mcp = _FakeMCP(tools=[])

    with _patch_provider(_ScriptedProvider([chunks])):
        events = await _drain(
            AgentExecutor(mcp=fake_mcp),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="bounded chunks",
        )

    messages = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == test_conversation.id)
            .order_by(Message.ordinal)
        )
    ).scalars().all()
    assert fake_mcp.calls == []
    if should_succeed:
        assert [event.type for event in events] == ["text_delta", "done"]
        assert [(message.role, message.content) for message in messages] == [
            ("user", "bounded chunks"),
            ("assistant", "ok"),
        ]
    else:
        assert [(event.type, event.error_code, event.finish_reason) for event in events] == [
            ("error", "unavailable", None),
            ("done", None, "error"),
        ]
        assert [message.role for message in messages] == ["user"]


async def test_executor_stops_unbounded_empty_chunks_without_model_side_effects(
    session, test_user, test_agent, test_conversation, monkeypatch
):
    monkeypatch.setattr(executor_module, "MAX_STREAM_CHUNKS", 3)
    yielded = 0

    class Provider(_ScriptedProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            nonlocal yielded
            while True:
                yielded += 1
                yield ChatChunk(type="text_delta", text="")

    fake_mcp = _FakeMCP(tools=[])
    with _patch_provider(Provider([])):
        events = await _drain(
            AgentExecutor(mcp=fake_mcp),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="infinite chunks",
        )

    assert yielded == 4
    assert fake_mcp.calls == []
    assert [event.type for event in events] == ["error", "done"]
    messages = (
        await session.execute(
            select(Message).where(Message.conversation_id == test_conversation.id)
        )
    ).scalars().all()
    assert [message.role for message in messages] == ["user"]


class _HostileChunkValue:
    def __getattribute__(self, _name):
        raise RuntimeError("model-derived-client-secret-sentinel")

    def __str__(self):
        raise RuntimeError("model-derived-client-secret-sentinel")

    def __repr__(self):
        raise RuntimeError("model-derived-client-secret-sentinel")

    def __hash__(self):
        raise RuntimeError("model-derived-client-secret-sentinel")

    def __eq__(self, _other):
        raise RuntimeError("model-derived-client-secret-sentinel")


def _uninitialized_runtime_chunk() -> ChatChunk:
    return object.__new__(ChatChunk)


def _runtime_chunk_with_deleted_field() -> ChatChunk:
    chunk = ChatChunk(type="text_delta", text="must-not-emit")
    del chunk.type
    return chunk


def _runtime_chunk_with_uninitialized_usage() -> ChatChunk:
    return ChatChunk(type="usage", usage=object.__new__(Usage))


def _runtime_chunk_with_deleted_usage_field() -> ChatChunk:
    usage = Usage(input_tokens=1, output_tokens=2)
    del usage.input_tokens
    return ChatChunk(type="usage", usage=usage)


@pytest.mark.parametrize(
    "chunks",
    [
        [_uninitialized_runtime_chunk()],
        [_runtime_chunk_with_deleted_field()],
        [_runtime_chunk_with_uninitialized_usage()],
        [_runtime_chunk_with_deleted_usage_field()],
        [ChatChunk(type="text_delta", text=cast(Any, _HostileChunkValue()))],
        [ChatChunk(type="text_delta", text="buffered-secret"), ChatChunk(type="text_delta", text="ok", tool_name="extra")],
        [ChatChunk(type="tool_call_start", tool_call_id=cast(Any, _HostileChunkValue()), tool_name="securo__list_accounts")],
        [ChatChunk(type="tool_call_start", tool_call_id="call", tool_name=cast(Any, _HostileChunkValue()))],
        [ChatChunk(type="tool_call_args_delta", tool_call_id="orphan", args_delta="")],
        [
            ChatChunk(type="tool_call_start", tool_call_id="call", tool_name="securo__list_accounts"),
            ChatChunk(type="tool_call_end", tool_call_id="call"),
            ChatChunk(type="tool_call_args_delta", tool_call_id="call", args_delta=""),
        ],
        [ChatChunk(type="tool_call_end", tool_call_id=cast(Any, _HostileChunkValue()))],
        [ChatChunk(type="usage", usage=cast(Any, _HostileChunkValue()))],
        [ChatChunk(type="usage", usage=Usage(input_tokens=cast(Any, True), output_tokens=0))],
        [
            ChatChunk(type="usage", usage=Usage(input_tokens=1, output_tokens=2)),
            ChatChunk(type="usage", usage=Usage(input_tokens=3, output_tokens=4)),
        ],
        [
            ChatChunk(type="usage", usage=Usage(input_tokens=1, output_tokens=2)),
            ChatChunk(type="text_delta", text="late"),
            ChatChunk(type="finish", finish_reason="stop"),
        ],
        [
            ChatChunk(type="tool_call_start", tool_call_id="call", tool_name="securo__list_accounts"),
            ChatChunk(type="usage", usage=Usage(input_tokens=1, output_tokens=2)),
        ],
        [ChatChunk(type="finish", finish_reason=cast(Any, []))],
        [ChatChunk(type="finish", finish_reason=None)],
        [ChatChunk(type="finish", finish_reason="stop", args_delta="extra")],
        [
            ChatChunk(type="finish", finish_reason="stop"),
            ChatChunk(type="finish", finish_reason="stop"),
        ],
        [ChatChunk(type=cast(Any, _HostileChunkValue()))],
        [ChatChunk(type=cast(Any, "bogus"))],
    ],
    ids=[
        "uninitialized-chunk", "deleted-chunk-field", "uninitialized-usage",
        "deleted-usage-field", "text-hostile", "text-extra-field", "start-hostile-id", "start-hostile-name",
        "orphan-empty-args", "post-end-empty-args", "end-hostile-id", "usage-hostile",
        "usage-bool", "usage-duplicate", "payload-after-usage", "usage-with-open-tool",
        "finish-list", "finish-none",
        "finish-extra-field", "finish-duplicate", "hostile-type", "unknown-type",
    ],
)
async def test_executor_rejects_malformed_chunk_shapes_atomically_with_sanitized_error(
    session, test_user, test_agent, test_conversation, chunks
):
    fake_mcp = _FakeMCP(
        tools=[
            ToolHandle(
                server="securo",
                name="list_accounts",
                description="d",
                parameters={"type": "object"},
            )
        ]
    )

    with _patch_provider(_ScriptedProvider([chunks])):
        events = await _drain(
            AgentExecutor(mcp=fake_mcp),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="malformed runtime chunk",
        )

    assert [(event.type, event.error_code, event.finish_reason) for event in events] == [
        ("error", "unavailable", None),
        ("done", None, "error"),
    ]
    assert events[0].error_message == "LLM provider is unavailable."
    assert "sentinel" not in (events[0].error_message or "")
    assert fake_mcp.calls == []
    messages = (
        await session.execute(
            select(Message).where(Message.conversation_id == test_conversation.id)
        )
    ).scalars().all()
    usage_rows = (
        await session.execute(
            select(LlmUsage).where(LlmUsage.conversation_id == test_conversation.id)
        )
    ).scalars().all()
    assert [(message.role, message.tool_calls) for message in messages] == [("user", None)]
    assert usage_rows == []


async def test_executor_sanitizes_unexpected_provider_exception(
    session, test_user, test_agent, test_conversation
):
    class Provider(_ScriptedProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            raise RuntimeError("model-derived-client-secret-sentinel")
            yield  # pragma: no cover

    with _patch_provider(Provider([])):
        events = await _drain(
            AgentExecutor(mcp=_FakeMCP(tools=[])),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="provider exception",
        )

    assert [(event.type, event.error_code, event.error_message, event.finish_reason) for event in events] == [
        ("error", "unavailable", "LLM provider failed unexpectedly.", None),
        ("done", None, None, "error"),
    ]


@pytest.mark.parametrize(
    ("failure_type", "expected_message"),
    [
        (RuntimeError, "LLM provider failed unexpectedly."),
        (LLMUnavailableError, "LLM provider is unavailable."),
        (SystemExit, "LLM provider failed unexpectedly."),
        (KeyboardInterrupt, "LLM provider failed unexpectedly."),
        (GeneratorExit, "LLM provider failed unexpectedly."),
        (BaseException, "LLM provider failed unexpectedly."),
        (asyncio.CancelledError, "LLM provider failed unexpectedly."),
    ],
)
async def test_executor_sanitizes_every_provider_failure_without_real_cancellation(
    session, test_user, test_agent, test_conversation, failure_type, expected_message
):
    class Provider(_ScriptedProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            raise failure_type("provider-failure-sentinel")
            yield  # pragma: no cover

    with _patch_provider(Provider([])):
        events = await asyncio.wait_for(_drain(
            AgentExecutor(mcp=_FakeMCP(tools=[])),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="provider failure",
        ), timeout=2)

    assert [(event.type, event.error_code, event.error_message, event.finish_reason) for event in events] == [
        ("error", "unavailable", expected_message, None),
        ("done", None, None, "error"),
    ]
    assert "sentinel" not in repr(events)




@pytest.mark.parametrize("cancel_count,uncancel_count", [(1, 0), (2, 0), (2, 1)])
async def test_executor_ignores_stale_cancellation_counts_on_later_operation(
    session, test_user, test_agent, test_conversation, cancel_count, uncancel_count
):
    class Provider(_ScriptedProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            raise RuntimeError("independent-provider-sentinel")
            yield  # pragma: no cover

    async def scenario():
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
        with _patch_provider(Provider([])):
            events = await _drain(
                AgentExecutor(mcp=_FakeMCP(tools=[])),
                session=session,
                agent=test_agent,
                user_id=test_user.id,
                conversation_id=test_conversation.id,
                user_message="later independent failure",
            )
        return baseline, task.cancelling(), events

    baseline, final_count, events = await asyncio.create_task(scenario())
    assert final_count == baseline == cancel_count - uncancel_count
    assert [event.type for event in events] == ["error", "done"]
    assert "sentinel" not in repr(events)


async def test_executor_delivers_cancellation_at_trusted_provider_checkpoint(
    session, test_user, test_agent, test_conversation
):
    entered = False
    cleaned = False
    parent_task: asyncio.Task | None = None

    class Source:
        def __aiter__(self):
            return self

        async def __anext__(self):
            nonlocal entered
            entered = True
            return ChatChunk(type="finish", finish_reason="stop")

        async def aclose(self):
            nonlocal cleaned
            cleaned = True

    class Provider(_ScriptedProvider):
        def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            assert parent_task is not None
            parent_task.cancel()
            return Source()

    async def scenario():
        nonlocal parent_task
        parent_task = asyncio.current_task()
        assert parent_task is not None
        with _patch_provider(Provider([])):
            return await _drain(
                AgentExecutor(mcp=_FakeMCP(tools=[])),
                session=session,
                agent=test_agent,
                user_id=test_user.id,
                conversation_id=test_conversation.id,
                user_message="cancel before provider await",
            )

    task = asyncio.create_task(scenario())
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await asyncio.wait_for(task, timeout=2)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert task.cancelling() == 1
    assert entered is False
    assert cleaned is True








async def test_executor_provider_runs_in_caller_task_with_context_continuity(
    session, test_user, test_agent, test_conversation
):
    marker = ContextVar("executor_provider_marker", default="parent")
    task_ids: set[int] = set()
    calls = 0

    class Source:
        def __aiter__(self):
            return self

        async def __anext__(self):
            nonlocal calls
            task = asyncio.current_task()
            assert task is not None
            task_ids.add(id(task))
            calls += 1
            if calls == 1:
                assert marker.get() == "parent"
                marker.set("provider-runner")
                return ChatChunk(type="text_delta", text="one")
            if calls == 2:
                assert marker.get() == "provider-runner"
                return ChatChunk(type="finish", finish_reason="stop")
            if calls == 3:
                assert marker.get() == "provider-runner"
                raise StopAsyncIteration
            raise AssertionError("executor resumed provider after exhaustion")

    class Provider(_ScriptedProvider):
        def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            return Source()

    caller_task = asyncio.current_task()
    assert caller_task is not None
    with _patch_provider(Provider([])):
        events = await _drain(
            AgentExecutor(mcp=_FakeMCP(tools=[])),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="persistent runner",
        )

    assert [event.type for event in events] == ["text_delta", "done"]
    assert task_ids == {id(caller_task)}
    assert calls == 3
    assert marker.get() == "provider-runner"






async def test_executor_rejects_unknown_chunk_before_emit_persist_or_dispatch(
    session, test_user, test_agent, test_conversation
):
    fake_mcp = _FakeMCP(
        tools=[
            ToolHandle(
                server="securo",
                name="list_accounts",
                description="d",
                parameters={"type": "object"},
            )
        ]
    )
    chunks = [
        ChatChunk(type="text_delta", text="buffered-secret"),
        ChatChunk(
            type="tool_call_start",
            tool_call_id="call",
            tool_name="securo__list_accounts",
        ),
        ChatChunk(type="tool_call_args_delta", tool_call_id="call", args_delta="{}"),
        ChatChunk(type="tool_call_end", tool_call_id="call"),
        ChatChunk(type=cast(Any, "bogus")),
        ChatChunk(type="finish", finish_reason="tool_calls"),
    ]

    with _patch_provider(_ScriptedProvider([chunks])):
        events = await _drain(
            AgentExecutor(mcp=fake_mcp),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="unknown chunk",
        )

    assert fake_mcp.calls == []
    assert [(event.type, event.error_code, event.finish_reason) for event in events] == [
        ("error", "unavailable", None),
        ("done", None, "error"),
    ]
    assert all("buffered-secret" not in (event.error_message or "") for event in events)
    messages = (
        await session.execute(
            select(Message).where(Message.conversation_id == test_conversation.id)
        )
    ).scalars().all()
    assert [message.role for message in messages] == ["user"]


async def test_executor_counts_unknown_chunk_before_type_inspection(
    session, test_user, test_agent, test_conversation, monkeypatch
):
    monkeypatch.setattr(executor_module, "MAX_STREAM_CHUNKS", 1)
    chunks = [
        ChatChunk(type="text_delta", text=""),
        ChatChunk(type=cast(Any, "bogus")),
    ]

    with _patch_provider(_ScriptedProvider([chunks])):
        events = await _drain(
            AgentExecutor(mcp=_FakeMCP(tools=[])),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="count unknown",
        )

    assert [event.type for event in events] == ["error", "done"]
    assert events[0].error_message == "LLM provider is unavailable."
    messages = (
        await session.execute(
            select(Message).where(Message.conversation_id == test_conversation.id)
        )
    ).scalars().all()
    assert [message.role for message in messages] == ["user"]


@pytest.mark.parametrize(
    "chunk",
    [
        ChatChunk(type="tool_call_start", tool_call_id="abc", tool_name="de"),
        ChatChunk(type="tool_call_end", tool_call_id="a", thought_signature="abcde"),
    ],
    ids=["start-metadata", "signature"],
)
async def test_process_chunk_bounds_tool_metadata_before_retention(monkeypatch, chunk):
    monkeypatch.setattr(executor_module, "_MAX_LLM_TOOL_METADATA_BYTES", 4)
    monkeypatch.setattr(executor_module, "_MAX_LLM_ALL_TOOL_METADATA_BYTES", 4)
    open_calls = {}
    if chunk.type == "tool_call_end":
        open_calls["a"] = {
            "id": "a",
            "name": "b",
            "args_parts": [],
            "args_bytes": 0,
            "ended": False,
        }
    totals = {"text_bytes": 0, "all_args_bytes": 0, "metadata_bytes": 0}

    with pytest.raises(LLMUnavailableError, match="metadata.*size limit"):
        _ = [
            event
            async for event in executor_module._process_chunk(
                chunk, [], open_calls, totals
            )
        ]


async def test_process_chunk_rejects_nul_argument_before_retention():
    open_calls = {
        "a": {
            "id": "a",
            "name": "lookup",
            "args_parts": [],
            "args_bytes": 0,
            "ended": False,
        }
    }
    totals = {"text_bytes": 0, "all_args_bytes": 0, "metadata_bytes": 0}

    with pytest.raises(LLMUnavailableError, match="invalid tool arguments"):
        _ = [
            event
            async for event in executor_module._process_chunk(
                ChatChunk(
                    type="tool_call_args_delta",
                    tool_call_id="a",
                    args_delta='{"value":"bad\x00argument"}',
                ),
                [],
                open_calls,
                totals,
            )
        ]

    assert open_calls["a"]["args_parts"] == []
    assert totals["all_args_bytes"] == 0


async def test_invalid_tool_result_is_sanitized_before_emit_or_persistence(
    session, test_user, test_agent, test_conversation
):
    fake_mcp = _FakeMCP(
        tools=[
            ToolHandle(
                server="securo",
                name="list_accounts",
                description="d",
                parameters={"type": "object"},
            )
        ],
        canned_result={"ok": True, "data": {"bad\x00key": "bad\x00value"}, "text": "bad\x00text"},
    )
    provider = _ScriptedProvider(
        [
            [
                ChatChunk(
                    type="tool_call_start",
                    tool_call_id="call",
                    tool_name="securo__list_accounts",
                ),
                ChatChunk(type="tool_call_args_delta", tool_call_id="call", args_delta="{}"),
                ChatChunk(type="tool_call_end", tool_call_id="call"),
                ChatChunk(type="finish", finish_reason="tool_calls"),
            ],
            [ChatChunk(type="text_delta", text="done"), ChatChunk(type="finish", finish_reason="stop")],
        ]
    )

    with _patch_provider(provider):
        events = await _drain(
            AgentExecutor(mcp=fake_mcp),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="call it",
        )

    tool_event = next(event for event in events if event.type == "tool_result")
    assert tool_event.tool_result == {"ok": False, "data": None, "text": None}
    messages = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == test_conversation.id)
            .order_by(Message.ordinal)
        )
    ).scalars().all()
    tool_message = next(message for message in messages if message.role == "tool")
    assert "\x00" not in tool_message.content
    assert "\x00" not in repr(tool_message.tool_result)


@pytest.mark.parametrize(
    "chunks",
    [
        [ChatChunk(type="text_delta", text="bad\x00text"), ChatChunk(type="finish")],
        [
            ChatChunk(type="usage", usage=Usage(input_tokens=2_147_483_648, output_tokens=0)),
            ChatChunk(type="finish"),
        ],
    ],
    ids=["nul-text", "usage-overflow"],
)
async def test_unpersistable_stream_values_fail_before_persistence(
    session, test_user, test_agent, test_conversation, chunks
):
    with _patch_provider(_ScriptedProvider([chunks])):
        events = await _drain(
            AgentExecutor(mcp=_FakeMCP(tools=[])),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="bad stream",
        )

    assert [(event.type, event.error_code, event.finish_reason) for event in events[-2:]] == [
        ("error", "unavailable", None),
        ("done", None, "error"),
    ]
    messages = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == test_conversation.id)
            .order_by(Message.ordinal)
        )
    ).scalars().all()
    assert [message.role for message in messages] == ["user"]


async def test_stream_aggregate_limits_accept_exact_multibyte_boundaries(
    session, test_user, test_agent, test_conversation, monkeypatch
):
    monkeypatch.setattr(executor_module, "_MAX_LLM_TEXT_BYTES", 4)
    monkeypatch.setattr(executor_module, "_MAX_LLM_TOOL_ARGUMENT_BYTES", 4)
    monkeypatch.setattr(executor_module, "_MAX_LLM_ALL_TOOL_ARGUMENT_BYTES", 4)
    monkeypatch.setattr(executor_module, "_MAX_LLM_TOOL_CALLS", 2)
    provider = _ScriptedProvider(
        [[
            ChatChunk(type="text_delta", text="é"),
            ChatChunk(type="text_delta", text="é"),
            ChatChunk(type="finish", finish_reason="stop"),
        ]]
    )

    with _patch_provider(provider):
        events = await _drain(
            AgentExecutor(mcp=_FakeMCP(tools=[])),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="boundary",
        )

    assert "".join(event.text or "" for event in events) == "éé"
    messages = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == test_conversation.id)
            .order_by(Message.ordinal)
        )
    ).scalars().all()
    assert [(message.role, message.content) for message in messages] == [
        ("user", "boundary"),
        ("assistant", "éé"),
    ]


async def test_tool_result_passed_to_llm_in_full(session, test_user, test_agent, test_conversation):
    """Regression: the executor used to truncate the tool result text to
    1500 chars before feeding it back to the model, causing the LLM to
    report "truncated" rows for any list response over a few items.
    The LLM must see the full structured payload."""
    big_items = [{"id": f"id-{i}", "description": f"Transaction number {i}", "amount": i * 10} for i in range(20)]
    fake_mcp = _FakeMCP(
        tools=[ToolHandle(server="securo", name="list_transactions", description="d", parameters={"type": "object"})],
        canned_result={
            "ok": True,
            "data": {"items": big_items, "total": 20},
            "text": "long unused text",
        },
    )

    captured_tool_messages: list[str] = []

    class _Capture(_ScriptedProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            captured_tool_messages.extend([m.content for m in messages if m.role == "tool"])
            async for c in super().chat_stream(messages, model=model, tools=tools, temperature=temperature, max_tokens=max_tokens):
                yield c

    provider = _Capture([
        # Turn 1: emit a tool call.
        [
            ChatChunk(type="tool_call_start", tool_call_id="t1", tool_name="securo__list_transactions"),
            ChatChunk(type="tool_call_args_delta", tool_call_id="t1", args_delta="{}"),
            ChatChunk(type="tool_call_end", tool_call_id="t1"),
            ChatChunk(type="finish", finish_reason="tool_calls"),
        ],
        # Turn 2: text reply.
        [
            ChatChunk(type="text_delta", text="ok"),
            ChatChunk(type="finish", finish_reason="stop"),
        ],
    ])

    executor = AgentExecutor(mcp=fake_mcp)
    with _patch_provider(provider):
        await _drain(
            executor,
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="list me everything",
        )

    assert captured_tool_messages, "executor should have fed a tool message into turn 2"
    last = captured_tool_messages[-1]
    # Every item must appear in the LLM-facing content — no silent truncation.
    for i in range(20):
        assert f"Transaction number {i}" in last, (
            f"item {i} missing from tool message of length {len(last)} — likely truncation regression"
        )


async def test_text_only_tool_result_round_trips_into_responses_history(
    session, test_user, test_agent, test_conversation
):
    fake_mcp = _FakeMCP(
        tools=[ToolHandle(server="securo", name="text_tool", description="d", parameters={"type": "object"})],
        canned_result={"ok": True, "data": None, "text": "complete text-only result"},
    )
    provider = _ScriptedProvider(
        [
            [
                ChatChunk(type="tool_call_start", tool_call_id="text-call", tool_name="securo__text_tool"),
                ChatChunk(type="tool_call_args_delta", tool_call_id="text-call", args_delta="{}"),
                ChatChunk(type="tool_call_end", tool_call_id="text-call"),
                ChatChunk(type="finish", finish_reason="tool_calls"),
            ],
            [ChatChunk(type="text_delta", text="done"), ChatChunk(type="finish", finish_reason="stop")],
        ]
    )
    executor = AgentExecutor(mcp=fake_mcp)
    with _patch_provider(provider):
        await _drain(
            executor,
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="run text tool",
        )
        await _drain(
            executor,
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="replay it",
        )

    restored_tools = [message for message in provider.requests[2] if message.role == "tool"]
    assert len(restored_tools) == 1
    assert restored_tools[0].tool_call_id == "text-call"
    assert restored_tools[0].content == '"complete text-only result"'

    from app.agents.providers.openai_chatgpt import _serialize_input

    _, responses_input = _serialize_input(provider.requests[2])
    function_items = [
        item
        for item in responses_input
        if item.get("type") in {"function_call", "function_call_output"}
    ]
    assert function_items == [
        {
            "type": "function_call",
            "namespace": "securo",
            "call_id": "text-call",
            "name": "securo__text_tool",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "text-call",
            "output": '"complete text-only result"',
        },
    ]


async def test_auto_context_primer_prepended_when_enabled(session, test_user, test_agent, test_conversation, test_account):
    """Captures the system messages the provider sees and verifies the
    primer is the first one when auto_context=True (default), with the
    agent's own system_prompt right after."""
    captured: dict[str, list] = {"system": []}

    class _Capture(_ScriptedProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            captured["system"] = [m.content for m in messages if m.role == "system"]
            async for c in super().chat_stream(messages, model=model, tools=tools, temperature=temperature, max_tokens=max_tokens):
                yield c

    provider = _Capture([[
        ChatChunk(type="text_delta", text="ok"),
        ChatChunk(type="finish", finish_reason="stop"),
    ]])
    executor = AgentExecutor(mcp=_FakeMCP(tools=[]))
    test_agent.auto_context = True
    test_agent.system_prompt = "You are helpful."
    await session.commit()

    with _patch_provider(provider):
        await _drain(
            executor,
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="hi",
        )
    sys_msgs = captured["system"]
    # Stack: guardrail [0] + identity primer [1] + agent system_prompt [2] + auto-context [3].
    assert len(sys_msgs) == 4, f"expected guardrail + identity + agent prompt + auto-context, got {len(sys_msgs)}"
    assert "Runtime rules" in sys_msgs[0]
    assert "propose_" in sys_msgs[0]
    assert "Securo" in sys_msgs[1]            # identity primer mentions the product
    assert sys_msgs[2] == "You are helpful."
    assert "Context for this conversation" in sys_msgs[3]
    assert "test@example.com" in sys_msgs[3]  # uses test_user fixture's email
    assert "Conta Corrente" in sys_msgs[3]    # account name from fixture


async def test_auto_context_primer_skipped_when_disabled(session, test_user, test_agent, test_conversation):
    captured: dict[str, list] = {"system": []}

    class _Capture(_ScriptedProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            captured["system"] = [m.content for m in messages if m.role == "system"]
            async for c in super().chat_stream(messages, model=model, tools=tools, temperature=temperature, max_tokens=max_tokens):
                yield c

    provider = _Capture([[
        ChatChunk(type="text_delta", text="ok"),
        ChatChunk(type="finish", finish_reason="stop"),
    ]])
    executor = AgentExecutor(mcp=_FakeMCP(tools=[]))
    test_agent.auto_context = False
    test_agent.system_prompt = "Just the agent prompt."
    await session.commit()

    with _patch_provider(provider):
        await _drain(
            executor,
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="hi",
        )
    sys_msgs = captured["system"]
    # Guardrail [0] + identity [1] + agent prompt [2]; no auto-context primer.
    assert len(sys_msgs) == 3
    assert "Runtime rules" in sys_msgs[0]
    assert "Securo" in sys_msgs[1]
    assert sys_msgs[2] == "Just the agent prompt."


async def test_runtime_guardrail_always_present_even_with_no_agent_prompt(session, test_user, test_agent, test_conversation):
    """Even when the user gave their agent no system_prompt and disabled
    auto_context, the runtime guardrail must still be there."""
    captured: dict[str, list] = {"system": []}

    class _Capture(_ScriptedProvider):
        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            captured["system"] = [m.content for m in messages if m.role == "system"]
            async for c in super().chat_stream(messages, model=model, tools=tools, temperature=temperature, max_tokens=max_tokens):
                yield c

    provider = _Capture([[ChatChunk(type="text_delta", text="ok"), ChatChunk(type="finish", finish_reason="stop")]])
    executor = AgentExecutor(mcp=_FakeMCP(tools=[]))
    test_agent.auto_context = False
    test_agent.system_prompt = ""
    await session.commit()

    with _patch_provider(provider):
        await _drain(
            executor, session=session, agent=test_agent,
            user_id=test_user.id, conversation_id=test_conversation.id,
            user_message="hi",
        )
    sys_msgs = captured["system"]
    # Guardrail [0] + identity primer [1] are the irreducible minimum —
    # both are always present even when the user gave no system_prompt
    # and turned auto-context off.
    assert len(sys_msgs) == 2
    assert "Runtime rules" in sys_msgs[0]
    assert "propose_" in sys_msgs[0]
    assert "Securo" in sys_msgs[1]


async def test_max_iterations_terminates_runaway_agent(session, test_user, test_agent, test_conversation):
    """If the LLM keeps emitting tool calls forever, executor stops at
    MAX_ITERS and emits a max_iterations error."""
    tools = [ToolHandle(server="securo", name="loop_tool", description="", parameters={"type": "object"})]
    fake_mcp = _FakeMCP(tools=tools)

    # Same tool-call turn repeated indefinitely.
    def _looping_turn():
        call_id = f"t{uuid.uuid4().hex[:6]}"
        return [
            ChatChunk(type="tool_call_start", tool_call_id=call_id, tool_name="securo__loop_tool"),
            ChatChunk(type="tool_call_args_delta", tool_call_id=call_id, args_delta="{}"),
            ChatChunk(type="tool_call_end", tool_call_id=call_id),
            ChatChunk(type="finish", finish_reason="tool_calls"),
        ]

    provider = _ScriptedProvider([_looping_turn() for _ in range(20)])
    executor = AgentExecutor(mcp=fake_mcp)
    with _patch_provider(provider):
        events = await _drain(
            executor,
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="loop forever",
        )

    err = next((e for e in events if e.type == "error"), None)
    done = next((e for e in events if e.type == "done"), None)
    assert err is not None and err.error_code == "max_iterations"
    assert done is not None and done.finish_reason == "max_iterations"


async def test_history_window_never_opens_on_an_orphaned_tool_result(
    session, test_user, test_agent, test_conversation,
):
    """A turn with parallel tool calls has an odd number of rows, so the
    history window can start on a tool result whose assistant call was cut
    off. Providers reject that, so the replay must start at a user message."""
    test_agent.max_history_messages = 3  # window of 3 * 2 + 2 = 8 rows
    await session.commit()

    async def add(role, content="", **kw):
        await conversation_service.append_message(
            session, conversation_id=test_conversation.id, role=role, content=content, **kw,
        )

    await add("user", "q0")
    await add("assistant", tool_calls=[
        {"id": "c1", "name": "securo__list_accounts", "arguments": {}},
        {"id": "c2", "name": "securo__list_budgets", "arguments": {}},
    ])
    await add("tool", "{}", tool_result={"tool_call_id": "c1"})
    await add("tool", "{}", tool_result={"tool_call_id": "c2"})
    await add("assistant", "a0")
    for i in (1, 2):
        await add("user", f"q{i}")
        await add("assistant", f"a{i}")

    provider = _ScriptedProvider([[
        ChatChunk(type="text_delta", text="ok"),
        ChatChunk(type="finish", finish_reason="stop"),
    ]])
    with _patch_provider(provider):
        await _drain(
            AgentExecutor(mcp=_FakeMCP(tools=[])),
            session=session,
            agent=test_agent,
            user_id=test_user.id,
            conversation_id=test_conversation.id,
            user_message="new question",
        )

    replayed = [m for m in provider.requests[0] if m.role != "system"]
    assert [(m.role, m.content) for m in replayed] == [
        ("user", "q1"), ("assistant", "a1"),
        ("user", "q2"), ("assistant", "a2"),
        ("user", "new question"),
    ]
