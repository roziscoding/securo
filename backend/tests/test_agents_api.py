"""Agent + conversation CRUD via HTTP. Verifies auth, multi-tenant scoping,
and field handling. Does not exercise the LLM (see test_agents_executor.py).
"""
import asyncio
import uuid
from contextvars import ContextVar
from typing import Any, cast
from unittest.mock import patch

import bcrypt as _bcrypt
import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.providers.base import LLMError
from app.models.user import User


pytestmark = pytest.mark.asyncio


# Patch MCPRegistry.discover so /api/agents/{id}/tools doesn't try to hit a
# real MCP server during HTTP tests. Returns an empty tool list.
@pytest.fixture(autouse=True)
def _mock_mcp_discover():
    async def _empty_discover(self, *, user_id, workspace_id=None, conversation_id=None, agent_id=None):
        return []
    with patch("app.agents.api.agents.MCPRegistry.discover", new=_empty_discover):
        yield


@pytest_asyncio.fixture
async def other_user(session: AsyncSession) -> User:
    """A second user in the same DB to verify cross-tenant isolation."""
    from app.services.workspace_service import create_personal_workspace_for_user

    hashed = _bcrypt.hashpw(b"otherpass123", _bcrypt.gensalt()).decode()
    user = User(
        id=uuid.uuid4(),
        email="other@example.com",
        hashed_password=hashed,
        is_active=True,
        is_superuser=False,
        is_verified=True,
        preferences={"language": "en", "currency_display": "USD"},
    )
    session.add(user)
    await session.flush()
    await create_personal_workspace_for_user(session, user)
    await session.commit()
    await session.refresh(user)
    return user


@pytest_asyncio.fixture
async def other_auth_headers(client: AsyncClient, other_user: User) -> dict:
    resp = await client.post(
        "/api/auth/login",
        data={"username": "other@example.com", "password": "otherpass123"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert resp.status_code == 200
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


# --- Info endpoints --------------------------------------------------------

async def test_app_info_reports_agents_enabled(client: AsyncClient):
    r = await client.get("/api/info")
    assert r.status_code == 200
    assert r.json()["features"]["agents"] is True


async def test_agents_info_lists_providers(client: AsyncClient, auth_headers: dict):
    r = await client.get("/api/agents/info", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert set(body["providers"]) >= {"openai", "anthropic", "ollama", "openai_compatible"}


async def test_agents_info_exposes_mcp_external_ttl(client: AsyncClient, auth_headers: dict):
    """The frontend reads mcp_external_ttl_days to label the token panel."""
    r = await client.get("/api/agents/info", headers=auth_headers)
    body = r.json()
    assert isinstance(body["mcp_external_ttl_days"], int)
    assert body["mcp_external_ttl_days"] >= 1


async def test_agents_info_exposes_external_mcp_url(client: AsyncClient, auth_headers: dict):
    """The frontend uses external_mcp_url (empty by default) so deployments
    behind an ingress can override the hardcoded :8765 endpoint."""
    r = await client.get("/api/agents/info", headers=auth_headers)
    body = r.json()
    assert "external_mcp_url" in body
    assert isinstance(body["external_mcp_url"], str)


async def test_agents_info_returns_configured_external_mcp_url(
    client: AsyncClient, auth_headers: dict, monkeypatch
):
    """When AGENTS_EXTERNAL_MCP_URL is set, it is surfaced verbatim (trimmed)."""
    from app.agents import config

    config.get_agent_settings.cache_clear()
    monkeypatch.setenv("AGENTS_EXTERNAL_MCP_URL", "  https://securo.example.com/mcp  ")
    try:
        r = await client.get("/api/agents/info", headers=auth_headers)
        assert r.json()["external_mcp_url"] == "https://securo.example.com/mcp"
    finally:
        monkeypatch.delenv("AGENTS_EXTERNAL_MCP_URL", raising=False)
        config.get_agent_settings.cache_clear()


# --- External MCP tokens ---------------------------------------------------

async def test_mcp_tokens_requires_auth(client: AsyncClient):
    """The mint endpoint must reject unauthenticated calls — otherwise any
    anonymous visitor could mint a long-lived token for an arbitrary user."""
    r = await client.post("/api/agents/mcp-tokens")
    assert r.status_code == 401


async def test_mcp_tokens_mint_returns_external_jwt(
    client: AsyncClient, auth_headers: dict, test_user
):
    """End-to-end: minted token decodes, carries the `ext` claim, scoped
    to the calling user, and the advertised TTL matches the response."""
    from app.agents.mcp.auth import JWT_ALGO, JWT_AUDIENCE, JWT_ISSUER
    from app.agents.config import get_agent_settings
    from jose import jwt

    r = await client.post("/api/agents/mcp-tokens", headers=auth_headers)
    assert r.status_code == 201, r.text
    body = r.json()
    assert "token" in body
    assert body["expires_in_days"] == get_agent_settings().mcp_external_ttl_days
    assert body["expires_in_seconds"] == body["expires_in_days"] * 86400

    payload = jwt.decode(
        body["token"],
        get_agent_settings().mcp_jwt_secret,
        algorithms=[JWT_ALGO],
        audience=JWT_AUDIENCE,
        issuer=JWT_ISSUER,
    )
    assert payload["sub"] == str(test_user.id)
    assert payload["ext"] is True
    # External tokens are detached from any conv/agent.
    assert "conv_id" not in payload
    assert "agent_id" not in payload


# --- Agent CRUD ------------------------------------------------------------

async def test_unauthenticated_agents_list_rejected(client: AsyncClient):
    r = await client.get("/api/agents")
    assert r.status_code == 401


async def test_create_list_get_update_delete(client: AsyncClient, auth_headers: dict):
    # Empty by default
    r = await client.get("/api/agents", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == []

    # Create
    payload = {
        "name": "Brazilian Tax Advisor",
        "description": "Helps with PT-BR tax questions",
        "system_prompt": "You are a tax advisor.",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.3,
    }
    r = await client.post("/api/agents", json=payload, headers=auth_headers)
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["name"] == "Brazilian Tax Advisor"
    assert created["provider"] == "openai"
    assert created["temperature"] == 0.3

    aid = created["id"]

    # List shows the new agent
    r = await client.get("/api/agents", headers=auth_headers)
    assert len(r.json()) == 1
    assert r.json()[0]["id"] == aid

    # Get by id
    r = await client.get(f"/api/agents/{aid}", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["id"] == aid

    # Update
    r = await client.patch(f"/api/agents/{aid}", json={"description": "New desc", "temperature": 0.7}, headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["description"] == "New desc"
    assert r.json()["temperature"] == 0.7

    # Delete
    r = await client.delete(f"/api/agents/{aid}", headers=auth_headers)
    assert r.status_code == 204

    # Gone
    r = await client.get(f"/api/agents/{aid}", headers=auth_headers)
    assert r.status_code == 404


async def test_multi_tenant_scoping(client: AsyncClient, auth_headers: dict, other_auth_headers: dict):
    """User A's agent is invisible to user B."""
    r = await client.post("/api/agents", json={"name": "Mine"}, headers=auth_headers)
    assert r.status_code == 201
    aid = r.json()["id"]

    # Other user lists their own agents — empty.
    r = await client.get("/api/agents", headers=other_auth_headers)
    assert r.status_code == 200
    assert r.json() == []

    # Other user can't read it directly.
    r = await client.get(f"/api/agents/{aid}", headers=other_auth_headers)
    assert r.status_code == 404

    # Or update.
    r = await client.patch(f"/api/agents/{aid}", json={"name": "hijack"}, headers=other_auth_headers)
    assert r.status_code == 404

    # Or delete.
    r = await client.delete(f"/api/agents/{aid}", headers=other_auth_headers)
    assert r.status_code == 404


async def test_create_validation_rejects_missing_name(client: AsyncClient, auth_headers: dict):
    r = await client.post("/api/agents", json={}, headers=auth_headers)
    assert r.status_code == 422


async def test_temperature_clamp(client: AsyncClient, auth_headers: dict):
    # Pydantic schema enforces 0..2
    r = await client.post("/api/agents", json={"name": "x", "temperature": 5.0}, headers=auth_headers)
    assert r.status_code == 422


# --- Conversations & messages ---------------------------------------------

async def test_conversations_empty(client: AsyncClient, auth_headers: dict):
    r = await client.get("/api/agents/conversations", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == []


async def test_conversation_filtered_by_agent(
    client: AsyncClient, auth_headers: dict, session: AsyncSession, test_user: User
):
    """Insert a conversation directly and verify the filter works."""
    from app.agents.models.agent import Agent
    from app.agents.models.conversation import Conversation

    a1 = Agent(id=uuid.uuid4(), user_id=test_user.id, name="A1")
    a2 = Agent(id=uuid.uuid4(), user_id=test_user.id, name="A2")
    session.add_all([a1, a2])
    await session.commit()

    c1 = Conversation(id=uuid.uuid4(), user_id=test_user.id, agent_id=a1.id, channel="web", title="hello")
    c2 = Conversation(id=uuid.uuid4(), user_id=test_user.id, agent_id=a2.id, channel="web", title="other")
    session.add_all([c1, c2])
    await session.commit()

    r = await client.get(f"/api/agents/conversations?agent_id={a1.id}", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["title"] == "hello"


async def test_rename_conversation(
    client: AsyncClient, auth_headers: dict, session: AsyncSession, test_user: User
):
    from app.agents.models.agent import Agent
    from app.agents.models.conversation import Conversation

    agent = Agent(id=uuid.uuid4(), user_id=test_user.id, name="A")
    conv = Conversation(id=uuid.uuid4(), user_id=test_user.id, agent_id=agent.id, channel="web", title="old")
    session.add_all([agent, conv])
    await session.commit()

    r = await client.patch(f"/api/agents/conversations/{conv.id}", json={"title": "renamed!"}, headers=auth_headers)
    assert r.status_code == 200, r.text
    assert r.json()["title"] == "renamed!"

    r = await client.get(f"/api/agents/conversations/{conv.id}", headers=auth_headers)
    assert r.json()["title"] == "renamed!"


async def test_rename_conversation_validates_empty(
    client: AsyncClient, auth_headers: dict, session: AsyncSession, test_user: User
):
    from app.agents.models.agent import Agent
    from app.agents.models.conversation import Conversation

    agent = Agent(id=uuid.uuid4(), user_id=test_user.id, name="A")
    conv = Conversation(id=uuid.uuid4(), user_id=test_user.id, agent_id=agent.id, channel="web", title="old")
    session.add_all([agent, conv])
    await session.commit()

    r = await client.patch(f"/api/agents/conversations/{conv.id}", json={"title": ""}, headers=auth_headers)
    assert r.status_code == 422


async def test_rename_conversation_404_other_user(
    client: AsyncClient, auth_headers: dict, other_auth_headers: dict, session: AsyncSession, test_user: User
):
    """auth_headers must come BEFORE other_auth_headers so test_user's
    clean_db setup runs first — otherwise it wipes the other user that
    other_auth_headers already created."""
    from app.agents.models.agent import Agent
    from app.agents.models.conversation import Conversation

    agent = Agent(id=uuid.uuid4(), user_id=test_user.id, name="A")
    conv = Conversation(id=uuid.uuid4(), user_id=test_user.id, agent_id=agent.id, channel="web", title="old")
    session.add_all([agent, conv])
    await session.commit()

    r = await client.patch(
        f"/api/agents/conversations/{conv.id}", json={"title": "hijack"}, headers=other_auth_headers
    )
    assert r.status_code == 404


async def test_delete_conversation(
    client: AsyncClient, auth_headers: dict, session: AsyncSession, test_user: User
):
    from app.agents.models.agent import Agent
    from app.agents.models.conversation import Conversation

    agent = Agent(id=uuid.uuid4(), user_id=test_user.id, name="A")
    conv = Conversation(id=uuid.uuid4(), user_id=test_user.id, agent_id=agent.id, channel="web")
    session.add_all([agent, conv])
    await session.commit()

    r = await client.delete(f"/api/agents/conversations/{conv.id}", headers=auth_headers)
    assert r.status_code == 204

    r = await client.get(f"/api/agents/conversations/{conv.id}", headers=auth_headers)
    assert r.status_code == 404


async def test_generate_title_uses_llm(
    client: AsyncClient, auth_headers: dict, session: AsyncSession, test_user: User
):
    """The endpoint asks the agent's LLM provider for a short summary
    and persists it. We patch _provider_and_model_for so no real LLM
    call is made."""
    from unittest.mock import patch
    from app.agents.models.agent import Agent
    from app.agents.models.conversation import Conversation, Message
    from app.agents.providers.base import (
        ChatResponse,
        LLMProvider,
        Usage,
    )

    class _Scripted(LLMProvider):
        name = "openai"

        async def chat_stream(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            # not used for title (we call .chat() which goes through .chat_stream)
            yield  # pragma: no cover
            return  # pragma: no cover

        async def chat(self, messages, *, model, tools=None, temperature=0.4, max_tokens=None):
            return ChatResponse(content="Brazilian Cuisine Picks", usage=Usage(2, 4))

        async def embed(self, texts, *, model):
            return []

    agent = Agent(id=uuid.uuid4(), user_id=test_user.id, name="A", provider="openai", model="gpt-4o-mini")
    conv = Conversation(id=uuid.uuid4(), user_id=test_user.id, agent_id=agent.id, channel="web", title="name three brazilian foods")
    session.add_all([agent, conv])
    await session.commit()

    session.add_all([
        Message(id=uuid.uuid4(), conversation_id=conv.id, ordinal=1, role="user", content="name three brazilian foods"),
        Message(id=uuid.uuid4(), conversation_id=conv.id, ordinal=2, role="assistant", content="Feijoada, Coxinha, Pão de Queijo"),
    ])
    await session.commit()

    with patch(
        "app.agents.runtime.executor._provider_and_model_for",
        return_value=(_Scripted(), "gpt-4o-mini"),
    ):
        r = await client.post(
            f"/api/agents/conversations/{conv.id}/generate-title",
            headers=auth_headers,
        )
    assert r.status_code == 200, r.text
    assert r.json()["title"] == "Brazilian Cuisine Picks"


async def test_generate_title_strips_reasoning_tags(
    client: AsyncClient, auth_headers: dict, session: AsyncSession, test_user: User
):
    """Local reasoning models often emit `<think>...</think>` before the
    answer. The endpoint must strip that and use only the title."""
    from unittest.mock import patch
    from app.agents.models.agent import Agent
    from app.agents.models.conversation import Conversation, Message
    from app.agents.providers.base import ChatResponse, LLMProvider, Usage

    class _Scripted(LLMProvider):
        name = "openai_compatible"

        async def chat_stream(self, *a, **kw):
            yield  # pragma: no cover

        async def chat(self, *a, **kw):
            return ChatResponse(
                content="<think>Let me summarize…</think>\nQuick Math Question",
                usage=Usage(0, 0),
            )

        async def embed(self, *a, **kw):
            return []

    agent = Agent(id=uuid.uuid4(), user_id=test_user.id, name="A", model="local")
    conv = Conversation(id=uuid.uuid4(), user_id=test_user.id, agent_id=agent.id, channel="web", title="2+2")
    session.add_all([agent, conv])
    await session.commit()
    session.add_all([
        Message(id=uuid.uuid4(), conversation_id=conv.id, ordinal=1, role="user", content="2+2"),
        Message(id=uuid.uuid4(), conversation_id=conv.id, ordinal=2, role="assistant", content="4"),
    ])
    await session.commit()

    with patch(
        "app.agents.runtime.executor._provider_and_model_for",
        return_value=(_Scripted(), "local"),
    ):
        r = await client.post(
            f"/api/agents/conversations/{conv.id}/generate-title",
            headers=auth_headers,
        )
    assert r.status_code == 200
    # Just the title — the <think> block is gone.
    assert r.json()["title"] == "Quick Math Question"


# --- Tools endpoint --------------------------------------------------------

async def test_tools_endpoint_returns_empty_when_no_mcp(client: AsyncClient, auth_headers: dict):
    r = await client.post("/api/agents", json={"name": "T"}, headers=auth_headers)
    aid = r.json()["id"]

    r = await client.get(f"/api/agents/{aid}/tools", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["tools"] == []


async def test_put_tools_persists_selection(
    client: AsyncClient, auth_headers: dict, session: AsyncSession, test_user: User
):
    """Even without a live MCP server, the user can pre-stage a whitelist."""
    from app.agents.models.agent import Agent

    a = Agent(id=uuid.uuid4(), user_id=test_user.id, name="X")
    session.add(a)
    await session.commit()

    r = await client.put(
        f"/api/agents/{a.id}/tools",
        json=[
            {"server": "securo", "tool_name": "list_transactions", "enabled": True},
            {"server": "securo", "tool_name": "propose_categorize", "enabled": False},
        ],
        headers=auth_headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["updated"] == 2

    # Verify rows actually exist via the service.
    from app.agents.services import agent_service

    pairs = await agent_service.allowed_tool_pairs(session, a.id)
    assert pairs == {("securo", "list_transactions")}


@pytest.mark.parametrize("failure_kind", ["exact", "hostile"])
async def test_chat_sse_fallback_never_exposes_replacement_exception(
    client: AsyncClient,
    auth_headers: dict,
    session: AsyncSession,
    test_user: User,
    failure_kind: str,
):
    from app.agents.models.agent import Agent
    from app.agents.providers.base import LLMUnavailableError
    from app.agents.runtime.executor import AgentExecutor

    sentinel = "api-sse-replacement-bearer-sentinel"

    class HostileFailure(LLMUnavailableError):
        def __getattribute__(self, _name):
            raise RuntimeError(f"accessor-{sentinel}")

        def __str__(self):
            raise RuntimeError(f"replacement-{sentinel}")

    async def fail(self, **_kwargs):
        if failure_kind == "exact":
            raise LLMUnavailableError(sentinel)
        raise HostileFailure(sentinel)
        yield  # pragma: no cover

    agent = Agent(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="SSE fallback",
        provider="openai",
        model="gpt-test",
    )
    session.add(agent)
    await session.commit()

    with patch.object(AgentExecutor, "run", fail):
        response = await client.post(
            f"/api/agents/{agent.id}/chat",
            json={"content": "trigger fallback"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert '"error_code": "unknown"' in response.text
    assert '"error_message": "Agent request failed unexpectedly."' in response.text
    assert sentinel not in response.text


async def test_chat_sse_fallback_bounds_hostile_baseexception(
    client: AsyncClient, auth_headers: dict, session: AsyncSession, test_user: User
):
    from app.agents.models.agent import Agent
    from app.agents.runtime.executor import AgentExecutor

    sentinel = "api-sse-hostile-baseexception-sentinel"

    class HostileFailure(BaseException):
        def __getattribute__(self, _name):
            raise RuntimeError(f"accessor-{sentinel}")

        def __str__(self):
            raise RuntimeError(f"str-{sentinel}")

        def __repr__(self):
            raise RuntimeError(f"repr-{sentinel}")

        def __hash__(self):
            raise RuntimeError(f"hash-{sentinel}")

        def __eq__(self, _other):
            raise RuntimeError(f"equality-{sentinel}")

    async def fail(self, **_kwargs):
        raise HostileFailure(sentinel)
        yield  # pragma: no cover

    agent = Agent(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="SSE hostile BaseException",
        provider="openai",
        model="gpt-test",
    )
    session.add(agent)
    await session.commit()

    with patch.object(AgentExecutor, "run", fail):
        response = await asyncio.wait_for(
            client.post(
                f"/api/agents/{agent.id}/chat",
                json={"content": "trigger hostile BaseException"},
                headers=auth_headers,
            ),
            timeout=2,
        )

    assert response.status_code == 200
    assert response.text.count("event: error") == 1
    assert '"error_code": "unknown"' in response.text
    assert '"error_message": "Agent request failed unexpectedly."' in response.text
    assert sentinel not in response.text


async def test_chat_sse_event_serialization_never_stringifies_hostile_values(
    client: AsyncClient, auth_headers: dict, session: AsyncSession, test_user: User
):
    from app.agents.models.agent import Agent
    from app.agents.runtime.executor import AgentExecutor, ExecutorEvent

    sentinel = "api-sse-event-bearer-sentinel"

    class HostileValue:
        def __str__(self):
            return sentinel

    async def hostile_event(self, **_kwargs):
        yield ExecutorEvent(
            type="error",
            error_message=cast(Any, HostileValue()),
        )

    agent = Agent(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="SSE serializer",
        provider="openai",
        model="gpt-test",
    )
    session.add(agent)
    await session.commit()

    with patch.object(AgentExecutor, "run", hostile_event):
        response = await client.post(
            f"/api/agents/{agent.id}/chat",
            json={"content": "trigger serializer"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert '"error_message": "Agent request failed unexpectedly."' in response.text
    assert sentinel not in response.text


@pytest.mark.parametrize(
    "mismatch_kind",
    [
        "message-phase",
        "reasoning-encrypted",
        "content-logprobs",
        "done-logprobs-omission",
        "function-name",
    ],
)
async def test_chat_output_item_mismatch_has_no_tool_or_runtime_side_effects(
    mismatch_kind: str,
    client: AsyncClient,
    auth_headers: dict,
    session: AsyncSession,
    test_user: User,
):
    import json
    from contextlib import asynccontextmanager

    from sqlalchemy import select

    from app.agents.mcp.client import MCPRegistry, ToolHandle
    from app.agents.models.agent import Agent
    from app.agents.models.conversation import Message
    from app.agents.models.usage import LlmUsage
    from app.agents.providers.openai_chatgpt import OpenAIChatGPTProvider

    if mismatch_kind == "function-name":
        added_item = {
            "type": "function_call",
            "namespace": "securo",
            "id": "item",
            "call_id": "call",
            "name": "securo__list_accounts",
            "arguments": "",
            "status": "in_progress",
        }
        done_item = {**added_item, "name": "mutated", "arguments": "{}", "status": "completed"}
        middle_events = [
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "item",
                "output_index": 0,
                "delta": "{}",
                "sequence_number": 2,
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "item",
                "output_index": 0,
                "arguments": "{}",
                "sequence_number": 3,
            },
        ]
    elif mismatch_kind == "message-phase":
        added_item = {
            "id": "item",
            "type": "message",
            "role": "assistant",
            "content": [],
            "status": "in_progress",
            "phase": "commentary",
        }
        done_item = {
            **added_item,
            "status": "completed",
            "phase": "final_answer",
        }
        middle_events = []
    elif mismatch_kind == "reasoning-encrypted":
        added_item = {
            "id": "item",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "prefix",
            "status": "in_progress",
        }
        done_item = {
            **added_item,
            "encrypted_content": "different",
            "status": "completed",
        }
        middle_events = []
    else:
        added_item = {
            "id": "item",
            "type": "message",
            "role": "assistant",
            "content": [],
            "status": "in_progress",
        }
        accepted_logprobs = [
            {
                "token": "{}",
                "bytes": [123, 125],
                "logprob": -0.1,
                "top_logprobs": [],
            }
        ]
        final_part = {
            "type": "output_text",
            "text": "{}",
            "annotations": [],
            **(
                {}
                if mismatch_kind == "done-logprobs-omission"
                else {"logprobs": []}
            ),
        }
        done_item = {
            **added_item,
            "content": [final_part],
            "status": "completed",
        }
        middle_events = [
            {
                "type": "response.content_part.added",
                "item_id": "item",
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": "",
                    "annotations": [],
                    "logprobs": (
                        []
                        if mismatch_kind == "done-logprobs-omission"
                        else [
                            {
                                "token": "INITIAL",
                                "bytes": [73],
                                "logprob": -0.1,
                                "top_logprobs": [],
                            }
                        ]
                    ),
                },
                "sequence_number": 2,
            },
            {
                "type": "response.output_text.delta",
                "item_id": "item",
                "output_index": 0,
                "content_index": 0,
                "delta": "{}",
                "logprobs": (
                    accepted_logprobs
                    if mismatch_kind == "done-logprobs-omission"
                    else []
                ),
                "sequence_number": 3,
            },
            {
                "type": "response.output_text.done",
                "item_id": "item",
                "output_index": 0,
                "content_index": 0,
                "text": "{}",
                **(
                    {}
                    if mismatch_kind == "done-logprobs-omission"
                    else {"logprobs": []}
                ),
                "sequence_number": 4,
            },
            {
                "type": "response.content_part.done",
                "item_id": "item",
                "output_index": 0,
                "content_index": 0,
                "part": final_part,
                "sequence_number": 5,
            },
        ]

    events = [
        {"type": "response.created", "response": {"id": "resp"}, "sequence_number": 0},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": added_item,
            "sequence_number": 1,
        },
        *middle_events,
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": done_item,
            "sequence_number": 2 + len(middle_events),
        },
        {
            "type": "response.completed",
            "response": {
                "id": "resp",
                "status": "completed",
                "output": [done_item],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
            "sequence_number": 3 + len(middle_events),
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
        yield cast(Any, None)

    provider = OpenAIChatGPTProvider(access_token=access)

    async def provider_stream(*_args, **_kwargs):
        async for chunk in provider._parse_stream(Response()):
            yield chunk

    provider.chat_stream = cast(Any, provider_stream)
    tool = ToolHandle(
        server="securo",
        name="list_accounts",
        description="d",
        parameters={"type": "object"},
    )
    calls: list[tuple[str, dict]] = []

    async def discover(self, **_kwargs):
        return [tool]

    async def call(self, *, wire_name, arguments, **_kwargs):
        calls.append((wire_name, arguments))
        return {"ok": True, "data": {}, "text": "must not run"}

    agent = Agent(
        id=uuid.uuid4(),
        user_id=test_user.id,
        name="Terminal reconciliation",
        provider="openai",
        model="gpt-test",
    )
    session.add(agent)
    await session.commit()

    with (
        patch("app.agents.runtime.executor._provider_for", return_value=provider),
        patch.object(MCPRegistry, "discover", new=discover),
        patch.object(MCPRegistry, "call", new=call),
    ):
        response = await client.post(
            f"/api/agents/{agent.id}/chat",
            json={"content": "run a tool"},
            headers=auth_headers,
        )

    assert response.status_code == 200
    assert "event: tool_call" not in response.text
    assert "event: tool_result" not in response.text
    assert "event: text_delta" not in response.text
    assert '"error_code": "unavailable"' in response.text
    assert calls == []

    conversation_id = uuid.UUID(
        json.loads(response.text.split("event: conversation\ndata: ", 1)[1].split("\n\n", 1)[0])[
            "conversation_id"
        ]
    )
    messages = (
        await session.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.ordinal)
        )
    ).scalars().all()
    assert [(message.role, message.content) for message in messages] == [
        ("user", "run a tool")
    ]
    usage_rows = (
        await session.execute(
            select(LlmUsage).where(LlmUsage.conversation_id == conversation_id)
        )
    ).scalars().all()
    assert usage_rows == []


async def _collect_chunks(stream):
    return [chunk async for chunk in stream]


@pytest.mark.parametrize(
    "base",
    [
        RuntimeError,
        LLMError,
        BaseException,
        KeyboardInterrupt,
        SystemExit,
        GeneratorExit,
        asyncio.CancelledError,
    ],
)
async def test_sse_boundary_sanitizes_provider_raised_control_subclasses(base):
    from app.agents.api.chat import _stream_executor_events

    sentinel = "api-sse-control-subclass-sentinel"
    cleaned = False

    class HostileControl(base):
        def __getattribute__(self, _name):
            raise RuntimeError(f"accessor-{sentinel}")

        def __str__(self):
            raise RuntimeError(f"str-{sentinel}")

        def __repr__(self):
            raise RuntimeError(f"repr-{sentinel}")

        def __hash__(self):
            raise RuntimeError(f"hash-{sentinel}")

        def __eq__(self, _other):
            raise RuntimeError(f"equality-{sentinel}")

    async def hostile_stream():
        nonlocal cleaned
        try:
            raise HostileControl(sentinel)
            yield  # pragma: no cover
        finally:
            cleaned = True

    chunks = await asyncio.wait_for(
        _collect_chunks(_stream_executor_events(hostile_stream())),
        timeout=2,
    )

    assert chunks == [
        b'event: error\ndata: {"error_code": "unknown", '
        b'"error_message": "Agent request failed unexpectedly."}\n\n'
    ]
    assert sentinel.encode() not in b"".join(chunks)
    assert cleaned is True




async def test_sse_boundary_reraises_active_task_cancellation_and_cleans_up():
    from app.agents.api.chat import _stream_executor_events

    entered = asyncio.Event()
    cleaned = asyncio.Event()

    async def blocked_stream():
        try:
            entered.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover
        finally:
            cleaned.set()

    task = asyncio.create_task(_collect_chunks(_stream_executor_events(blocked_stream())))
    await asyncio.wait_for(entered.wait(), timeout=2)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    await asyncio.wait_for(cleaned.wait(), timeout=2)






async def test_sse_boundary_uses_caller_task_without_prefetch():
    from app.agents.api.chat import _stream_executor_events
    from app.agents.runtime.executor import ExecutorEvent

    marker = ContextVar("sse_same_task_marker", default="parent")
    task_ids: set[int] = set()
    calls = 0
    in_flight = 0

    class Source:
        def __aiter__(self):
            return self

        async def __anext__(self):
            nonlocal calls, in_flight
            in_flight += 1
            assert in_flight == 1
            try:
                task = asyncio.current_task()
                assert task is not None
                task_ids.add(id(task))
                calls += 1
                if calls == 1:
                    assert marker.get() == "parent"
                    marker.set("runner")
                    return ExecutorEvent(type="text_delta", text="one")
                if calls == 2:
                    assert marker.get() == "runner"
                    return ExecutorEvent(type="done", finish_reason="stop")
                raise AssertionError("the SSE boundary prefetched past done")
            finally:
                in_flight -= 1

    chunks = await _collect_chunks(_stream_executor_events(Source()))

    assert [chunk.split(b"\n", 1)[0] for chunk in chunks] == [
        b"event: text_delta",
        b"event: done",
    ]
    assert calls == 2
    assert len(task_ids) == 1
    assert marker.get() == "runner"




@pytest.mark.parametrize("cancel_count,uncancel_count", [(1, 0), (2, 0), (2, 1)])
async def test_sse_boundary_ignores_stale_cancellation_counts_across_operations(
    cancel_count, uncancel_count
):
    from app.agents.api.chat import _stream_executor_events

    async def failed_stream(label):
        raise RuntimeError(f"independent-{label}-sentinel")
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
        first = await _collect_chunks(_stream_executor_events(failed_stream("first")))
        second = await _collect_chunks(_stream_executor_events(failed_stream("second")))
        return baseline, task.cancelling(), first, second

    baseline, final_count, first, second = await asyncio.create_task(scenario())
    assert final_count == baseline == cancel_count - uncancel_count
    assert first == second == [
        b'event: error\ndata: {"error_code": "unknown", '
        b'"error_message": "Agent request failed unexpectedly."}\n\n'
    ]
    assert b"sentinel" not in b"".join(first + second)


async def test_sse_boundary_delivers_cancellation_at_trusted_pre_iterator_checkpoint():
    from app.agents.api.chat import _stream_executor_events

    entered = False

    async def source():
        nonlocal entered
        entered = True
        yield cast(Any, None)

    async def scenario():
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        return await anext(_stream_executor_events(source()))

    task = asyncio.create_task(scenario())
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await asyncio.wait_for(task, timeout=2)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert entered is False


async def test_sse_boundary_generator_close_preserves_closure_and_cleans_up():
    from app.agents.api.chat import _stream_executor_events
    from app.agents.runtime.executor import ExecutorEvent

    cleaned = False

    async def source():
        nonlocal cleaned
        try:
            yield ExecutorEvent(type="text_delta", text="one")
            await asyncio.Event().wait()
        finally:
            cleaned = True

    stream = _stream_executor_events(source())
    first = await asyncio.wait_for(anext(stream), timeout=2)
    assert b"event: text_delta" in first

    await asyncio.wait_for(stream.aclose(), timeout=2)
    assert cleaned is True


async def test_sse_boundary_cancellation_during_generator_close_is_authoritative():
    from app.agents.api.chat import _stream_executor_events
    from app.agents.runtime.executor import ExecutorEvent

    close_entered = asyncio.Event()

    class Source:
        def __init__(self):
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                await asyncio.Event().wait()
            self.sent = True
            return ExecutorEvent(type="text_delta", text="one")

        async def aclose(self):
            close_entered.set()
            await asyncio.Event().wait()

    stream = _stream_executor_events(Source())
    await asyncio.wait_for(anext(stream), timeout=1)
    closing = asyncio.create_task(stream.aclose())
    await asyncio.wait_for(close_entered.wait(), timeout=1)
    closing.cancel()

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await asyncio.wait_for(closing, timeout=0.5)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert closing.cancelled()


async def test_sse_boundary_stops_after_done_without_second_terminal_event():
    from app.agents.api.chat import _stream_executor_events
    from app.agents.runtime.executor import ExecutorEvent

    class LateFailure(BaseException):
        pass

    cleaned = False

    async def source():
        nonlocal cleaned
        try:
            yield ExecutorEvent(type="done", finish_reason="stop")
            raise LateFailure
        finally:
            cleaned = True

    chunks = await asyncio.wait_for(
        _collect_chunks(_stream_executor_events(source())),
        timeout=2,
    )

    assert len(chunks) == 1
    assert chunks[0].startswith(b"event: done\n")
    assert b"event: error" not in chunks[0]
    assert cleaned is True
