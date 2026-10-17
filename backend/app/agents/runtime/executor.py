"""Channel-agnostic agent execution loop.

Takes (agent, user_id, conversation_id, user_message) and yields a stream
of structured events. Whether the caller is the web SSE endpoint or a
future WhatsApp gateway, this loop is the same.

Flow:
  1. Load conversation history from DB.
  2. Discover tools from MCP servers; filter by per-agent whitelist.
  3. Loop:
     a. Stream LLM response.
     b. If LLM emits tool calls, run them in parallel against MCP and
        feed results back as `role=tool` messages, then continue.
     c. Stop when finish_reason is `stop` (no tool call this turn).
  4. Persist user + assistant messages and any tool messages.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Literal, Optional, cast

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.config import get_agent_settings
from app.agents.mcp.client import MCPRegistry
from app.agents.models.agent import Agent
from app.agents.providers.base import (
    ChatChunk,
    ChatMessage,
    LLM_CLEANUP_TIMEOUT_SECONDS,
    LLM_OPERATION_TIMEOUT_SECONDS,
    LLMAuthError,
    LLMError,
    LLMNotSupportedError,
    LLMRateLimitError,
    LLMUnavailableError,
    MAX_STREAM_CHUNKS,
    OperationCancellationScope,
    Role,
    ToolCall,
    bounded_async_iterator,
    discard_untrusted_exception as _discard_untrusted_exception,
    validate_chat_chunk,
)
from app.agents.providers.registry import build_provider
from app.agents.services import agent_service, context_service, conversation_service, usage_service

logger = logging.getLogger(__name__)
_MAX_LLM_FIELD_BYTES = 1_000_000
_MAX_LLM_TEXT_BYTES = 4_000_000
_MAX_LLM_TOOL_ARGUMENT_BYTES = 1_000_000
_MAX_LLM_ALL_TOOL_ARGUMENT_BYTES = 4_000_000
_MAX_LLM_TOOL_CALLS = 128
_MAX_LLM_TOOL_METADATA_BYTES = 1_000_000
_MAX_LLM_ALL_TOOL_METADATA_BYTES = 4_000_000
_MAX_LLM_USAGE_TOKENS = 2_147_483_647
_MAX_LLM_JSON_NODES = 20_000
_MAX_LLM_JSON_DEPTH = 32


def _unique_json_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


@dataclass
class ExecutorEvent:
    type: Literal[
        "text_delta",
        "tool_call",      # tool name + args (after assembly)
        "tool_result",    # tool name + ok + summary
        "citation",
        "error",
        "done",
    ]
    text: Optional[str] = None
    tool_name: Optional[str] = None
    tool_args: Optional[dict[str, Any]] = None
    tool_result: Optional[dict[str, Any]] = None
    citation: Optional[dict[str, Any]] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    finish_reason: Optional[str] = None


def _provider_for(agent: Agent):
    """Build a provider from env-var defaults (no connection lookup).

    Kept as a separate function so tests can monkey-patch this single
    seam to inject a scripted provider. Production goes through
    `_provider_and_model_for` which prefers user-managed connections.
    """
    name = agent.provider or os.getenv("AGENTS_DEFAULT_PROVIDER", "ollama")
    api_key = ""
    base_url = None
    if name == "openai":
        api_key = os.getenv("AGENTS_OPENAI_API_KEY", "")
    elif name == "anthropic":
        api_key = os.getenv("AGENTS_ANTHROPIC_API_KEY", "")
    elif name == "ollama":
        base_url = os.getenv("AGENTS_OLLAMA_BASE_URL", "http://ollama:11434")
    elif name == "openai_compatible":
        api_key = os.getenv("AGENTS_OPENAI_COMPAT_API_KEY", "")
        base_url = os.getenv("AGENTS_OPENAI_COMPAT_BASE_URL")
    return build_provider(name, api_key=api_key, base_url=base_url, model=agent.model)


def _model_for(agent: Agent) -> str:
    if agent.model:
        return agent.model
    return os.getenv("AGENTS_DEFAULT_MODEL", "")


async def _provider_and_model_for(session, agent: Agent):
    """Resolve (provider, model) using this priority:
        1. agent.connection_id        — explicit user-managed connection
        2. user's default connection  — is_default=True
        3. _provider_for(agent)       — env-var fallback (testable seam)

    Returns (LLMProvider, model_id_str).
    """
    from app.agents.services import connection_service  # local import: cycle safety

    conn = None
    if agent.connection_id:
        conn = await connection_service.get_connection(session, agent.connection_id, agent.user_id)
        if conn is None:
            raise LLMAuthError("The configured connection is unavailable")
    else:
        conn = await connection_service.get_default_connection(session, agent.user_id)

    if conn is not None:
        if conn.kind == "openai_chatgpt":
            from contextlib import asynccontextmanager

            from app.agents.services.oauth_refresh import credential_access

            connection_id = conn.id
            connection_user_id = conn.user_id

            @asynccontextmanager
            async def access_token(*, force=False, rejected_version=None):
                async with credential_access(
                    session,
                    connection_id,
                    connection_user_id,
                    force=force,
                    rejected_version=rejected_version,
                ) as snapshot:
                    yield snapshot

            try:
                provider = connection_service.build_provider_for_connection(
                    conn, access_token=access_token
                )
            except LLMError:
                raise
            except Exception:
                raise LLMAuthError("ChatGPT OAuth provider is unavailable") from None
        else:
            provider = connection_service.build_provider_for_connection(conn)
        model = agent.model or conn.default_model or os.getenv("AGENTS_DEFAULT_MODEL", "")
        return provider, model

    return _provider_for(agent), _model_for(agent)


_RUNTIME_GUARDRAIL = (
    "## Runtime rules (always apply, regardless of the agent's own prompt)\n"
    "\n"
    "1. Tools whose name starts with `propose_` are PREVIEWS, not actions. "
    "Calling them does NOT change the user's data. They return a structured "
    "proposal and the UI renders an Apply button. After calling one, "
    "describe it as a proposal — say 'I prepared a proposal…' or 'Here's a "
    "preview…'. NEVER say 'I created', 'I added', 'Done', 'Ready', or "
    "anything that implies the action has been executed.\n"
    "\n"
    "2. Don't silently substitute entities the user named. If the user "
    "asks for a category/account/payee/group 'X' and the lookup returns "
    "no match, you MUST stop and tell the user before doing anything else. "
    "Never quietly pick a similar-looking one (e.g. user said 'Coleguinhas', "
    "only 'Amigos' exists → STOP, ask, do not build a proposal against "
    "'Amigos'). For categories specifically: if the user-named category "
    "doesn't exist, prefer leaving `category_id` null in the transaction "
    "proposal and mention it in your reply — don't chain a "
    "propose_create_category step unless the user explicitly asked to "
    "create the category.\n"
    "\n"
    "3. The auto-context primer (when present) is orientation only — never "
    "quote balances or counts from it; query the tools for live numbers.\n"
    "\n"
    "4. When responding in Portuguese (or any non-English language), keep "
    "your phrasing language-consistent — don't mix English snippets like "
    "'I prepared a proposal' into a Portuguese reply. Use 'Preparei uma "
    "proposta…' / 'Aqui está uma prévia…'.\n"
    "\n"
    "5. Charts: when a visualization would clearly help (trends over "
    "time, category breakdowns, comparisons), render one inline by "
    "emitting a fenced code block tagged `securo-chart`. The body is "
    "JSON. Multi-series example:\n"
    "\n"
    "```securo-chart\n"
    "{\n"
    '  "type": "line",            // line | bar | area | pie\n'
    '  "title": "Income vs expenses (last 6 months)",\n'
    '  "currency": "BRL",         // optional, formats Y axis as money\n'
    '  "data": [\n'
    '    {"x": "Jan", "income": 3200, "expense": 2100},\n'
    '    {"x": "Feb", "income": 3400, "expense": 2300}\n'
    "  ],\n"
    '  "series": [\n'
    '    {"key": "income",  "name": "Income"},\n'
    '    {"key": "expense", "name": "Expense"}\n'
    "  ]\n"
    "}\n"
    "```\n"
    "\n"
    "Single-series shorthand: omit `series` and use the key `y`. "
    'Example: `{\"type\":\"bar\",\"data\":[{\"x\":\"Food\",\"y\":500},'
    '{\"x\":\"Rent\",\"y\":1200}]}`. For pie use '
    '`{\"type\":\"pie\",\"data\":[{\"name\":\"Food\",\"value\":500},...]}`.\n'
    "\n"
    "Add a one-sentence summary above the chart; do NOT also list every "
    "data point in prose — let the chart speak. Only render charts when "
    "you have at least 2 data points to show.\n"
    "\n"
    "6. Knowledge base: if `search_knowledge_base` is available, the user "
    "has uploaded reference documents to this agent. Treat the KB as the "
    "authoritative source for anything that is NOT plainly transactional "
    "data in the user's accounts — including laws, regulations, tax rules, "
    "accounting standards, contracts, internal policies, project briefs, "
    "team/people names, internal codes, identifiers, dates, addresses, "
    "definitions, jargon, or any domain-specific knowledge.\n"
    "\n"
    "   - You MUST call `search_knowledge_base` before any answer that "
    "claims something is or isn't in the documents. Phrases like 'não "
    "encontrei nos documentos', 'I didn't find this in the docs', or 'isso "
    "não consta' are forbidden unless you actually called the tool first "
    "and got no relevant result. Pre-judging without searching is a bug, "
    "even if you think the question is out of scope.\n"
    "   - Use the user's own wording as the query (translate if needed). "
    "Try a second query with related terms if the first returns nothing.\n"
    "   - If the KB returns relevant chunks, answer from them and cite the "
    "doc title or filename inline (e.g. 'segundo o briefing X…').\n"
    "   - If the KB returns nothing relevant (no items, or only low-score "
    "chunks unrelated to the question), DO NOT invent an answer and DO NOT "
    "fall back to general world knowledge as if it were authoritative. "
    "Say plainly that you searched and didn't find this in the uploaded "
    "documents, and offer to look at it differently if the user can share "
    "a doc or rephrase. It's better to say 'I don't know' than to guess.\n"
    "\n"
    "7. Trust the tools' arithmetic. When `aggregate` or any other server-"
    "side total returns a value, that IS the answer — quote it directly. "
    "NEVER list the individual transactions and re-sum them by hand: model "
    "arithmetic over long lists is unreliable and will drift from the SQL "
    "result. If you need a keyword filter (merchant name, payee), pass "
    "`description_contains` to `aggregate` rather than listing and "
    "summing. The only acceptable place to do arithmetic yourself is a "
    "single combine step on tool outputs (e.g. `1847 * 12` = 22164).\n"
)


def _format_page_context(page_context: Optional[dict[str, Any]]) -> Optional[str]:
    """Render the frontend's page-context blob as a short system message.

    The frontend builds a free-form dict — typically:
      {"path": "/transactions", "label": "Transactions",
       "filters": {...}, "selection": {...}, "summary": "..."}
    We render it as a compact bullet list so the model can parse it but
    the cost stays small. Empty/None values are dropped.
    """
    if not page_context:
        return None
    path = page_context.get("path") or page_context.get("route")
    label = page_context.get("label") or path
    lines = ["## Current page context",
             "The user is sending this message from the page below — when they refer "
             "to 'this', 'these', 'aqui', etc., it most likely refers to what's on "
             "this page right now."]
    if label or path:
        lines.append(f"- **Page:** {label or '?'}{f' ({path})' if (path and label != path) else ''}")
    summary = page_context.get("summary")
    if summary:
        lines.append(f"- **Summary:** {summary}")
    filters = page_context.get("filters")
    if filters:
        try:
            parts = ", ".join(f"{k}={v}" for k, v in filters.items() if v not in (None, "", []))
            if parts:
                lines.append(f"- **Active filters:** {parts}")
        except Exception:  # noqa: BLE001
            lines.append(f"- **Active filters:** {filters!r}")
    selection = page_context.get("selection")
    if selection:
        try:
            n = len(selection) if hasattr(selection, "__len__") else None
            if isinstance(selection, list) and n:
                lines.append(f"- **Selected items ({n}):** {selection[:5]}{' …' if n > 5 else ''}")
            else:
                lines.append(f"- **Selection:** {selection!r}")
        except Exception:  # noqa: BLE001
            lines.append(f"- **Selection:** {selection!r}")
    extra = {k: v for k, v in page_context.items() if k not in {"path", "route", "label", "summary", "filters", "selection"}}
    for k, v in extra.items():
        if v in (None, "", [], {}):
            continue
        lines.append(f"- **{k}:** {v}")
    return "\n".join(lines) if len(lines) > 2 else None


def _validate_request_value(value: Any, *, label: str) -> None:
    nodes = 0
    total_bytes = 0
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_LLM_JSON_NODES or depth > _MAX_LLM_JSON_DEPTH:
            raise LLMNotSupportedError(f"{label} is too complex")
        if isinstance(item, str):
            if "\x00" in item:
                raise LLMNotSupportedError(f"{label} contains an invalid string")
            try:
                total_bytes += len(item.encode())
            except UnicodeEncodeError:
                raise LLMNotSupportedError(f"{label} contains invalid Unicode") from None
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise LLMNotSupportedError(f"{label} contains an invalid key")
                if "\x00" in key:
                    raise LLMNotSupportedError(f"{label} contains an invalid key")
                try:
                    total_bytes += len(key.encode())
                except UnicodeEncodeError:
                    raise LLMNotSupportedError(
                        f"{label} contains invalid Unicode"
                    ) from None
                stack.append((child, depth + 1))
        elif isinstance(item, (list, tuple)):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise LLMNotSupportedError(f"{label} contains a non-finite number")
        elif item is not None and not isinstance(item, (bool, int, float)):
            raise LLMNotSupportedError(f"{label} contains an unsupported value")
        if total_bytes > _MAX_LLM_FIELD_BYTES:
            raise LLMNotSupportedError(f"{label} exceeded the size limit")


def _build_agent_identity_primer(agent: Agent) -> str:
    """Baseline persona injected BEFORE the user's system_prompt.

    Tells the model two things it had no way to know before:
      1. It's running inside Securo — an open-source, self-hosted
         personal-finance app — and what kind of help that implies.
      2. Its own name and stated role, taken from the agent row itself
         (the user picked them in the UI).

    The user's `system_prompt` then runs AFTER this, so it can extend
    or override anything here without losing the product framing.
    """
    name = (agent.name or "Assistant").strip()
    description = (agent.description or "").strip()
    lines = [
        "## Who you are",
        f"You are **{name}**, an AI assistant running inside **Securo** — an "
        "open-source, self-hosted personal-finance app the user owns and "
        "runs on their own infrastructure. The user is the owner of the "
        "data you operate on; everything you read/write belongs to them.",
    ]
    if description:
        lines.append(f"\nYour stated role / specialty: {description}")
    lines.append(
        "\nGround rules:\n"
        "- Answer in the user's language (Portuguese, English, Spanish, etc.).\n"
        "- You have tools that read the user's data and `propose_*` tools "
        "that draft changes for the user to approve — those never apply "
        "by themselves.\n"
        "- Be concise, direct, and willing to make calls. Don't pad with "
        "warnings or boilerplate."
    )
    return "\n".join(lines)


def _safe_exception_kind(exc: BaseException) -> str:
    """Return an exact built-in type name without invoking exception hooks."""
    try:
        name = type.__getattribute__(type(exc), "__name__")
        return name if type(name) is str else "Exception"
    except BaseException:
        return "Exception"


def _classify_error(exc: BaseException) -> tuple[str, str]:
    """Map arbitrary failures to fixed public output without invoking hooks."""
    try:
        actual_type = type(exc)
        if actual_type is LLMAuthError:
            return ("auth", "LLM provider rejected the credentials.")
        if actual_type is LLMRateLimitError:
            return ("rate_limit", "LLM provider is rate-limiting.")
        if actual_type is LLMUnavailableError:
            return ("unavailable", "LLM provider is unavailable.")
        if actual_type is LLMNotSupportedError:
            return ("not_supported", "LLM provider does not support this request.")
        if actual_type is LLMError:
            return ("unknown", "LLM request failed.")
        try:
            mro = type.__getattribute__(actual_type, "__mro__")
            is_llm_failure = type(mro) is tuple and any(
                base is LLMError for base in tuple.__iter__(mro)
            )
        except BaseException:
            is_llm_failure = False
        if is_llm_failure:
            return ("unavailable", "LLM provider is unavailable.")
    except BaseException:
        pass
    return ("unknown", "LLM request failed.")


class AgentExecutor:
    def __init__(self, *, mcp: Optional[MCPRegistry] = None):
        self.mcp = mcp or MCPRegistry()
        self.settings = get_agent_settings()

    async def run(
        self,
        *,
        session: AsyncSession,
        agent: Agent,
        user_id: uuid.UUID,
        workspace_id: Optional[uuid.UUID] = None,
        conversation_id: uuid.UUID,
        user_message: str,
        channel: str = "web",
        page_context: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[ExecutorEvent]:
        # OAuth credential access deliberately rolls the request session back
        # around provider I/O. Snapshot scalar agent fields used afterwards so
        # an expired ORM instance never triggers implicit async I/O.
        agent_id = agent.id
        agent_temperature = agent.temperature
        try:
            _validate_request_value(user_message, label="User message")
            if page_context is not None:
                _validate_request_value(page_context, label="Page context")
        except LLMNotSupportedError as exc:
            code, msg = _classify_error(exc)
            yield ExecutorEvent(type="error", error_code=code, error_message=msg)
            yield ExecutorEvent(type="done", finish_reason="error")
            return
        # 1. Persist the user message first so it survives crashes.
        await conversation_service.append_message(
            session, conversation_id=conversation_id, role="user", content=user_message
        )
        await conversation_service.update_title_if_empty(session, conversation_id, user_message)

        # 2. Build the message list. Order, top to bottom:
        #      1. Runtime guardrail (app-level invariants — always)
        #      2. Agent identity primer (who you are + Securo framing)
        #      3. User-defined system_prompt (extends or overrides #2)
        #      4. Auto-context primer (user data: name, currency, accounts)
        #      5. Page-context primer (where the user is right now)
        #      6. Conversation history
        #      7. The new user message (appended in step 4 below)
        history = await conversation_service.list_messages(session, conversation_id, limit=agent.max_history_messages * 2 + 2)
        # The window can open mid-turn: a turn with parallel tool calls has an
        # odd number of rows, so the cut can land on a tool result whose
        # assistant call fell outside it. Providers reject an orphaned tool
        # result, so replay from the first user message in the window.
        while history and history[0].role != "user":
            history.pop(0)
        messages: list[ChatMessage] = []
        # Runtime guardrail goes FIRST and applies to every conversation,
        # regardless of agent settings or per-agent system prompt. Locks
        # in the propose-vs-action framing and the no-silent-substitution
        # rule. Agents can build on top of these but can't undo them.
        messages.append(ChatMessage(role="system", content=_RUNTIME_GUARDRAIL))
        # Agent identity — name + description + Securo framing. Ensures
        # the model knows what product it lives in and what role it was
        # configured for, even when the user leaves system_prompt blank.
        messages.append(ChatMessage(role="system", content=_build_agent_identity_primer(agent)))
        # User-defined system prompt extends/overrides the identity
        # primer. Goes BEFORE the data primers so it shapes the persona,
        # not just the per-turn answer.
        if agent.system_prompt and agent.system_prompt.strip():
            messages.append(ChatMessage(role="system", content=agent.system_prompt))
        # Optional context primer — user name, currency, accounts, etc.
        # Cheap orientation so the agent doesn't need to call list_accounts
        # on every "what's my balance?" question.
        if getattr(agent, "auto_context", True):
            try:
                from app.models.user import User
                user = await session.get(User, user_id)
                if user is not None:
                    primer = await context_service.build_context_primer(
                        session, user, workspace_id=agent.workspace_id
                    )
                    if primer:
                        messages.append(ChatMessage(role="system", content=primer))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "context primer failed; continuing without it (kind=%s)",
                    _safe_exception_kind(exc),
                )
        # Page-context primer — orientation about where the user is when
        # they sent THIS message. Goes after the agent's prompt so the
        # base persona always wins, but before history so prior turns
        # don't overshadow the current page.
        page_primer = _format_page_context(page_context)
        if page_primer:
            messages.append(ChatMessage(role="system", content=page_primer))
        for m in history:
            tcs = []
            for raw in (m.tool_calls or []):
                tcs.append(ToolCall(
                    id=raw.get("id"), name=raw.get("name"), arguments=raw.get("arguments") or {},
                    thought_signature=raw.get("thought_signature"),
                ))
            tool_call_id = (m.tool_result or {}).get("tool_call_id") if m.role == "tool" else None
            content = m.content
            if m.role == "tool":
                # Persisted content is the exact provider-facing function output.
                # Older rows may lack it, so retain the structured fallback.
                tr = m.tool_result or {}
                if content is None:
                    content = tr.get("text") or _safe_json(tr.get("data"))
            messages.append(ChatMessage(
                role=cast(Role, m.role),
                content=content,
                tool_calls=tcs,
                tool_call_id=tool_call_id,
            ))

        # 3. Discover tools from MCP and filter by per-agent whitelist.
        try:
            handles = await self.mcp.discover(
                user_id=user_id,
                workspace_id=workspace_id,
                conversation_id=conversation_id,
                agent_id=agent_id,
            )
        except Exception as exc:
            logger.error(
                "MCP discovery failed; running without tools (kind=%s)",
                _safe_exception_kind(exc),
            )
            handles = []
        allowed = await agent_service.allowed_tool_pairs(session, agent_id)
        tool_defs = self.mcp.to_provider_tools(handles, allowed=allowed)
        wire_name_counts: dict[str, int] = {}
        for tool in tool_defs:
            wire_name_counts[tool.name] = wire_name_counts.get(tool.name, 0) + 1
        tool_defs = [
            tool
            for tool in tool_defs
            if tool.name.count("__") == 1
            and all(tool.name.split("__", 1))
            and wire_name_counts[tool.name] == 1
        ]
        advertised_wire_names = frozenset(tool.name for tool in tool_defs)

        # Resolve provider+model once per request. Monkey-patched in tests
        # via _provider_for; production prefers _provider_and_model_for.
        try:
            provider, model = await _provider_and_model_for(session, agent)
        except LLMError as exc:
            code, msg = _classify_error(exc)
            logger.error("provider resolution failed (category=%s)", code)
            yield ExecutorEvent(type="error", error_code=code, error_message=msg)
            yield ExecutorEvent(type="done", finish_reason="error")
            return
        except Exception as exc:  # noqa: BLE001
            if agent.connection_id is not None:
                logger.error(
                    "explicit provider resolution failed (kind=%s)",
                    _safe_exception_kind(exc),
                )
                error = LLMAuthError("The configured connection is invalid")
                code, msg = _classify_error(error)
                yield ExecutorEvent(type="error", error_code=code, error_message=msg)
                yield ExecutorEvent(type="done", finish_reason="error")
                return
            logger.error(
                "provider resolution failed; falling back to legacy path (kind=%s)",
                _safe_exception_kind(exc),
            )
            provider = _provider_for(agent)
            model = _model_for(agent)
        if not model:
            yield ExecutorEvent(
                type="error",
                error_code="config",
                error_message="Agent has no model configured. Pick a connection or set agent.model.",
            )
            yield ExecutorEvent(type="done", finish_reason="error")
            return

        # 4. Tool-calling loop. Cap iterations to prevent runaway agents.
        MAX_ITERS = 6
        for iteration in range(MAX_ITERS):
            text_buf: list[str] = []
            model_events: list[ExecutorEvent] = []
            open_calls: dict[str, dict] = {}
            stream_totals = {
                "text_bytes": 0,
                "all_args_bytes": 0,
                "metadata_bytes": 0,
            }
            finish_reason = "stop"
            finish_seen = False
            usage_seen = False
            usage_input = 0
            usage_output = 0
            chunk_count = 0
            iter_start = time.time()
            provider_failure: BaseException | None = None
            provider_cancelled = False
            provider_source: AsyncIterator[ChatChunk] | None = None
            provider_stream: AsyncIterator[ChatChunk] | None = None
            cancellation = OperationCancellationScope()
            try:
                await cancellation.checkpoint()
                provider_source = provider.chat_stream(
                    messages,
                    model=model,
                    tools=tool_defs or None,
                    temperature=agent_temperature,
                )
                await asyncio.sleep(0)
                provider_stream = bounded_async_iterator(
                    provider_source,
                    timeout_seconds=LLM_OPERATION_TIMEOUT_SECONDS,
                )
                provider_source = None
                async for raw_chunk in provider_stream:
                    chunk_count += 1
                    if chunk_count > MAX_STREAM_CHUNKS:
                        raise LLMUnavailableError("LLM stream returned too many chunks")
                    chunk = validate_chat_chunk(
                        raw_chunk,
                        max_text_bytes=_MAX_LLM_TEXT_BYTES,
                        max_tool_argument_bytes=_MAX_LLM_TOOL_ARGUMENT_BYTES,
                        max_tool_metadata_bytes=_MAX_LLM_TOOL_METADATA_BYTES,
                        max_usage_tokens=_MAX_LLM_USAGE_TOKENS,
                    )
                    if finish_seen:
                        raise LLMUnavailableError("LLM stream continued after completion")
                    if usage_seen:
                        if chunk.type == "usage":
                            raise LLMUnavailableError("LLM returned duplicate usage")
                        if chunk.type != "finish":
                            raise LLMUnavailableError("LLM stream continued after usage")
                    async for ev in _process_chunk(
                        chunk, text_buf, open_calls, stream_totals
                    ):
                        model_events.append(ev)
                    if chunk.type == "finish":
                        if any(not tc["ended"] for tc in open_calls.values()):
                            raise LLMUnavailableError("LLM stream was incomplete")
                        finish_reason = cast(str, chunk.finish_reason)
                        finish_seen = True
                    elif chunk.type == "usage":
                        if any(not tc["ended"] for tc in open_calls.values()):
                            raise LLMUnavailableError("LLM stream was incomplete")
                        usage = chunk.usage
                        if usage is None:  # validated above; keep type narrowing explicit
                            raise LLMUnavailableError("LLM returned invalid usage")
                        usage_input = usage.input_tokens
                        usage_output = usage.output_tokens
                        usage_seen = True
                if (
                    provider_failure is None
                    and not provider_cancelled
                    and (not finish_seen or any(not tc["ended"] for tc in open_calls.values()))
                ):
                    raise LLMUnavailableError("LLM stream was incomplete")
            except asyncio.CancelledError as failure:
                if cancellation.requested():
                    provider_cancelled = True
                else:
                    provider_failure = failure
            except BaseException as failure:  # noqa: BLE001 - total provider boundary
                provider_failure = failure
            finally:
                if provider_source is not None:
                    try:
                        close = getattr(provider_source, "aclose", None)
                        if close is not None:
                            async with asyncio.timeout(LLM_CLEANUP_TIMEOUT_SECONDS):
                                await close()
                    except BaseException as cleanup_failure:
                        _discard_untrusted_exception(cleanup_failure)
                    provider_source = None
                if provider_stream is not None:
                    try:
                        await provider_stream.aclose()
                    except asyncio.CancelledError:
                        provider_cancelled = True
                    except BaseException as cleanup_failure:
                        if provider_failure is None:
                            provider_failure = LLMUnavailableError(
                                "LLM stream cleanup failed"
                            )
                        _discard_untrusted_exception(cleanup_failure)
                provider_stream = None

            if provider_cancelled:
                if provider_failure is not None:
                    _discard_untrusted_exception(provider_failure)
                provider_failure = None
                raise asyncio.CancelledError from None

            if provider_failure is not None:
                failure = provider_failure
                try:
                    failure_mro = type.__getattribute__(type(failure), "__mro__")
                    is_llm_failure = type(failure_mro) is tuple and any(
                        base is LLMError for base in tuple.__iter__(failure_mro)
                    )
                except BaseException:
                    is_llm_failure = False
                if is_llm_failure:
                    code, msg = _classify_error(failure)
                    log_message = "LLM provider call failed (category=%s)"
                    log_value = code
                else:
                    code, msg = ("unavailable", "LLM provider failed unexpectedly.")
                    log_message = "provider stream failed unexpectedly (kind=%s)"
                    log_value = _safe_exception_kind(failure)
                _discard_untrusted_exception(failure)
                failure = None
                provider_failure = None
                logger.error(log_message, log_value)
                yield ExecutorEvent(type="error", error_code=code, error_message=msg)
                yield ExecutorEvent(type="done", finish_reason="error")
                return

            assistant_text = "".join(text_buf)
            assembled_calls: list[ToolCall] = []
            try:
                for tc in open_calls.values():
                    import json

                    args_text = "".join(tc["args_parts"])
                    try:
                        args = (
                            json.loads(
                                args_text,
                                object_pairs_hook=_unique_json_members,
                                parse_constant=lambda _value: (_ for _ in ()).throw(
                                    ValueError("invalid JSON constant")
                                ),
                            )
                            if args_text
                            else {}
                        )
                    except (json.JSONDecodeError, ValueError, RecursionError):
                        raise LLMUnavailableError(
                            "LLM returned malformed tool arguments"
                        ) from None
                    if not isinstance(args, dict):
                        raise LLMUnavailableError(
                            "LLM returned non-object tool arguments"
                        )
                    try:
                        _validate_request_value(args, label="LLM tool arguments")
                    except LLMNotSupportedError:
                        raise LLMUnavailableError(
                            "LLM returned invalid tool arguments"
                        ) from None
                    assembled_calls.append(ToolCall(
                        id=tc["id"], name=tc["name"], arguments=args,
                        thought_signature=tc.get("thought_signature"),
                    ))
                _validate_tool_call_batch(
                    assembled_calls, advertised_wire_names=advertised_wire_names
                )
            except LLMError as exc:
                code, msg = _classify_error(exc)
                logger.error("LLM tool arguments were invalid (category=%s)", code)
                yield ExecutorEvent(type="error", error_code=code, error_message=msg)
                yield ExecutorEvent(type="done", finish_reason="error")
                return

            for event in model_events:
                yield event

            # Persist assistant turn.
            assistant_msg = await conversation_service.append_message(
                session,
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_text or None,
                tool_calls=[{
                    "id": c.id, "name": c.name, "arguments": c.arguments,
                    **({"thought_signature": c.thought_signature}
                       if c.thought_signature is not None else {}),
                } for c in assembled_calls] or None,
                input_tokens=usage_input or None,
                output_tokens=usage_output or None,
            )
            messages.append(ChatMessage(role="assistant", content=assistant_text or None, tool_calls=assembled_calls))

            # Record one usage row per provider call. Best-effort: a logging
            # failure should never break the user's chat.
            try:
                await usage_service.record_usage(
                    session,
                    user_id=user_id,
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    message_id=assistant_msg.id,
                    provider=provider.name,
                    model=model,
                    kind="chat",
                    input_tokens=usage_input,
                    output_tokens=usage_output,
                    latency_ms=int((time.time() - iter_start) * 1000),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "failed to record llm usage (kind=%s)",
                    _safe_exception_kind(exc),
                )

            if not assembled_calls:
                # If the model returned nothing at all (no text, no tool
                # call, no usage), the endpoint probably isn't actually
                # an LLM — surface that as a friendly error instead of a
                # silent close that leaves the UI hanging. Most common
                # cause: openai_compatible base_url missing the /v1 path.
                if not assistant_text and not usage_input and not usage_output:
                    yield ExecutorEvent(
                        type="error",
                        error_code="empty_response",
                        error_message=(
                            f"The {provider.name} endpoint returned no content. "
                            "Check that the connection's base URL is correct and points to an OpenAI-compatible /v1 root, "
                            "and that the model name matches what's loaded on the server."
                        ),
                    )
                    yield ExecutorEvent(type="done", finish_reason="error")
                    return
                yield ExecutorEvent(type="done", finish_reason=finish_reason)
                return

            # 5. Run tool calls in parallel, persist + emit results, loop.
            for ev in [ExecutorEvent(type="tool_call", tool_name=c.name, tool_args=c.arguments) for c in assembled_calls]:
                yield ev

            results = await asyncio.gather(*[
                _safe_call_tool(self.mcp, c, advertised_wire_names=advertised_wire_names, user_id=user_id, workspace_id=workspace_id, conversation_id=conversation_id, agent_id=agent_id)
                for c in assembled_calls
            ])
            for c, res in zip(assembled_calls, results):
                try:
                    _validate_request_value(res, label="Tool result")
                except LLMNotSupportedError:
                    res = {
                        "ok": False,
                        "data": None,
                        "text": "Tool output was invalid or exceeded the size limit",
                    }
                # Two views of the same result:
                #   - `summary` is a SHORT preview for the UI chip header
                #     (e.g. "5 items returned"). Truncating this is fine.
                #   - `llm_content` is the FULL JSON for the model to read.
                #     Truncating this is what caused the model to think the
                #     data was incomplete and report fake "truncated" rows.
                summary = _summarize_result(res)
                llm_content = _safe_json(res.get("data") if res.get("data") is not None else res.get("text"))
                yield ExecutorEvent(type="tool_result", tool_name=c.name, tool_result=summary)
                await conversation_service.append_message(
                    session,
                    conversation_id=conversation_id,
                    role="tool",
                    content=llm_content,
                    tool_result={
                        "tool_call_id": c.id,
                        "name": c.name,
                        "data": summary.get("data"),
                        "ok": summary.get("ok", False),
                    },
                )
                messages.append(ChatMessage(
                    role="tool",
                    content=llm_content,
                    tool_call_id=c.id,
                    name=c.name,
                ))

        # Reached the tool-call ceiling without the model producing a final
        # answer. Emit a visible fallback so the UI doesn't show an empty
        # assistant bubble, and persist it so the conversation has a
        # readable transcript. The friendliest message reuses any text we
        # accumulated mid-loop if there is some.
        fallback = (
            "Não consegui completar essa consulta — pedi muitas ferramentas em sequência "
            "e o limite foi atingido. Reformule a pergunta de forma mais específica e eu tento "
            "de novo (por exemplo, restrinja a um período ou a uma categoria)."
        )
        yield ExecutorEvent(type="text_delta", text=fallback)
        await conversation_service.append_message(
            session, conversation_id=conversation_id, role="assistant", content=fallback,
        )
        yield ExecutorEvent(type="error", error_code="max_iterations", error_message="Agent reached its tool-call limit.")
        yield ExecutorEvent(type="done", finish_reason="max_iterations")


async def _process_chunk(
    chunk: ChatChunk,
    text_buf: list[str],
    open_calls: dict[str, dict],
    stream_totals: dict[str, int],
) -> AsyncIterator[ExecutorEvent]:
    chunk = validate_chat_chunk(
        chunk,
        max_text_bytes=_MAX_LLM_TEXT_BYTES,
        max_tool_argument_bytes=_MAX_LLM_TOOL_ARGUMENT_BYTES,
        max_tool_metadata_bytes=_MAX_LLM_TOOL_METADATA_BYTES,
        max_usage_tokens=_MAX_LLM_USAGE_TOKENS,
    )
    if chunk.type == "text_delta":
        if not isinstance(chunk.text, str):
            raise LLMUnavailableError("LLM returned an invalid text delta")
        if "\x00" in chunk.text:
            raise LLMUnavailableError("LLM returned an invalid text delta")
        try:
            delta_bytes = len(chunk.text.encode())
        except UnicodeEncodeError:
            raise LLMUnavailableError("LLM returned invalid Unicode") from None
        if delta_bytes > _MAX_LLM_TEXT_BYTES - stream_totals["text_bytes"]:
            raise LLMUnavailableError("LLM response text exceeded the size limit")
        if not chunk.text:
            return
        stream_totals["text_bytes"] += delta_bytes
        text_buf.append(chunk.text)
        yield ExecutorEvent(type="text_delta", text=chunk.text)
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
        retained_metadata = stream_totals.setdefault("metadata_bytes", 0)
        if (
            id_bytes > _MAX_LLM_TOOL_METADATA_BYTES
            or name_bytes > _MAX_LLM_TOOL_METADATA_BYTES
            or metadata_bytes
            > _MAX_LLM_ALL_TOOL_METADATA_BYTES - retained_metadata
        ):
            raise LLMUnavailableError("LLM tool metadata exceeded the size limit")
        if chunk.tool_call_id in open_calls:
            raise LLMUnavailableError("LLM returned a duplicate tool call")
        if len(open_calls) >= _MAX_LLM_TOOL_CALLS:
            raise LLMUnavailableError("LLM returned too many tool calls")
        stream_totals["metadata_bytes"] = retained_metadata + metadata_bytes
        open_calls[chunk.tool_call_id] = {
            "id": chunk.tool_call_id,
            "name": chunk.tool_name or "",
            "args_parts": [],
            "args_bytes": 0,
            "ended": False,
        }
    elif chunk.type == "tool_call_args_delta":
        call_id = cast(str, chunk.tool_call_id)
        tc = open_calls.get(call_id)
        if tc is None:
            raise LLMUnavailableError("LLM returned tool arguments before the tool call")
        if tc["ended"]:
            raise LLMUnavailableError("LLM stream continued after completion")
        delta = cast(str, chunk.args_delta)
        delta_bytes = len(delta.encode())
        if not delta:
            return
        if (
            delta_bytes > _MAX_LLM_TOOL_ARGUMENT_BYTES - tc["args_bytes"]
            or delta_bytes
            > _MAX_LLM_ALL_TOOL_ARGUMENT_BYTES - stream_totals["all_args_bytes"]
        ):
            raise LLMUnavailableError("LLM tool arguments exceeded the size limit")
        tc["args_bytes"] += delta_bytes
        stream_totals["all_args_bytes"] += delta_bytes
        tc["args_parts"].append(delta)
    elif chunk.type == "tool_call_end":
        tc = open_calls.get(cast(str, chunk.tool_call_id))
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
            retained_metadata = stream_totals.setdefault("metadata_bytes", 0)
            if (
                signature_bytes > _MAX_LLM_TOOL_METADATA_BYTES
                or signature_bytes
                > _MAX_LLM_ALL_TOOL_METADATA_BYTES - retained_metadata
            ):
                raise LLMUnavailableError("LLM tool metadata exceeded the size limit")
            stream_totals["metadata_bytes"] = retained_metadata + signature_bytes
        tc["thought_signature"] = chunk.thought_signature
        tc["ended"] = True
    elif chunk.type in {"finish", "usage"}:
        return
    else:
        raise LLMUnavailableError("LLM returned an unknown chunk type")


def _validate_tool_call_batch(
    calls: list[ToolCall], *, advertised_wire_names: frozenset[str]
) -> None:
    """Authorize a complete model-produced batch before any observable side effect."""
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for call in calls:
        if (
            not isinstance(call.id, str)
            or not call.id
            or not isinstance(call.name, str)
            or call.name.count("__") != 1
            or not all(call.name.split("__", 1))
            or call.name not in advertised_wire_names
            or not isinstance(call.arguments, dict)
            or call.id in seen_ids
            or call.name in seen_names
        ):
            raise LLMUnavailableError("LLM returned an unauthorized tool call batch")
        seen_ids.add(call.id)
        seen_names.add(call.name)


async def _safe_call_tool(
    mcp: MCPRegistry,
    call: ToolCall,
    *,
    advertised_wire_names: frozenset[str],
    user_id: uuid.UUID,
    workspace_id: Optional[uuid.UUID] = None,
    conversation_id: uuid.UUID,
    agent_id: Optional[uuid.UUID] = None,
) -> dict[str, Any]:
    started = time.time()
    if call.name not in advertised_wire_names or not isinstance(call.arguments, dict):
        logger.warning("blocked unadvertised tool call")
        return {
            "ok": False,
            "data": None,
            "text": "Tool error: tool was not advertised for this turn",
            "elapsed_ms": 0,
        }
    try:
        return await mcp.call(
            wire_name=call.name,
            arguments=call.arguments,
            user_id=user_id,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
            agent_id=agent_id,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "tool call %s failed (kind=%s)", call.name, _safe_exception_kind(exc)
        )
        return {
            "ok": False,
            "data": None,
            "text": "Tool execution failed.",
            "elapsed_ms": int((time.time() - started) * 1000),
        }


def _summarize_result(res: dict[str, Any]) -> dict[str, Any]:
    """Compact summary used by the UI tool-call chip — just a one-line
    hint plus the structured `data` (which the chip's expand view shows
    in full). Never feed this `text` back to the LLM directly; the LLM
    needs the complete payload, see executor.run().
    """
    data = res.get("data")
    short = ""
    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            total = data.get("total")
            n = total if isinstance(total, int) else len(data["items"])
            short = f"{n} item(s) returned"
        elif "error" in data:
            short = f"error: {data['error']}"
        elif "kind" in data:
            short = f"{data['kind']}"
    return {
        "ok": bool(res.get("ok", False)),
        "data": data,
        "text": short or None,
    }


def _safe_json(obj: Any) -> str:
    """Serialize tool data for the LLM to read. No length cap — the
    model needs the full payload or it'll hallucinate truncation."""
    import json
    try:
        _validate_request_value(obj, label="Tool output")
        return json.dumps(obj, default=str)
    except Exception:
        return '"Tool output was invalid or exceeded the size limit"'
