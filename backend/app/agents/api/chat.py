"""SSE chat endpoint.

POST /api/agents/{agent_id}/chat with {content, conversation_id?, channel?}.
Returns Server-Sent Events streaming text + tool-call + result events. The
client UI renders deltas as they arrive and replaces tool-call spinners
with summaries when results land.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from app.agents.providers.base import (
    LLM_OPERATION_TIMEOUT_SECONDS,
    OperationCancellationScope,
    bounded_async_iterator,
    discard_untrusted_exception as _discard_untrusted_exception,
)
from app.agents.runtime.executor import AgentExecutor, ExecutorEvent
from app.agents.schemas.conversation import SendMessageRequest
from app.agents.services import agent_service, conversation_service
from app.core.database import get_async_session
from app.core.workspace_context import WorkspaceContext, current_writable_workspace

router = APIRouter(prefix="/api/agents", tags=["agents"])

_UNEXPECTED_ERROR_EVENT = (
    b'event: error\ndata: {"error_code": "unknown", '
    b'"error_message": "Agent request failed unexpectedly."}\n\n'
)


def _format_event(event: ExecutorEvent) -> bytes:
    payload = {k: v for k, v in asdict(event).items() if v is not None}
    return f"event: {event.type}\ndata: {json.dumps(payload)}\n\n".encode()


async def _stream_executor_events(
    events: AsyncIterator[ExecutorEvent],
) -> AsyncGenerator[bytes, None]:
    """Serialize executor events with total, operation-local boundaries."""
    stream: AsyncIterator[ExecutorEvent] | None = None
    cancelled = False
    terminal_error = False
    generator_closed = False
    cancellation = OperationCancellationScope()
    try:
        await cancellation.checkpoint()
        try:
            stream = bounded_async_iterator(
                events,
                timeout_seconds=LLM_OPERATION_TIMEOUT_SECONDS,
                timeout_error="Agent request timed out",
                cleanup_error="Agent stream cleanup failed",
            )
        except BaseException:  # noqa: BLE001 - untrusted iterator construction
            terminal_error = True

        while stream is not None and not terminal_error and not cancelled:
            try:
                event = await anext(stream)
                payload = _format_event(event)
            except StopAsyncIteration:
                break
            except asyncio.CancelledError:
                if cancellation.requested():
                    cancelled = True
                else:
                    terminal_error = True
                break
            except BaseException:  # noqa: BLE001 - model-derived event boundary
                terminal_error = True
                break

            # Keep the yield outside untrusted guards so ``aclose()`` retains
            # normal async-generator closure semantics.
            yield payload
            if (
                type(event) is ExecutorEvent
                and type(event.type) is str
                and event.type == "done"
            ):
                break
    except GeneratorExit:
        # Suppress the injected close signal until owned cleanup finishes so
        # cancellation delivered during cleanup can remain authoritative.
        generator_closed = True
    finally:
        if stream is not None:
            try:
                await stream.aclose()
            except asyncio.CancelledError:
                cancelled = True
            except BaseException as cleanup_failure:
                _discard_untrusted_exception(cleanup_failure)
                if not terminal_error and not generator_closed:
                    terminal_error = True
        stream = None
        del events

    if cancelled:
        raise asyncio.CancelledError from None
    if generator_closed:
        return
    if terminal_error:
        yield _UNEXPECTED_ERROR_EVENT


@router.post("/{agent_id}/chat")
async def chat(
    agent_id: uuid.UUID,
    body: SendMessageRequest,
    # Write-gated for what the agent can do on the caller's behalf, not for
    # the conversation row it persists. The tool set reachable from here
    # includes `propose_create_transaction` and its siblings, which write —
    # so a read-only member chatting could create financial data the HTTP
    # API would have refused them.
    #
    # This deliberately also blocks a viewer from *asking* questions, which
    # is a real use case (a read-only accountant reading the books). The way
    # to restore it is to make the tools role-aware so a viewer's session
    # only exposes the reading ones — not to drop this gate, which would
    # bring the write path back with it.
    ctx: WorkspaceContext = Depends(current_writable_workspace),
    session: AsyncSession = Depends(get_async_session),
):
    agent = await agent_service.get_agent(session, agent_id, ctx.workspace.id)
    if agent is None:
        raise HTTPException(status_code=404, detail="agent not found")

    conv = None
    if body.conversation_id:
        conv = await conversation_service.get_conversation(session, body.conversation_id, ctx.workspace.id)
        if conv is None or conv.agent_id != agent_id:
            raise HTTPException(status_code=404, detail="conversation not found")
    if conv is None:
        conv = await conversation_service.create_conversation(
            session,
            workspace_id=ctx.workspace.id,
            user_id=ctx.user_id,
            agent_id=agent_id,
            channel=body.channel,
        )

    executor = AgentExecutor()

    async def gen() -> AsyncIterator[bytes]:
        # Send the conversation id immediately so the client can update its URL.
        yield f"event: conversation\ndata: {json.dumps({'conversation_id': str(conv.id)})}\n\n".encode()
        stream = bounded_async_iterator(
            _stream_executor_events(
                executor.run(
                    session=session,
                    agent=agent,
                    user_id=ctx.user_id,
                    workspace_id=ctx.workspace.id,
                    conversation_id=conv.id,
                    user_message=body.content,
                    channel=body.channel,
                    page_context=body.page_context,
                )
            ),
            timeout_seconds=LLM_OPERATION_TIMEOUT_SECONDS,
            timeout_error="Agent request timed out",
            cleanup_error="Agent stream cleanup failed",
        )
        try:
            async for payload in stream:
                yield payload
        finally:
            await stream.aclose()

    return StreamingResponse(gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    })
