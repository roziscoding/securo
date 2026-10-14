from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.schemas.connection import ConnectionCreate, ConnectionRead, ConnectionTestResult, ConnectionUpdate
from app.agents.services import connection_service
from app.core.auth import current_active_user
from app.core.database import get_async_session
from app.models.user import User

# Mounted BEFORE the agents/{agent_id} router so the literal path wins.
router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("/connections", response_model=list[ConnectionRead])
async def list_connections(
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    rows = await connection_service.list_connections(session, user.id)
    return [ConnectionRead.from_orm_row(r) for r in rows]


@router.post("/connections", response_model=ConnectionRead, status_code=status.HTTP_201_CREATED)
async def create_connection(
    data: ConnectionCreate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    try:
        conn = await connection_service.create_connection(
            session, user.id,
            name=data.name, kind=data.kind, base_url=data.base_url, api_key=data.api_key,
            default_model=data.default_model, extra=data.extra, is_default=data.is_default,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ConnectionRead.from_orm_row(conn)


@router.get("/connections/{conn_id}", response_model=ConnectionRead)
async def get_connection(
    conn_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    conn = await connection_service.get_connection(session, conn_id, user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="connection not found")
    return ConnectionRead.from_orm_row(conn)


@router.patch("/connections/{conn_id}", response_model=ConnectionRead)
async def update_connection(
    conn_id: uuid.UUID,
    data: ConnectionUpdate,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    try:
        conn = await connection_service.update_connection(
            session, conn_id, user.id,
            name=data.name, base_url=data.base_url, api_key=data.api_key,
            default_model=data.default_model, extra=data.extra, is_default=data.is_default,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if conn is None:
        raise HTTPException(status_code=404, detail="connection not found")
    return ConnectionRead.from_orm_row(conn)


@router.delete("/connections/{conn_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_connection(
    conn_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    try:
        ok = await connection_service.delete_connection(session, conn_id, user.id)
    except ValueError as exc:
        await session.rollback()
        raise HTTPException(status_code=409, detail=str(exc)) from None
    if not ok:
        raise HTTPException(status_code=404, detail="connection not found")


@router.post("/connections/{conn_id}/revoke", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_connection(
    conn_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    conn = await connection_service.get_connection(session, conn_id, user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="connection not found")
    if conn.kind != "openai_chatgpt" or conn.oauth_credential is None:
        raise HTTPException(status_code=409, detail="connection has no ChatGPT OAuth credential")

    from app.agents.services.oauth_refresh import revoke_credential
    from app.agents.services.openai_oauth import OAuthLifecycleError

    try:
        await revoke_credential(session, conn.id, user.id)
    except OAuthLifecycleError as exc:
        raise HTTPException(
            status_code=503 if exc.retryable else 502, detail=str(exc)
        ) from None


@router.post("/connections/{conn_id}/test", response_model=ConnectionTestResult)
async def test_connection(
    conn_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    conn = await connection_service.get_connection(session, conn_id, user.id)
    if conn is None:
        raise HTTPException(status_code=404, detail="connection not found")
    if conn.kind == "openai_chatgpt":
        return await connection_service.test_connection(conn, session=session)
    return await connection_service.test_connection(conn)
