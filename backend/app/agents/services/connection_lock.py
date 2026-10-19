"""Serialize credential creation and connection removal on the owned parent."""
import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models.connection import LlmConnection


async def lock_owned_connection(session: AsyncSession, connection_id: uuid.UUID, user_id: uuid.UUID):
    owned = (LlmConnection.id == connection_id, LlmConnection.user_id == user_id)
    if session.get_bind().dialect.name == "sqlite":
        # SQLite ignores FOR UPDATE. A write acquires its transaction-wide
        # writer lock before reading the credential or lifecycle state.
        await session.execute(update(LlmConnection).where(*owned).values(id=LlmConnection.id))
    return (await session.execute(
        select(LlmConnection).where(*owned).with_for_update(of=LlmConnection)
        .execution_options(populate_existing=True)
    )).scalar_one_or_none()
