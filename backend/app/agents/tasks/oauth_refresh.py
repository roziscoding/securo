"""Periodic refresh of active SIWC credentials approaching access expiry."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from billiard.exceptions import SoftTimeLimitExceeded
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.agents.models.connection import LlmConnection
from app.agents.models.oauth_credential import OpenAIOAuthCredential
from app.agents.services.oauth_credentials import RESOURCE_CLEANUP_PENDING
from app.agents.services.oauth_refresh import refresh_credential
from app.agents.services.openai_oauth import OAuthLifecycleError, _discard_exception_secrets
from app.core.config import get_settings
from app.worker import celery_app

_REFRESH_HORIZON = timedelta(minutes=5)
_MAX_CANDIDATES = 100
_FAILURE_BACKOFF = timedelta(minutes=5)
_SELECTION_CLAIM = timedelta(seconds=55)


def _due_credentials_statement(current_time: datetime, *, limit: int = _MAX_CANDIDATES):
    cutoff = current_time + _REFRESH_HORIZON
    return (
        select(
            OpenAIOAuthCredential.connection_id,
            LlmConnection.user_id,
            OpenAIOAuthCredential.credential_version,
            OpenAIOAuthCredential.access_token_expires_at,
        )
        .join(LlmConnection)
        .where(
            OpenAIOAuthCredential.status == "active",
            OpenAIOAuthCredential.token_bundle_encrypted.is_not(None),
            or_(
                OpenAIOAuthCredential.last_error_code.is_(None),
                OpenAIOAuthCredential.last_error_code != RESOURCE_CLEANUP_PENDING,
            ),
            OpenAIOAuthCredential.access_token_expires_at <= cutoff,
            or_(
                OpenAIOAuthCredential.earliest_refresh_at.is_(None),
                OpenAIOAuthCredential.earliest_refresh_at <= current_time,
            ),
            or_(
                OpenAIOAuthCredential.refresh_retry_after.is_(None),
                OpenAIOAuthCredential.refresh_retry_after <= func.now(),
            ),
        )
        .order_by(
            OpenAIOAuthCredential.access_token_expires_at,
            OpenAIOAuthCredential.connection_id,
        )
        .limit(limit)
    )


async def _claim_due_credentials(
    session: AsyncSession, current_time: datetime, *, limit: int = _MAX_CANDIDATES
):
    """Atomically reserve a fair batch before releasing the database connection."""
    statement = _due_credentials_statement(current_time, limit=limit)
    if session.get_bind().dialect.name == "postgresql":
        statement = statement.with_for_update(
            skip_locked=True, of=OpenAIOAuthCredential
        )
    candidates = list((await session.execute(statement)).all())
    if not candidates:
        await session.rollback()
        return candidates
    connection_ids = [candidate.connection_id for candidate in candidates]
    claim_until = (
        func.now() + _SELECTION_CLAIM
        if session.get_bind().dialect.name == "postgresql"
        else current_time + _SELECTION_CLAIM
    )
    await session.execute(
        update(OpenAIOAuthCredential)
        .where(OpenAIOAuthCredential.connection_id.in_(connection_ids))
        .values(refresh_retry_after=claim_until)
    )
    await session.commit()
    return candidates


async def _record_attempt(
    session: AsyncSession,
    connection_id,
    user_id,
    *,
    expected_version: int,
    observed_access_expiry: datetime,
    failed: bool,
    current_time: datetime,
) -> bool:
    owned_connection = select(LlmConnection.id).where(
        LlmConnection.id == connection_id, LlmConnection.user_id == user_id
    )
    if failed:
        retry_after = (
            func.now() + _FAILURE_BACKOFF
            if session.get_bind().dialect.name == "postgresql"
            else current_time + _FAILURE_BACKOFF
        )
        values = {
            "refresh_attempt_count": OpenAIOAuthCredential.refresh_attempt_count + 1,
            "refresh_retry_after": retry_after,
        }
    else:
        values = {"refresh_attempt_count": 0, "refresh_retry_after": None}
    database_now = (
        func.now()
        if session.get_bind().dialect.name == "postgresql"
        else func.datetime("now")
    )
    result = await session.execute(
        update(OpenAIOAuthCredential)
        .where(
            OpenAIOAuthCredential.connection_id == connection_id,
            OpenAIOAuthCredential.connection_id.in_(owned_connection),
            OpenAIOAuthCredential.credential_version == expected_version,
            OpenAIOAuthCredential.status == "active",
            or_(
                OpenAIOAuthCredential.last_error_code.is_(None),
                OpenAIOAuthCredential.last_error_code != RESOURCE_CLEANUP_PENDING,
            ),
            OpenAIOAuthCredential.access_token_expires_at == observed_access_expiry,
            or_(
                OpenAIOAuthCredential.lifecycle_lease_token.is_(None),
                OpenAIOAuthCredential.lifecycle_lease_expires_at <= database_now,
            ),
        )
        .values(**values)
        .returning(OpenAIOAuthCredential.id)
    )
    await session.commit()
    return result.scalar_one_or_none() is not None


async def _do_refresh_due(session_maker, *, now: datetime | None = None) -> dict[str, int]:
    current_time = now or datetime.now(timezone.utc)
    async with session_maker() as session:
        candidates = await _claim_due_credentials(session, current_time)

    stats = {
        "selected": len(candidates),
        "refreshed": 0,
        "transient_failures": 0,
        "definitive_failures": 0,
        "unexpected_failures": 0,
    }
    for connection_id, user_id, observed_version, observed_expiry in candidates:
        async with session_maker() as session:
            try:
                await refresh_credential(
                    session,
                    connection_id,
                    user_id,
                )
                stats["refreshed"] += 1
                await _record_attempt(
                    session,
                    connection_id,
                    user_id,
                    expected_version=observed_version,
                    observed_access_expiry=observed_expiry,
                    failed=False,
                    current_time=current_time,
                )
            except SoftTimeLimitExceeded as exc:
                _discard_exception_secrets(exc)
                raise
            except OAuthLifecycleError as exc:
                key = "transient_failures" if exc.retryable else "definitive_failures"
                stats[key] += 1
                await _record_attempt(
                    session,
                    connection_id,
                    user_id,
                    expected_version=observed_version,
                    observed_access_expiry=observed_expiry,
                    failed=True,
                    current_time=current_time,
                )
            except Exception:  # noqa: BLE001
                # One malformed row or unforeseen provider error must not
                # prevent other credentials from being refreshed.
                await session.rollback()
                stats["unexpected_failures"] += 1
                await _record_attempt(
                    session,
                    connection_id,
                    user_id,
                    expected_version=observed_version,
                    observed_access_expiry=observed_expiry,
                    failed=True,
                    current_time=current_time,
                )
    return stats


async def _run_periodic_refresh() -> dict[str, int]:
    from app.agents.config import get_agent_settings

    if not get_agent_settings().enabled:
        return {"disabled": 1}
    engine = create_async_engine(
        get_settings().database_url,
        poolclass=NullPool,
        hide_parameters=True,
    )
    session_maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        return await _do_refresh_due(session_maker)
    finally:
        await engine.dispose()


@celery_app.task(
    name="app.agents.tasks.oauth_refresh.refresh_near_expiry",
    bind=True,
    max_retries=1,
    default_retry_delay=30,
    soft_time_limit=55,
    time_limit=65,
)
def refresh_near_expiry(self) -> dict[str, int]:
    """Refresh independently and retry a transient aggregate at most once."""
    from app.agents.config import get_agent_settings

    if not get_agent_settings().enabled:
        return {"disabled": 1}
    try:
        result = asyncio.run(_run_periodic_refresh())
    except SoftTimeLimitExceeded as exc:
        _discard_exception_secrets(exc)
        raise
    if result["transient_failures"]:
        raise self.retry()
    return result
