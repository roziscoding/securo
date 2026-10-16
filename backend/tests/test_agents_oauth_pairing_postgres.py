"""PostgreSQL locking and atomicity coverage for SIWC pairing."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from app.agents.models.connection import LlmConnection
from app.agents.models.oauth_credential import OpenAIOAuthCredential
from app.agents.models.oauth_pairing import OpenAIOAuthPairing
from app.agents.services import oauth_pairing
from app.agents.services import connection_service
from app.agents.services.oauth_credentials import CredentialBundle
from app.agents.services.openai_oauth import ValidatedTokenBundle
from app.models.user import User


async def _setup(factory):
    async with factory() as session:
        user = User(
            id=uuid.uuid4(), email=f"pair-{uuid.uuid4()}@example.com",
            hashed_password="unused", is_active=True, is_superuser=False,
            is_verified=True, preferences={},
        )
        connection = LlmConnection(
            id=uuid.uuid4(), user_id=user.id, name="ChatGPT",
            kind="openai_chatgpt", extra={},
        )
        session.add(user)
        await session.flush()
        session.add(connection)
        await session.commit()
        started = await oauth_pairing.start_pairing(session, connection.id, user.id)
        return user.id, connection.id, started


async def _claim(factory, started):
    attempt = uuid.uuid4()
    payload = {"access_token": "access", "refresh_token": "refresh", "id_token": "id"}
    async with factory() as session:
        await oauth_pairing.register_attempt(
            session, started.id, started.secret, attempt, "n" * 43,
            "http://127.0.0.1:42311/auth/callback",
        )
        lease = await oauth_pairing.claim_completion(
            session,
            started.id,
            started.secret,
            attempt,
            oauth_pairing.completion_digest(attempt, "issued", payload),
        )
    validated = ValidatedTokenBundle(
        bundle=CredentialBundle("access", "refresh", "id"),
        subject="subject", scopes=frozenset({"openid"}),
        access_token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        earliest_refresh_at=None, refresh_token_expires_at=None,
    )
    return lease, validated


@pytest.mark.asyncio
async def test_postgres_conflicting_attempt_race_has_one_winner(postgres_sessions):
    user_id, connection_id, started = await _setup(postgres_sessions)
    first, second = uuid.uuid4(), uuid.uuid4()

    async def register(attempt):
        async with postgres_sessions() as session:
            try:
                await oauth_pairing.register_attempt(
                    session, started.id, started.secret, attempt, "n" * 43,
                    "http://127.0.0.1:42311/auth/callback",
                )
                return "won"
            except oauth_pairing.PairingUnavailable:
                return "lost"

    assert sorted(await asyncio.gather(register(first), register(second))) == ["lost", "won"]
    async with postgres_sessions() as session:
        rows = await session.scalar(
            select(func.count()).select_from(OpenAIOAuthPairing).where(
                OpenAIOAuthPairing.connection_id == connection_id
            )
        )
        assert rows == 1


@pytest.mark.asyncio
async def test_postgres_cancelled_completion_cannot_write_credential(postgres_sessions):
    user_id, connection_id, started = await _setup(postgres_sessions)
    attempt = uuid.uuid4()
    async with postgres_sessions() as session:
        await oauth_pairing.register_attempt(
            session, started.id, started.secret, attempt, "n" * 43,
            "http://127.0.0.1:42311/auth/callback",
        )
        digest = oauth_pairing.completion_digest(attempt, "issued", {"access_token": "x"})
        lease = await oauth_pairing.claim_completion(
            session, started.id, started.secret, attempt, digest
        )
        assert await oauth_pairing.cancel_pairing(
            session, started.id, connection_id, user_id
        )
        validated = ValidatedTokenBundle(
            bundle=CredentialBundle("access", "refresh", "id"),
            subject="subject", scopes=frozenset({"openid"}),
            access_token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            earliest_refresh_at=None, refresh_token_expires_at=None,
        )
        with pytest.raises(oauth_pairing.PairingUnavailable):
            await oauth_pairing.persist_validated_completion(
                session, started.id, lease, "issued", validated
            )
        count = await session.scalar(
            select(func.count()).select_from(OpenAIOAuthCredential).where(
                OpenAIOAuthCredential.connection_id == connection_id
            )
        )
        assert count == 0


@pytest.mark.asyncio
async def test_postgres_twenty_exact_completions_serialize_and_replay(postgres_sessions):
    _user_id, connection_id, started = await _setup(postgres_sessions)
    attempt = uuid.uuid4()
    payload = {"access_token": "access", "refresh_token": "refresh", "id_token": "id"}
    digest = oauth_pairing.completion_digest(attempt, "issued", payload)
    async with postgres_sessions() as session:
        await oauth_pairing.register_attempt(
            session, started.id, started.secret, attempt, "n" * 43,
            "http://127.0.0.1:42311/auth/callback",
        )

    async def claim():
        async with postgres_sessions() as session:
            try:
                return await asyncio.wait_for(
                    oauth_pairing.claim_completion(
                        session, started.id, started.secret, attempt, digest
                    ),
                    timeout=10,
                )
            except oauth_pairing.CredentialBusy:
                return "busy"

    claims = await asyncio.gather(*(claim() for _ in range(20)))
    leases = [claim for claim in claims if claim != "busy"]
    assert len(leases) == 1
    validated = ValidatedTokenBundle(
        bundle=CredentialBundle("access", "refresh", "id"),
        subject="subject", scopes=frozenset({"openid"}),
        access_token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        earliest_refresh_at=None, refresh_token_expires_at=None,
    )
    async with postgres_sessions() as session:
        await oauth_pairing.persist_validated_completion(
            session, started.id, leases[0], "issued", validated
        )

    replays = await asyncio.gather(*(claim() for _ in range(20)))
    assert replays == ["completed"] * 20
    async with postgres_sessions() as session:
        credentials = await session.scalar(
            select(func.count()).select_from(OpenAIOAuthCredential).where(
                OpenAIOAuthCredential.connection_id == connection_id
            )
        )
        assert credentials == 1


@pytest.mark.asyncio
async def test_postgres_start_vs_complete_obeys_lock_order_without_deadlock(postgres_sessions):
    user_id, connection_id, started = await _setup(postgres_sessions)
    lease, validated = await _claim(postgres_sessions, started)

    async def complete():
        async with postgres_sessions() as session:
            await oauth_pairing.persist_validated_completion(
                session, started.id, lease, "issued", validated
            )
            return "completed"

    async def restart():
        async with postgres_sessions() as session:
            with pytest.raises(oauth_pairing.PairingUnavailable):
                await oauth_pairing.start_pairing(session, connection_id, user_id)
            return "blocked"

    assert sorted(
        await asyncio.wait_for(asyncio.gather(complete(), restart()), timeout=10)
    ) == ["blocked", "completed"]


@pytest.mark.asyncio
async def test_postgres_cancel_vs_complete_has_one_terminal_outcome_without_deadlock(
    postgres_sessions,
):
    user_id, connection_id, started = await _setup(postgres_sessions)
    lease, validated = await _claim(postgres_sessions, started)

    async def complete():
        async with postgres_sessions() as session:
            try:
                await oauth_pairing.persist_validated_completion(
                    session, started.id, lease, "issued", validated
                )
                return "completed"
            except oauth_pairing.PairingUnavailable:
                return "cancelled"

    async def cancel():
        async with postgres_sessions() as session:
            return await oauth_pairing.cancel_pairing(
                session, started.id, connection_id, user_id
            )

    completion, cancelled = await asyncio.wait_for(
        asyncio.gather(complete(), cancel()), timeout=10
    )
    assert (completion, cancelled) in {("completed", False), ("cancelled", True)}


@pytest.mark.asyncio
async def test_postgres_delete_vs_complete_has_no_deadlock_or_orphan(postgres_sessions):
    user_id, connection_id, started = await _setup(postgres_sessions)
    lease, validated = await _claim(postgres_sessions, started)

    async def complete():
        async with postgres_sessions() as session:
            try:
                await oauth_pairing.persist_validated_completion(
                    session, started.id, lease, "issued", validated
                )
                return "completed"
            except oauth_pairing.PairingUnavailable:
                return "deleted"

    async def delete():
        async with postgres_sessions() as session:
            try:
                return "deleted" if await connection_service.delete_connection(
                    session, connection_id, user_id
                ) else "missing"
            except ValueError:
                await session.rollback()
                return "credential_live"

    outcomes = await asyncio.wait_for(
        asyncio.gather(complete(), delete()), timeout=10
    )
    assert outcomes in [
        ["completed", "credential_live"],
        ["deleted", "deleted"],
    ]
    async with postgres_sessions() as session:
        credential = await session.scalar(
            select(OpenAIOAuthCredential).where(
                OpenAIOAuthCredential.connection_id == connection_id
            )
        )
        if outcomes[0] == "completed":
            assert credential is not None and credential.token_bundle_encrypted is not None
        else:
            assert credential is None
