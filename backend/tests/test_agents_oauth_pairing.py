"""Secure remote SIWC pairing state-machine coverage."""
from __future__ import annotations

import uuid
import asyncio
import types
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.requests import Request

from app.agents.api.oauth_pairing import (
    MAX_PAIRING_BODY,
    _secret,
    _status,
    _strict_body,
    helper_bootstrap,
    helper_status,
    user_pairing_status,
)
from app.agents.schemas.oauth_pairing import CompletionRequest, PairingStatus
from app.agents.models.oauth_credential import OpenAIOAuthCredential
from app.agents.models.connection import LlmConnection
from app.agents.models.oauth_pairing import OpenAIOAuthPairing
from app.agents.services import oauth_pairing
from app.agents.config import MAX_OAUTH_TIMEOUT_SECONDS
from app.main import app
from app.models.user import User


def test_pairing_status_contract_is_closed_and_sanitizes_failure_codes():
    now = datetime.now(timezone.utc)
    view = oauth_pairing.PairingView(uuid.uuid4(), "attempt_registered", now, failure_code="transient")

    assert _status(view, include_failure=True).failure_code == "transient"
    assert _status(view).failure_code is None
    assert _status(
        oauth_pairing.PairingView(view.id, "failed", now, failure_code="token=secret"),
        include_failure=True,
    ).failure_code is None
    with pytest.raises(ValueError):
        PairingStatus.model_validate(
            {"pairing_id": view.id, "state": "unknown", "expires_at": now}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["user", "helper_status", "helper_bootstrap"])
async def test_status_routes_fail_closed_on_hostile_persisted_state(
    route, monkeypatch, caplog
):
    sentinel = "hostile-persisted-state-sentinel"
    view = oauth_pairing.PairingView(
        uuid.uuid4(), sentinel, datetime.now(timezone.utc), failure_code=sentinel
    )

    async def hostile_view(*_args, **_kwargs):
        return view

    monkeypatch.setattr(
        oauth_pairing,
        "get_user_status" if route == "user" else "get_helper_status",
        hostile_view,
    )

    with pytest.raises(Exception) as caught:
        if route == "user":
            await user_pairing_status(
                uuid.uuid4(), view.id, cast(AsyncSession, object()),
                cast(User, types.SimpleNamespace(id=uuid.uuid4())),
            )
        elif route == "helper_status":
            await helper_status(view.id, "s" * 43, object())
        else:
            await helper_bootstrap(view.id, "s" * 43, object())

    assert getattr(caught.value, "status_code", None) == 404
    assert getattr(caught.value, "detail", None) == {"code": "pairing_not_available"}
    assert sentinel not in repr(caught.value)
    assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_user_status_sanitizes_hostile_persisted_failure_code(monkeypatch, caplog):
    sentinel = "hostile-persisted-failure-sentinel"
    view = oauth_pairing.PairingView(
        uuid.uuid4(), "failed", datetime.now(timezone.utc), failure_code=sentinel
    )

    async def hostile_view(*_args, **_kwargs):
        return view

    monkeypatch.setattr(oauth_pairing, "get_user_status", hostile_view)
    result = await user_pairing_status(
        uuid.uuid4(), view.id, cast(AsyncSession, object()),
        cast(User, types.SimpleNamespace(id=uuid.uuid4())),
    )

    assert result.failure_code is None
    assert sentinel not in result.model_dump_json()
    assert sentinel not in caplog.text


def test_pairing_status_openapi_is_closed_sanitized_and_secret_free():
    schema = app.openapi()["components"]["schemas"]["PairingStatus"]
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {
        "pairing_id", "state", "expires_at", "failure_code"
    }
    assert schema["properties"]["state"]["enum"] == [
        "pending", "attempt_registered", "completing", "completed", "cancelled", "failed"
    ]
    failure_schema = schema["properties"]["failure_code"]
    failure_enum = next(item["enum"] for item in failure_schema["anyOf"] if "enum" in item)
    assert set(failure_enum) == {
        "transient", "revoked", "scope_loss", "corrupt_token", "reauth_required",
        "too_early", "race_lost", "redirect_rejected", "malformed_response", "inactive",
        "invalid_client", "expired", "refresh_outcome_unknown", "revocation_pending",
        "resource_cleanup_pending", "credential_busy", "start_outcome_unknown",
    }
    assert "secret" not in str(schema).lower()


@pytest.mark.asyncio
async def test_start_stores_only_domain_separated_secret_digest(session, clean_db, test_user):
    connection = LlmConnection(
        user_id=test_user.id, name="ChatGPT", kind="openai_chatgpt", extra={}
    )
    session.add(connection)
    await session.commit()

    started = await oauth_pairing.start_pairing(session, connection.id, test_user.id)

    assert len(started.secret) == 43
    row = await session.scalar(select(OpenAIOAuthPairing).where(OpenAIOAuthPairing.id == started.id))
    assert row is not None
    assert row.secret_digest != started.secret
    raw = (await session.execute(text("select secret_digest from agent_openai_oauth_pairings"))).scalar_one()
    assert started.secret not in raw
    assert oauth_pairing.secret_matches(started.secret, raw)


@pytest.mark.asyncio
async def test_start_commit_failure_returns_fixed_clean_error_without_raw_secret(
    session, clean_db, test_user, monkeypatch
):
    connection = LlmConnection(
        user_id=test_user.id, name="ChatGPT", kind="openai_chatgpt", extra={}
    )
    session.add(connection)
    await session.commit()
    raw = bytes(range(32))
    encoded = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"
    generated = iter(raw)
    monkeypatch.setattr(oauth_pairing.secrets, "randbits", lambda count: next(generated))

    async def fail_commit():
        raise RuntimeError(encoded)

    monkeypatch.setattr(session, "commit", fail_commit)
    monkeypatch.setattr(
        oauth_pairing,
        "_new_start_cleanup_session",
        async_sessionmaker(session.bind, expire_on_commit=False),
    )
    with pytest.raises(oauth_pairing.PairingUnavailable) as caught:
        await oauth_pairing.start_pairing(session, connection.id, test_user.id)

    assert caught.value.args == ("pairing_not_available",)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    traceback = caught.value.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith(
            "/app/agents/services/oauth_pairing.py"
        ):
            values = traceback.tb_frame.f_locals.values()
            assert encoded not in values
            assert all(not (isinstance(value, bytearray) and bytes(value) == raw) for value in values)
        traceback = traceback.tb_next


@pytest.mark.asyncio
async def test_start_commit_then_raise_cancels_durable_row_in_independent_session(
    session, clean_db, test_user, monkeypatch
):
    connection = LlmConnection(
        user_id=test_user.id, name="ChatGPT", kind="openai_chatgpt", extra={}
    )
    session.add(connection)
    await session.commit()
    connection_id = connection.id
    user_id = test_user.id
    original_commit = session.commit
    cleanup_factory = async_sessionmaker(session.bind, expire_on_commit=False)
    cleanup_sessions = []
    closed_sessions = []

    async def commit_then_raise():
        await original_commit()
        raise RuntimeError("commit outcome unknown")

    def new_cleanup_session():
        cleanup = cleanup_factory()
        cleanup_sessions.append(cleanup)
        real_close = cleanup.close

        async def tracked_close():
            closed_sessions.append(cleanup)
            await real_close()

        monkeypatch.setattr(cleanup, "close", tracked_close)
        return cleanup

    monkeypatch.setattr(session, "commit", commit_then_raise)
    monkeypatch.setattr(oauth_pairing, "_new_start_cleanup_session", new_cleanup_session)

    with pytest.raises(oauth_pairing.PairingUnavailable, match="pairing_not_available"):
        await oauth_pairing.start_pairing(session, connection_id, user_id)

    assert cleanup_sessions and all(cleanup is not session for cleanup in cleanup_sessions)
    assert closed_sessions == cleanup_sessions
    async with cleanup_factory() as verify:
        row = await verify.scalar(
            select(OpenAIOAuthPairing).where(
                OpenAIOAuthPairing.connection_id == connection_id
            )
        )
        assert row is not None
        assert row.state == "cancelled"
        assert row.secret_digest is None
        assert row.attempt_id is None
        assert row.nonce is None
        assert row.redirect_uri is None
        assert row.completion_digest is None
        assert row.completion_lease_token is None
        assert row.completion_lease_expires_at is None


@pytest.mark.asyncio
async def test_start_cleanup_reconciles_when_its_first_commit_is_ambiguous(
    session, clean_db, test_user, monkeypatch
):
    connection = LlmConnection(
        user_id=test_user.id, name="ChatGPT", kind="openai_chatgpt", extra={}
    )
    session.add(connection)
    await session.commit()
    connection_id = connection.id
    user_id = test_user.id
    original_commit = session.commit
    cleanup_factory = async_sessionmaker(session.bind, expire_on_commit=False)
    cleanup_commits = 0

    async def commit_then_raise():
        await original_commit()
        raise RuntimeError("start commit outcome unknown")

    def new_cleanup_session():
        nonlocal cleanup_commits
        cleanup = cleanup_factory()
        real_commit = cleanup.commit

        async def ambiguous_first_cleanup_commit():
            nonlocal cleanup_commits
            cleanup_commits += 1
            await real_commit()
            if cleanup_commits == 1:
                raise RuntimeError("cleanup commit outcome unknown")

        monkeypatch.setattr(cleanup, "commit", ambiguous_first_cleanup_commit)
        return cleanup

    monkeypatch.setattr(session, "commit", commit_then_raise)
    monkeypatch.setattr(oauth_pairing, "_new_start_cleanup_session", new_cleanup_session)

    with pytest.raises(oauth_pairing.PairingUnavailable, match="pairing_not_available"):
        await oauth_pairing.start_pairing(session, connection_id, user_id)

    assert cleanup_commits == 1
    async with cleanup_factory() as verify:
        row = await verify.scalar(select(OpenAIOAuthPairing))
        assert row is not None and row.state == "cancelled"
        assert row.secret_digest is None


@pytest.mark.asyncio
async def test_start_cleanup_survives_caller_cancellation_without_retaining_secret(
    session, clean_db, test_user, monkeypatch
):
    connection = LlmConnection(
        user_id=test_user.id, name="ChatGPT", kind="openai_chatgpt", extra={}
    )
    session.add(connection)
    await session.commit()
    connection_id = connection.id
    user_id = test_user.id
    original_commit = session.commit
    cleanup_factory = async_sessionmaker(session.bind, expire_on_commit=False)
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    captured = []

    async def commit_then_raise():
        await original_commit()
        raise RuntimeError("start commit outcome unknown")

    real_cleanup = oauth_pairing._cleanup_start_pairing

    async def delayed_cleanup(identity):
        captured.append(identity)
        cleanup_started.set()
        await release_cleanup.wait()
        return await real_cleanup(identity)

    monkeypatch.setattr(session, "commit", commit_then_raise)
    monkeypatch.setattr(oauth_pairing, "_new_start_cleanup_session", cleanup_factory)
    monkeypatch.setattr(oauth_pairing, "_cleanup_start_pairing", delayed_cleanup)

    task = asyncio.create_task(
        oauth_pairing.start_pairing(session, connection_id, user_id)
    )
    await cleanup_started.wait()
    task.cancel()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(captured) == 1
    identity = captured[0]
    assert identity.pairing_id is not None
    assert identity.connection_id == connection_id
    assert identity.user_id == user_id
    assert not hasattr(identity, "session")
    assert not hasattr(identity, "secret")
    async with cleanup_factory() as verify:
        row = await verify.scalar(select(OpenAIOAuthPairing))
        assert row is not None and row.state == "cancelled"
        assert row.secret_digest is None


@pytest.mark.asyncio
async def test_start_encoding_failure_cancels_committed_pairing(
    session, clean_db, test_user, monkeypatch
):
    connection = LlmConnection(
        user_id=test_user.id, name="ChatGPT", kind="openai_chatgpt", extra={}
    )
    session.add(connection)
    await session.commit()

    def fail_encode(_raw):
        raise RuntimeError("encoding failed")

    monkeypatch.setattr(oauth_pairing.base64, "urlsafe_b64encode", fail_encode)
    with pytest.raises(oauth_pairing.PairingUnavailable) as caught:
        await oauth_pairing.start_pairing(session, connection.id, test_user.id)

    assert caught.value.__context__ is None
    row = await session.scalar(
        select(OpenAIOAuthPairing).where(OpenAIOAuthPairing.connection_id == connection.id)
    )
    assert row is not None
    assert row.state == "cancelled"
    assert row.secret_digest is None


@pytest.mark.asyncio
async def test_attempt_registration_is_exactly_idempotent_and_conflicts(session, clean_db, test_user):
    connection = LlmConnection(
        user_id=test_user.id, name="ChatGPT", kind="openai_chatgpt", extra={}
    )
    session.add(connection)
    await session.commit()
    started = await oauth_pairing.start_pairing(session, connection.id, test_user.id)
    attempt = uuid.uuid4()
    nonce = "n" * 43
    redirect_uri = "http://127.0.0.1:43123/auth/callback"

    first = await oauth_pairing.register_attempt(
        session, started.id, started.secret, attempt, nonce, redirect_uri
    )
    second = await oauth_pairing.register_attempt(
        session, started.id, started.secret, attempt, nonce, redirect_uri
    )
    assert first.attempt_id == second.attempt_id == attempt

    with pytest.raises(oauth_pairing.PairingUnavailable):
        await oauth_pairing.register_attempt(
            session, started.id, started.secret, attempt, "x" * 43, redirect_uri
        )


@pytest.mark.asyncio
async def test_cancel_while_worker_holds_lease_prevents_terminal_write(session, clean_db, test_user):
    connection = LlmConnection(
        user_id=test_user.id, name="ChatGPT", kind="openai_chatgpt", extra={}
    )
    session.add(connection)
    await session.commit()
    started = await oauth_pairing.start_pairing(session, connection.id, test_user.id)
    attempt = uuid.uuid4()
    await oauth_pairing.register_attempt(
        session, started.id, started.secret, attempt, "n" * 43,
        "http://127.0.0.1:43123/auth/callback",
    )
    digest = oauth_pairing.completion_digest(attempt, "client", {"access_token": "a"})
    lease = await oauth_pairing.claim_completion(
        session, started.id, started.secret, attempt, digest
    )
    assert lease

    assert await oauth_pairing.cancel_pairing(session, started.id, connection.id, test_user.id)
    with pytest.raises(oauth_pairing.PairingUnavailable):
        await oauth_pairing.assert_completion_lease(session, started.id, lease)


@pytest.mark.asyncio
async def test_expiry_uses_database_clock(session, clean_db, test_user):
    connection = LlmConnection(
        user_id=test_user.id, name="ChatGPT", kind="openai_chatgpt", extra={}
    )
    session.add(connection)
    await session.commit()
    started = await oauth_pairing.start_pairing(session, connection.id, test_user.id)
    await session.execute(
        text("update agent_openai_oauth_pairings set expires_at=:past where id=:id"),
        {"past": datetime.now(timezone.utc) - timedelta(seconds=1), "id": started.id.hex},
    )
    await session.commit()
    with pytest.raises(oauth_pairing.PairingUnavailable):
        await oauth_pairing.get_helper_status(session, started.id, started.secret)


def test_helper_auth_rejects_duplicate_authorization_headers():
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/pairing",
        "headers": [
            (b"authorization", b"SecuroPairing " + b"a" * 43),
            (b"Authorization", b"SecuroPairing " + b"a" * 43),
        ],
    })
    with pytest.raises(Exception) as caught:
        _secret(request)
    assert getattr(caught.value, "status_code", None) == 404


@pytest.mark.asyncio
async def test_helper_status_boundary_detaches_pairing_secret_on_unexpected_failure(
    monkeypatch,
):
    pairing_secret = "pairing-secret-sentinel-" + "s" * 43
    failure = RuntimeError(pairing_secret)

    async def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(oauth_pairing, "get_helper_status", fail)
    with pytest.raises(Exception) as caught:
        await helper_status(uuid.uuid4(), pairing_secret, object())

    assert getattr(caught.value, "status_code", None) == 500
    assert failure.args == ()
    assert failure.__traceback__ is None
    assert failure.__cause__ is None
    assert failure.__context__ is None


@pytest.mark.asyncio
async def test_helper_status_totally_scrubs_hostile_cancellation(monkeypatch):
    sentinel = "pairing-hostile-cancel-secret"

    class HostileCancellation(asyncio.CancelledError):
        __slots__ = ("request", "response", "retained")

        def __getattribute__(self, name):
            if name in {"args", "__dict__", "__traceback__", "__cause__", "__context__"}:
                raise RuntimeError("hostile-cancellation-accessor")
            return asyncio.CancelledError.__getattribute__(self, name)

        def __setattr__(self, name, value):
            if name in {"args", "__dict__", "__traceback__", "__cause__", "__context__"}:
                raise RuntimeError("hostile-cancellation-accessor")
            return asyncio.CancelledError.__setattr__(self, name, value)

    retained = HostileCancellation(sentinel)
    retained.request = {"authorization": sentinel}
    retained.response = {"token": sentinel}
    retained.retained = [sentinel]

    async def cancel(*_args, **_kwargs):
        raise retained

    monkeypatch.setattr(oauth_pairing, "get_helper_status", cancel)
    with pytest.raises(asyncio.CancelledError) as caught:
        await helper_status(uuid.uuid4(), "s" * 43, object())

    assert type(caught.value) is asyncio.CancelledError
    assert caught.value.args == ()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert BaseException.__dict__["args"].__get__(retained, HostileCancellation) == ()
    assert retained.request is None
    assert retained.response is None
    assert retained.retained is None

    pending: list[object] = [caught.value]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, str):
            assert sentinel not in value
        elif isinstance(value, BaseException):
            pending.extend(value.args)
            pending.extend((value.__cause__, value.__context__, value.__traceback__))
        elif isinstance(value, types.TracebackType):
            pending.extend((value.tb_next, value.tb_frame))
        elif isinstance(value, types.FrameType):
            if not value.f_code.co_filename.endswith("test_agents_oauth_pairing.py"):
                pending.extend(value.f_locals.values())


@pytest.mark.asyncio
async def test_strict_body_stops_stream_at_limit_plus_one():
    consumed = 0

    async def receive():
        nonlocal consumed
        consumed += 1
        return {
            "type": "http.request",
            "body": b"x",
            "more_body": consumed <= MAX_PAIRING_BODY,
        }

    request = Request({
        "type": "http", "method": "POST", "path": "/complete",
        "headers": [(b"content-type", b"application/json")],
    }, receive)
    with pytest.raises(Exception) as caught:
        await _strict_body(request, CompletionRequest)
    assert getattr(caught.value, "status_code", None) == 400
    assert consumed == MAX_PAIRING_BODY + 1


@pytest.mark.asyncio
async def test_start_rejects_live_credential_but_allows_revoked_scrubbed(
    session, clean_db, test_user
):
    connection = LlmConnection(
        user_id=test_user.id, name="ChatGPT", kind="openai_chatgpt", extra={}
    )
    session.add(connection)
    await session.flush()
    credential = OpenAIOAuthCredential(
        connection_id=connection.id, issuer="https://auth.openai.com", subject="subject",
        issued_client_id="issued", scopes=["openid"], status="active",
        token_bundle_encrypted="encrypted-live-bundle",
        access_token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        credential_version=1,
    )
    session.add(credential)
    await session.commit()
    connection_id = connection.id
    user_id = test_user.id
    with pytest.raises(oauth_pairing.PairingUnavailable):
        await oauth_pairing.start_pairing(session, connection_id, user_id)

    credential = await session.scalar(
        select(OpenAIOAuthCredential).where(
            OpenAIOAuthCredential.connection_id == connection_id
        )
    )
    assert credential is not None
    credential.status = "revoked"
    credential.token_bundle_encrypted = None
    await session.commit()
    first = await oauth_pairing.start_pairing(session, connection_id, user_id)
    first_view = await oauth_pairing.get_helper_status(session, first.id, first.secret)
    second = await oauth_pairing.start_pairing(session, connection_id, user_id)
    second_view = await oauth_pairing.get_helper_status(session, second.id, second.secret)
    assert first_view.ext_agent_host_id == second_view.ext_agent_host_id
    assert first_view.returning_client_id == second_view.returning_client_id == "issued"

    credential = await session.scalar(
        select(OpenAIOAuthCredential).where(
            OpenAIOAuthCredential.connection_id == connection_id
        )
    )
    assert credential is not None
    credential.status = "reauth_required"
    credential.last_error_code = "corrupt_token"
    await session.commit()
    scrubbed = await oauth_pairing.start_pairing(session, connection_id, user_id)
    scrubbed_view = await oauth_pairing.get_helper_status(
        session, scrubbed.id, scrubbed.secret
    )
    assert scrubbed_view.returning_client_id == "issued"


def test_host_identity_is_installation_stable_and_rotates_with_install_secret(monkeypatch):
    monkeypatch.setattr(oauth_pairing, "_key", lambda: b"installation-secret-one")
    first = oauth_pairing._host_id()
    second = oauth_pairing._host_id()

    monkeypatch.setattr(oauth_pairing, "_key", lambda: b"installation-secret-two")
    rotated = oauth_pairing._host_id()

    assert first == second
    assert first != rotated
    assert first.startswith("urn:uuid:")
    assert uuid.UUID(first.removeprefix("urn:uuid:")).version == 4
    assert "installation-secret" not in first


@pytest.mark.asyncio
async def test_host_identity_is_shared_by_different_connections(session, clean_db, test_user):
    connections = [
        LlmConnection(
            user_id=test_user.id, name=f"ChatGPT {index}", kind="openai_chatgpt", extra={}
        )
        for index in range(2)
    ]
    session.add_all(connections)
    await session.commit()

    started = [
        await oauth_pairing.start_pairing(session, connection.id, test_user.id)
        for connection in connections
    ]
    views = [
        await oauth_pairing.get_helper_status(session, pairing.id, pairing.secret)
        for pairing in started
    ]

    assert views[0].ext_agent_host_id == views[1].ext_agent_host_id


def test_completion_lease_exceeds_maximum_network_deadline_and_finalize_margin():
    assert oauth_pairing.COMPLETION_LEASE.total_seconds() > (
        MAX_OAUTH_TIMEOUT_SECONDS + oauth_pairing.COMPLETION_LEASE_MARGIN_SECONDS
    )
    assert oauth_pairing.COMPLETION_LEASE < oauth_pairing.PAIRING_TTL


@pytest.mark.asyncio
async def test_completion_is_not_claimed_when_pairing_cannot_outlive_full_lease(
    session, clean_db, test_user
):
    connection = LlmConnection(
        user_id=test_user.id, name="ChatGPT", kind="openai_chatgpt", extra={}
    )
    session.add(connection)
    await session.commit()
    started = await oauth_pairing.start_pairing(session, connection.id, test_user.id)
    attempt = uuid.uuid4()
    await oauth_pairing.register_attempt(
        session, started.id, started.secret, attempt, "n" * 43,
        "http://127.0.0.1:43123/auth/callback",
    )
    pairing = await session.get(OpenAIOAuthPairing, started.id)
    assert pairing is not None
    pairing.expires_at = (
        datetime.now(timezone.utc) + oauth_pairing.COMPLETION_LEASE - timedelta(seconds=2)
    )
    await session.commit()

    with pytest.raises(oauth_pairing.PairingUnavailable):
        await oauth_pairing.claim_completion(
            session,
            started.id,
            started.secret,
            attempt,
            oauth_pairing.completion_digest(attempt, "issued", {"id_token": "id"}),
        )


def test_pairable_credential_predicate_blocks_every_unsafe_resource_state():
    credential = OpenAIOAuthCredential(
        connection_id=uuid.uuid4(), issuer="https://auth.openai.com", subject="subject",
        issued_client_id="issued", scopes=["openid"], status="revoked",
        token_bundle_encrypted=None,
        access_token_expires_at=datetime.now(timezone.utc), credential_version=1,
    )
    assert oauth_pairing._credential_is_pairable(credential)

    for field, value in (
        ("status", "active"),
        ("token_bundle_encrypted", "old-refresh-token-bundle"),
        ("last_error_code", "refresh_outcome_unknown"),
        ("last_error_code", "revocation_pending"),
        ("last_error_code", "resource_cleanup_pending"),
        ("lifecycle_lease_token", "lease"),
        ("refresh_attempt_id", "attempt"),
    ):
        credential.status = "revoked"
        credential.token_bundle_encrypted = None
        credential.last_error_code = None
        credential.lifecycle_lease_token = None
        credential.lifecycle_lease_kind = None
        credential.lifecycle_lease_expires_at = None
        credential.refresh_attempt_id = None
        credential.refresh_attempt_started_at = None
        credential.refresh_restore_status = None
        setattr(credential, field, value)
        assert not oauth_pairing._credential_is_pairable(credential), field


@pytest.mark.asyncio
async def test_user_start_endpoint_returns_secret_once_and_no_store(
    client, auth_headers, session, clean_db, test_user
):
    from app.core.rate_limit import pairing_start_rate_limit
    from app.main import app

    connection = LlmConnection(
        user_id=test_user.id, name="ChatGPT", kind="openai_chatgpt", extra={}
    )
    session.add(connection)
    await session.commit()
    app.dependency_overrides[pairing_start_rate_limit] = lambda: None
    try:
        response = await client.post(
            f"/api/agents/connections/{connection.id}/chatgpt/pairings",
            headers=auth_headers,
        )
    finally:
        app.dependency_overrides.pop(pairing_start_rate_limit, None)
    assert response.status_code == 201
    assert response.headers["cache-control"] == "no-store"
    secret = response.json()["secret"]
    row = await session.scalar(select(OpenAIOAuthPairing))
    assert row is not None and secret not in row.secret_digest
    status_response = await client.get(
        f"/api/agents/connections/{connection.id}/chatgpt/pairings/{row.id}",
        headers=auth_headers,
    )
    assert status_response.status_code == 200
    assert "secret" not in status_response.json()
