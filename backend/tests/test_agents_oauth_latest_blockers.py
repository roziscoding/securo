from __future__ import annotations

import inspect
import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from app.agents.config import AgentSettings
from app.agents.services import connection_service, oauth_refresh, openai_oauth
from app.agents.services.oauth_credentials import decrypt_bundle
from app.agents.services.oauth_credentials import (
    CredentialBundle,
    StaleCredentialVersionError,
    create_credential,
    get_owned_credential,
    rotate_credential,
)
from app.agents.services.openai_oauth import OAuthFailureCode, OAuthLifecycleError
from app.models.user import User
from billiard.exceptions import SoftTimeLimitExceeded


@pytest.fixture(autouse=True)
def _stub_refresh_jwks_prefetch(monkeypatch):
    async def prefetch():
        return object()

    monkeypatch.setattr(oauth_refresh, "fetch_validated_jwks", prefetch)


def test_public_oauth_interfaces_do_not_accept_http_clients_or_transports():
    public_functions = (
        openai_oauth.exchange_authorization_code,
        openai_oauth.refresh_token_bundle,
        openai_oauth.revoke_token,
        openai_oauth.list_models,
        oauth_refresh.refresh_credential,
        oauth_refresh.revoke_credential,
        connection_service.test_connection,
    )
    for function in public_functions:
        signature = inspect.signature(function)
        for parameter in signature.parameters.values():
            assert parameter.name not in {"client", "http_client", "transport"}
            assert parameter.annotation is not httpx.AsyncClient


@pytest.mark.parametrize(
    "value",
    [
        " https://proxy.example",
        "https://proxy.example ",
        "https://proxy.example\n",
        "https://proxy.example\x00",
    ],
)
def test_proxy_rejects_noncanonical_whitespace_and_controls(value):
    with pytest.raises(ValidationError):
        AgentSettings(siwc_oauth_proxy_url=value)


def test_ca_bundle_validation_accepts_regular_readable_pem(tmp_path: Path):
    path = tmp_path / "ca.pem"
    path.write_text("-----BEGIN CERTIFICATE-----\nAA==\n-----END CERTIFICATE-----\n")
    assert AgentSettings(siwc_oauth_ca_bundle=str(path)).siwc_oauth_ca_bundle == str(path)


@pytest.mark.parametrize("kind", ["relative", "missing", "directory", "nul", "control"])
def test_ca_bundle_validation_rejects_invalid_paths(tmp_path: Path, kind: str):
    values = {
        "relative": "ca.pem",
        "missing": str(tmp_path / "missing.pem"),
        "directory": str(tmp_path),
        "nul": str(tmp_path / "ca.pem") + "\x00",
        "control": str(tmp_path / "ca.pem") + "\n",
    }
    with pytest.raises(ValidationError):
        AgentSettings(siwc_oauth_ca_bundle=values[kind])


def test_ca_bundle_validation_rejects_symlink_and_unreadable_file(tmp_path: Path):
    target = tmp_path / "target.pem"
    target.write_text("-----BEGIN CERTIFICATE-----\nAA==\n-----END CERTIFICATE-----\n")
    link = tmp_path / "link.pem"
    link.symlink_to(target)
    with pytest.raises(ValidationError):
        AgentSettings(siwc_oauth_ca_bundle=str(link))

    target.chmod(0)
    try:
        with pytest.raises(ValidationError):
            AgentSettings(siwc_oauth_ca_bundle=str(target))
    finally:
        target.chmod(0o600)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_all_strict_json_paths_reject_nonfinite_constants(constant: str):
    with pytest.raises(ValueError):
        openai_oauth._strict_json_bytes(f'{{"unknown":{constant}}}'.encode())
    response = httpx.Response(200, content=f'{{"keys":[],"unknown":{constant}}}'.encode())
    with pytest.raises(OAuthLifecycleError) as exc_info:
        openai_oauth._json_object(response)
    assert exc_info.value.code is OAuthFailureCode.MALFORMED_RESPONSE


def test_encrypted_bundle_parser_rejects_nonfinite_unknown_field(monkeypatch):
    monkeypatch.setattr(
        "app.agents.services.oauth_credentials.decrypt",
        lambda _: '{"access_token":"a","refresh_token":"r","id_token":"i","unknown":NaN}',
    )
    with pytest.raises(ValueError):
        decrypt_bundle("ciphertext")


async def _postgres_credential(postgres_sessions, label: str):
    user_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    async with postgres_sessions() as session:
        session.add(
            User(
                id=user_id,
                email=f"{label}-{user_id}@example.com",
                hashed_password="unused",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        await session.commit()
        connection = await connection_service.create_connection(
            session, user_id, name=label, kind="openai_chatgpt"
        )
        await create_credential(
            session,
            connection.id,
            user_id,
            bundle=CredentialBundle("old-access", "single-use-refresh", "old-id"),
            issuer="https://auth.openai.com",
            subject="subject-1",
            issued_client_id="oaiapp_test",
            account_email="verified@example.com",
            scopes=["openid", "chatgpt.tokens.use.direct"],
            access_token_expires_at=now,
            refresh_token_expires_at=now + timedelta(days=30),
        )
        return connection.id, user_id, now


@pytest.mark.asyncio
async def test_postgres_refresh_intent_is_durable_before_provider_io_and_cancellation_is_unknown(
    postgres_sessions, monkeypatch
):
    connection_id, user_id, now = await _postgres_credential(
        postgres_sessions, "durable-refresh-intent"
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    limit = asyncio.CancelledError()

    async def cancelled_provider(**kwargs):
        entered.set()
        await release.wait()
        raise limit

    monkeypatch.setattr(oauth_refresh, "refresh_token_bundle", cancelled_provider)
    async with postgres_sessions() as worker, postgres_sessions() as observer:
        task = asyncio.create_task(
            oauth_refresh.refresh_credential(worker, connection_id, user_id, now=now, force=True)
        )
        await entered.wait()
        intent = await get_owned_credential(observer, connection_id, user_id)
        assert intent is not None
        assert intent.status == "reauth_required"
        assert intent.last_error_code == "refresh_outcome_unknown"
        assert intent.refresh_attempt_id
        assert intent.refresh_attempt_started_at is not None
        assert decrypt_bundle(intent.token_bundle_encrypted).refresh_token == "single-use-refresh"
        assert intent.subject == "subject-1"
        assert intent.account_email == "verified@example.com"

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await observer.rollback()
        observer.expire_all()
        current = await get_owned_credential(observer, connection_id, user_id)
        assert current is not None
        assert current.status == "reauth_required"
        assert current.last_error_code == "refresh_outcome_unknown"
        assert current.token_bundle_encrypted is not None


@pytest.mark.asyncio
async def test_postgres_expired_unknown_attempt_never_reuses_old_refresh_token(
    postgres_sessions, monkeypatch
):
    connection_id, user_id, now = await _postgres_credential(
        postgres_sessions, "unknown-takeover"
    )
    calls = 0

    async def provider(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("unknown refresh token must never be dispatched")

    monkeypatch.setattr(oauth_refresh, "refresh_token_bundle", provider)
    async with postgres_sessions() as session:
        credential = await get_owned_credential(session, connection_id, user_id)
        assert credential is not None
        credential.status = "reauth_required"
        credential.last_error_code = "refresh_outcome_unknown"
        credential.refresh_attempt_id = str(uuid.uuid4())
        credential.refresh_attempt_started_at = now - timedelta(minutes=10)
        credential.lifecycle_lease_token = str(uuid.uuid4())
        credential.lifecycle_lease_kind = "refresh"
        credential.lifecycle_lease_expires_at = now - timedelta(minutes=1)
        await session.commit()

        with pytest.raises(StaleCredentialVersionError):
            await rotate_credential(
                session,
                connection_id,
                user_id,
                expected_version=credential.credential_version,
                expected_lease_token=credential.lifecycle_lease_token,
                expected_attempt_id=credential.refresh_attempt_id,
                bundle=CredentialBundle("stale-access", "stale-refresh", "stale-id"),
                access_token_expires_at=now + timedelta(hours=1),
            )

        with pytest.raises(OAuthLifecycleError) as exc_info:
            await oauth_refresh.refresh_credential(
                session, connection_id, user_id, now=now, force=True
            )
        assert exc_info.value.code is OAuthFailureCode.REFRESH_OUTCOME_UNKNOWN
        assert exc_info.value.retryable is False
        assert calls == 0


@pytest.mark.asyncio
async def test_postgres_revocation_intent_is_durable_before_provider_io(
    postgres_sessions, monkeypatch
):
    connection_id, user_id, _ = await _postgres_credential(
        postgres_sessions, "durable-revocation-intent"
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def provider(**kwargs):
        entered.set()
        await release.wait()

    monkeypatch.setattr(oauth_refresh, "revoke_token", provider)
    async with postgres_sessions() as worker, postgres_sessions() as observer:
        task = asyncio.create_task(oauth_refresh.revoke_credential(worker, connection_id, user_id))
        await entered.wait()
        pending = await get_owned_credential(observer, connection_id, user_id)
        assert pending is not None
        assert pending.status == "reauth_required"
        assert pending.last_error_code == "revocation_pending"
        assert pending.token_bundle_encrypted is not None
        release.set()
        await task


@pytest.mark.asyncio
async def test_postgres_revoke_soft_limit_preserves_same_exception_when_db_cleanup_fails(
    postgres_sessions, monkeypatch
):
    connection_id, user_id, _ = await _postgres_credential(
        postgres_sessions, "revoke-cleanup-failure"
    )
    limit = SoftTimeLimitExceeded("sentinel")

    async def stop(**kwargs):
        raise limit

    async def fail_cleanup(*args, **kwargs):
        raise RuntimeError("database cleanup failed")

    monkeypatch.setattr(oauth_refresh, "revoke_token", stop)
    monkeypatch.setattr(oauth_refresh, "scrub_credential_bundle", fail_cleanup)
    async with postgres_sessions() as worker, postgres_sessions() as observer:
        with pytest.raises(SoftTimeLimitExceeded) as exc_info:
            await oauth_refresh.revoke_credential(worker, connection_id, user_id)
        assert exc_info.value is limit
        pending = await get_owned_credential(observer, connection_id, user_id)
        assert pending is not None
        assert pending.status == "reauth_required"
        assert pending.last_error_code == "revocation_pending"
        assert pending.token_bundle_encrypted is not None


@pytest.mark.asyncio
async def test_request_soft_limit_wins_when_owned_transport_close_also_fails(monkeypatch):
    limit = SoftTimeLimitExceeded("soft-limit")

    class FailingCloseTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            raise limit

        async def aclose(self):
            raise RuntimeError("close failed")

    monkeypatch.setattr(openai_oauth, "_new_oauth_transport", FailingCloseTransport)
    with pytest.raises(SoftTimeLimitExceeded) as exc_info:
        await openai_oauth._request("GET", openai_oauth.JWKS_ENDPOINT)
    assert exc_info.value is limit
