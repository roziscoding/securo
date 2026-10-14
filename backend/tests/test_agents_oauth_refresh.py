"""Database-backed refresh, revocation, and recovery tests."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import uuid
import gc
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from billiard.exceptions import SoftTimeLimitExceeded
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt
from sqlalchemy import func, update

from app.agents.models.oauth_credential import OpenAIOAuthCredential
from app.agents.services import connection_service, oauth_refresh, openai_oauth
from app.agents.services.oauth_credentials import (
    CredentialBundle,
    StaleCredentialVersionError,
    begin_refresh_attempt,
    claim_credential_lease,
    decrypt_bundle,
    get_owned_credential,
    rotate_credential,
)
from app.agents.services.oauth_refresh import (
    disable_plan_usage,
    refresh_credential,
    require_reauthorization,
    revoke_credential,
)
from app.agents.services.openai_oauth import (
    ISSUER,
    REQUIRED_SCOPES,
    RESOURCE,
    OAuthFailureCode,
    OAuthLifecycleError,
    ValidatedTokenBundle,
)
from app.models.user import User


def _use_owned_test_transport(monkeypatch, client: httpx.AsyncClient) -> None:
    transport = client._transport  # type: ignore[attr-defined]
    assert type(transport) is httpx.MockTransport
    monkeypatch.setattr(openai_oauth, "_new_oauth_transport", lambda: transport)


def _b64uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _refresh_success_material(now: datetime):
    key: Any = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = key.public_key().public_numbers()
    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "kid": "refresh-test-key",
                "use": "sig",
                "alg": "RS256",
                "n": _b64uint(public.n),
                "e": _b64uint(public.e),
            }
        ]
    }
    issued_at = int(now.timestamp())
    expires_at = int((now + timedelta(minutes=10)).timestamp())
    scopes = " ".join(sorted(REQUIRED_SCOPES))
    access_token = jwt.encode(
        {
            "iss": ISSUER,
            "aud": RESOURCE,
            "sub": "subject-1",
            "client_id": "oaiapp_test",
            "scope": scopes,
            "nbf": issued_at,
            "jti": "refresh-access-id",
            "https://api.openai.com/auth": {
                "per_user_salt": "salt",
                "encrypted_auth_metadata": "metadata",
            },
            "iat": issued_at,
            "exp": expires_at,
        },
        key,
        algorithm="RS256",
        headers={"kid": "refresh-test-key"},
    )
    digest = hashlib.sha256(access_token.encode()).digest()
    at_hash = base64.urlsafe_b64encode(digest[: len(digest) // 2]).rstrip(b"=").decode()
    id_token = jwt.encode(
        {
            "iss": ISSUER,
            "aud": "oaiapp_test",
            "sub": "subject-1",
            "iat": issued_at,
            "exp": expires_at,
            "at_hash": at_hash,
        },
        key,
        algorithm="RS256",
        headers={"kid": "refresh-test-key"},
    )
    return jwks, {
        "access_token": access_token,
        "refresh_token": "rotated-refresh",
        "id_token": id_token,
        "token_type": "Bearer",
        "expires_in": 600,
        "scope": scopes,
        "earliest_refresh_at": issued_at,
    }


@pytest.fixture(autouse=True)
def _stub_refresh_jwks_prefetch(monkeypatch):
    snapshot = object()

    async def prefetch():
        return snapshot

    monkeypatch.setattr(oauth_refresh, "fetch_validated_jwks", prefetch)


async def _credential(
    session,
    test_user,
    *,
    expires_in=5,
    earliest_delta=-5,
    refresh_expires_delta=timedelta(days=30),
):
    from app.agents.services.oauth_credentials import create_credential

    connection = await connection_service.create_connection(
        session, test_user.id, name="ChatGPT", kind="openai_chatgpt"
    )
    now = datetime.now(timezone.utc)
    credential = await create_credential(
        session,
        connection.id,
        test_user.id,
        bundle=CredentialBundle("old-access", "old-refresh", "old-id"),
        issuer="https://auth.openai.com",
        subject="subject-1",
        issued_client_id="oaiapp_test",
        account_email="owner@example.com",
        scopes=[
            "openid",
            "profile",
            "email",
            "offline_access",
            "resource.invoke",
            "chatgpt.tokens.use.direct",
        ],
        access_token_expires_at=now + timedelta(seconds=expires_in),
        earliest_refresh_at=now + timedelta(seconds=earliest_delta),
        refresh_token_expires_at=now + refresh_expires_delta,
    )
    return connection, credential, now


@pytest.mark.asyncio
async def test_refresh_obeys_earliest_refresh_at(session, test_user):
    connection, original, now = await _credential(
        session, test_user, expires_in=5, earliest_delta=60
    )
    connection_id, user_id = connection.id, test_user.id
    calls = 0

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as _client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await refresh_credential(
                session, connection_id, user_id, now=now, force=True
            )
    assert exc_info.value.code is OAuthFailureCode.TOO_EARLY
    assert calls == 0
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.credential_version == original.credential_version
    assert decrypt_bundle(current.token_bundle_encrypted) == CredentialBundle(
        "old-access", "old-refresh", "old-id"
    )


@pytest.mark.asyncio
async def test_refresh_expired_refresh_token_scrubs_at_boundary_before_lease_or_network(
    session, test_user, monkeypatch
):
    connection, original, now = await _credential(
        session, test_user, refresh_expires_delta=timedelta(0)
    )
    connection_id = connection.id
    user_id = test_user.id
    original_version = original.credential_version
    lease_calls = 0
    provider_calls = 0

    async def unexpected_lease(*args, **kwargs):
        nonlocal lease_calls
        lease_calls += 1
        return True

    async def unexpected_refresh(*args, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        raise AssertionError("provider must not be called")

    monkeypatch.setattr("app.agents.services.oauth_refresh.begin_refresh_attempt", unexpected_lease)
    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_token_bundle", unexpected_refresh)

    async with httpx.AsyncClient() as _client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await refresh_credential(
                session, connection_id, user_id, now=now, force=True
            )

    assert exc_info.value.code is OAuthFailureCode.REAUTH_REQUIRED
    assert exc_info.value.definitive is True
    assert lease_calls == 0
    assert provider_calls == 0
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.credential_version == original_version + 1
    assert current.status == "reauth_required"
    assert current.last_error_code == "expired"
    assert current.token_bundle_encrypted is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_kind", "expected_code"),
    [
        ("transient", OAuthFailureCode.TRANSIENT),
        ("malformed", OAuthFailureCode.MALFORMED_RESPONSE),
        ("cancelled", None),
        ("soft-limit", None),
    ],
)
async def test_refresh_jwks_prerequisite_failure_precedes_decryption_and_durable_intent(
    session, test_user, monkeypatch, failure_kind, expected_code
):
    connection, original, now = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id
    original_version = original.credential_version
    original_bundle = decrypt_bundle(original.token_bundle_encrypted)
    decrypt_calls = 0
    intent_calls = 0
    requests: list[tuple[str, httpx.URL]] = []
    real_decrypt = oauth_refresh.decrypt_bundle
    real_begin = oauth_refresh.begin_refresh_attempt
    failure: BaseException | None = None
    if failure_kind == "cancelled":
        failure = asyncio.CancelledError()
    elif failure_kind == "soft-limit":
        failure = SoftTimeLimitExceeded("JWKS lookup exceeded task limit")

    def tracked_decrypt(*args, **kwargs):
        nonlocal decrypt_calls
        decrypt_calls += 1
        return real_decrypt(*args, **kwargs)

    async def tracked_begin(*args, **kwargs):
        nonlocal intent_calls
        intent_calls += 1
        return await real_begin(*args, **kwargs)

    def handler(request: httpx.Request):
        requests.append((request.method, request.url))
        assert request.url == openai_oauth.JWKS_ENDPOINT
        if failure_kind == "transient":
            return httpx.Response(503, json={"error": "temporarily_unavailable"})
        if failure_kind == "malformed":
            return httpx.Response(200, json={"keys": []})
        assert failure is not None
        raise failure

    monkeypatch.setattr(oauth_refresh, "decrypt_bundle", tracked_decrypt)
    monkeypatch.setattr(oauth_refresh, "begin_refresh_attempt", tracked_begin)
    monkeypatch.setattr(oauth_refresh, "fetch_validated_jwks", openai_oauth.fetch_validated_jwks)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        _use_owned_test_transport(monkeypatch, client)
        if isinstance(failure, asyncio.CancelledError):
            expectation = pytest.raises(asyncio.CancelledError)
        elif isinstance(failure, SoftTimeLimitExceeded):
            expectation = pytest.raises(SoftTimeLimitExceeded)
        else:
            expectation = pytest.raises(OAuthLifecycleError)
        with expectation as exc_info:
            await refresh_credential(session, connection_id, user_id, now=now, force=True)

    if expected_code is not None:
        assert isinstance(exc_info.value, OAuthLifecycleError)
        assert exc_info.value.code is expected_code
    if isinstance(failure, SoftTimeLimitExceeded):
        assert exc_info.value is failure
    assert decrypt_calls == 0
    assert intent_calls == 0
    assert requests == [("GET", httpx.URL(openai_oauth.JWKS_ENDPOINT))]

    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.status == "active"
    assert current.last_error_code is None
    assert current.refresh_attempt_id is None
    assert current.refresh_attempt_started_at is None
    assert current.lifecycle_lease_token is None
    assert current.lifecycle_lease_kind is None
    assert current.lifecycle_lease_expires_at is None
    assert current.credential_version == original_version
    assert decrypt_bundle(current.token_bundle_encrypted) == original_bundle


@pytest.mark.asyncio
async def test_refresh_atomically_persists_rotated_bundle(session, test_user, monkeypatch):
    connection, _, now = await _credential(session, test_user)
    connection_id, user_id = connection.id, test_user.id
    rotated = ValidatedTokenBundle(
        bundle=CredentialBundle("new-access", "new-refresh", "new-id"),
        subject="subject-1",
        scopes=frozenset(
            [
                "openid",
                "profile",
                "email",
                "offline_access",
                "resource.invoke",
                "chatgpt.tokens.use.direct",
            ]
        ),
        access_token_expires_at=now + timedelta(hours=1),
        earliest_refresh_at=now + timedelta(minutes=30),
        refresh_token_expires_at=now + timedelta(days=30),
        received_at=now,
    )
    seen: list[tuple[str, str, str]] = []
    events: list[str] = []
    prefetched = object()
    real_begin = oauth_refresh.begin_refresh_attempt

    async def prefetch():
        events.append("jwks_get")
        return prefetched

    async def recording_begin(*args, **kwargs):
        claimed = await real_begin(*args, **kwargs)
        events.append("intent_committed")
        return claimed

    async def fake_refresh(
        *, refresh_token, issued_client_id, expected_subject, jwks, **kwargs
    ):
        assert not session.in_transaction()
        assert jwks is prefetched
        events.append("token_post")
        seen.append((refresh_token, issued_client_id, expected_subject))
        return rotated

    monkeypatch.setattr(oauth_refresh, "fetch_validated_jwks", prefetch)
    monkeypatch.setattr(oauth_refresh, "begin_refresh_attempt", recording_begin)
    monkeypatch.setattr(oauth_refresh, "refresh_token_bundle", fake_refresh)
    async with httpx.AsyncClient() as _client:
        token = await refresh_credential(
            session, connection_id, user_id, now=now, force=True
        )
    assert token == "new-access"
    assert events == ["jwks_get", "intent_committed", "token_post"]
    assert seen == [("old-refresh", "oaiapp_test", "subject-1")]
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.credential_version == 2
    assert decrypt_bundle(current.token_bundle_encrypted) == rotated.bundle
    assert current.last_refreshed_at is not None
    assert current.last_refreshed_at.replace(tzinfo=timezone.utc) == now


@pytest.mark.asyncio
async def test_refresh_gets_jwks_then_commits_intent_then_posts_and_atomically_finalizes(
    session, test_user, monkeypatch
):
    connection, original, now = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id
    original_version = original.credential_version
    jwks, token_payload = _refresh_success_material(now)
    events: list[str] = []
    requests: list[tuple[str, httpx.URL]] = []
    real_begin = oauth_refresh.begin_refresh_attempt

    async def recording_begin(*args, **kwargs):
        claimed = await real_begin(*args, **kwargs)
        events.append("intent_committed")
        return claimed

    def handler(request: httpx.Request):
        requests.append((request.method, request.url))
        if request.url == openai_oauth.JWKS_ENDPOINT:
            events.append("jwks_get")
            return httpx.Response(200, json=jwks)
        assert request.url == openai_oauth.TOKEN_ENDPOINT
        events.append("token_post")
        return httpx.Response(200, json=token_payload)

    monkeypatch.setattr(oauth_refresh, "fetch_validated_jwks", openai_oauth.fetch_validated_jwks)
    monkeypatch.setattr(oauth_refresh, "begin_refresh_attempt", recording_begin)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        _use_owned_test_transport(monkeypatch, client)
        access_token = await refresh_credential(
            session,
            connection_id,
            user_id,
            now=now,
            force=True,
            receipt_clock=lambda: now,
        )

    assert access_token == token_payload["access_token"]
    assert events == ["jwks_get", "intent_committed", "token_post"]
    assert requests == [
        ("GET", httpx.URL(openai_oauth.JWKS_ENDPOINT)),
        ("POST", httpx.URL(openai_oauth.TOKEN_ENDPOINT)),
    ]
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.credential_version == original_version + 1
    assert current.status == "active"
    assert current.last_error_code is None
    assert current.refresh_attempt_id is None
    assert current.lifecycle_lease_token is None
    assert decrypt_bundle(current.token_bundle_encrypted).refresh_token == "rotated-refresh"


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe_email", ["x" * 321, "not-an-email"])
async def test_refresh_omits_unsafe_optional_email_without_losing_rotated_bundle(
    session, test_user, monkeypatch, unsafe_email
):
    connection, _, now = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id
    rotated = ValidatedTokenBundle(
        bundle=CredentialBundle("new-access", "new-refresh", "new-id"),
        subject="subject-1",
        account_email=unsafe_email,
        scopes=frozenset({"chatgpt.tokens.use.direct"}),
        access_token_expires_at=now + timedelta(hours=1),
        earliest_refresh_at=None,
        refresh_token_expires_at=now + timedelta(days=30),
    )

    async def fake_refresh(*args, **kwargs):
        return rotated

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_token_bundle", fake_refresh)
    async with httpx.AsyncClient() as _client:
        assert (
            await refresh_credential(
                session, connection_id, user_id, now=now, force=True
            )
            == "new-access"
        )

    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.account_email is None
    assert decrypt_bundle(current.token_bundle_encrypted) == rotated.bundle
    assert current.lifecycle_lease_token is None


@pytest.mark.asyncio
async def test_refresh_scope_loss_persists_rotated_sign_in_disabled_and_stops_inference(
    session, test_user, monkeypatch
):
    connection, _, now = await _credential(session, test_user)
    connection_id, user_id = connection.id, test_user.id
    rotated = ValidatedTokenBundle(
        bundle=CredentialBundle("new-access", "new-refresh", "new-id"),
        subject="subject-1",
        account_email="rotated@example.com",
        scopes=frozenset({"openid", "profile", "email", "offline_access", "resource.invoke"}),
        access_token_expires_at=now + timedelta(hours=1),
        earliest_refresh_at=now + timedelta(minutes=30),
        refresh_token_expires_at=now + timedelta(days=30),
    )
    provider_calls = 0

    async def fake_refresh(*args, **kwargs):
        nonlocal provider_calls
        provider_calls += 1
        return rotated

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_token_bundle", fake_refresh)
    async with httpx.AsyncClient() as _client:
        with pytest.raises(OAuthLifecycleError) as lost:
            await refresh_credential(
                session, connection_id, user_id, now=now, force=True
            )
        with pytest.raises(OAuthLifecycleError) as repeated:
            await refresh_credential(
                session, connection_id, user_id, now=now, force=True
            )

    assert lost.value.code is OAuthFailureCode.SCOPE_LOSS
    assert lost.value.definitive is True
    assert lost.value.retryable is False
    assert repeated.value.code is OAuthFailureCode.SCOPE_LOSS
    assert provider_calls == 1
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.status == "disabled"
    assert current.last_error_code == "scope_loss"
    assert current.subject == "subject-1"
    assert current.issued_client_id == "oaiapp_test"
    assert current.account_email == "rotated@example.com"
    assert set(current.scopes) == set(rotated.scopes)
    assert decrypt_bundle(current.token_bundle_encrypted) == rotated.bundle
    assert current.lifecycle_lease_token is None
    assert current.lifecycle_lease_kind is None
    assert current.lifecycle_lease_expires_at is None

    await session.refresh(connection)
    result = await connection_service.test_connection(connection, session=session, now=now)
    assert result["ok"] is False
    assert result["auth_state"] == "scope_loss"
    assert "disabled" in result["detail"].lower()


@pytest.mark.asyncio
async def test_transient_refresh_marks_outcome_unknown_and_retains_last_bundle(
    session, test_user, monkeypatch
):
    connection, original, now = await _credential(session, test_user)
    connection_id, user_id = connection.id, test_user.id
    original_version = original.credential_version

    async def fail(*args, **kwargs):
        raise OAuthLifecycleError(
            OAuthFailureCode.TRANSIENT, "temporary refresh failure", retryable=True
        )

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_token_bundle", fail)
    async with httpx.AsyncClient() as _client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await refresh_credential(
                session, connection.id, test_user.id, now=now, force=True
            )
    assert exc_info.value.code is OAuthFailureCode.TRANSIENT
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.status == "reauth_required"
    assert current.last_error_code == "refresh_outcome_unknown"
    assert current.refresh_attempt_id is not None
    assert current.credential_version == original_version
    assert decrypt_bundle(current.token_bundle_encrypted).refresh_token == "old-refresh"


@pytest.mark.asyncio
async def test_invalid_client_after_dispatch_marks_outcome_unknown_and_preserves_bundle(
    session, test_user, monkeypatch
):
    connection, original, now = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id
    original_version = original.credential_version

    async def fail(*args, **kwargs):
        raise OAuthLifecycleError(
            OAuthFailureCode.INVALID_CLIENT,
            "refresh rejected the client",
            definitive=False,
            retryable=False,
        )

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_token_bundle", fail)
    async with httpx.AsyncClient() as _client:
        with pytest.raises(OAuthLifecycleError) as failure:
            await refresh_credential(
                session, connection_id, user_id, now=now, force=True
            )

    assert failure.value.code is OAuthFailureCode.INVALID_CLIENT
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.status == "reauth_required"
    assert current.credential_version == original_version
    assert current.last_error_code == "refresh_outcome_unknown"
    assert decrypt_bundle(current.token_bundle_encrypted) == CredentialBundle(
        "old-access", "old-refresh", "old-id"
    )
    assert current.lifecycle_lease_token is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_code",
    [
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    ],
)
async def test_terminal_refresh_errors_scrub_bundle_and_require_reauth(
    session, test_user, monkeypatch, provider_code
):
    connection, _, now = await _credential(session, test_user)
    connection_id, user_id = connection.id, test_user.id

    async def fail(*args, **kwargs):
        raise OAuthLifecycleError(
            OAuthFailureCode.REAUTH_REQUIRED,
            f"refresh requires reauthorization ({provider_code})",
            definitive=True,
        )

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_token_bundle", fail)
    async with httpx.AsyncClient() as _client:
        with pytest.raises(OAuthLifecycleError):
            await refresh_credential(
                session, connection_id, user_id, now=now, force=True
            )
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.status == "reauth_required"
    assert current.last_error_code == "reauth_required"
    assert current.token_bundle_encrypted is None


@pytest.mark.asyncio
async def test_refresh_race_does_not_overwrite_winner(session, test_user, monkeypatch):
    connection, _, now = await _credential(session, test_user)
    rotated = ValidatedTokenBundle(
        bundle=CredentialBundle("loser-access", "loser-refresh", "loser-id"),
        subject="subject-1",
        scopes=frozenset({"chatgpt.tokens.use.direct"}),
        access_token_expires_at=now + timedelta(hours=1),
        earliest_refresh_at=None,
        refresh_token_expires_at=None,
    )

    async def fake_refresh(*args, **kwargs):
        return rotated

    async def stale(*args, **kwargs):
        from app.agents.services.oauth_credentials import StaleCredentialVersionError

        raise StaleCredentialVersionError("credential version is stale")

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_token_bundle", fake_refresh)
    monkeypatch.setattr("app.agents.services.oauth_refresh.rotate_credential", stale)
    async with httpx.AsyncClient() as _client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await refresh_credential(
                session, connection.id, test_user.id, now=now, force=True
            )
    assert exc_info.value.code is OAuthFailureCode.RACE_LOST


@pytest.mark.asyncio
async def test_revocation_is_idempotent_and_always_scrubs_local_bundle(
    session, test_user, monkeypatch
):
    connection, _, _ = await _credential(session, test_user)
    connection_id, user_id = connection.id, test_user.id
    calls = 0

    async def fake_revoke(*args, **kwargs):
        nonlocal calls
        assert not session.in_transaction()
        calls += 1

    monkeypatch.setattr("app.agents.services.oauth_refresh.revoke_token", fake_revoke)
    async with httpx.AsyncClient() as _client:
        await revoke_credential(session, connection_id, user_id)
        await revoke_credential(session, connection_id, user_id)
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.status == "revoked"
    assert current.token_bundle_encrypted is None
    assert calls == 1


@pytest.mark.asyncio
async def test_revocation_network_failure_scrubs_local_bundle_without_claiming_remote_success(
    session, test_user, monkeypatch
):
    connection, _, _ = await _credential(session, test_user)

    async def fail(*args, **kwargs):
        raise OAuthLifecycleError(
            OAuthFailureCode.TRANSIENT, "revocation unavailable", retryable=True
        )

    monkeypatch.setattr("app.agents.services.oauth_refresh.revoke_token", fail)
    async with httpx.AsyncClient() as _client:
        with pytest.raises(OAuthLifecycleError):
            await revoke_credential(session, connection.id, test_user.id)
    current = await get_owned_credential(session, connection.id, test_user.id)
    assert current is not None
    assert current.status == "revoked"
    assert current.last_error_code == "transient"
    assert current.token_bundle_encrypted is None

    async with httpx.AsyncClient() as _client:
        with pytest.raises(OAuthLifecycleError) as repeated:
            await revoke_credential(session, connection.id, test_user.id)
    assert repeated.value.code is OAuthFailureCode.TRANSIENT
    assert repeated.value.retryable is True


@pytest.mark.asyncio
async def test_revocation_runtime_failure_still_scrubs_and_records_unconfirmed(
    session, test_user, monkeypatch
):
    connection, _, _ = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id

    async def fail(*args, **kwargs):
        raise RuntimeError("unexpected hook failure")

    monkeypatch.setattr("app.agents.services.oauth_refresh.revoke_token", fail)
    async with httpx.AsyncClient() as _client:
        with pytest.raises(RuntimeError, match="^OAuth operation failed safely$"):
            await revoke_credential(session, connection_id, user_id)

    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.status == "revoked"
    assert current.last_error_code == "transient"
    assert current.token_bundle_encrypted is None

    async with httpx.AsyncClient() as _client:
        with pytest.raises(OAuthLifecycleError) as repeated:
            await revoke_credential(session, connection_id, user_id)
    assert repeated.value.code is OAuthFailureCode.TRANSIENT
    assert repeated.value.retryable is True


@pytest.mark.asyncio
async def test_revocation_does_not_execute_inherited_response_hook(
    session, test_user, monkeypatch
):
    connection, _, _ = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id
    retained: list[httpx.Response] = []

    async def explode(response: httpx.Response):
        retained.append(response)
        raise RuntimeError("response hook must not run")

    def handler(request: httpx.Request):
        return httpx.Response(200, content=b"provider-secret")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), event_hooks={"response": [explode]}
    ) as _client:
        _use_owned_test_transport(monkeypatch, _client)
        await revoke_credential(
            session,
            connection_id,
            user_id,
        )

    assert retained == []
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.status == "revoked"
    assert current.last_error_code is None
    assert current.token_bundle_encrypted is None


@pytest.mark.asyncio
async def test_revocation_cancellation_waits_for_local_scrub_before_propagating(
    session, test_user, monkeypatch
):
    connection, _, _ = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id
    entered = asyncio.Event()

    async def wait_forever(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("app.agents.services.oauth_refresh.revoke_token", wait_forever)
    async with httpx.AsyncClient() as _client:
        operation = asyncio.create_task(
            revoke_credential(session, connection_id, user_id)
        )
        await entered.wait()
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation

    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.status == "revoked"
    assert current.last_error_code == "transient"
    assert current.token_bundle_encrypted is None


@pytest.mark.asyncio
async def test_refresh_cancellation_waits_for_rotation_then_propagates_without_lockout(
    session, test_user, monkeypatch
):
    connection, _, now = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def rotate_after_release(*args, **kwargs):
        entered.set()
        await finish.wait()
        return ValidatedTokenBundle(
            bundle=CredentialBundle("new-access", "new-refresh", "new-id"),
            subject="subject-1",
            scopes=frozenset({"chatgpt.tokens.use.direct"}),
            access_token_expires_at=now + timedelta(hours=1),
            earliest_refresh_at=None,
            refresh_token_expires_at=now + timedelta(days=30),
        )

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_token_bundle", rotate_after_release)
    async with httpx.AsyncClient() as _client:
        operation = asyncio.create_task(
            refresh_credential(session, connection_id, user_id, now=now, force=True)
        )
        await entered.wait()
        operation.cancel()
        operation.cancel()
        await asyncio.sleep(0)
        assert not operation.done()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await operation

        assert (
            await refresh_credential(session, connection_id, user_id, now=now)
            == "new-access"
        )

    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert decrypt_bundle(current.token_bundle_encrypted).refresh_token == "new-refresh"
    assert current.lifecycle_lease_token is None


@pytest.mark.asyncio
async def test_provider_cooperative_cancellation_makes_refresh_nonretryable(
    session, test_user, monkeypatch
):
    connection, _, now = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id
    calls = 0

    async def cancel_then_rotate(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError()
        return ValidatedTokenBundle(
            bundle=CredentialBundle("new-access", "new-refresh", "new-id"),
            subject="subject-1",
            scopes=frozenset({"chatgpt.tokens.use.direct"}),
            access_token_expires_at=now + timedelta(hours=1),
            earliest_refresh_at=None,
            refresh_token_expires_at=now + timedelta(days=30),
        )

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_token_bundle", cancel_then_rotate)
    async with httpx.AsyncClient() as _client:
        with pytest.raises(asyncio.CancelledError):
            await refresh_credential(
                session, connection_id, user_id, now=now, force=True
            )
        with pytest.raises(OAuthLifecycleError) as unknown:
            await refresh_credential(
                session, connection_id, user_id, now=now, force=True
            )
        assert unknown.value.code is OAuthFailureCode.REFRESH_OUTCOME_UNKNOWN

    assert calls == 1


@pytest.mark.asyncio
async def test_revocation_base_exception_still_scrubs_before_sanitizing(
    session, test_user, monkeypatch
):
    class ProviderAbort(BaseException):
        pass

    connection, _, _ = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id

    secret = "revocation-base-exception-secret"
    provider_abort = ProviderAbort(secret)

    async def abort(*args, **kwargs):
        raise provider_abort

    monkeypatch.setattr("app.agents.services.oauth_refresh.revoke_token", abort)
    async with httpx.AsyncClient() as _client:
        with pytest.raises(RuntimeError, match="OAuth operation failed safely") as exc_info:
            await revoke_credential(session, connection_id, user_id)

    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.status == "revoked"
    assert current.last_error_code == "transient"
    assert current.token_bundle_encrypted is None
    assert secret not in _service_traceback_text(exc_info.value)
    assert provider_abort.args == ()
    assert provider_abort.__traceback__ is None


@pytest.mark.asyncio
async def test_connection_probe_discovers_official_model_list_without_redirects(
    session, test_user, monkeypatch
):
    connection, _, now = await _credential(session, test_user, expires_in=3600)

    def handler(request: httpx.Request):
        assert request.url == "https://api.openai.com/v1/models"
        assert request.headers["authorization"] == "Bearer old-access"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "gpt-listed",
                        "display_name": "GPT Listed",
                        "visibility": "list",
                    },
                    {
                        "slug": "hidden",
                        "display_name": "Hidden",
                        "visibility": "hidden",
                    },
                    {
                        "slug": "gpt-second",
                        "display_name": "GPT Second",
                        "visibility": "list",
                    },
                ],
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as _client:
        _use_owned_test_transport(monkeypatch, _client)
        result = await connection_service.test_connection(
            connection,
            session=session,
            now=now,
        )
    assert result == {
        "ok": True,
        "detail": "reachable (2 models)",
        "models": [
            {"slug": "gpt-listed", "display_name": "GPT Listed"},
            {"slug": "gpt-second", "display_name": "GPT Second"},
        ],
    }


@pytest.mark.asyncio
async def test_connection_probe_refreshes_once_after_unauthorized(session, test_user, monkeypatch):
    connection, _, now = await _credential(session, test_user, expires_in=3600)
    forces: list[bool] = []

    async def fake_refresh(*args, force=False, **kwargs):
        forces.append(force)
        return "fresh-access" if force else "old-access"

    responses = [
        httpx.Response(401),
        httpx.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "gpt-listed",
                        "display_name": "GPT Listed",
                        "visibility": "list",
                    }
                ]
            },
        ),
    ]
    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_credential", fake_refresh)

    def handler(request: httpx.Request):
        return responses.pop(0)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as _client:
        _use_owned_test_transport(monkeypatch, _client)
        result = await connection_service.test_connection(
            connection,
            session=session,
            now=now,
        )
    assert result["ok"] is True
    assert forces == [False, True]


@pytest.mark.asyncio
async def test_connection_probe_forbidden_does_not_refresh(session, test_user, monkeypatch):
    connection, _, now = await _credential(session, test_user, expires_in=3600)
    forces: list[bool] = []

    async def fake_refresh(*args, force=False, **kwargs):
        forces.append(force)
        return "old-access"

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_credential", fake_refresh)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(403))
    ) as _client:
        _use_owned_test_transport(monkeypatch, _client)
        result = await connection_service.test_connection(
            connection,
            session=session,
            now=now,
        )
    assert result["auth_state"] == "scope_loss"
    assert forces == [False]
    current = await get_owned_credential(session, connection.id, test_user.id)
    assert current is not None
    assert current.status == "disabled"
    assert current.last_error_code == "scope_loss"
    assert decrypt_bundle(current.token_bundle_encrypted).access_token == "old-access"


@pytest.mark.asyncio
async def test_connection_probe_second_unauthorized_scrubs_and_requires_reauth(
    session, test_user, monkeypatch
):
    connection, _, now = await _credential(session, test_user, expires_in=3600)

    async def fake_refresh(*args, force=False, **kwargs):
        return "fresh-access" if force else "old-access"

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_credential", fake_refresh)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(401))
    ) as _client:
        _use_owned_test_transport(monkeypatch, _client)
        result = await connection_service.test_connection(
            connection,
            session=session,
            now=now,
        )
    assert result["auth_state"] == "reauth_required"
    current = await get_owned_credential(session, connection.id, test_user.id)
    assert current is not None
    assert current.status == "reauth_required"
    assert current.last_error_code == "reauth_required"
    assert current.token_bundle_encrypted is None


@pytest.mark.asyncio
async def test_connection_probe_unauthorized_then_forbidden_scrubs_scope_loss(
    session, test_user, monkeypatch
):
    connection, _, now = await _credential(session, test_user, expires_in=3600)

    async def fake_refresh(*args, force=False, **kwargs):
        return "fresh-access" if force else "old-access"

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_credential", fake_refresh)
    responses = [httpx.Response(401), httpx.Response(403)]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: responses.pop(0))
    ) as _client:
        _use_owned_test_transport(monkeypatch, _client)
        result = await connection_service.test_connection(
            connection,
            session=session,
            now=now,
        )

    assert result["auth_state"] == "scope_loss"
    current = await get_owned_credential(session, connection.id, test_user.id)
    assert current is not None
    assert current.status == "disabled"
    assert current.last_error_code == "scope_loss"
    assert decrypt_bundle(current.token_bundle_encrypted).access_token == "old-access"


@pytest.mark.asyncio
async def test_stale_resource_401_reuses_rotation_winner_before_forced_refresh(
    session, test_user, monkeypatch
):
    connection, credential, now = await _credential(session, test_user, expires_in=3600)
    connection_id = connection.id
    user_id = test_user.id
    stale_version = credential.credential_version
    provider_refreshes = 0
    model_calls = 0

    async def forbidden_refresh(**kwargs):
        nonlocal provider_refreshes
        provider_refreshes += 1
        raise AssertionError("a stale 401 must not force-refresh the winner")

    async def models(access_token):
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            assert access_token == "old-access"
            await rotate_credential(
                session,
                connection_id,
                user_id,
                expected_version=stale_version,
                bundle=CredentialBundle("winner-access", "winner-refresh", "winner-id"),
                access_token_expires_at=now + timedelta(hours=1),
                refresh_token_expires_at=now + timedelta(days=30),
            )
            raise OAuthLifecycleError(
                OAuthFailureCode.REAUTH_REQUIRED,
                "stale access token rejected",
                definitive=True,
            )
        assert access_token == "winner-access"
        return []

    monkeypatch.setattr(oauth_refresh, "refresh_token_bundle", forbidden_refresh)
    monkeypatch.setattr(openai_oauth, "list_models", models)
    result = await connection_service.test_connection(connection, session=session, now=now)

    assert result["ok"] is True
    assert model_calls == 2
    assert provider_refreshes == 0


@pytest.mark.asyncio
async def test_connection_snapshot_rejects_unknown_attempt_before_provider_call(
    session, test_user, monkeypatch
):
    connection, credential, now = await _credential(session, test_user, expires_in=3600)
    provider_calls = 0

    async def becomes_unknown(*args, **kwargs):
        assert await begin_refresh_attempt(
            session,
            connection.id,
            test_user.id,
            expected_version=credential.credential_version,
            attempt_id=str(uuid.uuid4()),
            lease_duration=timedelta(minutes=5),
        )
        return "old-access"

    async def forbidden_models(access_token):
        nonlocal provider_calls
        provider_calls += 1
        return []

    monkeypatch.setattr(oauth_refresh, "refresh_credential", becomes_unknown)
    monkeypatch.setattr(openai_oauth, "list_models", forbidden_models)
    result = await connection_service.test_connection(connection, session=session, now=now)

    assert result["auth_state"] == "refresh_outcome_unknown"
    assert provider_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", [disable_plan_usage, require_reauthorization])
async def test_stale_provider_terminal_response_cannot_mutate_rotation_winner(
    session, test_user, helper
):
    connection, credential, now = await _credential(session, test_user, expires_in=0)
    connection_id = connection.id
    user_id = test_user.id
    stale_version = credential.credential_version
    winner = await rotate_credential(
        session,
        connection_id,
        user_id,
        expected_version=stale_version,
        bundle=CredentialBundle("winner-access", "winner-refresh", "winner-id"),
        access_token_expires_at=now + timedelta(hours=1),
        refresh_token_expires_at=now + timedelta(days=30),
    )
    winner_version = winner.credential_version

    with pytest.raises(OAuthLifecycleError) as exc_info:
        await helper(
            session,
            connection_id,
            user_id,
            expected_version=stale_version,
        )

    assert exc_info.value.code is OAuthFailureCode.RACE_LOST
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.credential_version == winner_version
    assert current.status == "active"
    assert decrypt_bundle(current.token_bundle_encrypted).access_token == "winner-access"


@pytest.mark.asyncio
async def test_valid_token_fast_path_closes_read_transaction_before_return(session, test_user):
    connection, _, now = await _credential(session, test_user, expires_in=3600)
    async with httpx.AsyncClient() as _client:
        assert (
            await refresh_credential(session, connection.id, test_user.id, now=now)
            == "old-access"
        )
    assert not session.in_transaction()


@pytest.mark.asyncio
async def test_valid_token_second_check_releases_transaction_before_return(
    session, test_user, monkeypatch
):
    connection, _, now = await _credential(session, test_user, expires_in=5)

    async def becomes_valid(current_session, connection_id, user_id):
        credential = await get_owned_credential(current_session, connection_id, user_id)
        assert credential is not None
        credential.access_token_expires_at = now + timedelta(hours=1)
        return credential

    monkeypatch.setattr("app.agents.services.oauth_refresh._locked_owned_credential", becomes_valid)
    async with httpx.AsyncClient() as _client:
        assert (
            await refresh_credential(session, connection.id, test_user.id, now=now)
            == "old-access"
        )

    assert not session.in_transaction()


@pytest.mark.asyncio
async def test_process_lock_registry_releases_unused_connection_locks():
    from app.agents.services import oauth_refresh

    ids = [uuid.uuid4() for _ in range(200)]
    for connection_id in ids:
        await oauth_refresh._local_lock(connection_id)
    gc.collect()
    assert len(oauth_refresh._local_locks) < 10


@pytest.mark.asyncio
async def test_invalid_client_revocation_scrubs_but_surfaces_meaningful_error(
    session, test_user, monkeypatch
):
    connection, _, _ = await _credential(session, test_user)

    async def fail(*args, **kwargs):
        raise OAuthLifecycleError(
            OAuthFailureCode.INVALID_CLIENT, "revocation rejected the client", definitive=True
        )

    monkeypatch.setattr("app.agents.services.oauth_refresh.revoke_token", fail)
    async with httpx.AsyncClient() as _client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await revoke_credential(session, connection.id, test_user.id)
    assert exc_info.value.code is OAuthFailureCode.INVALID_CLIENT
    current = await get_owned_credential(session, connection.id, test_user.id)
    assert current is not None
    assert current.token_bundle_encrypted is None
    assert current.last_error_code == "invalid_client"


def test_oauth_connection_provider_resolution_fails_closed():
    from app.agents.models.connection import LlmConnection

    connection = LlmConnection(
        name="ChatGPT", kind="openai_chatgpt", user_id=uuid.uuid4(), extra={}
    )
    with pytest.raises(OAuthLifecycleError) as exc_info:
        connection_service.build_provider_for_connection(connection)
    assert exc_info.value.code is OAuthFailureCode.INACTIVE


def test_connection_test_schema_preserves_typed_oauth_result():
    from app.agents.schemas.connection import ConnectionTestResult

    result = ConnectionTestResult.model_validate(
        {
            "ok": False,
            "detail": "Required ChatGPT plan scopes were not granted",
            "auth_state": "scope_loss",
            "models": [{"slug": "gpt-listed", "display_name": "Listed"}],
        }
    ).model_dump()
    assert result["auth_state"] == "scope_loss"
    assert result["models"] == [{"slug": "gpt-listed", "display_name": "Listed"}]


@pytest.mark.asyncio
async def test_postgres_refresh_lease_prevents_cross_process_refresh_token_reuse(
    postgres_sessions, monkeypatch
):
    user_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    async with postgres_sessions() as setup:
        setup.add(
            User(
                id=user_id,
                email=f"lease-{user_id}@example.com",
                hashed_password="unused",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        await setup.commit()
        connection = await connection_service.create_connection(
            setup, user_id, name="ChatGPT", kind="openai_chatgpt"
        )
        from app.agents.services.oauth_credentials import create_credential

        await create_credential(
            setup,
            connection.id,
            user_id,
            bundle=CredentialBundle("old-access", "one-time-refresh", "old-id"),
            issuer="https://auth.openai.com",
            subject="subject-1",
            issued_client_id="oaiapp_test",
            account_email=None,
            scopes=["openid"],
            access_token_expires_at=now,
            refresh_token_expires_at=now + timedelta(days=30),
        )
        connection_id = connection.id

    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def fake_refresh(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return ValidatedTokenBundle(
            bundle=CredentialBundle("new-access", "new-refresh", "new-id"),
            subject="subject-1",
            scopes=frozenset({"chatgpt.tokens.use.direct"}),
            access_token_expires_at=now + timedelta(hours=1),
            earliest_refresh_at=None,
            refresh_token_expires_at=None,
        )

    async def process_local_lock(_connection_id):
        return asyncio.Lock()

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_token_bundle", fake_refresh)
    monkeypatch.setattr("app.agents.services.oauth_refresh._local_lock", process_local_lock)
    async with postgres_sessions() as first, postgres_sessions() as second:
        async with httpx.AsyncClient() as _client:
            one = asyncio.create_task(
                refresh_credential(
                    first, connection_id, user_id, now=now, force=True
                )
            )
            await entered.wait()
            two = asyncio.create_task(
                refresh_credential(
                    second, connection_id, user_id, now=now, force=True
                )
            )
            await asyncio.sleep(0.05)
            release.set()
            results = await asyncio.gather(one, two, return_exceptions=True)

    assert calls == 1
    assert sum(result == "new-access" for result in results) == 1
    assert (
        sum(
            isinstance(result, OAuthLifecycleError)
            and result.code is OAuthFailureCode.REFRESH_OUTCOME_UNKNOWN
            for result in results
        )
        == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["disable", "reauthorize"])
async def test_postgres_active_refresh_lease_blocks_conflicting_status_transition(
    postgres_sessions, monkeypatch, transition
):
    user_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    async with postgres_sessions() as setup:
        setup.add(
            User(
                id=user_id,
                email=f"lease-transition-{transition}-{user_id}@example.com",
                hashed_password="unused",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        await setup.commit()
        connection = await connection_service.create_connection(
            setup, user_id, name="Lease transition", kind="openai_chatgpt"
        )
        from app.agents.services.oauth_credentials import create_credential

        await create_credential(
            setup,
            connection.id,
            user_id,
            bundle=CredentialBundle("old-access", "one-time-refresh", "old-id"),
            issuer="https://auth.openai.com",
            subject="subject-1",
            issued_client_id="oaiapp_test",
            account_email=None,
            scopes=["openid", "chatgpt.tokens.use.direct"],
            access_token_expires_at=now,
            refresh_token_expires_at=now + timedelta(days=30),
        )
        connection_id = connection.id

    entered = asyncio.Event()
    release = asyncio.Event()

    async def fake_refresh(*args, **kwargs):
        entered.set()
        await release.wait()
        return ValidatedTokenBundle(
            bundle=CredentialBundle("winner-access", "winner-refresh", "winner-id"),
            subject="subject-1",
            scopes=frozenset({"openid", "chatgpt.tokens.use.direct"}),
            access_token_expires_at=now + timedelta(hours=1),
            earliest_refresh_at=None,
            refresh_token_expires_at=now + timedelta(days=30),
        )

    async def process_local_lock(_connection_id):
        return asyncio.Lock()

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_token_bundle", fake_refresh)
    monkeypatch.setattr("app.agents.services.oauth_refresh._local_lock", process_local_lock)
    async with postgres_sessions() as refresher, postgres_sessions() as contender:
        async with httpx.AsyncClient() as _client:
            refresh = asyncio.create_task(
                refresh_credential(
                    refresher, connection_id, user_id, now=now, force=True
                )
            )
            await entered.wait()
            operation = disable_plan_usage if transition == "disable" else require_reauthorization
            with pytest.raises(OAuthLifecycleError) as busy:
                await operation(
                    contender, connection_id, user_id, expected_version=1
                )
            assert busy.value.code is OAuthFailureCode.RACE_LOST
            release.set()
            assert await refresh == "winner-access"

        winner = await get_owned_credential(contender, connection_id, user_id)
        assert winner is not None
        assert winner.status == "active"
        assert winner.credential_version == 2
        assert decrypt_bundle(winner.token_bundle_encrypted).refresh_token == "winner-refresh"
        assert winner.lifecycle_lease_token is None


@pytest.mark.asyncio
async def test_postgres_db_clock_controls_lease_takeover_and_rejects_stale_finalization(
    postgres_sessions,
):
    user_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    async with postgres_sessions() as setup:
        setup.add(
            User(
                id=user_id,
                email=f"lease-takeover-{user_id}@example.com",
                hashed_password="unused",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        await setup.commit()
        connection = await connection_service.create_connection(
            setup, user_id, name="Lease takeover", kind="openai_chatgpt"
        )
        from app.agents.services.oauth_credentials import create_credential

        await create_credential(
            setup,
            connection.id,
            user_id,
            bundle=CredentialBundle("old-access", "one-time-refresh", "old-id"),
            issuer="https://auth.openai.com",
            subject="subject-1",
            issued_client_id="oaiapp_test",
            account_email=None,
            scopes=["openid"],
            access_token_expires_at=now,
            refresh_token_expires_at=now + timedelta(days=30),
        )
        connection_id = connection.id

    async with postgres_sessions() as stale, postgres_sessions() as takeover:
        assert await claim_credential_lease(
            stale,
            connection_id,
            user_id,
            expected_version=1,
            lease_token="stale-owner",
            operation="test",
            lease_duration=timedelta(minutes=5),
        )
        # An absurdly fast application clock is no longer part of the claim API
        # and therefore cannot steal a lease that is live according to PostgreSQL.
        assert not await claim_credential_lease(
            takeover,
            connection_id,
            user_id,
            expected_version=1,
            lease_token="fast-clock-worker",
            operation="test",
            lease_duration=timedelta(minutes=5),
        )
        await stale.execute(
            update(OpenAIOAuthCredential)
            .where(OpenAIOAuthCredential.connection_id == connection_id)
            .values(lifecycle_lease_expires_at=func.now() - timedelta(seconds=1))
        )
        await stale.commit()
        assert await claim_credential_lease(
            takeover,
            connection_id,
            user_id,
            expected_version=1,
            lease_token="new-owner",
            operation="test",
            lease_duration=timedelta(minutes=5),
        )
        with pytest.raises(StaleCredentialVersionError):
            await rotate_credential(
                stale,
                connection_id,
                user_id,
                expected_version=1,
                expected_lease_token="stale-owner",
                bundle=CredentialBundle("stale-access", "stale-refresh", "stale-id"),
                access_token_expires_at=now + timedelta(hours=1),
            )
        winner = await rotate_credential(
            takeover,
            connection_id,
            user_id,
            expected_version=1,
            expected_lease_token="new-owner",
            bundle=CredentialBundle("winner-access", "winner-refresh", "winner-id"),
            access_token_expires_at=now + timedelta(hours=1),
        )

    assert decrypt_bundle(winner.token_bundle_encrypted).access_token == "winner-access"


async def _postgres_lifecycle_credential(postgres_sessions, label: str):
    user_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    async with postgres_sessions() as setup:
        setup.add(
            User(
                id=user_id,
                email=f"{label}-{user_id}@example.com",
                hashed_password="unused",
                is_active=True,
                is_superuser=False,
                is_verified=True,
            )
        )
        await setup.commit()
        connection = await connection_service.create_connection(
            setup, user_id, name=label, kind="openai_chatgpt"
        )
        from app.agents.services.oauth_credentials import create_credential

        await create_credential(
            setup,
            connection.id,
            user_id,
            bundle=CredentialBundle("old-access", "one-time-refresh", "old-id"),
            issuer="https://auth.openai.com",
            subject="subject-1",
            issued_client_id="oaiapp_test",
            account_email=None,
            scopes=["openid", "chatgpt.tokens.use.direct"],
            access_token_expires_at=now,
            refresh_token_expires_at=now + timedelta(days=30),
        )
        return connection.id, user_id, now


@pytest.mark.asyncio
async def test_postgres_snapshot_race_blocks_unknown_attempt_before_provider(
    postgres_sessions, monkeypatch
):
    connection_id, user_id, now = await _postgres_lifecycle_credential(
        postgres_sessions, "snapshot-vs-unknown"
    )
    snapshot_waiting = asyncio.Event()
    release_snapshot = asyncio.Event()
    provider_calls = 0
    from app.agents.services import oauth_credentials

    original_get_owned = oauth_credentials.get_owned_credential

    async def delayed_get_owned(session, target_connection_id, target_user_id):
        snapshot_waiting.set()
        await release_snapshot.wait()
        return await original_get_owned(session, target_connection_id, target_user_id)

    async def return_old_access(*args, **kwargs):
        return "old-access"

    async def forbidden_models(access_token):
        nonlocal provider_calls
        provider_calls += 1
        return []

    monkeypatch.setattr(oauth_refresh, "refresh_credential", return_old_access)
    monkeypatch.setattr(oauth_credentials, "get_owned_credential", delayed_get_owned)
    monkeypatch.setattr(openai_oauth, "list_models", forbidden_models)

    async with postgres_sessions() as probe, postgres_sessions() as contender:
        connection = await connection_service.get_connection(probe, connection_id, user_id)
        assert connection is not None
        task = asyncio.create_task(
            connection_service.test_connection(connection, session=probe, now=now)
        )
        await snapshot_waiting.wait()
        assert await begin_refresh_attempt(
            contender,
            connection_id,
            user_id,
            expected_version=1,
            attempt_id=str(uuid.uuid4()),
            lease_duration=timedelta(minutes=5),
        )
        release_snapshot.set()
        result = await task

    assert result["auth_state"] == "refresh_outcome_unknown"
    assert provider_calls == 0


@pytest.mark.asyncio
async def test_postgres_refresh_reloads_cached_terminal_state_before_fast_path(
    postgres_sessions,
):
    connection_id, user_id, now = await _postgres_lifecycle_credential(
        postgres_sessions, "cached-terminal-refresh"
    )
    async with postgres_sessions() as stale, postgres_sessions() as winner:
        cached = await get_owned_credential(stale, connection_id, user_id)
        assert cached is not None and cached.status == "active"
        await winner.execute(
            update(OpenAIOAuthCredential)
            .where(OpenAIOAuthCredential.connection_id == connection_id)
            .values(
                status="revoked",
                token_bundle_encrypted=None,
                credential_version=2,
            )
        )
        await winner.commit()

        with pytest.raises(OAuthLifecycleError) as exc_info:
            await refresh_credential(stale, connection_id, user_id, now=now)
        assert exc_info.value.code is OAuthFailureCode.REVOKED


@pytest.mark.asyncio
@pytest.mark.parametrize("helper", [disable_plan_usage, require_reauthorization])
async def test_postgres_expired_unknown_attempt_blocks_nonowner_terminal_transition(
    postgres_sessions, helper
):
    connection_id, user_id, _ = await _postgres_lifecycle_credential(
        postgres_sessions, f"unknown-vs-{helper.__name__}"
    )
    attempt_id = str(uuid.uuid4())
    async with postgres_sessions() as stale, postgres_sessions() as contender:
        cached = await get_owned_credential(stale, connection_id, user_id)
        assert cached is not None and cached.status == "active"
        assert await begin_refresh_attempt(
            contender,
            connection_id,
            user_id,
            expected_version=1,
            attempt_id=attempt_id,
            lease_duration=timedelta(seconds=1),
        )
        await contender.execute(
            update(OpenAIOAuthCredential)
            .where(OpenAIOAuthCredential.connection_id == connection_id)
            .values(lifecycle_lease_expires_at=func.now() - timedelta(seconds=1))
        )
        await contender.commit()

        with pytest.raises(OAuthLifecycleError) as exc_info:
            await helper(
                stale,
                connection_id,
                user_id,
                expected_version=1,
            )
        assert exc_info.value.code is OAuthFailureCode.RACE_LOST

        current = await get_owned_credential(stale, connection_id, user_id)
        assert current is not None
        assert current.status == "reauth_required"
        assert current.last_error_code == "refresh_outcome_unknown"
        assert current.refresh_attempt_id == attempt_id
        assert current.token_bundle_encrypted is not None
        assert decrypt_bundle(current.token_bundle_encrypted).refresh_token == "one-time-refresh"


@pytest.mark.asyncio
async def test_postgres_refresh_vs_revoke_never_consumes_then_retains_stale_bundle(
    postgres_sessions, monkeypatch
):
    connection_id, user_id, now = await _postgres_lifecycle_credential(
        postgres_sessions, "refresh-vs-revoke"
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fake_refresh(*args, **kwargs):
        entered.set()
        await release.wait()
        return ValidatedTokenBundle(
            bundle=CredentialBundle("winner-access", "winner-refresh", "winner-id"),
            subject="subject-1",
            scopes=frozenset({"openid", "chatgpt.tokens.use.direct"}),
            access_token_expires_at=now + timedelta(hours=1),
            earliest_refresh_at=None,
            refresh_token_expires_at=now + timedelta(days=30),
            received_at=now,
        )

    async def process_local_lock(_connection_id):
        return asyncio.Lock()

    monkeypatch.setattr("app.agents.services.oauth_refresh.refresh_token_bundle", fake_refresh)
    monkeypatch.setattr("app.agents.services.oauth_refresh._local_lock", process_local_lock)
    async with postgres_sessions() as refresher, postgres_sessions() as revoker:
        async with httpx.AsyncClient() as _client:
            refresh = asyncio.create_task(
                refresh_credential(
                    refresher, connection_id, user_id, now=now, force=True
                )
            )
            await entered.wait()
            with pytest.raises(OAuthLifecycleError) as busy:
                await revoke_credential(revoker, connection_id, user_id)
            assert busy.value.code is OAuthFailureCode.RACE_LOST
            release.set()
            assert await refresh == "winner-access"
        winner = await get_owned_credential(revoker, connection_id, user_id)
        assert winner is not None and winner.status == "active"
        assert decrypt_bundle(winner.token_bundle_encrypted).refresh_token == "winner-refresh"


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["disable", "reauthorize"])
async def test_postgres_revoke_vs_status_transition_keeps_mandatory_cleanup(
    postgres_sessions, monkeypatch, transition
):
    connection_id, user_id, _ = await _postgres_lifecycle_credential(
        postgres_sessions, f"revoke-vs-{transition}"
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    remote_calls = 0

    async def fake_revoke(*args, **kwargs):
        nonlocal remote_calls
        remote_calls += 1
        entered.set()
        await release.wait()

    async def process_local_lock(_connection_id):
        return asyncio.Lock()

    monkeypatch.setattr("app.agents.services.oauth_refresh.revoke_token", fake_revoke)
    monkeypatch.setattr("app.agents.services.oauth_refresh._local_lock", process_local_lock)
    async with postgres_sessions() as revoker, postgres_sessions() as contender:
        async with httpx.AsyncClient() as _client:
            revoke = asyncio.create_task(
                revoke_credential(revoker, connection_id, user_id)
            )
            await entered.wait()
            operation = disable_plan_usage if transition == "disable" else require_reauthorization
            with pytest.raises(OAuthLifecycleError) as busy:
                await operation(
                    contender, connection_id, user_id, expected_version=1
                )
            assert busy.value.code is OAuthFailureCode.RACE_LOST
            release.set()
            await revoke
        current = await get_owned_credential(contender, connection_id, user_id)
        assert current is not None and current.status == "revoked"
        assert current.token_bundle_encrypted is None
        assert remote_calls == 1


@pytest.mark.asyncio
async def test_postgres_cooperative_cancellation_contention_still_scrubs_revoke(
    postgres_sessions, monkeypatch
):
    connection_id, user_id, _ = await _postgres_lifecycle_credential(
        postgres_sessions, "cancelled-revoke"
    )
    entered = asyncio.Event()

    async def blocked_revoke(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    async def process_local_lock(_connection_id):
        return asyncio.Lock()

    monkeypatch.setattr("app.agents.services.oauth_refresh.revoke_token", blocked_revoke)
    monkeypatch.setattr("app.agents.services.oauth_refresh._local_lock", process_local_lock)
    async with postgres_sessions() as owner, postgres_sessions() as contender:
        async with httpx.AsyncClient() as _client:
            task = asyncio.create_task(
                revoke_credential(owner, connection_id, user_id)
            )
            await entered.wait()
            with pytest.raises(OAuthLifecycleError) as busy:
                await revoke_credential(contender, connection_id, user_id)
            assert busy.value.code is OAuthFailureCode.RACE_LOST
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        current = await get_owned_credential(contender, connection_id, user_id)
        assert current is not None and current.status == "revoked"
        assert current.token_bundle_encrypted is None


@pytest.mark.asyncio
async def test_refresh_recomputes_time_after_waiting_for_process_lock(
    session, test_user, monkeypatch
):
    from app.agents.services import oauth_refresh

    connection, _, initial = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id
    fresh = initial + timedelta(minutes=10)
    clock_values = iter([initial, fresh])
    monkeypatch.setattr(oauth_refresh, "_now", lambda: next(clock_values))

    lock = await oauth_refresh._local_lock(connection_id)
    await lock.acquire()
    real_claim = oauth_refresh.begin_refresh_attempt
    claim_durations: list[timedelta] = []

    async def recording_claim(*args, lease_duration, **kwargs):
        claim_durations.append(lease_duration)
        return await real_claim(*args, lease_duration=lease_duration, **kwargs)

    async def fake_refresh(*args, **kwargs):
        assert "now" not in kwargs
        return ValidatedTokenBundle(
            bundle=CredentialBundle("fresh-access", "fresh-refresh", "fresh-id"),
            subject="subject-1",
            scopes=frozenset({"chatgpt.tokens.use.direct"}),
            access_token_expires_at=fresh + timedelta(hours=1),
            earliest_refresh_at=None,
            refresh_token_expires_at=None,
            received_at=fresh,
        )

    monkeypatch.setattr(oauth_refresh, "begin_refresh_attempt", recording_claim)
    monkeypatch.setattr(oauth_refresh, "refresh_token_bundle", fake_refresh)
    async with httpx.AsyncClient() as _client:
        pending = asyncio.create_task(
            refresh_credential(session, connection_id, user_id, force=True)
        )
        await asyncio.sleep(0)
        lock.release()
        assert await pending == "fresh-access"

    assert claim_durations == [oauth_refresh._OPERATION_LEASE]
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.last_refreshed_at is not None
    assert current.last_refreshed_at.replace(tzinfo=timezone.utc) == fresh


@pytest.mark.asyncio
async def test_refresh_persists_provider_response_receipt_not_pre_io_time(
    session, test_user, monkeypatch
):
    from app.agents.services import oauth_refresh

    connection, _, attempt_time = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id
    receipt_time = attempt_time + timedelta(minutes=7)

    async def fake_refresh(*args, receipt_clock, **kwargs):
        assert "now" not in kwargs
        return ValidatedTokenBundle(
            bundle=CredentialBundle("fresh-access", "fresh-refresh", "fresh-id"),
            subject="subject-1",
            scopes=frozenset({"chatgpt.tokens.use.direct"}),
            access_token_expires_at=receipt_time + timedelta(hours=1),
            earliest_refresh_at=receipt_time + timedelta(minutes=1),
            refresh_token_expires_at=receipt_time + timedelta(days=30),
            received_at=receipt_clock(),
        )

    monkeypatch.setattr(oauth_refresh, "refresh_token_bundle", fake_refresh)
    async with httpx.AsyncClient() as _client:
        assert await refresh_credential(
            session,
            connection_id,
            user_id,
            now=attempt_time,
            force=True,
            receipt_clock=lambda: receipt_time,
        ) == "fresh-access"

    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None and current.last_refreshed_at is not None
    assert current.last_refreshed_at.replace(tzinfo=timezone.utc) == receipt_time


def _service_traceback_text(exc: BaseException) -> str:
    retained: list[str] = []
    seen: set[int] = set()

    def visit(error: BaseException | None):
        if error is None or id(error) in seen:
            return
        seen.add(id(error))
        retained.append(repr(error))
        traceback = error.__traceback__
        while traceback is not None:
            if "/app/agents/services/" in traceback.tb_frame.f_code.co_filename:
                retained.extend(repr(value) for value in traceback.tb_frame.f_locals.values())
            traceback = traceback.tb_next
        visit(error.__cause__)
        visit(error.__context__)

    visit(exc)
    return "\n".join(retained)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["refresh", "revoke"])
async def test_lifecycle_failure_discards_decrypted_bundle_from_all_tracebacks(
    session, test_user, monkeypatch, operation
):
    from app.agents.services import oauth_refresh

    connection, _, now = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id
    secrets = CredentialBundle(
        f"{operation}-unique-access-secret",
        f"{operation}-unique-refresh-secret",
        f"{operation}-unique-id-secret",
    )
    monkeypatch.setattr(oauth_refresh, "decrypt_bundle", lambda ciphertext: secrets)

    async def fail(*args, **kwargs):
        raise OAuthLifecycleError(
            OAuthFailureCode.TRANSIENT, "safe lifecycle failure", retryable=True
        )

    monkeypatch.setattr(
        oauth_refresh,
        "refresh_token_bundle" if operation == "refresh" else "revoke_token",
        fail,
    )
    async with httpx.AsyncClient() as _client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            if operation == "refresh":
                await refresh_credential(
                    session, connection_id, user_id, now=now, force=True
                )
            else:
                await revoke_credential(session, connection_id, user_id)

    retained = _service_traceback_text(exc_info.value)
    for secret in (secrets.access_token, secrets.refresh_token, secrets.id_token):
        assert secret not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert set(vars(exc_info.value)) == {"code", "definitive", "retryable"}


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["refresh", "revoke"])
async def test_soft_time_limit_cleanup_completes_before_same_exception_propagates(
    session, test_user, monkeypatch, operation
):
    from app.agents.services import oauth_refresh

    connection, _, now = await _credential(session, test_user)
    connection_id = connection.id
    user_id = test_user.id
    class SlottedSoftLimit(SoftTimeLimitExceeded):
        __slots__ = ("nested",)

    sentinel = f"{operation}-soft-limit-secret"
    limit = SlottedSoftLimit(sentinel)
    limit.nested = {"secret": bytearray(sentinel.encode())}
    try:
        raise RuntimeError(sentinel)
    except RuntimeError as cause:
        limit.__cause__ = cause

    async def stop(*args, **kwargs):
        raise limit

    monkeypatch.setattr(
        oauth_refresh,
        "refresh_token_bundle" if operation == "refresh" else "revoke_token",
        stop,
    )
    async with httpx.AsyncClient() as _client:
        with pytest.raises(SoftTimeLimitExceeded) as exc_info:
            if operation == "refresh":
                await refresh_credential(
                    session,
                    connection_id,
                    user_id,
                    now=now,
                    force=True,
                )
            else:
                await revoke_credential(session, connection_id, user_id)

    assert exc_info.value is limit
    assert limit.args == ()
    assert limit.__cause__ is None
    assert limit.__context__ is None
    assert limit.nested is None
    assert sentinel not in _service_traceback_text(limit)
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    if operation == "refresh":
        assert current.status == "reauth_required"
        assert current.last_error_code == "refresh_outcome_unknown"
        assert current.lifecycle_lease_token is not None
        assert current.token_bundle_encrypted is not None
    else:
        assert current.lifecycle_lease_token is None
        assert current.status == "revoked"
        assert current.token_bundle_encrypted is None


@pytest.mark.asyncio
async def test_refresh_unknown_validity_never_dispatches(session, test_user, monkeypatch):
    from app.agents.services import oauth_refresh
    connection, credential, now = await _credential(session, test_user)
    credential.refresh_token_expires_at = None
    await session.commit()
    dispatched = []

    async def forbidden(*args, **kwargs):
        dispatched.append(True)
        raise AssertionError("refresh validity was not established")

    monkeypatch.setattr(oauth_refresh, "refresh_token_bundle", forbidden)
    with pytest.raises(OAuthLifecycleError) as error:
        await oauth_refresh.refresh_credential(session, connection.id, test_user.id, force=True, now=now)
    assert error.value.code is OAuthFailureCode.REAUTH_REQUIRED
    assert not dispatched
