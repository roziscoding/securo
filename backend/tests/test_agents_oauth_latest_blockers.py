from __future__ import annotations

import inspect
import asyncio
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from pydantic import ValidationError

from app.agents.config import AgentSettings
from app.agents.providers.base import ChatMessage, LLMAuthError, LLMUnavailableError
from app.agents.providers.openai_chatgpt import OpenAIChatGPTProvider
from app.agents.services import connection_service, oauth_refresh, openai_oauth
from app.agents.services.oauth_credentials import decrypt_bundle
from app.agents.services.oauth_credentials import (
    CredentialBundle,
    RESOURCE_CLEANUP_PENDING,
    StaleCredentialVersionError,
    create_credential,
    get_owned_credential,
    rotate_credential,
)
from app.agents.services.openai_oauth import (
    OAuthFailureCode,
    OAuthLifecycleError,
    ValidatedTokenBundle,
)
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


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e9999"])
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


async def _postgres_resource_credential(postgres_sessions, label: str):
    connection_id, user_id, now = await _postgres_credential(postgres_sessions, label)
    async with postgres_sessions() as session:
        credential = await get_owned_credential(session, connection_id, user_id)
        assert credential is not None
        credential.access_token_expires_at = now + timedelta(hours=1)
        credential.earliest_refresh_at = now - timedelta(minutes=1)
        await session.commit()
    return connection_id, user_id, now


@asynccontextmanager
async def _connection_for(postgres_sessions, connection_id, user_id):
    async with postgres_sessions() as worker:
        connection = await connection_service.get_connection(
            worker, connection_id, user_id
        )
        assert connection is not None
        yield worker, connection


def _responses_client(responses, seen_headers):
    class Response:
        headers = {"content-type": "text/event-stream"}

        def __init__(self, status, body):
            self.status_code = status
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def aiter_raw(self):
            if self.body:
                yield self.body

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        def stream(self, _method, _url, **kwargs):
            seen_headers.append(kwargs["headers"])
            return Response(*responses.pop(0))

    return Client


@pytest.mark.asyncio
async def test_postgres_connection_test_blocks_existing_cleanup_pending_before_models(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "connection-test-cleanup-pending"
    )
    async with postgres_sessions() as setup:
        credential = await get_owned_credential(setup, connection_id, user_id)
        assert credential is not None
        credential.last_error_code = RESOURCE_CLEANUP_PENDING
        credential.lifecycle_lease_token = str(uuid.uuid4())
        credential.lifecycle_lease_kind = "resource"
        credential.lifecycle_lease_expires_at = now - timedelta(seconds=1)
        await setup.commit()

    model_calls = 0

    async def list_models(_access_token):
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("cleanup-pending credential reached the provider")

    monkeypatch.setattr(openai_oauth, "list_models", list_models)
    async with _connection_for(
        postgres_sessions, connection_id, user_id
    ) as (worker, connection):
        result = await connection_service.test_connection(
            connection, session=worker, now=now
        )

    assert result["ok"] is False
    assert result["auth_state"] == OAuthFailureCode.RACE_LOST.value
    assert model_calls == 0


@pytest.mark.asyncio
async def test_postgres_connection_test_race_after_refresh_never_calls_models(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "connection-test-refresh-race"
    )
    real_refresh = oauth_refresh.refresh_credential

    async def mark_cleanup_pending(*args, **kwargs):
        token = await real_refresh(*args, **kwargs)
        async with postgres_sessions() as racer:
            credential = await get_owned_credential(racer, connection_id, user_id)
            assert credential is not None
            credential.last_error_code = RESOURCE_CLEANUP_PENDING
            credential.lifecycle_lease_token = str(uuid.uuid4())
            credential.lifecycle_lease_kind = "resource"
            credential.lifecycle_lease_expires_at = now + timedelta(minutes=1)
            await racer.commit()
        return token

    model_calls = 0

    async def list_models(_access_token):
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("racing credential reached the provider")

    monkeypatch.setattr(oauth_refresh, "refresh_credential", mark_cleanup_pending)
    monkeypatch.setattr(openai_oauth, "list_models", list_models)
    async with _connection_for(
        postgres_sessions, connection_id, user_id
    ) as (worker, connection):
        result = await connection_service.test_connection(
            connection, session=worker, now=now
        )

    assert result["ok"] is False
    assert result["auth_state"] == OAuthFailureCode.RACE_LOST.value
    assert model_calls == 0


@pytest.mark.asyncio
async def test_postgres_connection_test_cannot_overlap_inference_resource_lease(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "connection-test-inference-race"
    )
    entered = asyncio.Event()
    release = asyncio.Event()

    async def inference():
        async with postgres_sessions() as inference_session:
            async with oauth_refresh.credential_access(
                inference_session, connection_id, user_id, now=now
            ):
                entered.set()
                await release.wait()

    inference_task = asyncio.create_task(inference())
    await entered.wait()
    model_calls = 0

    async def list_models(_access_token):
        nonlocal model_calls
        model_calls += 1
        raise AssertionError("connection test overlapped inference")

    monkeypatch.setattr(openai_oauth, "list_models", list_models)
    try:
        async with _connection_for(
            postgres_sessions, connection_id, user_id
        ) as (worker, connection):
            result = await connection_service.test_connection(
                connection, session=worker, now=now
            )
    finally:
        release.set()
        await inference_task

    assert result["ok"] is False
    assert result["auth_state"] == OAuthFailureCode.RACE_LOST.value
    assert model_calls == 0


@pytest.mark.asyncio
async def test_postgres_connection_test_401_refreshes_once_with_bundle_preserved(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "connection-test-401-refresh"
    )
    refresh_calls = 0
    model_tokens = []

    async def rotate(**_kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        async with postgres_sessions() as observer:
            current = await get_owned_credential(observer, connection_id, user_id)
            assert current is not None
            assert current.token_bundle_encrypted is not None
            assert decrypt_bundle(current.token_bundle_encrypted).refresh_token == (
                "single-use-refresh"
            )
        return ValidatedTokenBundle(
            bundle=CredentialBundle("rotated-access", "rotated-refresh", "rotated-id"),
            subject="subject-1",
            account_email="verified@example.com",
            scopes=frozenset({"openid", "chatgpt.tokens.use.direct"}),
            access_token_expires_at=now + timedelta(hours=2),
            earliest_refresh_at=now + timedelta(minutes=30),
            refresh_token_expires_at=now + timedelta(days=30),
        )

    async def list_models(access_token):
        model_tokens.append(access_token)
        async with postgres_sessions() as observer:
            current = await get_owned_credential(observer, connection_id, user_id)
            assert current is not None
            assert current.lifecycle_lease_kind == "resource"
            assert current.lifecycle_lease_token is not None
        if len(model_tokens) == 1:
            raise OAuthLifecycleError(
                OAuthFailureCode.REAUTH_REQUIRED,
                "rejected",
                definitive=True,
            )
        return [{"slug": "gpt-test", "display_name": "GPT Test"}]

    monkeypatch.setattr(oauth_refresh, "refresh_token_bundle", rotate)
    monkeypatch.setattr(openai_oauth, "list_models", list_models)
    async with _connection_for(
        postgres_sessions, connection_id, user_id
    ) as (worker, connection):
        result = await connection_service.test_connection(
            connection, session=worker, now=now
        )

    assert result["ok"] is True
    assert refresh_calls == 1
    assert model_tokens == ["old-access", "rotated-access"]
    async with postgres_sessions() as observer:
        current = await get_owned_credential(observer, connection_id, user_id)
        assert current is not None
        assert current.credential_version == 2
        assert current.lifecycle_lease_token is None


@pytest.mark.asyncio
async def test_postgres_connection_test_second_401_is_terminal_after_one_refresh(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "connection-test-second-401"
    )
    refresh_calls = 0
    model_calls = 0

    async def rotate(**_kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        return ValidatedTokenBundle(
            bundle=CredentialBundle("rotated-access", "rotated-refresh", "rotated-id"),
            subject="subject-1",
            account_email="verified@example.com",
            scopes=frozenset({"openid", "chatgpt.tokens.use.direct"}),
            access_token_expires_at=now + timedelta(hours=2),
            earliest_refresh_at=now + timedelta(minutes=30),
            refresh_token_expires_at=now + timedelta(days=30),
        )

    async def reject_models(_access_token):
        nonlocal model_calls
        model_calls += 1
        raise OAuthLifecycleError(
            OAuthFailureCode.REAUTH_REQUIRED,
            "rejected",
            definitive=True,
        )

    monkeypatch.setattr(oauth_refresh, "refresh_token_bundle", rotate)
    monkeypatch.setattr(openai_oauth, "list_models", reject_models)
    async with _connection_for(
        postgres_sessions, connection_id, user_id
    ) as (worker, connection):
        result = await connection_service.test_connection(
            connection, session=worker, now=now
        )

    assert result["ok"] is False
    assert result["auth_state"] == OAuthFailureCode.REAUTH_REQUIRED.value
    assert refresh_calls == 1
    assert model_calls == 2
    async with postgres_sessions() as observer:
        current = await get_owned_credential(observer, connection_id, user_id)
        assert current is not None
        assert current.status == "reauth_required"
        assert current.token_bundle_encrypted is None
        assert current.lifecycle_lease_token is None


@pytest.mark.asyncio
async def test_postgres_first_401_preserves_bundle_then_version_refreshes_and_retries(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "resource-401-refresh"
    )
    refresh_calls = 0
    observed_before_refresh = None

    async def rotate(**_kwargs):
        nonlocal refresh_calls, observed_before_refresh
        refresh_calls += 1
        async with postgres_sessions() as observer:
            observed_before_refresh = await get_owned_credential(
                observer, connection_id, user_id
            )
            assert observed_before_refresh is not None
            assert observed_before_refresh.token_bundle_encrypted is not None
        return ValidatedTokenBundle(
            bundle=CredentialBundle("rotated-access", "rotated-refresh", "rotated-id"),
            subject="subject-1",
            account_email="verified@example.com",
            scopes=frozenset({"openid", "chatgpt.tokens.use.direct"}),
            access_token_expires_at=now + timedelta(hours=2),
            earliest_refresh_at=now + timedelta(minutes=30),
            refresh_token_expires_at=now + timedelta(days=30),
        )

    monkeypatch.setattr(oauth_refresh, "refresh_token_bundle", rotate)
    complete = (
        b'data: {"type":"response.created","sequence_number":0,"response":{"id":"ok"}}\n\n'
        b'data: {"type":"response.completed","sequence_number":1,"response":{"id":"ok",'
        b'"status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
    )
    responses = [(401, b'{"error":"expired"}'), (200, complete)]
    seen_headers = []

    async with postgres_sessions() as worker:
        @asynccontextmanager
        async def access_token(*, force=False, rejected_version=None):
            async with oauth_refresh.credential_access(
                worker,
                connection_id,
                user_id,
                now=now,
                force=force,
                rejected_version=rejected_version,
            ) as snapshot:
                yield snapshot

        provider = OpenAIChatGPTProvider(access_token=access_token)
        client = _responses_client(responses, seen_headers)
        with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", client):
            chunks = [
                chunk
                async for chunk in provider.chat_stream(
                    [ChatMessage(role="user", content="retry")], model="gpt-test"
                )
            ]

    assert refresh_calls == 1
    assert observed_before_refresh is not None
    assert [headers["Authorization"] for headers in seen_headers] == [
        "Bearer old-access",
        "Bearer rotated-access",
    ]
    assert chunks[-1].type == "finish"
    async with postgres_sessions() as observer:
        current = await get_owned_credential(observer, connection_id, user_id)
        assert current is not None
        assert current.status == "active"
        assert current.credential_version == 2
        assert current.lifecycle_lease_token is None
        assert decrypt_bundle(current.token_bundle_encrypted) == CredentialBundle(
            "rotated-access", "rotated-refresh", "rotated-id"
        )


@pytest.mark.asyncio
async def test_postgres_first_401_terminal_refresh_requires_reauthorization(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "resource-401-terminal-refresh"
    )
    refresh_calls = 0

    async def reject_refresh(**_kwargs):
        nonlocal refresh_calls
        refresh_calls += 1
        raise OAuthLifecycleError(
            OAuthFailureCode.INVALID_CLIENT,
            "provider terminal detail",
            definitive=True,
        )

    monkeypatch.setattr(oauth_refresh, "refresh_token_bundle", reject_refresh)
    responses = [(401, b'{"error":"expired"}')]
    seen_headers = []

    async with postgres_sessions() as worker:
        @asynccontextmanager
        async def access_token(*, force=False, rejected_version=None):
            async with oauth_refresh.credential_access(
                worker,
                connection_id,
                user_id,
                now=now,
                force=force,
                rejected_version=rejected_version,
            ) as snapshot:
                yield snapshot

        provider = OpenAIChatGPTProvider(access_token=access_token)
        client = _responses_client(responses, seen_headers)
        with patch("app.agents.providers.openai_chatgpt.httpx.AsyncClient", client):
            with pytest.raises(LLMAuthError):
                _ = [
                    chunk
                    async for chunk in provider.chat_stream(
                        [ChatMessage(role="user", content="retry")], model="gpt-test"
                    )
                ]

    assert refresh_calls == 1
    assert len(seen_headers) == 1
    async with postgres_sessions() as observer:
        current = await get_owned_credential(observer, connection_id, user_id)
        assert current is not None
        assert current.status == "reauth_required"
        assert current.last_error_code == OAuthFailureCode.INVALID_CLIENT.value
        assert current.token_bundle_encrypted is None
        assert current.lifecycle_lease_token is None


@pytest.mark.asyncio
async def test_postgres_resource_cleanup_failure_is_sanitized_and_leaves_expiring_lease(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "resource-cleanup-failure"
    )

    async def fail_release(*_args, **_kwargs):
        raise RuntimeError("raw postgres cleanup sentinel")

    monkeypatch.setattr(oauth_refresh, "release_credential_lease", fail_release)
    async with postgres_sessions() as worker:
        with pytest.raises(LLMUnavailableError) as exc_info:
            async with oauth_refresh.credential_access(
                worker, connection_id, user_id, now=now
            ):
                pass

    assert str(exc_info.value) == "ChatGPT credential cleanup failed"
    assert "sentinel" not in repr(exc_info.value)
    async with postgres_sessions() as observer:
        current = await get_owned_credential(observer, connection_id, user_id)
        assert current is not None
        assert current.status == "active"
        assert current.token_bundle_encrypted is not None
        assert current.lifecycle_lease_kind == "resource"
        assert current.lifecycle_lease_token is not None
        assert current.lifecycle_lease_expires_at is not None
        assert current.last_error_code == RESOURCE_CLEANUP_PENDING

        current.lifecycle_lease_expires_at = now - timedelta(seconds=1)
        await observer.commit()
        with pytest.raises(OAuthLifecycleError) as blocked:
            async with oauth_refresh.credential_access(
                observer, connection_id, user_id, now=now
            ):
                raise AssertionError("cleanup-pending credential must not be reused")
        assert blocked.value.code is OAuthFailureCode.RACE_LOST

        refresh_calls = 0

        async def should_not_refresh(**_kwargs):
            nonlocal refresh_calls
            refresh_calls += 1
            raise AssertionError("cleanup-pending credential must not be refreshed")

        monkeypatch.setattr(oauth_refresh, "refresh_token_bundle", should_not_refresh)
        with pytest.raises(OAuthLifecycleError) as refresh_blocked:
            await oauth_refresh.refresh_credential(
                observer, connection_id, user_id, now=now, force=True
            )
        assert refresh_blocked.value.code is OAuthFailureCode.RACE_LOST
        assert refresh_calls == 0


@pytest.mark.asyncio
async def test_postgres_explicit_reauthorization_replaces_cleanup_pending_bundle(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "resource-explicit-reauthorization"
    )
    async with postgres_sessions() as session:
        current = await get_owned_credential(session, connection_id, user_id)
        assert current is not None
        current.last_error_code = RESOURCE_CLEANUP_PENDING
        current.lifecycle_lease_token = str(uuid.uuid4())
        current.lifecycle_lease_kind = "resource"
        current.lifecycle_lease_expires_at = now - timedelta(seconds=1)
        await session.commit()

        recovered = await rotate_credential(
            session,
            connection_id,
            user_id,
            expected_version=current.credential_version,
            bundle=CredentialBundle("new-access", "new-refresh", "new-id"),
            access_token_expires_at=now + timedelta(hours=1),
            refresh_token_expires_at=now + timedelta(days=30),
            status="active",
            last_error_code=None,
        )
        assert recovered.credential_version == 2
        assert recovered.status == "active"
        assert recovered.last_error_code is None
        assert recovered.lifecycle_lease_token is None
        assert decrypt_bundle(recovered.token_bundle_encrypted) == CredentialBundle(
            "new-access", "new-refresh", "new-id"
        )

        async with oauth_refresh.credential_access(
            session, connection_id, user_id, now=now
        ) as snapshot:
            assert snapshot.access_token == "new-access"
            assert snapshot.credential_version == 2


@pytest.mark.asyncio
async def test_postgres_terminal_resource_cleanup_survives_repeated_cancellation(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "resource-terminal-cancel"
    )
    entered = asyncio.Event()
    real_scrub = oauth_refresh.scrub_credential_bundle

    async def delayed_scrub(*args, **kwargs):
        entered.set()
        await asyncio.sleep(0.2)
        return await real_scrub(*args, **kwargs)

    monkeypatch.setattr(oauth_refresh, "scrub_credential_bundle", delayed_scrub)
    primary = LLMAuthError("terminal", status=401)

    async def use_credential():
        async with postgres_sessions() as request_session:
            async with oauth_refresh.credential_access(
                request_session, connection_id, user_id, now=now
            ):
                raise primary

    task = asyncio.create_task(use_credential())
    await entered.wait()
    for _ in range(5):
        task.cancel("cancel-spam")
        await asyncio.sleep(0)
    with pytest.raises(LLMAuthError) as exc_info:
        await task
    assert exc_info.value is primary

    async with postgres_sessions() as observer:
        current = await get_owned_credential(observer, connection_id, user_id)
        assert current is not None
        assert current.status == "reauth_required"
        assert current.token_bundle_encrypted is None
        assert current.lifecycle_lease_token is None

        with pytest.raises(OAuthLifecycleError) as access_error:
            async with oauth_refresh.credential_access(
                observer, connection_id, user_id, now=now + timedelta(days=1)
            ):
                raise AssertionError("terminal credential must never be reused")
        assert access_error.value.code is OAuthFailureCode.REAUTH_REQUIRED


@pytest.mark.asyncio
async def test_postgres_outer_timeout_waits_for_shielded_resource_release(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "resource-release-timeout"
    )
    real_release = oauth_refresh.release_credential_lease

    async def delayed_release(*args, **kwargs):
        await asyncio.sleep(0.2)
        return await real_release(*args, **kwargs)

    monkeypatch.setattr(oauth_refresh, "release_credential_lease", delayed_release)
    loop = asyncio.get_running_loop()
    started = loop.time()

    async with postgres_sessions() as request_session:
        access = oauth_refresh.credential_access(
            request_session, connection_id, user_id, now=now
        )
        await access.__aenter__()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(access.__aexit__(None, None, None), timeout=0.01)
    elapsed = loop.time() - started
    assert 0.15 <= elapsed < 1.0

    async with postgres_sessions() as observer:
        current = await get_owned_credential(observer, connection_id, user_id)
        assert current is not None
        assert current.lifecycle_lease_token is None


@pytest.mark.asyncio
async def test_postgres_resource_release_preserves_primary_cancellation(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "resource-release-cancel"
    )
    entered = asyncio.Event()
    real_release = oauth_refresh.release_credential_lease

    async def delayed_release(*args, **kwargs):
        entered.set()
        await asyncio.sleep(0.2)
        return await real_release(*args, **kwargs)

    monkeypatch.setattr(oauth_refresh, "release_credential_lease", delayed_release)
    primary = asyncio.CancelledError("primary cancellation")

    async def use_credential():
        async with postgres_sessions() as request_session:
            async with oauth_refresh.credential_access(
                request_session, connection_id, user_id, now=now
            ):
                raise primary

    task = asyncio.create_task(use_credential())
    await entered.wait()
    for _ in range(3):
        task.cancel("later cancellation")
        await asyncio.sleep(0)
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await task
    assert exc_info.value is primary

    async with postgres_sessions() as observer:
        current = await get_owned_credential(observer, connection_id, user_id)
        assert current is not None
        assert current.status == "active"
        assert current.lifecycle_lease_token is None
        assert current.token_bundle_encrypted is not None


@pytest.mark.asyncio
async def test_postgres_release_resolves_commit_then_raise_by_readback(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "resource-release-uncertain-commit"
    )
    real_release = oauth_refresh.release_credential_lease

    async def commit_then_raise(*args, **kwargs):
        await real_release(*args, **kwargs)
        raise RuntimeError("commit acknowledgement lost")

    monkeypatch.setattr(oauth_refresh, "release_credential_lease", commit_then_raise)
    async with postgres_sessions() as request_session:
        async with oauth_refresh.credential_access(
            request_session, connection_id, user_id, now=now
        ):
            pass

    async with postgres_sessions() as observer:
        current = await get_owned_credential(observer, connection_id, user_id)
        assert current is not None
        assert current.lifecycle_lease_token is None


@pytest.mark.asyncio
async def test_postgres_terminal_cleanup_resolves_commit_then_raise_by_readback(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "resource-uncertain-commit"
    )
    real_scrub = oauth_refresh.scrub_credential_bundle

    async def commit_then_raise(*args, **kwargs):
        await real_scrub(*args, **kwargs)
        raise RuntimeError("commit acknowledgement lost")

    monkeypatch.setattr(oauth_refresh, "scrub_credential_bundle", commit_then_raise)
    primary = LLMAuthError("terminal", status=401)
    async with postgres_sessions() as request_session:
        with pytest.raises(LLMAuthError) as exc_info:
            async with oauth_refresh.credential_access(
                request_session, connection_id, user_id, now=now
            ):
                raise primary
    assert exc_info.value is primary

    async with postgres_sessions() as observer:
        current = await get_owned_credential(observer, connection_id, user_id)
        assert current is not None
        assert current.status == "reauth_required"
        assert current.token_bundle_encrypted is None


@pytest.mark.asyncio
async def test_postgres_resource_cleanup_ignores_broken_request_session(
    postgres_sessions, monkeypatch
):
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    connection_id, user_id, now = await _postgres_resource_credential(
        postgres_sessions, "resource-broken-request-session"
    )
    request_context = postgres_sessions()
    request_session = await request_context.__aenter__()
    access = oauth_refresh.credential_access(
        request_session, connection_id, user_id, now=now
    )
    await access.__aenter__()
    await request_session.rollback()
    await request_context.__aexit__(None, None, None)
    await access.__aexit__(None, None, None)

    async with postgres_sessions() as observer:
        current = await get_owned_credential(observer, connection_id, user_id)
        assert current is not None
        assert current.lifecycle_lease_token is None
    assert not oauth_refresh._resource_cleanup_tasks


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


@pytest.mark.asyncio
async def test_oauth_transport_requests_identity_and_consumes_raw_stream(monkeypatch):
    observed = {}

    class RawStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"{}"

    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            observed["accept_encoding"] = request.headers.get("accept-encoding")
            return httpx.Response(200, stream=RawStream(), request=request)

    monkeypatch.setattr(openai_oauth, "_new_oauth_transport", Transport)
    response = await openai_oauth._request("GET", openai_oauth.JWKS_ENDPOINT)
    assert observed == {"accept_encoding": "identity"}
    assert response.content == b"{}"


@pytest.mark.asyncio
async def test_oauth_transport_rejects_content_encoding_before_raw_body(monkeypatch):
    consumed = 0

    class UnreadStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            nonlocal consumed
            consumed += 1
            yield b"compressed"

    class Transport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200,
                headers={"Content-Encoding": "gzip"},
                stream=UnreadStream(),
                request=request,
            )

    monkeypatch.setattr(openai_oauth, "_new_oauth_transport", Transport)
    with pytest.raises(OAuthLifecycleError) as caught:
        await openai_oauth._request("GET", openai_oauth.JWKS_ENDPOINT)
    assert caught.value.code is OAuthFailureCode.MALFORMED_RESPONSE
    assert consumed == 0
