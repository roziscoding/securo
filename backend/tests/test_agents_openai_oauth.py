"""Deterministic OpenAI SIWC protocol and HTTP contract tests."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
from contextvars import ContextVar
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from billiard.exceptions import SoftTimeLimitExceeded
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from jose import jwt

from app.agents.services import openai_oauth
from app.agents.services.openai_oauth import (
    ISSUER,
    JWKS_ENDPOINT,
    MODELS_ENDPOINT,
    REQUIRED_SCOPES,
    RESOURCE,
    REVOKE_ENDPOINT,
    TOKEN_ENDPOINT,
    OAuthFailureCode,
    OAuthLifecycleError,
    exchange_authorization_code as _exchange_authorization_code,
    list_models as _list_models,
    refresh_token_bundle as _refresh_token_bundle,
    revoke_token as _revoke_token,
    validate_id_token,
    _json_object,
    _new_oauth_transport as _new_oauth_transport_impl,
    _request as _request_impl,
)


# Protocol traffic ignores caller transports. Tests monkeypatch the private
# owned-transport factory; the task-local holder keeps concurrent calls isolated.

_owned_test_transport: ContextVar[httpx.MockTransport | None] = ContextVar(
    "owned_test_transport", default=None
)


@pytest.fixture(autouse=True)
def _patch_owned_transport_factory(monkeypatch):
    def factory():
        transport = _owned_test_transport.get()
        if transport is None:
            return httpx.AsyncHTTPTransport(trust_env=False)
        return transport

    monkeypatch.setattr(openai_oauth, "_new_oauth_transport", factory)


def _mock_transport(client: httpx.AsyncClient) -> httpx.MockTransport | None:
    transport = client._transport  # type: ignore[attr-defined]
    return transport if type(transport) is httpx.MockTransport else None


async def exchange_authorization_code(client, **kwargs):
    token = _owned_test_transport.set(_mock_transport(client))
    try:
        return await _exchange_authorization_code(**kwargs)
    finally:
        _owned_test_transport.reset(token)


async def refresh_token_bundle(client, **kwargs):
    token = _owned_test_transport.set(_mock_transport(client))
    try:
        return await _refresh_token_bundle(**kwargs)
    finally:
        _owned_test_transport.reset(token)


async def fetch_validated_jwks(client):
    token = _owned_test_transport.set(_mock_transport(client))
    try:
        return await openai_oauth.fetch_validated_jwks()
    finally:
        _owned_test_transport.reset(token)


async def revoke_token(client, **kwargs):
    token = _owned_test_transport.set(_mock_transport(client))
    try:
        return await _revoke_token(**kwargs)
    finally:
        _owned_test_transport.reset(token)


async def list_models(client, access_token):
    token = _owned_test_transport.set(_mock_transport(client))
    try:
        return await _list_models(access_token)
    finally:
        _owned_test_transport.reset(token)


async def _request(client, method, url, **kwargs):
    token = _owned_test_transport.set(_mock_transport(client))
    try:
        return await _request_impl(method, url, **kwargs)
    finally:
        _owned_test_transport.reset(token)


class _TrackingStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.yielded = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class _SlowStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.closed = False

    async def __aiter__(self):
        await openai_oauth.asyncio.sleep(60)
        yield b"late response"

    async def aclose(self) -> None:
        self.closed = True


class _ExplodingStream(httpx.AsyncByteStream):
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.closed = False

    async def __aiter__(self):
        yield b"provider-secret-prefix"
        raise self.error

    async def aclose(self) -> None:
        self.closed = True


def _b64uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


@pytest.fixture(scope="module")
def signing_material():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = key.public_key().public_numbers()
    jwk = {
        "kty": "RSA",
        "kid": "test-key",
        "use": "sig",
        "alg": "RS256",
        "n": _b64uint(public.n),
        "e": _b64uint(public.e),
    }
    return key, {"keys": [jwk]}


def _id_token(key, *, client_id="oaiapp_test", nonce="nonce-value", subject="subject-1", **claims):
    now = datetime.now(timezone.utc)
    payload = {
        "iss": ISSUER,
        "aud": client_id,
        "sub": subject,
        "nonce": nonce,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
        **claims,
    }
    return jwt.encode(payload, key, algorithm="RS256", headers={"kid": "test-key"})


def _at_hash(access_token: str) -> str:
    digest = hashlib.sha256(access_token.encode()).digest()
    return base64.urlsafe_b64encode(digest[: len(digest) // 2]).rstrip(b"=").decode()


def _access_token(key, *, subject="subject-1", audience=RESOURCE, **claims):
    now = datetime.now(timezone.utc)
    payload = {
        "iss": ISSUER,
        "aud": audience,
        "sub": subject,
        "client_id": "oaiapp_test",
        "scope": " ".join(sorted(REQUIRED_SCOPES)),
        "nbf": int(now.timestamp()),
        "jti": "access-test-id",
        "https://api.openai.com/auth": {"per_user_salt": "salt", "encrypted_auth_metadata": "metadata"},
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
        **claims,
    }
    return jwt.encode(payload, key, algorithm="RS256", headers={"kid": "test-key"})


def _signed_raw_jwt(key, header_json: str, claims_json: str) -> str:
    header = base64.urlsafe_b64encode(header_json.encode()).rstrip(b"=").decode()
    claims = base64.urlsafe_b64encode(claims_json.encode()).rstrip(b"=").decode()
    signing_input = f"{header}.{claims}".encode()
    signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    encoded_signature = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()
    return f"{header}.{claims}.{encoded_signature}"


def _token_payload(key, **overrides):
    access_token = _access_token(key, scope=overrides.get("scope", " ".join(sorted(REQUIRED_SCOPES))))
    access_claims = jwt.get_unverified_claims(access_token)
    payload = {
        "access_token": access_token,
        "refresh_token": "refresh-value",
        "id_token": _id_token(
            key,
            iat=access_claims["iat"],
            exp=access_claims["exp"],
            at_hash=_at_hash(access_token),
        ),
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": " ".join(sorted(REQUIRED_SCOPES)),
        "earliest_refresh_at": int(datetime.now(timezone.utc).timestamp()),
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio
async def test_prefetched_jwks_is_immutable_and_used_without_refetch_or_key_substitution(
    signing_material,
):
    key, fixture_jwks = signing_material
    source_jwks = json.loads(json.dumps(fixture_jwks))
    calls: list[httpx.URL] = []

    def handler(request: httpx.Request):
        calls.append(request.url)
        if request.url == JWKS_ENDPOINT:
            return httpx.Response(200, json=source_jwks)
        assert request.url == TOKEN_ENDPOINT
        return httpx.Response(200, json=_token_payload(key))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        prefetched = await fetch_validated_jwks(client)
        original_snapshot = prefetched.copy_for_verification()
        replacement_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        replacement_numbers = replacement_key.public_key().public_numbers()
        source_jwks["keys"][0]["n"] = _b64uint(replacement_numbers.n)
        with pytest.raises(FrozenInstanceError):
            setattr(prefetched, "_canonical_json", b"{}")

        rotated = await refresh_token_bundle(
            client,
            refresh_token="rotating-refresh",
            issued_client_id="oaiapp_test",
            expected_subject="subject-1",
            jwks=prefetched,
        )

    assert prefetched.copy_for_verification() == original_snapshot
    assert rotated.bundle.refresh_token == "refresh-value"
    assert calls == [httpx.URL(JWKS_ENDPOINT), httpx.URL(TOKEN_ENDPOINT)]


@pytest.mark.asyncio
async def test_authorization_exchange_validates_signature_claims_nonce_and_scopes(signing_material):
    key, jwks = signing_material
    calls: list[httpx.Request] = []
    token_forms: list[dict[str, str]] = []

    def handler(request: httpx.Request):
        calls.append(request)
        if request.url == JWKS_ENDPOINT:
            return httpx.Response(200, json=jwks)
        assert request.url == TOKEN_ENDPOINT
        token_forms.append(dict(item.split("=", 1) for item in request.content.decode().split("&")))
        return httpx.Response(200, json=_token_payload(key))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        result = await exchange_authorization_code(
            client,
            code="authorization-code",
            verifier="verifier-value",
            redirect_uri="http://127.0.0.1:1455/auth/callback",
            issued_client_id="oaiapp_test",
            nonce="nonce-value",
            receipt_clock=lambda: datetime.now(timezone.utc),
        )

    assert result.bundle.refresh_token == "refresh-value"
    assert result.subject == "subject-1"
    assert result.scopes == REQUIRED_SCOPES
    form = token_forms[0]
    assert form["grant_type"] == "authorization_code"
    assert "client_secret" not in form
    assert [request.url for request in calls] == [
        httpx.URL(JWKS_ENDPOINT),
        httpx.URL(TOKEN_ENDPOINT),
    ]
    assert all(not request.extensions.get("follow_redirects") for request in calls)


@pytest.mark.parametrize("missing_claim", ["iss", "aud", "exp", "iat", "sub"])
def test_id_token_requires_all_identity_claims(signing_material, missing_claim):
    key, jwks = signing_material
    access = _access_token(key)
    now = datetime.now(timezone.utc)
    claims = {
        "iss": ISSUER,
        "aud": "oaiapp_test",
        "sub": "subject-1",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=10)).timestamp()),
        "at_hash": _at_hash(access),
    }
    claims.pop(missing_claim)
    encoded = jwt.encode(claims, key, algorithm="RS256", headers={"kid": "test-key"})

    with pytest.raises(OAuthLifecycleError) as exc_info:
        validate_id_token(encoded, jwks=jwks, issued_client_id="oaiapp_test", access_token=access)

    assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


def test_id_token_requires_matching_azp_for_multiple_audiences(signing_material):
    key, jwks = signing_material
    access = _access_token(key)
    encoded = _id_token(
        key,
        aud=["oaiapp_test", "another-client"],
        azp="another-client",
        at_hash=_at_hash(access),
    )

    with pytest.raises(OAuthLifecycleError) as exc_info:
        validate_id_token(encoded, jwks=jwks, issued_client_id="oaiapp_test", access_token=access)

    assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


@pytest.mark.parametrize("valid", [True, False])
def test_id_token_explicitly_validates_at_hash(signing_material, valid):
    key, jwks = signing_material
    access = _access_token(key)
    encoded = _id_token(key, at_hash=_at_hash(access) if valid else _at_hash("other-token"))

    if valid:
        assert (
            validate_id_token(
                encoded, jwks=jwks, issued_client_id="oaiapp_test", access_token=access
            )["sub"]
            == "subject-1"
        )
    else:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            validate_id_token(
                encoded, jwks=jwks, issued_client_id="oaiapp_test", access_token=access
            )
        assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


@pytest.mark.parametrize(
    "claim_changes",
    [
        {"iat": float("nan")},
        {"iat": "not-numeric"},
        {"iat": "future"},
        {"exp": float("inf")},
        {"exp": "not-numeric"},
        {"exp": "before-iat"},
        {"exp": "too-long"},
    ],
)
def test_id_token_explicitly_bounds_numeric_times(signing_material, claim_changes):
    key, jwks = signing_material
    now = datetime(2026, 1, 2, 12, 0, tzinfo=timezone.utc)
    access = _access_token(key)
    replacements = dict(claim_changes)
    if replacements.get("iat") == "future":
        replacements["iat"] = int((now + timedelta(seconds=61)).timestamp())
    else:
        replacements.setdefault("iat", int(now.timestamp()))
    if replacements.get("exp") == "before-iat":
        replacements["exp"] = int(now.timestamp())
    elif replacements.get("exp") == "too-long":
        replacements["exp"] = int((now + timedelta(hours=24, seconds=1)).timestamp())
    else:
        replacements.setdefault("exp", int((now + timedelta(minutes=10)).timestamp()))
    encoded = _id_token(key, at_hash=_at_hash(access), **replacements)

    with pytest.raises(OAuthLifecycleError) as exc_info:
        validate_id_token(
            encoded,
            jwks=jwks,
            issued_client_id="oaiapp_test",
            access_token=access,
            now=now,
        )

    assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


@pytest.mark.asyncio
async def test_exchange_validates_jwt_access_token_signature_resource_and_identity(
    signing_material,
):
    key, jwks = signing_material
    wrong_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    now = datetime.now(timezone.utc)
    variants = [
        _access_token(wrong_key),
        _access_token(key, audience="https://evil.example"),
        _access_token(key, subject="different-subject"),
        _access_token(key, iat=int((now + timedelta(minutes=5)).timestamp())),
    ]
    for access in variants:

        def handler(request: httpx.Request, access=access):
            payload = _token_payload(
                key,
                access_token=access,
                id_token=_id_token(key, at_hash=_at_hash(access)),
            )
            return httpx.Response(200, json=jwks if request.url == JWKS_ENDPOINT else payload)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(OAuthLifecycleError) as exc_info:
                await exchange_authorization_code(
                    client,
                    code="c",
                    verifier="v",
                    redirect_uri="http://127.0.0.1:1/cb",
                    issued_client_id="oaiapp_test",
                    nonce="nonce-value",
                )
        assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN
        assert exc_info.value.definitive is True
        retained = _production_traceback_text(exc_info.value)
        assert access not in retained
        assert "refresh-value" not in retained


@pytest.mark.asyncio
async def test_exchange_accepts_conservatively_treated_opaque_access_token(signing_material):
    key, jwks = signing_material
    access = "opaque-access-token"
    payload = _token_payload(
        key, access_token=access, id_token=_id_token(key, at_hash=_at_hash(access))
    )

    def handler(request: httpx.Request):
        return httpx.Response(200, json=jwks if request.url == JWKS_ENDPOINT else payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await exchange_authorization_code(
            client,
            code="c",
            verifier="v",
            redirect_uri="http://127.0.0.1:1/cb",
            issued_client_id="oaiapp_test",
            nonce="nonce-value",
        )
    assert result.bundle.access_token == access


@pytest.mark.asyncio
async def test_exchange_treats_non_jose_three_segment_access_token_as_opaque(signing_material):
    key, jwks = signing_material
    access = "opaque.part.token"
    payload = _token_payload(
        key, access_token=access, id_token=_id_token(key, at_hash=_at_hash(access))
    )
    calls: list[str] = []

    def handler(request: httpx.Request):
        calls.append(str(request.url))
        return httpx.Response(200, json=jwks if request.url == JWKS_ENDPOINT else payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await exchange_authorization_code(
            client,
            code="c",
            verifier="v",
            redirect_uri="http://127.0.0.1:1/cb",
            issued_client_id="oaiapp_test",
            nonce="nonce-value",
        )

    assert result.bundle.access_token == access
    assert calls == [JWKS_ENDPOINT, TOKEN_ENDPOINT]


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", ["HS256", "none"])
async def test_exchange_never_downgrades_recognizable_jose_with_unsupported_alg(
    signing_material, algorithm
):
    key, jwks = signing_material
    header = (
        base64.urlsafe_b64encode(json.dumps({"alg": algorithm, "typ": "JWT"}).encode())
        .rstrip(b"=")
        .decode()
    )
    claims = base64.urlsafe_b64encode(b'{"sub":"subject-1"}').rstrip(b"=").decode()
    access = f"{header}.{claims}.signature"
    payload = _token_payload(
        key, access_token=access, id_token=_id_token(key, at_hash=_at_hash(access))
    )

    def handler(request: httpx.Request):
        return httpx.Response(200, json=jwks if request.url == JWKS_ENDPOINT else payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await exchange_authorization_code(
                client,
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )
    assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header",
    [
        {},
        {"jku": "https://evil.example/jwks"},
        {"jwk": {"kty": "RSA"}},
        {"x5c": ["certificate"]},
        {"typ": "JWT"},
        {"kid": "key-1"},
    ],
)
async def test_exchange_never_downgrades_any_json_object_header_to_opaque(
    signing_material, header
):
    key, jwks = signing_material
    header_segment = (
        base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode())
        .rstrip(b"=")
        .decode()
    )
    access = f"{header_segment}.e30.signature"
    payload = _token_payload(
        key, access_token=access, id_token=_id_token(key, at_hash=_at_hash(access))
    )

    def handler(request: httpx.Request):
        return httpx.Response(200, json=jwks if request.url == JWKS_ENDPOINT else payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await exchange_authorization_code(
                client,
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )

    assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_segment", ["not-json", "W10"])
async def test_exchange_never_downgrades_recognizable_jose_with_malformed_claims(
    signing_material, payload_segment
):
    key, jwks = signing_material
    header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
    access = f"{header}.{payload_segment}.signature"
    payload = _token_payload(
        key, access_token=access, id_token=_id_token(key, at_hash=_at_hash(access))
    )

    def handler(request: httpx.Request):
        return httpx.Response(200, json=jwks if request.url == JWKS_ENDPOINT else payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await exchange_authorization_code(
                client,
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )
    assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


@pytest.mark.asyncio
async def test_exchange_rejects_recognizable_jose_with_bad_signature_characters(signing_material):
    key, jwks = signing_material
    header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
    claims = base64.urlsafe_b64encode(b'{"sub":"subject-1"}').rstrip(b"=").decode()
    access = f"{header}.{claims}.bad!signature"
    payload = _token_payload(
        key, access_token=access, id_token=_id_token(key, at_hash=_at_hash(access))
    )

    def handler(request: httpx.Request):
        return httpx.Response(200, json=jwks if request.url == JWKS_ENDPOINT else payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await exchange_authorization_code(
                client,
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )

    assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


@pytest.mark.asyncio
async def test_exchange_prefetches_jwks_before_consuming_authorization_code(signing_material):
    key, jwks = signing_material
    calls: list[httpx.URL] = []

    def handler(request: httpx.Request):
        calls.append(request.url)
        return httpx.Response(
            200, json=jwks if request.url == JWKS_ENDPOINT else _token_payload(key)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await exchange_authorization_code(
            client,
            code="one-time-code",
            verifier="v",
            redirect_uri="http://127.0.0.1:1/cb",
            issued_client_id="oaiapp_test",
            nonce="nonce-value",
        )

    assert calls == [httpx.URL(JWKS_ENDPOINT), httpx.URL(TOKEN_ENDPOINT)]


@pytest.mark.asyncio
async def test_jwks_failure_does_not_consume_authorization_code_or_refresh_token():
    calls: list[httpx.URL] = []

    def handler(request: httpx.Request):
        calls.append(request.url)
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exchange_error:
            await exchange_authorization_code(
                client,
                code="one-time-code",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )
        with pytest.raises(OAuthLifecycleError) as refresh_error:
            await refresh_token_bundle(
                client,
                refresh_token="rotating-refresh",
                issued_client_id="oaiapp_test",
                expected_subject="subject-1",
            )

    assert exchange_error.value.retryable is True
    assert refresh_error.value.retryable is True
    assert calls == [httpx.URL(JWKS_ENDPOINT), httpx.URL(JWKS_ENDPOINT)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "jwks",
    [
        {"keys": []},
        {"keys": [{}]},
        {"keys": [{"kty": "EC", "kid": "key", "n": "AQ", "e": "AQAB"}]},
        {"keys": [{"kty": "RSA", "kid": "", "n": "AQ", "e": "AQAB"}]},
        {"keys": [{"kty": "RSA", "kid": "key", "n": "!", "e": "AQAB"}]},
        {"keys": [{"kty": "RSA", "kid": "key", "n": "AQ", "e": "AQAB", "alg": "HS256"}]},
        {"keys": [{"kty": "RSA", "kid": "key", "n": "AQ", "e": "AQAB", "use": "enc"}]},
        {"keys": [{"kty": "RSA", "kid": "key", "n": "AQ", "e": "AQAB", "key_ops": ["sign"]}]},
    ],
)
async def test_unusable_jwks_is_rejected_before_one_time_token_is_consumed(jwks):
    calls: list[httpx.URL] = []

    def handler(request: httpx.Request):
        calls.append(request.url)
        return httpx.Response(200, json=jwks)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await exchange_authorization_code(
                client,
                code="one-time-code",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )

    assert exc_info.value.code is OAuthFailureCode.MALFORMED_RESPONSE
    assert calls == [httpx.URL(JWKS_ENDPOINT)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "n,e",
    [
        (_b64uint(3), _b64uint(65537)),
        (_b64uint(1 << 2046), _b64uint(65537)),
        (_b64uint((1 << 2047) | 1), _b64uint(2)),
        (_b64uint((1 << 2047) | 1), _b64uint(1)),
        (_b64uint((1 << 2047) | 1), _b64uint((1 << 40) | 1)),
    ],
)
async def test_caller_supplied_unusable_jwks_is_rejected_before_refresh_grant(n, e):
    calls: list[httpx.URL] = []
    jwks = {"keys": [{"kty": "RSA", "kid": "key", "alg": "RS256", "use": "sig", "n": n, "e": e}]}

    def handler(request: httpx.Request):
        calls.append(request.url)
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await refresh_token_bundle(
                client,
                refresh_token="one-time-refresh",
                issued_client_id="oaiapp_test",
                expected_subject="subject-1",
                jwks=jwks,
            )

    assert exc_info.value.code is OAuthFailureCode.MALFORMED_RESPONSE
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"expires_in": float("inf")},
        {"expires_in": 0},
        {"expires_in": 10**12},
        {"earliest_refresh_at": float("nan")},
        {"earliest_refresh_at": 10**20},
        {"refresh_token_expires_at": 0},
    ],
)
async def test_exchange_rejects_nonfinite_or_implausible_expiries(signing_material, overrides):
    key, jwks = signing_material

    def handler(request: httpx.Request):
        payload = jwks if request.url == JWKS_ENDPOINT else _token_payload(key, **overrides)
        return httpx.Response(
            200, content=json.dumps(payload), headers={"content-type": "application/json"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await exchange_authorization_code(
                client,
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )
    assert exc_info.value.code is OAuthFailureCode.MALFORMED_RESPONSE
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("token_type", [None, "MAC", "bearer-token"])
async def test_exchange_requires_bearer_token_type(signing_material, token_type):
    key, jwks = signing_material

    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            json=jwks
            if request.url == JWKS_ENDPOINT
            else _token_payload(key, token_type=token_type),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError):
            await exchange_authorization_code(
                client,
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )


@pytest.mark.asyncio
async def test_exchange_accepts_bearer_token_type_case_insensitively(signing_material):
    key, jwks = signing_material

    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            json=jwks if request.url == JWKS_ENDPOINT else _token_payload(key, token_type="bEaReR"),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await exchange_authorization_code(
            client,
            code="c",
            verifier="v",
            redirect_uri="http://127.0.0.1:1/cb",
            issued_client_id="oaiapp_test",
            nonce="nonce-value",
        )
    assert result.subject == "subject-1"


@pytest.mark.asyncio
async def test_request_error_does_not_retain_token_bearing_httpx_exception_chain(signing_material):
    _, jwks = signing_material
    secret = "refresh-secret-in-request"

    def handler(request: httpx.Request):
        raise httpx.ReadError(f"failed request containing {secret}", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await refresh_token_bundle(
                client,
                refresh_token=secret,
                issued_client_id="oaiapp_test",
                expected_subject="subject-1",
                jwks=jwks,
            )
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in repr(exc_info.value)


def _production_traceback_text(exc: BaseException) -> str:
    values: list[str] = []
    traceback = exc.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_filename.endswith("app/agents/services/openai_oauth.py"):
            values.extend(repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next
    return "\n".join(values)


def _retained_secret_text(exc: BaseException) -> str:
    """Recursively inspect chained errors and production-frame object graphs."""
    seen: set[int] = set()
    rendered: list[str] = []

    def visit(value, depth=0):
        if value is None or depth > 8 or id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, (str, bytes, bytearray)):
            rendered.append(repr(value))
            return
        if isinstance(value, BaseException):
            rendered.append(repr(value))
            visit(value.__cause__, depth + 1)
            visit(value.__context__, depth + 1)
            traceback = value.__traceback__
            while traceback is not None:
                if "/app/agents/services/" in traceback.tb_frame.f_code.co_filename:
                    visit(traceback.tb_frame.f_locals, depth + 1)
                traceback = traceback.tb_next
            return
        if isinstance(value, dict):
            for key, item in value.items():
                visit(key, depth + 1)
                visit(item, depth + 1)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item, depth + 1)
            return
        if isinstance(value, (httpx.Response, httpx.Request)):
            visit(dict(value.headers), depth + 1)
            visit(value.extensions, depth + 1)
            try:
                visit(value.content, depth + 1)
            except (httpx.RequestNotRead, httpx.ResponseNotRead):
                pass
            visit(value.stream, depth + 1)
            if isinstance(value, httpx.Response):
                try:
                    visit(value.request, depth + 1)
                except RuntimeError:
                    pass
            return
        state = getattr(value, "__dict__", None)
        if isinstance(state, dict):
            visit(state, depth + 1)
        for cls in type(value).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in {"__dict__", "__weakref__"}:
                    continue
                try:
                    visit(object.__getattribute__(value, slot), depth + 1)
                except (AttributeError, TypeError):
                    pass

    visit(exc)
    return "\n".join(rendered)


@pytest.mark.asyncio
async def test_safe_oauth_boundary_preserves_lookup_semantics_without_secret_context():
    secret = "lookup-owner-secret"

    @openai_oauth.safe_oauth_boundary
    async def missing(value: str):
        raise LookupError(f"missing {value}")

    with pytest.raises(LookupError, match="^OAuth resource not found$") as exc_info:
        await missing(secret)

    assert not isinstance(exc_info.value, OAuthLifecycleError)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in _retained_secret_text(exc_info.value)


@pytest.mark.asyncio
async def test_safe_oauth_boundary_preserves_programming_defect_type_without_secret_context():
    secret = "assertion-programming-secret"

    @openai_oauth.safe_oauth_boundary
    async def broken(value: str):
        raise AssertionError(f"broken {value}")

    with pytest.raises(AssertionError, match="^OAuth operation failed safely$") as exc_info:
        await broken(secret)

    assert not isinstance(exc_info.value, OAuthLifecycleError)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in _retained_secret_text(exc_info.value)


@pytest.mark.asyncio
async def test_malformed_json_scrubs_response_request_and_traceback_secrets():
    request_secret = "request-refresh-secret"
    provider_secret = "provider-body-secret"
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request):
        response = httpx.Response(200, content=f'{{"token":"{provider_secret}"'.encode())
        responses.append(response)
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await _request(
            client,
            "POST",
            TOKEN_ENDPOINT,
            content=request_secret,
            headers={"Authorization": f"Bearer {request_secret}"},
        )
        with pytest.raises(OAuthLifecycleError) as exc_info:
            _json_object(response)

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert response.content == b""
    assert response.request.content == b""
    assert "authorization" not in response.request.headers
    retained = _production_traceback_text(exc_info.value)
    assert request_secret not in retained
    assert provider_secret not in retained


@pytest.mark.asyncio
async def test_http_failure_scrubs_provider_response_and_refresh_token_traceback(signing_material):
    _, jwks = signing_material
    refresh_secret = "rotating-refresh-secret"
    provider_secret = "provider-http-secret"
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request):
        response = httpx.Response(400, content=f'{{"detail":"{provider_secret}"'.encode())
        responses.append(response)
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await refresh_token_bundle(
                client,
                refresh_token=refresh_secret,
                issued_client_id="oaiapp_test",
                expected_subject="subject-1",
                jwks=jwks,
            )

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert responses[0].content == b""
    retained = _production_traceback_text(exc_info.value)
    assert refresh_secret not in retained
    assert provider_secret not in retained


@pytest.mark.asyncio
async def test_request_stops_streaming_at_actual_size_limit_and_closes_response():
    stream = _TrackingStream([b"a" * 500_000, b"b" * 500_000, b"c", b"must-not-read"])

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    ) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await _request(client, "GET", JWKS_ENDPOINT)

    assert exc_info.value.code is OAuthFailureCode.MALFORMED_RESPONSE
    assert stream.yielded == 3
    assert stream.closed is True


@pytest.mark.asyncio
async def test_request_rejects_oversized_content_length_without_consuming_body():
    stream = _TrackingStream([b"must-not-read"])

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-length": "1000001"},
                stream=stream,
            )
        )
    ) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await _request(client, "GET", JWKS_ENDPOINT)

    assert exc_info.value.code is OAuthFailureCode.MALFORMED_RESPONSE
    assert stream.yielded == 0
    assert stream.closed is True


@pytest.mark.asyncio
async def test_request_returns_usable_bounded_response_and_closes_source():
    stream = _TrackingStream([b'{"ok":', b"true}"])
    seen: list[httpx.Request] = []
    seen_timeouts: list[dict[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        seen_timeouts.append(dict(request.extensions["timeout"]))
        return httpx.Response(
            201,
            headers={"content-type": "application/json", "x-result": "kept"},
            stream=stream,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), timeout=httpx.Timeout(7.0, connect=2.0)
    ) as client:
        response = await _request(client, "POST", TOKEN_ENDPOINT, content=b"request-body")

    assert response.status_code == 201
    assert response.json() == {"ok": True}
    assert response.headers["x-result"] == "kept"
    assert response.request is not seen[0]
    assert response.request.content == b""
    assert "authorization" not in response.request.headers
    assert seen_timeouts[0] == {
        "connect": 5.0,
        "read": 30.0,
        "write": 30.0,
        "pool": 30.0,
    }
    assert stream.closed is True


@pytest.mark.asyncio
async def test_request_overall_timeout_is_sanitized_and_closes_response(monkeypatch):
    stream = _SlowStream()
    requested_timeouts: list[float | None] = []
    real_timeout = openai_oauth.asyncio.timeout

    def short_timeout(delay: float | None):
        requested_timeouts.append(delay)
        return real_timeout(0.001)

    monkeypatch.setattr(openai_oauth.asyncio, "timeout", short_timeout)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream))
    ) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await _request(client, "GET", JWKS_ENDPOINT)

    assert requested_timeouts == [30.0]
    assert exc_info.value.code is OAuthFailureCode.TRANSIENT
    assert exc_info.value.retryable is True
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert stream.closed is True


@pytest.mark.asyncio
async def test_request_does_not_execute_inherited_request_hook():
    secret = "hook-request-secret"
    seen: list[httpx.Request] = []

    async def explode(request: httpx.Request):
        seen.append(request)
        raise RuntimeError(f"hook failed with {secret}")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
        event_hooks={"request": [explode]},
    ) as client:
        response = await _request(
            client,
            "POST",
            TOKEN_ENDPOINT,
            content=secret,
            headers={"Authorization": f"Bearer {secret}"},
        )

    assert response.status_code == 200
    assert seen == []


@pytest.mark.asyncio
async def test_request_never_invokes_client_auth_or_sends_caller_cookies():
    class HostileAuth(httpx.Auth):
        invoked = False

        async def async_auth_flow(self, request: httpx.Request):
            self.invoked = True
            request.headers["x-auth-secret"] = "hostile-secret"
            yield request

    auth = HostileAuth()
    seen: list[tuple[str, str | None, str | None]] = []

    def handler(request: httpx.Request):
        seen.append(
            (
                request.headers["authorization"],
                request.headers.get("cookie"),
                request.headers.get("x-auth-secret"),
            )
        )
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=auth,
        cookies={"session": "oauth-cookie"},
    ) as client:
        response = await _request(
            client,
            "GET",
            MODELS_ENDPOINT,
            headers={"Authorization": "Bearer explicit-token"},
        )

    assert response.json() == {"ok": True}
    assert auth.invoked is False
    assert seen == [("Bearer explicit-token", None, None)]


@pytest.mark.asyncio
async def test_request_does_not_execute_inherited_response_hook():
    retained: list[httpx.Response] = []

    async def explode(response: httpx.Response):
        retained.append(response)
        raise RuntimeError("must not run")

    def handler(request: httpx.Request):
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), event_hooks={"response": [explode]}
    ) as client:
        response = await _request(client, "POST", TOKEN_ENDPOINT, content="secret")

    assert response.json() == {"ok": True}
    assert retained == []


@pytest.mark.asyncio
async def test_request_never_mutates_shared_hooks_during_concurrent_normal_request():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls: list[tuple[str, str]] = []

    async def first(response: httpx.Response):
        calls.append(("first", response.request.url.path))
        if response.request.url.path == "/normal":
            entered.set()
            await release.wait()

    async def second(response: httpx.Response):
        calls.append(("second", response.request.url.path))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True})),
        event_hooks={"response": [first, second]},
    ) as client:
        original_hooks = list(client.event_hooks["response"])
        normal = asyncio.create_task(client.get("https://example.test/normal"))
        await entered.wait()
        await _request(client, "GET", MODELS_ENDPOINT)
        assert client.event_hooks["response"] == original_hooks
        release.set()
        await normal

    assert calls == [
        ("first", "/normal"),
        ("second", "/normal"),
    ]


@pytest.mark.asyncio
async def test_request_hook_snapshots_are_isolated_across_clients():
    calls: list[str] = []

    async def first(response: httpx.Response):
        calls.append("first")

    async def second(response: httpx.Response):
        calls.append("second")

    async with (
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200)),
            event_hooks={"response": [first]},
        ) as first_client,
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200)),
            event_hooks={"response": [second]},
        ) as second_client,
    ):
        await asyncio.gather(
            _request(first_client, "GET", JWKS_ENDPOINT),
            _request(second_client, "GET", MODELS_ENDPOINT),
        )

    assert calls == []


@pytest.mark.asyncio
async def test_redirect_is_rejected_without_constructing_next_request(monkeypatch):
    redirect_builds = 0

    def handler(request: httpx.Request):
        return httpx.Response(307, headers={"location": TOKEN_ENDPOINT})

    def forbidden_redirect(*args, **kwargs):
        nonlocal redirect_builds
        redirect_builds += 1
        raise AssertionError("redirect request must not be constructed")

    monkeypatch.setattr(httpx.AsyncClient, "_build_redirect_request", forbidden_redirect)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await _request(
                client,
                "POST",
                TOKEN_ENDPOINT,
                content=b"original-secret",
                headers={"Authorization": "Bearer original-secret"},
            )

    assert exc_info.value.code is OAuthFailureCode.REDIRECT_REJECTED
    assert redirect_builds == 0


@pytest.mark.asyncio
async def test_hostile_user_hook_cannot_retain_oauth_response():
    retained: list[httpx.Response] = []
    later_calls = 0

    async def retain_then_fail(response: httpx.Response):
        retained.append(response)
        raise RuntimeError("user hook failed")

    async def late_capture_like_hook(response: httpx.Response):
        nonlocal later_calls
        later_calls += 1

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b"provider-secret")
        ),
        event_hooks={"response": [retain_then_fail, late_capture_like_hook]},
    ) as client:
        original_hooks = list(client.event_hooks["response"])
        response = await _request(client, "GET", JWKS_ENDPOINT)
        assert client.event_hooks["response"] == original_hooks

    assert later_calls == 0
    assert retained == []
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["request", "exchange", "refresh", "revoke", "models"])
async def test_arbitrary_base_exception_is_sanitized_at_every_public_boundary(
    operation, signing_material
):
    class ProviderAbort(BaseException):
        __slots__ = ("private_secret",)

    _, jwks = signing_material
    secret = f"{operation}-base-exception-secret"
    abort = ProviderAbort("external transport failed")
    abort.private_secret = secret

    def explode(request: httpx.Request):
        raise abort

    async with httpx.AsyncClient(transport=httpx.MockTransport(explode)) as client:
        with pytest.raises(RuntimeError, match="failed safely") as exc_info:
            if operation == "request":
                await _request(client, "GET", JWKS_ENDPOINT)
            elif operation == "exchange":
                await exchange_authorization_code(
                    client,
                    code=secret,
                    verifier=secret,
                    redirect_uri="http://127.0.0.1:1/cb",
                    issued_client_id="oaiapp_test",
                    nonce=secret,
                )
            elif operation == "refresh":
                await refresh_token_bundle(
                    client,
                    refresh_token=secret,
                    issued_client_id="oaiapp_test",
                    expected_subject="subject-1",
                    jwks=jwks,
                )
            elif operation == "revoke":
                await revoke_token(client, refresh_token=secret, issued_client_id="oaiapp_test")
            else:
                await list_models(client, secret)

    assert secret not in _retained_secret_text(exc_info.value)
    assert abort.args == ()
    assert abort.__traceback__ is None
    assert abort.private_secret is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("stream failed"), asyncio.CancelledError()])
async def test_request_scrubs_external_response_when_arbitrary_stream_failure_propagates(error):
    secret = "stream-request-secret"
    stream = _ExplodingStream(error)
    seen: list[tuple[httpx.Request, httpx.Response]] = []

    def handler(request: httpx.Request):
        response = httpx.Response(
            200,
            headers={"x-secret": "stream-response-secret"},
            extensions={"secret": "stream-extension-secret"},
            stream=stream,
        )
        seen.append((request, response))
        return response

    expected = asyncio.CancelledError if isinstance(error, asyncio.CancelledError) else RuntimeError
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(expected):
            await _request(
                client,
                "POST",
                TOKEN_ENDPOINT,
                content=secret,
                headers={"Authorization": f"Bearer {secret}"},
            )

    request, response = seen[0]
    assert request.content == b""
    assert dict(request.headers) == {}
    assert request.extensions == {}
    assert response.content == b""
    assert dict(response.headers) == {}
    assert response.extensions == {}
    assert stream.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token_changes,expected_code",
    [
        ({"id_token": "not-a-jwt"}, OAuthFailureCode.CORRUPT_TOKEN),
        ({"id_token": None}, OAuthFailureCode.MALFORMED_RESPONSE),
    ],
)
async def test_exchange_rejects_malformed_tokens(
    signing_material, token_changes, expected_code
):
    key, jwks = signing_material

    def handler(request: httpx.Request):
        return httpx.Response(
            200, json=jwks if request.url == JWKS_ENDPOINT else _token_payload(key, **token_changes)
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await exchange_authorization_code(
                client,
                code="code",
                verifier="verifier",
                redirect_uri="http://127.0.0.1:1/auth/callback",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )
    assert exc_info.value.code is expected_code
    assert "access-value" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_exchange_retains_valid_identity_when_direct_plan_scope_is_missing(signing_material):
    key, jwks = signing_material

    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            json=jwks
            if request.url == JWKS_ENDPOINT
            else _token_payload(
                key, scope="openid profile email offline_access resource.invoke"
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await exchange_authorization_code(
            client,
            code="code",
            verifier="verifier",
            redirect_uri="http://127.0.0.1:1/auth/callback",
            issued_client_id="oaiapp_test",
            nonce="nonce-value",
        )

    assert result.subject == "subject-1"
    assert result.bundle.access_token
    assert result.bundle.refresh_token == "refresh-value"
    assert result.bundle.id_token
    assert result.scopes == frozenset(
        {"openid", "profile", "email", "offline_access", "resource.invoke"}
    )


@pytest.mark.asyncio
async def test_exchange_rejects_nonce_issuer_audience_and_subject(signing_material):
    key, jwks = signing_material
    variants = [
        _id_token(key, nonce="wrong"),
        _id_token(key, iss="https://evil.example"),
        _id_token(key, client_id="another-client"),
        _id_token(key, subject=""),
    ]
    for encoded in variants:

        def handler(request: httpx.Request, encoded=encoded):
            return httpx.Response(
                200,
                json=jwks
                if request.url == JWKS_ENDPOINT
                else _token_payload(key, id_token=encoded),
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=False
        ) as client:
            with pytest.raises(OAuthLifecycleError) as exc_info:
                await exchange_authorization_code(
                    client,
                    code="code",
                    verifier="verifier",
                    redirect_uri="http://127.0.0.1:1/auth/callback",
                    issued_client_id="oaiapp_test",
                    nonce="nonce-value",
                )
        assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


@pytest.mark.asyncio
async def test_authenticated_requests_refuse_redirects(signing_material):
    _, jwks = signing_material
    refresh_secret = "redirect-refresh-secret"
    location_secret = "redirect-location-secret"
    seen: list[tuple[httpx.Request, httpx.Response]] = []

    def handler(request: httpx.Request):
        response = httpx.Response(
            302,
            headers={"location": f"https://evil.example/{location_secret}"},
            extensions={"provider_secret": location_secret},
        )
        seen.append((request, response))
        return response

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await refresh_token_bundle(
                client,
                refresh_token=refresh_secret,
                issued_client_id="oaiapp_test",
                expected_subject="subject-1",
                jwks=jwks,
            )
    assert exc_info.value.code is OAuthFailureCode.REDIRECT_REJECTED
    request, response = seen[0]
    for secret in (refresh_secret, location_secret):
        assert secret not in _retained_secret_text(exc_info.value)
        assert secret not in repr(dict(request.headers))
        assert secret not in repr(response.headers)
        assert secret not in repr(response.extensions)
    assert request.content == b""
    assert response.content == b""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal_code",
    [
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    ],
)
async def test_refresh_terminal_errors_require_reauthorization(signing_material, terminal_code):
    _, jwks = signing_material

    def handler(request: httpx.Request):
        return httpx.Response(
            400, json={"error": terminal_code, "error_description": "contains-secret"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as invalid:
            await refresh_token_bundle(
                client,
                refresh_token="refresh-value",
                issued_client_id="oaiapp_test",
                expected_subject="subject-1",
                jwks=jwks,
            )
    assert invalid.value.code is OAuthFailureCode.REAUTH_REQUIRED
    assert invalid.value.definitive is True
    assert "contains-secret" not in str(invalid.value)


@pytest.mark.asyncio
async def test_refresh_server_failure_is_transient(signing_material):
    _, jwks = signing_material
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, text="contains-secret"))
    ) as client:
        with pytest.raises(OAuthLifecycleError) as transient:
            await refresh_token_bundle(
                client,
                refresh_token="refresh-value",
                issued_client_id="oaiapp_test",
                expected_subject="subject-1",
                jwks=jwks,
            )
    assert transient.value.code is OAuthFailureCode.TRANSIENT
    assert transient.value.definitive is False
    assert "contains-secret" not in str(transient.value)


@pytest.mark.asyncio
async def test_refresh_invalid_client_is_nonretryable_but_not_definitive(signing_material):
    _, jwks = signing_material
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, json={"error": "invalid_client"})
        )
    ) as client:
        with pytest.raises(OAuthLifecycleError) as failure:
            await refresh_token_bundle(
                client,
                refresh_token="refresh-value",
                issued_client_id="oaiapp_test",
                expected_subject="subject-1",
                jwks=jwks,
            )

    assert failure.value.code is OAuthFailureCode.INVALID_CLIENT
    assert failure.value.definitive is False
    assert failure.value.retryable is False


@pytest.mark.asyncio
async def test_model_listing_parses_official_list_response_and_uses_official_endpoint():
    seen: list[httpx.Request] = []
    authorizations: list[str] = []

    def handler(request: httpx.Request):
        seen.append(request)
        authorizations.append(request.headers["authorization"])
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "gpt-5.4",
                        "display_name": "GPT 5.4",
                        "visibility": "list",
                    },
                    {
                        "slug": "internal-model",
                        "display_name": "Internal",
                        "visibility": "hidden",
                    },
                    {
                        "slug": "gpt-5-mini",
                        "display_name": "GPT 5 mini",
                        "visibility": "list",
                    },
                ],
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        models = await list_models(client, "access-value")
    assert models == [
        {"slug": "gpt-5.4", "display_name": "GPT 5.4"},
        {"slug": "gpt-5-mini", "display_name": "GPT 5 mini"},
    ]
    assert seen[0].url == MODELS_ENDPOINT
    assert authorizations == ["Bearer access-value"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["exchange", "refresh"])
async def test_successful_token_paths_scrub_bounded_responses_and_parsed_payloads(
    monkeypatch, signing_material, operation
):
    key, jwks = signing_material
    token_payload = _token_payload(key)
    jwks_payload = {"keys": [dict(jwks["keys"][0])]}
    responses: list[httpx.Response] = []

    def retained_response(payload: dict) -> httpx.Response:
        request = httpx.Request(
            "POST",
            TOKEN_ENDPOINT,
            content=b"successful-request-secret",
            headers={"Authorization": "Bearer successful-request-secret"},
            extensions={"request_secret": "successful-request-secret"},
        )
        response = httpx.Response(
            200,
            content=json.dumps(payload).encode(),
            headers={"x-secret": "successful-header-secret"},
            extensions={"response_secret": "successful-extension-secret"},
            request=request,
        )
        responses.append(response)
        return response

    async def fake_request(method, url, **kwargs):
        if url == JWKS_ENDPOINT:
            return retained_response(jwks_payload)
        return retained_response(token_payload)

    monkeypatch.setattr(openai_oauth, "_request", fake_request)
    async with httpx.AsyncClient() as client:
        if operation == "exchange":
            result = await exchange_authorization_code(
                client,
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )
        else:
            result = await refresh_token_bundle(
                client,
                refresh_token="refresh-value",
                issued_client_id="oaiapp_test",
                expected_subject="subject-1",
                jwks=jwks,
            )

    assert result.subject == "subject-1"
    for response in responses:
        assert response.content == b""
        assert dict(response.headers) == {}
        assert response.extensions == {}
        assert response.request.content == b""
        assert dict(response.request.headers) == {}
        assert response.request.extensions == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["revoke", "models"])
async def test_successful_non_token_paths_scrub_bounded_response_and_payload(
    monkeypatch, operation
):
    payload = (
        {
            "models": [
                {"slug": "gpt-5.4", "display_name": "GPT 5.4", "visibility": "list"}
            ]
        }
        if operation == "models"
        else {"ignored": "successful-provider-secret"}
    )
    request = httpx.Request(
        "POST",
        REVOKE_ENDPOINT,
        content=b"successful-request-secret",
        headers={"Authorization": "Bearer successful-request-secret"},
        extensions={"request_secret": "successful-request-secret"},
    )
    response = httpx.Response(
        200,
        content=(
            json.dumps(payload).encode()
            if operation == "models"
            else b"successful-provider-secret"
        ),
        headers={"x-secret": "successful-header-secret"},
        extensions={"response_secret": "successful-extension-secret"},
        request=request,
    )

    async def fake_request(method, url, **kwargs):
        return response

    monkeypatch.setattr(openai_oauth, "_request", fake_request)
    async with httpx.AsyncClient() as client:
        if operation == "models":
            result = await list_models(client, "access-value")
            assert result == [{"slug": "gpt-5.4", "display_name": "GPT 5.4"}]
        else:
            await revoke_token(
                client, refresh_token="refresh-value", issued_client_id="oaiapp_test"
            )

    assert response.content == b""
    assert dict(response.headers) == {}
    assert response.extensions == {}
    assert request.content == b""
    assert dict(request.headers) == {}
    assert request.extensions == {}


@pytest.mark.asyncio
async def test_model_listing_rejects_malformed_payload_and_redirect():
    responses = [
        httpx.Response(200, json={"models": "wrong"}),
        httpx.Response(307, headers={"location": RESOURCE}),
    ]

    def handler(request: httpx.Request):
        return responses.pop(0)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        with pytest.raises(OAuthLifecycleError) as malformed:
            await list_models(client, "access-value")
        with pytest.raises(OAuthLifecycleError) as redirect:
            await list_models(client, "access-value")
    assert malformed.value.code is OAuthFailureCode.MALFORMED_RESPONSE
    assert redirect.value.code is OAuthFailureCode.REDIRECT_REJECTED


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_error", ["invalid_token", "invalid_grant"])
async def test_revocation_uses_official_endpoint_and_treats_invalid_token_as_idempotent(
    provider_error,
):
    seen: list[httpx.Request] = []
    bodies: list[str] = []

    def handler(request: httpx.Request):
        seen.append(request)
        bodies.append(request.content.decode())
        return httpx.Response(400, json={"error": provider_error})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        await revoke_token(client, refresh_token="refresh-value", issued_client_id="oaiapp_test")
    assert seen[0].url == REVOKE_ENDPOINT
    body = bodies[0]
    assert "token_type_hint=refresh_token" in body
    assert "client_id=oaiapp_test" in body


@pytest.mark.asyncio
async def test_exchange_validates_access_token_times_independently_and_uses_earlier_expiry(
    signing_material,
):
    key, jwks = signing_material
    now = datetime.now(timezone.utc).replace(microsecond=0)
    access = _access_token(
        key,
        iat=int((now - timedelta(minutes=2)).timestamp()),
        exp=int((now + timedelta(minutes=7)).timestamp()),
    )
    payload = _token_payload(
        key,
        access_token=access,
        id_token=_id_token(
            key,
            iat=int((now - timedelta(minutes=1)).timestamp()),
            exp=int((now + timedelta(minutes=9)).timestamp()),
            at_hash=_at_hash(access),
        ),
        expires_in=3600,
    )

    def handler(request: httpx.Request):
        return httpx.Response(200, json=jwks if request.url == JWKS_ENDPOINT else payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await exchange_authorization_code(
            client,
            code="c",
            verifier="v",
            redirect_uri="http://127.0.0.1:1/cb",
            issued_client_id="oaiapp_test",
            nonce="nonce-value",
            receipt_clock=lambda: now,
        )

    assert result.access_token_expires_at == now + timedelta(minutes=7)


def test_id_token_rejects_mismatched_azp_with_scalar_audience(signing_material):
    key, jwks = signing_material
    access = _access_token(key)
    encoded = _id_token(key, azp="another-client", at_hash=_at_hash(access))

    with pytest.raises(OAuthLifecycleError) as exc_info:
        validate_id_token(
            encoded,
            jwks=jwks,
            issued_client_id="oaiapp_test",
            access_token=access,
        )

    assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


@pytest.mark.asyncio
async def test_opaque_access_token_expiry_uses_expires_in(signing_material):
    key, jwks = signing_material
    now = datetime.now(timezone.utc).replace(microsecond=0)
    access = "opaque-access-token"
    payload = _token_payload(
        key,
        access_token=access,
        id_token=_id_token(key, at_hash=_at_hash(access)),
        expires_in=90,
    )

    def handler(request: httpx.Request):
        return httpx.Response(200, json=jwks if request.url == JWKS_ENDPOINT else payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await exchange_authorization_code(
            client,
            code="c",
            verifier="v",
            redirect_uri="http://127.0.0.1:1/cb",
            issued_client_id="oaiapp_test",
            nonce="nonce-value",
            receipt_clock=lambda: now,
        )

    assert result.access_token_expires_at == now + timedelta(seconds=90)


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [True, False])
async def test_exchange_rejects_oversized_http_response_before_json_decoding(
    signing_material, declared
):
    key, _ = signing_material
    oversized = b"{" + b" " * 1_000_000 + b"}"

    def handler(request: httpx.Request):
        headers = {"content-length": "1000001"} if declared else {}
        content = b"{}" if declared else oversized
        return httpx.Response(200, content=content, headers=headers)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await exchange_authorization_code(
                client,
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )

    assert exc_info.value.code is OAuthFailureCode.MALFORMED_RESPONSE
    assert exc_info.value.__cause__ is None


def test_response_size_limit_is_checked_before_json_decoding(monkeypatch):
    response = httpx.Response(200, content=b"{}", headers={"content-length": "1000001"})

    def must_not_decode():
        raise AssertionError("JSON decoder must not run for oversized responses")

    monkeypatch.setattr(response, "json", must_not_decode)
    with pytest.raises(OAuthLifecycleError) as exc_info:
        _json_object(response)
    assert exc_info.value.code is OAuthFailureCode.MALFORMED_RESPONSE


@pytest.mark.asyncio
async def test_exchange_rejects_oversized_token_and_jwks_collections(signing_material):
    key, jwks = signing_material
    responses = [
        _token_payload(key, refresh_token="r" * 65_537),
        {"keys": jwks["keys"] * 33},
    ]

    def handler(request: httpx.Request):
        return httpx.Response(200, json=responses.pop(0))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await exchange_authorization_code(
                client,
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )
    assert exc_info.value.code is OAuthFailureCode.MALFORMED_RESPONSE

    responses[:] = [_token_payload(key), {"keys": jwks["keys"] * 33}]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await exchange_authorization_code(
                client,
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )
    assert exc_info.value.code is OAuthFailureCode.MALFORMED_RESPONSE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "models",
    [
        [
            {"slug": f"model-{index}", "display_name": "Model", "visibility": "list"}
            for index in range(501)
        ],
        [{"slug": "s" * 257, "display_name": "Model", "visibility": "list"}],
        [{"slug": "", "display_name": "Model", "visibility": "list"}],
        [{"slug": 123, "display_name": "Model", "visibility": "list"}],
        [{"slug": "model", "display_name": "", "visibility": "list"}],
        [{"slug": "model", "display_name": "D" * 257, "visibility": "list"}],
    ],
)
async def test_model_listing_bounds_collection_and_fields(models):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"models": models})
        )
    ) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await list_models(client, "access-value")
    assert exc_info.value.code is OAuthFailureCode.MALFORMED_RESPONSE


@pytest.mark.asyncio
async def test_model_listing_rejects_generic_api_model_list_shape():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"object": "collection", "data": [{"id": "gpt-5.4"}]}
            )
        )
    ) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await list_models(client, "access-value")
    assert exc_info.value.code is OAuthFailureCode.MALFORMED_RESPONSE


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_model_auth_error_recursively_scrubs_response_and_request(status):
    access_secret = f"model-access-{status}-secret"
    provider_secret = f"model-provider-{status}-secret"
    response_header_secret = f"model-header-{status}-secret"
    seen: list[tuple[httpx.Request, httpx.Response]] = []

    def handler(request: httpx.Request):
        response = httpx.Response(
            status,
            content=provider_secret,
            headers={"x-provider-secret": response_header_secret},
            extensions={"provider_secret": provider_secret},
        )
        seen.append((request, response))
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await list_models(client, access_secret)

    request, response = seen[0]
    for secret in (access_secret, provider_secret, response_header_secret):
        assert secret not in _retained_secret_text(exc_info.value)
        assert secret not in repr(dict(request.headers))
        assert secret not in repr(request.extensions)
        assert secret not in repr(response.headers)
        assert secret not in repr(response.extensions)
    assert request.content == b""
    assert response.content == b""


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [httpx.ReadError, httpx.ReadTimeout])
async def test_transport_or_timeout_error_recursively_discards_request_secret(error_type):
    secret = f"recursive-{error_type.__name__}-request-secret"
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request):
        seen.append(request)
        raise error_type("transport failed", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await _request(
                client,
                "POST",
                TOKEN_ENDPOINT,
                content=secret,
                headers={"Authorization": f"Bearer {secret}"},
            )

    assert secret not in _retained_secret_text(exc_info.value)
    assert secret not in repr(dict(seen[0].headers))
    assert seen[0].content == b""


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["exchange", "refresh", "revoke"])
async def test_public_oauth_failures_recursively_discard_all_provider_secrets(
    signing_material, operation
):
    _, jwks = signing_material
    request_secret = f"{operation}-request-secret"
    provider_secret = f"{operation}-provider-secret"
    header_secret = f"{operation}-header-secret"
    seen: list[tuple[httpx.Request, httpx.Response]] = []

    def handler(request: httpx.Request):
        if request.url == JWKS_ENDPOINT:
            return httpx.Response(200, json=jwks)
        response = httpx.Response(
            400,
            content=f'{{"error":"unknown","detail":"{provider_secret}"}}',
            headers={"x-provider-secret": header_secret},
            extensions={"provider_secret": provider_secret},
        )
        seen.append((request, response))
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            if operation == "exchange":
                await exchange_authorization_code(
                    client,
                    code=request_secret,
                    verifier=f"{request_secret}-verifier",
                    redirect_uri="http://127.0.0.1:1/cb",
                    issued_client_id="oaiapp_test",
                    nonce=f"{request_secret}-nonce",
                )
            elif operation == "refresh":
                await refresh_token_bundle(
                    client,
                    refresh_token=request_secret,
                    issued_client_id="oaiapp_test",
                    expected_subject="subject-1",
                    jwks=jwks,
                )
            else:
                await revoke_token(
                    client, refresh_token=request_secret, issued_client_id="oaiapp_test"
                )

    request, response = seen[0]
    for secret in (request_secret, provider_secret, header_secret):
        assert secret not in _retained_secret_text(exc_info.value)
        assert secret not in repr(dict(request.headers))
        assert secret not in repr(request.extensions)
        assert secret not in repr(response.headers)
        assert secret not in repr(response.extensions)
    assert request.content == b""
    assert response.content == b""
    assert set(vars(exc_info.value)) == {"code", "definitive", "retryable"}


def test_jwt_validation_failure_recursively_discards_encoded_secret(signing_material):
    _, jwks = signing_material
    header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
    encoded_secret = f"{header}.unique-secret-claims.signature"

    with pytest.raises(OAuthLifecycleError) as exc_info:
        validate_id_token(
            encoded_secret,
            jwks=jwks,
            issued_client_id="oaiapp_test",
            access_token="unique-bound-access-secret",
        )

    retained = _retained_secret_text(exc_info.value)
    assert encoded_secret not in retained
    assert "unique-bound-access-secret" not in retained
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.asyncio
async def test_request_isolates_caller_defaults_cookies_and_transport_lifetime():
    seen: list[tuple[httpx.URL, httpx.Headers]] = []

    def handler(request: httpx.Request):
        seen.append((request.url, httpx.Headers(request.headers)))
        return httpx.Response(200, headers={"set-cookie": "provider=secret; Path=/"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        transport=transport,
        headers={"x-caller-default": "must-not-leak"},
        cookies={"caller": "must-not-leak"},
    ) as client:
        await _request(
            client,
            "GET",
            MODELS_ENDPOINT,
            headers={"x-explicit": "preserved"},
        )
        assert client.cookies.get("provider") is None
        await client.get("https://ordinary.example/check")

    oauth_headers = seen[0][1]
    assert oauth_headers["x-explicit"] == "preserved"
    assert "x-caller-default" not in oauth_headers
    assert "cookie" not in oauth_headers
    assert seen[1][0] == "https://ordinary.example/check"


@pytest.mark.asyncio
async def test_request_rejects_untrusted_origin_before_transport_dispatch():
    calls = 0

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="failed safely"):
            await _request(client, "GET", "https://auth.openai.com.evil.test/token")

    assert calls == 0


@pytest.mark.parametrize(
    "explicit_expiry, expected_delta",
    [
        (None, timedelta(days=30)),
        (timedelta(days=10), timedelta(days=10)),
        (timedelta(days=31), timedelta(days=30)),
    ],
)
def test_rotating_refresh_token_gets_30_day_expiry_or_earlier_explicit_value(
    signing_material, explicit_expiry, expected_delta
):
    key, jwks = signing_material
    now = datetime.now(timezone.utc).replace(microsecond=0)
    payload = _token_payload(key)
    if explicit_expiry is not None:
        payload["refresh_token_expires_at"] = (now + explicit_expiry).timestamp()

    result = openai_oauth._validated_response(
        payload,
        jwks=jwks,
        issued_client_id="oaiapp_test",
        nonce="nonce-value",
        now=now,
    )

    assert result.refresh_token_expires_at == now + expected_delta


@pytest.mark.parametrize("email", ["x" * 321, "not-an-email", "bad email@example.com"])
def test_optional_invalid_account_email_is_omitted_from_valid_token_bundle(
    signing_material, email
):
    key, jwks = signing_material
    now = datetime.now(timezone.utc).replace(microsecond=0)
    payload = _token_payload(key, id_token=_id_token(key, email=email))

    result = openai_oauth._validated_response(
        payload,
        jwks=jwks,
        issued_client_id="oaiapp_test",
        nonce="nonce-value",
        now=now,
    )

    assert result.account_email is None


@pytest.mark.parametrize(
    "headers",
    [
        {"kid": "test-key", "crit": ["unsupported"], "unsupported": True},
        {},
        {"kid": "k" * 513},
    ],
)
def test_id_token_rejects_unsupported_critical_headers_and_invalid_kid(
    signing_material, headers
):
    key, jwks = signing_material
    access = _access_token(key)
    claims = {
        "iss": ISSUER,
        "aud": "oaiapp_test",
        "sub": "subject-1",
        "iat": int((datetime.now(timezone.utc) - timedelta(minutes=1)).timestamp()),
        "exp": int((datetime.now(timezone.utc) + timedelta(minutes=5)).timestamp()),
        "at_hash": _at_hash(access),
    }
    encoded = jwt.encode(claims, key, algorithm="RS256", headers=headers)

    with pytest.raises(OAuthLifecycleError) as exc_info:
        validate_id_token(
            encoded,
            jwks=jwks,
            issued_client_id="oaiapp_test",
            access_token=access,
        )

    assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


@pytest.mark.parametrize(
    "jwk_change",
    [
        {"use": "enc"},
        {"key_ops": ["sign"]},
    ],
)
def test_public_id_token_validation_rejects_unsuitable_supplied_jwk(
    signing_material, jwk_change
):
    key, jwks = signing_material
    access = _access_token(key)
    encoded = _id_token(key, at_hash=_at_hash(access))
    unsuitable = {"keys": [{**jwks["keys"][0], **jwk_change}]}

    with pytest.raises(OAuthLifecycleError):
        validate_id_token(
            encoded,
            jwks=unsuitable,
            issued_client_id="oaiapp_test",
            access_token=access,
        )


@pytest.mark.parametrize(
    "headers",
    [
        {"kid": "test-key", "crit": ["unsupported"], "unsupported": True},
        {},
    ],
)
def test_signed_access_token_rejects_unsupported_critical_headers_and_missing_kid(
    signing_material, headers
):
    key, jwks = signing_material
    now = datetime.now(timezone.utc).replace(microsecond=0)
    encoded = jwt.encode(
        {
            "iss": ISSUER,
            "aud": RESOURCE,
            "sub": "subject-1",
            "iat": int((now - timedelta(minutes=1)).timestamp()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
        },
        key,
        algorithm="RS256",
        headers=headers,
    )

    with pytest.raises(OAuthLifecycleError) as exc_info:
        openai_oauth._validate_access_token(
            encoded,
            jwks=jwks,
            identity_claims={"sub": "subject-1"},
            issued_client_id="oaiapp_test",
            granted_scopes=REQUIRED_SCOPES,
            now=now,
        )

    assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["exchange", "refresh"])
async def test_token_timing_uses_exact_post_response_receipt_time(
    signing_material, operation
):
    assert "now" not in inspect.signature(_exchange_authorization_code).parameters
    assert "now" not in inspect.signature(_refresh_token_bundle).parameters
    key, jwks = signing_material
    received_at = datetime.now(timezone.utc).replace(microsecond=0)
    attempt_started_at = received_at - timedelta(minutes=7)
    access = "opaque-access-token"
    response_complete = False
    clock_calls = 0

    def receipt_clock():
        nonlocal clock_calls
        assert response_complete
        clock_calls += 1
        return received_at
    payload = _token_payload(
        key,
        access_token=access,
        id_token=_id_token(key, at_hash=_at_hash(access)),
        expires_in=120,
        earliest_refresh_at=int((received_at + timedelta(seconds=30)).timestamp()),
    )

    def handler(request: httpx.Request):
        nonlocal response_complete
        if request.url == JWKS_ENDPOINT:
            return httpx.Response(200, json=jwks)
        response_complete = True
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        if operation == "exchange":
            result = await exchange_authorization_code(
                client,
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
                receipt_clock=receipt_clock,
            )
        else:
            result = await refresh_token_bundle(
                client,
                refresh_token="refresh-value",
                issued_client_id="oaiapp_test",
                expected_subject="subject-1",
                jwks=jwks,
                receipt_clock=receipt_clock,
            )

    assert clock_calls == 1
    assert result.received_at == received_at
    assert result.access_token_expires_at == received_at + timedelta(seconds=120)
    assert result.access_token_expires_at != attempt_started_at + timedelta(seconds=120)
    assert result.earliest_refresh_at == received_at + timedelta(seconds=30)
    assert result.refresh_token_expires_at == received_at + timedelta(days=30)


@pytest.mark.asyncio
@pytest.mark.parametrize("document", ["token", "jwks"])
async def test_provider_json_rejects_duplicate_members(signing_material, document):
    key, jwks = signing_material
    valid_token = json.dumps(_token_payload(key), separators=(",", ":"))
    duplicate_token = valid_token[:-1] + ',"access_token":"attacker"}'
    valid_jwk = json.dumps(jwks["keys"][0], separators=(",", ":"))
    duplicate_jwks = '{"keys":[' + valid_jwk[:-1] + ',"kid":"attacker"}]}'

    def handler(request: httpx.Request):
        body = duplicate_jwks if document == "jwks" else (
            json.dumps(jwks) if request.url == JWKS_ENDPOINT else duplicate_token
        )
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(OAuthLifecycleError) as exc_info:
            await exchange_authorization_code(
                client,
                code="c",
                verifier="v",
                redirect_uri="http://127.0.0.1:1/cb",
                issued_client_id="oaiapp_test",
                nonce="nonce-value",
            )
    assert exc_info.value.code in {
        OAuthFailureCode.MALFORMED_RESPONSE,
        OAuthFailureCode.CORRUPT_TOKEN,
    }


@pytest.mark.parametrize("duplicate", ["header_alg", "header_kid", "claims_sub"])
def test_signed_jwt_rejects_duplicate_security_members(signing_material, duplicate):
    key, jwks = signing_material
    now = datetime.now(timezone.utc)
    header = '{"alg":"RS256","kid":"test-key"}'
    claims = json.dumps(
        {
            "iss": ISSUER,
            "aud": "oaiapp_test",
            "sub": "subject-1",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=10)).timestamp()),
        },
        separators=(",", ":"),
    )
    if duplicate == "header_alg":
        header = '{"alg":"HS256","alg":"RS256","kid":"test-key"}'
    elif duplicate == "header_kid":
        header = '{"alg":"RS256","kid":"attacker","kid":"test-key"}'
    else:
        claims = claims[:-1] + ',"sub":"subject-1"}'
    encoded = _signed_raw_jwt(key, header, claims)

    with pytest.raises(OAuthLifecycleError) as exc_info:
        validate_id_token(encoded, jwks=jwks, issued_client_id="oaiapp_test", now=now)
    assert exc_info.value.code is OAuthFailureCode.CORRUPT_TOKEN


@pytest.mark.asyncio
async def test_hostile_readonly_transport_cannot_retain_credentials(monkeypatch):
    class ReadonlyAbort(BaseException):
        __slots__ = ("_copied",)

        def __init__(self, copied):
            self._copied = copied

        @property
        def copied(self):
            return self._copied

    class HostileTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.invoked = False
            self.error = ReadonlyAbort(None)

        async def handle_async_request(self, request):
            self.invoked = True
            self.error = ReadonlyAbort(await request.aread())
            raise self.error

    hostile = HostileTransport()
    internal = httpx.MockTransport(lambda request: httpx.Response(200, content=b"{}"))
    monkeypatch.setattr(openai_oauth, "_new_oauth_transport", lambda: internal)

    async with httpx.AsyncClient(transport=hostile) as client:
        response = await _request(
            client,
            "POST",
            TOKEN_ENDPOINT,
            content="readonly-external-sentinel-secret",
        )

    assert response.status_code == 200
    assert hostile.invoked is False
    assert hostile.error.copied is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://***@auth.openai.com/api/accounts/oauth/token",
        "https://@auth.openai.com/api/accounts/oauth/token",
        "https://:@auth.openai.com/api/accounts/oauth/token",
        "https://attacker:secret@api.openai.com/v1/models",
        "https://auth.openai.com/api/accounts/oauth/token#",
        "https://auth.openai.com/api/accounts/oauth/token#fragment",
        "https://auth.openai.com./api/accounts/oauth/token",
        "https://auth.openai.com.:443/api/accounts/oauth/token",
        "https://auth.openai.com:443/api/accounts/oauth/token",
        "https://auth.openai.com:0/api/accounts/oauth/token",
        "https://auth.openai.com:00/api/accounts/oauth/token",
        "https://auth.openai.com:444/api/accounts/oauth/token",
        "http://auth.openai.com/api/accounts/oauth/token",
        "HTTPS://auth.openai.com/api/accounts/oauth/token",
        "https://AUTH.OPENAI.COM/api/accounts/oauth/token",
        "https://auth.openai.com/api/accounts/oauth/token/",
        "https://auth.openai.com/api/accounts/oauth/%74oken",
        "https://auth.openai.com/.well-known/jwks.json?ignored=1",
    ],
)
async def test_trusted_host_userinfo_is_rejected_before_transport_selection(
    url, monkeypatch
):
    constructed = 0

    def transport_factory(*args, **kwargs):
        nonlocal constructed
        constructed += 1
        return httpx.MockTransport(lambda request: httpx.Response(200))

    monkeypatch.setattr(openai_oauth.httpx, "AsyncHTTPTransport", transport_factory)
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError, match="failed safely"):
            await _request(client, "GET", url)

    assert constructed == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "url"),
    [
        ("GET", TOKEN_ENDPOINT),
        ("POST", JWKS_ENDPOINT),
        ("POST", MODELS_ENDPOINT),
        ("GET", REVOKE_ENDPOINT),
    ],
)
async def test_request_rejects_wrong_method_for_canonical_endpoint_before_dispatch(
    method, url, monkeypatch
):
    calls = 0

    def transport_factory(*args, **kwargs):
        nonlocal calls
        calls += 1
        return httpx.MockTransport(lambda request: httpx.Response(200))

    monkeypatch.setattr(openai_oauth.httpx, "AsyncHTTPTransport", transport_factory)
    async with httpx.AsyncClient() as _client:
        with pytest.raises(RuntimeError, match="failed safely"):
            await _request_impl(method, url)

    assert calls == 0


def test_public_oauth_signatures_have_no_transport_injection():
    from app.agents.services import connection_service, oauth_refresh

    public_functions = (
        _exchange_authorization_code,
        _refresh_token_bundle,
        _revoke_token,
        _list_models,
        oauth_refresh.refresh_credential,
        oauth_refresh.revoke_credential,
        connection_service.test_connection,
    )
    for function in public_functions:
        parameters = inspect.signature(function).parameters
        assert "_test_transport" not in parameters
        assert "protocol_transport" not in parameters


@pytest.mark.asyncio
async def test_mock_transport_cannot_be_supplied_through_public_oauth_api(monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(200))
    with pytest.raises(TypeError):
        inspect.signature(_exchange_authorization_code).bind(
            object(),
            code="code",
            verifier="verifier",
            redirect_uri="http://127.0.0.1/cb",
            issued_client_id="client",
            nonce="nonce",
            _test_transport=transport,
        )

    factory_calls = 0

    def factory():
        nonlocal factory_calls
        factory_calls += 1
        return transport

    monkeypatch.setattr(openai_oauth, "_new_oauth_transport", factory)
    async with httpx.AsyncClient() as client:
        with pytest.raises(RuntimeError, match="OAuth operation failed safely"):
            await _exchange_authorization_code(
                client,
                code="code",
                verifier="verifier",
                redirect_uri="http://127.0.0.1/cb",
                issued_client_id="client",
                nonce="nonce",
                _test_transport=transport,  # type: ignore[call-arg]
            )

    assert factory_calls == 0


def test_owned_transport_ignores_hostile_proxy_and_ca_environment(monkeypatch):
    from app.agents.config import AgentSettings

    captured = {}
    ssl_context = object()
    default_paths = type(
        "DefaultPaths",
        (),
        {"openssl_cafile": "/system/ca.pem", "openssl_capath": "/system/certs"},
    )()
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.invalid:8080")
    monkeypatch.setenv("SSL_CERT_FILE", "/attacker/ca.pem")
    monkeypatch.setenv("SSL_CERT_DIR", "/attacker/certs")
    monkeypatch.setattr(openai_oauth, "get_agent_settings", lambda: AgentSettings())
    monkeypatch.setattr(openai_oauth.ssl, "get_default_verify_paths", lambda: default_paths)
    monkeypatch.setattr(
        openai_oauth.ssl,
        "create_default_context",
        lambda **kwargs: captured.update(kwargs) or ssl_context,
    )

    def transport_factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(openai_oauth.httpx, "AsyncHTTPTransport", transport_factory)

    assert _new_oauth_transport_impl() is not None
    assert captured["trust_env"] is False
    assert captured["proxy"] is None
    assert captured["cafile"] == "/system/ca.pem"
    assert captured["capath"] == "/system/certs"
    assert captured["verify"] is ssl_context


def test_owned_transport_uses_only_explicit_proxy_and_ca_settings(monkeypatch, tmp_path):
    from app.agents.config import AgentSettings

    captured = {}
    ssl_context = object()
    default_paths = type(
        "DefaultPaths",
        (),
        {"openssl_cafile": "/system/ca.pem", "openssl_capath": "/system/certs"},
    )()
    ca_bundle = tmp_path / "ca.pem"
    ca_bundle.write_text("-----BEGIN CERTIFICATE-----\nAA==\n-----END CERTIFICATE-----\n")
    settings = AgentSettings(
        siwc_oauth_proxy_url="https://trusted-proxy.example:8443",
        siwc_oauth_ca_bundle=str(ca_bundle),
    )
    monkeypatch.setattr(openai_oauth, "get_agent_settings", lambda: settings)
    monkeypatch.setattr(openai_oauth.ssl, "get_default_verify_paths", lambda: default_paths)
    monkeypatch.setattr(
        openai_oauth.ssl,
        "create_default_context",
        lambda **kwargs: captured.update(kwargs) or ssl_context,
    )

    def transport_factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(openai_oauth.httpx, "AsyncHTTPTransport", transport_factory)

    _new_oauth_transport_impl()
    assert captured["proxy"] == "https://trusted-proxy.example:8443"
    assert captured["cafile"] == str(ca_bundle)
    assert captured["capath"] is None
    assert captured["trust_env"] is False


@pytest.mark.parametrize(
    "proxy_url",
    [
        "socks5://proxy.example:1080",
        "http://user:password@proxy.example:8080",
        "http://@proxy.example:8080",
        "http://proxy.example:0",
        "http://proxy.example:99999",
        "http://proxy.example/path",
        "http://proxy.example?query=1",
        "http://proxy.example#fragment",
    ],
)
def test_explicit_oauth_proxy_setting_rejects_unsafe_urls(proxy_url):
    from pydantic import ValidationError

    from app.agents.config import AgentSettings

    with pytest.raises(ValidationError):
        AgentSettings(siwc_oauth_proxy_url=proxy_url)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel", "soft_limit"])
async def test_owned_transport_closes_on_every_request_outcome(outcome, monkeypatch):
    class TrackingTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.closed = False
            self.limit = SoftTimeLimitExceeded("transport-close-sentinel")

        async def handle_async_request(self, request):
            if outcome == "failure":
                raise httpx.ReadError("failed", request=request)
            if outcome == "cancel":
                raise asyncio.CancelledError()
            if outcome == "soft_limit":
                raise self.limit
            return httpx.Response(200, content=b"{}")

        async def aclose(self):
            self.closed = True

    transport = TrackingTransport()
    monkeypatch.setattr(openai_oauth, "_new_oauth_transport", lambda: transport)
    async with httpx.AsyncClient() as _caller:
        if outcome == "success":
            response = await _request_impl("GET", JWKS_ENDPOINT)
            assert response.status_code == 200
        elif outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await _request_impl("GET", JWKS_ENDPOINT)
        elif outcome == "soft_limit":
            with pytest.raises(SoftTimeLimitExceeded) as exc_info:
                await _request_impl("GET", JWKS_ENDPOINT)
            assert exc_info.value is transport.limit
        else:
            with pytest.raises(OAuthLifecycleError):
                await _request_impl("GET", JWKS_ENDPOINT)

    assert transport.closed is True


@pytest.mark.asyncio
async def test_owned_transport_closes_if_owned_client_construction_fails(monkeypatch):
    class TrackingTransport(httpx.AsyncBaseTransport):
        def __init__(self):
            self.closed = False

        async def handle_async_request(self, request):  # pragma: no cover
            raise AssertionError("request must not be dispatched")

        async def aclose(self):
            self.closed = True

    transport = TrackingTransport()
    caller = httpx.AsyncClient()
    request = httpx.Request("GET", JWKS_ENDPOINT)
    monkeypatch.setattr(openai_oauth, "_new_oauth_transport", lambda: transport)

    def fail_client_construction(**kwargs):
        raise RuntimeError("construction failed")

    monkeypatch.setattr(openai_oauth.httpx, "AsyncClient", fail_client_construction)
    try:
        with pytest.raises(RuntimeError, match="construction failed"):
            await openai_oauth._send_protocol_request(request)
    finally:
        await caller.aclose()

    assert transport.closed is True


def test_request_and_response_extension_graphs_are_recursively_scrubbed():
    request_secret = bytearray(b"nested-request-extension-secret")
    response_secret = bytearray(b"nested-response-extension-secret")
    retained_request: dict[str, Any] = {"secret": request_secret}
    retained_response: dict[str, Any] = {"secret": response_secret}
    retained_request["cycle"] = retained_response
    retained_response["cycle"] = retained_request
    request = httpx.Request(
        "GET", TOKEN_ENDPOINT, extensions={"nested": retained_request}
    )
    history_request = httpx.Request(
        "GET", TOKEN_ENDPOINT, extensions={"nested": retained_request}
    )
    history = httpx.Response(
        200,
        request=history_request,
        extensions={"nested": retained_response},
    )
    next_request = httpx.Request(
        "GET", TOKEN_ENDPOINT, extensions={"nested": retained_request}
    )
    response = httpx.Response(
        200,
        request=request,
        extensions={"nested": retained_response},
        history=[history],
    )
    response.next_request = next_request

    openai_oauth._scrub_response(response)

    assert request_secret == bytearray()
    assert response_secret == bytearray()
    assert retained_request == {}
    assert retained_response == {}
    assert request.extensions == {}
    assert history.extensions == {}
    assert history_request.extensions == {}
    assert next_request.extensions == {}


@pytest.mark.asyncio
async def test_request_scrubs_nested_cyclic_source_and_bounded_extensions():
    sentinel = bytearray(b"source-bounded-extension-sentinel")
    retained: dict[str, Any] = {"sentinel": sentinel}
    retained["cycle"] = retained
    history_request = httpx.Request(
        "GET", TOKEN_ENDPOINT, extensions={"nested": retained}
    )
    history = httpx.Response(
        200,
        request=history_request,
        extensions={"nested": retained},
    )
    next_request = httpx.Request(
        "GET", TOKEN_ENDPOINT, extensions={"nested": retained}
    )
    sources: list[httpx.Response] = []

    def handler(request):
        source = httpx.Response(
            200,
            content=b"{}",
            extensions={"nested": retained},
            history=[history],
        )
        source.next_request = next_request
        sources.append(source)
        return source

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        bounded = await _request(client, "GET", JWKS_ENDPOINT)

    assert sentinel == bytearray()
    assert retained == {}
    assert sources[0].extensions == {}
    assert history.extensions == {}
    assert history_request.extensions == {}
    assert next_request.extensions == {}
    openai_oauth._scrub_response(bounded)
    assert bounded.extensions == {}


@pytest.mark.asyncio
async def test_caller_transport_never_receives_credential_request(monkeypatch):
    copied: list[bytes] = []

    class HostileTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            copied.append(await request.aread())
            raise AssertionError("caller transport was invoked")

    internal_calls = 0

    def internal_factory(*args, **kwargs):
        nonlocal internal_calls

        def handler(request):
            nonlocal internal_calls
            internal_calls += 1
            return httpx.Response(200, content=b"{}")

        return httpx.MockTransport(handler)

    monkeypatch.setattr(openai_oauth.httpx, "AsyncHTTPTransport", internal_factory)
    async with httpx.AsyncClient(transport=HostileTransport()) as client:
        response = await _request(
            client,
            "POST",
            TOKEN_ENDPOINT,
            content="credential-sentinel",
        )

    assert response.status_code == 200
    assert copied == []
    assert internal_calls == 1


@pytest.mark.asyncio
async def test_soft_time_limit_propagates_unchanged_from_provider_io():
    limit = SoftTimeLimitExceeded()

    def handler(request):
        raise limit

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SoftTimeLimitExceeded) as exc_info:
            await _request(
                client,
                "POST",
                TOKEN_ENDPOINT,
                content="credential-sentinel",
            )

    assert exc_info.value is limit


@pytest.mark.asyncio
async def test_soft_time_limit_from_provider_io_is_scrubbed_in_place():
    sentinel = "soft-limit-provider-sentinel"
    retained_request = httpx.Request(
        "POST",
        TOKEN_ENDPOINT,
        headers={"authorization": f"Bearer {sentinel}"},
        content=sentinel,
    )
    retained_response = httpx.Response(
        503,
        request=retained_request,
        headers={"x-secret": sentinel},
        content=sentinel,
        extensions={"nested": {"secret": bytearray(sentinel.encode())}},
    )

    class SlottedSoftLimit(SoftTimeLimitExceeded):
        __slots__ = ("retained",)

    limit = SlottedSoftLimit(sentinel)
    limit.retained = {"response": retained_response}
    try:
        raise RuntimeError(sentinel)
    except RuntimeError as cause:
        limit.__cause__ = cause

    def handler(request):
        raise limit

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SoftTimeLimitExceeded) as exc_info:
            await _request(client, "POST", TOKEN_ENDPOINT, content=sentinel)

    assert exc_info.value is limit
    assert limit.args == ()
    assert limit.__cause__ is None
    assert limit.__context__ is None
    assert limit.retained is None
    assert sentinel not in _retained_secret_text(limit)
    assert retained_request.content == b""
    assert retained_response.content == b""


@pytest.mark.asyncio
async def test_safe_boundaries_propagate_soft_time_limit_unchanged():
    async_limit = SoftTimeLimitExceeded()
    sync_limit = SoftTimeLimitExceeded()

    @openai_oauth.safe_oauth_boundary
    async def async_stop():
        raise async_limit

    @openai_oauth._safe_oauth_sync_boundary
    def sync_stop():
        raise sync_limit

    with pytest.raises(SoftTimeLimitExceeded) as async_info:
        await async_stop()
    with pytest.raises(SoftTimeLimitExceeded) as sync_info:
        sync_stop()

    assert async_info.value is async_limit
    assert sync_info.value is sync_limit


@pytest.mark.parametrize("claim,value", [
    ("client_id", None), ("client_id", "another-client"),
    ("scope", None), ("scope", " ".join(sorted(REQUIRED_SCOPES)) + " extra"),
    ("nbf", None), ("nbf", 9999999999), ("jti", ""), ("jti", "x" * 513),
    ("https://api.openai.com/auth", None),
    ("https://api.openai.com/auth", {"per_user_salt": "", "encrypted_auth_metadata": "data"}),
])
def test_signed_access_requires_documented_binding(signing_material, claim, value):
    key, jwks = signing_material
    access = _access_token(key, **{claim: value})
    payload = _token_payload(key, access_token=access, id_token=_id_token(key, at_hash=_at_hash(access)))
    with pytest.raises(OAuthLifecycleError):
        openai_oauth._validated_response(payload, jwks=jwks, issued_client_id="oaiapp_test", now=datetime.now(timezone.utc))


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Protocol tests use synthetic HTTP/JWT inputs and require no database."""
    yield


@pytest.mark.parametrize("ignored", [float("nan"), float("inf"), {"nested": [float("-inf")]}, {1: "value"}, {"s": "x" * 65537}, object()])
def test_jwks_rejects_invalid_ignored_graph(signing_material, ignored):
    _, jwks = signing_material
    supplied = {"keys": jwks["keys"], "ignored": ignored}
    with pytest.raises(OAuthLifecycleError):
        openai_oauth._validated_jwks(supplied)


def test_opaque_access_requires_verified_id_token_hash_binding(signing_material):
    key, jwks = signing_material
    payload = _token_payload(key, access_token="opaque-access", id_token=_id_token(key))
    with pytest.raises(OAuthLifecycleError) as error:
        openai_oauth._validated_response(payload, jwks=jwks, issued_client_id="oaiapp_test", now=datetime.now(timezone.utc))
    assert error.value.code is OAuthFailureCode.CORRUPT_TOKEN
