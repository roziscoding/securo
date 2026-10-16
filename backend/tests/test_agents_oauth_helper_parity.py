"""Shared table-driven contract vectors for helper/backend OAuth parity."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from jose import jwt

sys.path.insert(0, str(Path(__file__).parents[2] / "tools/securo-siwc/src"))
from securo_siwc import openai as helper_oauth  # noqa: E402

from app.agents.services import openai_oauth as backend_oauth  # noqa: E402


@pytest.fixture(scope="module")
def oauth_contract_bundle() -> tuple[dict[str, Any], dict[str, Any], str, str, datetime, rsa.RSAPrivateKey]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public = key.public_key().public_numbers()

    def integer(value: int) -> str:
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    jwks = {"keys": [{
        "kty": "RSA", "kid": "contract", "alg": "RS256", "use": "sig",
        "n": integer(public.n), "e": integer(public.e),
    }]}
    client_id = "issued-client"
    nonce = "n" * 43
    now_value = int(time.time())
    now = datetime.fromtimestamp(now_value, tz=timezone.utc)
    access_claims = {
        "iss": helper_oauth.ISSUER,
        "aud": helper_oauth.RESOURCE,
        "sub": "subject",
        "client_id": client_id,
        "scope": helper_oauth.SCOPES,
        "iat": now_value,
        "nbf": now_value,
        "exp": now_value + 3600,
        "jti": "contract-jti",
        "https://api.openai.com/auth": {
            "per_user_salt": "salt",
            "encrypted_auth_metadata": "metadata",
        },
    }
    access = jwt.encode(access_claims, private, algorithm="RS256", headers={"kid": "contract"})
    at_hash = base64.urlsafe_b64encode(
        hashlib.sha256(access.encode()).digest()[:16]
    ).rstrip(b"=").decode()
    identity_claims = {
        "iss": helper_oauth.ISSUER,
        "aud": client_id,
        "sub": "subject",
        "nonce": nonce,
        "iat": now_value,
        "exp": now_value + 3600,
        "at_hash": at_hash,
        "email": "account@example.com",
    }
    identity = jwt.encode(identity_claims, private, algorithm="RS256", headers={"kid": "contract"})
    response = {
        "access_token": access,
        "refresh_token": "refresh-token",
        "id_token": identity,
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": helper_oauth.SCOPES,
        "earliest_refresh_at": now_value + 3000,
        "refresh_token_expires_at": now_value + 86400,
    }
    return response, jwks, client_id, nonce, now, key


def _resign(response: dict[str, Any], context: dict[str, Any]) -> None:
    private = context["private"]
    access = jwt.encode(
        context["access_claims"], private, algorithm="RS256", headers={"kid": "contract"}
    )
    response["access_token"] = access
    identity = context["identity_claims"]
    identity["at_hash"] = base64.urlsafe_b64encode(
        hashlib.sha256(access.encode()).digest()[:16]
    ).rstrip(b"=").decode()
    response["id_token"] = jwt.encode(
        identity, private, algorithm="RS256", headers={"kid": "contract"}
    )


def _raw_token(header: bytes, claims: bytes, key: rsa.RSAPrivateKey) -> str:
    def encode(value: bytes) -> bytes:
        return base64.urlsafe_b64encode(value).rstrip(b"=")

    signing_input = encode(header) + b"." + encode(claims)
    signature = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return (signing_input + b"." + encode(signature)).decode()


def _valid(_response: dict[str, Any], _context: dict[str, Any], _key: rsa.RSAPrivateKey) -> None:
    return


def _too_many_scopes(response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey) -> None:
    scopes = helper_oauth.SCOPES.split() + [f"extra-{index}" for index in range(59)]
    response["scope"] = " ".join(scopes)
    context["access_claims"]["scope"] = response["scope"]
    _resign(response, context)


def _oversized_access_claim_scope(
    response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey
) -> None:
    context["access_claims"]["scope"] = helper_oauth.SCOPES + (" " * 9000)
    _resign(response, context)


def _repeated_whitespace_access_claim_scope(
    response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey
) -> None:
    context["access_claims"]["scope"] = "  \t".join(
        helper_oauth.SCOPES.split() + helper_oauth.SCOPES.split()
    )
    _resign(response, context)


def _long_subject(response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey) -> None:
    context["access_claims"]["sub"] = "s" * 256
    context["identity_claims"]["sub"] = "s" * 256
    _resign(response, context)


def _long_lifetime(response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey) -> None:
    context["identity_claims"]["exp"] = context["identity_claims"]["iat"] + 86401
    _resign(response, context)


def _long_jti(response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey) -> None:
    context["access_claims"]["jti"] = "j" * 513
    _resign(response, context)


def _long_metadata(response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey) -> None:
    context["access_claims"]["https://api.openai.com/auth"]["per_user_salt"] = "s" * 8193
    _resign(response, context)


def _boolean_nbf(response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey) -> None:
    context["access_claims"]["nbf"] = True
    _resign(response, context)


def _duplicate_alg(response: dict[str, Any], context: dict[str, Any], key: rsa.RSAPrivateKey) -> None:
    claims = json.dumps(context["identity_claims"], separators=(",", ":")).encode()
    response["id_token"] = _raw_token(
        b'{"alg":"RS256","alg":"RS256","kid":"contract","typ":"JWT"}', claims, key
    )


def _duplicate_subject(response: dict[str, Any], context: dict[str, Any], key: rsa.RSAPrivateKey) -> None:
    claims = json.dumps(context["identity_claims"], separators=(",", ":")).encode()
    claims = claims[:-1] + b',"sub":"subject"}'
    response["id_token"] = _raw_token(
        b'{"alg":"RS256","kid":"contract","typ":"JWT"}', claims, key
    )


def _set_raw_access(
    response: dict[str, Any], context: dict[str, Any], key: rsa.RSAPrivateKey,
    *, header: bytes, claims: bytes,
) -> None:
    access = _raw_token(header, claims, key)
    response["access_token"] = access
    context["identity_claims"]["at_hash"] = base64.urlsafe_b64encode(
        hashlib.sha256(access.encode()).digest()[:16]
    ).rstrip(b"=").decode()
    response["id_token"] = jwt.encode(
        context["identity_claims"], context["private"],
        algorithm="RS256", headers={"kid": "contract"},
    )


def _duplicate_access_alg(response: dict[str, Any], context: dict[str, Any], key: rsa.RSAPrivateKey) -> None:
    claims = json.dumps(context["access_claims"], separators=(",", ":")).encode()
    _set_raw_access(
        response, context, key,
        header=b'{"alg":"RS256","alg":"RS256","kid":"contract","typ":"JWT"}',
        claims=claims,
    )


def _duplicate_access_claim(response: dict[str, Any], context: dict[str, Any], key: rsa.RSAPrivateKey) -> None:
    claims = json.dumps(context["access_claims"], separators=(",", ":")).encode()
    _set_raw_access(
        response, context, key,
        header=b'{"alg":"RS256","kid":"contract","typ":"JWT"}',
        claims=claims[:-1] + b',"jti":"contract-jti"}',
    )


def _wrong_resource(response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey) -> None:
    context["access_claims"]["aud"] = "https://api.openai.com/wrong"
    _resign(response, context)


def _invalid_azp(response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey) -> None:
    context["identity_claims"]["azp"] = 7
    _resign(response, context)


def _invalid_nonce(response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey) -> None:
    context["identity_claims"]["nonce"] = 7
    _resign(response, context)


def _invalid_refresh_timestamp(
    response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey
) -> None:
    response["refresh_token_expires_at"] = context["identity_claims"]["iat"] + 367 * 86400


def _invalid_account_type(
    response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey
) -> None:
    context["identity_claims"]["email"] = {"not": "an-email"}
    _resign(response, context)


def _invalid_audience_type(
    response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey
) -> None:
    context["identity_claims"]["aud"] = {"client": "issued-client"}
    _resign(response, context)


def _multiple_audiences_without_azp(
    response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey
) -> None:
    context["identity_claims"]["aud"] = ["issued-client", "other-client"]
    _resign(response, context)


def _long_scope_member(
    response: dict[str, Any], context: dict[str, Any], _key: rsa.RSAPrivateKey
) -> None:
    response["scope"] += " " + "s" * 257
    context["access_claims"]["scope"] = response["scope"]
    _resign(response, context)


def _boolean_expiry(
    response: dict[str, Any], _context: dict[str, Any], _key: rsa.RSAPrivateKey
) -> None:
    response["expires_in"] = True


def _unknown_response_member(
    response: dict[str, Any], _context: dict[str, Any], _key: rsa.RSAPrivateKey
) -> None:
    response["unknown"] = "value"


CONTRACT_VECTORS: list[tuple[str, bool, Callable[[dict[str, Any], dict[str, Any], rsa.RSAPrivateKey], None]]] = [
    ("valid", True, _valid),
    ("more_than_64_scopes", False, _too_many_scopes),
    ("access_claim_scope_over_8192", False, _oversized_access_claim_scope),
    ("access_claim_repeated_whitespace_scopes", True, _repeated_whitespace_access_claim_scope),
    ("subject_over_255", False, _long_subject),
    ("signed_lifetime_over_24_hours", False, _long_lifetime),
    ("jti_over_512", False, _long_jti),
    ("auth_metadata_over_8192", False, _long_metadata),
    ("boolean_not_before", False, _boolean_nbf),
    ("duplicate_alg", False, _duplicate_alg),
    ("duplicate_subject", False, _duplicate_subject),
    ("duplicate_access_alg", False, _duplicate_access_alg),
    ("duplicate_access_claim", False, _duplicate_access_claim),
    ("wrong_resource", False, _wrong_resource),
    ("invalid_authorized_party_type", False, _invalid_azp),
    ("invalid_nonce_type", False, _invalid_nonce),
    ("refresh_timestamp_over_366_days", False, _invalid_refresh_timestamp),
    ("invalid_account_is_safely_ignored", True, _invalid_account_type),
    ("invalid_audience_type", False, _invalid_audience_type),
    ("multiple_audiences_without_azp", False, _multiple_audiences_without_azp),
    ("scope_member_over_256", False, _long_scope_member),
    ("boolean_token_expiry", False, _boolean_expiry),
    ("unknown_token_response_member", False, _unknown_response_member),
]


@pytest.mark.parametrize(("_name", "accepted", "mutate"), CONTRACT_VECTORS)
def test_helper_and_backend_share_token_contract(
    _name: str,
    accepted: bool,
    mutate: Callable[[dict[str, Any], dict[str, Any], rsa.RSAPrivateKey], None],
    oauth_contract_bundle,
):
    original, jwks, client_id, nonce, now, key = oauth_contract_bundle
    response = copy.deepcopy(original)
    private = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    context = {
        "private": private,
        "access_claims": copy.deepcopy(jwt.get_unverified_claims(original["access_token"])),
        "identity_claims": copy.deepcopy(jwt.get_unverified_claims(original["id_token"])),
    }
    mutate(response, context, key)

    helper_ok = True
    try:
        helper_oauth.validate_token_response(
            copy.deepcopy(response), jwks=copy.deepcopy(jwks), client_id=client_id, nonce=nonce
        )
    except (ValueError, RuntimeError):
        helper_ok = False

    backend_ok = True
    try:
        backend_oauth._validated_response_impl(
            copy.deepcopy(response),
            jwks=copy.deepcopy(jwks),
            issued_client_id=client_id,
            nonce=nonce,
            now=now,
        )
    except backend_oauth.OAuthLifecycleError:
        backend_ok = False

    assert helper_ok is backend_ok is accepted


@pytest.mark.asyncio
@pytest.mark.parametrize("duplicate_kind", ["usable", "mixed", "unrelated"])
async def test_backend_rejects_duplicate_presented_kids_before_token_endpoint(
    duplicate_kind: str, oauth_contract_bundle, monkeypatch
):
    _response, original_jwks, client_id, nonce, _now, _key = oauth_contract_bundle
    usable = copy.deepcopy(original_jwks["keys"][0])
    if duplicate_kind == "usable":
        keys = [usable, copy.deepcopy(usable)]
    elif duplicate_kind == "mixed":
        keys = [usable, {"kty": "EC", "kid": usable["kid"], "crv": "P-256"}]
    else:
        keys = [usable, {"kty": "EC", "kid": "unrelated"}, {"kty": "oct", "kid": "unrelated"}]

    calls: list[str] = []

    async def request(method: str, url: str, **_kwargs: Any):
        calls.append(url)
        if url == backend_oauth.JWKS_ENDPOINT:
            return __import__("httpx").Response(
                200,
                json={"keys": keys},
                request=__import__("httpx").Request(method, url),
            )
        pytest.fail("duplicate JWKS kid reached the token endpoint")

    monkeypatch.setattr(backend_oauth, "_request", request)
    with pytest.raises(RuntimeError):
        await backend_oauth.exchange_authorization_code(
            code="one-use-code",
            verifier="v" * 64,
            redirect_uri="http://127.0.0.1:43111/auth/callback",
            issued_client_id=client_id,
            nonce=nonce,
        )
    assert calls == [backend_oauth.JWKS_ENDPOINT]
