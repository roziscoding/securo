"""Fixed-endpoint OpenAI OAuth protocol for the standalone helper."""
from __future__ import annotations

import base64
import hashlib
import json
import math
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import httpx
from jose import JWTError, jwk, jwt

from .security import _scrub_response, secret_boundary

ISSUER = "https://auth.openai.com"
AUTHORIZE_ENDPOINT = f"{ISSUER}/api/accounts/authorize"
TOKEN_ENDPOINT = f"{ISSUER}/api/accounts/oauth/token"
JWKS_ENDPOINT = f"{ISSUER}/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
REQUIRED = frozenset(SCOPES.split())
MAX_TOKEN = 65_536
MAX_RESPONSE = 1_000_000
MAX_JWKS_KEYS = 32
MAX_SCOPE_LENGTH = 8_192
MAX_SIGNED_TOKEN_LIFETIME = 24 * 60 * 60
_BASE64URL = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")


def _new_http_client() -> httpx.Client:
    """Build the production-owned transport boundary."""
    return httpx.Client(
        timeout=httpx.Timeout(30, connect=5),
        follow_redirects=False,
        trust_env=False,
        cookies=None,
        auth=None,
        event_hooks={},
    )


@dataclass(frozen=True)
class Authorization:
    state: str
    nonce: str
    verifier: str
    redirect_uri: str
    url: str


def create_authorization(
    port: int,
    ext_agent_host_id: str,
    returning_client_id: str | None = None,
    *,
    state: str | None = None,
) -> Authorization:
    try:
        host_uuid = __import__("uuid").UUID(ext_agent_host_id.removeprefix("urn:uuid:"))
    except (ValueError, AttributeError):
        raise ValueError("backend host identity is invalid") from None
    if ext_agent_host_id != f"urn:uuid:{host_uuid}":
        raise ValueError("backend host identity is invalid")
    if state is None:
        state = secrets.token_urlsafe(32)
    elif not state or len(state) > 256 or not state.isascii():
        raise ValueError("OAuth state is invalid")
    nonce = secrets.token_urlsafe(32)
    verifier = secrets.token_urlsafe(64)
    redirect_uri = f"http://127.0.0.1:{port}/auth/callback"
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    if returning_client_id is not None and (
        not returning_client_id
        or len(returning_client_id) > 255
        or not returning_client_id.isascii()
        or any(ord(character) <= 32 or ord(character) == 127 for character in returning_client_id)
        or returning_client_id == "dynamic_agent_client"
    ):
        raise ValueError("returning client identifier is invalid")
    dynamic = returning_client_id is None
    params = {
        "client_id": "dynamic_agent_client" if dynamic else returning_client_id,
        "ext_agent_host_id": ext_agent_host_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": SCOPES,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    if dynamic:
        params["agent_name_hint"] = "Securo"
    return Authorization(state, nonce, verifier, redirect_uri, AUTHORIZE_ENDPOINT + "?" + urlencode(params))


class OpenAIClient:
    def __init__(self) -> None:
        self.client = _new_http_client()

    def close(self) -> None:
        self.client.close()

    @secret_boundary("OpenAI request failed safely")
    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        allowed = {
            ("POST", TOKEN_ENDPOINT),
            ("GET", JWKS_ENDPOINT),
        }
        if (method, url) not in allowed:
            raise RuntimeError("untrusted OpenAI endpoint")
        raw = bytearray()
        source: httpx.Response | None = None
        try:
            headers = dict(kwargs.pop("headers", {}))
            headers["Accept-Encoding"] = "identity"
            if kwargs.keys() - {"content"}:
                raise RuntimeError("unsupported OpenAI request option")
            with self.client.stream(method, url, headers=headers, **kwargs) as source:
                self.client.cookies.clear()
                if source.is_redirect:
                    raise RuntimeError("OpenAI redirect rejected")
                declared = source.headers.get_list("content-length")
                if len(declared) > 1:
                    raise RuntimeError("OpenAI response size is invalid")
                if declared:
                    try:
                        length = int(declared[0])
                    except ValueError:
                        raise RuntimeError("OpenAI response size is invalid") from None
                    if length < 0 or length > MAX_RESPONSE:
                        raise RuntimeError("OpenAI response too large")
                if source.headers.get("content-encoding", "identity").casefold() not in {
                    "", "identity"
                }:
                    raise RuntimeError("OpenAI compressed response rejected")
                for chunk in source.iter_raw(chunk_size=min(MAX_RESPONSE + 1, 65_536)):
                    remaining = MAX_RESPONSE + 1 - len(raw)
                    raw.extend(chunk[:remaining])
                    if len(raw) > MAX_RESPONSE:
                        raise RuntimeError("OpenAI response too large")
                return httpx.Response(
                    source.status_code,
                    headers=source.headers,
                    content=bytes(raw),
                    request=httpx.Request(method, url),
                )
        except httpx.HTTPError:
            raise RuntimeError("OpenAI is unreachable") from None
        finally:
            if source is not None:
                _scrub_response(source)
            raw.clear()
            kwargs.clear()

    @secret_boundary("OpenAI token exchange failed safely")
    def exchange(self, authorization: Authorization, code: str, client_id: str) -> dict[str, Any]:
        if not code or len(code) > 8192 or not client_id or client_id == "dynamic_agent_client":
            raise ValueError("OAuth callback is invalid")
        jwks_response = self._request("GET", JWKS_ENDPOINT)
        try:
            if jwks_response.status_code != 200:
                raise RuntimeError("OpenAI signing keys unavailable")
            jwks = _validated_jwks(_json(jwks_response.content))
        finally:
            jwks_response.close()
        response = self._request(
            "POST",
            TOKEN_ENDPOINT,
            content=urlencode({
                "grant_type": "authorization_code",
                "client_id": client_id,
                "code": code,
                "code_verifier": authorization.verifier,
                "redirect_uri": authorization.redirect_uri,
                "resource": RESOURCE,
            }),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            if response.status_code != 200:
                raise RuntimeError("OpenAI token exchange failed")
            payload = _json(response.content)
        finally:
            response.close()
        if not isinstance(payload, dict):
            raise RuntimeError("OpenAI token response is malformed")
        validate_token_response(payload, jwks=jwks, client_id=client_id, nonce=authorization.nonce)
        return payload


def validate_token_response(
    payload: dict[str, Any], *, jwks: Any, client_id: str, nonce: str
) -> dict[str, Any]:
    allowed = {
        "access_token", "refresh_token", "id_token", "token_type", "expires_in",
        "scope", "earliest_refresh_at", "refresh_token_expires_at",
    }
    if type(payload) is not dict or set(payload) - allowed:
        raise ValueError("token response has unknown members")
    _validate_json_graph(payload, maximum_string_total=MAX_RESPONSE)
    if (
        not isinstance(client_id, str)
        or not client_id
        or client_id == "dynamic_agent_client"
        or len(client_id) > 255
        or not client_id.isascii()
        or any(ord(character) <= 32 or ord(character) == 127 for character in client_id)
    ):
        raise ValueError("client identifier is invalid")
    if type(nonce) is not str or not nonce or len(nonce) > 256:
        raise ValueError("nonce is invalid")
    for field in ("access_token", "refresh_token", "id_token"):
        value = payload.get(field)
        if type(value) is not str or not value or len(value) > MAX_TOKEN:
            raise ValueError("token response is incomplete")
    token_type = payload.get("token_type")
    if type(token_type) is not str or token_type.casefold() != "bearer":
        raise ValueError("token type is invalid")
    expires = payload.get("expires_in")
    if isinstance(expires, bool) or not isinstance(expires, (int, float)) or not math.isfinite(expires) or not 0 < expires <= 86400:
        raise ValueError("token expiry is invalid")
    scopes = payload.get("scope")
    if type(scopes) is not str or len(scopes) > MAX_SCOPE_LENGTH:
        raise ValueError("required scopes were not granted")
    granted = frozenset(scopes.split())
    if (
        len(granted) > 64
        or any(len(scope) > 256 for scope in granted)
        or not REQUIRED.issubset(granted)
    ):
        raise ValueError("required scopes were not granted")
    now = time.time()
    _optional_timestamp(
        payload.get("earliest_refresh_at"), minimum=now - 86400, maximum=now + float(expires)
    )
    _optional_timestamp(
        payload.get("refresh_token_expires_at"), minimum=now, maximum=now + 366 * 86400
    )
    if not isinstance(jwks, dict) or not isinstance(jwks.get("keys"), list) or not jwks["keys"]:
        raise ValueError("signing keys are invalid")
    strict_claims = _strict_jose_segment(payload["id_token"], 1)
    selected = _selected_keys(payload["id_token"], jwks)
    try:
        claims = jwt.decode(
            payload["id_token"], selected, algorithms=["RS256"], issuer=ISSUER,
            audience=client_id, access_token=payload["access_token"],
            options={"require_iss": True, "require_aud": True, "require_sub": True,
                     "require_iat": True, "require_exp": True},
        )
    except (JWTError, TypeError, ValueError):
        raise ValueError("ID token validation failed") from None
    if claims != strict_claims:
        raise ValueError("ID token claims are ambiguous")
    claims = strict_claims
    _validate_signed_times(claims, now=now)
    subject = claims.get("sub")
    if type(subject) is not str or not subject or len(subject) > 255:
        raise ValueError("ID token subject is invalid")
    claim_nonce = claims.get("nonce")
    if type(claim_nonce) is not str or not secrets.compare_digest(claim_nonce, nonce):
        raise ValueError("ID token nonce mismatch")
    aud = claims.get("aud")
    azp = claims.get("azp")
    _validate_audience(aud, client_id)
    if (
        (azp is not None and (type(azp) is not str or len(azp) > 255 or azp != client_id))
        or (type(aud) is list and len(aud) > 1 and azp != client_id)
    ):
        raise ValueError("ID token client binding failed")
    _validate_access_token(
        payload["access_token"], jwks=jwks, identity=claims,
        client_id=client_id, scopes=granted, now=now,
    )
    return claims


def _selected_keys(encoded: str, jwks: dict[str, Any]) -> dict[str, Any]:
    try:
        header = _strict_jose_segment(encoded, 0)
    except (JWTError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("token header is invalid") from None
    if not isinstance(header, dict) or header.get("alg") != "RS256" or header.get("crit") not in (None, []):
        raise ValueError("token algorithm is invalid")
    kid = header.get("kid")
    keys = jwks.get("keys")
    if not isinstance(kid, str) or not isinstance(keys, list):
        raise ValueError("token signing key is invalid")
    matching = [key for key in keys if isinstance(key, dict) and key.get("kid") == kid]
    if len(matching) != 1:
        raise ValueError("token signing key is invalid")
    return {"keys": matching}


def _decode_jwk_integer(value: Any) -> int:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 16_384
        or any(character not in _BASE64URL for character in value)
    ):
        raise ValueError("signing key integer is invalid")
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, TypeError):
        raise ValueError("signing key integer is invalid") from None
    if (
        not raw
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != value
        or (len(raw) > 1 and raw[0] == 0)
    ):
        raise ValueError("signing key integer is invalid")
    return int.from_bytes(raw, "big")


def _validated_jwks(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"keys"}:
        raise ValueError("signing keys are invalid")
    keys = value["keys"]
    if not isinstance(keys, list) or not keys or len(keys) > MAX_JWKS_KEYS:
        raise ValueError("signing keys are invalid")
    frozen: list[dict[str, Any]] = []
    kids: set[str] = set()
    for key in keys:
        if not isinstance(key, dict) or not key or len(key) > 32:
            raise ValueError("signing keys are invalid")
        kid = key.get("kid")
        if not isinstance(kid, str) or not kid or len(kid) > 512 or kid in kids:
            raise ValueError("signing keys are invalid")
        kids.add(kid)
        if key.get("kty") != "RSA":
            continue
        if key.get("alg") not in (None, "RS256") or key.get("use") not in (None, "sig"):
            continue
        operations = key.get("key_ops")
        if operations is not None and (
            not isinstance(operations, list)
            or len(operations) > 16
            or "verify" not in operations
            or any(not isinstance(operation, str) or len(operation) > 64 for operation in operations)
        ):
            continue
        modulus = _decode_jwk_integer(key.get("n"))
        exponent = _decode_jwk_integer(key.get("e"))
        if (
            modulus.bit_length() < 2048
            or modulus % 2 != 1
            or exponent < 3
            or exponent > 0xFFFFFFFF
            or exponent % 2 != 1
        ):
            continue
        candidate = _json(json.dumps(key, separators=(",", ":")).encode())
        if not isinstance(candidate, dict):
            raise ValueError("signing keys are invalid")
        try:
            jwk.construct(candidate, algorithm="RS256")
        except (JWTError, TypeError, ValueError):
            continue
        frozen.append(candidate)
    if not frozen:
        raise ValueError("signing keys are invalid")
    return {"keys": frozen}


def _validate_access_token(
    encoded: str,
    *,
    jwks: dict[str, Any],
    identity: dict[str, Any],
    client_id: str,
    scopes: frozenset[str],
    now: float | None = None,
) -> None:
    if not isinstance(encoded, str) or not encoded or len(encoded) > MAX_TOKEN:
        raise ValueError("access token is invalid")
    segments = encoded.split(".")
    if len(segments) != 3:
        # Opaque tokens are cryptographically bound by ID-token at_hash above.
        if not isinstance(identity.get("at_hash"), str):
            raise ValueError("opaque access token is not bound")
        return
    try:
        header_segment = segments[0]
        if not header_segment or any(character not in _BASE64URL for character in header_segment):
            raise ValueError
        header_raw = base64.urlsafe_b64decode(header_segment + "=" * (-len(header_segment) % 4))
        recognizable = json.loads(header_raw)
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        if not isinstance(identity.get("at_hash"), str):
            raise ValueError("opaque access token is not bound") from None
        return
    if not isinstance(recognizable, dict):
        if not isinstance(identity.get("at_hash"), str):
            raise ValueError("opaque access token is not bound")
        return
    try:
        header = _json(header_raw)
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("access token header is invalid") from None
    if type(header) is not dict:
        raise ValueError("access token header is invalid")
    if not all(part and all(character in _BASE64URL for character in part) for part in segments):
        raise ValueError("access token encoding is invalid")
    strict_claims = _strict_jose_segment(encoded, 1)
    selected = _selected_keys(encoded, jwks)
    try:
        claims = jwt.decode(
            encoded, selected, algorithms=["RS256"], issuer=ISSUER, audience=RESOURCE,
            options={"require_iss": True, "require_aud": True, "require_sub": True,
                     "require_iat": True, "require_exp": True},
        )
    except (JWTError, TypeError, ValueError):
        raise ValueError("access token validation failed") from None
    if claims != strict_claims:
        raise ValueError("access token claims are ambiguous")
    claims = strict_claims
    _validate_signed_times(claims, now=now or time.time())
    _validate_audience(claims.get("aud"), RESOURCE)
    raw_scope = claims.get("scope")
    metadata = claims.get("https://api.openai.com/auth")
    claim_scopes = (
        frozenset(raw_scope.split())
        if type(raw_scope) is str and len(raw_scope) <= MAX_SCOPE_LENGTH
        else None
    )
    if (
        claims.get("sub") != identity.get("sub")
        or claims.get("client_id") != client_id
        or claim_scopes is None
        or len(claim_scopes) > 64
        or any(len(scope) > 256 for scope in claim_scopes)
        or claim_scopes != scopes
        or type(claims.get("jti")) is not str
        or not claims["jti"].strip()
        or len(claims["jti"]) > 512
        or not isinstance(metadata, dict)
        or not all(
            type(metadata.get(field)) is str
            and metadata[field].strip()
            and len(metadata[field]) <= 8192
            for field in ("per_user_salt", "encrypted_auth_metadata")
        )
    ):
        raise ValueError("access token claims are invalid")
    nbf = claims.get("nbf")
    if (
        isinstance(nbf, bool)
        or not isinstance(nbf, (int, float))
        or not math.isfinite(float(nbf))
        or nbf > (now or time.time()) + 60
        or nbf < claims.get("iat", 0) - 60
        or nbf >= claims.get("exp", 0)
    ):
        raise ValueError("access token claims are invalid")


def _validate_signed_times(claims: dict[str, Any], *, now: float) -> None:
    issued = claims.get("iat")
    expires = claims.get("exp")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (issued, expires)
    ):
        raise ValueError("signed token times are invalid")
    assert isinstance(issued, (int, float))
    assert isinstance(expires, (int, float))
    if (
        issued > now + 60
        or expires <= issued
        or expires <= now
        or expires - issued > MAX_SIGNED_TOKEN_LIFETIME
    ):
        raise ValueError("signed token times are invalid")


def _validate_audience(value: Any, expected: str) -> None:
    if type(value) is str:
        audiences = [value]
    elif type(value) is list:
        audiences = value
    else:
        raise ValueError("token audience is invalid")
    if (
        not audiences
        or len(audiences) > 16
        or any(type(audience) is not str or not audience or len(audience) > 512 for audience in audiences)
        or expected not in audiences
        or len(set(audiences)) != len(audiences)
    ):
        raise ValueError("token audience is invalid")


def _optional_timestamp(value: Any, *, minimum: float, maximum: float) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < minimum
        or value > maximum
    ):
        raise ValueError("token response timestamp is invalid")


def _validate_json_graph(value: Any, *, maximum_string_total: int) -> None:
    pending = [(value, 0)]
    nodes = 0
    string_units = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 4096 or depth > 16:
            raise ValueError("JSON graph exceeds bounds")
        if item is None or type(item) is bool:
            continue
        if type(item) is str:
            string_units += len(item)
            if len(item) > MAX_TOKEN or string_units > maximum_string_total:
                raise ValueError("JSON string exceeds bounds")
        elif type(item) is int:
            if item.bit_length() > 1024:
                raise ValueError("JSON integer exceeds bounds")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite JSON number")
        elif type(item) is list:
            if len(item) > 4096:
                raise ValueError("JSON array exceeds bounds")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is dict:
            if len(item) > 128 or any(type(key) is not str or len(key) > 512 for key in item):
                raise ValueError("JSON object keys exceed bounds")
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        else:
            raise ValueError("unsupported JSON value")


def _strict_jose_segment(encoded: str, index: int) -> dict[str, Any]:
    if type(encoded) is not str or len(encoded) > MAX_TOKEN:
        raise ValueError("token encoding is invalid")
    segments = encoded.split(".")
    segment = segments[index] if len(segments) == 3 else ""
    if not segment or len(segment) > MAX_TOKEN or any(character not in _BASE64URL for character in segment):
        raise ValueError("token encoding is invalid")
    raw = base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))
    if base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != segment:
        raise ValueError("token encoding is noncanonical")
    parsed = _json(raw)
    if type(parsed) is not dict:
        raise ValueError("token object is invalid")
    return parsed


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def _json(raw: bytes) -> Any:
    if len(raw) > MAX_RESPONSE:
        raise ValueError("JSON too large")
    value = json.loads(
        raw,
        object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
    )
    _validate_json_graph(value, maximum_string_total=MAX_RESPONSE)
    return value
