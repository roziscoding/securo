"""OpenAI Sign in with ChatGPT OAuth protocol primitives.

This module is intentionally limited to documented OpenAI endpoints. Exceptions
contain stable metadata only: provider bodies and all credential material stay
out of logs and API errors.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
import json
import logging
import math
import re
import ssl
import traceback as traceback_module
import types
from typing import Any, Callable
from urllib.parse import urlencode

import httpx
from billiard.exceptions import SoftTimeLimitExceeded
from jose import JWTError, jwt

from app.agents.services.oauth_credentials import CredentialBundle, sanitize_account_email
from app.agents.config import get_agent_settings

ISSUER = "https://auth.openai.com"
TOKEN_ENDPOINT = f"{ISSUER}/api/accounts/oauth/token"
REVOKE_ENDPOINT = f"{ISSUER}/api/accounts/oauth/revoke"
JWKS_ENDPOINT = f"{ISSUER}/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
MODELS_ENDPOINT = f"{RESOURCE}/models"
REQUIRED_SCOPES = frozenset(
    {"openid", "profile", "email", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"}
)
BASE_REQUIRED_SCOPES = REQUIRED_SCOPES - {"chatgpt.tokens.use.direct"}
MAX_HTTP_RESPONSE_BYTES = 1_000_000
MAX_TOKEN_LENGTH = 65_536
MAX_SCOPE_LENGTH = 8_192
MAX_JWKS_KEYS = 32
MAX_MODELS = 500
MAX_MODEL_FIELD_LENGTH = 256
MAX_SIGNED_TOKEN_LIFETIME = timedelta(hours=24)
_BASE64URL_SEGMENT = re.compile(r"^[A-Za-z0-9_-]*$")
_SUPPORTED_HTTPX_VERSION = "0.28.1"
_logger = logging.getLogger(__name__)
_ENDPOINT_METHODS = {
    TOKEN_ENDPOINT: "POST",
    REVOKE_ENDPOINT: "POST",
    JWKS_ENDPOINT: "GET",
    MODELS_ENDPOINT: "GET",
}


class OAuthFailureCode(str, Enum):
    """Safe, actionable OAuth lifecycle failure categories."""

    TRANSIENT = "transient"
    REVOKED = "revoked"
    SCOPE_LOSS = "scope_loss"
    CORRUPT_TOKEN = "corrupt_token"
    REAUTH_REQUIRED = "reauth_required"
    TOO_EARLY = "too_early"
    RACE_LOST = "race_lost"
    REDIRECT_REJECTED = "redirect_rejected"
    MALFORMED_RESPONSE = "malformed_response"
    INACTIVE = "inactive"
    INVALID_CLIENT = "invalid_client"
    REFRESH_OUTCOME_UNKNOWN = "refresh_outcome_unknown"


class OAuthLifecycleError(RuntimeError):
    """Sanitized OAuth error that never embeds a provider response or token."""

    def __init__(
        self,
        code: OAuthFailureCode,
        message: str,
        *,
        definitive: bool = False,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.definitive = definitive or code is OAuthFailureCode.CORRUPT_TOKEN
        self.retryable = retryable


@dataclass(frozen=True)
class ValidatedTokenBundle:
    bundle: CredentialBundle
    subject: str
    scopes: frozenset[str]
    access_token_expires_at: datetime
    earliest_refresh_at: datetime | None
    refresh_token_expires_at: datetime | None
    account_email: str | None = None
    received_at: datetime | None = None


@dataclass(frozen=True)
class ValidatedJWKS:
    """An immutable snapshot of a fully validated provider signing-key set."""

    _canonical_json: bytes

    def __post_init__(self) -> None:
        try:
            parsed = _strict_json_bytes(self._canonical_json)
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            raise _safe_error(
                OAuthFailureCode.MALFORMED_RESPONSE, "OAuth JWKS snapshot is malformed"
            ) from None
        validated = _validated_jwks(parsed)
        canonical = json.dumps(validated, sort_keys=True, separators=(",", ":")).encode()
        if canonical != self._canonical_json:
            raise _safe_error(
                OAuthFailureCode.MALFORMED_RESPONSE, "OAuth JWKS snapshot is malformed"
            )

    def copy_for_verification(self) -> dict[str, Any]:
        copied = _strict_json_bytes(self._canonical_json)
        if not isinstance(copied, dict):  # pragma: no cover - constructor invariant
            raise RuntimeError("Validated JWKS snapshot is invalid")
        return copied


def _safe_error(
    code: OAuthFailureCode,
    message: str,
    *,
    definitive: bool = False,
    retryable: bool = False,
) -> OAuthLifecycleError:
    return OAuthLifecycleError(code, message, definitive=definitive, retryable=retryable)


def _scrub_reachable_value(value: Any, seen: set[int]) -> None:
    """Best-effort in-place erasure without invoking arbitrary descriptors."""
    if value is None or id(value) in seen:
        return
    if isinstance(value, httpx.Response):
        _scrub_response(value, _seen=seen)
        return
    if isinstance(value, httpx.Request):
        _scrub_request(value, _seen=seen)
        return
    if isinstance(value, BaseException):
        _discard_exception_secrets(value, _seen=seen)
        return
    if isinstance(value, (Enum, type, types.ModuleType, types.FunctionType, types.CodeType)):
        return
    seen.add(id(value))
    if isinstance(value, bytearray):
        value.clear()
        return
    if isinstance(value, dict):
        for key, item in list(value.items()):
            _scrub_reachable_value(key, seen)
            _scrub_reachable_value(item, seen)
        value.clear()
        return
    if isinstance(value, list):
        for item in value:
            _scrub_reachable_value(item, seen)
        value.clear()
        return
    if isinstance(value, set):
        for item in list(value):
            _scrub_reachable_value(item, seen)
        value.clear()
        return
    if isinstance(value, (tuple, frozenset)):
        for item in value:
            _scrub_reachable_value(item, seen)

        return
    # Use only native dictionary/member descriptors from the class graph.
    # getattr/object.__getattribute__ could still invoke an arbitrary property.
    for cls in type(value).__mro__:
        dictionary_descriptor = cls.__dict__.get("__dict__")
        if isinstance(dictionary_descriptor, types.GetSetDescriptorType):
            try:
                dictionary = dictionary_descriptor.__get__(value, type(value))
            except (AttributeError, TypeError):
                continue
            if isinstance(dictionary, dict):
                _scrub_reachable_value(dictionary, seen)
        for descriptor in cls.__dict__.values():
            if not isinstance(descriptor, types.MemberDescriptorType):
                continue
            try:
                item = descriptor.__get__(value, type(value))
            except AttributeError:
                continue
            _scrub_reachable_value(item, seen)
            try:
                descriptor.__set__(value, None)
            except (AttributeError, TypeError):
                # Native read-only slots cannot own a removable reference.
                continue


def _discard_exception_secrets(
    error: BaseException, *, _seen: set[int] | None = None
) -> None:
    """Detach exception graphs, including writable slots, without hostile getattr."""
    pending: list[BaseException] = [error]
    seen = _seen if _seen is not None else set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        traceback = current.__traceback__
        if traceback is not None:
            traceback_module.clear_frames(traceback)
        for item in current.args:
            _scrub_reachable_value(item, seen)
        for item in list(current.__dict__.values()):
            _scrub_reachable_value(item, seen)
        for cls in type(current).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot in slots:
                if slot in {"__dict__", "__weakref__"}:
                    continue
                descriptor = cls.__dict__.get(slot)
                if not isinstance(descriptor, types.MemberDescriptorType):
                    continue
                try:
                    item = descriptor.__get__(current, type(current))
                except AttributeError:
                    continue
                _scrub_reachable_value(item, seen)
                try:
                    descriptor.__set__(current, None)
                except (AttributeError, TypeError) as exc:
                    raise RuntimeError("OAuth exception slot could not be scrubbed") from exc
        current.args = ()
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None
        current.__dict__.clear()


def _reject_duplicate_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object member")
        result[key] = value
    return result


def _strict_json_bytes(data: bytes) -> Any:
    return json.loads(
        data,
        object_pairs_hook=_reject_duplicate_members,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}")
        ),
    )


def _strict_jose_segment(encoded: str, index: int) -> dict[str, Any]:
    segments = encoded.split(".")
    if len(segments) != 3 or not _BASE64URL_SEGMENT.fullmatch(segments[index]):
        raise ValueError("invalid JOSE segment")
    raw = base64.urlsafe_b64decode(segments[index] + "=" * (-len(segments[index]) % 4))
    parsed = _strict_json_bytes(raw)
    if not isinstance(parsed, dict):
        raise ValueError("invalid JOSE object")
    return parsed


def safe_oauth_boundary(function):
    """Detach internal tracebacks without hiding non-OAuth programming faults."""

    @wraps(function)
    async def guarded(*args, **kwargs):
        metadata: tuple[OAuthFailureCode, str, bool, bool] | None = None
        failure_kind: str | None = None
        internal_traceback = None
        try:
            return await function(*args, **kwargs)
        except SoftTimeLimitExceeded as exc:
            internal_traceback = exc.__traceback__
            _discard_exception_secrets(exc)
            raise
        except OAuthLifecycleError as exc:
            metadata = (exc.code, str(exc), exc.definitive, exc.retryable)
            internal_traceback = exc.__traceback__
            _discard_exception_secrets(exc)
        except asyncio.CancelledError as exc:
            failure_kind = "cancelled"
            internal_traceback = exc.__traceback__
            _discard_exception_secrets(exc)
        except LookupError as exc:
            failure_kind = "lookup"
            internal_traceback = exc.__traceback__
            _discard_exception_secrets(exc)
        except AssertionError as exc:
            failure_kind = "assertion"
            internal_traceback = exc.__traceback__
            _discard_exception_secrets(exc)
        except BaseException as exc:  # noqa: BLE001 - this is the public secret boundary
            failure_kind = "unexpected"
            internal_traceback = exc.__traceback__
            _discard_exception_secrets(exc)
        finally:
            if internal_traceback is not None:
                traceback_module.clear_frames(internal_traceback)
            args = ()
            kwargs.clear()
        internal_traceback = None
        if metadata is not None:
            raise OAuthLifecycleError(
                metadata[0],
                metadata[1],
                definitive=metadata[2],
                retryable=metadata[3],
            ) from None
        if failure_kind == "cancelled":
            raise asyncio.CancelledError() from None
        if failure_kind == "lookup":
            raise LookupError("OAuth resource not found") from None
        if failure_kind == "assertion":
            raise AssertionError("OAuth operation failed safely") from None
        raise RuntimeError("OAuth operation failed safely") from None

    return guarded


def _safe_oauth_sync_boundary(function):
    """Synchronous counterpart for token validators exposed to callers."""

    @wraps(function)
    def guarded(*args, **kwargs):
        metadata: tuple[OAuthFailureCode, str, bool, bool] | None = None
        internal_traceback = None
        try:
            return function(*args, **kwargs)
        except SoftTimeLimitExceeded as exc:
            internal_traceback = exc.__traceback__
            _discard_exception_secrets(exc)
            raise
        except OAuthLifecycleError as exc:
            metadata = (exc.code, str(exc), exc.definitive, exc.retryable)
            internal_traceback = exc.__traceback__
            _discard_exception_secrets(exc)
        except BaseException as exc:  # noqa: BLE001 - this is the public secret boundary
            metadata = (
                OAuthFailureCode.CORRUPT_TOKEN,
                "Token validation failed safely",
                True,
                False,
            )
            internal_traceback = exc.__traceback__
            _discard_exception_secrets(exc)
        if internal_traceback is not None:
            traceback_module.clear_frames(internal_traceback)
        args = ()
        kwargs.clear()
        kwargs = {}
        internal_traceback = None
        assert metadata is not None
        raise OAuthLifecycleError(
            metadata[0],
            metadata[1],
            definitive=metadata[2],
            retryable=metadata[3],
        ) from None

    return guarded


def _ensure_declared_response_size(response: httpx.Response) -> None:
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            raise _safe_error(
                OAuthFailureCode.MALFORMED_RESPONSE,
                "OAuth provider returned an invalid response size",
            ) from None
        if declared_length < 0 or declared_length > MAX_HTTP_RESPONSE_BYTES:
            raise _safe_error(
                OAuthFailureCode.MALFORMED_RESPONSE,
                "OAuth provider response exceeded the size limit",
            )


def _ensure_response_size(response: httpx.Response) -> None:
    _ensure_declared_response_size(response)
    if len(response.content) > MAX_HTTP_RESPONSE_BYTES:
        raise _safe_error(
            OAuthFailureCode.MALFORMED_RESPONSE,
            "OAuth provider response exceeded the size limit",
        )


def _sanitized_request(request: httpx.Request) -> httpx.Request:
    """Retain routing metadata without credential-bearing headers or content."""
    return httpx.Request(request.method, request.url)


def _scrub_request(request: httpx.Request, *, _seen: set[int] | None = None) -> None:
    """Erase all mutable request state that can contain bearer material."""
    seen = _seen if _seen is not None else set()
    if id(request) in seen:
        return
    seen.add(id(request))
    for value in list(request.extensions.values()):
        _scrub_reachable_value(value, seen)
    request.headers.clear()
    request.extensions.clear()
    request._content = b""
    request.stream = httpx.ByteStream(b"")


def _scrub_response(response: httpx.Response, *, _seen: set[int] | None = None) -> None:
    """Erase and detach every request/response reachable from a response."""
    seen = _seen if _seen is not None else set()
    if id(response) in seen:
        return
    seen.add(id(response))
    next_request = response.next_request
    history = list(response.history)
    extension_values = list(response.extensions.values())
    response.next_request = None
    response.history = []
    response._content = b""  # httpx intentionally exposes no public content scrubber
    response.headers.clear()
    for value in extension_values:
        _scrub_reachable_value(value, seen)
    response.extensions.clear()
    response.stream = httpx.ByteStream(b"")
    if next_request is not None:
        _scrub_request(next_request, _seen=seen)
    for historical_response in history:
        _scrub_response(historical_response, _seen=seen)
    try:
        request = response.request
    except RuntimeError:
        return
    _scrub_request(request, _seen=seen)


def _json_object(response: httpx.Response) -> dict[str, Any]:
    failure: OAuthLifecycleError | None = None
    payload: Any = None
    try:
        _ensure_response_size(response)
        payload = _strict_json_bytes(response.content)
    except OAuthLifecycleError as exc:
        failure = exc
    except (ValueError, TypeError):
        failure = _safe_error(
            OAuthFailureCode.MALFORMED_RESPONSE, "OAuth provider returned malformed JSON"
        )
    if failure is not None:
        _scrub_response(response)
        raise failure
    if not isinstance(payload, dict):
        _scrub_response(response)
        raise _safe_error(
            OAuthFailureCode.MALFORMED_RESPONSE, "OAuth provider returned malformed JSON"
        )
    return payload


def _reject_redirect(response: httpx.Response) -> None:
    if response.is_redirect:
        raise _safe_error(
            OAuthFailureCode.REDIRECT_REJECTED,
            "OAuth provider redirect was rejected",
            retryable=True,
        )


def _provider_failure(response: httpx.Response, *, operation: str) -> OAuthLifecycleError:
    _ensure_response_size(response)
    status_code = response.status_code
    error_code = ""
    try:
        payload = _strict_json_bytes(response.content)
        if isinstance(payload, dict):
            error_code = str(payload.get("error") or "")
    except (ValueError, TypeError):
        pass
    _scrub_response(response)
    terminal_refresh_errors = {
        "invalid_grant",
        "invalid_refresh_token",
        "token_expired",
        "refresh_token_expired",
        "refresh_token_invalidated",
        "refresh_token_reused",
    }
    if error_code == "invalid_token" or (
        operation == "token refresh" and error_code in terminal_refresh_errors
    ):
        return _safe_error(
            OAuthFailureCode.REAUTH_REQUIRED,
            f"OAuth {operation} requires reauthorization",
            definitive=True,
        )
    if error_code == "invalid_client":
        return _safe_error(
            OAuthFailureCode.INVALID_CLIENT,
            f"OAuth {operation} rejected the client",
        )
    if status_code in {408, 425, 429} or status_code >= 500:
        return _safe_error(
            OAuthFailureCode.TRANSIENT,
            f"OAuth {operation} is temporarily unavailable",
            retryable=True,
        )
    return _safe_error(
        OAuthFailureCode.MALFORMED_RESPONSE,
        f"OAuth {operation} failed",
    )


def _trusted_origin(url: httpx.URL, method: str) -> bool:
    if url.username or url.password or url.fragment:
        return False
    if url.raw_scheme != b"https" or url.raw_host not in {
        b"auth.openai.com",
        b"api.openai.com",
    }:
        return False
    port = 443 if url.port is None else url.port
    return port == 443 and _ENDPOINT_METHODS.get(str(url)) == method


def _new_oauth_transport() -> httpx.AsyncBaseTransport:
    settings = get_agent_settings()
    default_paths = ssl.get_default_verify_paths()
    ssl_context = ssl.create_default_context(
        cafile=settings.siwc_oauth_ca_bundle or default_paths.openssl_cafile,
        capath=None if settings.siwc_oauth_ca_bundle else default_paths.openssl_capath,
    )
    return httpx.AsyncHTTPTransport(
        verify=ssl_context,
        proxy=settings.siwc_oauth_proxy_url or None,
        trust_env=False,
    )


async def _send_protocol_request(
    request: httpx.Request,
) -> tuple[httpx.Response, httpx.AsyncClient]:
    """Dispatch once through a protocol-owned client and transport."""
    if httpx.__version__ != _SUPPORTED_HTTPX_VERSION:
        raise RuntimeError("Unsupported httpx version for secure OAuth requests")

    # Reject the complete destination before constructing or selecting a transport.
    if not _trusted_origin(request.url, request.method):
        raise RuntimeError("OAuth request target is not trusted")
    transport = _new_oauth_transport()
    settings = get_agent_settings()
    try:
        isolated = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(
                settings.siwc_oauth_timeout_seconds,
                connect=settings.siwc_oauth_connect_timeout_seconds,
            ),
            follow_redirects=False,
            trust_env=False,
        )
    except BaseException:
        try:
            await transport.aclose()
        except BaseException:
            _logger.warning("OAuth transport cleanup failed after client construction failure")
        raise
    try:
        isolated._set_timeout(request)  # type: ignore[attr-defined]
        # This is deliberately the last check before the one-shot transport call.
        if not _trusted_origin(request.url, request.method):
            raise RuntimeError("OAuth request target is not trusted")
        response = await isolated._send_single_request(request)  # type: ignore[attr-defined]
    except BaseException:
        try:
            await isolated.aclose()
        except BaseException:
            _logger.warning("OAuth client cleanup failed after request failure")
        raise
    return response, isolated


async def _request(method: str, url: str, **kwargs: Any) -> httpx.Response:
    bounded_response: httpx.Response | None = None
    source_response: httpx.Response | None = None
    protocol_client: httpx.AsyncClient | None = None
    outgoing_request: httpx.Request | None = None
    content = bytearray()
    oauth_failure: tuple[OAuthFailureCode, str, bool, bool] | None = None
    unexpected_kind: str | None = None
    soft_failure: SoftTimeLimitExceeded | None = None
    cleanup_failure: BaseException | None = None
    internal_traceback = None
    try:
        if _ENDPOINT_METHODS.get(url) != method:
            raise RuntimeError("OAuth request target is not trusted")
        unsupported = set(kwargs) - {"content", "headers"}
        if unsupported:
            raise RuntimeError("Unsupported OAuth request options")
        outgoing_request = httpx.Request(
            method,
            url,
            content=kwargs.get("content"),
            headers=kwargs.get("headers"),
        )
        async with asyncio.timeout(get_agent_settings().siwc_oauth_timeout_seconds):
            source_response, protocol_client = await _send_protocol_request(outgoing_request)
            _reject_redirect(source_response)
            _ensure_declared_response_size(source_response)
            async for chunk in source_response.aiter_bytes():
                if len(content) + len(chunk) > MAX_HTTP_RESPONSE_BYTES:
                    raise _safe_error(
                        OAuthFailureCode.MALFORMED_RESPONSE,
                        "OAuth provider response exceeded the size limit",
                    )
                content.extend(chunk)
            bounded_response = httpx.Response(
                source_response.status_code,
                headers=source_response.headers,
                content=bytes(content),
                request=_sanitized_request(outgoing_request),
                extensions=dict(source_response.extensions),
            )
    except SoftTimeLimitExceeded as exc:
        soft_failure = exc
        internal_traceback = exc.__traceback__
        _discard_exception_secrets(exc)
    except OAuthLifecycleError as exc:
        oauth_failure = (exc.code, str(exc), exc.definitive, exc.retryable)
        internal_traceback = exc.__traceback__
        _discard_exception_secrets(exc)
    except (httpx.HTTPError, TimeoutError) as exc:
        request = getattr(exc, "request", None)
        response = getattr(exc, "response", None)
        if isinstance(request, httpx.Request):
            _scrub_request(request)
        if isinstance(response, httpx.Response):
            _scrub_response(response)
        oauth_failure = (
            OAuthFailureCode.TRANSIENT,
            "OAuth provider is temporarily unreachable",
            False,
            True,
        )
        internal_traceback = exc.__traceback__
        _discard_exception_secrets(exc)
    except asyncio.CancelledError as exc:
        unexpected_kind = "cancelled"
        internal_traceback = exc.__traceback__
        _discard_exception_secrets(exc)
    except BaseException as exc:  # noqa: BLE001 - this is the public secret boundary
        unexpected_kind = "runtime"
        internal_traceback = exc.__traceback__
        _discard_exception_secrets(exc)
    finally:
        kwargs.clear()
        content.clear()
        requests_to_scrub: list[httpx.Request] = []
        if outgoing_request is not None:
            requests_to_scrub.append(outgoing_request)
        if source_response is not None:
            try:
                requests_to_scrub.append(source_response.request)
            except RuntimeError:
                pass
        if source_response is not None:
            try:
                await source_response.aclose()
            except BaseException as exc:  # cleanup must not mask the primary failure
                cleanup_failure = exc
            finally:
                _scrub_response(source_response)
        if protocol_client is not None:
            try:
                await protocol_client.aclose()
            except BaseException as exc:  # still scrub requests after close failure
                cleanup_failure = cleanup_failure or exc
        try:
            scrubbed_request_ids: set[int] = set()
            for request_to_scrub in requests_to_scrub:
                request_id = id(request_to_scrub)
                if request_id in scrubbed_request_ids:
                    continue
                scrubbed_request_ids.add(request_id)
                _scrub_request(request_to_scrub)
            requests_to_scrub.clear()
            if internal_traceback is not None:
                traceback_module.clear_frames(internal_traceback)
        except BaseException as exc:
            cleanup_failure = cleanup_failure or exc
    internal_traceback = None
    if soft_failure is not None:
        if cleanup_failure is not None:
            _logger.warning("OAuth request cleanup failed during soft time limit handling")
        raise soft_failure
    if oauth_failure is not None:
        raise OAuthLifecycleError(
            oauth_failure[0],
            oauth_failure[1],
            definitive=oauth_failure[2],
            retryable=oauth_failure[3],
        ) from None
    if unexpected_kind == "cancelled":
        raise asyncio.CancelledError() from None
    if unexpected_kind is not None:
        raise RuntimeError("OAuth request failed safely") from None
    if cleanup_failure is not None:
        raise RuntimeError("OAuth request cleanup failed safely") from None
    if bounded_response is None:
        raise RuntimeError("OAuth request failed safely")
    return bounded_response


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or len(value) > MAX_TOKEN_LENGTH:
        raise _safe_error(
            OAuthFailureCode.MALFORMED_RESPONSE,
            "OAuth provider returned an incomplete token response",
        )
    return value


def _bounded_input(value: str, *, field: str, maximum: int) -> None:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise _safe_error(
            OAuthFailureCode.MALFORMED_RESPONSE,
            f"OAuth {field} is invalid",
        )


def _scopes(payload: dict[str, Any]) -> frozenset[str]:
    raw = payload.get("scope")
    if not isinstance(raw, str) or len(raw) > MAX_SCOPE_LENGTH:
        raise _safe_error(
            OAuthFailureCode.MALFORMED_RESPONSE, "OAuth provider omitted token scopes"
        )
    granted = frozenset(raw.split())
    if len(granted) > 64 or any(len(scope) > 256 for scope in granted):
        raise _safe_error(OAuthFailureCode.MALFORMED_RESPONSE, "OAuth token scopes are invalid")
    if not BASE_REQUIRED_SCOPES.issubset(granted):
        raise _safe_error(
            OAuthFailureCode.SCOPE_LOSS,
            "Required OAuth identity or resource scopes were not granted",
            definitive=True,
        )
    return granted


def _timestamp(
    value: Any,
    *,
    field: str,
    minimum: datetime,
    maximum: datetime,
) -> datetime | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise _safe_error(
            OAuthFailureCode.MALFORMED_RESPONSE, f"OAuth provider returned invalid {field}"
        )
    try:
        result = datetime.fromtimestamp(value, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        raise _safe_error(
            OAuthFailureCode.MALFORMED_RESPONSE, f"OAuth provider returned invalid {field}"
        ) from None
    if result < minimum or result > maximum:
        raise _safe_error(
            OAuthFailureCode.MALFORMED_RESPONSE, f"OAuth provider returned invalid {field}"
        )
    return result


@_safe_oauth_sync_boundary
def validate_id_token(
    encoded: str,
    *,
    jwks: dict[str, Any] | ValidatedJWKS,
    issued_client_id: str,
    expected_subject: str | None = None,
    nonce: str | None = None,
    access_token: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify signature and identity claims, adding nonce/subject binding."""
    _bounded_input(encoded, field="ID token", maximum=MAX_TOKEN_LENGTH)
    _bounded_input(issued_client_id, field="client identifier", maximum=512)
    try:
        _strict_jose_segment(encoded, 0)
        _strict_jose_segment(encoded, 1)
    except (ValueError, binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "ID token encoding is invalid") from None
    selected_jwks = _selected_verification_jwks(encoded, jwks=jwks, token_name="ID")
    try:
        claims = jwt.decode(
            encoded,
            selected_jwks,
            algorithms=["RS256"],
            issuer=ISSUER,
            audience=issued_client_id,
            access_token=access_token,
            options={
                "require_iss": True,
                "require_aud": True,
                "require_exp": False,
                "require_iat": False,
                "require_sub": True,
                "verify_exp": False,
                "verify_iat": False,
            },
        )
    except (JWTError, ValueError, TypeError):
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "ID token validation failed") from None
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject or len(subject) > 255:
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "ID token subject is invalid")
    if expected_subject is not None and subject != expected_subject:
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "ID token identity changed")
    _validate_signed_times(claims, now=now or datetime.now(timezone.utc), token_name="ID")
    if nonce is not None and claims.get("nonce") != nonce:
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "ID token nonce validation failed")
    audience = claims.get("aud")
    authorized_party = claims.get("azp")
    if (authorized_party is not None and authorized_party != issued_client_id) or (
        isinstance(audience, list) and len(audience) > 1 and authorized_party != issued_client_id
    ):
        raise _safe_error(
            OAuthFailureCode.CORRUPT_TOKEN, "ID token authorized party validation failed"
        )
    return claims


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _validate_signed_times(claims: dict[str, Any], *, now: datetime, token_name: str) -> datetime:
    issued_at = claims.get("iat")
    expires_at = claims.get("exp")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in (issued_at, expires_at)
    ):
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, f"{token_name} token times are invalid")
    assert isinstance(issued_at, (int, float))
    assert isinstance(expires_at, (int, float))
    try:
        issued = datetime.fromtimestamp(float(issued_at), tz=timezone.utc)
        expires = datetime.fromtimestamp(float(expires_at), tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        raise _safe_error(
            OAuthFailureCode.CORRUPT_TOKEN, f"{token_name} token times are invalid"
        ) from None
    reference = _utc_datetime(now)
    if (
        issued > reference + timedelta(seconds=60)
        or expires <= issued
        or expires <= reference
        or expires - issued > MAX_SIGNED_TOKEN_LIFETIME
    ):
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, f"{token_name} token times are invalid")
    return expires


def _selected_verification_jwks(
    encoded: str, *, jwks: dict[str, Any] | ValidatedJWKS, token_name: str
) -> dict[str, Any]:
    validated = (
        jwks.copy_for_verification()
        if isinstance(jwks, ValidatedJWKS)
        else _validated_jwks(jwks)
    )
    try:
        header = _strict_jose_segment(encoded, 0)
    except (ValueError, binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        raise _safe_error(
            OAuthFailureCode.CORRUPT_TOKEN, f"{token_name} token header is invalid"
        ) from None
    if not isinstance(header, dict) or header.get("alg") != "RS256":
        raise _safe_error(
            OAuthFailureCode.CORRUPT_TOKEN,
            f"{token_name} token uses an unsupported algorithm",
        )
    critical = header.get("crit")
    if critical not in (None, []):
        raise _safe_error(
            OAuthFailureCode.CORRUPT_TOKEN,
            f"{token_name} token uses unsupported critical headers",
        )
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid or len(kid) > 512:
        raise _safe_error(
            OAuthFailureCode.CORRUPT_TOKEN, f"{token_name} token key identifier is invalid"
        )
    matching = [key for key in validated["keys"] if key.get("kid") == kid]
    if len(matching) != 1:
        raise _safe_error(
            OAuthFailureCode.CORRUPT_TOKEN, f"{token_name} token signing key is invalid"
        )
    return {"keys": matching}


def _validate_access_token(
    encoded: str,
    *,
    jwks: dict[str, Any] | ValidatedJWKS,
    identity_claims: dict[str, Any],
    now: datetime,
    issued_client_id: str,
    granted_scopes: frozenset[str],
) -> datetime | None:
    """Validate structurally recognizable RS256 JWTs; preserve opaque tokens."""
    segments = encoded.split(".")
    if len(segments) != 3:
        return None
    try:
        header_segment = segments[0]
        if not _BASE64URL_SEGMENT.fullmatch(header_segment):
            return None
        header_bytes = base64.urlsafe_b64decode(header_segment + "=" * (-len(header_segment) % 4))
        recognizable_header = _strict_json_bytes(header_bytes)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(recognizable_header, dict):
        return None
    try:
        header = _strict_json_bytes(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError):
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "Access token header is invalid") from None
    if not all(part and _BASE64URL_SEGMENT.fullmatch(part) for part in segments):
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "Access token encoding is invalid")
    if header.get("alg") != "RS256":
        raise _safe_error(
            OAuthFailureCode.CORRUPT_TOKEN, "Access token uses an unsupported algorithm"
        )
    selected_jwks = _selected_verification_jwks(encoded, jwks=jwks, token_name="Access")
    try:
        unverified_claims = _strict_jose_segment(encoded, 1)
    except (ValueError, binascii.Error, UnicodeDecodeError, json.JSONDecodeError):
        raise _safe_error(
            OAuthFailureCode.CORRUPT_TOKEN, "Access token claims are invalid"
        ) from None
    if not isinstance(unverified_claims, dict):
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "Access token claims are invalid")
    try:
        claims = jwt.decode(
            encoded,
            selected_jwks,
            algorithms=["RS256"],
            issuer=ISSUER,
            audience=RESOURCE,
            options={
                "require_iss": True,
                "require_aud": True,
                "require_exp": False,
                "require_iat": False,
                "require_sub": True,
                "verify_exp": False,
                "verify_iat": False,
            },
        )
    except (JWTError, ValueError, TypeError):
        raise _safe_error(
            OAuthFailureCode.CORRUPT_TOKEN, "Access token validation failed"
        ) from None
    if claims.get("sub") != identity_claims.get("sub"):
        raise _safe_error(
            OAuthFailureCode.CORRUPT_TOKEN,
            "Access token identity does not match the ID token",
        )
    client_id = claims.get("client_id")
    if not isinstance(client_id, str) or not 0 < len(client_id) <= 255 or client_id != issued_client_id:
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "Access token client binding is invalid")
    if _scopes(claims) != granted_scopes:
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "Access token grants do not match")
    jti = claims.get("jti")
    metadata = claims.get("https://api.openai.com/auth")
    if not isinstance(jti, str) or not jti.strip() or len(jti) > 512:
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "Access token identifier is invalid")
    if not isinstance(metadata, dict) or any(
        not isinstance(metadata.get(field), str)
        or not metadata[field].strip()
        or len(metadata[field]) > 8192
        for field in ("per_user_salt", "encrypted_auth_metadata")
    ):
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "Access token auth metadata is invalid")
    nbf = claims.get("nbf")
    if (isinstance(nbf, bool) or not isinstance(nbf, (int, float))
        or not math.isfinite(float(nbf)) or nbf > _utc_datetime(now).timestamp() + 60
        or nbf < claims.get("iat", 0) - 60 or nbf >= claims.get("exp", 0)):
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "Access token not-before is invalid")
    return _validate_signed_times(claims, now=now, token_name="Access")


async def _fetch_jwks() -> ValidatedJWKS:
    response = await _request("GET", JWKS_ENDPOINT)
    payload: dict[str, Any] = {}
    try:
        if response.status_code != 200:
            raise _provider_failure(response, operation="JWKS lookup")
        payload = _json_object(response)
        validated = _validated_jwks(payload)
        return ValidatedJWKS(
            json.dumps(validated, sort_keys=True, separators=(",", ":")).encode()
        )
    finally:
        payload.clear()
        _scrub_response(response)


async def fetch_validated_jwks() -> ValidatedJWKS:
    """Fetch and freeze signing keys before consuming a one-shot credential."""
    return await _fetch_jwks()


def _validate_json_graph(value: Any) -> None:
    """Bound all supplied JSON, including fields the verifier does not use."""
    pending = [(value, 0)]
    nodes = 0
    string_size = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > 16 or nodes > 4096:
            raise ValueError("JSON graph exceeds bounds")
        if item is None or type(item) is bool:
            continue
        if type(item) is str:
            string_size += len(item)
            if len(item) > 65536 or string_size > MAX_HTTP_RESPONSE_BYTES:
                raise ValueError("JSON string exceeds bounds")
        elif type(item) is int:
            if item.bit_length() > 1024:
                raise ValueError("JSON integer exceeds bounds")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("JSON number is nonfinite")
        elif type(item) is list:
            if len(item) > 4096:
                raise ValueError("JSON array exceeds bounds")
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is dict:
            if len(item) > 128 or any(type(key) is not str or len(key) > 512 for key in item):
                raise ValueError("JSON object keys exceed bounds")
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        else:
            raise ValueError("JSON value is incompatible")


def _validated_jwks(jwks: Any) -> dict[str, Any]:
    try:
        _validate_json_graph(jwks)
    except ValueError:
        raise _safe_error(OAuthFailureCode.MALFORMED_RESPONSE, "OAuth JWKS graph is malformed") from None
    if not isinstance(jwks, dict):
        raise _safe_error(OAuthFailureCode.MALFORMED_RESPONSE, "OAuth JWKS payload is malformed")
    keys = jwks.get("keys")
    if not isinstance(keys, list) or not keys or len(keys) > MAX_JWKS_KEYS:
        raise _safe_error(OAuthFailureCode.MALFORMED_RESPONSE, "OAuth JWKS payload is malformed")
    usable = [dict(key) for key in keys if _usable_rsa_verification_jwk(key)]
    if not usable:
        raise _safe_error(OAuthFailureCode.MALFORMED_RESPONSE, "OAuth JWKS payload is malformed")
    return {"keys": usable}


def _usable_rsa_verification_jwk(key: Any) -> bool:
    if not isinstance(key, dict) or len(key) > 32 or key.get("kty") != "RSA":
        return False
    kid = key.get("kid")
    if not isinstance(kid, str) or not kid or len(kid) > 512:
        return False
    if key.get("alg") not in (None, "RS256") or key.get("use") not in (None, "sig"):
        return False
    key_ops = key.get("key_ops")
    if key_ops is not None and (
        not isinstance(key_ops, list)
        or len(key_ops) > 16
        or "verify" not in key_ops
        or any(not isinstance(operation, str) or len(operation) > 64 for operation in key_ops)
    ):
        return False
    numbers: dict[str, int] = {}
    for field in ("n", "e"):
        encoded = key.get(field)
        if (
            not isinstance(encoded, str)
            or not encoded
            or len(encoded) > 16_384
            or not _BASE64URL_SEGMENT.fullmatch(encoded)
        ):
            return False
        try:
            decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        except (binascii.Error, ValueError):
            return False
        if not decoded:
            return False
        numbers[field] = int.from_bytes(decoded, "big")
    modulus = numbers["n"]
    exponent = numbers["e"]
    return (
        modulus.bit_length() >= 2048
        and modulus % 2 == 1
        and 3 <= exponent <= 0xFFFFFFFF
        and exponent % 2 == 1
    )


def _validated_response_impl(
    payload: dict[str, Any],
    *,
    jwks: dict[str, Any] | ValidatedJWKS,
    issued_client_id: str,
    now: datetime,
    nonce: str | None = None,
    expected_subject: str | None = None,
) -> ValidatedTokenBundle:
    access_token = _required_string(payload, "access_token")
    refresh_token = _required_string(payload, "refresh_token")
    id_token = _required_string(payload, "id_token")
    token_type = payload.get("token_type")
    if not isinstance(token_type, str) or token_type.casefold() != "bearer":
        raise _safe_error(
            OAuthFailureCode.MALFORMED_RESPONSE, "OAuth provider returned invalid token type"
        )
    expires_in = payload.get("expires_in")
    if (
        isinstance(expires_in, bool)
        or not isinstance(expires_in, (int, float))
        or not math.isfinite(float(expires_in))
        or expires_in <= 0
        or expires_in > 24 * 60 * 60
    ):
        raise _safe_error(
            OAuthFailureCode.MALFORMED_RESPONSE, "OAuth provider returned invalid expiry"
        )
    granted = _scopes(payload)
    claims = validate_id_token(
        id_token,
        jwks=jwks,
        issued_client_id=issued_client_id,
        expected_subject=expected_subject,
        nonce=nonce,
        access_token=access_token,
        now=now,
    )
    signed_access_expiry = _validate_access_token(
        access_token, jwks=jwks, identity_claims=claims, now=now,
        issued_client_id=issued_client_id, granted_scopes=granted
    )
    if signed_access_expiry is None and not isinstance(claims.get("at_hash"), str):
        raise _safe_error(OAuthFailureCode.CORRUPT_TOKEN, "Opaque access token requires ID token hash binding")
    advertised_expiry = now + timedelta(seconds=float(expires_in))
    access_expires_at = min(advertised_expiry, signed_access_expiry or advertised_expiry)
    return ValidatedTokenBundle(
        bundle=CredentialBundle(access_token, refresh_token, id_token),
        subject=claims["sub"],
        account_email=sanitize_account_email(claims.get("email")),
        scopes=granted,
        access_token_expires_at=access_expires_at,
        earliest_refresh_at=_timestamp(
            payload.get("earliest_refresh_at"),
            field="earliest_refresh_at",
            minimum=now - timedelta(days=1),
            maximum=access_expires_at,
        ),
        refresh_token_expires_at=min(
            now + timedelta(days=30),
            _timestamp(
                payload.get("refresh_token_expires_at"),
                field="refresh_token_expires_at",
                minimum=now,
                maximum=now + timedelta(days=366),
            )
            or now + timedelta(days=30),
        ),
        received_at=now,
    )


def _validated_response(
    payload: dict[str, Any],
    *,
    jwks: dict[str, Any] | ValidatedJWKS,
    issued_client_id: str,
    now: datetime,
    nonce: str | None = None,
    expected_subject: str | None = None,
) -> ValidatedTokenBundle:
    failure_meta: tuple[OAuthFailureCode, str, bool, bool] | None = None
    result: ValidatedTokenBundle | None = None
    try:
        result = _validated_response_impl(
            payload,
            jwks=jwks,
            issued_client_id=issued_client_id,
            now=now,
            nonce=nonce,
            expected_subject=expected_subject,
        )
    except OAuthLifecycleError as exc:
        failure_meta = (exc.code, str(exc), exc.definitive, exc.retryable)
    finally:
        payload.clear()
    if failure_meta is not None:
        raise OAuthLifecycleError(
            failure_meta[0],
            failure_meta[1],
            definitive=failure_meta[2],
            retryable=failure_meta[3],
        )
    if result is None:
        raise RuntimeError("OAuth token validation failed safely")
    return result


@safe_oauth_boundary
async def exchange_authorization_code(
    *,
    code: str,
    verifier: str,
    redirect_uri: str,
    issued_client_id: str,
    nonce: str,
    receipt_clock: Callable[[], datetime] | None = None,
) -> ValidatedTokenBundle:
    """Exchange a code and validate the complete authorization result."""
    _bounded_input(code, field="authorization code", maximum=8_192)
    _bounded_input(verifier, field="PKCE verifier", maximum=512)
    _bounded_input(redirect_uri, field="redirect URI", maximum=2_048)
    _bounded_input(issued_client_id, field="client identifier", maximum=512)
    _bounded_input(nonce, field="nonce", maximum=1_024)
    jwks = await _fetch_jwks()
    response = await _request(
        "POST",
        TOKEN_ENDPOINT,
        content=urlencode(
            {
                "grant_type": "authorization_code",
                "client_id": issued_client_id,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
                "resource": RESOURCE,
            }
        ),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    received_at = _utc_datetime((receipt_clock or (lambda: datetime.now(timezone.utc)))())
    payload: dict[str, Any] = {}
    try:
        if response.status_code != 200:
            raise _provider_failure(response, operation="authorization exchange")
        payload = _json_object(response)
        return _validated_response(
            payload,
            jwks=jwks,
            issued_client_id=issued_client_id,
            nonce=nonce,
            now=received_at,
        )
    finally:
        payload.clear()
        _scrub_response(response)
        code = verifier = nonce = ""


@safe_oauth_boundary
async def refresh_token_bundle(
    *,
    refresh_token: str,
    issued_client_id: str,
    expected_subject: str,
    jwks: dict[str, Any] | ValidatedJWKS | None = None,
    receipt_clock: Callable[[], datetime] | None = None,
) -> ValidatedTokenBundle:
    _bounded_input(refresh_token, field="refresh token", maximum=MAX_TOKEN_LENGTH)
    _bounded_input(issued_client_id, field="client identifier", maximum=512)
    _bounded_input(expected_subject, field="subject", maximum=255)
    key_set = (
        jwks
        if isinstance(jwks, ValidatedJWKS)
        else _validated_jwks(jwks)
        if jwks is not None
        else await _fetch_jwks()
    )
    response = await _request(
        "POST",
        TOKEN_ENDPOINT,
        content=urlencode(
            {
                "grant_type": "refresh_token",
                "client_id": issued_client_id,
                "refresh_token": refresh_token,
                "resource": RESOURCE,
            }
        ),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    received_at = _utc_datetime((receipt_clock or (lambda: datetime.now(timezone.utc)))())
    payload: dict[str, Any] = {}
    try:
        if response.status_code != 200:
            raise _provider_failure(response, operation="token refresh")
        payload = _json_object(response)
        return _validated_response(
            payload,
            jwks=key_set,
            issued_client_id=issued_client_id,
            expected_subject=expected_subject,
            now=received_at,
        )
    finally:
        payload.clear()
        _scrub_response(response)
        refresh_token = ""


@safe_oauth_boundary
async def revoke_token(
    *,
    refresh_token: str,
    issued_client_id: str,
) -> None:
    """Revoke a refresh token; already-invalid tokens count as revoked."""
    _bounded_input(refresh_token, field="refresh token", maximum=MAX_TOKEN_LENGTH)
    _bounded_input(issued_client_id, field="client identifier", maximum=512)
    response = await _request(
        "POST",
        REVOKE_ENDPOINT,
        content=urlencode(
            {
                "token": refresh_token,
                "token_type_hint": "refresh_token",
                "client_id": issued_client_id,
            }
        ),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    if response.status_code < 300:
        _scrub_response(response)
        return
    _ensure_response_size(response)
    try:
        payload = _strict_json_bytes(response.content)
    except (ValueError, TypeError):
        payload = {}
    if isinstance(payload, dict) and payload.get("error") in {"invalid_token", "invalid_grant"}:
        payload.clear()
        _scrub_response(response)
        return
    raise _provider_failure(response, operation="token revocation")


@safe_oauth_boundary
async def list_models(
    access_token: str,
) -> list[dict[str, str]]:
    _bounded_input(access_token, field="access token", maximum=MAX_TOKEN_LENGTH)
    response = await _request(
        "GET",
        MODELS_ENDPOINT,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if response.status_code == 401:
        _scrub_response(response)
        raise _safe_error(
            OAuthFailureCode.REAUTH_REQUIRED,
            "ChatGPT plan authorization was rejected",
            definitive=True,
        )
    if response.status_code == 403:
        _scrub_response(response)
        raise _safe_error(
            OAuthFailureCode.SCOPE_LOSS,
            "ChatGPT plan scope or entitlement was rejected",
            definitive=True,
        )
    if response.status_code != 200:
        raise _provider_failure(response, operation="model discovery")
    payload: dict[str, Any] = {}
    try:
        payload = _json_object(response)
        raw_models = payload.get("models")
        if not isinstance(raw_models, list) or len(raw_models) > MAX_MODELS:
            raise _safe_error(
                OAuthFailureCode.MALFORMED_RESPONSE, "Model catalog payload is malformed"
            )
        models: list[dict[str, str]] = []
        for raw in raw_models:
            if not isinstance(raw, dict):
                raise _safe_error(
                    OAuthFailureCode.MALFORMED_RESPONSE, "Model catalog payload is malformed"
                )
            if raw.get("visibility") != "list":
                continue
            slug = raw.get("slug")
            display_name = raw.get("display_name")
            if (
                not isinstance(slug, str)
                or not slug
                or len(slug) > MAX_MODEL_FIELD_LENGTH
                or not isinstance(display_name, str)
                or not display_name
                or len(display_name) > MAX_MODEL_FIELD_LENGTH
            ):
                raise _safe_error(
                    OAuthFailureCode.MALFORMED_RESPONSE, "Model catalog payload is malformed"
                )
            models.append({"slug": slug, "display_name": display_name})
        return models
    finally:
        payload.clear()
        _scrub_response(response)
