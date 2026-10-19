"""Secure user-owned and secret-authenticated SIWC pairing endpoints."""
from __future__ import annotations

import asyncio
from functools import wraps
import json
import math
import uuid
from typing import TypeVar, cast

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.schemas.oauth_pairing import (
    AttemptRegistration,
    CompletionRequest,
    CompletionResult,
    PairingBootstrap,
    PairingStart,
    PairingStatus,
    PairingFailureCode,
    PairingState,
)
from app.agents.services import oauth_pairing, openai_oauth
from app.agents.siwc_distribution import (
    bootstrap_script,
    helper_command,
    helper_wheel,
    public_frontend_origin,
)
from app.agents.siwc_observability import observe_siwc
from app.core.auth import current_active_user
from app.core.config import get_settings
from app.core.database import get_async_session
from app.core.rate_limit import pairing_helper_rate_limit, pairing_start_rate_limit
from app.models.user import User

router = APIRouter(prefix="/api/agents", tags=["agents"])
T = TypeVar("T", bound=BaseModel)
MAX_PAIRING_BODY = 256 * 1024


class _InvalidRequest(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=400, detail={"code": "invalid_request"})


class _NotAvailable(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=404, detail={"code": "pairing_not_available"})


class _CredentialBusy(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=409, detail={"code": "credential_busy"})


class _ProviderUnavailable(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=503, detail={"code": "provider_unavailable"})


class _InvalidCredential(HTTPException):
    def __init__(self) -> None:
        super().__init__(status_code=400, detail={"code": "invalid_credential"})


_SAFE_HELPER_FAILURES: dict[type[BaseException], tuple[int, dict[str, str]]] = {
    _InvalidRequest: (400, {"code": "invalid_request"}),
    _NotAvailable: (404, {"code": "pairing_not_available"}),
    _CredentialBusy: (409, {"code": "credential_busy"}),
    _ProviderUnavailable: (503, {"code": "provider_unavailable"}),
    _InvalidCredential: (400, {"code": "invalid_credential"}),
}


def _not_available() -> HTTPException:
    return _NotAvailable()


def _secret(request: Request) -> str:
    values = [
        value for name, value in request.scope.get("headers", [])
        if name.lower() == b"authorization"
    ]
    if len(values) != 1:
        raise _not_available()
    try:
        authorization = values[0].decode("ascii")
    except UnicodeDecodeError:
        raise _not_available() from None
    prefix = "SecuroPairing "
    secret = authorization[len(prefix):] if authorization.startswith(prefix) else ""
    if not 40 <= len(secret) <= 256 or not secret.isascii() or any(char.isspace() for char in secret):
        raise _not_available()
    return secret


def _unique_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate member")
        result[key] = value
    return result


def _validate_graph(value: object) -> None:
    pending = [(value, 0)]
    nodes = 0
    string_units = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 4096 or depth > 16:
            raise ValueError("JSON graph exceeds bounds")
        if item is None or type(item) in {bool, int}:
            continue
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite JSON number")
            continue
        if isinstance(item, str):
            string_units += len(item)
            if len(item) > 65_536 or string_units > MAX_PAIRING_BODY:
                raise ValueError("JSON string exceeds bounds")
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, dict):
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        else:
            raise ValueError("unsupported JSON value")


async def _strict_body(request: Request, model: type[T]) -> T:
    content_types = [
        value
        for name, value in request.scope.get("headers", [])
        if name.lower() == b"content-type"
    ]
    if content_types != [b"application/json"]:
        raise _InvalidRequest()
    declared_values = [
        value
        for name, value in request.scope.get("headers", [])
        if name.lower() == b"content-length"
    ]
    if len(declared_values) > 1:
        raise _InvalidRequest()
    if declared_values:
        try:
            declared = int(declared_values[0].decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            raise _InvalidRequest() from None
        if declared <= 0 or declared > MAX_PAIRING_BODY:
            raise _InvalidRequest()
    body = bytearray()
    try:
        async for chunk in request.stream():
            remaining = MAX_PAIRING_BODY + 1 - len(body)
            if remaining <= 0:
                break
            body.extend(chunk[:remaining])
            if len(body) > MAX_PAIRING_BODY:
                raise ValueError("request too large")
        if not body:
            raise ValueError("empty request")
        payload = json.loads(
            bytes(body),
            object_pairs_hook=_unique_members,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")),
        )
        _validate_graph(payload)
        return model.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, ValidationError, RecursionError):
        raise _InvalidRequest() from None
    finally:
        body.clear()


_PUBLIC_PAIRING_FAILURES = {
    "transient", "revoked", "scope_loss", "corrupt_token", "reauth_required",
    "too_early", "race_lost", "redirect_rejected", "malformed_response", "inactive",
    "invalid_client", "expired", "refresh_outcome_unknown", "revocation_pending",
    "resource_cleanup_pending", "credential_busy", "start_outcome_unknown",
}
_PUBLIC_PAIRING_STATES = {
    "pending", "attempt_registered", "completing", "completed", "cancelled", "failed",
}


def _status(
    view: oauth_pairing.PairingView, *, include_failure: bool = False
) -> PairingStatus:
    if view.state not in _PUBLIC_PAIRING_STATES:
        raise oauth_pairing.PairingUnavailable("pairing_not_available")
    failure = view.failure_code if view.failure_code in _PUBLIC_PAIRING_FAILURES else None
    return PairingStatus(
        pairing_id=view.id,
        state=cast(PairingState, view.state),
        expires_at=view.expires_at,
        failure_code=cast(PairingFailureCode, failure) if include_failure and failure else None,
    )


def _scrub_completion_failure(error: BaseException) -> None:
    openai_oauth._discard_exception_secrets(error)


def _pairing_secret_boundary(function):
    @wraps(function)
    async def guarded(*args, **kwargs):
        http_failure: tuple[int, object] | None = None
        cancelled = False
        fatal: type[BaseException] | None = None
        internal_traceback: object = None
        try:
            return await function(*args, **kwargs)
        except HTTPException as exc:
            http_failure = _SAFE_HELPER_FAILURES.get(
                type(exc), (500, {"code": "pairing_failed"})
            )
            internal_traceback = openai_oauth._base_exception_get(exc, "__traceback__")
            _scrub_completion_failure(exc)
        except asyncio.CancelledError as exc:
            cancelled = True
            internal_traceback = openai_oauth._base_exception_get(exc, "__traceback__")
            _scrub_completion_failure(exc)
        except (SystemExit, KeyboardInterrupt) as exc:
            fatal = type(exc)
            internal_traceback = openai_oauth._base_exception_get(exc, "__traceback__")
            _scrub_completion_failure(exc)
        except BaseException as exc:  # noqa: BLE001 - public credential boundary
            http_failure = (500, {"code": "pairing_failed"})
            internal_traceback = openai_oauth._base_exception_get(exc, "__traceback__")
            _scrub_completion_failure(exc)
        finally:
            openai_oauth._clear_exception_traceback(internal_traceback)
            internal_traceback = None
            kwargs.clear()
            args = ()
        if cancelled:
            raise asyncio.CancelledError() from None
        if fatal is SystemExit:
            raise SystemExit(1) from None
        if fatal is KeyboardInterrupt:
            raise KeyboardInterrupt() from None
        assert http_failure is not None
        raise HTTPException(status_code=http_failure[0], detail=http_failure[1]) from None

    return guarded


def _configured_frontend_origin() -> str:
    try:
        return public_frontend_origin(get_settings().frontend_url)
    except ValueError:
        raise _ProviderUnavailable() from None


@router.get("/chatgpt/securo-siwc.py", include_in_schema=False)
async def download_siwc_bootstrap() -> Response:
    origin = _configured_frontend_origin()
    _, _, digest = helper_wheel()
    return Response(
        content=bootstrap_script(origin, digest),
        media_type="text/x-python",
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/chatgpt/securo-siwc.whl", include_in_schema=False)
async def download_siwc_wheel() -> Response:
    wheel, filename, digest = helper_wheel()
    return Response(
        content=wheel,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'attachment; filename="{filename}"',
            "ETag": f'"sha256:{digest}"',
        },
    )


@router.post(
    "/connections/{connection_id}/chatgpt/pairings",
    response_model=PairingStart,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(pairing_start_rate_limit)],
)
@observe_siwc(component="pairing", operation="start")
async def start_pairing(
    connection_id: uuid.UUID,
    response: Response,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    origin = _configured_frontend_origin()
    try:
        started = await oauth_pairing.start_pairing(session, connection_id, user.id)
    except oauth_pairing.PairingUnavailable:
        raise _not_available() from None
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    return PairingStart(
        pairing_id=started.id,
        secret=started.secret,
        expires_at=started.expires_at,
        helper_command=helper_command(origin, started.id),
    )


@router.get(
    "/connections/{connection_id}/chatgpt/pairings/{pairing_id}",
    response_model=PairingStatus,
)
async def user_pairing_status(
    connection_id: uuid.UUID,
    pairing_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    try:
        return _status(
            await oauth_pairing.get_user_status(session, pairing_id, connection_id, user.id),
            include_failure=True,
        )
    except oauth_pairing.PairingUnavailable:
        raise _not_available() from None


@router.delete(
    "/connections/{connection_id}/chatgpt/pairings/{pairing_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
@observe_siwc(component="pairing", operation="cancel")
async def cancel_pairing(
    connection_id: uuid.UUID,
    pairing_id: uuid.UUID,
    session: AsyncSession = Depends(get_async_session),
    user: User = Depends(current_active_user),
):
    if not await oauth_pairing.cancel_pairing(session, pairing_id, connection_id, user.id):
        raise _not_available()


@router.get(
    "/chatgpt/pairings/{pairing_id}/bootstrap",
    response_model=PairingBootstrap,
    dependencies=[Depends(pairing_helper_rate_limit)],
)
@_pairing_secret_boundary
async def helper_bootstrap(
    pairing_id: uuid.UUID,
    secret: str = Depends(_secret),
    session: AsyncSession = Depends(get_async_session),
):
    try:
        view = await oauth_pairing.get_helper_status(session, pairing_id, secret)
        return PairingBootstrap(
            **_status(view).model_dump(),
            ext_agent_host_id=view.ext_agent_host_id,
            returning_client_id=view.returning_client_id,
        )
    except oauth_pairing.PairingUnavailable:
        raise _not_available() from None


@router.get(
    "/chatgpt/pairings/{pairing_id}/status",
    response_model=PairingStatus,
    dependencies=[Depends(pairing_helper_rate_limit)],
)
@_pairing_secret_boundary
async def helper_status(
    pairing_id: uuid.UUID,
    secret: str = Depends(_secret),
    session: AsyncSession = Depends(get_async_session),
):
    try:
        return _status(await oauth_pairing.get_helper_status(session, pairing_id, secret))
    except oauth_pairing.PairingUnavailable:
        raise _not_available() from None


@router.post(
    "/chatgpt/pairings/{pairing_id}/attempt",
    response_model=AttemptRegistration,
    dependencies=[Depends(pairing_helper_rate_limit)],
)
@_pairing_secret_boundary
@observe_siwc(component="pairing", operation="attempt")
async def helper_attempt(
    pairing_id: uuid.UUID,
    request: Request,
    secret: str = Depends(_secret),
    session: AsyncSession = Depends(get_async_session),
):
    data = await _strict_body(request, AttemptRegistration)
    try:
        registered = await oauth_pairing.register_attempt(
            session, pairing_id, secret, data.attempt_id, data.nonce, data.redirect_uri
        )
    except oauth_pairing.PairingUnavailable:
        raise _not_available() from None
    return AttemptRegistration(
        attempt_id=registered.attempt_id,
        nonce=registered.nonce,
        redirect_uri=registered.redirect_uri,
    )


@router.post(
    "/chatgpt/pairings/{pairing_id}/complete",
    response_model=CompletionResult,
    dependencies=[Depends(pairing_helper_rate_limit)],
)
@_pairing_secret_boundary
@observe_siwc(component="pairing", operation="complete")
async def helper_complete(
    pairing_id: uuid.UUID,
    request: Request,
    secret: str = Depends(_secret),
    session: AsyncSession = Depends(get_async_session),
):
    data = await _strict_body(request, CompletionRequest)
    try:
        digest = oauth_pairing.completion_digest(
            data.attempt_id, data.issued_client_id, data.token_response
        )
        lease = await oauth_pairing.claim_completion(
            session, pairing_id, secret, data.attempt_id, digest
        )
        if lease == "completed":
            view = await oauth_pairing.get_helper_status(session, pairing_id, secret)
            if view.subject is None:
                raise oauth_pairing.PairingUnavailable("pairing_not_available")
            return CompletionResult(subject=view.subject)
        pairing = await oauth_pairing.assert_completion_lease(session, pairing_id, lease)
        nonce = pairing.nonce
        await session.rollback()
        if nonce is None:
            raise oauth_pairing.PairingUnavailable("pairing_not_available")
        try:
            validated = await openai_oauth.validate_transferred_token_response(
                token_response=data.token_response,
                issued_client_id=data.issued_client_id,
                nonce=nonce,
            )
        except openai_oauth.OAuthLifecycleError as exc:
            await oauth_pairing.fail_completion(
                session, pairing_id, lease, exc.code.value, retryable=exc.retryable
            )
            raise (_ProviderUnavailable() if exc.retryable else _InvalidCredential()) from None
        view = await oauth_pairing.persist_validated_completion(
            session, pairing_id, lease, data.issued_client_id, validated
        )
        return CompletionResult(subject=view.subject or "")
    except oauth_pairing.CredentialBusy:
        if "lease" in locals() and lease != "completed":
            await oauth_pairing.fail_completion(
                session, pairing_id, lease, "credential_busy", retryable=False
            )
        raise _CredentialBusy() from None
    except oauth_pairing.PairingUnavailable:
        raise _not_available() from None
