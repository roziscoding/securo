"""Database-authoritative secure remote SIWC pairing state machine."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models.connection import LlmConnection
from app.agents.models.oauth_credential import OpenAIOAuthCredential
from app.agents.models.oauth_pairing import OpenAIOAuthPairing
from app.agents.services.oauth_credentials import encrypt_bundle
from app.agents.services.openai_oauth import ISSUER, ValidatedTokenBundle
from app.agents.config import MAX_OAUTH_TIMEOUT_SECONDS
from app.core.config import get_settings
from app.core.database import async_session_maker
from app.core.secret_scrub import clear_traceback, scrub_graph

PAIRING_TTL = timedelta(minutes=10)
COMPLETION_LEASE_MARGIN_SECONDS = 30
COMPLETION_LEASE_GUARD_SECONDS = 30
COMPLETION_LEASE = timedelta(
    seconds=(
        MAX_OAUTH_TIMEOUT_SECONDS
        + COMPLETION_LEASE_MARGIN_SECONDS
        + COMPLETION_LEASE_GUARD_SECONDS
    )
)
assert COMPLETION_LEASE.total_seconds() > (
    MAX_OAUTH_TIMEOUT_SECONDS + COMPLETION_LEASE_MARGIN_SECONDS
)
assert COMPLETION_LEASE < PAIRING_TTL
_SECRET_DOMAIN = b"securo-siwc-pairing-secret-v1\0"
_COMPLETION_DOMAIN = b"securo-siwc-pairing-completion-v1\0"
_LOOPBACK = re.compile(r"^http://127\.0\.0\.1:([1-9][0-9]{0,4})/auth/callback$")
_START_CLEANUP_ATTEMPTS = 3
_START_CLEANUP_TIMEOUT_SECONDS = 5.0


class PairingUnavailable(LookupError):
    """Stable indistinguishable failure for absent, unauthorized, or unusable pairings."""


class CredentialBusy(RuntimeError):
    """Credential lifecycle state makes replacement outcome-unsafe."""


@dataclass(frozen=True)
class StartedPairing:
    id: uuid.UUID
    secret: str
    expires_at: datetime


@dataclass(frozen=True)
class RegisteredAttempt:
    attempt_id: uuid.UUID
    nonce: str
    redirect_uri: str


@dataclass(frozen=True)
class PairingView:
    id: uuid.UUID
    state: str
    expires_at: datetime
    subject: str | None = None
    ext_agent_host_id: str = ""
    returning_client_id: str | None = None
    failure_code: str | None = None


@dataclass(frozen=True)
class _StartCleanupIdentity:
    pairing_id: uuid.UUID
    connection_id: uuid.UUID
    user_id: uuid.UUID
    expected_digest: str


def _new_start_cleanup_session() -> AsyncSession:
    return async_session_maker()


def _start_row_is_scrubbed(pairing: OpenAIOAuthPairing) -> bool:
    return (
        pairing.state == "cancelled"
        and pairing.secret_digest is None
        and pairing.attempt_id is None
        and pairing.nonce is None
        and pairing.redirect_uri is None
        and pairing.completion_digest is None
        and pairing.completion_lease_token is None
        and pairing.completion_lease_expires_at is None
    )


async def _cleanup_start_pairing(identity: _StartCleanupIdentity) -> str:
    """Reconcile an unknown start commit using only non-secret immutable identity."""
    deadline = asyncio.get_running_loop().time() + _START_CLEANUP_TIMEOUT_SECONDS
    for attempt in range(_START_CLEANUP_ATTEMPTS):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return "unresolved"
        try:
            async with asyncio.timeout(remaining):
                async with _new_start_cleanup_session() as cleanup_session:
                    pairing = await cleanup_session.scalar(
                        select(OpenAIOAuthPairing)
                        .where(
                            OpenAIOAuthPairing.id == identity.pairing_id,
                            OpenAIOAuthPairing.connection_id == identity.connection_id,
                            OpenAIOAuthPairing.user_id == identity.user_id,
                        )
                        .with_for_update()
                    )
                    if pairing is None:
                        await cleanup_session.rollback()
                        return "absent"
                    if _start_row_is_scrubbed(pairing):
                        await cleanup_session.rollback()
                        return "cancelled"
                    if (
                        pairing.secret_digest != identity.expected_digest
                        or pairing.state not in {"pending", "attempt_registered"}
                    ):
                        await cleanup_session.rollback()
                        return "unexpected_terminal"
                    pairing.state = "cancelled"
                    pairing.secret_digest = None
                    pairing.attempt_id = None
                    pairing.nonce = None
                    pairing.redirect_uri = None
                    pairing.completion_digest = None
                    pairing.completion_lease_token = None
                    pairing.completion_lease_expires_at = None
                    pairing.failure_code = "start_outcome_unknown"
                    pairing.expires_at = _db_now(cleanup_session)
                    await cleanup_session.commit()
                    await cleanup_session.refresh(pairing)
                    return "cancelled" if _start_row_is_scrubbed(pairing) else "unresolved"
        except BaseException as cleanup_error:
            scrub_graph(cleanup_error)
        if attempt + 1 < _START_CLEANUP_ATTEMPTS:
            await asyncio.sleep(0.05 * (attempt + 1))
    return "unresolved"


async def _await_start_cleanup(
    identity: _StartCleanupIdentity,
) -> asyncio.CancelledError | None:
    """Finish independently-owned cleanup despite cancellation of the caller."""
    task = asyncio.create_task(
        _cleanup_start_pairing(identity), name="oauth-pairing-start-cleanup"
    )
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
        except BaseException:
            break
    try:
        task.result()
    except BaseException as cleanup_error:
        scrub_graph(cleanup_error)
    return cancellation


def _key() -> bytes:
    return get_settings().secret_key.get_secret_value().encode("utf-8")


def _digest(domain: bytes, value: bytes | bytearray) -> str:
    digest = hmac.new(_key(), digestmod=hashlib.sha256)
    digest.update(domain)
    digest.update(value)
    return digest.hexdigest()


def secret_digest(secret: str) -> str:
    try:
        raw = base64.urlsafe_b64decode(secret + "=" * (-len(secret) % 4))
    except (ValueError, UnicodeEncodeError):
        raw = b""
    if (
        len(raw) != 32
        or base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != secret
    ):
        return _digest(_SECRET_DOMAIN, b"")
    return _digest(_SECRET_DOMAIN, raw)


def secret_matches(secret: str, expected: str | None) -> bool:
    if expected is None or not 40 <= len(secret) <= 256 or not secret.isascii():
        return False
    return hmac.compare_digest(secret_digest(secret), expected)


def completion_digest(attempt_id: uuid.UUID, issued_client_id: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"attempt_id": str(attempt_id), "issued_client_id": issued_client_id, "token_response": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return _digest(_COMPLETION_DOMAIN, canonical)


def _host_id() -> str:
    """Return the opaque, installation-wide SIWC host identity.

    Rotating the application secret intentionally rotates this identifier.
    """
    raw = bytearray.fromhex(
        _digest(b"securo-siwc-installation-host-id-v1\0", b"")[:32]
    )
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return f"urn:uuid:{uuid.UUID(bytes=bytes(raw))}"


def _credential_is_pairable(credential: OpenAIOAuthCredential | None) -> bool:
    return credential is None or (
        credential.status in {"revoked", "reauth_required"}
        and credential.token_bundle_encrypted is None
        and credential.lifecycle_lease_token is None
        and credential.lifecycle_lease_kind is None
        and credential.lifecycle_lease_expires_at is None
        and credential.refresh_attempt_id is None
        and credential.refresh_attempt_started_at is None
        and credential.refresh_restore_status is None
        and credential.last_error_code
        not in {
            "refresh_outcome_unknown",
            "revocation_pending",
            "resource_cleanup_pending",
        }
    )


def _db_now(session: AsyncSession):
    return func.clock_timestamp() if session.get_bind().dialect.name == "postgresql" else func.datetime("now")


def _db_plus(session: AsyncSession, seconds: int):
    if session.get_bind().dialect.name == "postgresql":
        return _db_now(session) + timedelta(seconds=seconds)
    return func.datetime("now", f"+{seconds} seconds")


def _aware(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


async def start_pairing(
    session: AsyncSession, connection_id: uuid.UUID, user_id: uuid.UUID
) -> StartedPairing:
    connection = await session.scalar(
        select(LlmConnection).where(
            LlmConnection.id == connection_id, LlmConnection.user_id == user_id
        ).with_for_update()
    )
    if connection is None or connection.kind != "openai_chatgpt":
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    existing_pairing = await session.scalar(
        select(OpenAIOAuthPairing)
        .where(OpenAIOAuthPairing.connection_id == connection_id)
        .with_for_update()
    )
    if (
        existing_pairing is not None
        and existing_pairing.state == "completing"
        and existing_pairing.completion_lease_expires_at is not None
        and await session.scalar(
            select(OpenAIOAuthPairing.id).where(
                OpenAIOAuthPairing.id == existing_pairing.id,
                OpenAIOAuthPairing.completion_lease_expires_at > _db_now(session),
            )
        )
        is not None
    ):
        await session.rollback()
        raise PairingUnavailable("connection_not_pairable")
    credential = await session.scalar(
        select(OpenAIOAuthCredential)
        .where(OpenAIOAuthCredential.connection_id == connection_id)
        .with_for_update()
    )
    if not _credential_is_pairable(credential):
        await session.rollback()
        raise PairingUnavailable("connection_not_pairable")
    await session.execute(
        delete(OpenAIOAuthPairing).where(OpenAIOAuthPairing.connection_id == connection_id)
    )
    raw_secret = bytearray()
    failure_kind: str | None = None
    internal_traceback: object = None
    pairing: OpenAIOAuthPairing | None = None
    pairing_id: uuid.UUID | None = None
    expires_at: datetime | None = None
    expected_digest: str | None = None
    try:
        for _ in range(32):
            raw_secret.append(secrets.randbits(8))
        expected_digest = _digest(_SECRET_DOMAIN, raw_secret)
        pairing = OpenAIOAuthPairing(
            connection_id=connection_id,
            user_id=user_id,
            secret_digest=expected_digest,
            state="pending",
            expires_at=_db_plus(session, int(PAIRING_TTL.total_seconds())),
        )
        session.add(pairing)
        await session.flush()
        await session.refresh(pairing)
        pairing_id = pairing.id
        expires_at = _aware(pairing.expires_at)
        await session.commit()
    except asyncio.CancelledError as exc:
        failure_kind = "cancelled"
        internal_traceback = exc.__traceback__
        scrub_graph(exc)
    except (KeyboardInterrupt, SystemExit) as exc:
        failure_kind = "fatal_interrupt" if isinstance(exc, KeyboardInterrupt) else "fatal_exit"
        internal_traceback = exc.__traceback__
        scrub_graph(exc)
    except BaseException as exc:
        failure_kind = "failure"
        internal_traceback = exc.__traceback__
        scrub_graph(exc)

    if failure_kind is not None:
        raw_secret.clear()
        pairing = None
        clear_traceback(internal_traceback)
        internal_traceback = None
        try:
            await session.rollback()
        except BaseException as cleanup_error:
            scrub_graph(cleanup_error)
        cleanup_cancellation = None
        if pairing_id is not None and expected_digest is not None:
            cleanup_cancellation = await _await_start_cleanup(
                _StartCleanupIdentity(
                    pairing_id, connection_id, user_id, expected_digest
                )
            )
        expected_digest = None
        if cleanup_cancellation is not None:
            raise asyncio.CancelledError() from None
        if failure_kind == "cancelled":
            raise asyncio.CancelledError() from None
        if failure_kind == "fatal_interrupt":
            raise KeyboardInterrupt() from None
        if failure_kind == "fatal_exit":
            raise SystemExit(1) from None
        raise PairingUnavailable("pairing_not_available") from None

    assert pairing_id is not None and expires_at is not None
    post_commit_kind: str | None = None
    try:
        encoded_secret = base64.urlsafe_b64encode(raw_secret).rstrip(b"=").decode("ascii")
        if len(encoded_secret) != 43:
            raise RuntimeError("pairing secret encoding failed")
        result = StartedPairing(pairing_id, encoded_secret, expires_at)
    except BaseException as exc:
        post_commit_kind = (
            "cancelled" if isinstance(exc, asyncio.CancelledError)
            else "fatal_interrupt" if isinstance(exc, KeyboardInterrupt)
            else "fatal_exit" if isinstance(exc, SystemExit)
            else "failure"
        )
        post_commit_traceback = exc.__traceback__
        scrub_graph(exc)
        raw_secret.clear()
        encoded_secret = ""
        result = None
        try:
            await session.execute(
                update(OpenAIOAuthPairing)
                .where(OpenAIOAuthPairing.id == pairing_id)
                .values(state="cancelled", secret_digest=None, expires_at=_db_now(session))
            )
            await session.commit()
        except BaseException as cleanup_error:
            scrub_graph(cleanup_error)
        clear_traceback(post_commit_traceback)
        post_commit_traceback = None
    finally:
        raw_secret.clear()
    if post_commit_kind == "cancelled":
        raise asyncio.CancelledError() from None
    if post_commit_kind == "fatal_interrupt":
        raise KeyboardInterrupt() from None
    if post_commit_kind == "fatal_exit":
        raise SystemExit(1) from None
    if post_commit_kind is not None:
        raise PairingUnavailable("pairing_not_available") from None
    assert result is not None
    return result


async def _helper_pairing(
    session: AsyncSession, pairing_id: uuid.UUID, secret: str, *, lock: bool = False
) -> OpenAIOAuthPairing:
    statement = select(OpenAIOAuthPairing).where(
        OpenAIOAuthPairing.id == pairing_id,
        OpenAIOAuthPairing.expires_at > _db_now(session),
    )
    if lock:
        statement = statement.with_for_update()
    pairing = await session.scalar(statement)
    if pairing is None:
        await session.execute(
            update(OpenAIOAuthPairing)
            .where(
                OpenAIOAuthPairing.id == pairing_id,
                OpenAIOAuthPairing.expires_at <= _db_now(session),
            )
            .values(secret_digest=None, nonce=None)
        )
        await session.commit()
        raise PairingUnavailable("pairing_not_available")
    if not secret_matches(secret, pairing.secret_digest):
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    return pairing


async def _lock_connection_then_pairing(
    session: AsyncSession,
    pairing_id: uuid.UUID,
) -> tuple[LlmConnection, OpenAIOAuthPairing]:
    identity = (
        await session.execute(
            select(OpenAIOAuthPairing.connection_id, OpenAIOAuthPairing.user_id).where(
                OpenAIOAuthPairing.id == pairing_id
            )
        )
    ).one_or_none()
    if identity is None:
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    connection = await session.scalar(
        select(LlmConnection)
        .where(
            LlmConnection.id == identity.connection_id,
            LlmConnection.user_id == identity.user_id,
            LlmConnection.kind == "openai_chatgpt",
        )
        .with_for_update()
    )
    if connection is None:
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    pairing = await session.scalar(
        select(OpenAIOAuthPairing)
        .where(
            OpenAIOAuthPairing.id == pairing_id,
            OpenAIOAuthPairing.connection_id == connection.id,
            OpenAIOAuthPairing.user_id == connection.user_id,
        )
        .with_for_update()
    )
    if pairing is None:
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    return connection, pairing


async def get_helper_status(
    session: AsyncSession, pairing_id: uuid.UUID, secret: str
) -> PairingView:
    pairing = await _helper_pairing(session, pairing_id, secret)
    credential = await session.scalar(
        select(OpenAIOAuthCredential).where(
            OpenAIOAuthCredential.connection_id == pairing.connection_id
        )
    )
    returning = (
        credential.issued_client_id
        if credential is not None and _credential_is_pairable(credential)
        else None
    )
    return PairingView(
        pairing.id,
        pairing.state,
        _aware(pairing.expires_at),
        pairing.result_subject,
        _host_id(),
        returning,
        pairing.failure_code,
    )


async def get_user_status(
    session: AsyncSession, pairing_id: uuid.UUID, connection_id: uuid.UUID, user_id: uuid.UUID
) -> PairingView:
    pairing = await session.scalar(
        select(OpenAIOAuthPairing).where(
            OpenAIOAuthPairing.id == pairing_id,
            OpenAIOAuthPairing.connection_id == connection_id,
            OpenAIOAuthPairing.user_id == user_id,
            OpenAIOAuthPairing.expires_at > _db_now(session),
        )
    )
    if pairing is None:
        raise PairingUnavailable("pairing_not_available")
    return PairingView(
        pairing.id,
        pairing.state,
        _aware(pairing.expires_at),
        pairing.result_subject,
        failure_code=pairing.failure_code,
    )


async def register_attempt(
    session: AsyncSession,
    pairing_id: uuid.UUID,
    secret: str,
    attempt_id: uuid.UUID,
    nonce: str,
    redirect_uri: str,
) -> RegisteredAttempt:
    if not 32 <= len(nonce) <= 256 or not nonce.isascii():
        raise PairingUnavailable("pairing_not_available")
    match = _LOOPBACK.fullmatch(redirect_uri)
    if match is None or int(match.group(1)) > 65535:
        raise PairingUnavailable("pairing_not_available")
    _, pairing = await _lock_connection_then_pairing(session, pairing_id)
    live = await session.scalar(
        select(OpenAIOAuthPairing.id).where(
            OpenAIOAuthPairing.id == pairing.id,
            OpenAIOAuthPairing.expires_at > _db_now(session),
        )
    )
    if live is None or not secret_matches(secret, pairing.secret_digest):
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    if pairing.state == "attempt_registered":
        if (
            pairing.attempt_id == attempt_id
            and hmac.compare_digest(pairing.nonce or "", nonce)
            and hmac.compare_digest(pairing.redirect_uri or "", redirect_uri)
        ):
            await session.rollback()
            return RegisteredAttempt(attempt_id, nonce, redirect_uri)
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    if pairing.state != "pending":
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    pairing.state = "attempt_registered"
    pairing.attempt_id = attempt_id
    pairing.nonce = nonce
    pairing.redirect_uri = redirect_uri
    await session.commit()
    return RegisteredAttempt(attempt_id, nonce, redirect_uri)


async def claim_completion(
    session: AsyncSession,
    pairing_id: uuid.UUID,
    secret: str,
    attempt_id: uuid.UUID,
    payload_digest: str,
) -> str:
    _, pairing = await _lock_connection_then_pairing(session, pairing_id)
    live = await session.scalar(
        select(OpenAIOAuthPairing.id).where(
            OpenAIOAuthPairing.id == pairing.id,
            OpenAIOAuthPairing.expires_at
            > _db_plus(session, int(COMPLETION_LEASE.total_seconds())),
        )
    )
    if live is None or not secret_matches(secret, pairing.secret_digest):
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    if pairing.attempt_id != attempt_id:
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    if pairing.state == "completed":
        completed_digest = pairing.completion_digest
        await session.rollback()
        if completed_digest == payload_digest:
            return "completed"
        raise PairingUnavailable("pairing_not_available")
    now = _db_now(session)
    if pairing.state == "completing" and pairing.completion_lease_expires_at is not None:
        live_id = await session.scalar(
            select(OpenAIOAuthPairing.id).where(
                OpenAIOAuthPairing.id == pairing.id,
                OpenAIOAuthPairing.completion_lease_expires_at > now,
            )
        )
        if live_id is not None:
            await session.rollback()
            raise CredentialBusy("completion_in_progress")
    elif pairing.state != "attempt_registered":
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    if pairing.completion_digest not in (None, payload_digest):
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    lease = str(uuid.uuid4())
    pairing.state = "completing"
    pairing.completion_digest = payload_digest
    pairing.completion_lease_token = lease
    pairing.completion_lease_expires_at = _db_plus(
        session, int(COMPLETION_LEASE.total_seconds())
    )
    await session.commit()
    return lease


async def assert_completion_lease(
    session: AsyncSession, pairing_id: uuid.UUID, lease: str
) -> OpenAIOAuthPairing:
    pairing = await session.scalar(
        select(OpenAIOAuthPairing).where(
            OpenAIOAuthPairing.id == pairing_id,
            OpenAIOAuthPairing.state == "completing",
            OpenAIOAuthPairing.completion_lease_token == lease,
            OpenAIOAuthPairing.completion_lease_expires_at > _db_now(session),
            OpenAIOAuthPairing.expires_at > _db_now(session),
        ).with_for_update()
    )
    if pairing is None:
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    return pairing


async def cancel_pairing(
    session: AsyncSession, pairing_id: uuid.UUID, connection_id: uuid.UUID, user_id: uuid.UUID
) -> bool:
    try:
        connection, _pairing = await _lock_connection_then_pairing(session, pairing_id)
    except PairingUnavailable:
        return False
    if connection.id != connection_id or connection.user_id != user_id:
        await session.rollback()
        return False
    result = await session.execute(
        update(OpenAIOAuthPairing)
        .where(
            OpenAIOAuthPairing.id == pairing_id,
            OpenAIOAuthPairing.connection_id == connection_id,
            OpenAIOAuthPairing.user_id == user_id,
            OpenAIOAuthPairing.expires_at > _db_now(session),
            OpenAIOAuthPairing.state.in_({"pending", "attempt_registered", "completing"}),
        )
        .values(
            state="cancelled", secret_digest=None, nonce=None,
            completion_lease_token=None, completion_lease_expires_at=None,
        )
        .returning(OpenAIOAuthPairing.id)
    )
    cancelled = result.scalar_one_or_none() is not None
    await session.commit()
    return cancelled


async def fail_completion(
    session: AsyncSession, pairing_id: uuid.UUID, lease: str, code: str, *, retryable: bool
) -> None:
    values: dict[str, Any]
    if retryable:
        values = {
            "state": "attempt_registered", "completion_lease_token": None,
            "completion_lease_expires_at": None, "failure_code": code,
        }
    else:
        values = {
            "state": "failed", "secret_digest": None, "nonce": None,
            "completion_lease_token": None, "completion_lease_expires_at": None,
            "failure_code": code,
        }
    await session.execute(
        update(OpenAIOAuthPairing).where(
            OpenAIOAuthPairing.id == pairing_id,
            OpenAIOAuthPairing.state == "completing",
            OpenAIOAuthPairing.completion_lease_token == lease,
            OpenAIOAuthPairing.completion_lease_expires_at > _db_now(session),
        ).values(**values)
    )
    await session.commit()


async def persist_validated_completion(
    session: AsyncSession,
    pairing_id: uuid.UUID,
    lease: str,
    issued_client_id: str,
    validated: ValidatedTokenBundle,
) -> PairingView:
    connection, pairing = await _lock_connection_then_pairing(session, pairing_id)
    lease_is_live = await session.scalar(
        select(OpenAIOAuthPairing.id).where(
            OpenAIOAuthPairing.id == pairing_id,
            OpenAIOAuthPairing.state == "completing",
            OpenAIOAuthPairing.completion_lease_token == lease,
            OpenAIOAuthPairing.completion_lease_expires_at > _db_now(session),
            OpenAIOAuthPairing.expires_at > _db_now(session),
        )
    )
    if lease_is_live is None:
        await session.rollback()
        raise PairingUnavailable("pairing_not_available")
    credential = await session.scalar(
        select(OpenAIOAuthCredential).where(
            OpenAIOAuthCredential.connection_id == connection.id
        ).with_for_update()
    )
    if not _credential_is_pairable(credential):
        await session.rollback()
        raise CredentialBusy("credential_busy")
    if credential is not None and credential.subject != validated.subject:
        await session.rollback()
        raise PairingUnavailable("identity_mismatch")
    ciphertext = encrypt_bundle(validated.bundle)
    values = dict(
        issuer=ISSUER,
        subject=validated.subject,
        issued_client_id=issued_client_id,
        account_email=validated.account_email,
        scopes=sorted(validated.scopes),
        status="active",
        token_bundle_encrypted=ciphertext,
        access_token_expires_at=validated.access_token_expires_at,
        earliest_refresh_at=validated.earliest_refresh_at,
        refresh_token_expires_at=validated.refresh_token_expires_at,
        last_error_code=None,
        refresh_attempt_count=0,
        refresh_retry_after=None,
        refresh_attempt_id=None,
        refresh_attempt_started_at=None,
        refresh_restore_status=None,
        lifecycle_lease_token=None,
        lifecycle_lease_kind=None,
        lifecycle_lease_expires_at=None,
    )
    if credential is None:
        credential = OpenAIOAuthCredential(connection_id=connection.id, credential_version=1, **values)
        session.add(credential)
    else:
        for name, value in values.items():
            setattr(credential, name, value)
        credential.credential_version += 1
    pairing.state = "completed"
    pairing.result_subject = validated.subject
    pairing.completed_at = _db_now(session)
    pairing.nonce = None
    pairing.completion_lease_token = None
    pairing.completion_lease_expires_at = None
    await session.commit()
    return PairingView(pairing.id, "completed", _aware(pairing.expires_at), validated.subject)
