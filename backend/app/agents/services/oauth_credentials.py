"""Encrypted, ownership-scoped storage for OpenAI SIWC credentials."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import Interval, bindparam, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models.connection import LlmConnection
from app.agents.models.oauth_credential import OpenAIOAuthCredential
from app.agents.services.crypto import decrypt, encrypt


CREDENTIAL_STATUSES = frozenset(
    {"active", "disabled", "revoked", "reauth_required"}
)


def sanitize_account_email(value: object) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 320 or value.count("@") != 1:
        return None
    local, domain = value.split("@")
    if not local or len(local) > 64 or not domain or len(domain) > 255:
        return None
    if any(character.isspace() or ord(character) < 32 for character in value):
        return None
    return value


class CredentialAlreadyExistsError(RuntimeError):
    """The connection already owns a credential; safe to expose to callers."""


def _validate_status(status: str) -> None:
    if status not in CREDENTIAL_STATUSES:
        raise ValueError(f"invalid credential status: {status}")


@dataclass(frozen=True)
class CredentialBundle:
    """The rotating token set, deliberately opaque in repr and logs."""

    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    id_token: str = field(repr=False)


_BUNDLE_FIELDS = frozenset({"access_token", "refresh_token", "id_token"})


def _validated_bundle_data(data: object) -> dict[str, str]:
    if not isinstance(data, dict) or set(data) != _BUNDLE_FIELDS:
        raise ValueError("credential bundle is invalid")
    if any(
        not isinstance(data[field], str)
        or not data[field]
        or len(data[field]) > 65_536
        for field in _BUNDLE_FIELDS
    ):
        raise ValueError("credential bundle is invalid")
    return {field: data[field] for field in _BUNDLE_FIELDS}


def encrypt_bundle(bundle: CredentialBundle) -> str:
    if type(bundle) is not CredentialBundle:
        raise ValueError("credential bundle is invalid")
    data = _validated_bundle_data(
        {
            "access_token": bundle.access_token,
            "refresh_token": bundle.refresh_token,
            "id_token": bundle.id_token,
        }
    )
    plaintext = json.dumps(data, sort_keys=True, separators=(",", ":"))
    ciphertext = encrypt(plaintext)
    if ciphertext is None:  # pragma: no cover - a non-empty JSON object cannot encrypt to None
        raise ValueError("credential bundle encryption failed")
    return ciphertext


def _unique_bundle_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("credential bundle has duplicate members")
        result[key] = value
    return result


def decrypt_bundle(ciphertext: str | None) -> CredentialBundle:
    if ciphertext is None:
        raise ValueError("credential bundle cannot be decrypted")
    plaintext = decrypt(ciphertext)
    if plaintext is None:
        raise ValueError("credential bundle cannot be decrypted")
    try:
        data = json.loads(
            plaintext,
            object_pairs_hook=_unique_bundle_members,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("credential bundle is invalid") from exc
    validated = _validated_bundle_data(data)
    return CredentialBundle(**validated)


async def get_owned_credential(
    session: AsyncSession, connection_id: uuid.UUID, user_id: uuid.UUID
) -> OpenAIOAuthCredential | None:
    return (
        await session.execute(
            select(OpenAIOAuthCredential)
            .join(LlmConnection)
            .where(
                OpenAIOAuthCredential.connection_id == connection_id,
                LlmConnection.user_id == user_id,
            )
        )
    ).scalar_one_or_none()


async def create_credential(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    bundle: CredentialBundle,
    issuer: str,
    subject: str,
    issued_client_id: str,
    account_email: str | None,
    scopes: list[str],
    access_token_expires_at: datetime,
    earliest_refresh_at: datetime | None = None,
    refresh_token_expires_at: datetime | None = None,
    status: str = "active",
) -> OpenAIOAuthCredential:
    _validate_status(status)
    from app.agents.services.connection_lock import lock_owned_connection

    connection = await lock_owned_connection(session, connection_id, user_id)
    if connection is None:
        raise LookupError("connection not found")
    if connection.kind != "openai_chatgpt":
        raise ValueError("OAuth credentials require an openai_chatgpt connection")

    existing = await get_owned_credential(session, connection_id, user_id)
    if existing is not None:
        await session.rollback()
        raise CredentialAlreadyExistsError("OAuth credential already exists")

    credential = OpenAIOAuthCredential(
        connection=connection,
        issuer=issuer,
        subject=subject,
        issued_client_id=issued_client_id,
        account_email=sanitize_account_email(account_email),
        scopes=list(scopes),
        status=status,
        token_bundle_encrypted=encrypt_bundle(bundle),
        access_token_expires_at=access_token_expires_at,
        earliest_refresh_at=earliest_refresh_at,
        refresh_token_expires_at=refresh_token_expires_at,
        credential_version=1,
    )
    session.add(credential)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise CredentialAlreadyExistsError("OAuth credential already exists") from None
    await session.refresh(credential)
    return credential


class StaleCredentialVersionError(RuntimeError):
    """A newer rotating bundle won the race; the caller must not overwrite it."""


async def claim_credential_lease(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    expected_version: int,
    lease_token: str,
    operation: str,
    lease_duration: timedelta,
    allowed_statuses: frozenset[str] = frozenset({"active"}),
) -> bool:
    """Commit a short cross-process claim before any provider network I/O."""
    duration_seconds = lease_duration.total_seconds()
    if duration_seconds <= 0 or duration_seconds != int(duration_seconds):
        raise ValueError("lease duration must be a positive whole number of seconds")
    if not allowed_statuses or not allowed_statuses.issubset(CREDENTIAL_STATUSES):
        raise ValueError("lease statuses are invalid")
    dialect_name = session.get_bind().dialect.name
    if dialect_name == "postgresql":
        database_now = func.now()
        database_expiry = database_now + bindparam(
            "oauth_lease_duration", lease_duration, type_=Interval()
        )
    elif dialect_name == "sqlite":
        database_now = func.datetime("now")
        database_expiry = func.datetime(
            "now",
            func.printf(
                "+%d seconds", bindparam("oauth_lease_seconds", int(duration_seconds))
            ),
        )
    else:  # pragma: no cover - production and tests use the two dialects above
        raise RuntimeError(f"unsupported database dialect for OAuth leases: {dialect_name}")
    owned_connection = select(LlmConnection.id).where(
        LlmConnection.id == connection_id, LlmConnection.user_id == user_id
    )
    result = await session.execute(
        update(OpenAIOAuthCredential)
        .where(
            OpenAIOAuthCredential.connection_id == connection_id,
            OpenAIOAuthCredential.connection_id.in_(owned_connection),
            OpenAIOAuthCredential.credential_version == expected_version,
            OpenAIOAuthCredential.status.in_(allowed_statuses),
            OpenAIOAuthCredential.token_bundle_encrypted.is_not(None),
            or_(
                OpenAIOAuthCredential.lifecycle_lease_token.is_(None),
                OpenAIOAuthCredential.lifecycle_lease_expires_at <= database_now,
            ),
        )
        .values(
            lifecycle_lease_token=lease_token,
            lifecycle_lease_kind=operation,
            lifecycle_lease_expires_at=database_expiry,
        )
        .returning(OpenAIOAuthCredential.id)
    )
    claimed = result.scalar_one_or_none() is not None
    await session.commit()
    return claimed


async def begin_refresh_attempt(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    expected_version: int,
    attempt_id: str,
    lease_duration: timedelta,
) -> bool:
    """Commit non-retryable refresh intent before the token can leave the process."""
    duration_seconds = lease_duration.total_seconds()
    if duration_seconds <= 0 or duration_seconds != int(duration_seconds):
        raise ValueError("lease duration must be a positive whole number of seconds")
    dialect_name = session.get_bind().dialect.name
    if dialect_name == "postgresql":
        database_now = func.now()
        database_expiry = database_now + bindparam(
            "oauth_refresh_lease_duration", lease_duration, type_=Interval()
        )
    elif dialect_name == "sqlite":
        database_now = func.datetime("now")
        database_expiry = func.datetime(
            "now",
            func.printf(
                "+%d seconds", bindparam("oauth_refresh_lease_seconds", int(duration_seconds))
            ),
        )
    else:  # pragma: no cover
        raise RuntimeError(f"unsupported database dialect for OAuth leases: {dialect_name}")
    owned_connection = select(LlmConnection.id).where(
        LlmConnection.id == connection_id, LlmConnection.user_id == user_id
    )
    result = await session.execute(
        update(OpenAIOAuthCredential)
        .where(
            OpenAIOAuthCredential.connection_id == connection_id,
            OpenAIOAuthCredential.connection_id.in_(owned_connection),
            OpenAIOAuthCredential.credential_version == expected_version,
            OpenAIOAuthCredential.status == "active",
            OpenAIOAuthCredential.token_bundle_encrypted.is_not(None),
            OpenAIOAuthCredential.refresh_attempt_id.is_(None),
            or_(
                OpenAIOAuthCredential.lifecycle_lease_token.is_(None),
                OpenAIOAuthCredential.lifecycle_lease_expires_at <= database_now,
            ),
        )
        .values(
            status="reauth_required",
            last_error_code="refresh_outcome_unknown",
            refresh_attempt_id=attempt_id,
            refresh_attempt_started_at=database_now,
            refresh_restore_status="active",
            lifecycle_lease_token=attempt_id,
            lifecycle_lease_kind="refresh",
            lifecycle_lease_expires_at=database_expiry,
        )
        .returning(OpenAIOAuthCredential.id)
    )
    claimed = result.scalar_one_or_none() is not None
    await session.commit()
    return claimed


async def begin_revocation_attempt(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    expected_version: int,
    lease_token: str,
    lease_duration: timedelta,
) -> bool:
    """Commit revocation-pending state before provider network I/O."""
    duration_seconds = lease_duration.total_seconds()
    if duration_seconds <= 0 or duration_seconds != int(duration_seconds):
        raise ValueError("lease duration must be a positive whole number of seconds")
    dialect_name = session.get_bind().dialect.name
    if dialect_name == "postgresql":
        database_now = func.now()
        database_expiry = database_now + bindparam(
            "oauth_revoke_lease_duration", lease_duration, type_=Interval()
        )
    elif dialect_name == "sqlite":
        database_now = func.datetime("now")
        database_expiry = func.datetime(
            "now",
            func.printf(
                "+%d seconds", bindparam("oauth_revoke_lease_seconds", int(duration_seconds))
            ),
        )
    else:  # pragma: no cover
        raise RuntimeError(f"unsupported database dialect for OAuth leases: {dialect_name}")
    owned_connection = select(LlmConnection.id).where(
        LlmConnection.id == connection_id, LlmConnection.user_id == user_id
    )
    result = await session.execute(
        update(OpenAIOAuthCredential)
        .where(
            OpenAIOAuthCredential.connection_id == connection_id,
            OpenAIOAuthCredential.connection_id.in_(owned_connection),
            OpenAIOAuthCredential.credential_version == expected_version,
            OpenAIOAuthCredential.status.in_({"active", "disabled", "reauth_required"}),
            OpenAIOAuthCredential.token_bundle_encrypted.is_not(None),
            or_(
                OpenAIOAuthCredential.lifecycle_lease_token.is_(None),
                OpenAIOAuthCredential.lifecycle_lease_expires_at <= database_now,
            ),
        )
        .values(
            status="reauth_required",
            last_error_code="revocation_pending",
            lifecycle_lease_token=lease_token,
            lifecycle_lease_kind="revoke",
            lifecycle_lease_expires_at=database_expiry,
        )
        .returning(OpenAIOAuthCredential.id)
    )
    claimed = result.scalar_one_or_none() is not None
    await session.commit()
    return claimed


def _live_attempt_lease(session: AsyncSession, token: str):
    database_now = (
        func.clock_timestamp() if session.get_bind().dialect.name == "postgresql"
        else func.datetime("now")
    )
    return (
        OpenAIOAuthCredential.lifecycle_lease_token == token,
        OpenAIOAuthCredential.lifecycle_lease_expires_at > database_now,
        or_(
            OpenAIOAuthCredential.lifecycle_lease_kind != "refresh",
            OpenAIOAuthCredential.refresh_attempt_id == token,
        ),
    )


async def release_credential_lease(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    expected_version: int,
    lease_token: str,
    last_error_code: str | None = None,
) -> None:
    """Release a failed operation only if this worker still owns its claim."""
    owned_connection = select(LlmConnection.id).where(
        LlmConnection.id == connection_id, LlmConnection.user_id == user_id
    )
    await session.execute(
        update(OpenAIOAuthCredential)
        .where(
            OpenAIOAuthCredential.connection_id == connection_id,
            OpenAIOAuthCredential.connection_id.in_(owned_connection),
            OpenAIOAuthCredential.credential_version == expected_version,
            *_live_attempt_lease(session, lease_token),
        )
        .values(
            lifecycle_lease_token=None,
            lifecycle_lease_kind=None,
            lifecycle_lease_expires_at=None,
            last_error_code=last_error_code,
        )
    )
    await session.commit()


async def rotate_credential(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    expected_version: int,
    bundle: CredentialBundle,
    access_token_expires_at: datetime,
    earliest_refresh_at: datetime | None = None,
    refresh_token_expires_at: datetime | None = None,
    last_refreshed_at: datetime | None = None,
    status: str = "active",
    last_error_code: str | None = None,
    expected_lease_token: str | None = None,
    expected_attempt_id: str | None = None,
    subject: str | None = None,
    account_email: str | None = None,
    scopes: list[str] | None = None,
) -> OpenAIOAuthCredential:
    _validate_status(status)
    owned_connection = select(LlmConnection.id).where(
        LlmConnection.id == connection_id, LlmConnection.user_id == user_id
    )
    statement = update(OpenAIOAuthCredential).where(
        OpenAIOAuthCredential.connection_id == connection_id,
        OpenAIOAuthCredential.connection_id.in_(owned_connection),
        OpenAIOAuthCredential.credential_version == expected_version,
    )
    if expected_lease_token is not None:
        statement = statement.where(*_live_attempt_lease(session, expected_lease_token))
    if expected_attempt_id is not None:
        if expected_lease_token != expected_attempt_id:
            raise StaleCredentialVersionError("credential attempt ownership is stale")
        statement = statement.where(
            OpenAIOAuthCredential.refresh_attempt_id == expected_attempt_id,
            OpenAIOAuthCredential.status == "reauth_required",
            OpenAIOAuthCredential.last_error_code == "refresh_outcome_unknown",
        )
    values = {
        "token_bundle_encrypted": encrypt_bundle(bundle),
        "access_token_expires_at": access_token_expires_at,
        "earliest_refresh_at": earliest_refresh_at,
        "refresh_token_expires_at": refresh_token_expires_at,
        "last_refreshed_at": last_refreshed_at,
        "status": status,
        "last_error_code": last_error_code,
        "refresh_attempt_count": 0,
        "refresh_retry_after": None,
        "refresh_attempt_id": None,
        "refresh_attempt_started_at": None,
        "refresh_restore_status": None,
        "credential_version": OpenAIOAuthCredential.credential_version + 1,
        "lifecycle_lease_token": None,
        "lifecycle_lease_kind": None,
        "lifecycle_lease_expires_at": None,
    }
    if subject is not None:
        values["subject"] = subject
        values["account_email"] = sanitize_account_email(account_email)
    if scopes is not None:
        values["scopes"] = list(scopes)
    result = await session.execute(
        statement.values(**values).returning(OpenAIOAuthCredential.id)
    )
    if result.scalar_one_or_none() is None:
        current = await get_owned_credential(session, connection_id, user_id)
        if current is None:
            raise LookupError("connection not found")
        raise StaleCredentialVersionError("credential version is stale")

    await session.commit()
    credential = await get_owned_credential(session, connection_id, user_id)
    if credential is None:  # pragma: no cover - protected by the atomic UPDATE
        raise LookupError("connection not found")
    await session.commit()
    return credential


async def transition_credential_status(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    expected_version: int,
    status: str,
    last_error_code: str | None = None,
    expected_lease_token: str | None = None,
) -> OpenAIOAuthCredential:
    _validate_status(status)
    owned_connection = select(LlmConnection.id).where(
        LlmConnection.id == connection_id, LlmConnection.user_id == user_id
    )
    statement = update(OpenAIOAuthCredential).where(
        OpenAIOAuthCredential.connection_id == connection_id,
        OpenAIOAuthCredential.connection_id.in_(owned_connection),
        OpenAIOAuthCredential.credential_version == expected_version,
    )
    if expected_lease_token is not None:
        statement = statement.where(
            *_live_attempt_lease(session, expected_lease_token)
        )
    else:
        database_now = (
            func.now()
            if session.get_bind().dialect.name == "postgresql"
            else func.datetime("now")
        )
        statement = statement.where(
            or_(
                OpenAIOAuthCredential.lifecycle_lease_token.is_(None),
                OpenAIOAuthCredential.lifecycle_lease_expires_at <= database_now,
            ),
            OpenAIOAuthCredential.refresh_attempt_id.is_(None),
            or_(
                OpenAIOAuthCredential.last_error_code.is_(None),
                OpenAIOAuthCredential.last_error_code.not_in(
                    {"refresh_outcome_unknown", "revocation_pending"}
                ),
            ),
        )
    result = await session.execute(
        statement.values(
            status=status,
            last_error_code=last_error_code,
            credential_version=OpenAIOAuthCredential.credential_version + 1,
            lifecycle_lease_token=None,
            lifecycle_lease_kind=None,
            lifecycle_lease_expires_at=None,
        ).returning(OpenAIOAuthCredential.id)
    )
    if result.scalar_one_or_none() is None:
        current = await get_owned_credential(session, connection_id, user_id)
        if current is None:
            raise LookupError("connection not found")
        raise StaleCredentialVersionError("credential version is stale")
    await session.commit()
    credential = await get_owned_credential(session, connection_id, user_id)
    if credential is None:  # pragma: no cover - protected by the atomic UPDATE
        raise LookupError("connection not found")
    await session.commit()
    return credential


async def scrub_credential_bundle(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    expected_version: int,
    status: str,
    last_error_code: str | None = None,
    expected_lease_token: str | None = None,
) -> OpenAIOAuthCredential:
    """Atomically remove all tokens while retaining recovery metadata."""
    _validate_status(status)
    owned_connection = select(LlmConnection.id).where(
        LlmConnection.id == connection_id, LlmConnection.user_id == user_id
    )
    statement = update(OpenAIOAuthCredential).where(
        OpenAIOAuthCredential.connection_id == connection_id,
        OpenAIOAuthCredential.connection_id.in_(owned_connection),
        OpenAIOAuthCredential.credential_version == expected_version,
    )
    if expected_lease_token is not None:
        statement = statement.where(
            *_live_attempt_lease(session, expected_lease_token)
        )
    else:
        database_now = (
            func.now()
            if session.get_bind().dialect.name == "postgresql"
            else func.datetime("now")
        )
        statement = statement.where(
            or_(
                OpenAIOAuthCredential.lifecycle_lease_token.is_(None),
                OpenAIOAuthCredential.lifecycle_lease_expires_at <= database_now,
            ),
            OpenAIOAuthCredential.refresh_attempt_id.is_(None),
            or_(
                OpenAIOAuthCredential.last_error_code.is_(None),
                OpenAIOAuthCredential.last_error_code.not_in(
                    {"refresh_outcome_unknown", "revocation_pending"}
                ),
            ),
        )
    result = await session.execute(
        statement
        .values(
            token_bundle_encrypted=None,
            status=status,
            last_error_code=last_error_code,
            refresh_attempt_id=None,
            refresh_attempt_started_at=None,
            refresh_restore_status=None,
            credential_version=OpenAIOAuthCredential.credential_version + 1,
            lifecycle_lease_token=None,
            lifecycle_lease_kind=None,
            lifecycle_lease_expires_at=None,
        )
        .returning(OpenAIOAuthCredential.id)
    )
    if result.scalar_one_or_none() is None:
        current = await get_owned_credential(session, connection_id, user_id)
        if current is None:
            raise LookupError("connection not found")
        raise StaleCredentialVersionError("credential version is stale")
    await session.commit()
    credential = await get_owned_credential(session, connection_id, user_id)
    if credential is None:  # pragma: no cover - protected by the atomic UPDATE
        raise LookupError("connection not found")
    await session.commit()
    return credential
