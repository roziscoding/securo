"""Encrypted, ownership-scoped storage for OpenAI SIWC credentials."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models.connection import LlmConnection
from app.agents.models.oauth_credential import OpenAIOAuthCredential
from app.agents.services.crypto import decrypt, encrypt


CREDENTIAL_STATUSES = frozenset(
    {"active", "disabled", "expired", "revoked", "error", "reauth_required"}
)


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
    if any(not isinstance(data[field], str) or not data[field] for field in _BUNDLE_FIELDS):
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


def decrypt_bundle(ciphertext: str) -> CredentialBundle:
    plaintext = decrypt(ciphertext)
    if plaintext is None:
        raise ValueError("credential bundle cannot be decrypted")
    try:
        data = json.loads(plaintext)
    except json.JSONDecodeError as exc:
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
    connection = (
        await session.execute(
            select(LlmConnection).where(
                LlmConnection.id == connection_id, LlmConnection.user_id == user_id
            )
        )
    ).scalar_one_or_none()
    if connection is None:
        raise LookupError("connection not found")
    if connection.kind != "openai_chatgpt":
        raise ValueError("OAuth credentials require an openai_chatgpt connection")

    credential = OpenAIOAuthCredential(
        connection=connection,
        issuer=issuer,
        subject=subject,
        issued_client_id=issued_client_id,
        account_email=account_email,
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
) -> OpenAIOAuthCredential:
    _validate_status(status)
    owned_connection = select(LlmConnection.id).where(
        LlmConnection.id == connection_id, LlmConnection.user_id == user_id
    )
    result = await session.execute(
        update(OpenAIOAuthCredential)
        .where(
            OpenAIOAuthCredential.connection_id == connection_id,
            OpenAIOAuthCredential.connection_id.in_(owned_connection),
            OpenAIOAuthCredential.credential_version == expected_version,
        )
        .values(
            token_bundle_encrypted=encrypt_bundle(bundle),
            access_token_expires_at=access_token_expires_at,
            earliest_refresh_at=earliest_refresh_at,
            refresh_token_expires_at=refresh_token_expires_at,
            last_refreshed_at=last_refreshed_at,
            status=status,
            last_error_code=last_error_code,
            credential_version=OpenAIOAuthCredential.credential_version + 1,
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
    return credential


async def transition_credential_status(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    expected_version: int,
    status: str,
    last_error_code: str | None = None,
) -> OpenAIOAuthCredential:
    _validate_status(status)
    owned_connection = select(LlmConnection.id).where(
        LlmConnection.id == connection_id, LlmConnection.user_id == user_id
    )
    result = await session.execute(
        update(OpenAIOAuthCredential)
        .where(
            OpenAIOAuthCredential.connection_id == connection_id,
            OpenAIOAuthCredential.connection_id.in_(owned_connection),
            OpenAIOAuthCredential.credential_version == expected_version,
        )
        .values(
            status=status,
            last_error_code=last_error_code,
            credential_version=OpenAIOAuthCredential.credential_version + 1,
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
    return credential
