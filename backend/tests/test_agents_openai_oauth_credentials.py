"""Encrypted persistence coverage for SIWC credential foundations."""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import JSONB, dialect as postgresql_dialect

from app.agents.models.oauth_credential import OpenAIOAuthCredential
from app.agents.services import connection_service
from app.agents.services.crypto import encrypt
from app.agents.services.oauth_credentials import (
    CredentialBundle,
    StaleCredentialVersionError,
    create_credential,
    decrypt_bundle,
    encrypt_bundle,
    get_owned_credential,
    rotate_credential,
    transition_credential_status,
)
from app.core.database import Base, engine
from app.models.user import User


def test_credential_bundle_encryption_roundtrip_hides_secrets():
    bundle = CredentialBundle(
        access_token="access-secret-value",
        refresh_token="refresh-secret-value",
        id_token="id-secret-value",
    )

    ciphertext = encrypt_bundle(bundle)

    assert decrypt_bundle(ciphertext) == bundle
    assert "access-secret-value" not in ciphertext
    assert "refresh-secret-value" not in ciphertext
    assert "id-secret-value" not in ciphertext
    assert "secret-value" not in repr(bundle)


@pytest.mark.parametrize(
    "bundle",
    [
        CredentialBundle("", "refresh", "id"),
        CredentialBundle("access", "", "id"),
        CredentialBundle("access", "refresh", ""),
        CredentialBundle(cast(Any, None), "refresh", "id"),
        {"access_token": "access", "refresh_token": "refresh", "id_token": "id"},
    ],
)
def test_encrypt_bundle_rejects_non_exact_or_invalid_bundle(bundle):
    with pytest.raises(ValueError, match="credential bundle is invalid"):
        encrypt_bundle(bundle)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"access_token": "access", "refresh_token": "refresh"},
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "id_token": "id",
            "nested_secret": {"token": "must-not-appear"},
        },
        {"access_token": 7, "refresh_token": "refresh", "id_token": "id"},
        {"access_token": "", "refresh_token": "refresh", "id_token": "id"},
    ],
)
def test_decrypt_bundle_rejects_non_exact_or_invalid_object_without_leaking_values(payload):
    ciphertext = encrypt(json.dumps(payload))
    assert ciphertext is not None

    with pytest.raises(ValueError) as exc_info:
        decrypt_bundle(ciphertext)

    assert str(exc_info.value) == "credential bundle is invalid"
    assert "must-not-appear" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_credential_is_stored_as_ciphertext_with_safe_metadata(session, test_user):
    connection = await connection_service.create_connection(
        session, test_user.id, name="ChatGPT", kind="openai_chatgpt"
    )
    now = datetime.now(timezone.utc)
    bundle = CredentialBundle("access-token-plain", "refresh-token-plain", "id-token-plain")

    credential = await create_credential(
        session,
        connection.id,
        test_user.id,
        bundle=bundle,
        issuer="https://auth.openai.com",
        subject="user-subject",
        issued_client_id="oaiapp_client",
        account_email="owner@example.com",
        scopes=["openid", "chatgpt.tokens.use.direct"],
        access_token_expires_at=now + timedelta(hours=1),
        earliest_refresh_at=now + timedelta(minutes=30),
        refresh_token_expires_at=now + timedelta(days=30),
    )

    stored = (await session.execute(
        text(
            "SELECT token_bundle_encrypted FROM agent_openai_oauth_credentials "
            "WHERE connection_id = :connection_id"
        ),
        {"connection_id": connection.id.hex},
    )).scalar_one()
    assert "access-token-plain" not in stored
    assert "refresh-token-plain" not in stored
    assert "id-token-plain" not in stored
    assert decrypt_bundle(stored) == bundle
    assert credential.status == "active"
    assert credential.credential_version == 1
    assert credential.scopes == ["openid", "chatgpt.tokens.use.direct"]
    assert "token-plain" not in repr(credential)

    fetched = await get_owned_credential(session, connection.id, test_user.id)
    assert fetched is not None
    assert fetched.id == credential.id
    assert (await session.execute(select(OpenAIOAuthCredential))).scalar_one().id == credential.id


@pytest.mark.asyncio
async def test_duplicate_credential_raises_sanitized_domain_error_and_rolls_back(
    session, test_user
):
    connection = await connection_service.create_connection(
        session, test_user.id, name="ChatGPT", kind="openai_chatgpt"
    )
    connection_id = connection.id
    now = datetime.now(timezone.utc)
    common: dict[str, Any] = {
        "issuer": "https://auth.openai.com",
        "subject": "subject",
        "issued_client_id": "oaiapp_client",
        "account_email": None,
        "scopes": ["openid"],
        "access_token_expires_at": now + timedelta(hours=1),
    }
    await create_credential(
        session,
        connection.id,
        test_user.id,
        bundle=CredentialBundle("first-access", "first-refresh", "first-id"),
        **common,
    )
    duplicate_secrets = ("duplicate-access", "duplicate-refresh", "duplicate-id")

    with pytest.raises(RuntimeError) as exc_info:
        await create_credential(
            session,
            connection.id,
            test_user.id,
            bundle=CredentialBundle(*duplicate_secrets),
            **common,
        )

    assert type(exc_info.value).__name__ == "CredentialAlreadyExistsError"
    assert str(exc_info.value) == "OAuth credential already exists"
    assert all(secret not in str(exc_info.value) for secret in duplicate_secrets)
    assert (await session.execute(select(OpenAIOAuthCredential))).scalar_one().connection_id == (
        connection_id
    )


@pytest.mark.asyncio
async def test_rotation_rejects_non_owner(session, test_user):
    connection = await connection_service.create_connection(
        session, test_user.id, name="ChatGPT", kind="openai_chatgpt"
    )
    now = datetime.now(timezone.utc)
    original = CredentialBundle("access-one", "refresh-one", "id-one")
    await create_credential(
        session,
        connection.id,
        test_user.id,
        bundle=original,
        issuer="https://auth.openai.com",
        subject="subject",
        issued_client_id="oaiapp_client",
        account_email=None,
        scopes=["openid"],
        access_token_expires_at=now + timedelta(hours=1),
    )

    wrong_user_id = test_user.id.__class__(int=test_user.id.int ^ 1)
    assert await get_owned_credential(session, connection.id, wrong_user_id) is None
    with pytest.raises(LookupError, match="connection not found"):
        await rotate_credential(
            session,
            connection.id,
            wrong_user_id,
            expected_version=1,
            bundle=CredentialBundle("stolen-access", "stolen-refresh", "stolen-id"),
            access_token_expires_at=now + timedelta(hours=2),
        )

    owned = await get_owned_credential(session, connection.id, test_user.id)
    assert owned is not None
    assert decrypt_bundle(owned.token_bundle_encrypted) == original


@pytest.mark.asyncio
async def test_stale_rotation_cannot_overwrite_newer_bundle(session, test_user):
    connection = await connection_service.create_connection(
        session, test_user.id, name="ChatGPT", kind="openai_chatgpt"
    )
    now = datetime.now(timezone.utc)
    await create_credential(
        session,
        connection.id,
        test_user.id,
        bundle=CredentialBundle("access-one", "refresh-one", "id-one"),
        issuer="https://auth.openai.com",
        subject="subject",
        issued_client_id="oaiapp_client",
        account_email=None,
        scopes=["openid"],
        access_token_expires_at=now + timedelta(hours=1),
    )
    winner = CredentialBundle("access-two", "refresh-two", "id-two")
    rotated = await rotate_credential(
        session,
        connection.id,
        test_user.id,
        expected_version=1,
        bundle=winner,
        access_token_expires_at=now + timedelta(hours=2),
        last_refreshed_at=now,
    )
    assert rotated.credential_version == 2

    with pytest.raises(StaleCredentialVersionError, match="stale"):
        await rotate_credential(
            session,
            connection.id,
            test_user.id,
            expected_version=1,
            bundle=CredentialBundle("stale-access", "stale-refresh", "stale-id"),
            access_token_expires_at=now + timedelta(hours=3),
        )

    connection_id = connection.id
    user_id = test_user.id
    session.expire_all()
    current = await get_owned_credential(session, connection_id, user_id)
    assert current is not None
    assert current.credential_version == 2
    assert decrypt_bundle(current.token_bundle_encrypted) == winner


@pytest.mark.asyncio
async def test_deleting_connection_rejects_unrevoked_oauth_credential(session, test_user):
    connection = await connection_service.create_connection(
        session, test_user.id, name="ChatGPT", kind="openai_chatgpt"
    )
    connection_id = connection.id
    user_id = test_user.id
    now = datetime.now(timezone.utc)
    await create_credential(
        session,
        connection_id,
        user_id,
        bundle=CredentialBundle("access", "refresh", "id"),
        issuer="https://auth.openai.com",
        subject="subject",
        issued_client_id="oaiapp_client",
        account_email=None,
        scopes=["openid"],
        access_token_expires_at=now + timedelta(hours=1),
    )

    with pytest.raises(ValueError, match="revoke"):
        await connection_service.delete_connection(session, connection_id, user_id)
    await session.rollback()
    assert await get_owned_credential(session, connection_id, user_id) is not None


@pytest.mark.asyncio
async def test_status_transition_is_owned_and_versioned(session, test_user):
    connection = await connection_service.create_connection(
        session, test_user.id, name="ChatGPT", kind="openai_chatgpt"
    )
    now = datetime.now(timezone.utc)
    await create_credential(
        session,
        connection.id,
        test_user.id,
        bundle=CredentialBundle("access", "refresh", "id"),
        issuer="https://auth.openai.com",
        subject="subject",
        issued_client_id="oaiapp_client",
        account_email=None,
        scopes=["openid"],
        access_token_expires_at=now + timedelta(hours=1),
    )

    transitioned = await transition_credential_status(
        session,
        connection.id,
        test_user.id,
        expected_version=1,
        status="reauth_required",
        last_error_code="invalid_refresh_token",
    )

    assert transitioned.status == "reauth_required"
    assert transitioned.last_error_code == "invalid_refresh_token"
    assert transitioned.credential_version == 2
    with pytest.raises(StaleCredentialVersionError):
        await transition_credential_status(
            session,
            connection.id,
            test_user.id,
            expected_version=1,
            status="revoked",
        )


def test_oauth_credential_table_is_registered_in_base_metadata():
    table = Base.metadata.tables["agent_openai_oauth_credentials"]
    assert table.c.token_bundle_encrypted.type.__class__.__name__ == "Text"
    assert table.c.connection_id.unique is True
    assert next(iter(table.c.connection_id.foreign_keys)).ondelete == "CASCADE"


def test_oauth_scopes_orm_uses_postgresql_jsonb_like_migration():
    scopes_type = Base.metadata.tables["agent_openai_oauth_credentials"].c.scopes.type

    assert isinstance(scopes_type.dialect_impl(postgresql_dialect()), JSONB)


def test_main_database_engine_hides_bind_parameters():
    assert engine.sync_engine.hide_parameters is True


@pytest.mark.asyncio
async def test_postgres_concurrent_rotation_allows_only_one_winner(postgres_sessions):
    user_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    async with postgres_sessions() as setup:
        setup.add(
            User(
                id=user_id,
                email=f"siwc-{user_id}@example.com",
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
        connection_id = connection.id
        await create_credential(
            setup,
            connection_id,
            user_id,
            bundle=CredentialBundle("access-one", "refresh-one", "id-one"),
            issuer="https://auth.openai.com",
            subject="subject",
            issued_client_id="oaiapp_client",
            account_email=None,
            scopes=["openid"],
            access_token_expires_at=now + timedelta(hours=1),
        )

    async with postgres_sessions() as first, postgres_sessions() as second:
        results = await asyncio.gather(
            rotate_credential(
                first,
                connection_id,
                user_id,
                expected_version=1,
                bundle=CredentialBundle("access-a", "refresh-a", "id-a"),
                access_token_expires_at=now + timedelta(hours=2),
            ),
            rotate_credential(
                second,
                connection_id,
                user_id,
                expected_version=1,
                bundle=CredentialBundle("access-b", "refresh-b", "id-b"),
                access_token_expires_at=now + timedelta(hours=2),
            ),
            return_exceptions=True,
        )

    assert sum(isinstance(result, OpenAIOAuthCredential) for result in results) == 1
    assert sum(isinstance(result, StaleCredentialVersionError) for result in results) == 1
