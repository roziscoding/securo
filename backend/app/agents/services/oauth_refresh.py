"""Database-backed SIWC refresh, revocation, and recovery orchestration."""
from __future__ import annotations

import asyncio
import logging
import uuid
import weakref
from datetime import datetime, timedelta, timezone
from typing import Callable

from billiard.exceptions import SoftTimeLimitExceeded
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.config import MAX_OAUTH_TIMEOUT_SECONDS
from app.agents.models.oauth_credential import OpenAIOAuthCredential
from app.agents.services.oauth_credentials import (
    StaleCredentialVersionError,
    begin_refresh_attempt,
    begin_revocation_attempt,
    decrypt_bundle,
    get_owned_credential,
    rotate_credential,
    scrub_credential_bundle,
    transition_credential_status,
)
from app.agents.services.openai_oauth import (
    OAuthFailureCode,
    OAuthLifecycleError,
    _discard_exception_secrets,
    fetch_validated_jwks,
    refresh_token_bundle,
    revoke_token,
    safe_oauth_boundary,
)

_REFRESH_WINDOW = timedelta(minutes=5)
# Refresh can fetch JWKS and then exchange the token. Each owned request has
# a bounded whole-request timeout; use the maximum accepted configuration.
_MAX_OPERATION_NETWORK_SECONDS = 2 * MAX_OAUTH_TIMEOUT_SECONDS
_OPERATION_LEASE = timedelta(seconds=_MAX_OPERATION_NETWORK_SECONDS + 120)


def _refresh_validity_margin() -> timedelta:
    return timedelta(seconds=_MAX_OPERATION_NETWORK_SECONDS + 60)


_logger = logging.getLogger(__name__)
_local_locks: weakref.WeakValueDictionary[uuid.UUID, asyncio.Lock] = (
    weakref.WeakValueDictionary()
)
_locks_guard = asyncio.Lock()


async def _local_lock(connection_id: uuid.UUID) -> asyncio.Lock:
    async with _locks_guard:
        return _local_locks.setdefault(connection_id, asyncio.Lock())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


async def _locked_owned_credential(
    session: AsyncSession, connection_id: uuid.UUID, user_id: uuid.UUID
) -> OpenAIOAuthCredential:
    """Reload a credential; serialization is provided by the committed lease."""
    credential = await get_owned_credential(session, connection_id, user_id)
    if credential is None:
        raise LookupError("connection not found")
    await session.refresh(credential)
    return credential


def _inactive(status: str, last_error_code: str | None = None) -> OAuthLifecycleError:
    if status == "reauth_required" and last_error_code == "refresh_outcome_unknown":
        return OAuthLifecycleError(
            OAuthFailureCode.REFRESH_OUTCOME_UNKNOWN,
            "OAuth refresh outcome is unknown; explicit reauthorization is required",
            definitive=True,
        )
    if status == "revoked":
        return OAuthLifecycleError(
            OAuthFailureCode.REVOKED, "OAuth credential is not active", definitive=True
        )
    if status == "disabled":
        return OAuthLifecycleError(
            OAuthFailureCode.SCOPE_LOSS, "ChatGPT plan usage is disabled", definitive=True
        )
    return OAuthLifecycleError(
        OAuthFailureCode.REAUTH_REQUIRED, "OAuth credential is not active", definitive=True
    )


def _race_lost() -> OAuthLifecycleError:
    return OAuthLifecycleError(
        OAuthFailureCode.RACE_LOST,
        "A concurrent OAuth lifecycle operation is in progress",
        retryable=True,
    )


def _previous_revocation_failure(last_error_code: str | None) -> OAuthLifecycleError:
    try:
        code = OAuthFailureCode(last_error_code)
    except (TypeError, ValueError):
        code = OAuthFailureCode.MALFORMED_RESPONSE
    return OAuthLifecycleError(
        code,
        "Previous OAuth revocation was not remotely confirmed",
        definitive=code is OAuthFailureCode.INVALID_CLIENT,
        retryable=code in {
            OAuthFailureCode.TRANSIENT,
            OAuthFailureCode.REDIRECT_REJECTED,
        },
    )


@safe_oauth_boundary
async def refresh_credential(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    now: datetime | None = None,
    force: bool = False,
    rejected_version: int | None = None,
    refresh_window: timedelta = _REFRESH_WINDOW,
    receipt_clock: Callable[[], datetime] | None = None,
) -> str:
    """Return a usable token, using a committed lease around provider I/O."""
    current_time = _utc(now or _now())
    current = await get_owned_credential(session, connection_id, user_id)
    if current is None:
        await session.rollback()
        raise LookupError("connection not found")
    await session.refresh(current)
    if current.status != "active":
        inactive_status = current.status
        inactive_error = current.last_error_code
        await session.rollback()
        raise _inactive(inactive_status, inactive_error)
    if rejected_version is not None and current.credential_version != rejected_version:
        try:
            if current.token_bundle_encrypted is None:
                raise ValueError("missing bundle")
            access_token = decrypt_bundle(current.token_bundle_encrypted).access_token
        except ValueError:
            await session.rollback()
            raise OAuthLifecycleError(
                OAuthFailureCode.CORRUPT_TOKEN,
                "OAuth credential cannot be decrypted",
                definitive=True,
            ) from None
        await session.rollback()
        return access_token
    observed_version = current.credential_version
    if (
        not force
        and _utc(current.access_token_expires_at) > current_time + refresh_window
        and (
            current.refresh_token_expires_at is None
            or _utc(current.refresh_token_expires_at) > current_time
        )
    ):
        try:
            if current.token_bundle_encrypted is None:
                raise ValueError("missing bundle")
            access_token = decrypt_bundle(current.token_bundle_encrypted).access_token
        except ValueError:
            pass
        else:
            await session.rollback()
            return access_token
    await session.rollback()

    lock = await _local_lock(connection_id)
    async with lock:
        current_time = _utc(now or _now())
        credential = await _locked_owned_credential(session, connection_id, user_id)
        if credential.status != "active":
            inactive_status = credential.status
            inactive_error = credential.last_error_code
            await session.rollback()
            raise _inactive(inactive_status, inactive_error)
        if rejected_version is not None and credential.credential_version != rejected_version:
            if credential.token_bundle_encrypted is None:
                await session.rollback()
                raise _inactive(credential.status, credential.last_error_code)
            try:
                access_token = decrypt_bundle(credential.token_bundle_encrypted).access_token
            except ValueError:
                await session.rollback()
                raise OAuthLifecycleError(
                    OAuthFailureCode.CORRUPT_TOKEN,
                    "OAuth credential cannot be decrypted",
                    definitive=True,
                ) from None
            await session.rollback()
            return access_token
        if credential.refresh_token_expires_at is None or _utc(
            credential.refresh_token_expires_at
        ) <= current_time + _refresh_validity_margin():
            try:
                await scrub_credential_bundle(
                    session,
                    connection_id,
                    user_id,
                    expected_version=credential.credential_version,
                    status="reauth_required",
                    last_error_code="expired",
                )
            except StaleCredentialVersionError as exc:
                await session.rollback()
                raise _race_lost() from exc
            raise OAuthLifecycleError(
                OAuthFailureCode.REAUTH_REQUIRED,
                "OAuth refresh token has expired",
                definitive=True,
            )
        # A caller that waited on the process lock joins the completed
        # single-flight even when it originally requested a forced refresh.
        if credential.credential_version != observed_version:
            if credential.token_bundle_encrypted is None:
                inactive_status = credential.status
                await session.rollback()
                raise _inactive(inactive_status)
            access_token = decrypt_bundle(credential.token_bundle_encrypted).access_token
            await session.rollback()
            return access_token
        if not force and _utc(credential.access_token_expires_at) > current_time + refresh_window:
            if credential.token_bundle_encrypted is None:
                await session.rollback()
                raise OAuthLifecycleError(
                    OAuthFailureCode.REAUTH_REQUIRED,
                    "OAuth credential requires reauthorization",
                    definitive=True,
                )
            try:
                access_token = decrypt_bundle(credential.token_bundle_encrypted).access_token
            except ValueError:
                await scrub_credential_bundle(
                    session,
                    connection_id,
                    user_id,
                    expected_version=credential.credential_version,
                    status="reauth_required",
                    last_error_code="corrupt_token",
                )
                raise OAuthLifecycleError(
                    OAuthFailureCode.CORRUPT_TOKEN,
                    "OAuth credential cannot be decrypted",
                    definitive=True,
                ) from None
            await session.rollback()
            return access_token
        if credential.earliest_refresh_at is not None and current_time < _utc(
            credential.earliest_refresh_at
        ):
            await session.rollback()
            raise OAuthLifecycleError(
                OAuthFailureCode.TOO_EARLY,
                "OAuth credential cannot be refreshed yet",
                retryable=True,
            )
        if credential.token_bundle_encrypted is None:
            await session.rollback()
            raise OAuthLifecycleError(
                OAuthFailureCode.REAUTH_REQUIRED,
                "OAuth credential requires reauthorization",
                definitive=True,
            )
        encrypted_bundle = credential.token_bundle_encrypted
        expected_version = credential.credential_version
        issued_client_id = credential.issued_client_id
        subject = credential.subject
        await session.rollback()

        # Establish signing-key readiness before exposing the rotating refresh
        # token or committing an irreversible dispatch-intent marker.
        jwks = await fetch_validated_jwks()
        try:
            old_bundle = decrypt_bundle(encrypted_bundle)
        except ValueError:
            await scrub_credential_bundle(
                session,
                connection_id,
                user_id,
                expected_version=expected_version,
                status="reauth_required",
                last_error_code="corrupt_token",
            )
            raise OAuthLifecycleError(
                OAuthFailureCode.CORRUPT_TOKEN,
                "OAuth credential cannot be decrypted",
                definitive=True,
            ) from None

        lease_token = str(uuid.uuid4())
        claimed = await begin_refresh_attempt(
            session,
            connection_id,
            user_id,
            expected_version=expected_version,
            attempt_id=lease_token,
            lease_duration=_OPERATION_LEASE,
        )
        if not claimed:
            old_bundle = None
            raise _race_lost()

        refresh_token = old_bundle.refresh_token
        old_bundle = None

        async def refresh_and_finalize() -> str:
            try:
                rotated = await refresh_token_bundle(
                    refresh_token=refresh_token,
                    issued_client_id=issued_client_id,
                    expected_subject=subject,
                    jwks=jwks,
                    receipt_clock=receipt_clock,
                )
            except BaseException as exc:  # cleanup also covers cooperative cancellation
                try:
                    if isinstance(exc, OAuthLifecycleError) and exc.definitive:
                        try:
                            await scrub_credential_bundle(
                                session,
                                connection_id,
                                user_id,
                                expected_version=expected_version,
                                expected_lease_token=lease_token,
                                status="reauth_required",
                                last_error_code=exc.code.value,
                            )
                        except StaleCredentialVersionError:
                            pass
                    else:
                        # Provider dispatch may already have consumed this
                        # rotating token. Leave the durable unknown marker and
                        # encrypted recovery bundle intact forever.
                        await session.rollback()
                except BaseException:
                    if isinstance(exc, SoftTimeLimitExceeded):
                        _logger.warning(
                            "OAuth refresh database cleanup failed during soft time limit handling"
                        )
                    else:
                        raise
                if isinstance(exc, SoftTimeLimitExceeded):
                    _discard_exception_secrets(exc)
                raise

            plan_usage_enabled = "chatgpt.tokens.use.direct" in rotated.scopes
            try:
                saved = await rotate_credential(
                    session,
                    connection_id,
                    user_id,
                    expected_version=expected_version,
                    expected_lease_token=lease_token,
                    expected_attempt_id=lease_token,
                    bundle=rotated.bundle,
                    access_token_expires_at=rotated.access_token_expires_at,
                    earliest_refresh_at=rotated.earliest_refresh_at,
                    refresh_token_expires_at=rotated.refresh_token_expires_at,
                    last_refreshed_at=rotated.received_at,
                    status="active" if plan_usage_enabled else "disabled",
                    last_error_code=(
                        None if plan_usage_enabled else OAuthFailureCode.SCOPE_LOSS.value
                    ),
                    subject=rotated.subject,
                    account_email=rotated.account_email,
                    scopes=sorted(rotated.scopes),
                )
            except StaleCredentialVersionError as exc:
                await session.rollback()
                raise _race_lost() from exc
            if saved.token_bundle_encrypted is None:  # pragma: no cover
                await session.rollback()
                raise OAuthLifecycleError(
                    OAuthFailureCode.CORRUPT_TOKEN,
                    "OAuth refresh persistence failed",
                    definitive=True,
                )
            access_token = decrypt_bundle(saved.token_bundle_encrypted).access_token
            await session.rollback()
            if not plan_usage_enabled:
                raise OAuthLifecycleError(
                    OAuthFailureCode.SCOPE_LOSS,
                    "ChatGPT plan usage is disabled",
                    definitive=True,
                )
            return access_token

        operation_task = asyncio.create_task(refresh_and_finalize())
        cancellation: asyncio.CancelledError | None = None
        while not operation_task.done():
            try:
                await asyncio.shield(operation_task)
            except asyncio.CancelledError as exc:
                cancellation = exc
            except BaseException:
                break
        try:
            result = operation_task.result()
        except BaseException:
            if cancellation is not None:
                raise asyncio.CancelledError() from None
            raise
        if cancellation is not None:
            raise asyncio.CancelledError() from None
        return result


@safe_oauth_boundary
async def revoke_credential(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
) -> None:
    """Revoke remotely without a DB transaction, then erase the local bundle."""
    lock = await _local_lock(connection_id)
    async with lock:
        credential = await _locked_owned_credential(session, connection_id, user_id)
        if credential.status == "revoked" and credential.token_bundle_encrypted is None:
            last_error_code = credential.last_error_code
            await session.rollback()
            if last_error_code is None:
                return
            raise _previous_revocation_failure(last_error_code)
        recoverable_pending = (
            credential.status == "reauth_required"
            and credential.last_error_code in {"refresh_outcome_unknown", "revocation_pending"}
            and credential.token_bundle_encrypted is not None
        )
        if credential.status not in {"active", "disabled"} and not recoverable_pending:
            inactive_status = credential.status
            inactive_error = credential.last_error_code
            await session.rollback()
            raise _inactive(inactive_status, inactive_error)
        if credential.token_bundle_encrypted is None:
            await session.rollback()
            return
        try:
            bundle = decrypt_bundle(credential.token_bundle_encrypted)
        except ValueError:
            await scrub_credential_bundle(
                session,
                connection_id,
                user_id,
                expected_version=credential.credential_version,
                status="revoked",
                last_error_code=OAuthFailureCode.CORRUPT_TOKEN.value,
            )
            raise OAuthLifecycleError(
                OAuthFailureCode.CORRUPT_TOKEN,
                "OAuth credential cannot be decrypted",
                definitive=True,
            ) from None

        expected_version = credential.credential_version
        issued_client_id = credential.issued_client_id
        lease_token = str(uuid.uuid4())
        claimed = await begin_revocation_attempt(
            session,
            connection_id,
            user_id,
            expected_version=expected_version,
            lease_token=lease_token,
            lease_duration=_OPERATION_LEASE,
        )
        if not claimed:
            raise _race_lost()

        async def scrub_local_bundle(last_error_code: str | None) -> None:
            try:
                await scrub_credential_bundle(
                    session,
                    connection_id,
                    user_id,
                    expected_version=expected_version,
                    expected_lease_token=lease_token,
                    status="revoked",
                    last_error_code=last_error_code,
                )
            finally:
                if session.in_transaction():
                    await session.rollback()

        async def finish_local_scrub(last_error_code: str | None) -> asyncio.CancelledError | None:
            # Shielding alone is insufficient because the outer await still
            # raises immediately. Keep waiting until the child commits.
            cleanup_task = asyncio.create_task(scrub_local_bundle(last_error_code))
            cleanup_cancellation: asyncio.CancelledError | None = None
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError as exc:
                    cleanup_cancellation = exc
                except BaseException:
                    break
            cleanup_task.result()
            return cleanup_cancellation

        failure: BaseException | None = None
        try:
            await revoke_token(
                refresh_token=bundle.refresh_token,
                issued_client_id=issued_client_id,
            )
        except SoftTimeLimitExceeded as exc:
            bundle = None
            try:
                await finish_local_scrub(OAuthFailureCode.TRANSIENT.value)
            except BaseException:
                # The scheduler's soft limit must retain priority, but the
                # durable pending state is already committed before provider I/O.
                _logger.warning(
                    "OAuth revocation database cleanup failed during soft time limit handling"
                )
            _discard_exception_secrets(exc)
            raise
        except BaseException as exc:  # cleanup is mandatory even for cancellation/system exits
            failure = exc
        finally:
            bundle = None

        last_error_code = (
            failure.code.value
            if isinstance(failure, OAuthLifecycleError)
            else OAuthFailureCode.TRANSIENT.value
            if failure is not None
            else None
        )

        try:
            cleanup_cancellation = await finish_local_scrub(last_error_code)
        except BaseException as exc:
            if isinstance(exc, StaleCredentialVersionError):
                if isinstance(failure, asyncio.CancelledError):
                    raise asyncio.CancelledError() from None
                raise _race_lost() from exc
            raise

        if cleanup_cancellation is not None:
            raise asyncio.CancelledError() from None
        if failure is not None:
            raise failure


async def require_reauthorization(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    expected_version: int,
    last_error_code: str = "reauth_required",
) -> None:
    """Scrub only the credential version rejected by the provider."""
    lock = await _local_lock(connection_id)
    async with lock:
        credential = await get_owned_credential(session, connection_id, user_id)
        if credential is None:
            await session.rollback()
            raise LookupError("connection not found")
        await session.refresh(credential)
        if credential.credential_version != expected_version:
            await session.rollback()
            raise _race_lost()
        if credential.refresh_attempt_id is not None or credential.last_error_code in {
            "refresh_outcome_unknown",
            "revocation_pending",
        }:
            await session.rollback()
            raise _race_lost()
        if credential.token_bundle_encrypted is None and credential.status == "reauth_required":
            await session.rollback()
            return
        try:
            await scrub_credential_bundle(
                session,
                connection_id,
                user_id,
                expected_version=expected_version,
                status="reauth_required",
                last_error_code=last_error_code,
            )
        except StaleCredentialVersionError:
            await session.rollback()
            raise _race_lost() from None


async def disable_plan_usage(
    session: AsyncSession,
    connection_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    expected_version: int,
) -> None:
    """Disable only the credential version rejected by the provider."""
    lock = await _local_lock(connection_id)
    async with lock:
        credential = await get_owned_credential(session, connection_id, user_id)
        if credential is None:
            await session.rollback()
            raise LookupError("connection not found")
        await session.refresh(credential)
        if credential.credential_version != expected_version:
            await session.rollback()
            raise _race_lost()
        if credential.refresh_attempt_id is not None or credential.last_error_code in {
            "refresh_outcome_unknown",
            "revocation_pending",
        }:
            await session.rollback()
            raise _race_lost()
        if credential.status == "disabled":
            await session.rollback()
            return
        try:
            await transition_credential_status(
                session,
                connection_id,
                user_id,
                expected_version=expected_version,
                status="disabled",
                last_error_code=OAuthFailureCode.SCOPE_LOSS.value,
            )
        except StaleCredentialVersionError:
            await session.rollback()
            raise _race_lost() from None
