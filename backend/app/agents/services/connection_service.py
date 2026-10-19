"""User-managed LLM connections (provider endpoints + API keys)."""
from __future__ import annotations

import uuid
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.models.connection import LlmConnection
from app.agents.providers.registry import build_provider, list_providers
from app.agents.services.crypto import decrypt, encrypt
from app.agents.services.openai_oauth import (
    OAuthFailureCode,
    OAuthLifecycleError,
)


VALID_KINDS = set(list_providers()) | {"openai_chatgpt"}


async def list_connections(session: AsyncSession, user_id: uuid.UUID) -> list[LlmConnection]:
    return list((await session.execute(
        select(LlmConnection).where(LlmConnection.user_id == user_id).order_by(LlmConnection.created_at.desc())
    )).scalars().all())


async def get_connection(
    session: AsyncSession, conn_id: uuid.UUID, user_id: uuid.UUID
) -> Optional[LlmConnection]:
    return (await session.execute(
        select(LlmConnection).where(LlmConnection.id == conn_id, LlmConnection.user_id == user_id)
    )).scalar_one_or_none()


async def get_default_connection(session: AsyncSession, user_id: uuid.UUID) -> Optional[LlmConnection]:
    return (await session.execute(
        select(LlmConnection).where(LlmConnection.user_id == user_id, LlmConnection.is_default.is_(True)).limit(1)
    )).scalar_one_or_none()


async def create_connection(
    session: AsyncSession,
    user_id: uuid.UUID,
    *,
    name: str,
    kind: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    default_model: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
    is_default: bool = False,
) -> LlmConnection:
    if kind not in VALID_KINDS:
        raise ValueError(f"unknown kind '{kind}', expected one of {sorted(VALID_KINDS)}")
    if kind == "openai_compatible" and not base_url:
        raise ValueError("openai_compatible requires base_url")
    if kind == "openai_chatgpt":
        if api_key is not None:
            raise ValueError("openai_chatgpt does not accept api_key")
        if base_url is not None:
            raise ValueError("openai_chatgpt does not accept base_url")
        if extra:
            raise ValueError("openai_chatgpt does not accept extra")

    if is_default:
        await _clear_default(session, user_id)

    conn = LlmConnection(
        user_id=user_id,
        name=name.strip(),
        kind=kind,
        base_url=(base_url or None),
        api_key_encrypted=encrypt(api_key) if api_key else None,
        default_model=(default_model or None),
        extra=extra or {},
        is_default=is_default,
    )
    session.add(conn)
    await session.commit()
    await session.refresh(conn)
    return conn


async def update_connection(
    session: AsyncSession,
    conn_id: uuid.UUID,
    user_id: uuid.UUID,
    *,
    name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,  # None = leave unchanged; "" = clear
    default_model: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
    is_default: Optional[bool] = None,
) -> Optional[LlmConnection]:
    conn = await get_connection(session, conn_id, user_id)
    if conn is None:
        return None
    if conn.kind == "openai_chatgpt":
        if api_key is not None:
            raise ValueError("openai_chatgpt does not accept api_key")
        if base_url is not None:
            raise ValueError("openai_chatgpt does not accept base_url")
        if extra:
            raise ValueError("openai_chatgpt does not accept extra")
    if name is not None:
        conn.name = name.strip()
    if base_url is not None:
        conn.base_url = base_url or None
    if api_key is not None:
        conn.api_key_encrypted = encrypt(api_key) if api_key else None
    if default_model is not None:
        conn.default_model = default_model or None
    if extra is not None:
        conn.extra = extra
    if is_default is True:
        await _clear_default(session, user_id, except_id=conn.id)
        conn.is_default = True
    elif is_default is False:
        conn.is_default = False
    await session.commit()
    await session.refresh(conn)
    return conn


async def delete_connection(session: AsyncSession, conn_id: uuid.UUID, user_id: uuid.UUID) -> bool:
    from app.agents.services.connection_lock import lock_owned_connection
    from app.agents.models.oauth_credential import OpenAIOAuthCredential

    conn = await lock_owned_connection(session, conn_id, user_id)
    if conn is None:
        return False
    credential_row = (
        await session.execute(
            select(
                OpenAIOAuthCredential,
                and_(
                    OpenAIOAuthCredential.lifecycle_lease_token.is_not(None),
                    OpenAIOAuthCredential.lifecycle_lease_expires_at.is_not(None),
                    OpenAIOAuthCredential.lifecycle_lease_expires_at > func.now(),
                ).label("lease_is_live"),
            )
            .where(OpenAIOAuthCredential.connection_id == conn_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).one_or_none()
    credential = credential_row[0] if credential_row is not None else None
    lease_is_live = bool(credential_row[1]) if credential_row is not None else False
    if credential is not None and (
        credential.token_bundle_encrypted is not None
        or credential.refresh_attempt_id is not None
        or credential.last_error_code
        in {"refresh_outcome_unknown", "revocation_pending"}
        or lease_is_live
    ):
        raise ValueError("revoke the ChatGPT credential before deleting this connection")
    await session.delete(conn)
    await session.commit()
    return True


async def _clear_default(session: AsyncSession, user_id: uuid.UUID, *, except_id: Optional[uuid.UUID] = None) -> None:
    stmt = update(LlmConnection).where(LlmConnection.user_id == user_id, LlmConnection.is_default.is_(True))
    if except_id is not None:
        stmt = stmt.where(LlmConnection.id != except_id)
    await session.execute(stmt.values(is_default=False))


def build_provider_for_connection(conn: LlmConnection):
    """Materialize a runtime LLMProvider instance for this connection."""
    if conn.kind == "openai_chatgpt":
        # Layer 3 supplies the Responses provider. Until then, never reinterpret
        # an OAuth connection as API-key OpenAI (which changes account/billing).
        raise OAuthLifecycleError(
            OAuthFailureCode.INACTIVE,
            "ChatGPT OAuth provider is not available",
            definitive=True,
        )
    return build_provider(
        conn.kind,
        api_key=decrypt(conn.api_key_encrypted) or "",
        base_url=conn.base_url or None,
        model=conn.default_model,
    )


def _in_container() -> bool:
    """Whether this process runs inside a container. Only used to decide
    whether a routing hint would help; never to change behaviour."""
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def _routing_hint(url: str) -> str:
    """Point at the usual cause when a container can't reach a model server
    the user can reach from their browser: the address is theirs, not the
    container's."""
    if not url or not _in_container():
        return ""
    host = (urlparse(url).hostname or "").strip("[]").lower()
    if not host:
        return ""
    port = urlparse(url).port
    suggestion = f"http://host.docker.internal:{port}" if port else "http://host.docker.internal"

    if host == "localhost":
        loopback, private = True, False
    else:
        try:
            ip = ip_address(host)
        except ValueError:
            return ""
        loopback, private = ip.is_loopback, ip.is_private

    if loopback:
        return (
            f". Securo runs in a container, where localhost is the container itself. "
            f"Use {suggestion} when the model server runs on the host machine."
        )
    if private:
        return (
            f". Securo runs in a container, which often can't reach the host's LAN address. "
            f"Use {suggestion} when the model server runs on the host machine."
        )
    return ""


def _unreachable(url: str, exc: Exception) -> dict[str, Any]:
    """Never return a bare 'unreachable:'. Connect timeouts stringify to an
    empty message, which left the UI showing no reason at all."""
    reason = str(exc).strip() or type(exc).__name__
    where = f" at {url}" if url else ""
    return {"ok": False, "detail": f"unreachable{where}: {reason}{_routing_hint(url)}"}


async def test_connection(
    conn: LlmConnection,
    *,
    session: AsyncSession | None = None,
    now=None,
) -> dict[str, Any]:
    """Live probe. For Ollama we list models; for OpenAI/Anthropic we do
    a tiny embedding or a 1-token chat. Returns {ok, detail, models?}."""
    import httpx

    api_key = decrypt(conn.api_key_encrypted) or ""
    timeout = httpx.Timeout(10.0, connect=5.0)
    # Tracked so the failure path can name the address that was tried.
    attempted = conn.base_url or ""

    try:
        if conn.kind == "openai_chatgpt":
            from app.agents.services.oauth_credentials import (
                decrypt_bundle,
                get_owned_credential,
            )
            from app.agents.services.oauth_refresh import (
                disable_plan_usage,
                refresh_credential,
                require_reauthorization,
            )
            from app.agents.services.openai_oauth import list_models

            if session is None:
                raise OAuthLifecycleError(
                    OAuthFailureCode.INACTIVE,
                    "OAuth connection test requires credential storage",
                    definitive=True,
                )
            connection_id = conn.id
            connection_user_id = conn.user_id

            async def access_snapshot(
                *, force: bool = False, rejected_version: int | None = None
            ) -> tuple[str, int]:
                await refresh_credential(
                    session,
                    connection_id,
                    connection_user_id,
                    now=now,
                    force=force,
                    rejected_version=rejected_version,
                )
                credential = await get_owned_credential(
                    session, connection_id, connection_user_id
                )
                if credential is not None:
                    await session.refresh(credential)
                if credential is not None and (
                    credential.status != "active"
                    or credential.refresh_attempt_id is not None
                    or credential.last_error_code
                    in {"refresh_outcome_unknown", "revocation_pending"}
                ):
                    status = credential.status
                    error_code = credential.last_error_code
                    await session.rollback()
                    if error_code == "refresh_outcome_unknown":
                        raise OAuthLifecycleError(
                            OAuthFailureCode.REFRESH_OUTCOME_UNKNOWN,
                            "OAuth refresh outcome is unknown; explicit reauthorization is required",
                            definitive=True,
                        )
                    if error_code == "revocation_pending":
                        raise OAuthLifecycleError(
                            OAuthFailureCode.RACE_LOST,
                            "OAuth revocation is in progress",
                            retryable=True,
                        )
                    if status == "disabled":
                        raise OAuthLifecycleError(
                            OAuthFailureCode.SCOPE_LOSS,
                            "ChatGPT plan usage is disabled",
                            definitive=True,
                        )
                    if status == "revoked":
                        raise OAuthLifecycleError(
                            OAuthFailureCode.REVOKED,
                            "OAuth credential is not active",
                            definitive=True,
                        )
                    raise OAuthLifecycleError(
                        OAuthFailureCode.REAUTH_REQUIRED,
                        "OAuth credential is not active",
                        definitive=True,
                    )
                if credential is None or credential.token_bundle_encrypted is None:
                    await session.rollback()
                    raise OAuthLifecycleError(
                        OAuthFailureCode.RACE_LOST,
                        "OAuth credential changed during provider use",
                        retryable=True,
                    )
                try:
                    access_token = decrypt_bundle(
                        credential.token_bundle_encrypted
                    ).access_token
                except ValueError:
                    await session.rollback()
                    raise OAuthLifecycleError(
                        OAuthFailureCode.CORRUPT_TOKEN,
                        "OAuth credential cannot be decrypted",
                        definitive=True,
                    ) from None
                observed_version = credential.credential_version
                await session.commit()
                return access_token, observed_version

            access_token, observed_version = await access_snapshot()
            try:
                models = await list_models(access_token)
            except OAuthLifecycleError as exc:
                if exc.code is OAuthFailureCode.SCOPE_LOSS:
                    await disable_plan_usage(
                        session,
                        connection_id,
                        connection_user_id,
                        expected_version=observed_version,
                    )
                    raise
                if exc.code is not OAuthFailureCode.REAUTH_REQUIRED:
                    raise
                # Provider-side invalidation can precede our local expiry.
                # Force exactly one rotation/retry; never fall back to an API key.
                access_token, observed_version = await access_snapshot(
                    force=True, rejected_version=observed_version
                )
                try:
                    models = await list_models(access_token)
                except OAuthLifecycleError as retry_exc:
                    if retry_exc.code is OAuthFailureCode.SCOPE_LOSS:
                        await disable_plan_usage(
                            session,
                            connection_id,
                            connection_user_id,
                            expected_version=observed_version,
                        )
                    elif retry_exc.code is OAuthFailureCode.REAUTH_REQUIRED:
                        await require_reauthorization(
                            session,
                            connection_id,
                            connection_user_id,
                            expected_version=observed_version,
                            last_error_code=retry_exc.code.value,
                        )
                    raise
            return {
                "ok": True,
                "detail": f"reachable ({len(models)} models)",
                "models": models,
            }

        if conn.kind == "ollama":
            url = (conn.base_url or "http://ollama:11434").rstrip("/") + "/api/tags"
            attempted = url
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url)
                r.raise_for_status()
                models = [m.get("name") for m in (r.json().get("models") or [])]
            return {"ok": True, "detail": f"reachable ({len(models)} models)", "models": models}

        if conn.kind in ("openai", "openai_compatible"):
            # Use the same normalization the runtime provider applies, so
            # "test" hits the same URL "chat" will use. Detects bare-host
            # URLs (e.g. http://lmstudio:1234) and auto-appends /v1.
            from app.agents.providers.openai import normalize_openai_base_url

            base = normalize_openai_base_url(conn.base_url or "https://api.openai.com/v1")
            attempted = base
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{base}/models", headers=headers)
                if r.status_code == 401 or r.status_code == 403:
                    return {"ok": False, "detail": "auth rejected (check api key)"}
                if r.status_code >= 400:
                    return {"ok": False, "detail": f"HTTP {r.status_code}: {r.text[:200]}"}
                # Some servers (LM Studio with a bad path) return 200 with
                # no JSON body — surface that as a useful warning instead
                # of a misleading "ok".
                try:
                    payload = r.json()
                except Exception:
                    return {"ok": False, "detail": f"reachable but {base}/models did not return JSON — wrong path?"}
                models = [m.get("id") for m in (payload.get("data") or [])][:50]
                if not models and conn.kind == "openai_compatible":
                    return {
                        "ok": False,
                        "detail": f"reachable but {base}/models returned no models — check the base URL",
                    }
            return {"ok": True, "detail": f"reachable at {base} ({len(models)} models)", "models": models}

        if conn.kind == "anthropic":
            base = (conn.base_url or "https://api.anthropic.com/v1").rstrip("/")
            attempted = base
            headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(f"{base}/models", headers=headers)
                if r.status_code in (401, 403):
                    return {"ok": False, "detail": "auth rejected (check api key)"}
                if r.status_code >= 400:
                    return {"ok": False, "detail": f"HTTP {r.status_code}: {r.text[:200]}"}
                models = [m.get("id") for m in (r.json().get("data") or [])][:50]
            return {"ok": True, "detail": f"reachable ({len(models)} models)", "models": models}

        return {"ok": False, "detail": f"unknown kind: {conn.kind}"}

    except OAuthLifecycleError as exc:
        return {"ok": False, "detail": str(exc), "auth_state": exc.code.value}
    except httpx.HTTPError as exc:
        return _unreachable(attempted, exc)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "detail": str(exc).strip() or type(exc).__name__}
