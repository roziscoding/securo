"""HTTP coverage for /api/agents/connections — LLM connection CRUD plus
the /test probe endpoint."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from app.agents.services import connection_service
from app.agents.services.oauth_credentials import CredentialBundle, create_credential


@pytest.mark.asyncio
async def test_list_connections_empty(client, auth_headers, test_user):
    r = await client.get("/api/agents/connections", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_create_connection_persists_and_hides_api_key(client, auth_headers, test_user):
    payload = {
        "name": "My OpenAI",
        "kind": "openai",
        "api_key": "sk-secret-123",
        "default_model": "gpt-4o-mini",
    }
    r = await client.post("/api/agents/connections", json=payload, headers=auth_headers)
    assert r.status_code == 201, r.text
    body = r.json()
    # has_api_key reflects presence; the literal key never round-trips.
    assert body["has_api_key"] is True
    assert "api_key" not in body
    assert body["default_model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_create_connection_400_for_unknown_kind(client, auth_headers, test_user):
    r = await client.post(
        "/api/agents/connections",
        json={"name": "x", "kind": "bogus"},
        headers=auth_headers,
    )
    assert r.status_code == 400
    assert "unknown kind" in r.json()["detail"]


@pytest.mark.asyncio
async def test_create_connection_400_for_openai_compat_without_base_url(client, auth_headers, test_user):
    r = await client.post(
        "/api/agents/connections",
        json={"name": "x", "kind": "openai_compatible"},
        headers=auth_headers,
    )
    assert r.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forbidden_field,forbidden_value",
    [("api_key", "must-not-store"), ("base_url", "https://evil.example/v1")],
)
async def test_openai_chatgpt_rejects_key_and_base_url(
    client, auth_headers, test_user, forbidden_field, forbidden_value
):
    r = await client.post(
        "/api/agents/connections",
        json={
            "name": "ChatGPT",
            "kind": "openai_chatgpt",
            forbidden_field: forbidden_value,
        },
        headers=auth_headers,
    )

    assert r.status_code == 400
    assert forbidden_field in r.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forbidden_field,forbidden_value",
    [("api_key", "must-not-store"), ("base_url", "https://evil.example/v1")],
)
async def test_openai_chatgpt_update_rejects_key_and_base_url(
    client, auth_headers, test_user, forbidden_field, forbidden_value
):
    created = await client.post(
        "/api/agents/connections",
        json={"name": "ChatGPT", "kind": "openai_chatgpt"},
        headers=auth_headers,
    )
    assert created.status_code == 201

    r = await client.patch(
        f"/api/agents/connections/{created.json()['id']}",
        json={forbidden_field: forbidden_value},
        headers=auth_headers,
    )

    assert r.status_code == 400
    assert forbidden_field in r.json()["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["create", "update"])
async def test_openai_chatgpt_api_rejects_nonempty_extra_without_echoing_nested_secret(
    client, auth_headers, test_user, method
):
    if method == "create":
        response = await client.post(
            "/api/agents/connections",
            json={
                "name": "ChatGPT",
                "kind": "openai_chatgpt",
                "extra": {"oauth": {"refresh_token": "api-nested-secret"}},
            },
            headers=auth_headers,
        )
    else:
        created = await client.post(
            "/api/agents/connections",
            json={"name": "ChatGPT", "kind": "openai_chatgpt"},
            headers=auth_headers,
        )
        assert created.status_code == 201
        response = await client.patch(
            f"/api/agents/connections/{created.json()['id']}",
            json={"extra": {"oauth": {"access_token": "api-nested-secret"}}},
            headers=auth_headers,
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "openai_chatgpt does not accept extra"
    assert "api-nested-secret" not in response.text


@pytest.mark.asyncio
async def test_chatgpt_connection_read_forces_empty_extra(
    client, auth_headers, session, test_user
):
    connection = await connection_service.create_connection(
        session, test_user.id, name="ChatGPT", kind="openai_chatgpt"
    )
    connection.extra = {"oauth": {"refresh_token": "legacy-nested-secret"}}
    await session.commit()

    response = await client.get(
        f"/api/agents/connections/{connection.id}", headers=auth_headers
    )

    assert response.status_code == 200
    assert response.json()["extra"] == {}
    assert "legacy-nested-secret" not in response.text


@pytest.mark.asyncio
async def test_chatgpt_connection_read_exposes_safe_typed_auth_status_only(
    client, auth_headers, session, test_user
):
    created = await client.post(
        "/api/agents/connections",
        json={"name": "ChatGPT", "kind": "openai_chatgpt"},
        headers=auth_headers,
    )
    assert created.status_code == 201
    connection_id = uuid.UUID(created.json()["id"])
    connection = await connection_service.get_connection(session, connection_id, test_user.id)
    assert connection is not None
    now = datetime.now(timezone.utc)
    await create_credential(
        session,
        connection.id,
        test_user.id,
        bundle=CredentialBundle("api-access-secret", "api-refresh-secret", "api-id-secret"),
        issuer="https://auth.openai.com",
        subject="subject-not-for-api",
        issued_client_id="client-not-for-api",
        account_email="owner@example.com",
        scopes=["openid", "chatgpt.tokens.use.direct"],
        access_token_expires_at=now + timedelta(hours=1),
    )

    response = await client.get(
        f"/api/agents/connections/{connection_id}", headers=auth_headers
    )

    assert response.status_code == 200
    body = response.json()
    assert body["auth_method"] == "chatgpt_oauth"
    assert body["credential_status"] == "active"
    assert body["account_email"] == "owner@example.com"
    assert body["scopes"] == ["openid", "chatgpt.tokens.use.direct"]
    assert body["needs_reauthorization"] is False
    assert body["has_api_key"] is False
    serialized = response.text
    for forbidden in (
        "api-access-secret",
        "api-refresh-secret",
        "api-id-secret",
        "subject-not-for-api",
        "client-not-for-api",
        "token_bundle_encrypted",
        "issued_client_id",
        "subject",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_get_update_delete_connection(client, auth_headers, test_user):
    r = await client.post(
        "/api/agents/connections",
        json={"name": "A", "kind": "openai"},
        headers=auth_headers,
    )
    cid = r.json()["id"]

    # GET
    r = await client.get(f"/api/agents/connections/{cid}", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["name"] == "A"

    # PATCH
    r = await client.patch(
        f"/api/agents/connections/{cid}",
        json={"name": "Renamed", "is_default": True},
        headers=auth_headers,
    )
    assert r.status_code == 200
    assert r.json()["name"] == "Renamed"
    assert r.json()["is_default"] is True

    # DELETE
    r = await client.delete(f"/api/agents/connections/{cid}", headers=auth_headers)
    assert r.status_code == 204

    # 404 after delete
    r = await client.get(f"/api/agents/connections/{cid}", headers=auth_headers)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_get_connection_404(client, auth_headers, test_user):
    r = await client.get(
        f"/api/agents/connections/{uuid.uuid4()}", headers=auth_headers
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_update_connection_404(client, auth_headers, test_user):
    r = await client.patch(
        f"/api/agents/connections/{uuid.uuid4()}",
        json={"name": "x"},
        headers=auth_headers,
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_connection_404(client, auth_headers, test_user):
    r = await client.delete(
        f"/api/agents/connections/{uuid.uuid4()}", headers=auth_headers
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_test_connection_uses_service_probe(client, auth_headers, test_user):
    """The /test endpoint should delegate to connection_service.test_connection
    and return its structured result."""
    r = await client.post(
        "/api/agents/connections",
        json={"name": "A", "kind": "ollama"},
        headers=auth_headers,
    )
    cid = r.json()["id"]

    async def fake_probe(conn):
        return {"ok": True, "detail": "fake reachable", "models": ["llama3"]}

    with patch(
        "app.agents.api.connections.connection_service.test_connection",
        side_effect=fake_probe,
    ):
        r = await client.post(
            f"/api/agents/connections/{cid}/test", headers=auth_headers
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["detail"] == "fake reachable"
    assert body["models"] == ["llama3"]


@pytest.mark.asyncio
async def test_test_connection_404_for_unknown(client, auth_headers, test_user):
    r = await client.post(
        f"/api/agents/connections/{uuid.uuid4()}/test", headers=auth_headers
    )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_unauthenticated_connections_rejected(client):
    r = await client.get("/api/agents/connections")
    assert r.status_code == 401
