from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest

from app.agents.providers.openai_chatgpt import _RESPONSES_URL
from app.agents.services.openai_oauth import JWKS_ENDPOINT, MODELS_ENDPOINT, REVOKE_ENDPOINT, TOKEN_ENDPOINT
from tests.integration.fake_openai import FakeOpenAI


def _challenge(verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()


def _issued_bundle(
    fake: FakeOpenAI, *, marker: str = "", client_id: str = "dynamic_agent_client"
) -> dict:
    verifier = "response-verifier-" * 4
    redirect = "http://127.0.0.1:43111/auth/callback"
    callback = fake.authorize(
        client_id=client_id,
        redirect_uri=redirect,
        nonce=("response-nonce-" * 4) + marker,
        code_challenge=_challenge(verifier),
        state=("response-state-" * 4) + marker,
    )
    values = parse_qs(urlsplit(callback).query)
    response = fake.handle_async(httpx.Request(
        "POST",
        TOKEN_ENDPOINT,
        content=urlencode({
            "grant_type": "authorization_code",
            "client_id": values["client_id"][0],
            "code": values["code"][0],
            "code_verifier": verifier,
            "redirect_uri": redirect,
            "resource": fake.resource,
        }),
        headers={"content-type": "application/x-www-form-urlencoded"},
    ))
    assert response.status_code == 200
    return response.json()


def _responses(fake: FakeOpenAI, access_token: str, payload: dict) -> httpx.Response:
    return fake.handle_async(httpx.Request(
        "POST",
        _RESPONSES_URL,
        json=payload,
        headers={"authorization": f"Bearer {access_token}"},
    ))


def _raw_responses(fake: FakeOpenAI, access_token: str, content: bytes) -> httpx.Response:
    return fake.handle_async(httpx.Request(
        "POST",
        _RESPONSES_URL,
        content=content,
        headers={
            "authorization": f"Bearer {access_token}",
            "content-type": "application/json",
        },
    ))


def _tool_payload() -> dict:
    return {
        "model": "gpt-integration",
        "input": [{"role": "user", "content": "Use the integration tool"}],
        "store": False,
        "stream": True,
        "tools": [{
            "type": "namespace",
            "name": "securo",
            "description": "Securo client-side tools",
            "tools": [{
                "type": "function",
                "name": "integration__lookup",
                "description": "Deterministic integration lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }],
        }],
    }


def _issued_call(response: httpx.Response) -> tuple[str, str, dict]:
    completed = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ][-1]
    call = completed["response"]["output"][0]
    return completed["response"]["id"], call["call_id"], call


def _continuation_payload(call: dict, *, call_id: str | None = None, output: str = '{"value": 42}') -> dict:
    payload = _tool_payload()
    payload["input"] = [
        {"role": "user", "content": "Use the integration tool"},
        {
            "type": "function_call",
            "namespace": "securo",
            "call_id": call["call_id"],
            "name": call["name"],
            "arguments": "{\"query\":\"integration\"}",
        },
        {
            "type": "function_call_output",
            "call_id": call_id or call["call_id"],
            "output": output,
        },
    ]
    return payload


def test_fake_enforces_code_binding_rotation_revocation_and_secret_free_journal():
    fake = FakeOpenAI()
    verifier = "v" * 64
    redirect = "http://127.0.0.1:43111/auth/callback"
    authorization = fake.authorize(
        client_id="dynamic_agent_client",
        redirect_uri=redirect,
        nonce="n" * 43,
        code_challenge=_challenge(verifier),
        state="s" * 43,
    )
    query = parse_qs(urlsplit(authorization).query)
    assert query["state"] == ["s" * 43]
    code = query["code"][0]
    issued = query["client_id"][0]

    def token(grant: dict[str, str]) -> httpx.Response:
        request = httpx.Request(
            "POST", TOKEN_ENDPOINT,
            content=urlencode(grant),
            headers={"content-type": "application/x-www-form-urlencoded"},
        )
        return fake.handle_async(request)

    wrong = token({
        "grant_type": "authorization_code", "client_id": issued, "code": code,
        "code_verifier": "wrong", "redirect_uri": redirect, "resource": fake.resource,
    })
    assert wrong.status_code == 400
    first = token({
        "grant_type": "authorization_code", "client_id": issued, "code": code,
        "code_verifier": verifier, "redirect_uri": redirect, "resource": fake.resource,
    })
    assert first.status_code == 200
    payload = first.json()
    assert token({
        "grant_type": "authorization_code", "client_id": issued, "code": code,
        "code_verifier": verifier, "redirect_uri": redirect, "resource": fake.resource,
    }).status_code == 400

    rotated = token({
        "grant_type": "refresh_token", "client_id": issued,
        "refresh_token": payload["refresh_token"], "resource": fake.resource,
    })
    assert rotated.status_code == 200
    assert rotated.json()["refresh_token"] != payload["refresh_token"]
    assert token({
        "grant_type": "refresh_token", "client_id": issued,
        "refresh_token": payload["refresh_token"], "resource": fake.resource,
    }).status_code == 400

    for url, method in ((JWKS_ENDPOINT, "GET"), (MODELS_ENDPOINT, "GET"), (REVOKE_ENDPOINT, "POST")):
        assert fake.handle_async(httpx.Request(method, url)).status_code in {200, 400, 401}
    journal = repr(fake.journal)
    for secret in (code, verifier, payload["access_token"], payload["refresh_token"]):
        assert secret not in journal


def test_fake_authorization_state_is_unique_one_use_and_bound():
    fake = FakeOpenAI()
    arguments = {
        "client_id": "dynamic_agent_client",
        "redirect_uri": "http://127.0.0.1:43111/auth/callback",
        "nonce": "n" * 43,
        "code_challenge": _challenge("v" * 64),
        "state": "s" * 43,
    }

    callback = fake.authorize(**arguments)
    assert parse_qs(urlsplit(callback).query)["state"] == [arguments["state"]]
    for mutation, value in (
        (None, None),
        ("client_id", "other-client"),
        ("redirect_uri", "http://127.0.0.1:43112/auth/callback"),
        ("nonce", "x" * 43),
        ("code_challenge", _challenge("x" * 64)),
    ):
        replay = dict(arguments)
        if mutation is not None:
            assert value is not None
            replay[mutation] = value
        with pytest.raises(ValueError, match="authorization state"):
            fake.authorize(**replay)


def test_fake_revoke_requires_exact_production_wire_contract():
    fake = FakeOpenAI()
    bundle = _issued_bundle(fake)
    client_id = fake.last_client_id
    assert client_id is not None
    valid_form = [
        ("token", bundle["refresh_token"]),
        ("token_type_hint", "refresh_token"),
        ("client_id", client_id),
    ]

    malformed = [
        httpx.Request("GET", REVOKE_ENDPOINT),
        httpx.Request("POST", REVOKE_ENDPOINT, json=dict(valid_form)),
        httpx.Request(
            "POST", REVOKE_ENDPOINT, content=urlencode(valid_form[:-1]),
            headers={"content-type": "application/x-www-form-urlencoded"},
        ),
        httpx.Request(
            "POST", REVOKE_ENDPOINT,
            content=urlencode(valid_form + [("token", bundle["refresh_token"])]),
            headers={"content-type": "application/x-www-form-urlencoded"},
        ),
        httpx.Request(
            "POST", REVOKE_ENDPOINT,
            content=urlencode(valid_form + [("extra", "not-allowed")]),
            headers={"content-type": "application/x-www-form-urlencoded"},
        ),
        httpx.Request(
            "POST", REVOKE_ENDPOINT,
            content=urlencode({**dict(valid_form), "token_type_hint": "access_token"}),
            headers={"content-type": "application/x-www-form-urlencoded"},
        ),
        httpx.Request(
            "POST", REVOKE_ENDPOINT,
            content=urlencode({**dict(valid_form), "client_id": "wrong-client"}),
            headers={"content-type": "application/x-www-form-urlencoded"},
        ),
        httpx.Request(
            "POST", REVOKE_ENDPOINT,
            content=urlencode(valid_form),
            headers={
                "content-type": "application/x-www-form-urlencoded",
                "authorization": "Bearer must-not-be-present",
            },
        ),
        httpx.Request(
            "POST", REVOKE_ENDPOINT + "/redirected",
            content=urlencode(valid_form),
            headers={"content-type": "application/x-www-form-urlencoded"},
        ),
    ]
    for request in malformed:
        response = fake.handle_async(request)
        assert response.status_code in {400, 404, 405, 413, 415}, request

    accepted = fake.handle_async(httpx.Request(
        "POST", REVOKE_ENDPOINT, content=urlencode(valid_form),
        headers={"content-type": "application/x-www-form-urlencoded"},
    ))
    assert accepted.status_code == 200
    assert accepted.content == b""

    assert fake.handle_async(httpx.Request(
        "GET", MODELS_ENDPOINT,
        headers={"authorization": f"Bearer {bundle['access_token']}"},
    )).status_code == 401


def test_fake_endpoints_reject_permissive_method_media_type_and_shape():
    fake = FakeOpenAI()
    bundle = _issued_bundle(fake)
    assert fake.handle_async(httpx.Request("POST", JWKS_ENDPOINT)).status_code == 405
    assert fake.handle_async(httpx.Request("GET", JWKS_ENDPOINT, content=b"x")).status_code == 400
    assert fake.handle_async(httpx.Request("POST", MODELS_ENDPOINT)).status_code == 405
    assert fake.handle_async(httpx.Request(
        "GET", MODELS_ENDPOINT,
        headers={"authorization": f"bearer {bundle['access_token']}"},
    )).status_code == 401
    assert fake.handle_async(httpx.Request(
        "POST", TOKEN_ENDPOINT,
        content="grant_type=refresh_token&grant_type=refresh_token",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )).status_code in {400, 415}
    assert fake.handle_async(httpx.Request(
        "POST", _RESPONSES_URL, content=json.dumps(_tool_payload()),
        headers={"authorization": f"Bearer {bundle['access_token']}"},
    )).status_code == 415
    assert fake.handle_async(httpx.Request(
        "POST", _RESPONSES_URL, json={**_tool_payload(), "unexpected": True},
        headers={"authorization": f"Bearer {bundle['access_token']}"},
    )).status_code == 400


def test_fake_generates_ephemeral_signing_material_without_repository_private_key():
    first = FakeOpenAI()
    second = FakeOpenAI()

    assert first.jwks["keys"][0]["n"] != second.jwks["keys"][0]["n"]
    integration_dir = Path(__file__).parent
    assert not list(integration_dir.glob("*.pem"))


def test_fake_scripted_redirect_is_exactly_once():
    fake = FakeOpenAI()
    fake.script(TOKEN_ENDPOINT, httpx.Response(307, headers={"location": "https://evil.invalid"}))
    request = httpx.Request("POST", TOKEN_ENDPOINT, content=b"")
    assert fake.handle_async(request).status_code == 307
    assert fake.handle_async(request).status_code == 415


def test_fake_can_lose_a_response_after_consuming_a_refresh_token():
    fake = FakeOpenAI()
    verifier = "v" * 64
    redirect = "http://127.0.0.1:43111/auth/callback"
    callback = fake.authorize(
        client_id="dynamic_agent_client",
        redirect_uri=redirect,
        nonce="n" * 43,
        code_challenge=_challenge(verifier),
        state="s" * 43,
    )
    values = parse_qs(urlsplit(callback).query)
    issued = values["client_id"][0]

    def token(grant: dict[str, str]) -> httpx.Response:
        return fake.handle_async(httpx.Request(
            "POST",
            TOKEN_ENDPOINT,
            content=urlencode(grant),
            headers={"content-type": "application/x-www-form-urlencoded"},
        ))

    initial = token({
        "grant_type": "authorization_code", "client_id": issued,
        "code": values["code"][0], "code_verifier": verifier,
        "redirect_uri": redirect, "resource": fake.resource,
    }).json()
    refresh_grant = {
        "grant_type": "refresh_token", "client_id": issued,
        "refresh_token": initial["refresh_token"], "resource": fake.resource,
    }
    fake.script_after(TOKEN_ENDPOINT, httpx.ReadError("response lost"))
    with pytest.raises(httpx.ReadError, match="response lost"):
        token(refresh_grant)
    assert token(refresh_grant).status_code == 400


@pytest.mark.asyncio
async def test_fake_async_transport_and_sync_helper_transport_share_state():
    fake = FakeOpenAI()
    assert type(fake.async_transport).__name__ == "_AsyncTransport"
    assert type(fake.sync_transport).__name__ == "_SyncTransport"
    async with httpx.AsyncClient(transport=fake.async_transport) as client:
        assert (await client.get(JWKS_ENDPOINT)).status_code == 200
    with httpx.Client(transport=fake.sync_transport) as client:
        assert client.get(JWKS_ENDPOINT).status_code == 200
    assert [entry["endpoint"] for entry in fake.journal] == ["jwks", "jwks"]


def test_fake_rejects_malformed_outbound_protocol_requests():
    fake = FakeOpenAI()
    assert fake.handle_async(httpx.Request("POST", TOKEN_ENDPOINT, content=b"grant_type=refresh_token")).status_code == 415
    assert fake.handle_async(httpx.Request("GET", MODELS_ENDPOINT, content=b"unexpected")).status_code == 400
    assert fake.handle_async(httpx.Request(
        "POST",
        "https://api.openai.com/v1/responses",
        json={"model": "gpt-integration", "input": [], "store": True, "stream": True},
        headers={"authorization": "Bearer invalid"},
    )).status_code == 400


@pytest.mark.parametrize(
    "mutation",
    ["unknown_call", "mismatched_call", "malformed_order", "wrong_output"],
)
def test_fake_rejects_invalid_responses_tool_continuations(mutation: str):
    fake = FakeOpenAI()
    bundle = _issued_bundle(fake)
    first = _responses(fake, bundle["access_token"], _tool_payload())
    assert first.status_code == 200
    _response_id, call_id, call = _issued_call(first)
    payload = _continuation_payload(call)
    if mutation == "unknown_call":
        payload["input"][-1]["call_id"] = "call-never-issued"
    elif mutation == "mismatched_call":
        payload["input"][-2]["name"] = "integration__other"
    elif mutation == "malformed_order":
        payload["input"][-2:] = reversed(payload["input"][-2:])
    else:
        payload["input"][-1]["output"] = '{"value": 41}'

    rejected = _responses(fake, bundle["access_token"], payload)

    assert rejected.status_code == 400
    assert rejected.json() == {"error": {"code": "invalid_tool_continuation"}}
    # A rejected continuation must not consume the one-use issued call.
    assert _responses(
        fake, bundle["access_token"], _continuation_payload(call)
    ).status_code == 200
    assert call_id


def test_fake_accepts_exact_continuation_once_and_never_finalizes_duplicate():
    fake = FakeOpenAI()
    bundle = _issued_bundle(fake)
    first = _responses(fake, bundle["access_token"], _tool_payload())
    _response_id, _call_id, call = _issued_call(first)
    continuation = _continuation_payload(call)

    accepted = _responses(fake, bundle["access_token"], continuation)
    duplicate = _responses(fake, bundle["access_token"], continuation)

    assert accepted.status_code == 200
    assert "assembled integration complete" in accepted.text
    assert duplicate.status_code == 400
    assert duplicate.json() == {"error": {"code": "invalid_tool_continuation"}}


def test_fake_rejects_extra_unknown_function_call_anywhere_in_history():
    fake = FakeOpenAI()
    bundle = _issued_bundle(fake)
    first = _responses(fake, bundle["access_token"], _tool_payload())
    _response_id, _call_id, call = _issued_call(first)
    payload = _continuation_payload(call)
    payload["input"].insert(1, {
        "type": "function_call",
        "namespace": "securo",
        "call_id": "call-never-issued",
        "name": "integration__lookup",
        "arguments": "{\"query\":\"integration\"}",
    })

    rejected = _responses(fake, bundle["access_token"], payload)

    assert rejected.status_code == 400
    assert rejected.json() == {"error": {"code": "invalid_tool_continuation"}}
    assert _responses(fake, bundle["access_token"], _continuation_payload(call)).status_code == 200


@pytest.mark.parametrize(
    "orientation",
    [
        "unknown_call_before", "unknown_call_after", "duplicate_call_before_output",
        "duplicate_call_after_output", "extra_output_before", "extra_output_after",
        "known_pairs_reversed", "known_call_wrong_client", "known_call_wrong_lineage",
        "malformed_unrelated_item_before", "malformed_unrelated_item_after",
    ],
)
def test_fake_rejects_complete_tool_history_orientation_matrix(orientation: str):
    fake = FakeOpenAI()
    first_bundle = _issued_bundle(fake)
    first_client_id = fake.last_client_id
    assert first_client_id is not None
    first_response = _responses(fake, first_bundle["access_token"], _tool_payload())
    _response_id, _call_id, first_call = _issued_call(first_response)
    second_response = _responses(fake, first_bundle["access_token"], _tool_payload())
    _response_id, _call_id, second_call = _issued_call(second_response)
    first_pair = _continuation_payload(first_call)["input"][-2:]
    second_pair = _continuation_payload(second_call)["input"][-2:]
    payload = _tool_payload()
    payload["input"] = [{"role": "user", "content": "Use the integration tool"}, *first_pair]
    unknown_call = {
        "type": "function_call", "namespace": "securo", "call_id": "call-never-issued",
        "name": "integration__lookup", "arguments": "{\"query\":\"integration\"}",
    }
    unknown_output = {
        "type": "function_call_output", "call_id": "call-never-issued",
        "output": '{"value": 42}',
    }
    if orientation == "unknown_call_before":
        payload["input"].insert(1, unknown_call)
    elif orientation == "unknown_call_after":
        payload["input"].append(unknown_call)
    elif orientation == "duplicate_call_before_output":
        payload["input"].insert(-1, dict(first_pair[0]))
    elif orientation == "duplicate_call_after_output":
        payload["input"].append(dict(first_pair[0]))
    elif orientation == "extra_output_before":
        payload["input"].insert(1, unknown_output)
    elif orientation == "extra_output_after":
        payload["input"].append(unknown_output)
    elif orientation == "known_pairs_reversed":
        payload["input"] = [payload["input"][0], *second_pair, *first_pair]
    elif orientation in {"known_call_wrong_client", "known_call_wrong_lineage"}:
        other_bundle = _issued_bundle(
            fake,
            marker=orientation,
            client_id=(
                first_client_id
                if orientation == "known_call_wrong_lineage"
                else "dynamic_agent_client"
            ),
        )
        other_response = _responses(fake, other_bundle["access_token"], _tool_payload())
        _response_id, _call_id, other_call = _issued_call(other_response)
        payload["input"] = [payload["input"][0], *_continuation_payload(other_call)["input"][-2:]]
    elif orientation == "malformed_unrelated_item_before":
        payload["input"].insert(1, {"role": "user", "content": "x", "extra": True})
    else:
        payload["input"].append({"role": "assistant", "content": 7})

    rejected = _responses(fake, first_bundle["access_token"], payload)

    assert rejected.status_code == 400


def test_fake_accepts_multiple_exact_issued_pairs_in_issue_order_once():
    fake = FakeOpenAI()
    bundle = _issued_bundle(fake)
    calls = [
        _issued_call(_responses(fake, bundle["access_token"], _tool_payload()))[2]
        for _ in range(2)
    ]
    payload = _tool_payload()
    payload["input"] = [
        {"role": "user", "content": "Use the integration tool"},
        *[item for call in calls for item in _continuation_payload(call)["input"][-2:]],
    ]

    accepted = _responses(fake, bundle["access_token"], payload)

    assert accepted.status_code == 200
    assert _responses(fake, bundle["access_token"], payload).status_code == 400


@pytest.mark.parametrize(
    "tools",
    [
        [],
        [{"type": "function", "name": "integration__lookup"}],
        [{
            "type": "namespace",
            "name": "wrong",
            "description": "Securo client-side tools",
            "tools": [{
                "type": "function",
                "name": "integration__lookup",
                "description": "Deterministic integration lookup",
                "parameters": {"type": "object"},
            }],
        }],
    ],
)
def test_fake_rejects_unadvertised_or_malformed_tool_schema(tools: list[dict]):
    fake = FakeOpenAI()
    bundle = _issued_bundle(fake)
    payload = _tool_payload()
    payload["tools"] = tools

    response = _responses(fake, bundle["access_token"], payload)

    assert response.status_code == 400
    assert response.json() == {"error": {"code": "invalid_tools"}}


@pytest.mark.parametrize(
    "content",
    [
        b'{"model":"gpt-integration","model":"duplicate","input":[],"store":false,"stream":true}',
        b'{"model":"gpt-integration","input":[],"store":false,"stream":true,"unknown":null}',
        b'{"model":"gpt-integration","input":[],"store":false,"stream":true,"temperature":NaN}',
        b'{"model":"gpt-integration","input":[],"store":false,"stream":true,"temperature":Infinity}',
        b'{"model":"gpt-integration","input":[{"role":"user","content":"ok","extra":true}],"store":false,"stream":true}',
        b'{"model":"gpt-integration","input":[{"role":"system","content":"forbidden"}],"store":false,"stream":true}',
        b'{"model":"gpt-integration","input":[{"type":"function_call_output","call_id":"x"}],"store":false,"stream":true}',
        b'{"model":"gpt-integration","input":[{"type":"invented","call_id":"x","output":"y"}],"store":false,"stream":true}',
        (
            b'{"model":"gpt-integration","input":[],"store":false,"stream":true,'
            b'"instructions":"' + b'x' * 1_000_001 + b'"}'
        ),
        (
            b'{"model":"gpt-integration","input":[],"store":false,"stream":true,'
            b'"instructions":' + b'[' * 40 + b'null' + b']' * 40 + b'}'
        ),
    ],
)
def test_fake_responses_strict_parser_rejects_adversarial_json(content: bytes):
    fake = FakeOpenAI()
    bundle = _issued_bundle(fake)

    response = _raw_responses(fake, bundle["access_token"], content)

    assert response.status_code in {400, 413}


def test_fake_responses_strict_parser_rejects_excessive_nodes_and_body_bytes():
    fake = FakeOpenAI()
    bundle = _issued_bundle(fake)
    too_many_nodes = json.dumps({
        **_tool_payload(),
        "input": [{"role": "user", "content": "x"} for _ in range(20_001)],
    }).encode()
    oversized = b"{" + b" " * 1_000_000 + b"}"

    assert _raw_responses(fake, bundle["access_token"], too_many_nodes).status_code in {400, 413}
    assert _raw_responses(fake, bundle["access_token"], oversized).status_code == 413
