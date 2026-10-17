"""Stateful, authentic OpenAI SIWC fake used only by integration tests.

The fake keeps production URLs intact. Its journal stores request hashes and
non-secret identifiers, never authorization codes or bearer/refresh tokens.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode

import httpx
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

from app.agents.services.openai_oauth import (
    ISSUER,
    JWKS_ENDPOINT,
    MODELS_ENDPOINT,
    REQUIRED_SCOPES,
    RESOURCE,
    REVOKE_ENDPOINT,
    TOKEN_ENDPOINT,
)
from app.agents.providers.openai_chatgpt import _RESPONSES_URL

_MAX_RESPONSES_BODY_BYTES = 1_000_000
_MAX_RESPONSES_STRING_BYTES = 1_000_000
_MAX_RESPONSES_JSON_DEPTH = 32
_MAX_RESPONSES_JSON_NODES = 20_000
_RESPONSES_FIELDS = frozenset({"model", "input", "store", "stream", "tools", "instructions"})


def _b64_int(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _digest(value: str | bytes) -> str:
    raw = value.encode() if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _at_hash(access_token: str) -> str:
    raw = hashlib.sha256(access_token.encode()).digest()
    return base64.urlsafe_b64encode(raw[: len(raw) // 2]).rstrip(b"=").decode()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def _bounded_json(data: bytes) -> Any:
    if len(data) > _MAX_RESPONSES_BODY_BYTES:
        raise OverflowError("request body exceeds bound")
    value = json.loads(
        data,
        object_pairs_hook=_unique_json_object,
        parse_constant=lambda _value: (_ for _ in ()).throw(
            ValueError("non-finite JSON number")
        ),
    )
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_RESPONSES_JSON_NODES or depth > _MAX_RESPONSES_JSON_DEPTH:
            raise ValueError("JSON graph exceeds bound")
        if isinstance(item, str):
            if "\x00" in item or len(item.encode("utf-8")) > _MAX_RESPONSES_STRING_BYTES:
                raise ValueError("JSON string exceeds bound")
        elif isinstance(item, dict):
            for key, child in item.items():
                if (
                    not isinstance(key, str)
                    or "\x00" in key
                    or len(key.encode("utf-8")) > _MAX_RESPONSES_STRING_BYTES
                ):
                    raise ValueError("JSON key exceeds bound")
                stack.append((child, depth + 1))
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif item is not None and not isinstance(item, (bool, int, float)):
            raise ValueError("non-JSON value")
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError("non-finite JSON number")
    return value


def _valid_response_input_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    if set(item) == {"role", "content"}:
        return item["role"] in {"user", "assistant"} and isinstance(item["content"], str)
    if set(item) == {"type", "namespace", "call_id", "name", "arguments"}:
        return (
            item["type"] == "function_call"
            and item["namespace"] == "securo"
            and all(isinstance(item[name], str) and item[name] for name in ("call_id", "name"))
            and isinstance(item["arguments"], str)
        )
    if set(item) == {"type", "call_id", "output"}:
        return (
            item["type"] == "function_call_output"
            and isinstance(item["call_id"], str)
            and bool(item["call_id"])
            and isinstance(item["output"], str)
        )
    return False


@dataclass
class _Code:
    client_id: str
    redirect_uri: str
    nonce: str
    challenge: str
    subject: str
    state: str
    used: bool = False


@dataclass
class _Refresh:
    client_id: str
    subject: str
    lineage: str
    used: bool = False
    revoked: bool = False


@dataclass
class _ResponseCall:
    order: int
    response_id: str
    client_id: str
    lineage: str
    call_id: str
    item_id: str
    name: str
    arguments: str
    expected_output: str
    used: bool = False


class _AsyncBytes(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def __aiter__(self):
        yield self.content


class DispatchBarrier:
    """Test-only provider-dispatch rendezvous without weakening production locks."""

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self._release = asyncio.Event()
        self.arrivals = 0

    async def wait(self) -> None:
        self.arrivals += 1
        self.reached.set()
        await self._release.wait()

    def release(self) -> None:
        self._release.set()


class _SyncTransport(httpx.BaseTransport):
    def __init__(self, engine: "FakeOpenAI") -> None:
        self.engine = engine

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self.engine.handle_async(request)
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
            extensions=response.extensions,
        )


class _AsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, engine: "FakeOpenAI") -> None:
        self.engine = engine

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        event = self.engine._barriers.get(str(request.url))
        if event is not None:
            await event.wait()
        response = self.engine.handle_async(request)
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            stream=_AsyncBytes(response.content),
            request=request,
            extensions=response.extensions,
        )


class FakeOpenAI:
    """Deterministic state machine with real RS256/JWKS verification material."""

    resource = RESOURCE

    def __init__(self) -> None:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self._key = key
        numbers = key.public_key().public_numbers()
        self.jwks = {
            "keys": [{
                "kty": "RSA", "kid": "securo-integration", "alg": "RS256", "use": "sig",
                "n": _b64_int(numbers.n), "e": _b64_int(numbers.e),
            }]
        }
        self._counter = 0
        self._lock = threading.RLock()
        self.default_subject = self._next("subject")
        self._codes: dict[str, _Code] = {}
        self._refresh: dict[str, _Refresh] = {}
        self._access: dict[str, tuple[str, str, int]] = {}
        self._revoked_lineages: set[str] = set()
        self._response_calls: dict[str, _ResponseCall] = {}
        self._response_call_order = 0
        self._id_tokens: set[str] = set()
        self._subjects: set[str] = set()
        self._client_ids: set[str] = set()
        self._nonces: set[str] = set()
        self._authorization_states: dict[str, tuple[str, str, str, str]] = {}
        self._next_identity_claim_overrides: dict[str, Any] = {}
        self._scripts: dict[str, deque[httpx.Response]] = defaultdict(deque)
        self._after_scripts: dict[str, deque[httpx.Response | Exception]] = defaultdict(deque)
        self._barriers: dict[str, DispatchBarrier] = {}
        self.journal: list[dict[str, Any]] = []
        self.models = [
            {"slug": "gpt-integration", "display_name": "GPT Integration", "visibility": "list"}
        ]
        self.refresh_calls = 0
        self.responses_calls = 0
        self.last_client_id: str | None = None
        self.async_transport = _AsyncTransport(self)
        self.sync_transport = _SyncTransport(self)

    def _next(self, prefix: str) -> str:
        with self._lock:
            self._counter += 1
            return f"{prefix}-{self._counter:04d}"

    def script(self, url: str, response: httpx.Response) -> None:
        with self._lock:
            self._scripts[url].append(response)

    def script_after(self, url: str, result: httpx.Response | Exception) -> None:
        """Replace/lose one response only after the endpoint mutates state."""
        with self._lock:
            self._after_scripts[url].append(result)

    def barrier(self, url: str) -> DispatchBarrier:
        barrier = DispatchBarrier()
        self._barriers[url] = barrier
        return barrier

    def override_next_identity_claim(self, name: str, value: Any) -> None:
        """Inject one signed bad claim for independent validator tests."""
        with self._lock:
            self._next_identity_claim_overrides[name] = value

    def authorize(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        nonce: str,
        code_challenge: str,
        state: str,
        subject: str | None = None,
    ) -> str:
        with self._lock:
            if not redirect_uri.startswith("http://127.0.0.1:"):
                raise ValueError("redirect_uri must use literal loopback")
            binding = (client_id, redirect_uri, nonce, code_challenge)
            if state in self._authorization_states:
                raise ValueError("authorization state was already consumed")
            self._authorization_states[state] = binding
            if client_id == "dynamic_agent_client":
                client_id = self._next("oaiapp-securo")
            subject = subject or self.default_subject
            code = self._next("code")
            self._codes[code] = _Code(
                client_id, redirect_uri, nonce, code_challenge, subject, state
            )
            self._subjects.add(subject)
            self._client_ids.add(client_id)
            self._nonces.add(nonce)
            self.last_client_id = client_id
            return redirect_uri + "?" + urlencode({"state": state, "code": code, "client_id": client_id})

    def expire_access_tokens(self) -> None:
        with self._lock:
            self._access = {
                token: (client_id, lineage, 0)
                for token, (client_id, lineage, _expires) in self._access.items()
            }

    def _journal(self, request: httpx.Request, endpoint: str, **ids: Any) -> None:
        authorization = request.headers.get("authorization", "")
        self.journal.append({
            "endpoint": endpoint,
            "request_id": self._next("request"),
            "body_sha256": _digest(request.content),
            "authorization_sha256": _digest(authorization) if authorization else None,
            **ids,
        })

    def _scripted(self, request: httpx.Request) -> httpx.Response | None:
        queue = self._scripts[str(request.url)]
        if not queue:
            return None
        response = queue.popleft()
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=response.content,
            request=request,
        )

    def _after_scripted(
        self, request: httpx.Request, response: httpx.Response
    ) -> httpx.Response:
        queue = self._after_scripts[str(request.url)]
        if not queue:
            return response
        result = queue.popleft()
        if isinstance(result, Exception):
            raise result
        return httpx.Response(
            result.status_code,
            headers=result.headers,
            content=result.content,
            request=request,
        )

    def handle_async(self, request: httpx.Request) -> httpx.Response:
        with self._lock:
            return self._handle(request)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        scripted = self._scripted(request)
        if scripted is not None:
            self._journal(request, "scripted")
            return scripted
        url = str(request.url)
        endpoint_methods = {
            JWKS_ENDPOINT: "GET",
            TOKEN_ENDPOINT: "POST",
            MODELS_ENDPOINT: "GET",
            REVOKE_ENDPOINT: "POST",
            _RESPONSES_URL: "POST",
        }
        expected_method = endpoint_methods.get(url)
        if expected_method is not None and request.method != expected_method:
            self._journal(request, "method_rejected")
            return httpx.Response(405, headers={"allow": expected_method}, request=request)
        if url == JWKS_ENDPOINT:
            if request.content or request.headers.get("authorization"):
                return httpx.Response(400, json={"error": "invalid_request"}, request=request)
            self._journal(request, "jwks")
            return httpx.Response(200, json=self.jwks, request=request)
        if url == TOKEN_ENDPOINT:
            return self._after_scripted(request, self._token(request))
        if url == MODELS_ENDPOINT:
            return self._models(request)
        if url == REVOKE_ENDPOINT:
            return self._revoke(request)
        if url == _RESPONSES_URL:
            return self._responses(request)
        self._journal(request, "unknown")
        return httpx.Response(404, json={"error": "not_found"}, request=request)

    @staticmethod
    def _form(request: httpx.Request) -> dict[str, str] | None:
        if request.headers.get("content-type") != "application/x-www-form-urlencoded":
            return None
        if request.headers.get("authorization") or len(request.content) > 16_384:
            return None
        try:
            pairs = parse_qsl(
                request.content.decode("ascii"),
                keep_blank_values=True,
                strict_parsing=True,
                max_num_fields=8,
            )
        except (UnicodeDecodeError, ValueError):
            return None
        if len({key for key, _value in pairs}) != len(pairs):
            return None
        return dict(pairs)

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = self._form(request)
        if form is None:
            return httpx.Response(415, json={"error": "unsupported_media_type"}, request=request)
        grant = form.get("grant_type")
        client_id = form.get("client_id", "")
        self._journal(
            request,
            "token",
            grant_type=grant,
            client_id_sha256=_digest(client_id) if client_id else None,
        )
        if grant == "authorization_code":
            if set(form) != {
                "grant_type", "client_id", "code", "code_verifier", "redirect_uri", "resource"
            } or not all(form.values()):
                return httpx.Response(400, json={"error": "invalid_request"}, request=request)
            record = self._codes.get(form.get("code", ""))
            verifier = form.get("code_verifier", "")
            challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            if (
                record is None or record.used or record.client_id != client_id
                or record.redirect_uri != form.get("redirect_uri")
                or form.get("resource") != RESOURCE or challenge != record.challenge
            ):
                return httpx.Response(400, json={"error": "invalid_grant"}, request=request)
            record.used = True
            lineage = self._next("lineage")
            return self._issued(request, client_id, record.subject, record.nonce, lineage)
        if grant == "refresh_token":
            if set(form) != {"grant_type", "client_id", "refresh_token", "resource"} or not all(
                form.values()
            ):
                return httpx.Response(400, json={"error": "invalid_request"}, request=request)
            self.refresh_calls += 1
            token = form.get("refresh_token", "")
            record = self._refresh.get(token)
            if (
                record is None or record.used or record.revoked
                or record.lineage in self._revoked_lineages
                or record.client_id != client_id or form.get("resource") != RESOURCE
            ):
                return httpx.Response(400, json={"error": "refresh_token_reused"}, request=request)
            record.used = True
            return self._issued(request, client_id, record.subject, None, record.lineage)
        return httpx.Response(400, json={"error": "unsupported_grant_type"}, request=request)

    def _issued(
        self,
        request: httpx.Request,
        client_id: str,
        subject: str,
        nonce: str | None,
        lineage: str,
    ) -> httpx.Response:
        now = int(time.time())
        expires = now + 3600
        access_id = self._next("access")
        scopes = " ".join(sorted(REQUIRED_SCOPES))
        access = jwt.encode(
            {
                "iss": ISSUER, "aud": RESOURCE, "sub": subject, "client_id": client_id,
                "scope": scopes, "iat": now, "nbf": now, "exp": expires,
                "jti": access_id,
                "https://api.openai.com/auth": {
                    "per_user_salt": "integration-salt",
                    "encrypted_auth_metadata": "integration-metadata",
                },
            },
            self._key,
            algorithm="RS256",
            headers={"kid": "securo-integration"},
        )
        identity: dict[str, Any] = {
            "iss": ISSUER, "aud": client_id, "sub": subject,
            "iat": now, "exp": expires, "at_hash": _at_hash(access),
            "email": f"account-{subject.removeprefix('subject-')}@example.test",
        }
        if nonce is not None:
            identity["nonce"] = nonce
        identity.update(self._next_identity_claim_overrides)
        self._next_identity_claim_overrides.clear()
        id_token = jwt.encode(
            identity, self._key, algorithm="RS256", headers={"kid": "securo-integration"}
        )
        self._id_tokens.add(id_token)
        refresh = self._next("refresh")
        self._refresh[refresh] = _Refresh(client_id, subject, lineage)
        self._access[access] = (client_id, lineage, expires)
        return httpx.Response(200, json={
            "access_token": access,
            "refresh_token": refresh,
            "id_token": id_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": scopes,
            "earliest_refresh_at": now,
            "refresh_token_expires_at": now + 86400,
        }, request=request)

    def _bearer(self, request: httpx.Request) -> tuple[str, str] | None:
        header = request.headers.get("authorization", "")
        token = header.removeprefix("Bearer ") if header.startswith("Bearer ") else ""
        record = self._access.get(token)
        if record is None:
            return None
        client_id, lineage, expires = record
        if lineage in self._revoked_lineages or expires <= int(time.time()):
            return None
        return client_id, lineage

    def _models(self, request: httpx.Request) -> httpx.Response:
        if request.content:
            return httpx.Response(400, json={"error": "invalid_request"}, request=request)
        auth = self._bearer(request)
        self._journal(
            request,
            "models",
            client_id_sha256=_digest(auth[0]) if auth else None,
        )
        if auth is None:
            return httpx.Response(401, json={"error": "invalid_token"}, request=request)
        return httpx.Response(200, json={"models": self.models}, request=request)

    def _revoke(self, request: httpx.Request) -> httpx.Response:
        form = self._form(request)
        if (
            form is None
            or set(form) != {"token", "token_type_hint", "client_id"}
            or not all(form.values())
            or form["token_type_hint"] != "refresh_token"
        ):
            return httpx.Response(400, json={"error": "invalid_request"}, request=request)
        record = self._refresh.get(form.get("token", ""))
        client_id = form.get("client_id", "")
        self._journal(
            request,
            "revoke",
            client_id_sha256=_digest(client_id) if client_id else None,
        )
        if record is None or record.client_id != form["client_id"]:
            return httpx.Response(400, json={"error": "invalid_request"}, request=request)
        if record is not None:
            self._revoked_lineages.add(record.lineage)
            for refresh in self._refresh.values():
                if refresh.lineage == record.lineage:
                    refresh.revoked = True
        return httpx.Response(200, content=b"", request=request)

    @staticmethod
    def _wire(events: list[dict[str, Any]]) -> bytes:
        for sequence, event in enumerate(events):
            event["sequence_number"] = sequence
        return b"".join(
            b"data: " + json.dumps(event, separators=(",", ":")).encode() + b"\n\n"
            for event in events
        )

    def _responses(self, request: httpx.Request) -> httpx.Response:
        if request.headers.get("content-type") != "application/json":
            return httpx.Response(415, json={"error": {"code": "invalid_media_type"}}, request=request)
        if len(request.content) > _MAX_RESPONSES_BODY_BYTES:
            return httpx.Response(413, json={"error": {"code": "request_too_large"}}, request=request)
        auth = self._bearer(request)
        self.responses_calls += 1
        try:
            payload = _bounded_json(request.content)
        except OverflowError:
            return httpx.Response(
                413, json={"error": {"code": "request_too_large"}}, request=request
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError):
            payload = {}
        self._journal(
            request,
            "responses",
            client_id_sha256=_digest(auth[0]) if auth else None,
            model=payload.get("model") if isinstance(payload, dict) else None,
        )
        if (
            not isinstance(payload, dict)
            or not set(payload).issubset(_RESPONSES_FIELDS)
            or payload.get("store") is not False
            or payload.get("stream") is not True
            or not isinstance(payload.get("model"), str)
            or not payload.get("model")
            or not isinstance(payload.get("input"), list)
            or len(payload["input"]) > 1024
            or not all(_valid_response_input_item(item) for item in payload["input"])
            or (
                "instructions" in payload
                and not isinstance(payload["instructions"], str)
            )
        ):
            return httpx.Response(400, json={"error": {"code": "invalid_request"}}, request=request)
        if auth is None:
            return httpx.Response(401, json={"error": {"code": "invalid_token"}}, request=request)
        client_id, lineage = auth
        inputs = payload.get("input", []) if isinstance(payload, dict) else []
        tools = payload.get("tools")
        expected_tools = [{
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
        }]
        if tools != expected_tools:
            return httpx.Response(
                400, json={"error": {"code": "invalid_tools"}}, request=request
            )
        assert isinstance(tools, list)
        function_items = [
            (index, item)
            for index, item in enumerate(inputs)
            if isinstance(item, dict)
            and item.get("type") in {"function_call", "function_call_output"}
        ]
        response_id = self._next("response")
        if not function_items:
            tool = tools[0]["tools"][0]
            item = {
                "type": "function_call", "namespace": "securo", "id": self._next("item"),
                "call_id": self._next("call"), "name": tool["name"],
                "arguments": "{\"query\":\"integration\"}", "status": "completed",
            }
            self._response_call_order += 1
            self._response_calls[item["call_id"]] = _ResponseCall(
                order=self._response_call_order,
                response_id=response_id,
                client_id=client_id,
                lineage=lineage,
                call_id=item["call_id"],
                item_id=item["id"],
                name=item["name"],
                arguments=item["arguments"],
                expected_output='{\"value\": 42}',
            )
            added = {**item, "arguments": "", "status": "in_progress"}
            events = [
                {"type": "response.created", "response": {"id": response_id}},
                {"type": "response.output_item.added", "response_id": response_id, "output_index": 0, "item": added},
                {"type": "response.function_call_arguments.delta", "response_id": response_id, "output_index": 0, "item_id": item["id"], "delta": item["arguments"]},
                {"type": "response.function_call_arguments.done", "response_id": response_id, "output_index": 0, "item_id": item["id"], "arguments": item["arguments"]},
                {"type": "response.output_item.done", "response_id": response_id, "output_index": 0, "item": item},
                {"type": "response.completed", "response": {"id": response_id, "status": "completed", "output": [item], "usage": {"input_tokens": 7, "output_tokens": 3}}},
            ]
        else:
            valid = len(function_items) % 2 == 0
            issued_calls: list[_ResponseCall] = []
            for offset in range(0, len(function_items), 2):
                if not valid:
                    break
                (call_index, function_call), (output_index, output) = function_items[
                    offset:offset + 2
                ]
                call_id = function_call.get("call_id")
                issued = self._response_calls.get(call_id) if isinstance(call_id, str) else None
                if not (
                    output_index == call_index + 1
                    and issued is not None
                    and not issued.used
                    and issued.client_id == client_id
                    and issued.lineage == lineage
                    and function_call == {
                        "type": "function_call",
                        "namespace": "securo",
                        "call_id": issued.call_id,
                        "name": issued.name,
                        "arguments": issued.arguments,
                    }
                    and output == {
                        "type": "function_call_output",
                        "call_id": issued.call_id,
                        "output": issued.expected_output,
                    }
                ):
                    valid = False
                    break
                issued_calls.append(issued)
            if valid and (
                not issued_calls
                or len({issued.call_id for issued in issued_calls}) != len(issued_calls)
                or [issued.order for issued in issued_calls]
                != sorted(issued.order for issued in issued_calls)
            ):
                valid = False
            if not valid:
                return httpx.Response(
                    400,
                    json={"error": {"code": "invalid_tool_continuation"}},
                    request=request,
                )
            for issued in issued_calls:
                issued.used = True
            text = "assembled integration complete"
            item = {
                "id": self._next("message"), "type": "message", "role": "assistant",
                "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
            events = [
                {"type": "response.created", "response": {"id": response_id}},
                {"type": "response.output_text.delta", "response_id": response_id, "output_index": 0, "content_index": 0, "item_id": item["id"], "delta": text},
                {"type": "response.output_text.done", "response_id": response_id, "output_index": 0, "content_index": 0, "item_id": item["id"], "text": text},
                {"type": "response.completed", "response": {"id": response_id, "status": "completed", "output": [item], "usage": {"input_tokens": 9, "output_tokens": 4}}},
            ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=self._wire(events),
            request=request,
        )

    def sensitive_values_for_test(self) -> set[str]:
        """Return live sentinels so tests can prove observability remains opaque."""
        return {
            *self._codes.keys(),
            *self._refresh.keys(),
            *self._access.keys(),
            *self._id_tokens,
            *self._subjects,
            *self._client_ids,
            *self._nonces,
            *(f"account-{subject.removeprefix('subject-')}@example.test" for subject in self._subjects),
        }
