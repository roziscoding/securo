from __future__ import annotations

import base64
import hashlib
import inspect
import json
import queue
import socket
import threading
import types
import time
import uuid
from unittest.mock import patch
from urllib.parse import urlencode
from urllib.request import urlopen

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from jose import jwt

from securo_siwc.backend import MAX_REQUEST, PairingClient, _strict_json
from securo_siwc.cli import build_parser
from securo_siwc import loopback as loopback_module
from securo_siwc.loopback import CallbackFailure, CallbackResult, LoopbackListener
from securo_siwc.openai import (
    ISSUER,
    JWKS_ENDPOINT,
    RESOURCE,
    SCOPES,
    TOKEN_ENDPOINT,
    OpenAIClient,
    _validate_access_token,
    _json as _openai_json,
    create_authorization,
)
from securo_siwc.protocol import validate_server_origin
from securo_siwc.security import secret_boundary


class _StreamingMockTransport(httpx.BaseTransport):
    def __init__(self, transport: httpx.BaseTransport):
        self._transport = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self._transport.handle_request(request)
        if not response.is_stream_consumed:
            return response
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            stream=httpx.ByteStream(response.content),
            request=request,
            extensions=dict(response.extensions),
        )

    def close(self) -> None:
        self._transport.close()


def _pairing_client(*args, transport, **kwargs) -> PairingClient:
    transport = _StreamingMockTransport(transport)
    with patch(
        "securo_siwc.backend._new_http_client",
        lambda: httpx.Client(transport=transport, trust_env=False, follow_redirects=False),
    ):
        return PairingClient(*args, **kwargs)


def _openai_client(transport) -> OpenAIClient:
    transport = _StreamingMockTransport(transport)
    with patch(
        "securo_siwc.openai._new_http_client",
        lambda: httpx.Client(transport=transport, trust_env=False, follow_redirects=False),
    ):
        return OpenAIClient()


def _exception_graph_contains(error: BaseException, needle: str) -> bool:
    pending: list[object] = [error]
    seen: set[int] = set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, (str, bytes)):
            if needle in (value.decode(errors="ignore") if isinstance(value, bytes) else value):
                return True
            continue
        if isinstance(value, BaseException):
            pending.extend(value.args)
            pending.extend((value.__cause__, value.__context__, value.__traceback__))
            pending.extend(value.__dict__.values())
        elif isinstance(value, types.TracebackType):
            pending.extend((value.tb_next, value.tb_frame))
        elif isinstance(value, types.FrameType):
            if not value.f_code.co_filename.endswith("test_helper.py"):
                pending.extend(value.f_locals.values())
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            pending.extend(value)
        elif isinstance(value, httpx.Request):
            pending.extend((value.headers, value.content, value.url))
        elif isinstance(value, httpx.Response):
            pending.extend((value.headers, value.content))
            try:
                pending.append(value.request)
            except RuntimeError:
                pass
        elif hasattr(value, "__dict__"):
            pending.extend(vars(value).values())
    return False


def test_server_origin_requires_https_or_literal_test_loopback():
    assert validate_server_origin("https://securo.example", allow_http_loopback=False) == "https://securo.example"
    assert validate_server_origin("http://127.0.0.1:8000", allow_http_loopback=True) == "http://127.0.0.1:8000"
    for value in ("http://example.com", "https://example.com/path", "https://user@example.com"):
        with pytest.raises(ValueError):
            validate_server_origin(value, allow_http_loopback=False)


def test_unarmed_listener_rejects_without_consuming_then_arms_once():
    outcomes: queue.Queue[CallbackResult | CallbackFailure] = queue.Queue(maxsize=1)
    listener = LoopbackListener(outcomes)
    listener.start()
    try:
        base = f"http://127.0.0.1:{listener.port}/auth/callback?"
        with pytest.raises(Exception):
            urlopen(base + urlencode({"state": "expected", "code": "poison"}), timeout=2)
        assert outcomes.empty()

        listener.arm("expected", None)
        with pytest.raises(RuntimeError, match="already armed"):
            listener.arm("replacement", None)
        response = urlopen(base + urlencode({"state": "expected", "error": "access_denied"}), timeout=2)
        assert response.status == 200
        result = outcomes.get(timeout=1)
        assert isinstance(result, CallbackFailure)
        assert result.code == "access_denied"
    finally:
        listener.close()


def test_wrong_state_does_not_poison_listener():
    outcomes: queue.Queue[CallbackResult | CallbackFailure] = queue.Queue(maxsize=1)
    listener = LoopbackListener(outcomes)
    listener.start()
    listener.arm("expected", None)
    try:
        base = f"http://127.0.0.1:{listener.port}/auth/callback?"
        with pytest.raises(Exception):
            urlopen(base + urlencode({"state": "wrong", "code": "poison"}), timeout=2)
        assert outcomes.empty()
    finally:
        listener.close()


def test_concurrent_local_attackers_cannot_poison_listener_arming():
    outcomes: queue.Queue[CallbackResult | CallbackFailure] = queue.Queue(maxsize=1)
    listener = LoopbackListener(outcomes)
    listener.start()
    release = threading.Event()
    base = f"http://127.0.0.1:{listener.port}/auth/callback?"

    def attack(index: int) -> None:
        release.wait()
        try:
            urlopen(
                base + urlencode({"state": f"attacker-{index}", "code": "poison"}),
                timeout=2,
            )
        except Exception:
            pass

    attackers = [threading.Thread(target=attack, args=(index,)) for index in range(32)]
    for attacker in attackers:
        attacker.start()
    release.set()
    listener.arm("expected", "issued")
    for attacker in attackers:
        attacker.join(timeout=3)
    try:
        assert outcomes.empty()
        assert urlopen(
            base + urlencode({"state": "expected", "code": "right"}), timeout=2
        ).status == 200
        assert outcomes.get(timeout=1) == CallbackResult("right", "issued")
    finally:
        listener.close()


def test_duplicate_callback_first_terminal_result_wins():
    outcomes: queue.Queue[CallbackResult | CallbackFailure] = queue.Queue(maxsize=1)
    listener = LoopbackListener(outcomes)
    listener.start()
    listener.arm("expected", "issued-client")
    try:
        url = f"http://127.0.0.1:{listener.port}/auth/callback?" + urlencode(
            {"state": "expected", "code": "first"}
        )
        assert urlopen(url, timeout=2).status == 200
        with pytest.raises(Exception):
            urlopen(url.replace("first", "second"), timeout=2)
        result = outcomes.get(timeout=1)
        assert isinstance(result, CallbackResult)
        assert result.code == "first"
    finally:
        listener.close()


def test_callback_state_remains_consumed_after_result_is_dequeued():
    outcomes: queue.Queue[CallbackResult | CallbackFailure] = queue.Queue(maxsize=1)
    listener = LoopbackListener(outcomes)
    listener.start()
    listener.arm("expected", "issued-client")
    try:
        base = f"http://127.0.0.1:{listener.port}/auth/callback?"
        first = base + urlencode({"state": "expected", "code": "first"})
        assert urlopen(first, timeout=2).status == 200
        assert outcomes.get(timeout=1) == CallbackResult("first", "issued-client")
        with pytest.raises(Exception):
            urlopen(base + urlencode({"state": "expected", "code": "replay"}), timeout=2)
        assert outcomes.empty()
    finally:
        listener.close()


def test_concurrent_exact_state_callbacks_publish_exactly_once():
    outcomes: queue.Queue[CallbackResult | CallbackFailure] = queue.Queue(maxsize=8)
    listener = LoopbackListener(outcomes)
    listener.start()
    listener.arm("expected", "issued-client")
    release = threading.Event()
    statuses: list[int] = []

    def callback(index: int) -> None:
        release.wait()
        try:
            response = urlopen(
                f"http://127.0.0.1:{listener.port}/auth/callback?"
                + urlencode({"state": "expected", "code": f"code-{index}"}),
                timeout=3,
            )
            statuses.append(response.status)
        except Exception:
            statuses.append(409)

    workers = [threading.Thread(target=callback, args=(index,)) for index in range(20)]
    try:
        for worker in workers:
            worker.start()
        release.set()
        for worker in workers:
            worker.join(timeout=4)
        assert statuses.count(200) == 1
        assert outcomes.qsize() == 1
    finally:
        listener.close()


def test_slowloris_connections_are_bounded_and_close_without_survivors(capsys):
    outcomes: queue.Queue[CallbackResult | CallbackFailure] = queue.Queue(maxsize=1)
    listener = LoopbackListener(outcomes)
    listener.start()
    listener.arm("expected", "issued")
    peers: list[socket.socket] = []
    try:
        for _ in range(loopback_module.MAX_CONCURRENT_HANDLERS + 6):
            peer = socket.create_connection(("127.0.0.1", listener.port), timeout=1)
            peer.sendall(b"GET /auth/callback?state=expected HTTP/1.1\r\nHost: 127.0.0.1\r\n")
            peers.append(peer)
        time.sleep(0.2)
        handler_threads = listener._handler_threads_for_test()
        assert len(handler_threads) <= loopback_module.MAX_CONCURRENT_HANDLERS

        listener.close()

        assert not listener._handler_threads_for_test()
        assert all(not thread.is_alive() for thread in handler_threads)
        assert capsys.readouterr().err == ""
    finally:
        listener.close()
        for peer in peers:
            peer.close()


def test_listener_normal_callback_survives_disconnect_race_and_repeated_close(capsys):
    outcomes: queue.Queue[CallbackResult | CallbackFailure] = queue.Queue(maxsize=1)
    listener = LoopbackListener(outcomes)
    listener.start()
    listener.arm("expected", "issued")
    disconnected = socket.create_connection(("127.0.0.1", listener.port), timeout=1)
    disconnected.sendall(
        b"GET /auth/callback?state=wrong&code=gone HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"
    )
    disconnected.close()
    try:
        response = urlopen(
            f"http://127.0.0.1:{listener.port}/auth/callback?"
            + urlencode({"state": "expected", "code": "right"}),
            timeout=2,
        )
        assert response.status == 200
        assert outcomes.get(timeout=1) == CallbackResult("right", "issued")
    finally:
        listener.close()
        listener.close()
    assert not listener._handler_threads_for_test()
    assert capsys.readouterr().err == ""


def test_public_client_signatures_do_not_accept_transport_or_client():
    for constructor in (PairingClient, OpenAIClient):
        parameters = inspect.signature(constructor).parameters
        assert parameters.keys().isdisjoint({"transport", "client", "http_client"})


def test_pairing_secret_only_appears_in_exact_authorization_header():
    secret = "s" * 43
    seen: list[tuple[list[str], str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(
            (
                request.headers.get_list("authorization"),
                str(request.url),
                request.content,
            )
        )
        return httpx.Response(200, json={"state": "pending"})

    pairing_id = uuid.uuid4()
    with _pairing_client(
        "http://127.0.0.1:8000", pairing_id, secret,
        allow_http_loopback=True, transport=httpx.MockTransport(handler),
    ) as client:
        client.bootstrap()
    assert len(seen) == 1
    authorization, url, content = seen[0]
    assert authorization == ["SecuroPairing " + secret]
    assert secret not in url
    assert content == b""


def test_cli_has_no_secret_argument_or_environment_fallback(monkeypatch):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            str(uuid.uuid4()), "--server", "https://securo.example", "--secret", "unsafe",
        ])
    monkeypatch.setenv("SECURO_PAIRING_SECRET", "unsafe")
    args = parser.parse_args([str(uuid.uuid4()), "--server", "https://securo.example"])
    assert vars(args).keys().isdisjoint({"secret", "pairing_secret"})


def _b64_int(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def test_end_to_end_transfer_with_fake_openai_and_backend():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    signing_key = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public = key.public_key().public_numbers()
    jwk = {"kty": "RSA", "kid": "test", "alg": "RS256", "use": "sig",
           "n": _b64_int(public.n), "e": _b64_int(public.e)}
    auth = create_authorization(43111, "urn:uuid:11111111-1111-4111-8111-111111111111")
    client_id = "issued-client"
    now = int(time.time())
    access_claims = {
        "iss": ISSUER, "aud": RESOURCE, "sub": "subject", "client_id": client_id,
        "scope": SCOPES, "iat": now, "nbf": now, "exp": now + 3600, "jti": "unique",
        "https://api.openai.com/auth": {
            "per_user_salt": "salt", "encrypted_auth_metadata": "metadata",
        },
    }
    access = jwt.encode(access_claims, signing_key, algorithm="RS256", headers={"kid": "test"})
    at_hash = base64.urlsafe_b64encode(hashlib.sha256(access.encode()).digest()[:16]).rstrip(b"=").decode()
    identity_claims = {
        "iss": ISSUER, "aud": client_id, "sub": "subject", "nonce": auth.nonce,
        "iat": now, "exp": now + 3600, "at_hash": at_hash,
    }
    identity = jwt.encode(identity_claims, signing_key, algorithm="RS256", headers={"kid": "test"})
    token_response = {
        "access_token": access, "refresh_token": "refresh-token", "id_token": identity,
        "token_type": "Bearer", "expires_in": 3600, "scope": SCOPES,
        "earliest_refresh_at": now + 3000,
    }

    def openai_handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == TOKEN_ENDPOINT:
            return httpx.Response(200, json=token_response)
        assert str(request.url) == JWKS_ENDPOINT
        return httpx.Response(200, json={"keys": [jwk]})

    oauth = _openai_client(httpx.MockTransport(openai_handler))
    try:
        transferred = oauth.exchange(auth, "authorization-code", client_id)
    finally:
        oauth.close()

    backend_requests: list[httpx.Request] = []

    def backend_handler(request: httpx.Request) -> httpx.Response:
        backend_requests.append(request)
        if request.url.path.endswith("/attempt"):
            return httpx.Response(200, json=json.loads(request.content))
        if request.url.path.endswith("/complete"):
            return httpx.Response(200, json={"state": "completed", "subject": "subject"})
        return httpx.Response(200, json={"state": "pending"})

    attempt = uuid.uuid4()
    with _pairing_client(
        "http://127.0.0.1:8000", uuid.uuid4(), "s" * 43,
        allow_http_loopback=True, transport=httpx.MockTransport(backend_handler),
    ) as backend:
        backend.bootstrap()
        backend.register_attempt(attempt, auth.nonce, auth.redirect_uri)
        result = backend.complete(attempt, client_id, transferred)
    assert result["state"] == "completed"
    assert [request.url.path.rsplit("/", 1)[-1] for request in backend_requests] == [
        "bootstrap", "attempt", "complete",
    ]
    assert all("access_token" not in str(request.url) for request in backend_requests)


def test_openai_fetches_and_validates_jwks_before_consuming_code():
    order: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        order.append(str(request.url))
        if str(request.url) == JWKS_ENDPOINT:
            return httpx.Response(200, json={"keys": []})
        pytest.fail("one-use authorization code was consumed before JWKS validation")

    auth = create_authorization(43111, "urn:uuid:11111111-1111-4111-8111-111111111111")
    oauth = _openai_client(httpx.MockTransport(handler))
    try:
        with pytest.raises((ValueError, RuntimeError)):
            oauth.exchange(auth, "one-use-code", "issued-client")
    finally:
        oauth.close()
    assert order == [JWKS_ENDPOINT]


@pytest.mark.parametrize(
    "keys",
    [
        [{"kid": "not-a-key"}],
        [{"kty": "RSA", "kid": "duplicate", "n": _b64_int(2**2047 + 1), "e": _b64_int(65537)},
         {"kty": "RSA", "kid": "duplicate", "n": _b64_int(2**2047 + 3), "e": _b64_int(65537)}],
        [{"kty": "RSA", "kid": "weak", "n": _b64_int(2**1023 + 1), "e": _b64_int(65537)}],
        [{"kty": "RSA", "kid": "exponent", "n": _b64_int(2**2047 + 1), "e": _b64_int(2)}],
    ],
)
def test_unusable_jwks_never_consumes_authorization_code(keys):
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if str(request.url) == JWKS_ENDPOINT:
            return httpx.Response(200, json={"keys": keys})
        pytest.fail("unusable JWKS consumed the authorization code")

    oauth = _openai_client(httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeError):
            oauth.exchange(
                create_authorization(43111, "urn:uuid:11111111-1111-4111-8111-111111111111"),
                "one-use-code",
                "issued-client",
            )
    finally:
        oauth.close()
    assert calls == [JWKS_ENDPOINT]


def test_three_segment_opaque_access_token_is_not_misclassified_as_jwt():
    _validate_access_token(
        "opaque.part.token",
        jwks={"keys": []},
        identity={"at_hash": "bound"},
        client_id="issued-client",
        scopes=frozenset(SCOPES.split()),
    )


@pytest.mark.parametrize(
    "token",
    [
        base64.urlsafe_b64encode(b'{"alg":"HS256"}').rstrip(b"=").decode() + ".e30.signature",
        base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode() + ".not+base64.signature",
    ],
)
def test_recognizable_but_unsupported_or_malformed_jose_fails_closed(token):
    with pytest.raises(ValueError):
        _validate_access_token(
            token,
            jwks={"keys": []},
            identity={"at_hash": "bound"},
            client_id="issued-client",
            scopes=frozenset(SCOPES.split()),
        )


def test_pairing_response_stream_stops_after_limit_plus_one(monkeypatch):
    from securo_siwc import backend as backend_module

    consumed = 0

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            nonlocal consumed
            while True:
                consumed += 1
                yield b"x"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=Stream())

    monkeypatch.setattr(backend_module, "MAX_RESPONSE", 32)
    with _pairing_client(
        "http://127.0.0.1:8000", uuid.uuid4(), "s" * 43,
        allow_http_loopback=True, transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(RuntimeError, match="response too large"):
            client.bootstrap()
    assert consumed == 33


def test_pairing_request_encoding_stops_before_traversing_huge_list():
    visited = 0
    requests = 0

    class CountingList(list):
        def __iter__(self):
            nonlocal visited
            for item in super().__iter__():
                visited += 1
                yield item

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={})

    payload = {"values": CountingList(["x" * 1024] * (MAX_REQUEST * 4))}
    with _pairing_client(
        "http://127.0.0.1:8000", uuid.uuid4(), "s" * 43,
        allow_http_loopback=True, transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(RuntimeError, match="request too large"):
            client._request("POST", "attempt", payload)
    assert visited <= MAX_REQUEST // 1024 + 2
    assert requests == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"value": float("nan")},
        {"value": object()},
        {"value": "x" * 65_537},
        {"value": (item for item in range(1))},
    ],
)
def test_pairing_request_rejects_non_json_or_oversize_values_without_request(payload):
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={})

    with _pairing_client(
        "http://127.0.0.1:8000", uuid.uuid4(), "s" * 43,
        allow_http_loopback=True, transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(RuntimeError):
            client._request("POST", "attempt", payload)
    assert requests == 0


def test_pairing_request_rejects_cycles_depth_nodes_members_and_custom_mappings():
    cycle: list[object] = []
    cycle.append(cycle)
    deep: object = 0
    for _ in range(17):
        deep = [deep]

    class CustomMapping(dict):
        pass

    rejected = [
        {"value": cycle},
        {"value": deep},
        {"value": [None] * 4096},
        {f"key-{index}": index for index in range(129)},
        {"value": 1 << 1024},
        {"value": CustomMapping(value="secret")},
    ]
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(200, json={})

    with _pairing_client(
        "http://127.0.0.1:8000", uuid.uuid4(), "s" * 43,
        allow_http_loopback=True, transport=httpx.MockTransport(handler),
    ) as client:
        for payload in rejected:
            with pytest.raises(RuntimeError, match="request too large"):
                client._request("POST", "attempt", payload)
    assert requests == 0


def test_authorization_uses_precommitted_state():
    from urllib.parse import parse_qs, urlsplit

    state = "precommitted-state"
    authorization = create_authorization(
        43111,
        "urn:uuid:11111111-1111-4111-8111-111111111111",
        state=state,
    )
    assert authorization.state == state
    assert parse_qs(urlsplit(authorization.url).query)["state"] == [state]


def test_authorization_host_identity_is_backend_owned_and_stable():
    host = "urn:uuid:11111111-1111-4111-8111-111111111111"
    first = create_authorization(43111, host)
    second = create_authorization(43112, host)
    from urllib.parse import parse_qs, urlsplit
    assert parse_qs(urlsplit(first.url).query)["ext_agent_host_id"] == [host]
    assert parse_qs(urlsplit(second.url).query)["ext_agent_host_id"] == [host]


def test_returning_authorization_reuses_client_without_dynamic_hint():
    from urllib.parse import parse_qs, urlsplit

    host = "urn:uuid:11111111-1111-4111-8111-111111111111"
    dynamic = parse_qs(urlsplit(create_authorization(43111, host).url).query)
    returning = parse_qs(
        urlsplit(create_authorization(43111, host, "retained-client").url).query
    )

    assert dynamic["client_id"] == ["dynamic_agent_client"]
    assert dynamic["agent_name_hint"] == ["Securo"]
    assert returning["client_id"] == ["retained-client"]
    assert "agent_name_hint" not in returning


def test_secret_boundary_never_stringifies_or_inspects_hostile_exception_values():
    class Hostile:
        def __str__(self):
            raise AssertionError("stringified hostile value")

        def __getattribute__(self, name):
            if name == "__dict__":
                raise AssertionError("inspected hostile value")
            return object.__getattribute__(self, name)

    class HostileError(RuntimeError):
        def __str__(self):
            raise AssertionError("stringified hostile exception")

        def __getattribute__(self, name):
            if name in {"args", "__dict__", "__traceback__", "__cause__", "__context__"}:
                raise AssertionError("inspected hostile exception")
            return RuntimeError.__getattribute__(self, name)

        def __setattr__(self, name, value):
            if name in {"args", "__traceback__", "__cause__", "__context__"}:
                raise AssertionError("mutated hostile exception")
            return RuntimeError.__setattr__(self, name, value)

    @secret_boundary("safe failure")
    def fail():
        raise HostileError({"nested": Hostile()})

    with pytest.raises(RuntimeError, match="^safe failure$") as caught:
        fail()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_helper_boundary_scrubs_3000_deep_and_very_wide_graphs_iteratively():
    sentinel = "bounded-helper-scrub-sentinel"
    deep: object = sentinel
    for _ in range(3000):
        deep = [deep]
    wide: list[object] = [bytearray(sentinel.encode()) for _ in range(10_000)]
    retained = RuntimeError({"deep": deep, "wide": wide})

    @secret_boundary("safe failure")
    def fail():
        raise retained

    with pytest.raises(RuntimeError, match="^safe failure$"):
        fail()
    assert retained.args == ()
    assert retained.__dict__ == {}
    assert deep == []
    assert wide == []


def test_helper_boundary_scrubs_externally_retained_hostile_exception_cycle():
    sentinel = "retained-helper-bearer-sentinel"

    class Hostile(RuntimeError):
        __slots__ = ("retained",)

        def __getattribute__(self, name):
            if name in {"__cause__", "__context__", "__traceback__", "args", "__dict__"}:
                raise RuntimeError("hostile-helper-accessor-sentinel")
            return RuntimeError.__getattribute__(self, name)

        def __setattr__(self, name, value):
            if name in {"__cause__", "__context__", "__traceback__", "args", "__dict__"}:
                raise RuntimeError("hostile-helper-accessor-sentinel")
            return RuntimeError.__setattr__(self, name, value)

    retained = Hostile(sentinel)
    retained.retained = {"bearer": sentinel}
    nested = RuntimeError({"nested": sentinel})
    BaseException.__dict__["__cause__"].__set__(nested, retained)
    BaseException.__dict__["__context__"].__set__(retained, nested)

    @secret_boundary("safe failure")
    def fail():
        raise retained

    with pytest.raises(RuntimeError, match="^safe failure$"):
        fail()
    assert BaseException.__dict__["args"].__get__(retained, Hostile) == ()
    assert retained.retained is None


def test_pairing_network_failure_detaches_authorization_secret():
    secret = "pairing-secret-sentinel-" + "s" * 43

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("failed", request=request)

    client = _pairing_client(
        "http://127.0.0.1:8000", uuid.uuid4(), secret,
        allow_http_loopback=True, transport=httpx.MockTransport(fail),
    )
    try:
        with pytest.raises(RuntimeError) as caught:
            client.bootstrap()
    finally:
        client.close()
    assert not _exception_graph_contains(caught.value, secret)


def test_openai_exchange_failure_detaches_code_and_verifier():
    code = "authorization-code-sentinel"
    auth = create_authorization(
        43111, "urn:uuid:11111111-1111-4111-8111-111111111111"
    )
    verifier = auth.verifier

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == JWKS_ENDPOINT:
            return httpx.Response(200, json={"keys": [{"kid": "structural"}]})
        raise httpx.ConnectError("failed", request=request)

    client = _openai_client(httpx.MockTransport(handler))
    try:
        with pytest.raises(RuntimeError) as caught:
            client.exchange(auth, code, "issued-client")
    finally:
        client.close()
    assert not _exception_graph_contains(caught.value, code)
    assert not _exception_graph_contains(caught.value, verifier)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":1e9999}',
        (b'[' * 18) + b'0' + (b']' * 18),
        b'{"extra":"' + (b'x' * 65_537) + b'"}',
    ],
)
def test_helper_json_parser_rejects_ambiguous_or_excessive_graphs(raw):
    with pytest.raises(ValueError):
        _strict_json(raw)
    with pytest.raises(ValueError):
        _openai_json(raw)


@pytest.mark.parametrize(
    ("headers", "message"),
    [
        ({"Content-Length": "2000000"}, "response too large"),
        ({"Content-Encoding": "gzip"}, "compressed response rejected"),
    ],
)
def test_pairing_rejects_declared_oversize_or_compression_before_body(headers, message):
    consumed = 0

    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            nonlocal consumed
            consumed += 1
            yield b"{}"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers=headers, stream=Stream())

    with _pairing_client(
        "http://127.0.0.1:8000", uuid.uuid4(), "s" * 43,
        allow_http_loopback=True, transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(RuntimeError, match=message):
            client.bootstrap()
    assert consumed == 0


def test_completion_ambiguous_failure_retries_exact_identical_post():
    requests: list[tuple[bytes, list[str]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/complete")
        requests.append((request.content, request.headers.get_list("authorization")))
        if len(requests) == 1:
            raise httpx.ReadError("ambiguous", request=request)
        return httpx.Response(200, json={"state": "completed", "subject": "subject"})

    with _pairing_client(
        "http://127.0.0.1:8000", uuid.uuid4(), "s" * 43,
        allow_http_loopback=True, transport=httpx.MockTransport(handler),
    ) as client:
        result = client.complete(uuid.uuid4(), "issued", {"access_token": "token"})
    assert result == {"state": "completed", "subject": "subject"}
    assert len(requests) == 2
    assert requests[0] == requests[1]


def test_completion_never_infers_success_when_altered_completion_wins():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path.endswith("/complete")
        if calls == 1:
            raise httpx.ReadError("ambiguous", request=request)
        return httpx.Response(409, json={"detail": "pairing_not_available"})

    with _pairing_client(
        "http://127.0.0.1:8000", uuid.uuid4(), "s" * 43,
        allow_http_loopback=True, transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(RuntimeError, match="completion could not be confirmed"):
            client.complete(uuid.uuid4(), "issued", {"access_token": "token"})
    assert calls == 3


def test_completion_retries_are_bounded_when_request_never_arrives(monkeypatch):
    calls = 0
    monkeypatch.setattr("securo_siwc.backend.time.sleep", lambda _delay: None)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("not reached", request=request)

    with _pairing_client(
        "http://127.0.0.1:8000", uuid.uuid4(), "s" * 43,
        allow_http_loopback=True, transport=httpx.MockTransport(handler),
    ) as client:
        with pytest.raises(RuntimeError, match="completion could not be confirmed"):
            client.complete(uuid.uuid4(), "issued", {"access_token": "token"})
    assert calls == 3


def test_listener_rejects_unknown_query_without_poisoning():
    outcomes: queue.Queue[CallbackResult | CallbackFailure] = queue.Queue(maxsize=1)
    listener = LoopbackListener(outcomes)
    listener.start()
    listener.arm("expected", "issued")
    try:
        base = f"http://127.0.0.1:{listener.port}/auth/callback?"
        with pytest.raises(Exception):
            urlopen(base + urlencode({"state": "expected", "code": "bad", "extra": "x"}), timeout=2)
        assert outcomes.empty()
        assert urlopen(
            base + urlencode({"state": "expected", "code": "good"}), timeout=2
        ).status == 200
        assert outcomes.get(timeout=1) == CallbackResult("good", "issued")
    finally:
        listener.close()


def test_pairing_boundary_scrubs_before_propagating_keyboard_interrupt():
    secret = "interrupt-secret-" + "s" * 43

    def interrupt(request: httpx.Request) -> httpx.Response:
        raise KeyboardInterrupt(secret)

    client = _pairing_client(
        "http://127.0.0.1:8000", uuid.uuid4(), secret,
        allow_http_loopback=True, transport=httpx.MockTransport(interrupt),
    )
    try:
        with pytest.raises(KeyboardInterrupt) as caught:
            client.bootstrap()
    finally:
        client.close()
    assert caught.value.args == ()
    assert not _exception_graph_contains(caught.value, secret)
