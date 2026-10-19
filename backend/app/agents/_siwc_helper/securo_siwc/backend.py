"""Secret-safe Securo pairing API client."""
from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any, cast

import httpx

from .protocol import validate_server_origin
from .security import _scrub_response, secret_boundary

MAX_RESPONSE = 1_000_000
MAX_REQUEST = 256 * 1024
MAX_JSON_NODES = 4096
MAX_JSON_DEPTH = 16
MAX_JSON_MEMBERS = 128
MAX_JSON_STRING = 65_536


def _new_http_client() -> httpx.Client:
    """Build the production-owned transport boundary."""
    return httpx.Client(
        timeout=httpx.Timeout(30, connect=5),
        follow_redirects=False,
        trust_env=False,
        headers={"Accept": "application/json"},
        cookies=None,
        auth=None,
        event_hooks={},
    )


def _bounded_request_json(payload: dict[str, Any]) -> bytes:
    """Encode strict JSON without retaining or traversing an oversized graph."""
    encoder = json.JSONEncoder(separators=(",", ":"), allow_nan=False)
    pending: list[tuple[object, int, bool]] = [(payload, 0, False)]
    active: set[int] = set()
    nodes = 0
    string_bytes = 0
    try:
        while pending:
            item, depth, leaving = pending.pop()
            if leaving:
                active.discard(id(item))
                continue
            nodes += 1
            if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
                raise RuntimeError("Securo request too large")
            if item is None or type(item) is bool:
                continue
            if type(item) is int:
                if item.bit_length() > 1024:
                    raise RuntimeError("Securo request too large")
                continue
            if type(item) is float:
                if not math.isfinite(item):
                    raise RuntimeError("Securo request too large")
                continue
            if type(item) is str:
                if len(item) > MAX_JSON_STRING:
                    raise RuntimeError("Securo request too large")
                string_bytes += len(item.encode("utf-8"))
                if string_bytes > MAX_REQUEST:
                    raise RuntimeError("Securo request too large")
                continue
            if type(item) not in {list, dict}:
                raise RuntimeError("Securo request too large")
            identity = id(item)
            if identity in active:
                raise RuntimeError("Securo request too large")
            active.add(identity)
            pending.append((item, depth, True))
            if type(item) is dict:
                mapping = cast(dict[str, object], item)
                if len(mapping) > MAX_JSON_MEMBERS:
                    raise RuntimeError("Securo request too large")
                children: list[tuple[object, int, bool]] = []
                for key, value in mapping.items():
                    if type(key) is not str or len(key) > 512:
                        raise RuntimeError("Securo request too large")
                    children.extend(((key, depth + 1, False), (value, depth + 1, False)))
                pending.extend(reversed(children))
            else:
                sequence = cast(list[object], item)
                if len(sequence) > MAX_JSON_NODES:
                    raise RuntimeError("Securo request too large")
                pending.extend((child, depth + 1, False) for child in reversed(sequence))

        encoded = bytearray()
        for chunk in encoder.iterencode(payload):
            for offset in range(0, len(chunk), 4096):
                raw = chunk[offset:offset + 4096].encode("utf-8")
                remaining = MAX_REQUEST + 1 - len(encoded)
                encoded.extend(raw[:remaining])
                if len(encoded) > MAX_REQUEST:
                    encoded.clear()
                    raise RuntimeError("Securo request too large")
        return bytes(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError):
        raise RuntimeError("Securo request too large") from None
    finally:
        pending.clear()
        active.clear()


class PairingClient:
    def __init__(
        self,
        server: str,
        pairing_id: uuid.UUID,
        secret: str,
        *,
        allow_http_loopback: bool = False,
    ) -> None:
        self.server = validate_server_origin(server, allow_http_loopback=allow_http_loopback)
        if not 40 <= len(secret) <= 256 or not secret.isascii() or any(c.isspace() for c in secret):
            raise ValueError("pairing secret is invalid")
        self.pairing_id = pairing_id
        self._authorization = "SecuroPairing " + secret
        self._client = _new_http_client()

    def close(self) -> None:
        self._authorization = ""
        self._client.close()

    def __enter__(self) -> "PairingClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @secret_boundary("Securo pairing request failed safely")
    def _request(self, method: str, suffix: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {"Authorization": self._authorization, "Accept-Encoding": "identity"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        encoded = _bounded_request_json(payload) if payload is not None else None
        raw = bytearray()
        response: httpx.Response | None = None
        status_code = 0
        redirect = False
        try:
            with self._client.stream(
                method,
                f"{self.server}/api/agents/chatgpt/pairings/{self.pairing_id}/{suffix}",
                headers=headers,
                content=encoded,
                extensions={"timeout": {"pool": 30.0, "connect": 5.0, "read": 30.0, "write": 30.0}},
            ) as response:
                self._client.cookies.clear()
                redirect = response.is_redirect
                status_code = response.status_code
                declared = response.headers.get_list("content-length")
                if len(declared) > 1:
                    raise RuntimeError("Securo returned invalid response size")
                if declared:
                    try:
                        length = int(declared[0])
                    except ValueError:
                        raise RuntimeError("Securo returned invalid response size") from None
                    if length < 0 or length > MAX_RESPONSE:
                        raise RuntimeError("Securo response too large")
                encoding = response.headers.get("content-encoding", "identity").casefold()
                if encoding not in {"", "identity"}:
                    raise RuntimeError("Securo compressed response rejected")
                for chunk in response.iter_raw(chunk_size=min(MAX_RESPONSE + 1, 65_536)):
                    remaining = MAX_RESPONSE + 1 - len(raw)
                    raw.extend(chunk[:remaining])
                    if len(raw) > MAX_RESPONSE:
                        raise RuntimeError("Securo response too large")
        except httpx.HTTPError:
            raise RuntimeError("Securo is unreachable") from None
        finally:
            if response is not None:
                _scrub_response(response)
            headers.clear()
            encoded = b""
        if redirect:
            raw.clear()
            raise RuntimeError("Securo redirect rejected")
        if status_code >= 400:
            raw.clear()
            raise RuntimeError(f"Securo pairing request failed ({status_code})")
        try:
            result = _strict_json(bytes(raw))
        finally:
            raw.clear()
        if not isinstance(result, dict):
            raise RuntimeError("Securo returned malformed JSON")
        return result

    @secret_boundary("Securo pairing request failed safely")
    def bootstrap(self) -> dict[str, Any]:
        return self._request("GET", "bootstrap")

    @secret_boundary("Securo pairing request failed safely")
    def register_attempt(self, attempt_id: uuid.UUID, nonce: str, redirect_uri: str) -> None:
        self._request("POST", "attempt", {
            "attempt_id": str(attempt_id), "nonce": nonce, "redirect_uri": redirect_uri,
        })

    @secret_boundary("Securo completion could not be confirmed")
    def complete(self, attempt_id: uuid.UUID, issued_client_id: str, token_response: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "attempt_id": str(attempt_id),
            "issued_client_id": issued_client_id,
            "token_response": token_response,
        }
        last_error: RuntimeError | None = None
        for delay in (0.0, 0.5, 1.5):
            if delay:
                time.sleep(delay)
            try:
                return self._request("POST", "complete", payload)
            except RuntimeError:
                last_error = RuntimeError("Securo completion could not be confirmed")
        assert last_error is not None
        raise last_error


def _pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate JSON member")
        result[key] = value
    return result


def _strict_json(raw: bytes) -> Any:
    if len(raw) > MAX_RESPONSE:
        raise ValueError("JSON too large")
    value = json.loads(
        raw,
        object_pairs_hook=_pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )
    pending = [(value, 0)]
    nodes = 0
    string_units = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 4096 or depth > 16:
            raise ValueError("JSON graph exceeds bounds")
        if isinstance(item, str):
            string_units += len(item)
            if len(item) > 65_536 or string_units > MAX_RESPONSE:
                raise ValueError("JSON string exceeds bounds")
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, dict):
            pending.extend((child, depth + 1) for pair in item.items() for child in pair)
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("nonfinite JSON number")
        elif item is not None and type(item) not in {bool, int}:
            raise ValueError("unsupported JSON value")
    return value
