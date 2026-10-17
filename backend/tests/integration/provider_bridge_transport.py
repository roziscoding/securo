"""Test-only IPC transports for a fixed-origin provider fake."""
from __future__ import annotations

import base64
from typing import Any

import httpx


def _encode(request: httpx.Request) -> dict[str, Any]:
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": list(request.headers.multi_items()),
        "content": base64.b64encode(request.content).decode("ascii"),
    }


def _decode(payload: dict[str, Any], request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        payload["status"],
        headers=payload["headers"],
        content=base64.b64decode(payload["content"], validate=True),
        request=request,
    )


class _AsyncBytes(httpx.AsyncByteStream):
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def __aiter__(self):
        yield self.content


def _decode_async(payload: dict[str, Any], request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        payload["status"],
        headers=payload["headers"],
        stream=_AsyncBytes(base64.b64decode(payload["content"], validate=True)),
        request=request,
    )


class ProviderBridgeSyncTransport(httpx.BaseTransport):
    def __init__(self, origin: str) -> None:
        self.origin = origin

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        with httpx.Client(trust_env=False, follow_redirects=False, timeout=10) as client:
            response = client.post(self.origin + "/request", json=_encode(request))
            response.raise_for_status()
            return _decode(response.json(), request)


class ProviderBridgeAsyncTransport(httpx.AsyncBaseTransport):
    def __init__(self, origin: str) -> None:
        self.origin = origin

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async with httpx.AsyncClient(
            trust_env=False, follow_redirects=False, timeout=10
        ) as client:
            response = await client.post(self.origin + "/request", json=_encode(request))
            response.raise_for_status()
            return _decode_async(response.json(), request)


def authorize_via_bridge(origin: str, values: dict[str, str]) -> str:
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=10) as client:
        response = client.post(origin + "/authorize", json=values)
        response.raise_for_status()
        callback = response.json().get("callback")
    if not isinstance(callback, str):
        raise RuntimeError("provider bridge returned an invalid callback")
    return callback
