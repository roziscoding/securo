"""Pinned HTTP and protocol primitives for the local helper."""
from __future__ import annotations

from urllib.parse import urlsplit


def validate_server_origin(value: str, *, allow_http_loopback: bool) -> str:
    if value != value.strip() or value.endswith("/"):
        raise ValueError("server must be an absolute origin without a trailing slash")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("server origin is invalid") from exc
    allowed_http = (
        allow_http_loopback and parsed.scheme == "http" and parsed.hostname == "127.0.0.1"
    )
    if (
        (parsed.scheme != "https" and not allowed_http)
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
        or port == 0
    ):
        raise ValueError("server must be an HTTPS origin")
    return value
