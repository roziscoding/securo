"""Child-process adapter that executes the public helper CLI against local fakes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from urllib.request import urlopen

import httpx

from tests.integration.fake_openai import FakeOpenAI


class _StreamingTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport) -> None:
        self.inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self.inner.handle_request(request)
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            stream=httpx.ByteStream(response.read()),
            request=request,
            extensions=response.extensions,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--provider-bridge")
    parser.add_argument("cli_args", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    cli_args = options.cli_args
    if cli_args[:1] == ["--"]:
        cli_args = cli_args[1:]

    from securo_siwc import backend as backend_module  # ty: ignore[unresolved-import]
    from securo_siwc import cli  # ty: ignore[unresolved-import]
    from securo_siwc import openai as openai_module  # ty: ignore[unresolved-import]

    fake = FakeOpenAI() if options.provider_bridge is None else None
    captured: dict[str, object] = {"callbacks": [], "shutdown": "process-exit"}
    generated: dict[str, set[str]] = {
        "authorization": set(),
        "credentials": set(),
        "identity": set(),
    }

    original_create = cli.create_authorization
    def create_authorization(*args, **kwargs):
        authorization = original_create(*args, **kwargs)
        generated["authorization"].update(
            {authorization.state, authorization.nonce, authorization.verifier}
        )
        return authorization

    def replace_browser(url: str) -> bool:
        values = parse_qs(urlsplit(url).query, strict_parsing=True)
        authorization_values = {
            key: values[key][0]
            for key in ("client_id", "redirect_uri", "nonce", "code_challenge", "state")
        }
        if options.provider_bridge is None:
            assert fake is not None
            callback = fake.authorize(**authorization_values)
        else:
            from tests.integration.provider_bridge_transport import authorize_via_bridge

            callback = authorize_via_bridge(options.provider_bridge, authorization_values)
        with urlopen(callback, timeout=5) as response:
            body = response.read()
            captured["callbacks"] = [{
                "status": response.status,
                "body_bytes": len(body),
                "loopback_host": urlsplit(callback).hostname,
            }]
        return True

    def openai_client() -> httpx.Client:
        if options.provider_bridge is None:
            assert fake is not None
            transport: httpx.BaseTransport = fake.sync_transport
        else:
            from tests.integration.provider_bridge_transport import ProviderBridgeSyncTransport

            transport = ProviderBridgeSyncTransport(options.provider_bridge)
        return httpx.Client(
            transport=_StreamingTransport(transport),
            timeout=30,
            follow_redirects=False,
            trust_env=False,
            cookies=None,
            auth=None,
            event_hooks={},
        )

    cli.create_authorization = create_authorization

    cli.webbrowser.open = replace_browser
    openai_module._new_http_client = openai_client

    sys.argv = ["securo-siwc", *cli_args]
    result = cli.main()
    if fake is not None:
        generated["credentials"].update(fake.sensitive_values_for_test())
        generated["identity"].update(fake._subjects)
    options.artifacts.write_text(json.dumps(captured, sort_keys=True), encoding="utf-8")
    options.manifest.write_text(
        json.dumps({key: sorted(values) for key, values in generated.items()}, sort_keys=True),
        encoding="utf-8",
    )
    # Ensure the runner did not accidentally use ambient proxy/client state.
    assert backend_module._new_http_client is not openai_client
    return result


if __name__ == "__main__":
    raise SystemExit(main())
