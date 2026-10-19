# Securo Sign in with ChatGPT helper

This standalone Python 3.11+ tool completes a Securo pairing on the same computer as your browser. OAuth credentials go directly from this process to Securo and are never sent through the Securo web frontend.

## Install

```console
python -m pip install ./tools/securo-siwc
```

## Run

Copy the **public pairing ID** from Securo, then run:

```console
securo-siwc 00000000-0000-0000-0000-000000000000 --server https://securo.example
```

The helper prompts invisibly for the separate pairing secret. For protected pipe input, use `--secret-stdin`; the tool reads exactly one bounded line. There is intentionally no secret command-line option, environment variable, config file, or disk persistence.

The server must be an absolute HTTPS origin. Redirects, ambient proxy variables, cookies, and credential-bearing URLs are not used. A hidden `--allow-http-loopback` switch accepts only literal `http://127.0.0.1:<port>` for automated tests.

## Protocol

1. Authenticate to the selected pairing with `Authorization: SecuroPairing …`.
2. Bind an already-created listener to literal `127.0.0.1` on an ephemeral port.
3. Register a fresh attempt ID, nonce, and exact callback URI before opening the browser.
4. Reuse retained OAuth client metadata when available; otherwise request OpenAI dynamic client registration with PKCE S256 and all six required scopes.
5. Accept only the first matching-state terminal callback; wrong-state callbacks do not consume the listener.
6. Exchange and validate the token response against OpenAI's fixed public JWKS endpoint.
7. Transfer the exact response directly to Securo. Securo independently validates it and atomically consumes the pairing.

No credentials are written to disk. If completion has an ambiguous network outcome, the helper retries only the identical payload and checks sanitized pairing status.

Securo derives `ext_agent_host_id` from a domain-separated installation secret, so all pairings and connections from one installation use the same opaque UUID across restarts. Keep that installation secret stable. Rotating it intentionally rotates the host identity; retained client IDs can still be reused, while later dynamic registrations use the new host identity. The source secret is never returned to the helper.
