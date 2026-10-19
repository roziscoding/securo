# Securo Sign in with ChatGPT helper

This standalone Python 3.11+ tool completes a Securo pairing on the same computer as your browser. OAuth credentials go directly from this process to Securo and are never sent through the Securo web frontend. Read the full [user and operations guide](../../../docs/SIGN_IN_WITH_CHATGPT.md) before using it against a remote or production installation.

## Run from a Securo installation

Start a pairing in Securo and copy the command shown by the pairing dialog. The running backend serves a version-matched bootstrap and helper wheel, so normal users do not need to clone this repository or install a helper package.

## Development install

For helper development or a protected secret-manager pipe, install from a trusted checkout:

```console
python -m venv .venv-siwc
# Linux/macOS: . .venv-siwc/bin/activate
# Windows PowerShell: .\.venv-siwc\Scripts\Activate.ps1
python -m pip install ./backend/tools/securo-siwc
securo-siwc --help
```

The standalone package is built from the helper source embedded in the Securo backend. This repository does not claim a PyPI release or native installer; use the checkout matching the server.

## Run

Copy the **public pairing ID** from Securo, then run:

```console
securo-siwc 00000000-0000-0000-0000-000000000000 --server https://securo.example
```

The helper prompts invisibly for the separate pairing secret. The secret must
be entered only at that hidden prompt. For a protected local pipe from a secret
manager, `--secret-stdin` reads exactly one bounded line. There is intentionally
no secret command-line option, environment variable, config file, URL field, or
disk persistence; do not redirect it from a world-readable file.

The server must be an absolute HTTPS origin. Redirects, ambient proxy variables,
cookies, and credential-bearing URLs are not used. A hidden
`--allow-http-loopback` switch accepts only literal
`http://127.0.0.1:<port>` for automated tests; it is not a deployment option.
The OpenAI callback itself always binds literal `127.0.0.1` on this computer,
including when Securo runs on a remote server.

## Protocol

1. Authenticate to the selected pairing with a `SecuroPairing` authorization header.
2. Bind an already-created listener to literal `127.0.0.1` on an ephemeral port.
3. Register a fresh attempt ID, nonce, and exact callback URI before opening the browser.
4. Reuse retained OAuth client metadata when available; otherwise request OpenAI dynamic client registration with PKCE S256 and all six required scopes.
5. Accept only the first matching-state terminal callback; wrong-state callbacks do not consume the listener.
6. Exchange and validate the token response against OpenAI's fixed public JWKS endpoint.
7. Transfer the exact response directly to Securo. Securo independently validates it and atomically consumes the pairing.

No credentials are written to disk. If completion has an ambiguous network outcome, the helper retries only the identical payload and checks sanitized pairing status.

Securo derives `ext_agent_host_id` from a domain-separated installation secret, so all pairings and connections from one installation use the same opaque UUID across restarts. Keep that installation secret stable. Rotating it intentionally rotates the host identity; retained client IDs can still be reused, while later dynamic registrations use the new host identity. The source secret is never returned to the helper.
