# Spike 001 — SIWC open-source dynamic registration

## Question

**Given** a local or self-hosted Securo runtime, **when** it performs OpenAI's OSS
`dynamic_agent_client` OAuth flow, **then** can it validate the returned identity,
list subscription-backed models, and complete a streamed Responses request without
an API key?

Risk order:

1. Dynamic registration and loopback callback really work for a third-party OSS app.
2. ID-token and direct-plan scope validation work with the issued client ID.
3. The bearer token lists account models through the public API.
4. `POST /v1/responses` reaches `response.completed`.

## What this spike contains

- `siwc_spike.py`
  - Authorization Code + PKCE S256.
  - Stable `urn:uuid:` host ID.
  - Literal `127.0.0.1` loopback callback.
  - First-time dynamic registration and returning-client checks.
  - Code exchange at OpenAI's public token endpoint.
  - RS256/JWKS ID-token validation including issuer, audience, expiry, subject,
    and nonce.
  - Required `chatgpt.tokens.use.direct` scope check.
  - Atomic credential file with mode `0600`.
  - Account-specific model listing.
  - Public Responses API inference with `store=false`, `stream=true`, and a hard
    requirement for `response.completed`.
- `test_siwc_spike.py`
  - Protocol, callback, JWT, scope, SSE, error, and credential-file tests.

Generated credentials and host state live under this spike's `.state/`, anchored
to the script directory regardless of the caller's working directory and ignored
by Git.
The CLI never prints tokens.

## Run

From the repository root:

```bash
cd backend
mise x -- uv sync --all-extras --frozen
cd ../spikes/001-siwc-open-source
../../backend/.venv/bin/python -m unittest -v test_siwc_spike.py
../../backend/.venv/bin/python siwc_spike.py dry-run
../../backend/.venv/bin/python siwc_spike.py login
../../backend/.venv/bin/python siwc_spike.py probe-saved
```

The live command opens a browser, stores credentials at
`.state/credentials.json`, lists models, and sends this default prompt:

```text
Say exactly: Securo SIWC works.
```

### Remote host through SSH

When the browser runs on another machine, pin the listener port:

```bash
# On Judith
../../backend/.venv/bin/python siwc_spike.py login --no-browser --port 1455 --timeout 900

# On the browser machine
ssh -N -L 1455:127.0.0.1:1455 cohen@judith
```

Then open the authorization URL printed by the first command in a normal local
browser. The OpenAI redirect to the browser machine's `127.0.0.1:1455` crosses the
SSH tunnel to the spike listener.

## Evidence collected

- Unit suite: 18 tests pass, including regressions for callback poisoning,
  credential-path containment, redirect token leakage, and exact tool calls.
- Dry-run output has the documented authorization endpoint, dynamic client,
  Securo name hint, exact scope set, API resource, stable host ID, PKCE S256, and
  literal loopback callback.
- In a manually observed live run, dynamic registration completed through an
  SSH-forwarded loopback callback.
- OpenAI issued an `oaiapp_...` client ID with all six requested scopes.
- RS256 ID-token validation passed for issuer, issued-client audience, expiry,
  subject, and the original nonce.
- The token response returned `expires_in=3600` and an integer
  `earliest_refresh_at` corresponding to roughly eight minutes before expiry in
  this observed response.
- In that live run, the account-specific public model catalog returned seven visible models.
- A real public Responses request using `gpt-6.1-sol`, `store=false`, and
  `stream=true` completed with the exact text `Securo SIWC works.` and usage
  `input=15`, `output=10`.
- A second real Responses request forced a function call. It completed with
  `securo_spike_probe({"proof":"live"})` and usage `input=62`, `output=21`.
- The credential record is 4006 bytes and mode `0600`; `.state/` remains ignored.

## Verdict: VALIDATED

### What worked

- Authorization request construction and callback invariants.
- Signed JWT/JWKS validation behavior with a real generated RSA key.
- Required-scope enforcement.
- Responses SSE success/failure semantics.
- Atomic owner-only credential persistence.
- Dynamic registration with no pre-issued client ID, API key, or client secret.
- SSH-forwarded loopback authorization for a remote self-hosted runtime.
- Issued-client ID, complete direct-plan scope grant, and signed identity.
- Account model listing through `GET https://api.openai.com/v1/models`.
- Text inference and function calling through the public Responses API.
- Stream completion and usage parsing.

### What remains for a production implementation

- Refresh-token rotation after `earliest_refresh_at`; it was not safe or necessary
  to wait for expiry during this spike.
- Revocation and reauthorization recovery.
- Multi-account/workspace switching.
- Production backend pairing, encrypted storage, locking, and frontend UX.

### Surprise

The automated browser was challenged by Cloudflare, but the fixed-port listener
plus an SSH `-L` tunnel allowed the normal user browser to satisfy both the
security check and OpenAI's strict `127.0.0.1` callback requirement. This validates
the proposed remote-helper boundary rather than weakening it.

### Recommendation for the real build

Proceed with a production implementation using a separate Responses provider and
a local helper/pairing flow around the same loopback boundary. Implement refresh
rotation and revocation next, before frontend work, and keep API-key OpenAI as a
separate connection path.
