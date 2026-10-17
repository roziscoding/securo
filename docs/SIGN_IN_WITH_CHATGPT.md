# Sign in with ChatGPT: user and operations guide

Securo can use an eligible ChatGPT plan for an AI-agent connection without an OpenAI API key. This integration is an OpenAI preview for eligible open-source, locally hosted applications. Eligibility, plan entitlements, model availability, limits, and the protocol may change. A hosted/SaaS Securo service must confirm its eligibility with OpenAI before enabling this path.

This is a separate connection type from **OpenAI — API key**. It has no API-key fallback, does not use an instance-wide OpenAI key, and cannot provide embeddings. Keep a separate embedding provider for agent knowledge/RAG.

## Security model and supported endpoints

The browser authorizes OpenAI, but the local helper receives the loopback callback and transfers the resulting credential bundle directly to the Securo backend. Users must never paste an access token, refresh token, ID token, authorization code, or credential bundle into Securo. The frontend does not receive OAuth credentials and they are not stored in browser local storage, session storage, URLs, analytics, or logs. Securo stores the token bundle encrypted with the installation's `SECRET_KEY`.

Production SIWC code is restricted to these public endpoints:

| Purpose | Method and endpoint |
|---|---|
| authorization | `GET https://auth.openai.com/api/accounts/authorize` |
| token exchange and refresh | `POST https://auth.openai.com/api/accounts/oauth/token` |
| signing keys | `GET https://auth.openai.com/.well-known/jwks.json` |
| revocation | `POST https://auth.openai.com/api/accounts/oauth/revoke` |
| model catalog | `GET https://api.openai.com/v1/models` |
| inference | `POST https://api.openai.com/v1/responses` |

The private `chatgpt.com/backend-api` surface is forbidden. Do not configure, proxy, or patch Securo to use it. SIWC inference uses only the public Responses API with `store: false` and streaming enabled.

The helper binds the callback to literal `127.0.0.1`, not `localhost`, on the computer running the helper. It never exposes that listener to the LAN. The Securo server may be remote; the loopback listener still belongs on the user's browser computer.

## Prerequisites

- An eligible ChatGPT account/plan with direct token use enabled by OpenAI.
- Securo with `AGENTS_ENABLED=true` and the agents services running.
- A stable production `SECRET_KEY`; changing it makes existing encrypted credentials unreadable.
- PostgreSQL migrated through the current head.
- A public Securo **HTTPS origin** reachable from the user's computer. The helper rejects plain HTTP, redirects, URL credentials, paths, query strings, and fragments. Literal `http://127.0.0.1:<port>` is a hidden test-only exception, not a deployment mode.
- Python 3.11 or newer on the computer running the helper (Linux, macOS, or Windows).
- For scheduled proactive refresh, both a Celery worker and Celery beat. Beat enqueues `refresh-agent-oauth-near-expiry` every five minutes only while agents are enabled. On-demand inference can refresh without beat, but running worker/beat avoids preventable expiry at first use.

For remote Docker, VM, LXC, or Kubernetes installations, put Securo behind a TLS-terminating reverse proxy and set `FRONTEND_URL` to the external HTTPS origin. Preserve the original host and scheme headers according to the existing Securo reverse-proxy configuration. The helper connects to that public origin; do not forward the helper's `127.0.0.1` callback port to the server.

## Install the helper

The helper currently ships in the Securo source tree; there is no claimed PyPI release. Install from the same trusted checkout/version as the server when possible.

### Linux and macOS

```console
git clone https://github.com/securo-finance/securo.git
cd securo
python3 -m venv .venv-siwc
. .venv-siwc/bin/activate
python -m pip install ./tools/securo-siwc
securo-siwc --help
```

For an existing checkout, update it using your normal verified upgrade process before installing. A source archive or built wheel produced by that checkout can be copied to the user's computer and installed with `python -m pip install <wheel>`.

### Windows PowerShell

```powershell
git clone https://github.com/securo-finance/securo.git
Set-Location securo
py -3.11 -m venv .venv-siwc
.\.venv-siwc\Scripts\Activate.ps1
python -m pip install .\tools\securo-siwc
securo-siwc --help
```

Do not add the pairing secret to an installer command, shell profile, environment variable, configuration file, URL, or shortcut.

## Pair a ChatGPT plan

1. In Securo, open **Settings → AI Agents → Connections**.
2. Choose **OpenAI — ChatGPT plan**, give the connection a name, and create the pairing.
3. Copy the public pairing ID and keep the separately displayed one-time pairing secret private.
4. On the same computer as the browser, run:

   ```console
   securo-siwc <PAIRING_ID> --server https://securo.example
   ```

5. Enter the pairing secret only at the helper's hidden `Pairing secret:` prompt. It is intentionally unsupported in argv, environment variables, configuration files, or URLs. `--secret-stdin` exists for a protected local pipe from a secret manager; avoid interactive shell history and never use a world-readable file.
6. Complete authorization in the browser. The browser returns to a temporary literal `http://127.0.0.1:<ephemeral-port>/auth/callback` listener on that computer.
7. Wait for `ChatGPT authorization transferred to Securo.` and for Securo to show an active connection. Close any authorization tab that remains open.
8. Test the connection and select one of the models returned for that account.

The public pairing ID is not sufficient by itself. The one-time secret is sent only in an authorization header over HTTPS. Pairings expire and are consumed atomically; start a new pairing rather than reusing an old secret.

`--no-browser` prints the OpenAI authorization URL for environments that cannot launch a browser. Treat that URL as short-lived sensitive workflow material: do not paste it into tickets, chat, or logs. It never replaces the hidden pairing-secret prompt.

## Proxy, private CA, and TLS settings

Ambient `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and CA environment variables are deliberately ignored for backend SIWC OAuth/model traffic. Configure the implemented agent settings explicitly:

```dotenv
# HTTP(S) proxy origin only; no path, query, fragment, or credentials.
AGENTS_SIWC_OAUTH_PROXY_URL=http://proxy.internal:3128

# Optional absolute path inside the API/worker container or host.
# It must be a readable, regular, non-symlink PEM certificate bundle.
AGENTS_SIWC_OAUTH_CA_BUNDLE=/app/secrets/corporate-ca.pem

# Optional; each must be greater than 0 and no more than 300 seconds.
AGENTS_SIWC_OAUTH_TIMEOUT_SECONDS=30
AGENTS_SIWC_OAUTH_CONNECT_TIMEOUT_SECONDS=5
```

Mount the CA bundle read-only into every backend and worker process that can refresh or test a connection. Proxy credentials are not accepted in the URL. If the proxy requires authentication, use a network-layer mechanism that does not put credentials in Securo configuration. Restart the API, worker, and beat after changing these values.

These backend settings do not configure the standalone helper. The helper also ignores ambient proxy variables and uses the operating system trust store. Run it from a network that can directly reach the Securo HTTPS origin and the documented OpenAI hosts. Do not disable certificate validation.

## Credential lifecycle and recovery

| State or error | Meaning and safe recovery |
|---|---|
| `active` | Credential can be used. Test the connection if model discovery or inference fails. |
| `disabled` | ChatGPT-plan use is disabled locally. Re-enable/reconnect from the UI as appropriate. |
| `reauth_required` | Start a fresh pairing. Never paste or restore old tokens manually. |
| `revoked` | Remote revocation was confirmed (or OpenAI confirmed the token was already invalid) and the local bundle was scrubbed. The connection may then be deleted or reconnected. |
| `refresh_outcome_unknown` | A rotating refresh may have reached OpenAI, but Securo did not receive a conclusive response. The old refresh token is never retried. Explicitly reconnect. |
| `revocation_pending` | Revocation had an ambiguous network outcome. Retry disconnect/revocation after connectivity is restored; do not claim the remote session is revoked yet. |
| `resource_cleanup_pending` | Remote revocation completed, but local cleanup still needs reconciliation. Retry cleanup/reconnect; automatic refresh excludes this credential. |
| `scope_loss` / entitlement rejected | The account did not grant required direct-use scope or is not eligible. Check plan/workspace policy, then reconnect. Do not switch silently to an API key. |
| `invalid_client` | Retained client metadata was rejected. Create a new pairing; repeated failure needs operator investigation. |
| `transient` / unavailable / rate limit | Preserve the credential, wait, and retry with bounded backoff. Usage-limit failures are not fixed by repeating OAuth. |
| corrupt/unreadable credential | Usually indicates changed `SECRET_KEY`, damaged data, or an invalid token. Restore the matching key from backup or reconnect; never print/decrypt the bundle while troubleshooting. |

Deletion is deliberately blocked while a credential has a live bundle, refresh attempt, revocation-pending state, or lifecycle lease. Revoke/disconnect first. If revocation cannot be confirmed, retain the record and retry rather than deleting proof of pending remote cleanup. A successful reconnect replaces the usable credential through the pairing lifecycle; it does not make an unknown refresh outcome safe to retry.

## Safe verification and troubleshooting

Safe checks reveal status and routing, not credential material:

```console
# Backend health and migration chain
cd backend
python3 scripts/check_migration_chain.py

# Helper packaging/entry point
securo-siwc --help

# Container/process health (adapt service names to the deployment)
docker compose ps
```

Then use the UI to inspect only sanitized fields: connection name, account email, credential state, expiry time, failure code, and available model display names. A test connection should perform model discovery without returning tokens. An agent test should call a harmless read-only Securo tool and show only the tool result.

For failures:

1. Confirm the public Securo URL is HTTPS and does not redirect to another origin.
2. Confirm DNS and TLS trust for the Securo origin, `auth.openai.com`, and `api.openai.com` from the relevant machine/process.
3. Confirm `AGENTS_ENABLED=true`; confirm API, worker, beat, Redis, PostgreSQL, and the built-in MCP server are healthy.
4. If a private CA/proxy is required, verify the explicit settings above in API and worker containers. Do not rely on ambient proxy variables.
5. Check sanitized event names, status codes, connection IDs, and failure codes. Never enable HTTP body/header dumps or debug logging in `httpx`/TLS/proxy components.
6. For `refresh_outcome_unknown`, reconnect. For `revocation_pending`, restore connectivity and retry revocation. For `resource_cleanup_pending`, retry local cleanup/reconnect.
7. If `SECRET_KEY` changed, restore the original key with the matching database backup or reconnect every affected credential.

Never collect or publish:

- `Authorization` or `SecuroPairing` headers;
- pairing secrets, authorization URLs, callback query strings, codes, PKCE values, nonces, access/refresh/ID tokens, encrypted bundles, or cookies;
- database rows containing `token_bundle_encrypted`;
- TLS key logs, packet captures, core dumps, process environments, or verbose proxy traces from a live pairing.

When opening an issue, record only Securo/helper version, operating system, sanitized state/failure code, whether the server is remote, whether an explicit proxy/private CA is configured, and timestamps rounded enough for correlation.

## Backup, upgrade, and rollback

Back up the PostgreSQL database and the exact `SECRET_KEY` together before upgrading. Protect both as credentials. Exclude transient helper output, browser data, logs, and pairing secrets; the helper persists none of them. Test restore procedures in an isolated environment without contacting OpenAI.

Upgrade in this order:

1. Stop API, worker, and beat so no refresh/revocation can race the backup.
2. Back up PostgreSQL and deployment secrets/configuration.
3. Deploy matching backend/frontend/helper source and dependencies.
4. Run Alembic upgrade to the current head.
5. Start API, worker, beat, and MCP services; verify sanitized connection states and test one connection.

Rollback application code and database schema as one compatible unit. Migrations have intentionally destructive downgrade behavior:

- Downgrading **099 → 098** drops all short-lived OAuth pairing rows. Users must start new pairings.
- Downgrading **098 → 097** deletes OAuth credential rows whose encrypted bundle has already been scrubbed, because 097 requires a non-null bundle. This includes terminal/revoked/reauthorization rows with no bundle. It also drops lifecycle lease and refresh-attempt metadata.
- Downgrading **097 → 096** drops the entire `agent_openai_oauth_credentials` table and therefore every ChatGPT-plan credential.

Existing API-key connections in `agent_llm_connections` are preserved across 097/098/099 upgrades and the tested 099 → 098 → 099 cycle. Nevertheless, take a backup before any downgrade. Prefer forward fixes; restore the pre-upgrade database and matching `SECRET_KEY` for a full rollback instead of relying on destructive migration downgrades.

## Opt-in live smoke checklist

This is a manual release/operator checklist, **not CI**. It contacts real OpenAI services, consumes real plan capacity, and requires ignored local credentials. Never add its secrets, output, screenshots, browser profile, or copied authorization URLs to the repository or CI secrets. Record only sanitized facts: platform, Securo/helper revision, pass/fail, sanitized state/failure code, model display name/slug, and approximate timestamp.

Prepare one disposable test user/connection on a non-production Securo instance with a valid HTTPS origin. Run the matrix on Linux, macOS, and Windows where available, plus at least one remote Securo installation behind the supported reverse proxy/TLS path.

- [ ] Confirm backups, `SECRET_KEY`, migrations, API, Redis, Celery worker/beat, PostgreSQL, and MCP service are healthy.
- [ ] Install the helper from the checked-out source (or its locally built wheel) in a fresh Python 3.11+ environment; `securo-siwc --help` succeeds.
- [ ] Create **OpenAI — ChatGPT plan** in the UI and record only the public pairing ID long enough to run the helper.
- [ ] Run the helper with the public ID and HTTPS server origin; enter the separate secret only at the hidden prompt. Confirm no secret appears in argv, environment, URL, terminal scrollback, browser storage, logs, or process listing.
- [ ] Complete the browser flow and confirm the callback is literal `127.0.0.1`, including when Securo is remote. Do not port-forward the callback to the server.
- [ ] Confirm the connection becomes `active`; test model discovery and select a returned visible model.
- [ ] Send a harmless agent prompt and verify a real, read-only Securo tool call completes through the public Responses API.
- [ ] If the test account and timing make it safe, use the product's explicit refresh action or a controlled near-expiry test to force one refresh. Confirm the connection remains active and that no token/header/body is logged. Do not replay a refresh after an ambiguous outcome.
- [ ] Disconnect/revoke. Confirm a conclusive `revoked` state and local bundle cleanup; if the result is `revocation_pending`, record that sanitized state and retry only after restoring connectivity.
- [ ] Reconnect the same account, confirm retained client metadata is accepted, then test model discovery again.
- [ ] Revoke once more and delete the connection. Confirm deletion is blocked until revocation/local cleanup is complete, then succeeds.
- [ ] Repeat the pairing path through the production reverse proxy, including any explicit backend proxy/private-CA configuration.
- [ ] Delete the disposable test user/data and any local virtual environment or wheel. Confirm no credential-bearing artifact was retained.

## Limitations

- OpenAI controls preview eligibility, models, limits, scopes, and account/workspace policy.
- SIWC is chat/inference only: no embeddings and no RAG embedding fallback.
- No API-key fallback is attempted when OAuth is unavailable, ineligible, expired, revoked, or over limit.
- Hosted tools and unsupported Responses capabilities are not silently substituted. Securo's local function tools are the supported tool path.
- The local helper currently requires Python and is distributed from source/current repository packaging, not as an asserted PyPI release or native installer.
- The helper cannot traverse an authenticated proxy and intentionally ignores ambient proxy configuration.
- A remote Securo server still requires a helper on the browser computer because OpenAI returns to literal loopback.

For protocol and design history, see [the implementation study](SIWC_IMPLEMENTATION_STUDY.md). For helper-specific development details, see [`tools/securo-siwc/README.md`](../tools/securo-siwc/README.md).
