#!/usr/bin/env python3
"""Throwaway SIWC OSS protocol spike for Securo.

Run with the Securo backend interpreter so python-jose is available. State paths
are anchored beside this script, independent of the current working directory:

    backend/.venv/bin/python spikes/001-siwc-open-source/siwc_spike.py login
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import queue
import secrets
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from collections.abc import Iterable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

ISSUER = "https://auth.openai.com"
AUTHORIZE_ENDPOINT = f"{ISSUER}/api/accounts/authorize"
TOKEN_ENDPOINT = f"{ISSUER}/api/accounts/oauth/token"
JWKS_ENDPOINT = f"{ISSUER}/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
MODELS_ENDPOINT = f"{RESOURCE}/models"
RESPONSES_ENDPOINT = f"{RESOURCE}/responses"
REQUIRED_SCOPE_STRING = (
    "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
)
SPIKE_DIR = Path(__file__).resolve().parent
SPIKE_STATE_DIR = SPIKE_DIR / ".state"


@dataclass(frozen=True)
class PendingAuthorization:
    authorization_url: str
    redirect_uri: str
    state: str
    nonce: str
    verifier: str
    client_id: str


@dataclass(frozen=True)
class CallbackResult:
    code: str
    client_id: str


@dataclass(frozen=True)
class CallbackFailure:
    message: str


CallbackOutcome = CallbackResult | CallbackFailure


@dataclass(frozen=True)
class FunctionCall:
    call_id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class StreamResult:
    text: str
    completed: bool
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: tuple[FunctionCall, ...] = ()


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def create_authorization(
    *,
    host_id: str,
    port: int,
    state: str | None = None,
    nonce: str | None = None,
    verifier: str | None = None,
    returning_client_id: str | None = None,
) -> PendingAuthorization:
    state = state or secrets.token_urlsafe(32)
    nonce = nonce or secrets.token_urlsafe(32)
    verifier = verifier or secrets.token_urlsafe(64)
    client_id = returning_client_id or "dynamic_agent_client"
    redirect_uri = f"http://127.0.0.1:{port}/auth/callback"
    params = {
        "client_id": client_id,
        "ext_agent_host_id": host_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": REQUIRED_SCOPE_STRING,
        "resource": RESOURCE,
        "state": state,
        "nonce": nonce,
        "code_challenge_method": "S256",
        "code_challenge": _b64url(hashlib.sha256(verifier.encode("ascii")).digest()),
    }
    if returning_client_id is None:
        params["agent_name_hint"] = "Securo"
    return PendingAuthorization(
        authorization_url=f"{AUTHORIZE_ENDPOINT}?{urlencode(params)}",
        redirect_uri=redirect_uri,
        state=state,
        nonce=nonce,
        verifier=verifier,
        client_id=client_id,
    )


def _one(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key) or []
    return values[0] if values else None


def validate_callback(
    params: dict[str, list[str]],
    *,
    expected_state: str,
    returning_client_id: str | None,
) -> CallbackResult:
    if not secrets.compare_digest(_one(params, "state") or "", expected_state):
        raise ValueError("OAuth state mismatch")
    if error := _one(params, "error"):
        description = _one(params, "error_description") or error
        raise ValueError(f"OpenAI authorization failed: {description}")
    code = _one(params, "code")
    if not code:
        raise ValueError("callback did not contain an authorization code")
    supplied_client_id = _one(params, "client_id")
    if returning_client_id is None:
        if not supplied_client_id or supplied_client_id == "dynamic_agent_client":
            raise ValueError("new registration did not return an issued client_id")
        client_id = supplied_client_id
    else:
        if supplied_client_id and supplied_client_id != returning_client_id:
            raise ValueError("callback returned a different client_id")
        client_id = returning_client_id
    return CallbackResult(code=code, client_id=client_id)


def offer_callback(
    params: dict[str, list[str]],
    *,
    expected_state: str,
    returning_client_id: str | None,
    result_queue: queue.Queue[CallbackOutcome],
) -> bool:
    if not secrets.compare_digest(_one(params, "state") or "", expected_state):
        return False
    try:
        outcome: CallbackOutcome = validate_callback(
            params,
            expected_state=expected_state,
            returning_client_id=returning_client_id,
        )
    except ValueError as exc:
        outcome = CallbackFailure(str(exc))
    try:
        result_queue.put_nowait(outcome)
    except queue.Full:
        return False
    return True


def require_plan_scope(scope: str) -> set[str]:
    scopes = {item for item in scope.split() if item}
    if "chatgpt.tokens.use.direct" not in scopes:
        raise ValueError("token response lacks chatgpt.tokens.use.direct")
    return scopes


def parse_response_stream(lines: Iterable[str]) -> StreamResult:
    text: list[str] = []
    tool_calls: list[FunctionCall] = []
    completed = False
    input_tokens = 0
    output_tokens = 0
    for raw in lines:
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        event = json.loads(payload)
        event_type = event.get("type")
        if event_type == "response.output_text.delta":
            text.append(event.get("delta") or "")
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                raw_arguments = item.get("arguments") or "{}"
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": raw_arguments}
                tool_calls.append(
                    FunctionCall(
                        call_id=item.get("call_id") or item.get("id") or "",
                        name=item.get("name") or "",
                        arguments=arguments,
                    )
                )
        elif event_type == "response.failed":
            error = (event.get("response") or {}).get("error") or event.get("error") or {}
            code = error.get("code") or "response_failed"
            message = error.get("message") or "Responses API reported failure"
            raise RuntimeError(f"{code}: {message}")
        elif event_type == "response.incomplete":
            raise RuntimeError("response.incomplete")
        elif event_type == "response.completed":
            completed = True
            usage = (event.get("response") or {}).get("usage") or {}
            input_tokens = int(usage.get("input_tokens") or 0)
            output_tokens = int(usage.get("output_tokens") or 0)
    if not completed:
        raise RuntimeError("stream ended without response.completed")
    return StreamResult(
        text="".join(text),
        completed=True,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tool_calls=tuple(tool_calls),
    )


def write_credentials(path: Path, credentials: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(credentials, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(tmp_name).unlink(missing_ok=True)
        raise


def load_or_create_host_id(path: Path) -> str:
    if path.exists():
        value = path.read_text(encoding="utf-8").strip()
        if value:
            return value
    value = f"urn:uuid:{uuid.uuid4()}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(value + "\n")
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_no_redirect(request: Request, timeout: float):
    return build_opener(_NoRedirect).open(request, timeout=timeout)


def request_json(request: Request, timeout: float = 30.0) -> dict:
    try:
        with _open_no_redirect(request, timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        finally:
            exc.close()
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc


def exchange_code(pending: PendingAuthorization, callback: CallbackResult) -> dict:
    body = urlencode(
        {
            "grant_type": "authorization_code",
            "client_id": callback.client_id,
            "code": callback.code,
            "code_verifier": pending.verifier,
            "redirect_uri": pending.redirect_uri,
            "resource": RESOURCE,
        }
    ).encode("ascii")
    return request_json(
        Request(
            TOKEN_ENDPOINT,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
    )


def validate_id_token_payload(
    id_token: str,
    *,
    jwks: dict,
    client_id: str,
    nonce: str,
    access_token: str | None = None,
) -> dict:
    try:
        from jose import jwt
    except ImportError as exc:
        raise RuntimeError("python-jose is required; run this through backend/uv") from exc
    claims = jwt.decode(
        id_token,
        jwks,
        algorithms=["RS256"],
        audience=client_id,
        issuer=ISSUER,
        access_token=access_token,
    )
    if not secrets.compare_digest(str(claims.get("nonce") or ""), nonce):
        raise ValueError("ID token nonce mismatch")
    if not claims.get("sub"):
        raise ValueError("ID token has no subject")
    return claims


def validate_id_token(token_response: dict, *, client_id: str, nonce: str) -> dict:
    jwks = request_json(Request(JWKS_ENDPOINT))
    return validate_id_token_payload(
        token_response["id_token"],
        jwks=jwks,
        client_id=client_id,
        nonce=nonce,
        access_token=token_response.get("access_token"),
    )


def list_models(access_token: str) -> list[dict[str, str]]:
    payload = request_json(
        Request(MODELS_ENDPOINT, headers={"Authorization": f"Bearer {access_token}"})
    )
    return [
        {"slug": model["slug"], "display_name": model.get("display_name") or model["slug"]}
        for model in payload.get("models", [])
        if model.get("visibility") == "list" and model.get("slug")
    ]


def _post_response(access_token: str, payload: dict) -> StreamResult:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        RESPONSES_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with _open_no_redirect(request, 120) as response:
            lines = (raw.decode("utf-8", errors="replace") for raw in response)
            return parse_response_stream(lines)
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Inference HTTP {exc.code}: {error_body[:2000]}") from exc


def run_inference(access_token: str, *, model: str, prompt: str) -> StreamResult:
    return _post_response(
        access_token,
        {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "store": False,
            "stream": True,
        },
    )


def build_tool_probe_payload(model: str) -> dict:
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": "Call securo_spike_probe exactly once with proof set to live.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "name": "securo_spike_probe",
                "description": "Proves that subscription-backed function calling works.",
                "parameters": {
                    "type": "object",
                    "properties": {"proof": {"type": "string"}},
                    "required": ["proof"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ],
        "tool_choice": "required",
        "store": False,
        "stream": True,
    }


def run_tool_probe(access_token: str, *, model: str) -> StreamResult:
    return _post_response(access_token, build_tool_probe_payload(model))


def validate_tool_probe(result: StreamResult) -> None:
    expected = (
        len(result.tool_calls) == 1
        and bool(result.tool_calls[0].call_id)
        and result.tool_calls[0].name == "securo_spike_probe"
        and result.tool_calls[0].arguments == {"proof": "live"}
    )
    if not expected:
        raise RuntimeError(
            "expected exactly one securo_spike_probe call with proof='live' and a call_id"
        )


class _CallbackHandler(BaseHTTPRequestHandler):
    result_queue: queue.Queue[CallbackOutcome]
    expected_state: str
    returning_client_id: str | None

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/auth/callback":
            self.send_error(404)
            return
        accepted = offer_callback(
            parse_qs(parsed.query, keep_blank_values=True),
            expected_state=self.expected_state,
            returning_client_id=self.returning_client_id,
            result_queue=self.result_queue,
        )
        if not accepted:
            self.send_error(400, "Invalid or duplicate OAuth callback")
            return
        body = (
            b"<!doctype html><meta charset=utf-8><title>Securo SIWC spike</title>"
            b"<h1>Authorization received</h1>"
            b"<p>You can close this tab and return to the terminal.</p>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def wait_for_callback(
    *,
    host_id: str,
    port: int,
    timeout: int,
    open_browser: bool,
    returning_client_id: str | None,
) -> tuple[PendingAuthorization, CallbackResult]:
    result_queue: queue.Queue[CallbackOutcome] = queue.Queue(maxsize=1)
    server = None
    pending = create_authorization(
        host_id=host_id,
        port=port,
        returning_client_id=returning_client_id,
    )
    handler = type(
        "CallbackHandler",
        (_CallbackHandler,),
        {
            "result_queue": result_queue,
            "expected_state": pending.state,
            "returning_client_id": returning_client_id,
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    if port == 0:
        pending = create_authorization(
            host_id=host_id,
            port=server.server_address[1],
            state=pending.state,
            nonce=pending.nonce,
            verifier=pending.verifier,
            returning_client_id=returning_client_id,
        )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        print("Open this URL in your system browser:\n")
        print(pending.authorization_url)
        print("\nWaiting for the loopback callback…", flush=True)
        if open_browser:
            webbrowser.open(pending.authorization_url)
        try:
            outcome = result_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"no callback received within {timeout} seconds") from exc
        match outcome:
            case CallbackFailure(message=message):
                raise RuntimeError(message)
            case CallbackResult():
                return pending, outcome
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _safe_summary(credentials: dict, claims: dict, models: list[dict[str, str]]) -> dict:
    return {
        "issuer": claims.get("iss"),
        "subject_present": bool(claims.get("sub")),
        "email": claims.get("email"),
        "client_id": credentials.get("client_id"),
        "scope": credentials.get("scope"),
        "expires_in": credentials.get("expires_in"),
        "earliest_refresh_at": credentials.get("earliest_refresh_at"),
        "model_count": len(models),
        "models": models,
    }


def login(args: argparse.Namespace) -> int:
    host_id = load_or_create_host_id(args.host_id_file)
    pending, callback = wait_for_callback(
        host_id=host_id,
        port=args.port,
        timeout=args.timeout,
        open_browser=not args.no_browser,
        returning_client_id=args.client_id,
    )
    token_response = exchange_code(pending, callback)
    scopes = require_plan_scope(str(token_response.get("scope") or ""))
    claims = validate_id_token(token_response, client_id=callback.client_id, nonce=pending.nonce)
    now = int(time.time())
    credentials = {
        "email": claims.get("email"),
        "issuer": claims["iss"],
        "subject": claims["sub"],
        "client_id": callback.client_id,
        "ext_agent_host_id": host_id,
        "id_token": token_response["id_token"],
        "access_token": token_response["access_token"],
        "refresh_token": token_response.get("refresh_token"),
        "token_type": token_response.get("token_type"),
        "expires_in": token_response.get("expires_in"),
        "earliest_refresh_at": token_response.get("earliest_refresh_at"),
        "scopes": sorted(scopes),
        "saved_at": now,
    }
    write_credentials(args.output, credentials)
    models = list_models(credentials["access_token"])
    print("\nValidated credential summary (tokens redacted):")
    print(json.dumps(_safe_summary({**token_response, "client_id": callback.client_id}, claims, models), indent=2))
    print(f"\nProtected credential file written to {args.output} (mode 0600).")
    if args.prompt:
        if not models and not args.model:
            raise RuntimeError("the account model catalog returned no visible models")
        model = args.model or models[0]["slug"]
        result = run_inference(credentials["access_token"], model=model, prompt=args.prompt)
        print(f"\nInference model: {model}")
        print(result.text)
        print(f"\nUsage: input={result.input_tokens}, output={result.output_tokens}")
    return 0


def dry_run(args: argparse.Namespace) -> int:
    host_id = load_or_create_host_id(args.host_id_file)
    pending = create_authorization(host_id=host_id, port=args.port)
    parsed = urlparse(pending.authorization_url)
    params = parse_qs(parsed.query)
    safe = {
        key: ("<redacted>" if key in {"state", "nonce", "code_challenge"} else values[0])
        for key, values in params.items()
    }
    print(json.dumps({"endpoint": f"{parsed.scheme}://{parsed.netloc}{parsed.path}", "params": safe}, indent=2))
    return 0


def probe_saved(args: argparse.Namespace) -> int:
    credentials = json.loads(args.credentials.read_text(encoding="utf-8"))
    access_token = credentials.get("access_token")
    if not access_token:
        raise ValueError("credential file has no access_token")
    models = list_models(access_token)
    if not models and not args.model:
        raise RuntimeError("the account model catalog returned no visible models")
    model = args.model or models[0]["slug"]
    result = run_tool_probe(access_token, model=model)
    validate_tool_probe(result)
    print(
        json.dumps(
            {
                "model": model,
                "completed": result.completed,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in result.tool_calls
                ],
                "usage": {
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Securo SIWC OSS protocol spike")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dry = subparsers.add_parser("dry-run", help="print a redacted authorization request")
    dry.add_argument("--port", type=int, default=1455)
    dry.add_argument("--host-id-file", type=Path, default=SPIKE_STATE_DIR / "host-id")
    dry.set_defaults(func=dry_run)

    auth = subparsers.add_parser("login", help="perform live dynamic registration")
    auth.add_argument("--host-id-file", type=Path, default=SPIKE_STATE_DIR / "host-id")
    auth.add_argument("--output", type=Path, default=SPIKE_STATE_DIR / "credentials.json")
    auth.add_argument("--timeout", type=int, default=300)
    auth.add_argument(
        "--port",
        type=int,
        default=0,
        help="fixed loopback port; useful with an SSH -L tunnel (default: choose a free port)",
    )
    auth.add_argument("--no-browser", action="store_true")
    auth.add_argument("--client-id", help="issued client ID for returning authorization")
    auth.add_argument("--prompt", default="Say exactly: Securo SIWC works.")
    auth.add_argument("--model")
    auth.set_defaults(func=login)

    probe = subparsers.add_parser(
        "probe-saved", help="run a live function-call probe with saved credentials"
    )
    probe.add_argument(
        "--credentials", type=Path, default=SPIKE_STATE_DIR / "credentials.json"
    )
    probe.add_argument("--model")
    probe.set_defaults(func=probe_saved)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (ValueError, RuntimeError, TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
