"""Command-line entry point; secret input never enters argv or environment."""
from __future__ import annotations

import argparse
import getpass
import queue
import secrets
import sys
import uuid
import webbrowser

from .backend import PairingClient
from .loopback import CallbackFailure, CallbackResult, LoopbackListener
from .openai import OpenAIClient, create_authorization
from .security import safe_exception_message

MAX_SECRET_INPUT = 512


def _secret(stdin: bool) -> str:
    if stdin:
        raw = sys.stdin.buffer.readline(MAX_SECRET_INPUT + 1)
        if len(raw) > MAX_SECRET_INPUT or not raw.endswith(b"\n"):
            raise ValueError("pairing secret input is invalid")
        try:
            return raw.rstrip(b"\r\n").decode("ascii")
        except UnicodeDecodeError:
            raise ValueError("pairing secret input is invalid") from None
    return getpass.getpass("Pairing secret: ")


def run(args: argparse.Namespace) -> int:
    secret = _secret(args.secret_stdin)
    outcomes: queue.Queue[CallbackResult | CallbackFailure] = queue.Queue(maxsize=1)
    authorization = None
    listener: LoopbackListener | None = None
    try:
        attempt_id = uuid.uuid4()
        with PairingClient(
            args.server, args.pairing_id, secret,
            allow_http_loopback=args.allow_http_loopback,
        ) as backend:
            bootstrap = backend.bootstrap()
            host_id = bootstrap.get("ext_agent_host_id")
            returning_client_id = bootstrap.get("returning_client_id")
            if not isinstance(host_id, str) or (
                returning_client_id is not None and not isinstance(returning_client_id, str)
            ):
                raise RuntimeError("Securo bootstrap response is invalid")
            state = secrets.token_urlsafe(32)
            listener = LoopbackListener(outcomes)
            # Bind first so the registered redirect uses the exact socket, but
            # reject every callback until the completed authorization is armed.
            listener.start()
            authorization = create_authorization(
                listener.port, host_id, returning_client_id, state=state
            )
            backend.register_attempt(
                attempt_id, authorization.nonce, authorization.redirect_uri
            )
            listener.arm(state, returning_client_id)
            print("Opening the OpenAI authorization page in your browser.")
            if args.no_browser:
                print(authorization.url)
            else:
                webbrowser.open(authorization.url)
            try:
                outcome = outcomes.get(timeout=args.timeout)
            except queue.Empty:
                raise RuntimeError("authorization timed out") from None
            if isinstance(outcome, CallbackFailure):
                raise RuntimeError(f"authorization ended: {outcome.code}")
            oauth = OpenAIClient()
            token_response = None
            try:
                token_response = oauth.exchange(authorization, outcome.code, outcome.client_id)
                backend.complete(attempt_id, outcome.client_id, token_response)
            finally:
                oauth.close()
                if token_response is not None:
                    token_response.clear()
        print("ChatGPT authorization transferred to Securo.")
        return 0
    finally:
        secret = ""
        if listener is not None:
            listener.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pair Securo with Sign in with ChatGPT")
    parser.add_argument("pairing_id", type=uuid.UUID, help="public pairing ID shown by Securo")
    parser.add_argument("--server", required=True, help="Securo HTTPS origin")
    parser.add_argument("--secret-stdin", action="store_true", help="read the secret from one stdin line")
    parser.add_argument("--no-browser", action="store_true", help="print rather than open the authorization URL")
    parser.add_argument("--timeout", type=int, default=300, choices=range(30, 901), metavar="SECONDS")
    parser.add_argument(
        "--allow-http-loopback", action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (ValueError, RuntimeError) as exc:
        # All protocol exceptions are deliberately sanitized before this boundary.
        print(
            "error: " + safe_exception_message(exc, "operation failed safely"),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
