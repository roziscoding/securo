"""Loopback callback listener with first-terminal-result semantics."""
from __future__ import annotations

import hmac
import queue
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit


MAX_CONCURRENT_HANDLERS = 8
SOCKET_READ_TIMEOUT_SECONDS = 1.0
SHUTDOWN_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class CallbackResult:
    code: str
    client_id: str


@dataclass(frozen=True)
class CallbackFailure:
    code: str


Outcome = CallbackResult | CallbackFailure


class _Server(HTTPServer):
    allow_reuse_address = False
    request_queue_size = MAX_CONCURRENT_HANDLERS + 16

    def __init__(self, *args, **kwargs):
        self._lifecycle_lock = threading.Lock()
        self._accepting = True
        self._active_sockets: set[socket.socket] = set()
        self._handler_threads: set[threading.Thread] = set()
        self._handler_slots = threading.BoundedSemaphore(MAX_CONCURRENT_HANDLERS)
        super().__init__(*args, **kwargs)

    def get_request(self):
        request, client_address = super().get_request()
        request.settimeout(SOCKET_READ_TIMEOUT_SECONDS)
        with self._lifecycle_lock:
            if not self._accepting:
                request.close()
                raise OSError("server is closing")
            self._active_sockets.add(request)
        return request, client_address

    def process_request(self, request, client_address) -> None:
        if not self._handler_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        thread = threading.Thread(
            target=self._process_request_thread,
            args=(request, client_address),
            name="securo-loopback-handler",
            daemon=False,
        )
        with self._lifecycle_lock:
            accepting = self._accepting
            if accepting:
                self._handler_threads.add(thread)
                thread.start()
        if not accepting:
            self._handler_slots.release()
            self.shutdown_request(request)

    def _process_request_thread(self, request, client_address) -> None:
        try:
            self.finish_request(request, client_address)
        except (TimeoutError, socket.timeout, BrokenPipeError, ConnectionResetError, OSError):
            pass
        except BaseException:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)
            with self._lifecycle_lock:
                self._handler_threads.discard(threading.current_thread())
            self._handler_slots.release()

    def close_request(self, request) -> None:
        with self._lifecycle_lock:
            self._active_sockets.discard(request)
        try:
            super().close_request(request)
        except OSError:
            pass

    def handle_error(self, request, client_address) -> None:
        return

    def stop_accepting_and_close_clients(self) -> tuple[threading.Thread, ...]:
        with self._lifecycle_lock:
            self._accepting = False
            requests = tuple(self._active_sockets)
            threads = tuple(self._handler_threads)
        for request in requests:
            try:
                request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                request.close()
            except OSError:
                pass
        return threads

    def handler_threads(self) -> tuple[threading.Thread, ...]:
        with self._lifecycle_lock:
            return tuple(self._handler_threads)


class LoopbackListener:
    def __init__(self, outcomes: queue.Queue[Outcome]):
        self._outcomes = outcomes
        self._configuration: tuple[str, str | None] | None = None
        self._armed_once = False
        self._configuration_lock = threading.Lock()
        self._server: _Server | None = None
        self._thread: threading.Thread | None = None
        self._close_lock = threading.Lock()

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("listener is not started")
        return int(self._server.server_address[1])

    def arm(self, state: str, returning_client_id: str | None) -> None:
        if not state or len(state) > 256 or not state.isascii():
            raise ValueError("OAuth state is invalid")
        with self._configuration_lock:
            if self._armed_once:
                raise RuntimeError("listener is already armed")
            self._configuration = (state, returning_client_id)
            self._armed_once = True

    def start(self) -> None:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                with owner._configuration_lock:
                    configuration = owner._configuration
                if configuration is None:
                    self.send_error(503)
                    return
                expected_state, returning_client_id = configuration
                if (
                    len(self.path) > 16_384
                    or len(self.headers) > 32
                    or sum(len(key) + len(value) for key, value in self.headers.items()) > 16_384
                    or len(self.headers.get_all("Host", [])) != 1
                ):
                    self.send_error(431)
                    return
                parsed = urlsplit(self.path)
                if parsed.path != "/auth/callback" or parsed.fragment:
                    self.send_error(404)
                    return
                try:
                    params = parse_qs(
                        parsed.query,
                        keep_blank_values=True,
                        strict_parsing=False,
                        max_num_fields=8,
                    )
                except ValueError:
                    self.send_error(400)
                    return
                if set(params) - {
                    "state", "code", "client_id", "scope", "error", "error_description"
                }:
                    self.send_error(400)
                    return
                states = params.get("state", [])
                scopes = params.get("scope", [])
                if (
                    len(states) != 1
                    or not hmac.compare_digest(states[0], expected_state)
                    or len(scopes) > 1
                ):
                    self.send_error(400)
                    return
                outcome: Outcome
                errors = params.get("error", [])
                codes = params.get("code", [])
                clients = params.get("client_id", [])
                if len(errors) == 1 and not codes:
                    outcome = CallbackFailure(errors[0][:128] or "authorization_denied")
                elif len(codes) == 1 and not errors:
                    if returning_client_id is None:
                        if len(clients) != 1 or clients[0] == "dynamic_agent_client":
                            self.send_error(400)
                            return
                        client_id = clients[0]
                    else:
                        if clients and (len(clients) != 1 or clients[0] != returning_client_id):
                            self.send_error(400)
                            return
                        client_id = returning_client_id
                    if not code_is_safe(codes[0]) or not client_id_is_safe(client_id):
                        self.send_error(400)
                        return
                    outcome = CallbackResult(codes[0], client_id)
                else:
                    self.send_error(400)
                    return
                with owner._configuration_lock:
                    if owner._configuration != configuration:
                        self.send_error(409)
                        return
                    owner._configuration = None
                    try:
                        owner._outcomes.put_nowait(outcome)
                    except queue.Full:
                        self.send_error(409)
                        return
                body = b"<!doctype html><meta charset=utf-8><meta http-equiv=Content-Security-Policy content=\"default-src 'none'; style-src 'unsafe-inline'\"><title>Securo</title><h1>Authorization received</h1><p>You may close this tab.</p>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'; base-uri 'none'")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = _Server(("127.0.0.1", 0), Handler, bind_and_activate=True)
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="securo-loopback-server",
            daemon=False,
        )
        self._thread.start()

    def close(self) -> None:
        with self._close_lock:
            server = self._server
            server_thread = self._thread
            if server is None:
                return
            deadline = time.monotonic() + SHUTDOWN_TIMEOUT_SECONDS
            handler_threads = server.stop_accepting_and_close_clients()
            server.shutdown()
            server.server_close()
            for thread in handler_threads:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if server_thread is not None:
                server_thread.join(timeout=max(0.0, deadline - time.monotonic()))
            if any(thread.is_alive() for thread in handler_threads) or (
                server_thread is not None and server_thread.is_alive()
            ):
                raise RuntimeError("loopback listener did not shut down cleanly")
            self._server = None
            self._thread = None

    def _handler_threads_for_test(self) -> tuple[threading.Thread, ...]:
        server = self._server
        return () if server is None else server.handler_threads()


def code_is_safe(value: str) -> bool:
    return 0 < len(value) <= 8192 and all(ord(char) >= 32 and ord(char) != 127 for char in value)


def client_id_is_safe(value: str) -> bool:
    return 0 < len(value) <= 255 and all(32 < ord(char) < 127 for char in value)
