from __future__ import annotations

# The standalone helper intentionally is not installed into the backend venv.
# This integration test adds its source tree before importing either app.
# ruff: noqa: E402

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import parse_qs, urlencode, urlsplit
from unittest.mock import patch

import httpx
import pytest
from sqlalchemy import select, update

_HELPER_SRC = Path(__file__).parents[3] / "tools" / "securo-siwc" / "src"
_ROOT = Path(__file__).parents[3]
if str(_HELPER_SRC) not in sys.path:
    sys.path.insert(0, str(_HELPER_SRC))

from securo_siwc.backend import PairingClient  # ty: ignore[unresolved-import]
from securo_siwc.loopback import CallbackResult, LoopbackListener  # ty: ignore[unresolved-import]
from securo_siwc.openai import (  # ty: ignore[unresolved-import]
    OpenAIClient,
    create_authorization,
    validate_token_response as helper_validate_token_response,
)

from app.agents.models.agent import Agent, AgentTool
from app.agents.models.connection import LlmConnection
from app.agents.models.conversation import Conversation, Message
from app.agents.models.oauth_credential import OpenAIOAuthCredential
from app.agents.models.oauth_pairing import OpenAIOAuthPairing
from app.agents.providers.base import ToolDefinition
from app.agents.providers import openai_chatgpt
from app.agents.runtime.executor import AgentExecutor
from app.agents.services import oauth_pairing, oauth_refresh, openai_oauth
from app.agents.services.oauth_credentials import decrypt_bundle
from app.core.auth import current_active_user
from app.core.database import get_async_session
from app.main import app
from app.models.user import User
from app.models.workspace import Workspace, WorkspaceMember
from tests.integration.fake_openai import FakeOpenAI


class _ProviderBridgeHandler(BaseHTTPRequestHandler):
    fake: FakeOpenAI

    def log_message(self, format: str, *_args: Any) -> None:  # noqa: A002
        del format

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        if self.path == "/authorize":
            callback = self.fake.authorize(**payload)
            result: dict[str, object] = {"callback": callback}
        elif self.path == "/request":
            request = httpx.Request(
                payload["method"],
                payload["url"],
                headers=payload["headers"],
                content=base64.b64decode(payload["content"], validate=True),
            )
            response = self.fake.handle_async(request)
            result = {
                "status": response.status_code,
                "headers": list(response.headers.multi_items()),
                "content": base64.b64encode(response.content).decode("ascii"),
            }
        else:
            self.send_error(404)
            return
        encoded = json.dumps(result, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class _StreamingSyncTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport) -> None:
        self.inner = inner

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self.inner.handle_request(request)
        content = response.read()
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            stream=httpx.ByteStream(content),
            request=request,
            extensions=response.extensions,
        )


class _ASGISyncTransport(httpx.BaseTransport):
    """Run sync helper requests on pytest's owning ASGI/event loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        async def send() -> tuple[int, list[tuple[bytes, bytes]], bytes]:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
                response = await client.request(
                    request.method,
                    str(request.url),
                    headers=request.headers,
                    content=request.content,
                )
                return response.status_code, response.headers.raw, response.content

        status, headers, content = asyncio.run_coroutine_threadsafe(send(), self.loop).result(20)
        return httpx.Response(
            status,
            headers=headers,
            stream=httpx.ByteStream(content),
            request=request,
        )


class _MCP:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def discover(self, **_kwargs):
        return []

    def to_provider_tools(self, _handles, *, allowed):
        assert ("integration", "lookup") in allowed
        return [
            ToolDefinition(
                name="integration__lookup",
                description="Deterministic integration lookup",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            )
        ]

    async def call(self, *, wire_name: str, arguments: dict, **_kwargs):
        self.calls.append((wire_name, arguments))
        return {"ok": True, "data": {"value": 42}, "text": None}


def _successful_helper_subprocesses(tmp_path: Path) -> tuple[list[dict[str, object]], set[str]]:
    pairing_secrets = {
        str(uuid.uuid4()): f"pairing-{uuid.uuid4().hex}-{uuid.uuid4().hex}",
        str(uuid.uuid4()): f"pairing-{uuid.uuid4().hex}-{uuid.uuid4().hex}",
    }
    state: dict[str, object] = {
        "returning_client_id": None,
        "attempts": 0,
        "completions": 0,
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *_args: Any) -> None:  # noqa: A002
            del format
            return

        def _reply(self, status: int, payload: dict[str, object]) -> None:
            encoded = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _pairing(self) -> tuple[str, str, str] | None:
            parts = self.path.split("/")
            if len(parts) != 7 or parts[1:5] != ["api", "agents", "chatgpt", "pairings"]:
                return None
            pairing_id, action = parts[5], parts[6]
            secret = pairing_secrets.get(pairing_id)
            if secret is None or self.headers.get("Authorization") != f"SecuroPairing {secret}":
                return None
            return pairing_id, action, secret

        def do_GET(self):  # noqa: N802
            pairing = self._pairing()
            if pairing is None or pairing[1] != "bootstrap":
                self._reply(404, {"error": "not_found"})
                return
            self._reply(200, {
                "ext_agent_host_id": f"urn:uuid:{uuid.uuid4()}",
                "returning_client_id": state["returning_client_id"],
            })

        def do_POST(self):  # noqa: N802
            pairing = self._pairing()
            if pairing is None:
                self._reply(404, {"error": "not_found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            assert 0 < length <= 256 * 1024
            payload = json.loads(self.rfile.read(length))
            if pairing[1] == "attempt":
                assert set(payload) == {"attempt_id", "nonce", "redirect_uri"}
                assert payload["redirect_uri"].startswith("http://127.0.0.1:")
                state["attempts"] = cast(int, state["attempts"]) + 1
                self._reply(200, {})
                return
            if pairing[1] == "complete":
                assert set(payload) == {"attempt_id", "issued_client_id", "token_response"}
                assert isinstance(payload["token_response"], dict)
                state["returning_client_id"] = payload["issued_client_id"]
                state["completions"] = cast(int, state["completions"]) + 1
                self._reply(200, {"state": "completed"})
                return
            self._reply(404, {"error": "not_found"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    artifacts: list[dict[str, object]] = []
    sentinels = set(pairing_secrets.values())
    try:
        for index, (pairing_id, secret) in enumerate(pairing_secrets.items()):
            artifact_path = tmp_path / f"helper-child-{index}.json"
            manifest_path = tmp_path / f"helper-child-{index}-manifest.json"
            helper_env = {
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": os.pathsep.join((str(_ROOT / "backend"), str(_HELPER_SRC))),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            argv = [
                sys.executable,
                "-m",
                "tests.integration.helper_cli_runner",
                "--artifacts",
                str(artifact_path),
                "--manifest",
                str(manifest_path),
                "--",
                pairing_id,
                "--server",
                f"http://127.0.0.1:{server.server_port}",
                "--secret-stdin",
                "--allow-http-loopback",
                "--timeout",
                "30",
            ]
            process = subprocess.run(
                argv,
                input=secret + "\n",
                text=True,
                capture_output=True,
                check=False,
                env=helper_env,
                cwd=_ROOT,
                timeout=40,
            )
            assert process.returncode == 0, (
                process.stdout
                + process.stderr
                + (artifact_path.read_text(encoding="utf-8") if artifact_path.exists() else "")
            )
            child_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            for values in child_manifest.values():
                sentinels.update(values)
            artifacts.append({
                "argv": process.args,
                "env": {key: helper_env[key] for key in sorted(helper_env)},
                "stdout": process.stdout,
                "stderr": process.stderr,
                "callbacks": json.loads(artifact_path.read_text(encoding="utf-8")),
                "returncode": process.returncode,
            })
        assert state["attempts"] == 2
        assert state["completions"] == 2
        assert isinstance(state["returning_client_id"], str)
        return artifacts, sentinels
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


async def _run_helper_flow(
    *,
    fake: FakeOpenAI,
    loop: asyncio.AbstractEventLoop,
    pairing: dict,
    sentinels: set[str],
    runtime_artifacts: list[object],
) -> tuple[str, dict]:
    backend_transport = _ASGISyncTransport(loop)
    openai_transport = _StreamingSyncTransport(fake.sync_transport)

    def work() -> tuple[str, dict]:
        with (
            patch(
                "securo_siwc.backend._new_http_client",
                lambda: httpx.Client(
                    transport=backend_transport, trust_env=False, follow_redirects=False
                ),
            ),
            patch(
                "securo_siwc.openai._new_http_client",
                lambda: httpx.Client(
                    transport=openai_transport, trust_env=False, follow_redirects=False
                ),
            ),
        ):
            with PairingClient(
                "http://127.0.0.1",
                uuid.UUID(pairing["pairing_id"]),
                pairing["secret"],
                allow_http_loopback=True,
            ) as backend:
                bootstrap = backend.bootstrap()
                outcomes: queue.Queue = queue.Queue(maxsize=1)
                listener = LoopbackListener(outcomes)
                listener.start()
                try:
                    literal_origin = f"http://127.0.0.1:{listener.port}"
                    prearm = httpx.get(
                        literal_origin + "/auth/callback?state=prearm&code=poison",
                        timeout=2,
                    )
                    assert prearm.status_code == 503
                    runtime_artifacts.append(("prearm", prearm.status_code, prearm.text))
                    authorization = create_authorization(
                        listener.port,
                        bootstrap["ext_agent_host_id"],
                        bootstrap.get("returning_client_id"),
                        state=f"assembled-{uuid.uuid4().hex}",
                    )
                    assert authorization.redirect_uri.startswith(literal_origin + "/")
                    attempt = uuid.uuid4()
                    backend.register_attempt(
                        attempt, authorization.nonce, authorization.redirect_uri
                    )
                    listener.arm(authorization.state, bootstrap.get("returning_client_id"))
                    callback = fake.authorize(
                        client_id=(
                            bootstrap.get("returning_client_id")
                            or "dynamic_agent_client"
                        ),
                        redirect_uri=authorization.redirect_uri,
                        nonce=authorization.nonce,
                        code_challenge=parse_qs(urlsplit(authorization.url).query)[
                            "code_challenge"
                        ][0],
                        state=authorization.state,
                    )
                    callback_values = parse_qs(urlsplit(callback).query)
                    sentinels.update({
                        pairing["secret"],
                        authorization.state,
                        authorization.nonce,
                        authorization.verifier,
                        callback_values["code"][0],
                        callback_values["client_id"][0],
                    })
                    accepted = httpx.get(callback, timeout=2)
                    assert accepted.status_code == 200
                    runtime_artifacts.append(("accepted", accepted.status_code, accepted.text))
                    outcome = outcomes.get(timeout=2)
                    assert isinstance(outcome, CallbackResult)
                    replay = httpx.get(callback, timeout=2)
                    assert replay.status_code == 503
                    runtime_artifacts.append(("replay", replay.status_code, replay.text))
                    issued_client_id = outcome.client_id
                    oauth = OpenAIClient()
                    try:
                        transferred = oauth.exchange(
                            authorization, outcome.code, issued_client_id
                        )
                    finally:
                        oauth.close()
                    result = backend.complete(attempt, issued_client_id, transferred)
                    return issued_client_id, result
                finally:
                    listener.close()
                    assert listener._handler_threads_for_test() == ()
                    runtime_artifacts.append(("listener_threads", listener._handler_threads_for_test()))

    return await asyncio.to_thread(work)


@pytest.mark.asyncio
async def test_helper_subprocess_uses_real_asgi_listener_and_reconnects(
    postgres_sessions, monkeypatch, tmp_path
):
    fake = FakeOpenAI()
    _ProviderBridgeHandler.fake = fake
    bridge = ThreadingHTTPServer(("127.0.0.1", 0), _ProviderBridgeHandler)
    bridge_thread = threading.Thread(target=bridge.serve_forever)
    bridge_thread.start()

    bind = postgres_sessions.kw["bind"]
    schema = bind.get_execution_options()["schema_translate_map"][None]
    database_url = os.environ["POSTGRES_TEST_URL"]
    user_id = uuid.uuid4()
    connection_id = uuid.uuid4()
    async with postgres_sessions() as setup:
        setup.add(User(
            id=user_id,
            email=f"helper-real-{user_id}@example.test",
            hashed_password="unused",
            is_active=True,
            is_superuser=False,
            is_verified=True,
            preferences={},
        ))
        await setup.flush()
        setup.add(LlmConnection(
            id=connection_id,
            user_id=user_id,
            name="Real helper subprocess",
            kind="openai_chatgpt",
            base_url=None,
            api_key_encrypted=None,
            default_model="gpt-integration",
            extra={},
            is_default=False,
        ))
        await setup.commit()

    import socket

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    server_port = probe.getsockname()[1]
    probe.close()
    bridge_origin = f"http://127.0.0.1:{bridge.server_port}"
    server_origin = f"http://127.0.0.1:{server_port}"
    server_env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "SECURO_TEST_DATABASE_URL": database_url,
        "SECURO_TEST_DATABASE_SCHEMA": schema,
        "SECURO_TEST_PROVIDER_BRIDGE": bridge_origin,
        "DATABASE_URL": database_url,
    }
    server_log = tmp_path / "uvicorn.log"
    with server_log.open("w+", encoding="utf-8") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "tests.integration.real_asgi_server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(server_port),
                "--log-level",
                "info",
                "--lifespan",
                "off",
            ],
            cwd=_ROOT / "backend",
            env=server_env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            ready = False
            with httpx.Client(trust_env=False, timeout=0.2) as readiness:
                for _ in range(2400):
                    if server.poll() is not None:
                        break
                    try:
                        response = readiness.get(server_origin + "/api/docs")
                        if response.status_code == 200:
                            ready = True
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(0.05)
            if not ready:
                pytest.fail(
                    "real ASGI server did not start: "
                    + server_log.read_text(encoding="utf-8")
                )
            assert server.poll() is None, server_log.read_text(encoding="utf-8")

            async def run_pairing(index: int) -> tuple[str, dict[str, object]]:
                async with postgres_sessions() as session:
                    started = await oauth_pairing.start_pairing(
                        session, connection_id, user_id
                    )
                with httpx.Client(trust_env=False, timeout=5) as backend_probe:
                    wrong = backend_probe.get(
                        f"{server_origin}/api/agents/chatgpt/pairings/"
                        f"{started.id}/bootstrap",
                        headers={"authorization": "SecuroPairing definitely-wrong"},
                    )
                assert wrong.status_code == 404
                artifact = tmp_path / f"real-helper-{index}.json"
                manifest = tmp_path / f"real-helper-{index}-manifest.json"
                helper_env = {
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": os.pathsep.join((str(_ROOT / "backend"), str(_HELPER_SRC))),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
                process = await asyncio.to_thread(
                    subprocess.run,
                    [
                        sys.executable,
                        "-m",
                        "tests.integration.helper_cli_runner",
                        "--artifacts",
                        str(artifact),
                        "--manifest",
                        str(manifest),
                        "--provider-bridge",
                        bridge_origin,
                        "--",
                        str(started.id),
                        "--server",
                        server_origin,
                        "--secret-stdin",
                        "--allow-http-loopback",
                        "--timeout",
                        "30",
                    ],
                    input=started.secret + "\n",
                    text=True,
                    capture_output=True,
                    check=False,
                    env=helper_env,
                    cwd=_ROOT,
                    timeout=40,
                )
                assert isinstance(process.stdout, str)
                assert isinstance(process.stderr, str)
                assert process.returncode == 0, (
                    process.stdout + process.stderr + server_log.read_text(encoding="utf-8")
                )
                captured = json.loads(artifact.read_text(encoding="utf-8"))
                callback = captured["callbacks"][0]
                assert callback["loopback_host"] == "127.0.0.1"
                assert callback["status"] == 200
                async with postgres_sessions() as verify:
                    credential = await verify.scalar(
                        select(OpenAIOAuthCredential).where(
                            OpenAIOAuthCredential.connection_id == connection_id
                        )
                    )
                    assert credential is not None
                    return credential.issued_client_id, captured

            first_client_id, _first = await run_pairing(1)
            monkeypatch.setattr(openai_oauth, "_new_oauth_transport", lambda: fake.async_transport)
            async with postgres_sessions() as revoke_session:
                await oauth_refresh.revoke_credential(
                    revoke_session, connection_id, user_id
                )
            second_client_id, _second = await run_pairing(2)
            assert second_client_id == first_client_id
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            bridge.shutdown()
            bridge.server_close()
            bridge_thread.join(timeout=5)
            assert not bridge_thread.is_alive()


@pytest.mark.asyncio
async def test_assembled_pair_refresh_revoke_reconnect_executor_lifecycle(
    postgres_sessions, monkeypatch, caplog, tmp_path
):
    fake = FakeOpenAI()
    caplog.set_level(logging.INFO, logger="app.agents.siwc")
    sentinels: set[str] = set()
    runtime_artifacts: list[object] = []
    monkeypatch.setattr(openai_oauth, "_new_oauth_transport", lambda: fake.async_transport)
    monkeypatch.setattr(openai_chatgpt, "_new_responses_transport", lambda: fake.async_transport)
    monkeypatch.setattr(oauth_refresh, "_resource_session_maker", postgres_sessions)
    monkeypatch.setattr(oauth_pairing, "_new_start_cleanup_session", postgres_sessions)

    user_id = uuid.uuid4()
    workspace_id = uuid.uuid4()
    async with postgres_sessions() as setup:
        user = User(
            id=user_id,
            email=f"assembled-{user_id}@example.test",
            hashed_password="unused",
            is_active=True,
            is_superuser=False,
            is_verified=True,
            preferences={},
        )
        workspace = Workspace(
            id=workspace_id,
            name="Integration",
            kind="personal",
            created_by_user_id=user_id,
            default_currency="USD",
            locale="en",
        )
        setup.add(user)
        await setup.flush()
        setup.add(workspace)
        await setup.flush()
        setup.add(
            WorkspaceMember(
                id=uuid.uuid4(), workspace_id=workspace_id, user_id=user_id, role="owner"
            )
        )
        await setup.commit()

    async def db_override():
        async with postgres_sessions() as session:
            yield session

    async def user_override():
        async with postgres_sessions() as session:
            loaded = await session.get(User, user_id)
            assert loaded is not None
            return loaded

    app.dependency_overrides[get_async_session] = db_override
    app.dependency_overrides[current_active_user] = user_override
    loop = asyncio.get_running_loop()
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            created = await api.post(
                "/api/agents/connections",
                json={
                    "name": "Assembled ChatGPT",
                    "kind": "openai_chatgpt",
                    "default_model": "gpt-integration",
                    "is_default": True,
                },
            )
            assert created.status_code == 201, created.text
            connection_id = uuid.UUID(created.json()["id"])
            started = await api.post(
                f"/api/agents/connections/{connection_id}/chatgpt/pairings"
            )
            assert started.status_code == 201, started.text
            first_client_id, completed = await _run_helper_flow(
                fake=fake,
                loop=loop,
                pairing=started.json(),
                sentinels=sentinels,
                runtime_artifacts=runtime_artifacts,
            )
            assert completed["state"] == "completed"
            assert completed["subject"] in fake.sensitive_values_for_test()

            status = await api.get(f"/api/agents/connections/{connection_id}")
            assert status.status_code == 200
            assert status.json()["credential_status"] == "active"
            tested = await api.post(f"/api/agents/connections/{connection_id}/test")
            assert tested.status_code == 200
            assert tested.json()["ok"] is True
            assert tested.json()["models"] == [
                {"slug": "gpt-integration", "display_name": "GPT Integration"}
            ]

        async with postgres_sessions() as session:
            agent_id = uuid.uuid4()
            agent = Agent(
                id=agent_id,
                user_id=user_id,
                workspace_id=workspace_id,
                name="Assembled",
                description=None,
                system_prompt="",
                connection_id=connection_id,
                provider=None,
                model="gpt-integration",
                temperature=0.0,
                max_history_messages=10,
                auto_context=False,
                extra={},
            )
            conversation = Conversation(
                id=uuid.uuid4(),
                agent_id=agent.id,
                user_id=user_id,
                workspace_id=workspace_id,
                channel="web",
            )
            conversation_id = conversation.id
            session.add(agent)
            await session.flush()
            session.add(AgentTool(agent_id=agent.id, server="integration", tool_name="lookup"))
            session.add(conversation)
            await session.commit()
            # Exercise the assembled provider's one allowed pre-output auth
            # retry. The forced refresh must complete before any model event is
            # yielded, then the normal tool-call stream continues.
            fake.script(
                str(openai_chatgpt._RESPONSES_URL),
                httpx.Response(401, json={"error": "expired_access"}),
            )
            mcp = _MCP()
            events = [
                event
                async for event in AgentExecutor(mcp=cast(Any, mcp)).run(
                    session=session,
                    agent=agent,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    conversation_id=conversation_id,
                    user_message="Use the integration tool",
                )
            ]
            assert mcp.calls == [("integration__lookup", {"query": "integration"})]
            assert any(event.type == "tool_result" for event in events)
            assert "assembled integration complete" in "".join(
                event.text or "" for event in events if event.type == "text_delta"
            )
            messages = (
                await session.execute(
                    select(Message).where(Message.conversation_id == conversation_id)
                )
            ).scalars().all()
            assert any(message.role == "tool" and message.content == '{"value": 42}' for message in messages)
            assistant_usage = [
                (message.input_tokens, message.output_tokens)
                for message in messages
                if message.role == "assistant"
            ]
            assert assistant_usage == [(7, 3), (9, 4)]

        async with postgres_sessions() as session:
            race_conversation_ids = [uuid.uuid4(), uuid.uuid4()]
            for race_conversation_id in race_conversation_ids:
                session.add(Conversation(
                    id=race_conversation_id,
                    agent_id=agent_id,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    channel="web",
                ))
            await session.execute(
                update(OpenAIOAuthCredential)
                .where(OpenAIOAuthCredential.connection_id == connection_id)
                .values(
                    access_token_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
                    earliest_refresh_at=datetime.now(timezone.utc) - timedelta(seconds=1),
                )
            )
            await session.commit()
            credential = await session.scalar(
                select(OpenAIOAuthCredential).where(
                    OpenAIOAuthCredential.connection_id == connection_id
                )
            )
            assert credential is not None
            assert credential.token_bundle_encrypted is not None
            version_before_concurrency = credential.credential_version
            bundle_before_concurrency = credential.token_bundle_encrypted
            refreshed_at_before_concurrency = credential.last_refreshed_at
            refresh_token_before_concurrency = decrypt_bundle(
                bundle_before_concurrency
            ).refresh_token
        calls_before = fake.refresh_calls

        # Model two separate workers: each executor owns an independent DB
        # session and the test-only lock factory deliberately removes only the
        # process-local optimization. PostgreSQL's durable refresh marker is
        # therefore the sole single-flight authority under test.
        async def independent_local_lock(_connection_id):
            return asyncio.Lock()

        monkeypatch.setattr(oauth_refresh, "_local_lock", independent_local_lock)
        dispatch = fake.barrier(openai_oauth.TOKEN_ENDPOINT)

        async def execute_once(conversation_id: uuid.UUID):
            async with postgres_sessions() as session:
                loaded_agent = await session.get(Agent, agent_id)
                assert loaded_agent is not None
                mcp = _MCP()
                events = [
                    event
                    async for event in AgentExecutor(mcp=cast(Any, mcp)).run(
                        session=session,
                        agent=loaded_agent,
                        user_id=user_id,
                        workspace_id=workspace_id,
                        conversation_id=conversation_id,
                        user_message="Concurrent expired-access inference",
                    )
                ]
                return events, mcp.calls

        executor_tasks = [
            asyncio.create_task(execute_once(conversation_id))
            for conversation_id in race_conversation_ids
        ]
        await asyncio.wait_for(dispatch.reached.wait(), timeout=5)
        done_while_dispatch_blocked, _ = await asyncio.wait(
            executor_tasks, timeout=5, return_when=asyncio.FIRST_COMPLETED
        )
        assert len(done_while_dispatch_blocked) == 1
        dispatch.release()
        outcomes = await asyncio.gather(*executor_tasks)

        assert fake.refresh_calls - calls_before == 1
        assert dispatch.arrivals == 1
        successful = [
            (events, calls)
            for events, calls in outcomes
            if any(event.finish_reason == "stop" for event in events)
        ]
        failed = [
            (events, calls)
            for events, calls in outcomes
            if any(event.finish_reason == "error" for event in events)
        ]
        assert len(successful) == 1
        assert len(failed) == 1
        assert successful[0][1] == [
            ("integration__lookup", {"query": "integration"})
        ]
        assert any(event.type == "tool_result" for event in successful[0][0])
        assert failed[0][1] == []
        loser_errors = [
            (event.error_code, event.error_message)
            for event in failed[0][0]
            if event.type == "error"
        ]
        assert loser_errors == [
            ("unavailable", "LLM provider is unavailable.")
        ]
        assert [
            event.finish_reason for event in failed[0][0] if event.type == "done"
        ] == ["error"]
        # A still-live refresh intent is contention, not an abandoned attempt.
        # It remains retryable while the winner owns the unexpired lease.
        assert "LLM provider call failed (category=unavailable)" in caplog.text

        async with postgres_sessions() as session:
            winner = await session.scalar(
                select(OpenAIOAuthCredential).where(
                    OpenAIOAuthCredential.connection_id == connection_id
                )
            )
            assert winner is not None
            assert winner.status == "active"
            assert winner.credential_version == version_before_concurrency + 1
            assert winner.token_bundle_encrypted is not None
            assert winner.last_refreshed_at is not None
            assert winner.last_refreshed_at != refreshed_at_before_concurrency
            assert winner.refresh_attempt_id is None
            assert winner.refresh_attempt_started_at is None
            assert winner.refresh_restore_status is None
            assert winner.lifecycle_lease_token is None
            assert winner.lifecycle_lease_kind is None
            assert winner.lifecycle_lease_expires_at is None
            winning_bundle_encrypted = winner.token_bundle_encrypted
            winning_refreshed_at = winner.last_refreshed_at
            assert not hmac.compare_digest(
                winning_bundle_encrypted, bundle_before_concurrency
            )
            winning_bundle = decrypt_bundle(winning_bundle_encrypted)
            assert not hmac.compare_digest(
                winning_bundle.refresh_token, refresh_token_before_concurrency
            )

        assert fake.refresh_calls - calls_before == 1
        async with postgres_sessions() as session:
            authoritative = await session.scalar(
                select(OpenAIOAuthCredential).where(
                    OpenAIOAuthCredential.connection_id == connection_id
                )
            )
            assert authoritative is not None
            assert authoritative.status == "active"
            assert authoritative.credential_version == version_before_concurrency + 1
            assert authoritative.token_bundle_encrypted is not None
            assert hmac.compare_digest(
                authoritative.token_bundle_encrypted, winning_bundle_encrypted
            )
            assert authoritative.last_refreshed_at == winning_refreshed_at
            assert authoritative.refresh_attempt_id is None
            assert authoritative.refresh_attempt_started_at is None
            assert authoritative.refresh_restore_status is None
            assert authoritative.lifecycle_lease_token is None
            assert authoritative.lifecycle_lease_kind is None
            assert authoritative.lifecycle_lease_expires_at is None
            authoritative_bundle = decrypt_bundle(authoritative.token_bundle_encrypted)
            assert not hmac.compare_digest(
                authoritative_bundle.refresh_token, refresh_token_before_concurrency
            )

        # A refresh that has durably claimed its one-use token lineage must
        # beat a concurrent revoke before either operation reaches provider
        # output. The loser is the exact retryable race classification, not
        # an arbitrary exception accepted by the test.
        async with postgres_sessions() as expire_for_revoke_race:
            await expire_for_revoke_race.execute(
                update(OpenAIOAuthCredential)
                .where(OpenAIOAuthCredential.connection_id == connection_id)
                .values(
                    access_token_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
                    earliest_refresh_at=datetime.now(timezone.utc) - timedelta(seconds=1),
                )
            )
            await expire_for_revoke_race.commit()
        refresh_revoke_barrier = fake.barrier(openai_oauth.TOKEN_ENDPOINT)

        async def refresh_for_revoke_race():
            async with postgres_sessions() as refresh_session:
                return await oauth_refresh.refresh_credential(
                    refresh_session, connection_id, user_id, force=True
                )

        refresh_task = asyncio.create_task(refresh_for_revoke_race())
        await asyncio.wait_for(refresh_revoke_barrier.reached.wait(), timeout=5)
        try:
            async with postgres_sessions() as revoke_loser_session:
                with pytest.raises(openai_oauth.OAuthLifecycleError) as revoke_loser:
                    await oauth_refresh.revoke_credential(
                        revoke_loser_session, connection_id, user_id
                    )
            assert revoke_loser.value.code is openai_oauth.OAuthFailureCode.RACE_LOST
            assert revoke_loser.value.retryable is True
            assert revoke_loser.value.definitive is False
        finally:
            refresh_revoke_barrier.release()
        refreshed_access = await asyncio.wait_for(refresh_task, timeout=10)
        assert isinstance(refreshed_access, str) and refreshed_access
        async with postgres_sessions() as refreshed_state_session:
            refreshed_state = await refreshed_state_session.scalar(
                select(OpenAIOAuthCredential).where(
                    OpenAIOAuthCredential.connection_id == connection_id
                )
            )
            assert refreshed_state is not None
            assert refreshed_state.status == "active"
            assert refreshed_state.credential_version == version_before_concurrency + 2
            assert refreshed_state.token_bundle_encrypted is not None
            winning_bundle = decrypt_bundle(refreshed_state.token_bundle_encrypted)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            responses_before = fake.responses_calls
            revoke_journal_before = len(fake.journal)
            revoked = await api.post(f"/api/agents/connections/{connection_id}/revoke")
            assert revoked.status_code == 204, revoked.text
            revoke_entries = fake.journal[revoke_journal_before:]
            assert len(revoke_entries) == 1
            assert revoke_entries[0] == {
                "endpoint": "revoke",
                "request_id": revoke_entries[0]["request_id"],
                "body_sha256": hashlib.sha256(urlencode({
                    "token": winning_bundle.refresh_token,
                    "token_type_hint": "refresh_token",
                    "client_id": first_client_id,
                }).encode()).hexdigest(),
                "authorization_sha256": None,
                "client_id_sha256": hashlib.sha256(first_client_id.encode()).hexdigest(),
            }
            assert fake.handle_async(httpx.Request(
                "GET",
                openai_oauth.MODELS_ENDPOINT,
                headers={"authorization": f"Bearer {winning_bundle.access_token}"},
            )).status_code == 401
            async with postgres_sessions() as revoked_state_session:
                revoked_state = await revoked_state_session.scalar(
                    select(OpenAIOAuthCredential).where(
                        OpenAIOAuthCredential.connection_id == connection_id
                    )
                )
                assert revoked_state is not None
                assert revoked_state.status == "revoked"
                assert revoked_state.credential_version == version_before_concurrency + 3
                assert revoked_state.token_bundle_encrypted is None
                assert revoked_state.last_error_code is None
                assert revoked_state.refresh_attempt_id is None
                assert revoked_state.lifecycle_lease_token is None
            failed = await api.post(f"/api/agents/connections/{connection_id}/test")
            assert failed.status_code == 200
            assert failed.json()["ok"] is False
            assert failed.json()["auth_state"] == "revoked"
            assert fake.responses_calls == responses_before

            fallback_calls = 0

            def forbidden_api_key_fallback(_agent):
                nonlocal fallback_calls
                fallback_calls += 1
                raise AssertionError("explicit OAuth connection fell back to an API key")

            monkeypatch.setattr(
                "app.agents.runtime.executor._provider_for",
                forbidden_api_key_fallback,
            )
            async with postgres_sessions() as revoked_session:
                revoked_agent = await revoked_session.get(Agent, agent_id)
                assert revoked_agent is not None
                revoked_conversation = Conversation(
                    id=uuid.uuid4(),
                    agent_id=agent_id,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    channel="web",
                )
                revoked_session.add(revoked_conversation)
                await revoked_session.flush()
                revoked_mcp = _MCP()
                revoked_events = [
                    event
                    async for event in AgentExecutor(mcp=cast(Any, revoked_mcp)).run(
                        session=revoked_session,
                        agent=revoked_agent,
                        user_id=user_id,
                        workspace_id=workspace_id,
                        conversation_id=revoked_conversation.id,
                        user_message="This must fail closed after revoke",
                    )
                ]
            assert any(event.type == "error" for event in revoked_events)
            assert any(event.finish_reason == "error" for event in revoked_events)
            assert revoked_mcp.calls == []
            assert fake.responses_calls == responses_before
            assert fallback_calls == 0

            restarted = await api.post(
                f"/api/agents/connections/{connection_id}/chatgpt/pairings"
            )
            assert restarted.status_code == 201, restarted.text
            second_client_id, recompleted = await _run_helper_flow(
                fake=fake,
                loop=loop,
                pairing=restarted.json(),
                sentinels=sentinels,
                runtime_artifacts=runtime_artifacts,
            )
            assert second_client_id == first_client_id
            assert recompleted["state"] == "completed"
            retested = await api.post(f"/api/agents/connections/{connection_id}/test")
            assert retested.status_code == 200 and retested.json()["ok"] is True

            async with postgres_sessions() as reconnected_session:
                reconnected_agent = await reconnected_session.get(Agent, agent_id)
                assert reconnected_agent is not None
                reconnected_conversation = Conversation(
                    id=uuid.uuid4(),
                    agent_id=agent_id,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    channel="web",
                )
                reconnected_session.add(reconnected_conversation)
                await reconnected_session.flush()
                reconnected_mcp = _MCP()
                reconnected_before = fake.responses_calls
                reconnected_events = [
                    event
                    async for event in AgentExecutor(
                        mcp=cast(Any, reconnected_mcp)
                    ).run(
                        session=reconnected_session,
                        agent=reconnected_agent,
                        user_id=user_id,
                        workspace_id=workspace_id,
                        conversation_id=reconnected_conversation.id,
                        user_message="Infer with the tool after reconnect",
                    )
                ]
            assert reconnected_mcp.calls == [
                ("integration__lookup", {"query": "integration"})
            ]
            assert any(event.type == "tool_result" for event in reconnected_events)
            assert any(event.finish_reason == "stop" for event in reconnected_events)
            assert fake.responses_calls - reconnected_before == 2
            assert fallback_calls == 0

            # A provider can report a structured credential failure only after
            # emitting visible text. The executor must buffer that partial turn,
            # perform no refresh/replay, and fail without persisting or exposing it.
            partial_conversation = uuid.uuid4()
            async with postgres_sessions() as partial_setup:
                partial_setup.add(Conversation(
                    id=partial_conversation,
                    agent_id=agent_id,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    channel="web",
                ))
                await partial_setup.commit()
            partial_response_id = "response-partial-auth"
            partial_item_id = "message-partial-auth"
            fake.script(
                str(openai_chatgpt._RESPONSES_URL),
                httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=fake._wire([
                        {
                            "type": "response.created",
                            "response": {"id": partial_response_id},
                        },
                        {
                            "type": "response.output_text.delta",
                            "response_id": partial_response_id,
                            "output_index": 0,
                            "content_index": 0,
                            "item_id": partial_item_id,
                            "delta": "must remain buffered",
                        },
                        {
                            "type": "response.failed",
                            "response": {
                                "id": partial_response_id,
                                "status": "failed",
                                "error": {"code": "subscription_sharing_invalid_user"},
                            },
                        },
                    ]),
                ),
            )
            partial_responses_before = fake.responses_calls
            partial_journal_before = len(fake.journal)
            partial_refreshes_before = fake.refresh_calls
            async with postgres_sessions() as partial_session:
                partial_agent = await partial_session.get(Agent, agent_id)
                assert partial_agent is not None
                partial_events = [
                    event
                    async for event in AgentExecutor(mcp=cast(Any, _MCP())).run(
                        session=partial_session,
                        agent=partial_agent,
                        user_id=user_id,
                        workspace_id=workspace_id,
                        conversation_id=partial_conversation,
                        user_message="Fail safely after partial provider output",
                    )
                ]
            assert fake.responses_calls == partial_responses_before
            assert [
                entry["endpoint"] for entry in fake.journal[partial_journal_before:]
            ] == ["scripted"]
            assert fake.refresh_calls == partial_refreshes_before
            assert not any(event.type == "text_delta" for event in partial_events)
            assert [event.error_code for event in partial_events if event.type == "error"] == [
                "auth"
            ]
            assert [
                event.finish_reason for event in partial_events if event.type == "done"
            ] == ["error"]

            pending = await api.post(
                f"/api/agents/connections/{connection_id}/chatgpt/pairings"
            )
            assert pending.status_code == 201
            pending_pairing_id = uuid.UUID(pending.json()["pairing_id"])
            assert (await api.delete(f"/api/agents/connections/{connection_id}")).status_code == 204
            assert (await api.get(f"/api/agents/connections/{connection_id}")).status_code == 404
            assert (
                await api.get(
                    f"/api/agents/connections/{connection_id}/chatgpt/pairings/{pending_pairing_id}"
                )
            ).status_code == 404

        async with postgres_sessions() as session:
            assert await session.get(LlmConnection, connection_id) is None
            assert await session.scalar(
                select(OpenAIOAuthCredential).where(
                    OpenAIOAuthCredential.connection_id == connection_id
                )
            ) is None
            assert await session.get(OpenAIOAuthPairing, pending_pairing_id) is None

        # A separate assembled credential proves consume-then-response-lost
        # without making the primary revoke/reconnect lifecycle unrecoverable.
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as api:
            loss_created = await api.post(
                "/api/agents/connections",
                json={
                    "name": "Ambiguous Refresh",
                    "kind": "openai_chatgpt",
                    "default_model": "gpt-integration",
                    "is_default": False,
                },
            )
            assert loss_created.status_code == 201
            loss_connection_id = uuid.UUID(loss_created.json()["id"])
            loss_pairing = await api.post(
                f"/api/agents/connections/{loss_connection_id}/chatgpt/pairings"
            )
            assert loss_pairing.status_code == 201
            await _run_helper_flow(
                fake=fake,
                loop=loop,
                pairing=loss_pairing.json(),
                sentinels=sentinels,
                runtime_artifacts=runtime_artifacts,
            )

        loss_agent_id = uuid.uuid4()
        loss_conversation_id = uuid.uuid4()
        async with postgres_sessions() as loss_setup:
            loss_setup.add(Agent(
                id=loss_agent_id,
                user_id=user_id,
                workspace_id=workspace_id,
                name="Ambiguous",
                description=None,
                system_prompt="",
                connection_id=loss_connection_id,
                provider=None,
                model="gpt-integration",
                temperature=0.0,
                max_history_messages=10,
                auto_context=False,
                extra={},
            ))
            loss_setup.add(AgentTool(
                agent_id=loss_agent_id, server="integration", tool_name="lookup"
            ))
            loss_setup.add(Conversation(
                id=loss_conversation_id,
                agent_id=loss_agent_id,
                user_id=user_id,
                workspace_id=workspace_id,
                channel="web",
            ))
            await loss_setup.commit()

        unknown_calls_before = fake.refresh_calls
        fake.script_after(
            openai_oauth.TOKEN_ENDPOINT,
            httpx.ReadError("assembled refresh response lost"),
        )
        async with postgres_sessions() as lost_session:
            preserved = await lost_session.scalar(
                select(OpenAIOAuthCredential).where(
                    OpenAIOAuthCredential.connection_id == loss_connection_id
                )
            )
            assert preserved is not None
            assert preserved.token_bundle_encrypted is not None
            preserved_encrypted = preserved.token_bundle_encrypted
            preserved_bundle = decrypt_bundle(preserved_encrypted)
            preserved_subject = preserved.subject
            preserved_email = preserved.account_email
            with pytest.raises(openai_oauth.OAuthLifecycleError):
                await oauth_refresh.refresh_credential(
                    lost_session, loss_connection_id, user_id, force=True
                )
        assert fake.refresh_calls - unknown_calls_before == 1

        async with postgres_sessions() as observer:
            unknown = await observer.scalar(
                select(OpenAIOAuthCredential).where(
                    OpenAIOAuthCredential.connection_id == loss_connection_id
                )
            )
            assert unknown is not None
            assert unknown.status == "reauth_required"
            assert unknown.last_error_code == "refresh_outcome_unknown"
            assert unknown.token_bundle_encrypted is not None
            assert hmac.compare_digest(
                unknown.token_bundle_encrypted, preserved_encrypted
            )
            recovered = decrypt_bundle(unknown.token_bundle_encrypted)
            assert hmac.compare_digest(
                recovered.refresh_token, preserved_bundle.refresh_token
            )
            assert unknown.refresh_attempt_id is not None
            assert unknown.refresh_attempt_started_at is not None
            assert unknown.refresh_restore_status == "active"
            assert unknown.subject == preserved_subject
            assert unknown.account_email == preserved_email

        blocked_before = fake.responses_calls
        async with postgres_sessions() as blocked_session:
            blocked_agent = await blocked_session.get(Agent, loss_agent_id)
            assert blocked_agent is not None
            blocked_mcp = _MCP()
            blocked_events = [
                event
                async for event in AgentExecutor(mcp=cast(Any, blocked_mcp)).run(
                    session=blocked_session,
                    agent=blocked_agent,
                    user_id=user_id,
                    workspace_id=workspace_id,
                    conversation_id=loss_conversation_id,
                    user_message="Do not retry an ambiguously consumed refresh token",
                )
            ]
        assert any(event.finish_reason == "error" for event in blocked_events)
        assert blocked_mcp.calls == []
        assert fake.responses_calls == blocked_before
        assert fake.refresh_calls - unknown_calls_before == 1

        sentinels.update(fake.sensitive_values_for_test())
        sentinels.add(f"assembled-{user_id}@example.test")
        child_artifacts, child_sentinels = await asyncio.to_thread(
            _successful_helper_subprocesses, tmp_path
        )
        sentinels.update(child_sentinels)
        helper_artifacts = {
            "children": child_artifacts,
            "callbacks": runtime_artifacts,
        }
        observable = "\n".join((
            caplog.text,
            repr(fake.journal),
            repr(helper_artifacts),
        ))
        assert sentinels
        for sentinel in sentinels:
            assert sentinel not in observable
        runtime_log = tmp_path / "siwc-runtime.log"
        runtime_log.write_text(observable, encoding="utf-8")
        sentinel_file = tmp_path / "siwc-sentinel-manifest"
        sentinel_file.write_text("\n".join(sorted(sentinels)) + "\n", encoding="utf-8")
        sentinel_file.chmod(0o600)
        distributions = tmp_path / "helper-distributions"
        built = subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--outdir",
                str(distributions),
            ],
            cwd=_ROOT / "tools" / "securo-siwc",
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert built.returncode == 0, built.stdout + built.stderr
        scanner = subprocess.run(
            [
                sys.executable,
                str(_ROOT / ".github" / "ci" / "scan_siwc_artifacts.py"),
                "--sentinels-file",
                str(sentinel_file),
                "--sentinels-only",
                str(runtime_log),
                str(_ROOT / "backend" / "app"),
                str(_ROOT / "backend" / "tests"),
                str(_ROOT / "frontend" / "src"),
                str(_ROOT / "docs"),
                str(_ROOT / "tools" / "securo-siwc" / "README.md"),
                str(_ROOT / "tools" / "securo-siwc" / "src"),
                str(_ROOT / "tools" / "securo-siwc" / "tests"),
                str(distributions),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert scanner.returncode == 0, scanner.stdout + scanner.stderr
    finally:
        app.dependency_overrides.pop(get_async_session, None)
        app.dependency_overrides.pop(current_active_user, None)


@pytest.mark.asyncio
async def test_fake_redirect_and_revoked_or_expired_credentials_fail_closed(monkeypatch):
    fake = FakeOpenAI()
    fake.script(
        str(openai_chatgpt._RESPONSES_URL),
        httpx.Response(307, headers={"location": "https://evil.invalid"}),
    )
    monkeypatch.setattr(openai_chatgpt, "_new_responses_transport", lambda: fake.async_transport)

    @asynccontextmanager
    async def invalid_access(**_kwargs):
        from app.agents.providers.openai_chatgpt import AccessTokenSnapshot

        yield AccessTokenSnapshot("expired-or-revoked", 1)

    provider = openai_chatgpt.OpenAIChatGPTProvider(access_token=invalid_access)
    from app.agents.providers.base import ChatMessage, LLMAuthError, LLMUnavailableError

    with pytest.raises(LLMUnavailableError):
        _ = [
            chunk
            async for chunk in provider.chat_stream(
                [ChatMessage(role="user", content="redirect")], model="gpt-integration"
            )
        ]
    with pytest.raises((LLMAuthError, LLMUnavailableError)):
        _ = [
            chunk
            async for chunk in provider.chat_stream(
                [ChatMessage(role="user", content="revoked")], model="gpt-integration"
            )
        ]
    # The provider performs its single allowed pre-output retry, then fails
    # closed without an API-key/global fallback.
    assert fake.responses_calls == 2


def test_helper_jwks_failure_precedes_code_consumption_and_code_replay_is_rejected():
    fake = FakeOpenAI()
    authorization = create_authorization(
        43111,
        f"urn:uuid:{uuid.uuid4()}",
        None,
        state=f"state-{uuid.uuid4().hex}",
    )
    callback = fake.authorize(
        client_id="dynamic_agent_client",
        redirect_uri=authorization.redirect_uri,
        nonce=authorization.nonce,
        code_challenge=parse_qs(urlsplit(authorization.url).query)["code_challenge"][0],
        state=authorization.state,
    )
    values = parse_qs(urlsplit(callback).query)
    code = values["code"][0]
    client_id = values["client_id"][0]
    fake.script(openai_oauth.JWKS_ENDPOINT, httpx.Response(503, json={"error": "unavailable"}))

    with patch(
        "securo_siwc.openai._new_http_client",
        lambda: httpx.Client(
            transport=_StreamingSyncTransport(fake.sync_transport),
            trust_env=False,
            follow_redirects=False,
        ),
    ):
        helper = OpenAIClient()
        try:
            with pytest.raises(RuntimeError, match="signing keys"):
                helper.exchange(authorization, code, client_id)
            assert not any(entry["endpoint"] == "token" for entry in fake.journal)

            bundle = helper.exchange(authorization, code, client_id)
            assert bundle["token_type"] == "Bearer"
            with pytest.raises(RuntimeError, match="token exchange"):
                helper.exchange(authorization, code, client_id)
        finally:
            helper.close()

    assert [entry["endpoint"] for entry in fake.journal].count("token") == 2
    assert code not in repr(fake.journal)


@pytest.mark.asyncio
async def test_wrong_identity_claim_is_independently_rejected_by_helper_and_backend(monkeypatch):
    fake = FakeOpenAI()
    authorization = create_authorization(
        43111,
        f"urn:uuid:{uuid.uuid4()}",
        None,
        state=f"state-{uuid.uuid4().hex}",
    )
    callback = fake.authorize(
        client_id="dynamic_agent_client",
        redirect_uri=authorization.redirect_uri,
        nonce=authorization.nonce,
        code_challenge=parse_qs(urlsplit(authorization.url).query)["code_challenge"][0],
        state=authorization.state,
    )
    values = parse_qs(urlsplit(callback).query)
    client_id = values["client_id"][0]
    fake.override_next_identity_claim("aud", "wrong-audience")
    response = fake.handle_async(httpx.Request(
        "POST",
        openai_oauth.TOKEN_ENDPOINT,
        content=urlencode({
            "grant_type": "authorization_code",
            "client_id": client_id,
            "code": values["code"][0],
            "code_verifier": authorization.verifier,
            "redirect_uri": authorization.redirect_uri,
            "resource": openai_oauth.RESOURCE,
        }),
        headers={"content-type": "application/x-www-form-urlencoded"},
    ))
    assert response.status_code == 200
    bundle = response.json()

    with pytest.raises(ValueError, match="ID token validation failed"):
        helper_validate_token_response(
            dict(bundle),
            jwks=fake.jwks,
            client_id=client_id,
            nonce=authorization.nonce,
        )

    monkeypatch.setattr(openai_oauth, "_new_oauth_transport", lambda: fake.async_transport)
    with pytest.raises(openai_oauth.OAuthLifecycleError) as backend_error:
        await openai_oauth.validate_transferred_token_response(
            token_response=dict(bundle),
            issued_client_id=client_id,
            nonce=authorization.nonce,
        )
    assert backend_error.value.code is openai_oauth.OAuthFailureCode.CORRUPT_TOKEN
