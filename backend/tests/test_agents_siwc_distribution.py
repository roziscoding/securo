"""Distribution contract for the installation-served SIWC helper."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import tomllib
import types
import uuid
import zipfile
from email.message import Message
from pathlib import Path

import pytest

from app.agents.siwc_distribution import (
    bootstrap_script,
    helper_command,
    helper_wheel,
    public_frontend_origin,
)


def test_helper_wheel_is_deterministic_complete_and_self_consistent():
    first, filename, digest = helper_wheel()
    second, second_filename, second_digest = helper_wheel()

    assert first == second
    assert filename == second_filename == "securo_siwc-0.1.0-py3-none-any.whl"
    assert digest == second_digest == hashlib.sha256(first).hexdigest()
    with zipfile.ZipFile(io.BytesIO(first)) as wheel:
        names = wheel.namelist()
        source_root = Path(__file__).parents[1] / "app" / "agents" / "_siwc_helper" / "securo_siwc"
        source_members = {f"securo_siwc/{path.name}" for path in source_root.glob("*.py")}
        assert {name for name in names if name.startswith("securo_siwc/")} == source_members
        assert "securo_siwc/cli.py" in names
        assert "securo_siwc-0.1.0.dist-info/METADATA" in names
        assert "securo_siwc-0.1.0.dist-info/WHEEL" in names
        assert "securo_siwc-0.1.0.dist-info/entry_points.txt" in names
        project = tomllib.loads(
            (Path(__file__).parents[1] / "tools" / "securo-siwc" / "pyproject.toml").read_text("utf-8")
        )["project"]
        metadata = wheel.read("securo_siwc-0.1.0.dist-info/METADATA").decode("utf-8")
        assert f"Name: {project['name']}\n" in metadata
        assert f"Version: {project['version']}\n" in metadata
        assert f"Requires-Python: {project['requires-python']}\n" in metadata
        for dependency in project["dependencies"]:
            assert f"Requires-Dist: {dependency}\n" in metadata
        record_name = "securo_siwc-0.1.0.dist-info/RECORD"
        rows = list(csv.reader(io.StringIO(wheel.read(record_name).decode("utf-8"))))
        assert [row[0] for row in rows] == names
        for member, encoded_digest, size in rows[:-1]:
            payload = wheel.read(member)
            expected = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
            assert encoded_digest == f"sha256={expected}"
            assert size == str(len(payload))
        assert rows[-1] == [record_name, "", ""]


def test_helper_requirements_lock_matches_backend_uv_wheels():
    backend_root = Path(__file__).parents[1]
    lock_text = (backend_root / "app" / "agents" / "_siwc_helper" / "requirements.lock").read_text("utf-8")
    parsed: dict[str, tuple[str, set[str]]] = {}
    current: tuple[str, str] | None = None
    hashes: set[str] = set()
    for line in lock_text.splitlines():
        if not line or line.startswith("#"):
            continue
        if not line.startswith(" "):
            if current is not None:
                parsed[current[0]] = (current[1], hashes)
            requirement = line.removesuffix(" \\")
            name, version = requirement.split("==", 1)
            current = (name, version)
            hashes = set()
        else:
            hashes.add(line.strip().removesuffix(" \\").removeprefix("--hash="))
    assert current is not None
    parsed[current[0]] = (current[1], hashes)

    uv_packages = {
        package["name"]: package
        for package in tomllib.loads((backend_root / "uv.lock").read_text("utf-8"))["package"]
    }
    assert set(parsed) == {
        "anyio", "certifi", "cffi", "cryptography", "ecdsa", "h11", "httpcore",
        "httpx", "idna", "pyasn1", "pycparser", "python-jose", "rsa", "six",
        "typing-extensions",
    }
    for name, (version, package_hashes) in parsed.items():
        package = uv_packages[name]
        assert version == package["version"]
        assert package_hashes == {wheel["hash"] for wheel in package["wheels"]}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://securo.example", "https://securo.example"),
        ("https://securo.example/", "https://securo.example"),
        ("https://securo.example:8443", "https://securo.example:8443"),
    ],
)
def test_public_frontend_origin_accepts_only_canonical_https_origins(value, expected):
    assert public_frontend_origin(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "http://securo.example",
        "https://securo.example:443",
        "https://user@securo.example",
        "https://securo.example/path",
        "https://securo.example?query=yes",
        "https://securo.example/#fragment",
        "https://securo.example\nmalicious",
        "https://securo.example'$(touch nope)'",
        "https://[fe80::1%25eth0]",
        "//securo.example",
    ],
)
def test_public_frontend_origin_rejects_unsafe_or_non_https_values(value):
    with pytest.raises(ValueError, match="FRONTEND_URL"):
        public_frontend_origin(value)


def test_bootstrap_and_command_are_pipeable_secret_free_and_bound_to_archive():
    pairing_id = uuid.UUID("235e24a4-df88-4e3d-b6cf-750bc109d4d0")
    origin = "https://securo.example"
    wheel, wheel_filename, digest = helper_wheel()

    script = bootstrap_script(origin, digest)
    compile(script, "securo-siwc.py", "exec")
    command = helper_command(origin, pairing_id)

    assert wheel and wheel_filename.endswith(".whl")
    assert digest.encode() in script
    assert b"/api/agents/chatgpt/securo-siwc.whl" in script
    assert b"--require-hashes" in script
    assert b"--only-binary=:all:" in script
    assert b"--no-deps" in script
    assert b"--secret-stdin is unavailable when the bootstrap is read from stdin" in script
    assert command == (
        "curl --disable --noproxy '*' --proto '=https' --tlsv1.2 "
        "--fail --silent --show-error "
        "https://securo.example/api/agents/chatgpt/securo-siwc.py | "
        "python3 - 235e24a4-df88-4e3d-b6cf-750bc109d4d0 "
        "--server https://securo.example"
    )
    assert "secret" not in command.casefold()


@pytest.mark.parametrize("declared", ["-1", "999", "+1", " 1"])
def test_bootstrap_rejects_invalid_or_inexact_content_lengths(
    declared, monkeypatch, tmp_path
):
    import urllib.request

    payload = b"archive"
    script = bootstrap_script("https://securo.example", hashlib.sha256(payload).hexdigest())
    namespace = {"__name__": "served_bootstrap_test"}
    exec(script, namespace)

    headers = Message()
    headers["Content-Type"] = "application/octet-stream"
    headers["Content-Length"] = declared

    class FakeResponse(io.BytesIO):
        status = 200

        def __init__(self):
            super().__init__(payload)
            self.headers = headers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            del exc_type, exc_val, exc_tb
            self.close()

    class FakeOpener:
        def open(self, *_args, **_kwargs):
            return FakeResponse()

    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: FakeOpener())
    download = namespace["_download"]
    assert callable(download)
    with pytest.raises(RuntimeError, match="archive size"):
        download(tmp_path / "helper.zip")


@pytest.mark.asyncio
async def test_public_helper_routes_serve_exact_version_without_authentication(
    client, monkeypatch
):
    from app.agents.api import oauth_pairing as pairing_api

    monkeypatch.setattr(
        pairing_api, "get_settings", lambda: types.SimpleNamespace(
            frontend_url="https://securo.example"
        )
    )
    expected_wheel, wheel_filename, expected_digest = helper_wheel()

    script_response = await client.get("/api/agents/chatgpt/securo-siwc.py")
    assert script_response.status_code == 200
    assert script_response.headers["content-type"] == "text/x-python; charset=utf-8"
    assert script_response.headers["cache-control"] == "no-store"
    assert script_response.headers["x-content-type-options"] == "nosniff"
    assert expected_digest.encode() in script_response.content
    compile(script_response.content, "served-securo-siwc.py", "exec")

    wheel_response = await client.get("/api/agents/chatgpt/securo-siwc.whl")
    assert wheel_response.status_code == 200
    assert wheel_response.headers["content-type"] == "application/octet-stream"
    assert wheel_response.headers["cache-control"] == "no-store"
    assert wheel_response.headers["etag"] == f'"sha256:{expected_digest}"'
    assert wheel_response.headers["content-disposition"] == f'attachment; filename="{wheel_filename}"'
    assert wheel_response.content == expected_wheel
