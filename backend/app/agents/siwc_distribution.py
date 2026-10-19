"""Version-matched, installation-served distribution for the SIWC helper."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import ipaddress
import re
import uuid
import zipfile
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from app.agents._siwc_helper.securo_siwc import __version__ as _HELPER_VERSION

_HELPER_ROUTE = "/api/agents/chatgpt/securo-siwc.py"
_WHEEL_ROUTE = "/api/agents/chatgpt/securo-siwc.whl"
_HELPER_ROOT = Path(__file__).resolve().parent / "_siwc_helper"
_HELPER_SOURCE = _HELPER_ROOT / "securo_siwc"
_HELPER_FILES = (
    "__init__.py",
    "__main__.py",
    "backend.py",
    "cli.py",
    "loopback.py",
    "openai.py",
    "protocol.py",
    "security.py",
)
_HELPER_PROJECT = {
    "name": "securo-siwc",
    "version": _HELPER_VERSION,
    "description": "Local Sign in with ChatGPT pairing helper for Securo",
    "requires-python": ">=3.11",
    "dependencies": ["httpx==0.28.1", "python-jose[cryptography]>=3.3,<4"],
}
_ZIP_TIMESTAMP = (2020, 1, 1, 0, 0, 0)


def _host_is_safe(host: str) -> bool:
    if "%" in host:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        labels = host.split(".")
        return (
            len(host) <= 253
            and all(
                label
                and len(label) <= 63
                and re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
                for label in labels
            )
        )


def public_frontend_origin(value: str) -> str:
    """Return a canonical HTTPS origin safe to place in a shell command."""
    if not isinstance(value, str) or not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("FRONTEND_URL must be a canonical HTTPS origin")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise ValueError("FRONTEND_URL must be a canonical HTTPS origin") from None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or port == 443
        or parsed.username is not None
        or parsed.password is not None
        or not _host_is_safe(parsed.hostname)
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.hostname != parsed.hostname.encode("idna").decode("ascii")
    ):
        raise ValueError("FRONTEND_URL must be a canonical HTTPS origin")
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    authority = f"{host}:{port}" if port is not None else host
    canonical = urlunsplit(("https", authority, "", "", ""))
    if value not in (canonical, canonical + "/"):
        raise ValueError("FRONTEND_URL must be a canonical HTTPS origin")
    return canonical


def _record_digest(payload: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return "sha256=" + encoded.decode("ascii")


@lru_cache(maxsize=1)
def helper_wheel() -> tuple[bytes, str, str]:
    """Build a deterministic pure-Python wheel without a networked build backend."""
    project = _HELPER_PROJECT
    version = project["version"]
    distribution = "securo_siwc"
    dist_info = f"{distribution}-{version}.dist-info"
    filename = f"{distribution}-{version}-py3-none-any.whl"
    members: dict[str, bytes] = {}
    for relative in sorted(_HELPER_FILES):
        members[f"securo_siwc/{relative}"] = (_HELPER_SOURCE / relative).read_bytes()

    metadata_lines = [
        "Metadata-Version: 2.3",
        f"Name: {project['name']}",
        f"Version: {version}",
        f"Summary: {project['description']}",
        f"Requires-Python: {project['requires-python']}",
        *(f"Requires-Dist: {dependency}" for dependency in project["dependencies"]),
        "",
        "",
    ]
    members[f"{dist_info}/METADATA"] = "\n".join(metadata_lines).encode("utf-8")
    members[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: securo-installation\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n\n"
    ).encode("ascii")
    members[f"{dist_info}/entry_points.txt"] = (
        "[console_scripts]\nsecuro-siwc = securo_siwc.cli:main\n"
    ).encode("utf-8")

    record_name = f"{dist_info}/RECORD"
    ordered_names = sorted(members)
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name in ordered_names:
        payload = members[name]
        writer.writerow((name, _record_digest(payload), len(payload)))
    writer.writerow((record_name, "", ""))
    members[record_name] = record.getvalue().encode("utf-8")

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as wheel:
        for name in [*ordered_names, record_name]:
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            wheel.writestr(info, members[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    payload = output.getvalue()
    return payload, filename, hashlib.sha256(payload).hexdigest()


def bootstrap_script(origin: str, archive_digest: str) -> bytes:
    """Return a dependency-free bootstrap that installs and runs the matching source."""
    origin = public_frontend_origin(origin)
    if len(archive_digest) != 64 or any(char not in "0123456789abcdef" for char in archive_digest):
        raise ValueError("helper archive digest is invalid")
    wheel_url = origin + _WHEEL_ROUTE
    _, wheel_filename, _ = helper_wheel()
    requirements_lock = (_HELPER_ROOT / "requirements.lock").read_text("utf-8")
    if len(requirements_lock) > 64 * 1024 or "\r" in requirements_lock or "\x00" in requirements_lock:
        raise ValueError("helper requirements lock is invalid")
    template = r'''#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import venv
from pathlib import Path

WHEEL_URL = __WHEEL_URL__
WHEEL_FILENAME = __WHEEL_FILENAME__
WHEEL_SHA256 = __WHEEL_SHA256__
REQUIREMENTS_LOCK = __REQUIREMENTS_LOCK__
MAX_WHEEL_BYTES = 2 * 1024 * 1024


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _download(destination: Path) -> None:
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        _NoRedirect(),
    )
    request = urllib.request.Request(
        WHEEL_URL,
        headers={"Accept": "application/octet-stream", "Accept-Encoding": "identity"},
        method="GET",
    )
    digest = hashlib.sha256()
    total = 0
    try:
        with opener.open(request, timeout=30) as response, destination.open("wb") as target:
            if response.status != 200 or response.headers.get_content_type() != "application/octet-stream":
                raise RuntimeError("Securo returned an invalid helper archive")
            if response.headers.get("Content-Encoding", "identity").casefold() not in ("", "identity"):
                raise RuntimeError("Securo returned an encoded helper archive")
            declared = response.headers.get("Content-Length")
            if (
                declared is None
                or not declared.isascii()
                or not declared.isdecimal()
                or not 0 < int(declared) <= MAX_WHEEL_BYTES
            ):
                raise RuntimeError("Securo returned an invalid helper archive size")
            declared_length = int(declared)
            while True:
                chunk = response.read(min(65_536, MAX_WHEEL_BYTES + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_WHEEL_BYTES:
                    raise RuntimeError("Securo helper archive is too large")
                digest.update(chunk)
                target.write(chunk)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise RuntimeError("Unable to download the Securo helper safely") from None
    if total != declared_length:
        raise RuntimeError("Securo returned an invalid helper archive size")
    if digest.hexdigest() != WHEEL_SHA256:
        raise RuntimeError("Securo helper archive failed integrity verification")


def main() -> int:
    if sys.version_info < (3, 11):
        raise RuntimeError("securo-siwc requires Python 3.11 or newer")
    arguments = sys.argv[1:]
    if "--secret-stdin" in arguments:
        raise RuntimeError(
            "--secret-stdin is unavailable when the bootstrap is read from stdin; "
            "install securo-siwc for secret-manager piping"
        )
    temporary = Path(tempfile.mkdtemp(prefix="securo-siwc-"))
    try:
        os.chmod(temporary, 0o700)
        wheel = temporary / WHEEL_FILENAME
        requirements = temporary / "requirements.lock"
        _download(wheel)
        requirements.write_text(REQUIREMENTS_LOCK, encoding="utf-8", newline="\n")
        os.chmod(requirements, 0o600)
        environment = temporary / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        child_environment = dict(os.environ)
        for key in tuple(child_environment):
            if key.upper().startswith(("PIP_", "PYTHON")) or key.upper() in {
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"
            }:
                child_environment.pop(key, None)
        child_environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        child_environment["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [
                str(python), "-m", "pip", "install", "--isolated", "--no-input",
                "--disable-pip-version-check", "--require-hashes", "--only-binary=:all:",
                "--no-deps", "--index-url", "https://pypi.org/simple", "-r", str(requirements),
            ],
            check=True,
            env=child_environment,
        )
        subprocess.run(
            [
                str(python), "-m", "pip", "install", "--isolated", "--no-input",
                "--disable-pip-version-check", "--no-index", "--no-deps", str(wheel),
            ],
            check=True,
            env=child_environment,
        )
        return subprocess.run(
            [str(python), "-m", "securo_siwc", *arguments],
            env=child_environment,
        ).returncode
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
'''
    return (
        template.replace("__WHEEL_URL__", repr(wheel_url))
        .replace("__WHEEL_FILENAME__", repr(wheel_filename))
        .replace("__WHEEL_SHA256__", repr(archive_digest))
        .replace("__REQUIREMENTS_LOCK__", repr(requirements_lock))
        .encode("utf-8")
    )


def helper_command(origin: str, pairing_id: uuid.UUID) -> str:
    origin = public_frontend_origin(origin)
    return (
        "curl --disable --noproxy '*' --proto '=https' --tlsv1.2 "
        "--fail --silent --show-error "
        f"{origin}{_HELPER_ROUTE} | python3 - {pairing_id} --server {origin}"
    )
