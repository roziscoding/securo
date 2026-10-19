"""Exercise the exact installation-served SIWC artifacts and wheel matrix."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).parents[2]
BACKEND = ROOT / "backend"
ASSETS = BACKEND / "app" / "agents" / "_siwc_helper"
sys.path.insert(0, str(BACKEND))

from app.agents.siwc_distribution import helper_wheel


def _run(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(args, check=True, env=env)


def _check_backend_wheel(root: Path) -> None:
    root.mkdir()
    _run(sys.executable, "-m", "build", "--wheel", "--outdir", str(root), str(BACKEND))
    wheels = tuple(root.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("backend wheel build did not produce exactly one wheel")
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
    required = {
        "app/agents/_siwc_helper/requirements.lock",
        "app/agents/_siwc_helper/securo_siwc/cli.py",
        "app/agents/_siwc_helper/securo_siwc/openai.py",
    }
    missing = required - names
    if missing:
        raise RuntimeError(f"backend wheel omitted SIWC helper assets: {sorted(missing)}")


def _check_host_install(root: Path) -> None:
    root.mkdir()
    wheel, filename, _digest = helper_wheel()
    wheel_path = root / filename
    wheel_path.write_bytes(wheel)
    environment = root / "venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith(("PIP_", "PYTHON"))
        and key.upper() not in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"}
    }
    clean_env.update(PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1")
    _run(
        str(python), "-m", "pip", "install", "--isolated", "--no-input",
        "--disable-pip-version-check", "--require-hashes", "--only-binary=:all:",
        "--no-deps", "--index-url", "https://pypi.org/simple",
        "-r", str(ASSETS / "requirements.lock"), env=clean_env,
    )
    _run(
        str(python), "-m", "pip", "install", "--isolated", "--no-input",
        "--disable-pip-version-check", "--no-index", "--no-deps", str(wheel_path),
        env=clean_env,
    )
    _run(str(python), "-m", "securo_siwc", "--help", env=clean_env)


def _check_platform_wheels(root: Path) -> None:
    matrix = (
        ("3.11", "manylinux_2_17_aarch64"),
        ("3.14", "manylinux_2_17_x86_64"),
        ("3.11", "macosx_10_15_x86_64"),
        ("3.14", "macosx_11_0_arm64"),
        ("3.11", "win_amd64"),
        ("3.14", "win_amd64"),
    )
    for python_version, platform in matrix:
        destination = root / f"{python_version}-{platform}"
        destination.mkdir()
        abi = "cp" + python_version.replace(".", "")
        _run(
            sys.executable, "-m", "pip", "download", "--quiet",
            "--disable-pip-version-check", "--require-hashes", "--only-binary=:all:",
            "--no-deps", "--implementation", "cp", "--python-version", python_version,
            "--abi", abi, "--platform", platform, "--dest", str(destination),
            "-r", str(ASSETS / "requirements.lock"),
        )
        if len(tuple(destination.iterdir())) != 15:
            raise RuntimeError(f"incomplete SIWC wheel set for Python {python_version} {platform}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="securo-siwc-ci-") as temporary:
        root = Path(temporary)
        _check_backend_wheel(root / "backend-wheel")
        _check_host_install(root / "host")
        matrix_root = root / "matrix"
        matrix_root.mkdir()
        _check_platform_wheels(matrix_root)
    print("installation-served SIWC distribution check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
