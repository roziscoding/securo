"""Repository-level SIWC endpoint and secret-artifact policy contracts."""
from __future__ import annotations

import ast
import re
from pathlib import Path

from app.agents.providers import openai_chatgpt
from app.agents.services import openai_oauth


ROOT = Path(__file__).parents[2]
HELPER = ROOT / "backend" / "app" / "agents" / "_siwc_helper" / "securo_siwc" / "openai.py"
EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "node_modules",
    "tests",
    "docs",
    "spikes",
}


def _helper_constants() -> dict[str, str]:
    tree = ast.parse(HELPER.read_text(encoding="utf-8"), filename=str(HELPER))
    literals: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and isinstance(statement.value, ast.Constant):
            if isinstance(statement.value.value, str):
                literals[target.id] = statement.value.value
    issuer = literals["ISSUER"]
    return {
        "issuer": issuer,
        "authorize": f"{issuer}/api/accounts/authorize",
        "token": f"{issuer}/api/accounts/oauth/token",
        "jwks": f"{issuer}/.well-known/jwks.json",
        "resource": literals["RESOURCE"],
    }


def _production_sources() -> list[Path]:
    roots = [ROOT / "backend" / "app", ROOT / "frontend" / "src"]
    return [
        path
        for root in roots
        for path in root.rglob("*")
        if path.suffix in {".py", ".ts", ".tsx"}
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
    ]


def _artifact_candidates() -> list[Path]:
    candidates: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        lowered = {part.casefold() for part in relative.parts}
        if EXCLUDED_PARTS.intersection(relative.parts):
            continue
        if (
            path.name in {".env.example"}
            or path.suffix.casefold() in {".log", ".pem"}
            or lowered.intersection({"artifacts", "artifact", "examples", "example", "dist"})
        ):
            candidates.append(path)
    return candidates


def test_production_siwc_endpoint_inventory_is_exact_and_public():
    helper = _helper_constants()
    assert helper == {
        "issuer": "https://auth.openai.com",
        "authorize": "https://auth.openai.com/api/accounts/authorize",
        "token": "https://auth.openai.com/api/accounts/oauth/token",
        "jwks": "https://auth.openai.com/.well-known/jwks.json",
        "resource": "https://api.openai.com/v1",
    }
    assert openai_oauth._ENDPOINT_METHODS == {
        "https://auth.openai.com/api/accounts/oauth/token": "POST",
        "https://auth.openai.com/api/accounts/oauth/revoke": "POST",
        "https://auth.openai.com/.well-known/jwks.json": "GET",
        "https://api.openai.com/v1/models": "GET",
    }
    assert openai_chatgpt._RESPONSES_URL == "https://api.openai.com/v1/responses"


def test_private_chatgpt_backend_is_absent_from_production_sources():
    forbidden = "chatgpt.com/" + "backend-api"
    offenders = [
        str(path.relative_to(ROOT))
        for path in _production_sources()
        if forbidden in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_ci_always_scans_repository_sources_independently_of_helper_selection():
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    scan_job = workflow.split("  repository-credential-scan:", 1)[1].split(
        "\n  siwc-helper:", 1
    )[0]
    assert "if:" not in scan_job
    assert "Detect helper-relevant changes" not in scan_job
    assert "python3 .github/ci/scan_siwc_artifacts.py" in scan_job
    for path in (
        "backend/app",
        "backend/tests",
        "frontend/src",
        "docs",
        "backend/app/agents/_siwc_helper",
        "backend/tools/securo-siwc/tests",
        "backend/tools/securo-siwc/README.md",
        "backend/tools/securo-siwc/pyproject.toml",
        "backend/app/agents/_siwc_helper/requirements.lock",
        ".github/ci",
        ".github/workflows/ci.yml",
    ):
        assert path in scan_job


def test_siwc_integration_uses_private_transport_adapters_not_production_test_seams():
    page = ROOT / "frontend" / "src" / "pages" / "agent-connections.tsx"
    source = page.read_text(encoding="utf-8")
    assert "pairingBrowserLocation" not in source
    assert "Injectable only for deterministic" not in source


def test_generated_artifacts_and_examples_contain_no_credentials_or_fixture_keys():
    private_key_marker = "-----BEGIN " + "PRIVATE KEY-----"
    credential_patterns = [
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        re.compile(r"\bBearer\s+[A-Za-z0-9._~-]{24,}\b", re.IGNORECASE),
        re.compile(r"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"),
        re.compile(r"(?im)^\s*(?:access_token|refresh_token|id_token)\s*[=:]\s*[^\s#][^\r\n]*$"),
    ]
    offenders: list[str] = []
    for path in _artifact_candidates():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if private_key_marker in text or any(pattern.search(text) for pattern in credential_patterns):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
