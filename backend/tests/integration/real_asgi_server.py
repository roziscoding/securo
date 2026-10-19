"""Uvicorn entry point for real-network SIWC helper integration tests."""
from __future__ import annotations

import os

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.agents.services import openai_oauth
from app.agents.api.oauth_pairing import pairing_helper_rate_limit
from app.core.database import get_async_session
from app.main import app
from tests.integration.provider_bridge_transport import ProviderBridgeAsyncTransport

_database_url = os.environ["SECURO_TEST_DATABASE_URL"]
_schema = os.environ["SECURO_TEST_DATABASE_SCHEMA"]
_provider_bridge = os.environ["SECURO_TEST_PROVIDER_BRIDGE"]
_engine = create_async_engine(_database_url).execution_options(
    schema_translate_map={None: _schema}
)
_sessions = async_sessionmaker(_engine, class_=AsyncSession, expire_on_commit=False)


async def _session_override():
    async with _sessions() as session:
        yield session


async def _rate_limit_override() -> None:
    return None


app.dependency_overrides[get_async_session] = _session_override
app.dependency_overrides[pairing_helper_rate_limit] = _rate_limit_override


def _provider_transport() -> httpx.AsyncBaseTransport:
    return ProviderBridgeAsyncTransport(_provider_bridge)


setattr(openai_oauth, "_new_oauth_transport", _provider_transport)
