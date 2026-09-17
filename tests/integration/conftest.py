"""Every API test gets a fresh schema's data and fresh rate-control state."""
import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.database import SessionLocal, engine
from app.main import app
from app.services.ai.gemini_service import GeminiService
from app.services.rate_control_service import RateControlService, get_rate_control_service
from tests.fakes.fake_gemini import FakeGemini
from tests.support.environment import validate_test_database_url
from tests.support.seed import TERMS, reset_database, seed_database


@pytest_asyncio.fixture(autouse=True)
async def database():
    if not os.environ.get("TEST_DATABASE_URL"):
        pytest.fail("Integration tests need TEST_DATABASE_URL; see TESTING.md or run pytest tests/unit.")
    validate_test_database_url(os.environ["TEST_DATABASE_URL"])
    await reset_database()
    await seed_database()
    yield
    await reset_database()
    await engine.dispose()


@pytest.fixture
def fake_gemini():
    return FakeGemini()


@pytest.fixture
def rate_control():
    return RateControlService()


@pytest_asyncio.fixture
async def client(database, fake_gemini, rate_control):
    app.dependency_overrides[GeminiService] = lambda: fake_gemini
    app.dependency_overrides[get_rate_control_service] = lambda: rate_control
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as visitor:
            yield visitor
    finally:
        app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def consented_client(client):
    response = await client.post("/api/consent", json={"conditionText": TERMS["condition"]})
    assert response.status_code == 200, response.text
    return client


@pytest_asyncio.fixture
async def db(database):
    async with SessionLocal() as session:
        yield session
