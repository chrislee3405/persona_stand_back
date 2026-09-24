"""Test-only entrypoint mounted by ec2yml; never included in the app image."""
from contextlib import asynccontextmanager

from tests.support.environment import configure_test_environment

configure_test_environment()

from app.main import app  # noqa: E402
from app.database import engine, coordination_engine  # noqa: E402
from app.services.ai import gemini_service  # noqa: E402
from tests.fakes.fake_gemini import FakeGemini  # noqa: E402
from tests.support.seed import reset_database, seed_database  # noqa: E402


def forbidden_live_client():
    raise AssertionError("Real Gemini is forbidden in the browser-test application")


gemini_service._get_client = forbidden_live_client
fake = FakeGemini()
app.dependency_overrides[gemini_service.GeminiService] = lambda: fake


@asynccontextmanager
async def test_lifespan(application):
    await reset_database()
    await seed_database()
    try:
        yield
    finally:
        await engine.dispose()
        await coordination_engine.dispose()


app.router.lifespan_context = test_lifespan
