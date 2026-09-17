"""Set safe settings before pytest imports any application modules."""
import pytest

from tests.support.environment import configure_test_environment

configure_test_environment(required=False)


@pytest.fixture(autouse=True)
def forbid_live_gemini(monkeypatch):
    from app.services.ai import gemini_service

    def forbidden():
        raise AssertionError("A test attempted to create the real Gemini client")

    monkeypatch.setattr(gemini_service, "_get_client", forbidden)
