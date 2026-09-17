"""Guard every test database reset before importing the real application."""
import os
from urllib.parse import urlsplit


def validate_test_database_url(value: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme not in {"postgresql", "postgresql+asyncpg", "postgres"}
        or parts.hostname not in {"localhost", "127.0.0.1", "::1", "db", "postgres"}
        or parts.path != "/persona_test"
        or parts.query
        or parts.fragment
    ):
        raise RuntimeError(
            "Tests require PostgreSQL database persona_test on localhost, "
            "127.0.0.1, ::1, db or postgres, without URL parameters. "
            "Never supply a development or production DATABASE_URL."
        )
    return value


def configure_test_environment(*, required: bool = True) -> str:
    value = os.environ.get("TEST_DATABASE_URL")
    if not value:
        if required:
            raise RuntimeError("Set TEST_DATABASE_URL to the disposable database; see TESTING.md.")
        # Unit tests may import models but never connect. Ignore DATABASE_URL
        # inherited from a developer shell or loaded from the local .env file.
        value = "postgresql://persona_test:persona_test@127.0.0.1:55432/persona_test"
    os.environ["DATABASE_URL"] = validate_test_database_url(value)
    os.environ["SESSION_SECRET_KEY"] = "fictional-test-session-secret-never-use-in-production"
    os.environ["ENV"] = "test"
    # python-dotenv must not fill other settings from a developer's .env.
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    return value
