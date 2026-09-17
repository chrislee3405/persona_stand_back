import pytest

from tests.support.environment import validate_test_database_url


@pytest.mark.parametrize("url", [
    "postgresql://user:secret@production.example/persona_test",
    "postgresql://user:secret@localhost/persona_stand",
    "postgresql://user:secret@db/postgres",
    "sqlite:///persona_test",
    "postgresql://user:secret@localhost/persona_test?host=production.example",
])
def test_reset_refuses_unsafe_database_targets(url):
    with pytest.raises(RuntimeError, match="persona_test"):
        validate_test_database_url(url)


def test_disposable_compose_database_is_allowed():
    value = "postgresql+asyncpg://persona_test:persona_test@db:5432/persona_test"
    assert validate_test_database_url(value) == value
