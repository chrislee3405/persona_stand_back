"""LOG_LEVEL and SESSION_COOKIE_SECURE: explicit values, the ENV fallback, and refusal of typos."""
import logging

import pytest

from app.runtime_settings import log_level, session_cookie_secure


@pytest.fixture
def env(monkeypatch):
    for name in ("ENV", "LOG_LEVEL", "SESSION_COOKIE_SECURE"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_match_the_old_development_behaviour(env):
    assert log_level() == logging.DEBUG
    assert session_cookie_secure() is False


def test_legacy_env_production_still_sets_both(env):
    env.setenv("ENV", "production")
    assert log_level() == logging.INFO
    assert session_cookie_secure() is True


def test_logging_can_be_production_grade_before_tls(env):
    # The whole point of the split: INFO logging with a non-Secure cookie.
    env.setenv("LOG_LEVEL", "info")
    env.setenv("SESSION_COOKIE_SECURE", "false")
    assert log_level() == logging.INFO
    assert session_cookie_secure() is False


def test_explicit_values_override_env(env):
    env.setenv("ENV", "production")
    env.setenv("LOG_LEVEL", "WARNING")
    env.setenv("SESSION_COOKIE_SECURE", "0")
    assert log_level() == logging.WARNING
    assert session_cookie_secure() is False


@pytest.mark.parametrize("value", ["true", "TRUE", "1", "yes", "on"])
def test_secure_cookie_accepts_the_usual_true_spellings(env, value):
    env.setenv("SESSION_COOKIE_SECURE", value)
    assert session_cookie_secure() is True


def test_a_log_level_typo_is_refused_rather_than_left_at_debug(env):
    env.setenv("LOG_LEVEL", "INFOO")
    with pytest.raises(RuntimeError, match="LOG_LEVEL"):
        log_level()


def test_an_unclear_cookie_setting_is_refused(env):
    env.setenv("SESSION_COOKIE_SECURE", "production")
    with pytest.raises(RuntimeError, match="SESSION_COOKIE_SECURE"):
        session_cookie_secure()
