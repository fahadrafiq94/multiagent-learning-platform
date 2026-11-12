from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError

from app.config.settings import Settings
from app.identity.pilot_tokens import issue_token, verify_token

SECRET = SecretStr("pilot-test-secret-" * 4)


def test_pilot_token_binds_internal_identity_and_checks_signature():
    user_id = uuid4()
    token = issue_token(user_id, SECRET)
    assert verify_token(token, SECRET) == user_id
    for invalid in [token + "x", "untrusted", "x.y.z", "", "x" * 1025]:
        with pytest.raises(ValueError, match="Invalid or expired"):
            verify_token(invalid, SECRET)
    with pytest.raises(ValueError):
        verify_token(token, SecretStr("another-secret-" * 4))


def test_expired_tokens_and_excessive_lifetimes_are_rejected(monkeypatch):
    token = issue_token(uuid4(), SECRET, lifetime_seconds=1)
    monkeypatch.setattr("app.identity.pilot_tokens.time.time", lambda: 10**12)
    with pytest.raises(ValueError):
        verify_token(token, SECRET)
    for lifetime in [0, 86401]:
        with pytest.raises(ValueError):
            issue_token(uuid4(), SECRET, lifetime_seconds=lifetime)


def test_settings_mask_secrets_and_validate_storage_and_pool():
    config = Settings(_env_file=None, application_pilot_auth_secret=SECRET)
    assert SECRET.get_secret_value() not in repr(config)
    for kwargs in [
        {"application_pilot_auth_secret": "private-marker"},
        {"application_storage": "unknown"},
        {"application_database_pool_size": 0},
    ]:
        with pytest.raises(ValidationError) as error:
            Settings(_env_file=None, **kwargs)
        assert "private-marker" not in str(error.value)
