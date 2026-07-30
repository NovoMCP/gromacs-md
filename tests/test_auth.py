"""Tests for API key authentication."""

import pytest
from fastapi import HTTPException

from gromacs_md import config


def test_validate_api_key_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "test-key-123")
    # Re-import after monkeypatch to pick up the new value
    from gromacs_md.auth import validate_api_key
    # The function reads config.API_KEY at call time
    result = validate_api_key(api_key="test-key-123")
    assert result == "test-key-123"


def test_validate_api_key_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "test-key-123")
    from gromacs_md.auth import validate_api_key
    with pytest.raises(HTTPException) as exc_info:
        validate_api_key(api_key="wrong-key")
    assert exc_info.value.status_code == 401


def test_validate_api_key_rejects_empty_key(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "test-key-123")
    from gromacs_md.auth import validate_api_key
    with pytest.raises(HTTPException) as exc_info:
        validate_api_key(api_key="")
    assert exc_info.value.status_code == 401


def test_validate_api_key_returns_503_when_unconfigured(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "")
    from gromacs_md.auth import validate_api_key
    with pytest.raises(HTTPException) as exc_info:
        validate_api_key(api_key="anything")
    assert exc_info.value.status_code == 503
