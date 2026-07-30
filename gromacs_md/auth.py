"""API key authentication."""

from fastapi import Header, HTTPException

from . import config


def validate_api_key(api_key: str = Header(None, alias="API-Key")):
    if not config.API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Service misconfigured: API_KEY environment variable not set",
        )
    if not api_key or api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key
