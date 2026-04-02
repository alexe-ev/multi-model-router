"""Bearer token auth dependency for the REST API."""

from __future__ import annotations

import os

from fastapi import HTTPException, Request


async def verify_api_key(request: Request) -> None:
    """FastAPI dependency: check Bearer token against MMROUTER_API_KEY env var.

    If MMROUTER_API_KEY is not set, auth is disabled (all requests pass).
    If set, missing or wrong key returns 401.
    """
    expected_key = os.environ.get("MMROUTER_API_KEY")
    if not expected_key:
        return

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth_header[7:]  # Strip "Bearer "
    if token != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
