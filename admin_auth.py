# HTTP Basic authentication for administrative endpoints when server.use_auth is true.

from __future__ import annotations

import logging
import secrets
from typing import Annotated, Optional

from fastapi import HTTPException, Security
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from config import config_manager

logger = logging.getLogger(__name__)

_http_basic = HTTPBasic(auto_error=False)


def verify_admin_access(
    credentials: Annotated[Optional[HTTPBasicCredentials], Security(_http_basic)],
) -> None:
    """Require valid HTTP Basic credentials when server.use_auth is enabled."""
    if not config_manager.get_bool("server.use_auth", False):
        return
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required for this endpoint.",
            headers={"WWW-Authenticate": "Basic"},
        )
    expected_user = config_manager.get_string("server.auth_username", "user")
    expected_pass = config_manager.get_string("server.auth_password", "password")
    user_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"), expected_user.encode("utf-8")
    )
    pass_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"), expected_pass.encode("utf-8")
    )
    if not (user_ok and pass_ok):
        logger.warning("Failed admin auth attempt for user %r", credentials.username)
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials.",
            headers={"WWW-Authenticate": "Basic"},
        )
