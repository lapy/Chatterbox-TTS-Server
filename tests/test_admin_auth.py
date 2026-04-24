from unittest.mock import patch

import pytest
from fastapi import HTTPException

from admin_auth import verify_admin_access


def test_admin_auth_skipped_when_disabled():
    with patch("config.config_manager.get_bool", return_value=False):
        verify_admin_access(None)  # no exception


def test_admin_auth_missing_credentials():
    with patch("config.config_manager.get_bool", return_value=True):
        with pytest.raises(HTTPException) as exc:
            verify_admin_access(None)
        assert exc.value.status_code == 401
