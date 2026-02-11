# tests/test_utils/test_user_id.py
"""Tests for per-user identification via browser cookies."""

from unittest.mock import MagicMock, patch

import pytest

from src.utils.user_id import COOKIE_NAME, SESSION_KEY, get_or_create_user_id


@pytest.fixture(autouse=True)
def _clean_session_state():
    """Ensure session state is clean before each test."""
    with patch("src.utils.user_id.st") as mock_st:
        mock_st.session_state = {}
        mock_st.context = MagicMock()
        mock_st.context.cookies = {}
        yield mock_st


class TestGetOrCreateUserId:
    """Tests for get_or_create_user_id lookup priority."""

    def test_returns_existing_session_state_id(self, _clean_session_state):
        """Session state takes highest priority."""
        mock_st = _clean_session_state
        mock_st.session_state[SESSION_KEY] = "existing-session-id"

        result = get_or_create_user_id()

        assert result == "existing-session-id"

    def test_returns_cookie_and_caches_in_session(self, _clean_session_state):
        """Cookie value is returned and cached in session state."""
        mock_st = _clean_session_state
        mock_st.context.cookies = {COOKIE_NAME: "cookie-user-id"}

        result = get_or_create_user_id()

        assert result == "cookie-user-id"
        assert mock_st.session_state[SESSION_KEY] == "cookie-user-id"

    @patch("src.utils.user_id.components")
    @patch("src.utils.user_id.uuid")
    def test_generates_new_id_when_none_exists(
        self, mock_uuid, mock_components, _clean_session_state
    ):
        """New UUID generated, cached, and cookie set via JS."""
        mock_st = _clean_session_state
        mock_uuid.uuid4.return_value = MagicMock(hex="abc123deadbeef")

        result = get_or_create_user_id()

        assert result == "abc123deadbeef"
        assert mock_st.session_state[SESSION_KEY] == "abc123deadbeef"
        mock_components.html.assert_called_once()
        # Verify the JS contains the cookie name and value
        js_arg = mock_components.html.call_args[0][0]
        assert COOKIE_NAME in js_arg
        assert "abc123deadbeef" in js_arg

    def test_session_state_takes_priority_over_cookie(self, _clean_session_state):
        """Session state ID returned even when cookie differs."""
        mock_st = _clean_session_state
        mock_st.session_state[SESSION_KEY] = "session-id"
        mock_st.context.cookies = {COOKIE_NAME: "different-cookie-id"}

        result = get_or_create_user_id()

        assert result == "session-id"

    @patch("src.utils.user_id.components")
    @patch("src.utils.user_id.uuid")
    def test_cookie_js_sets_lax_samesite(
        self, mock_uuid, mock_components, _clean_session_state
    ):
        """Cookie JS snippet includes SameSite=Lax."""
        mock_uuid.uuid4.return_value = MagicMock(hex="testid")

        get_or_create_user_id()

        js_arg = mock_components.html.call_args[0][0]
        assert "SameSite=Lax" in js_arg

    @patch("src.utils.user_id.components")
    @patch("src.utils.user_id.uuid")
    def test_cookie_js_invisible_iframe(
        self, mock_uuid, mock_components, _clean_session_state
    ):
        """JS injection uses zero-size component (invisible)."""
        mock_uuid.uuid4.return_value = MagicMock(hex="testid")

        get_or_create_user_id()

        call_kwargs = mock_components.html.call_args[1]
        assert call_kwargs["height"] == 0
        assert call_kwargs["width"] == 0
