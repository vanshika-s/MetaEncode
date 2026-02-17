# src/utils/user_id.py
"""Per-user identification via browser cookies.

Generates a persistent UUID for each browser, stored as a cookie and cached
in session state. Used to namespace per-user data (e.g., selection history)
so multi-user deployments don't share state.
"""

import uuid

import streamlit as st
import streamlit.components.v1 as components

COOKIE_NAME = "metaencode_user_id"
SESSION_KEY = "_metaencode_user_id"


def _set_cookie(user_id: str) -> None:
    """Inject a JS snippet to persist the user ID as a browser cookie."""
    max_age = 365 * 24 * 60 * 60  # 1 year
    js = f"""
    <script>
    document.cookie = "{COOKIE_NAME}={user_id}; path=/; max-age={max_age}; SameSite=Lax";
    </script>
    """
    components.html(js, height=0, width=0)


def get_or_create_user_id() -> str:
    """Return a stable per-browser user ID, creating one if needed.

    Lookup priority:
        1. st.session_state (fastest, avoids cookie round-trip lag)
        2. st.context.cookies (persisted from a previous visit)
        3. Generate a new UUID and set the cookie via JS injection

    Returns:
        A UUID string identifying this browser/user.
    """
    # 1. Session state (already resolved this session)
    if SESSION_KEY in st.session_state:
        return st.session_state[SESSION_KEY]

    # 2. Existing browser cookie
    cookie_val = st.context.cookies.get(COOKIE_NAME)
    if cookie_val:
        st.session_state[SESSION_KEY] = cookie_val
        return cookie_val

    # 3. First visit — generate new ID and persist
    new_id = uuid.uuid4().hex
    st.session_state[SESSION_KEY] = new_id
    _set_cookie(new_id)
    return new_id
