"""Phase A test helper: satisfy the global auth gate for integration clients.

After Phase A every `/api/v1/*` route (except `/auth/*` and `/instance/*`)
requires a valid Bearer JWT. Existing integration tests build their own
TestClient without a token, so they now 401. This helper seeds an owner +
sharing-mode instance_config, logs in through the app, and stamps the
Authorization header onto the client so every subsequent request passes the
gate.

Usage in a per-file client fixture:

    from ._auth_helper import authenticate_test_client
    ...
    client = TestClient(app)
    authenticate_test_client(client, app)
    yield client
"""
from __future__ import annotations

TEST_JWT_SECRET = "test-secret-please-do-not-use-in-prod-32-chars-or-more"
_OWNER_EMAIL = "testowner@example.com"
_OWNER_PASSWORD = "testpass12345"


def authenticate_test_client(client, app):
    """Attach auth to `app.state` (if lifespan didn't run), seed an owner +
    sharing instance_config, log in, and set the Bearer header on `client`.

    Idempotent: safe whether or not lifespan ran and whether or not an owner
    already exists. Returns `client` for chaining."""
    from app.resources.metadata_db import MetadataDB
    from app.services.auth_service import AuthService, OwnerAlreadyExistsError

    # Tests that `yield TestClient(app)` without a `with` block never run the
    # lifespan, so app.state.auth_service is unset — create one here. When
    # lifespan DID run we reuse its instance so the JWT secret matches.
    if getattr(app.state, "auth_service", None) is None:
        app.state.auth_service = AuthService(jwt_secret=TEST_JWT_SECRET)
    auth = app.state.auth_service

    if MetadataDB.get_instance_config() is None:
        MetadataDB.set_instance_config(mode="sharing", created_at=1.0)

    try:
        auth.create_owner(email=_OWNER_EMAIL, password=_OWNER_PASSWORD)
    except OwnerAlreadyExistsError:
        pass

    resp = client.post(
        "/api/v1/auth/login",
        json={"email": _OWNER_EMAIL, "password": _OWNER_PASSWORD},
    )
    token = resp.json()["token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client
