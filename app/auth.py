"""
Lightweight session-token authentication for the MedAgent web UI.

Deliberately simple (this is a demo-grade auth layer, not a production
identity system): PBKDF2-HMAC password hashing (stdlib `hashlib`, no
extra dependency), opaque random session tokens stored in SQLite, no
password reset / email verification flow.

The session token doubles as the `session_id` used everywhere else in
the app (patient profile, conversation memory, session findings,
history) — logging in gives a user continuity across every feature
that was previously only session-scoped.
"""
import hashlib
import logging
import os
import secrets

from app import database as db

logger = logging.getLogger("medagent.auth")

PBKDF2_ITERATIONS = 260_000


def _hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return digest.hex(), salt.hex()


def verify_password(password: str, password_hash: str, salt_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    computed, _ = _hash_password(password, salt)
    return secrets.compare_digest(computed, password_hash)


class AuthError(Exception):
    pass


def register(name: str, email: str, password: str) -> dict:
    email = email.strip().lower()
    if len(password) < 6:
        raise AuthError("Password must be at least 6 characters.")
    if db.get_user_by_email(email):
        raise AuthError("An account with this email already exists.")

    password_hash, salt = _hash_password(password)
    user_id = db.create_user(name=name.strip(), email=email, password_hash=password_hash, salt=salt)
    token = _issue_session(user_id)
    user = db.get_user_by_id(user_id)
    logger.info(f"Registered new user id={user_id}")
    return {"token": token, "user": user}


def login(email: str, password: str) -> dict:
    email = email.strip().lower()
    user_row = db.get_user_by_email(email)
    if not user_row or not verify_password(password, user_row["password_hash"], user_row["salt"]):
        raise AuthError("Incorrect email or password.")
    token = _issue_session(user_row["id"])
    user = db.get_user_by_id(user_row["id"])
    return {"token": token, "user": user}


def _issue_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    db.create_session(token, user_id)
    return token


def get_user_from_token(token: str) -> dict | None:
    if not token:
        return None
    return db.get_user_by_session_token(token)


def logout(token: str) -> None:
    db.delete_session(token)
