"""
services/auth_service.py - Gestión de espacios de trabajo (workspaces), control
de acceso de operador (HMAC) y enfriamiento de escaneos en vivo para Streamlit.
"""

import hmac
import math
import re
import time
from typing import Any, Tuple

import streamlit as st

import config

_DEFAULT_WORKSPACE_ID = "default"
_WORKSPACE_ALLOWED_RE = re.compile(r"[^a-z0-9_.@-]+")
_CTRL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]+")


def _normalize_workspace_id(raw_id: Any) -> str:
    """Normaliza el ID del workspace a minúsculas con caracteres [a-z0-9_.@-]."""
    cleaned = str(raw_id or "").strip().lower()
    cleaned = _WORKSPACE_ALLOWED_RE.sub("", cleaned)[:64]
    return cleaned or _DEFAULT_WORKSPACE_ID


def get_current_workspace_id() -> str:
    """Retorna el identificador normalizado del espacio de trabajo activo."""
    return _normalize_workspace_id(st.session_state.get("auth_workspace_id"))


def set_current_workspace_id(raw_id: str) -> str:
    """Sanitiza, persiste en st.session_state y retorna el workspace activo."""
    clean_id = _normalize_workspace_id(raw_id)
    st.session_state["auth_workspace_id"] = clean_id
    return clean_id


def is_admin_unlocked() -> bool:
    """
    Verifica si el modo operador está desbloqueado.
    Si config.ADMIN_ACCESS_CODE está vacío, retorna True automáticamente.
    """
    expected = str(getattr(config, "ADMIN_ACCESS_CODE", "") or "").strip()
    if not expected:
        return True
    return bool(st.session_state.get("admin_unlocked", False))


def verify_admin_code(input_code: str) -> bool:
    """
    Compara de forma segura (tiempo constante con hmac.compare_digest) el código
    ingresado contra config.ADMIN_ACCESS_CODE y desbloquea la sesión si coincide.
    """
    expected = str(getattr(config, "ADMIN_ACCESS_CODE", "") or "").strip()
    if not expected:
        st.session_state["admin_unlocked"] = True
        return True

    candidate = str(input_code or "").strip()
    if hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8")):
        st.session_state["admin_unlocked"] = True
        return True
    return False


def lock_admin() -> None:
    """Bloquea el acceso de operador en la sesión actual."""
    st.session_state["admin_unlocked"] = False


def check_scan_cooldown() -> Tuple[bool, int]:
    """
    Evalúa el enfriamiento entre escaneos en vivo según SCAN_COOLDOWN_SECONDS.
    Retorna (True, 0) si está habilitado o (False, segundos_restantes).
    """
    try:
        cooldown_sec = max(0, int(getattr(config, "SCAN_COOLDOWN_SECONDS", 45) or 0))
    except (TypeError, ValueError):
        cooldown_sec = 45

    if cooldown_sec == 0:
        return True, 0

    last_ts = st.session_state.get("_last_scan_ts")
    if last_ts is None:
        return True, 0

    try:
        elapsed = time.time() - float(last_ts)
    except (TypeError, ValueError):
        return True, 0

    if elapsed < 0 or elapsed >= cooldown_sec:
        return True, 0

    remaining = max(1, int(math.ceil(cooldown_sec - elapsed)))
    return False, remaining


def record_scan_triggered() -> None:
    """Registra la marca de tiempo del último escaneo en vivo ejecutado."""
    st.session_state["_last_scan_ts"] = time.time()


def sanitize_user_text(text: str, max_len: int = 280) -> str:
    """Limpia caracteres de control/saltos peligrosos, espacios extra y recorta a max_len."""
    if text is None:
        return ""
    limit = max(0, int(max_len))
    cleaned = _CTRL_CHARS_RE.sub(" ", str(text))
    cleaned = " ".join(cleaned.split())
    if limit == 0:
        return ""
    return cleaned[:limit]
