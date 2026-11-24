"""
services/__init__.py - Reexportación de servicios comerciales, financieros,
PDF, autenticación por workspace e inteligencia de mercado.
"""

from services.auth_service import (
    check_scan_cooldown,
    get_current_workspace_id,
    is_admin_unlocked,
    lock_admin,
    record_scan_triggered,
    sanitize_user_text,
    set_current_workspace_id,
    verify_admin_code,
)
from services.deal_analyzer import (
    DEFAULT_DEAL_COSTS,
    calculate_net_deal_metrics,
    enrich_dataframe_with_net_deal_and_confidence,
)
from services.market_trends import (
    compute_market_executive_summary,
    evaluate_radar_matches,
)
from services.pdf_report import generate_valuation_pdf_bytes

__all__ = [
    "DEFAULT_DEAL_COSTS",
    "calculate_net_deal_metrics",
    "enrich_dataframe_with_net_deal_and_confidence",
    "generate_valuation_pdf_bytes",
    "get_current_workspace_id",
    "set_current_workspace_id",
    "is_admin_unlocked",
    "verify_admin_code",
    "lock_admin",
    "check_scan_cooldown",
    "record_scan_triggered",
    "sanitize_user_text",
    "compute_market_executive_summary",
    "evaluate_radar_matches",
]
