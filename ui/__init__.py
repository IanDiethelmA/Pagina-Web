"""
ui/__init__.py - Reexportación unificada del sistema visual, dominio y componentes SaaS.
"""

from ui.components import (
    build_empty_state_html,
    build_hero_header_html,
    build_kpi_strip_html,
    clean_html,
    render_deal_cost_controls,
    render_empty_state,
    render_hero_header,
    render_kpi_strip,
    render_vehicle_card_html,
)
from ui.domain import (
    COMUNA_TO_REGION,
    NLP_LABELS_MAP,
    assign_portal_and_region_columns,
    extract_nlp_tags,
    format_clp,
    get_vehicle_emoji,
    normalize_model_input,
    resolve_portal_info,
    resolve_region,
)
from ui.theme import (
    THEME_MODES,
    build_theme_css,
    get_plotly_template,
    inject_theme_css,
    is_light_theme,
)

__all__ = [
    # ui.domain
    "NLP_LABELS_MAP",
    "COMUNA_TO_REGION",
    "resolve_region",
    "format_clp",
    "resolve_portal_info",
    "get_vehicle_emoji",
    "normalize_model_input",
    "assign_portal_and_region_columns",
    "extract_nlp_tags",
    # ui.theme
    "THEME_MODES",
    "is_light_theme",
    "get_plotly_template",
    "build_theme_css",
    "inject_theme_css",
    # ui.components
    "clean_html",
    "build_hero_header_html",
    "render_hero_header",
    "build_kpi_strip_html",
    "render_kpi_strip",
    "build_empty_state_html",
    "render_empty_state",
    "render_deal_cost_controls",
    "render_vehicle_card_html",
]
