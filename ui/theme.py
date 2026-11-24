"""
ui/theme.py - Sistema de diseño SaaS fintech/automotriz (Modo Oscuro / Claro WCAG AA).
"""

from __future__ import annotations

import streamlit as st

THEME_MODES = ["Oscuro", "Claro"]


def is_light_theme() -> bool:
    """Retorna True si el tema activo en sesión corresponde a Modo Claro."""
    mode = str(st.session_state.get("ui_theme_mode") or "")
    return mode == THEME_MODES[1] or "Claro" in mode


def get_plotly_template() -> str:
    """Retorna la plantilla de Plotly acorde al tema activo."""
    return "plotly_white" if is_light_theme() else "plotly_dark"


def _strip_css_lines(raw_html: str) -> str:
    """Elimina sangría inicial y líneas vacías para evitar bloques de código CommonMark."""
    return "\n".join(line.strip() for line in raw_html.splitlines() if line.strip())


def build_theme_css(light_mode: bool) -> str:
    """Construye la hoja de estilos CSS adaptativa limpia para Modo Claro u Oscuro."""
    if light_mode:
        theme_vars = """
        :root {
            --bg-app: #F8FAFC;
            --bg-surface: #FFFFFF;
            --bg-surface-alt: #F1F5F9;
            --bg-card: #FFFFFF;
            --bg-card-subtle: #F8FAFC;
            --border-subtle: #E2E8F0;
            --border-strong: #CBD5E1;
            --text-primary: #0F172A;
            --text-secondary: #475569;
            --text-muted: #64748B;
            --accent-emerald: #059669;
            --accent-emerald-text: #047857;
            --accent-ia: #0284C7;
            --accent-ia-text: #0369A1;
            --accent-amber: #D97706;
            --accent-amber-text: #B45309;
            --accent-danger-text: #B91C1C;

            --hero-bg: linear-gradient(135deg, #0F172A 0%, #134E4A 58%, #065F46 100%);
            --hero-border: rgba(15, 23, 42, 0.14);
            --hero-shadow: 0 10px 28px -8px rgba(15, 23, 42, 0.22);
            --hero-title: #FFFFFF;
            --hero-subtitle: #E2E8F0;
            --hero-pill-bg: rgba(255, 255, 255, 0.12);
            --hero-pill-border: rgba(255, 255, 255, 0.22);
            --hero-pill-text: #F8FAFC;

            --card-border: #E2E8F0;
            --card-border-left: #059669;
            --card-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 6px 16px -4px rgba(15, 23, 42, 0.06);
            --card-hover-shadow: 0 10px 24px -6px rgba(15, 23, 42, 0.12);
            --card-code-bg: #F1F5F9;
            --card-code-color: #0F172A;

            --price-grid-bg: #F8FAFC;
            --price-grid-border: #E2E8F0;
            --price-label: #475569;
            --price-pub: #0F172A;
            --price-ia: #0369A1;
            --price-target: #047857;

            --profit-bg: #ECFDF5;
            --profit-border: #A7F3D0;
            --profit-color: #047857;
            --profit-warn-bg: #FEF2F2;
            --profit-warn-border: #FECACA;
            --profit-warn-color: #B91C1C;

            --mlc-bg: #FEF3C7; --mlc-color: #92400E; --mlc-border: #F59E0B; --mlc-dot: #D97706;
            --ac-bg: #E0F2FE; --ac-color: #075985; --ac-border: #7DD3FC; --ac-dot: #0284C7;
            --yapo-bg: #FFEDD5; --yapo-color: #9A3412; --yapo-border: #FDBA74; --yapo-dot: #EA580C;
            --cl-bg: #D1FAE5; --cl-color: #065F46; --cl-border: #6EE7B7; --cl-dot: #059669;

            --comuna-bg: #F1F5F9; --comuna-color: #334155; --comuna-border: #CBD5E1;
            --drop-bg: #FEF3C7; --drop-color: #92400E; --drop-border: #FCD34D;
            --multi-bg: #E0F2FE; --multi-color: #0369A1; --multi-border: #BAE6FD;
            --days-bg: #F1F5F9; --days-color: #475569; --days-border: #E2E8F0;
            --crm-bg: #EDE9FE; --crm-color: #5B21B6; --crm-border: #C4B5FD;
            --tier-bg: #ECFDF5; --tier-color: #047857; --tier-border: #A7F3D0;
            --liq-bg: #F1F5F9; --liq-color: #334155; --liq-border: #CBD5E1;

            --conf-high-bg: #ECFDF5; --conf-high-color: #047857; --conf-high-border: #A7F3D0;
            --conf-med-bg: #FEF3C7; --conf-med-color: #92400E; --conf-med-border: #FDE68A;
            --conf-low-bg: #FEE2E2; --conf-low-color: #991B1B; --conf-low-border: #FECACA;
        }
        """
        mode_overrides = """
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(248, 250, 252, 0.94) !important;
            border-bottom: 1px solid #E2E8F0 !important;
            color: #0F172A !important;
        }
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
            color: #0F172A !important;
        }
        .stApp [data-testid="stMarkdownContainer"] h1,
        .stApp [data-testid="stMarkdownContainer"] h2,
        .stApp [data-testid="stMarkdownContainer"] h3,
        .stApp [data-testid="stMarkdownContainer"] h4,
        .stApp [data-testid="stMarkdownContainer"] h5,
        .stApp [data-testid="stMarkdownContainer"] h6,
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricLabel"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            color: #0F172A !important;
        }
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #475569 !important;
        }
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] label p,
        [data-testid="stCheckbox"] label,
        [data-testid="stCheckbox"] label p,
        [data-testid="stToggle"] label,
        [data-testid="stToggle"] label p {
            color: #0F172A !important;
        }
        [data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary p {
            color: #0F172A !important;
        }
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"],
        .stTextInput input,
        .stNumberInput input,
        .stTextArea textarea {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border-color: #CBD5E1 !important;
        }
        [data-baseweb="select"] input,
        [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
            color: #0F172A !important;
        }
        [data-baseweb="select"] > div > div > div:not([data-baseweb="tag"]) {
            color: #0F172A !important;
        }
        [data-baseweb="tag"] {
            border-radius: 6px !important;
        }
        [data-baseweb="tag"] span,
        [data-baseweb="tag"] svg {
            color: #FFFFFF !important;
            fill: currentColor !important;
        }
        [data-baseweb="popover"],
        [data-testid="stPopoverBody"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.14) !important;
        }
        [data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] h3,
        [data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] h4,
        [data-testid="stPopoverBody"] [data-testid="stWidgetLabel"],
        [data-testid="stPopoverBody"] [data-testid="stWidgetLabel"] p,
        [data-testid="stPopoverBody"] label {
            color: #0F172A !important;
        }
        [data-testid="stPopoverBody"] [data-testid="stCaptionContainer"],
        [data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] p {
            color: #475569 !important;
        }
        [data-baseweb="menu"],
        ul[role="listbox"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #E2E8F0 !important;
        }
        li[role="option"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background-color: #F1F5F9 !important;
            color: #0F172A !important;
        }
        button[kind="secondary"],
        [data-testid="stBaseButton-secondary"],
        [data-testid="stBaseButton-tertiary"],
        [data-testid="stPopover"] > button {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
        }
        button[kind="secondary"] p,
        [data-testid="stBaseButton-secondary"] p,
        [data-testid="stPopover"] > button p {
            color: #0F172A !important;
        }
        button[kind="secondary"]:hover,
        [data-testid="stBaseButton-secondary"]:hover,
        [data-testid="stPopover"] > button:hover {
            background-color: #F1F5F9 !important;
            border-color: #94A3B8 !important;
        }
        button[kind="primary"],
        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
            border: 1px solid #047857 !important;
            color: #FFFFFF !important;
        }
        button[kind="primary"] p,
        [data-testid="stBaseButton-primary"] p {
            color: #FFFFFF !important;
        }
        [data-testid="stButtonGroup"] button[aria-checked="true"],
        [data-testid="stBaseButton-segmented_controlActive"] {
            background-color: #047857 !important;
            color: #FFFFFF !important;
            border-color: #047857 !important;
        }
        [data-testid="stButtonGroup"] button[aria-checked="true"] p,
        [data-testid="stBaseButton-segmented_controlActive"] p {
            color: #FFFFFF !important;
        }
        [data-testid="stButtonGroup"] button[aria-checked="false"],
        [data-testid="stBaseButton-segmented_control"] {
            background-color: #FFFFFF !important;
            color: #0F172A !important;
            border-color: #CBD5E1 !important;
        }
        [data-testid="stButtonGroup"] button[aria-checked="false"] p,
        [data-testid="stBaseButton-segmented_control"] p {
            color: #0F172A !important;
        }
        [data-testid="stAlert"],
        [data-testid="stStatusWidget"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            color: #0F172A !important;
        }
        [data-testid="stAlert"] p,
        [data-testid="stStatusWidget"] p {
            color: #0F172A !important;
        }
        [data-testid="stNumberInput"] button {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
            border-color: #CBD5E1 !important;
        }
        [data-testid="stSlider"] [data-testid="stThumbValue"],
        [data-testid="stSlider"] [data-testid="stTickBarMin"],
        [data-testid="stSlider"] [data-testid="stTickBarMax"] {
            color: #475569 !important;
        }
        .stApp .hero-banner .hero-title { color: #FFFFFF !important; }
        .stApp .hero-banner .hero-subtitle { color: #E2E8F0 !important; }
        .stApp .hero-banner .hero-pill { color: #F8FAFC !important; }
        .stApp .opp-card .car-title { color: var(--text-primary) !important; }
        .stApp .opp-card .car-subtitle-trim { color: var(--text-secondary) !important; }
        .stApp .opp-card .car-meta { color: var(--text-secondary) !important; }
        .stApp .opp-card .price-val-pub { color: var(--price-pub) !important; }
        .stApp .opp-card .price-val-ia { color: var(--price-ia) !important; }
        .stApp .opp-card .price-val-target { color: var(--price-target) !important; }
        .stApp .opp-card .profit-banner,
        .stApp .opp-card .profit-banner span { color: var(--profit-color) !important; }
        .stApp .opp-card .profit-banner.is-negative,
        .stApp .opp-card .profit-banner.is-negative span { color: var(--profit-warn-color) !important; }
        .stApp .opp-card .btn-portal { color: #FFFFFF !important; }
        """
    else:
        theme_vars = """
        :root {
            --bg-app: #0B1120;
            --bg-surface: #111827;
            --bg-surface-alt: #0F172A;
            --bg-card: #1E293B;
            --bg-card-subtle: #162032;
            --border-subtle: rgba(148, 163, 184, 0.16);
            --border-strong: rgba(148, 163, 184, 0.28);
            --text-primary: #F8FAFC;
            --text-secondary: #CBD5E1;
            --text-muted: #94A3B8;
            --accent-emerald: #10B981;
            --accent-emerald-text: #34D399;
            --accent-ia: #38BDF8;
            --accent-ia-text: #38BDF8;
            --accent-amber: #F59E0B;
            --accent-amber-text: #FBBF24;
            --accent-danger-text: #FCA5A5;

            --hero-bg: linear-gradient(135deg, #0F172A 0%, #111827 55%, #064E3B 100%);
            --hero-border: rgba(56, 189, 248, 0.20);
            --hero-shadow: 0 12px 30px -8px rgba(0, 0, 0, 0.45);
            --hero-title: #F8FAFC;
            --hero-subtitle: #CBD5E1;
            --hero-pill-bg: rgba(15, 23, 42, 0.55);
            --hero-pill-border: rgba(148, 163, 184, 0.24);
            --hero-pill-text: #E2E8F0;

            --card-border: rgba(148, 163, 184, 0.18);
            --card-border-left: #10B981;
            --card-shadow: 0 4px 18px rgba(0, 0, 0, 0.28);
            --card-hover-shadow: 0 10px 26px rgba(16, 185, 129, 0.16);
            --card-code-bg: rgba(148, 163, 184, 0.14);
            --card-code-color: #E2E8F0;

            --price-grid-bg: rgba(11, 17, 32, 0.55);
            --price-grid-border: rgba(148, 163, 184, 0.14);
            --price-label: #94A3B8;
            --price-pub: #F8FAFC;
            --price-ia: #38BDF8;
            --price-target: #34D399;

            --profit-bg: rgba(16, 185, 129, 0.14);
            --profit-border: rgba(16, 185, 129, 0.38);
            --profit-color: #34D399;
            --profit-warn-bg: rgba(239, 68, 68, 0.14);
            --profit-warn-border: rgba(248, 113, 113, 0.40);
            --profit-warn-color: #FCA5A5;

            --mlc-bg: rgba(245, 158, 11, 0.16); --mlc-color: #FDE047; --mlc-border: rgba(245, 158, 11, 0.38); --mlc-dot: #FACC15;
            --ac-bg: rgba(56, 189, 248, 0.16); --ac-color: #7DD3FC; --ac-border: rgba(56, 189, 248, 0.38); --ac-dot: #38BDF8;
            --yapo-bg: rgba(249, 115, 22, 0.16); --yapo-color: #FDBA74; --yapo-border: rgba(249, 115, 22, 0.38); --yapo-dot: #FB923C;
            --cl-bg: rgba(16, 185, 129, 0.16); --cl-color: #6EE7B7; --cl-border: rgba(16, 185, 129, 0.38); --cl-dot: #10B981;

            --comuna-bg: rgba(148, 163, 184, 0.14); --comuna-color: #CBD5E1; --comuna-border: rgba(148, 163, 184, 0.24);
            --drop-bg: rgba(245, 158, 11, 0.16); --drop-color: #FCD34D; --drop-border: rgba(245, 158, 11, 0.40);
            --multi-bg: rgba(56, 189, 248, 0.14); --multi-color: #7DD3FC; --multi-border: rgba(56, 189, 248, 0.32);
            --days-bg: rgba(100, 116, 139, 0.22); --days-color: #CBD5E1; --days-border: rgba(148, 163, 184, 0.20);
            --crm-bg: rgba(139, 92, 246, 0.18); --crm-color: #DDD6FE; --crm-border: rgba(167, 139, 250, 0.40);
            --tier-bg: rgba(16, 185, 129, 0.16); --tier-color: #6EE7B7; --tier-border: rgba(16, 185, 129, 0.36);
            --liq-bg: rgba(56, 189, 248, 0.12); --liq-color: #BAE6FD; --liq-border: rgba(56, 189, 248, 0.25);

            --conf-high-bg: rgba(16, 185, 129, 0.16); --conf-high-color: #6EE7B7; --conf-high-border: rgba(16, 185, 129, 0.40);
            --conf-med-bg: rgba(245, 158, 11, 0.16); --conf-med-color: #FCD34D; --conf-med-border: rgba(245, 158, 11, 0.40);
            --conf-low-bg: rgba(239, 68, 68, 0.16); --conf-low-color: #FCA5A5; --conf-low-border: rgba(248, 113, 113, 0.40);
        }
        """
        mode_overrides = """
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0B1120 !important;
            color: #F8FAFC !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(11, 17, 32, 0.90) !important;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12) !important;
        }
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
            background-color: #111827 !important;
            border-right: 1px solid rgba(148, 163, 184, 0.14) !important;
        }
        [data-testid="stExpander"] {
            background-color: #111827 !important;
            border: 1px solid rgba(148, 163, 184, 0.16) !important;
            border-radius: 12px !important;
        }
        [data-baseweb="popover"],
        [data-testid="stPopoverBody"] {
            background-color: #111827 !important;
            border: 1px solid rgba(148, 163, 184, 0.22) !important;
            border-radius: 12px !important;
        }
        button[kind="primary"],
        [data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
            border: 1px solid #10B981 !important;
            color: #FFFFFF !important;
        }
        [data-testid="stButtonGroup"] button[aria-checked="true"],
        [data-testid="stBaseButton-segmented_controlActive"] {
            background-color: #059669 !important;
            color: #FFFFFF !important;
            border-color: #10B981 !important;
        }
        [data-testid="stButtonGroup"] button[aria-checked="false"],
        [data-testid="stBaseButton-segmented_control"] {
            background-color: #111827 !important;
            color: #CBD5E1 !important;
            border-color: rgba(148, 163, 184, 0.22) !important;
        }
        """

    base_css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    {theme_vars}
    {mode_overrides}
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebar"] {{
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    [data-testid="stMetricValue"],
    [data-testid="stDataFrame"],
    .saas-kpi-value,
    .price-val-pub,
    .price-val-ia,
    .price-val-target,
    .profit-banner,
    .net-breakdown-row,
    table {{
        font-variant-numeric: tabular-nums;
        font-feature-settings: "tnum" 1;
    }}
    .block-container {{
        padding-top: 1.15rem;
        padding-bottom: 2.5rem;
        max-width: 1440px;
    }}
    .hero-banner {{
        background: var(--hero-bg);
        border: 1px solid var(--hero-border);
        border-radius: 16px;
        padding: 1.25rem 1.65rem;
        margin-bottom: 1.1rem;
        color: var(--hero-title);
        box-shadow: var(--hero-shadow);
    }}
    .hero-title {{
        font-size: 1.52rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.025em;
        line-height: 1.2;
        color: var(--hero-title);
    }}
    .hero-subtitle {{
        font-size: 0.90rem;
        color: var(--hero-subtitle);
        margin-top: 0.35rem;
        margin-bottom: 0;
        line-height: 1.45;
    }}
    .hero-pill-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.78rem;
    }}
    .hero-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        background: var(--hero-pill-bg);
        border: 1px solid var(--hero-pill-border);
        color: var(--hero-pill-text);
        padding: 0.24rem 0.68rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        font-variant-numeric: tabular-nums;
    }}
    .saas-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.75rem;
        margin: 0.6rem 0 1.05rem 0;
    }}
    .saas-kpi-card {{
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-top: 3px solid var(--border-strong);
        border-radius: 12px;
        padding: 0.85rem 0.95rem;
        box-shadow: var(--card-shadow);
        min-width: 0;
    }}
    .saas-kpi-card.tone-emerald {{
        border-top-color: var(--accent-emerald);
    }}
    .saas-kpi-card.tone-blue {{
        border-top-color: var(--accent-ia);
    }}
    .saas-kpi-card.tone-amber {{
        border-top-color: var(--accent-amber);
    }}
    .saas-kpi-card.tone-neutral {{
        border-top-color: var(--border-strong);
    }}
    .saas-kpi-label {{
        font-size: 0.71rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--text-muted);
        margin-bottom: 0.28rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .saas-kpi-value {{
        font-size: 1.28rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        line-height: 1.2;
        color: var(--text-primary);
    }}
    .saas-kpi-card.tone-emerald .saas-kpi-value {{
        color: var(--accent-emerald-text);
    }}
    .saas-kpi-card.tone-blue .saas-kpi-value {{
        color: var(--accent-ia-text);
    }}
    .saas-kpi-card.tone-amber .saas-kpi-value {{
        color: var(--accent-amber-text);
    }}
    .saas-kpi-sub {{
        font-size: 0.73rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin-top: 0.25rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .opp-card {{
        background: var(--bg-card);
        border: 1px solid var(--card-border);
        border-left: 4px solid var(--card-border-left);
        border-radius: 14px;
        padding: 1.08rem 1.12rem;
        margin-bottom: 0.85rem;
        color: var(--text-primary);
        box-shadow: var(--card-shadow);
        transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
    }}
    .opp-card:hover {{
        transform: translateY(-2px);
        box-shadow: var(--card-hover-shadow);
    }}
    .opp-card.tier-elite {{
        border-left: 5px solid var(--accent-emerald);
    }}
    .opp-card.tier-good {{
        border-left: 4px solid var(--accent-ia);
    }}
    .opp-card.tier-crm {{
        border-left: 4px solid var(--accent-amber);
    }}
    .opp-card code {{
        background: var(--card-code-bg);
        color: var(--card-code-color);
        padding: 0.10rem 0.36rem;
        border-radius: 5px;
        font-size: 0.74rem;
        font-weight: 600;
    }}
    .car-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.5rem;
        margin-bottom: 0.42rem;
    }}
    .car-title {{
        font-size: 1.06rem;
        font-weight: 700;
        letter-spacing: -0.015em;
        line-height: 1.25;
        color: var(--text-primary);
        margin: 0;
    }}
    .car-badges {{
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        align-items: center;
        gap: 0.3rem;
        flex-shrink: 0;
    }}
    .car-subtitle-trim {{
        font-size: 0.79rem;
        font-weight: 500;
        color: var(--text-secondary);
        margin-bottom: 0.48rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .car-meta {{
        font-size: 0.79rem;
        color: var(--text-secondary);
        margin-bottom: 0.72rem;
        line-height: 1.5;
    }}
    .car-meta-row {{
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.35rem;
        margin-top: 0.32rem;
    }}
    .price-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem;
        background: var(--price-grid-bg);
        border: 1px solid var(--price-grid-border);
        padding: 0.68rem 0.78rem;
        border-radius: 10px;
        margin-bottom: 0.68rem;
    }}
    .price-label {{
        font-size: 0.67rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--price-label);
        margin-bottom: 0.12rem;
    }}
    .price-val-pub {{
        font-size: 1.08rem;
        font-weight: 700;
        color: var(--price-pub);
    }}
    .price-val-ia {{
        font-size: 1.08rem;
        font-weight: 700;
        color: var(--price-ia);
    }}
    .price-val-target {{
        font-size: 1.22rem;
        font-weight: 800;
        color: var(--price-target);
    }}
    .profit-banner {{
        background: var(--profit-bg);
        border: 1px solid var(--profit-border);
        color: var(--profit-color);
        padding: 0.54rem 0.82rem;
        border-radius: 9px;
        font-weight: 700;
        font-size: 0.88rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.58rem;
    }}
    .profit-banner.is-negative {{
        background: var(--profit-warn-bg);
        border-color: var(--profit-warn-border);
        color: var(--profit-warn-color);
    }}
    .net-breakdown-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        color: var(--text-secondary);
        padding: 0.28rem 0.1rem;
        border-bottom: 1px dashed var(--border-subtle);
        margin-bottom: 0.58rem;
    }}
    .btn-portal {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        text-align: center;
        background: linear-gradient(135deg, #047857 0%, #059669 100%);
        color: #FFFFFF !important;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 0.84rem;
        padding: 0.54rem 0.9rem;
        border-radius: 9px;
        transition: opacity 0.15s ease, transform 0.12s ease;
    }}
    .btn-portal:hover {{
        opacity: 0.94;
    }}
    .portal-badge-mlc,
    .portal-badge-ac,
    .portal-badge-yapo,
    .portal-badge-cl {{
        display: inline-flex;
        align-items: center;
        gap: 0.32rem;
        padding: 0.15rem 0.54rem;
        border-radius: 999px;
        font-size: 0.69rem;
        font-weight: 700;
        white-space: nowrap;
    }}
    .portal-badge-mlc::before,
    .portal-badge-ac::before,
    .portal-badge-yapo::before,
    .portal-badge-cl::before {{
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }}
    .portal-badge-mlc {{
        background: var(--mlc-bg); color: var(--mlc-color); border: 1px solid var(--mlc-border);
    }}
    .portal-badge-mlc::before {{ background: var(--mlc-dot); }}
    .portal-badge-ac {{
        background: var(--ac-bg); color: var(--ac-color); border: 1px solid var(--ac-border);
    }}
    .portal-badge-ac::before {{ background: var(--ac-dot); }}
    .portal-badge-yapo {{
        background: var(--yapo-bg); color: var(--yapo-color); border: 1px solid var(--yapo-border);
    }}
    .portal-badge-yapo::before {{ background: var(--yapo-dot); }}
    .portal-badge-cl {{
        background: var(--cl-bg); color: var(--cl-color); border: 1px solid var(--cl-border);
    }}
    .portal-badge-cl::before {{ background: var(--cl-dot); }}
    .conf-badge {{
        display: inline-flex;
        align-items: center;
        padding: 0.13rem 0.50rem;
        border-radius: 999px;
        font-size: 0.69rem;
        font-weight: 700;
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }}
    .conf-badge.conf-high {{
        background: var(--conf-high-bg); color: var(--conf-high-color); border: 1px solid var(--conf-high-border);
    }}
    .conf-badge.conf-med {{
        background: var(--conf-med-bg); color: var(--conf-med-color); border: 1px solid var(--conf-med-border);
    }}
    .conf-badge.conf-low {{
        background: var(--conf-low-bg); color: var(--conf-low-color); border: 1px solid var(--conf-low-border);
    }}
    .comuna-pill {{
        display: inline-block;
        background: var(--comuna-bg);
        color: var(--comuna-color);
        border: 1px solid var(--comuna-border);
        padding: 0.12rem 0.48rem;
        border-radius: 999px;
        font-size: 0.69rem;
        font-weight: 600;
        white-space: nowrap;
    }}
    .drop-badge {{
        display: inline-block;
        background: var(--drop-bg);
        color: var(--drop-color);
        border: 1px solid var(--drop-border);
        padding: 0.12rem 0.50rem;
        border-radius: 999px;
        font-size: 0.69rem;
        font-weight: 700;
        white-space: nowrap;
    }}
    .multi-badge {{
        display: inline-block;
        background: var(--multi-bg);
        color: var(--multi-color);
        border: 1px solid var(--multi-border);
        padding: 0.12rem 0.50rem;
        border-radius: 999px;
        font-size: 0.69rem;
        font-weight: 600;
    }}
    .days-pill {{
        display: inline-block;
        background: var(--days-bg);
        color: var(--days-color);
        border: 1px solid var(--days-border);
        padding: 0.11rem 0.45rem;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 600;
        white-space: nowrap;
    }}
    .crm-pill {{
        display: inline-block;
        background: var(--crm-bg);
        color: var(--crm-color);
        border: 1px solid var(--crm-border);
        padding: 0.14rem 0.52rem;
        border-radius: 999px;
        font-size: 0.69rem;
        font-weight: 700;
        white-space: nowrap;
    }}
    .tier-pill {{
        display: inline-block;
        background: var(--tier-bg);
        color: var(--tier-color);
        border: 1px solid var(--tier-border);
        padding: 0.14rem 0.50rem;
        border-radius: 999px;
        font-size: 0.69rem;
        font-weight: 700;
        white-space: nowrap;
    }}
    .liq-pill {{
        display: inline-block;
        background: var(--liq-bg);
        color: var(--liq-color);
        border: 1px solid var(--liq-border);
        padding: 0.12rem 0.48rem;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 600;
        white-space: nowrap;
    }}
    .empty-state-box {{
        background: var(--bg-card);
        border: 1px dashed var(--border-strong);
        border-radius: 14px;
        padding: 2.1rem 1.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
    }}
    .empty-state-icon {{
        font-size: 1.85rem;
        margin-bottom: 0.45rem;
        line-height: 1;
    }}
    .empty-state-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.32rem;
    }}
    .empty-state-subtitle {{
        font-size: 0.86rem;
        color: var(--text-secondary);
        max-width: 540px;
        margin: 0 auto;
        line-height: 1.45;
    }}
    @media (max-width: 768px) {{
        .block-container {{
            padding-top: 0.65rem;
            padding-left: 0.75rem;
            padding-right: 0.75rem;
            padding-bottom: 1.75rem;
        }}
        .hero-banner {{
            padding: 0.95rem 1.05rem;
            border-radius: 14px;
            margin-bottom: 0.85rem;
        }}
        .hero-title {{
            font-size: 1.18rem;
        }}
        .hero-subtitle {{
            font-size: 0.82rem;
        }}
        .hero-pill-row {{
            gap: 0.35rem;
            margin-top: 0.6rem;
        }}
        .hero-pill {{
            font-size: 0.71rem;
            padding: 0.20rem 0.54rem;
        }}
        .saas-kpi-grid {{
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.5rem 0 0.85rem 0;
        }}
        .saas-kpi-card {{
            padding: 0.70rem 0.78rem;
        }}
        .saas-kpi-value {{
            font-size: 1.08rem;
        }}
        .opp-card {{
            padding: 0.88rem 0.92rem;
            border-radius: 12px;
        }}
        .car-header {{
            flex-wrap: wrap;
            gap: 0.35rem;
        }}
        .car-badges {{
            justify-content: flex-start;
        }}
        .price-grid {{
            gap: 0.5rem;
            padding: 0.58rem 0.65rem;
        }}
        .price-val-pub,
        .price-val-ia {{
            font-size: 0.98rem;
        }}
        .profit-banner {{
            flex-wrap: wrap;
            font-size: 0.82rem;
            padding: 0.48rem 0.68rem;
        }}
        .btn-portal {{
            min-height: 42px;
            font-size: 0.86rem;
            padding: 0.60rem 0.85rem;
        }}
        .stButton > button,
        [data-testid="stPopover"] > button,
        [data-testid="stDownloadButton"] > button {{
            min-height: 40px;
        }}
    }}
    </style>
    """
    return _strip_css_lines(base_css)


def inject_theme_css(light_mode: bool) -> None:
    """Inyecta el CSS adaptativo del sistema visual SaaS en Streamlit."""
    st.markdown(build_theme_css(light_mode), unsafe_allow_html=True)
