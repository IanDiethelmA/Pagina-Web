"""
app.py - MÓDULO 5: Suite Avanzada de Arbitraje Automotriz e Inteligencia de Mercado (Streamlit + Plotly).
Conecta en tiempo real con el motor backend del Sistema de Arbitraje Automotriz de Chile:
- Base de Datos SQLite WAL ('data/chileautos_arbitraje.db': tablas 'autos_historico' y 'precio_historial').
- Datasets ('data/autos_clean.csv' y 'data/oportunidades_detectadas.csv').
- Modelo IA LightGBM + Target Encoding Jerárquico ('models/model.pkl').
- Scraper Multi-Portal en vivo (MercadoLibre 'MLC-', Autocosmos 'AC-', Yapo 'YAPO-', Chileautos 'CL-').
- Filtros avanzados unificados en Búsqueda en Vivo, Oportunidades, Curvas y Explorador General.
- Visualizaciones complejas: Curvas Multi-Capa, Dispersión 3D (Año-Km-Precio), BoxPlots por Generación/Portal,
  Heatmaps de Rentabilidad, Treemap Jerárquico y Matriz ROI (Capital vs Margen).
"""

from contextlib import closing
import html
import importlib
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from database import DatabaseManager
from pipeline import CarDataPipeline
from train import CarPricePredictor, HierarchicalCarEncoder

# Asegurar compatibilidad de deserialización para joblib en el namespace de Streamlit
setattr(sys.modules["__main__"], "HierarchicalCarEncoder", HierarchicalCarEncoder)

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA Y ESTILOS UI/UX (MODO ANCHO)
# ==============================================================================
st.set_page_config(
    page_title="Arbitraje Automotriz IA Chile | Suite Analítica Multi-Portal",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

THEME_MODES = ["🌙 Oscuro", "☀️ Claro"]


def is_light_theme() -> bool:
    """Retorna True si el usuario seleccionó el Modo Claro en la barra lateral."""
    return st.session_state.get("ui_theme_mode") == THEME_MODES[1]


def get_plotly_template() -> str:
    """Retorna la plantilla de Plotly acorde al tema activo (Claro u Oscuro)."""
    return "plotly_white" if is_light_theme() else "plotly_dark"


def _clean_html(raw_html: str) -> str:
    """
    Elimina espacios iniciales y líneas vacías en bloques HTML para evitar que
    el parser CommonMark de Streamlit interprete líneas indentadas como código.
    """
    return "\n".join(line.strip() for line in raw_html.splitlines() if line.strip())


def inject_theme_css(light_mode: bool) -> None:
    """Inyecta el CSS adaptativo para Modo Oscuro o Modo Claro."""
    if light_mode:
        theme_vars = """
        :root {
            --hero-bg: linear-gradient(135deg, #0f766e 0%, #0369a1 60%, #1e3a8a 100%);
            --hero-border: rgba(15, 23, 42, 0.12);
            --hero-shadow: 0 10px 24px -6px rgba(15, 23, 42, 0.18);
            --hero-title: #ffffff;
            --hero-subtitle: #e0f2fe;

            --card-bg: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            --card-border: rgba(16, 185, 129, 0.45);
            --card-border-left: #059669;
            --card-text: #0f172a;
            --card-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
            --card-hover-shadow: 0 8px 22px rgba(5, 150, 105, 0.18);
            --card-title: #0f172a;
            --card-subtitle: #475569;
            --card-meta: #475569;
            --card-code-bg: #e2e8f0;
            --card-code-color: #0f172a;

            --price-grid-bg: #f1f5f9;
            --price-label: #64748b;
            --price-pub: #0f172a;
            --price-ia: #0284c7;
            --price-target: #047857;

            --profit-bg: rgba(16, 185, 129, 0.12);
            --profit-border: rgba(5, 150, 105, 0.40);
            --profit-color: #047857;

            --mlc-bg: #fef3c7; --mlc-color: #92400e; --mlc-border: #f59e0b;
            --ac-bg: #e0f2fe; --ac-color: #0369a1; --ac-border: #38bdf8;
            --yapo-bg: #ffedd5; --yapo-color: #9a3412; --yapo-border: #fb923c;
            --cl-bg: #d1fae5; --cl-color: #065f46; --cl-border: #10b981;

            --comuna-bg: #e2e8f0; --comuna-color: #334155; --comuna-border: #cbd5e1;
            --drop-bg: #fef3c7; --drop-color: #b45309; --drop-border: #f59e0b;
            --multi-color: #0284c7;
            --days-bg: #e2e8f0; --days-color: #334155;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #f8fafc !important;
            color: #0f172a !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(248, 250, 252, 0.92) !important;
            color: #0f172a !important;
        }
        [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
            background-color: #f1f5f9 !important;
            border-right: 1px solid #e2e8f0 !important;
            color: #0f172a !important;
        }
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
        .stApp p, .stApp label, .stApp li,
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] span,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] {
            color: #0f172a !important;
        }
        .stApp .hero-banner .hero-title { color: #ffffff !important; }
        .stApp .hero-banner .hero-subtitle, .stApp .hero-banner b { color: #e0f2fe !important; }
        .stApp .opp-card .car-title { color: var(--card-title) !important; }
        .stApp .opp-card .car-subtitle-trim { color: var(--card-subtitle) !important; }
        .stApp .opp-card .car-meta { color: var(--card-meta) !important; }
        .stApp .opp-card .profit-banner, .stApp .opp-card .profit-banner span { color: var(--profit-color) !important; }
        .stApp .opp-card .btn-portal { color: #ffffff !important; }
        .stApp .opp-card .price-val-ia { color: var(--price-ia) !important; }
        .stApp .opp-card .price-val-target { color: var(--price-target) !important; }

        [data-testid="stExpander"] {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 10px !important;
        }
        [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {
            color: #0f172a !important;
        }
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"],
        .stTextInput input,
        .stNumberInput input {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border-color: #cbd5e1 !important;
        }
        [data-baseweb="popover"], [data-baseweb="menu"], ul[role="listbox"], li[role="option"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
        li[role="option"]:hover {
            background-color: #f1f5f9 !important;
        }
        button[kind="secondary"], [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-tertiary"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
        }
        button[kind="secondary"]:hover, [data-testid="stBaseButton-secondary"]:hover {
            background-color: #f1f5f9 !important;
            border-color: #94a3b8 !important;
        }
        button[kind="primary"], [data-testid="stBaseButton-primary"] {
            color: #ffffff !important;
        }
        button[kind="primary"] p, [data-testid="stBaseButton-primary"] p {
            color: #ffffff !important;
        }
        [data-testid="stButtonGroup"] button[aria-checked="true"],
        [data-testid="stBaseButton-segmented_controlActive"] {
            background-color: #0f766e !important;
            color: #ffffff !important;
            border-color: #0f766e !important;
        }
        [data-testid="stButtonGroup"] button[aria-checked="true"] p,
        [data-testid="stBaseButton-segmented_controlActive"] p {
            color: #ffffff !important;
        }
        [data-testid="stButtonGroup"] button[aria-checked="false"],
        [data-testid="stBaseButton-segmented_control"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border-color: #cbd5e1 !important;
        }
        [data-testid="stAlert"] {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
        }
        [data-testid="stAlert"] p {
            color: #0f172a !important;
        }
        """
    else:
        theme_vars = """
        :root {
            --hero-bg: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0f766e 100%);
            --hero-border: rgba(255, 255, 255, 0.12);
            --hero-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
            --hero-title: #ffffff;
            --hero-subtitle: #cbd5e1;

            --card-bg: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            --card-border: rgba(16, 185, 129, 0.38);
            --card-border-left: #10b981;
            --card-text: #f8fafc;
            --card-shadow: 0 4px 16px rgba(0, 0, 0, 0.22);
            --card-hover-shadow: 0 8px 24px rgba(16, 185, 129, 0.25);
            --card-title: #ffffff;
            --card-subtitle: #cbd5e1;
            --card-meta: #94a3b8;
            --card-code-bg: rgba(148, 163, 184, 0.16);
            --card-code-color: #e2e8f0;

            --price-grid-bg: rgba(255, 255, 255, 0.04);
            --price-label: #94a3b8;
            --price-pub: #f8fafc;
            --price-ia: #38bdf8;
            --price-target: #34d399;

            --profit-bg: rgba(16, 185, 129, 0.16);
            --profit-border: rgba(16, 185, 129, 0.45);
            --profit-color: #34d399;

            --mlc-bg: rgba(250, 204, 21, 0.18); --mlc-color: #fde047; --mlc-border: rgba(250, 204, 21, 0.45);
            --ac-bg: rgba(56, 189, 248, 0.18); --ac-color: #7dd3fc; --ac-border: rgba(56, 189, 248, 0.45);
            --yapo-bg: rgba(249, 115, 22, 0.18); --yapo-color: #fdba74; --yapo-border: rgba(249, 115, 22, 0.45);
            --cl-bg: rgba(16, 185, 129, 0.18); --cl-color: #6ee7b7; --cl-border: rgba(16, 185, 129, 0.45);

            --comuna-bg: rgba(148, 163, 184, 0.15); --comuna-color: #cbd5e1; --comuna-border: rgba(148, 163, 184, 0.3);
            --drop-bg: rgba(245, 158, 11, 0.18); --drop-color: #fcd34d; --drop-border: rgba(245, 158, 11, 0.45);
            --multi-color: #38bdf8;
            --days-bg: rgba(100, 116, 139, 0.22); --days-color: #cbd5e1;
        }
        """

    base_css = f"""
    <style>
    {theme_vars}
    .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
    }}
    .hero-banner {{
        background: var(--hero-bg);
        border: 1px solid var(--hero-border);
        border-radius: 16px;
        padding: 1.25rem 1.75rem;
        margin-bottom: 1.2rem;
        color: var(--hero-title);
        box-shadow: var(--hero-shadow);
    }}
    .hero-title {{
        font-size: 1.68rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        color: var(--hero-title);
    }}
    .hero-subtitle {{
        font-size: 0.92rem;
        color: var(--hero-subtitle);
        margin-top: 0.3rem;
        margin-bottom: 0;
    }}
    .opp-card {{
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-left: 5px solid var(--card-border-left);
        border-radius: 14px;
        padding: 1.15rem;
        margin-bottom: 1.05rem;
        color: var(--card-text);
        box-shadow: var(--card-shadow);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .opp-card:hover {{
        transform: translateY(-3px);
        box-shadow: var(--card-hover-shadow);
        border-color: var(--card-border-left);
    }}
    .opp-card code {{
        background: var(--card-code-bg);
        color: var(--card-code-color);
        padding: 0.08rem 0.32rem;
        border-radius: 4px;
        font-size: 0.76rem;
    }}
    .car-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.5rem;
        margin-bottom: 0.55rem;
    }}
    .car-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--card-title);
        margin: 0;
    }}
    .car-meta {{
        font-size: 0.81rem;
        color: var(--card-meta);
        margin-bottom: 0.75rem;
        line-height: 1.45;
    }}
    .price-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem;
        background: var(--price-grid-bg);
        padding: 0.68rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
    }}
    .price-label {{
        font-size: 0.69rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--price-label);
    }}
    .price-val-pub {{
        font-size: 1.06rem;
        font-weight: 700;
        color: var(--price-pub);
    }}
    .price-val-ia {{
        font-size: 1.06rem;
        font-weight: 700;
        color: var(--price-ia);
    }}
    .price-val-target {{
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--price-target);
    }}
    .profit-banner {{
        background: var(--profit-bg);
        border: 1px solid var(--profit-border);
        color: var(--profit-color);
        padding: 0.52rem 0.8rem;
        border-radius: 9px;
        font-weight: 700;
        font-size: 0.92rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.75rem;
    }}
    .btn-portal {{
        display: block;
        text-align: center;
        background: linear-gradient(90deg, #059669 0%, #10b981 100%);
        color: #ffffff !important;
        text-decoration: none !important;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 0.52rem 0.85rem;
        border-radius: 8px;
        transition: opacity 0.15s ease;
    }}
    .btn-portal:hover {{
        opacity: 0.92;
    }}
    .portal-badge-mlc {{
        background: var(--mlc-bg);
        color: var(--mlc-color);
        border: 1px solid var(--mlc-border);
        padding: 0.15rem 0.52rem;
        border-radius: 999px;
        font-size: 0.71rem;
        font-weight: 700;
        white-space: nowrap;
    }}
    .portal-badge-ac {{
        background: var(--ac-bg);
        color: var(--ac-color);
        border: 1px solid var(--ac-border);
        padding: 0.15rem 0.52rem;
        border-radius: 999px;
        font-size: 0.71rem;
        font-weight: 700;
        white-space: nowrap;
    }}
    .portal-badge-yapo {{
        background: var(--yapo-bg);
        color: var(--yapo-color);
        border: 1px solid var(--yapo-border);
        padding: 0.15rem 0.52rem;
        border-radius: 999px;
        font-size: 0.71rem;
        font-weight: 700;
        white-space: nowrap;
    }}
    .portal-badge-cl {{
        background: var(--cl-bg);
        color: var(--cl-color);
        border: 1px solid var(--cl-border);
        padding: 0.15rem 0.52rem;
        border-radius: 999px;
        font-size: 0.71rem;
        font-weight: 700;
        white-space: nowrap;
    }}
    .comuna-pill {{
        background: var(--comuna-bg);
        color: var(--comuna-color);
        border: 1px solid var(--comuna-border);
        padding: 0.14rem 0.5rem;
        border-radius: 999px;
        font-size: 0.71rem;
        font-weight: 600;
        white-space: nowrap;
    }}
    .car-subtitle-trim {{
        font-size: 0.78rem;
        color: var(--card-subtitle);
        margin-bottom: 0.4rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .drop-badge {{
        display: inline-block;
        background: var(--drop-bg);
        color: var(--drop-color);
        border: 1px solid var(--drop-border);
        padding: 0.12rem 0.48rem;
        border-radius: 999px;
        font-size: 0.70rem;
        font-weight: 700;
        margin-top: 0.25rem;
        margin-right: 0.25rem;
    }}
    .multi-badge {{
        font-size: 0.75rem;
        color: var(--multi-color);
        margin-top: 0.3rem;
        font-weight: 600;
    }}
    .days-pill {{
        display: inline-block;
        background: var(--days-bg);
        color: var(--days-color);
        padding: 0.10rem 0.42rem;
        border-radius: 999px;
        font-size: 0.69rem;
        font-weight: 600;
    }}
    @media (max-width: 768px) {{
        .block-container {{
            padding-top: 0.7rem;
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }}
        .hero-banner {{
            padding: 0.9rem 1.1rem;
        }}
        .hero-title {{
            font-size: 1.25rem;
        }}
        .opp-card {{
            padding: 0.9rem;
        }}
    }}
    </style>
    """
    st.markdown(_clean_html(base_css), unsafe_allow_html=True)

MAIN_SECTIONS = [
    "🔥 Oportunidades",
    "⭐ Favoritos",
    "📈 Curvas y Analítica 3D",
    "🧮 Tasador IA en Vivo",
    "🔍 Explorador y Rebajas",
]

CURVE_SUBVIEWS = [
    "📉 1. Curva 2D + Banda IA",
    "🧊 2. Superficie 3D (Año × Km × Precio)",
    "📦 3. BoxPlot por Generación y Portal",
    "🔥 4. Heatmap de Precios y Descuentos",
    "🗺️ 5. Mapa Global del Mercado (Treemap & IA)",
]

# Estado persistente de sesión para sincronizar Búsqueda en Vivo con Navegación y Filtros
DEFAULT_SESSION_KEYS = {
    "focused_marca": None,
    "focused_modelo": None,
    "last_scan_message": None,
}
for k, v in DEFAULT_SESSION_KEYS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ==============================================================================
# 2. UTILIDADES DE DOMINIO, PORTALES Y CARGA DE ARTEFACTOS
# ==============================================================================
NLP_LABELS_MAP = {
    "nlp_unico_dueno": "Único Dueño",
    "nlp_mantenciones_al_dia": "Mant. al Día",
    "nlp_cuero": "Tapiz Cuero",
    "nlp_sellado_inmaculado": "Impecable",
    "nlp_techo_panoramico": "Sunroof",
    "nlp_full_equipo": "Full Equipo",
    "nlp_laminas_seguridad": "Láminas Seg.",
    "nlp_neumaticos_nuevos": "Neum. Nuevos",
    "nlp_urgencia_conversable": "Vende por Apuro",
}

COMUNA_TO_REGION = {
    # Región de Arica y Parinacota (XV)
    "ARICA": "Arica y Parinacota", "PUTRE": "Arica y Parinacota",
    # Región de Tarapacá (I)
    "IQUIQUE": "Tarapacá", "ALTO HOSPICIO": "Tarapacá", "POZO ALMONTE": "Tarapacá",
    # Región de Antofagasta (II)
    "ANTOFAGASTA": "Antofagasta", "CALAMA": "Antofagasta", "TOCOPILLA": "Antofagasta",
    "MEJILLONES": "Antofagasta", "SAN PEDRO DE ATACAMA": "Antofagasta",
    # Región de Atacama (III)
    "COPIAPO": "Atacama", "VALLENAR": "Atacama", "CALDERA": "Atacama", "CHANARAL": "Atacama",
    # Región de Coquimbo (IV)
    "LA SERENA": "Coquimbo", "COQUIMBO": "Coquimbo", "OVALLE": "Coquimbo",
    "ILLAPEL": "Coquimbo", "SALAMANCA": "Coquimbo", "VICUNA": "Coquimbo",
    # Región de Valparaíso (V)
    "VALPARAISO": "Valparaíso", "VINA DEL MAR": "Valparaíso", "CONCON": "Valparaíso",
    "QUILPUE": "Valparaíso", "VILLA ALEMANA": "Valparaíso", "LIMACHE": "Valparaíso",
    "QUILLOTA": "Valparaíso", "LA CALERA": "Valparaíso", "SAN ANTONIO": "Valparaíso",
    "SANTO DOMINGO": "Valparaíso", "CARTAGENA": "Valparaíso", "EL QUISCO": "Valparaíso",
    "EL TABO": "Valparaíso", "ALGARROBO": "Valparaíso", "CASABLANCA": "Valparaíso",
    "LOS ANDES": "Valparaíso", "SAN FELIPE": "Valparaíso", "CALLE LARGA": "Valparaíso",
    "LA LIGUA": "Valparaíso", "ZAPALLAR": "Valparaíso", "PAPUDO": "Valparaíso",
    "PUCHUNCAVI": "Valparaíso", "QUINTERO": "Valparaíso",
    # Región Metropolitana de Santiago (RM)
    "SANTIAGO": "Metropolitana (RM)", "LAS CONDES": "Metropolitana (RM)",
    "VITACURA": "Metropolitana (RM)", "LO BARNECHEA": "Metropolitana (RM)",
    "PROVIDENCIA": "Metropolitana (RM)", "NUNOA": "Metropolitana (RM)",
    "LA REINA": "Metropolitana (RM)", "MACUL": "Metropolitana (RM)",
    "PENALOLEN": "Metropolitana (RM)", "LA FLORIDA": "Metropolitana (RM)",
    "PUENTE ALTO": "Metropolitana (RM)", "PIRQUE": "Metropolitana (RM)",
    "SAN JOSE DE MAIPO": "Metropolitana (RM)", "MAIPU": "Metropolitana (RM)",
    "CERRILLOS": "Metropolitana (RM)", "ESTACION CENTRAL": "Metropolitana (RM)",
    "PUDAHUEL": "Metropolitana (RM)", "LO PRADO": "Metropolitana (RM)",
    "PRADO": "Metropolitana (RM)", "CERRO NAVIA": "Metropolitana (RM)",
    "QUINTA NORMAL": "Metropolitana (RM)", "RENCA": "Metropolitana (RM)",
    "QUILICURA": "Metropolitana (RM)", "HUECHURABA": "Metropolitana (RM)",
    "CONCHALI": "Metropolitana (RM)", "RECOLETA": "Metropolitana (RM)",
    "INDEPENDENCIA": "Metropolitana (RM)", "SAN MIGUEL": "Metropolitana (RM)",
    "SAN JOAQUIN": "Metropolitana (RM)", "PEDRO AGUIRRE CERDA": "Metropolitana (RM)",
    "LO ESPEJO": "Metropolitana (RM)", "LA CISTERNA": "Metropolitana (RM)",
    "EL BOSQUE": "Metropolitana (RM)", "SAN RAMON": "Metropolitana (RM)",
    "LA GRANJA": "Metropolitana (RM)", "LA PINTANA": "Metropolitana (RM)",
    "SAN BERNARDO": "Metropolitana (RM)", "BUIN": "Metropolitana (RM)",
    "PAINE": "Metropolitana (RM)", "CALERA DE TANGO": "Metropolitana (RM)",
    "COLINA": "Metropolitana (RM)", "CHICUREO": "Metropolitana (RM)",
    "LAMPA": "Metropolitana (RM)", "TILTIL": "Metropolitana (RM)",
    "TALAGANTE": "Metropolitana (RM)", "PENAFLOR": "Metropolitana (RM)",
    "PADRE HURTADO": "Metropolitana (RM)", "ISLA DE MAIPO": "Metropolitana (RM)",
    "EL MONTE": "Metropolitana (RM)", "MELIPILLA": "Metropolitana (RM)",
    "CURACAVI": "Metropolitana (RM)",
    # Región de O'Higgins (VI)
    "RANCAGUA": "O'Higgins", "MACHALI": "O'Higgins", "RENGO": "O'Higgins",
    "SAN FERNANDO": "O'Higgins", "SANTA CRUZ": "O'Higgins", "PICHILEMU": "O'Higgins",
    "MOSTAZAL": "O'Higgins", "GRANEROS": "O'Higgins", "SAN VICENTE": "O'Higgins",
    "CHEPICA": "O'Higgins",
    # Región del Maule (VII)
    "TALCA": "Maule", "CURICO": "Maule", "LINARES": "Maule", "CAUQUENES": "Maule",
    "CONSTITUCION": "Maule", "MOLINA": "Maule", "PARRAL": "Maule",
    "SAN CLEMENTE": "Maule", "SAN JAVIER": "Maule", "VILLA ALEGRE": "Maule",
    "BUCALEMU": "O'Higgins",
    # Región de Ñuble (XVI)
    "CHILLAN": "Ñuble", "CHILLAN VIEJO": "Ñuble", "SAN CARLOS": "Ñuble",
    "BULNES": "Ñuble", "PORTEZUELO": "Ñuble", "QUIRIHUE": "Ñuble",
    # Región del Biobío (VIII)
    "CONCEPCION": "Biobío", "TALCAHUANO": "Biobío", "SAN PEDRO DE LA PAZ": "Biobío",
    "CHIGUAYANTE": "Biobío", "HUALPEN": "Biobío", "CORONEL": "Biobío",
    "LOTA": "Biobío", "TOME": "Biobío", "PENCO": "Biobío", "LOS ANGELES": "Biobío",
    "ARAUCO": "Biobío", "CANETE": "Biobío", "CURANILAHUE": "Biobío",
    "CABRERO": "Biobío", "YUMBEL": "Biobío", "LAJA": "Biobío", "SANTA JUANA": "Biobío",
    "MULCHEN": "Biobío",
    # Región de La Araucanía (IX)
    "TEMUCO": "La Araucanía", "PADRE LAS CASAS": "La Araucanía", "VILLARRICA": "La Araucanía",
    "PUCON": "La Araucanía", "ANGOL": "La Araucanía", "VICTORIA": "La Araucanía",
    "LAUTARO": "La Araucanía", "PITRUFQUEN": "La Araucanía", "CUNCO": "La Araucanía",
    "GORBEA": "La Araucanía", "NUEVA IMPERIAL": "La Araucanía",
    # Región de Los Ríos (XIV)
    "VALDIVIA": "Los Ríos", "LA UNION": "Los Ríos", "RIO BUENO": "Los Ríos",
    "PANGUIPULLI": "Los Ríos", "FUTRONO": "Los Ríos", "CORRAL": "Los Ríos",
    # Región de Los Lagos (X)
    "PUERTO MONTT": "Los Lagos", "PUERTO VARAS": "Los Lagos", "OSORNO": "Los Lagos",
    "CASTRO": "Los Lagos", "ANCUD": "Los Lagos", "QUELLON": "Los Lagos",
    "FRUTILLAR": "Los Lagos", "LLANQUIHUE": "Los Lagos", "CALBUCO": "Los Lagos",
    "LOS MUERMOS": "Los Lagos", "PURRANQUE": "Los Lagos",
    # Región de Aysén (XI)
    "COYHAIQUE": "Aysén", "PUERTO AYSEN": "Aysén",
    # Región de Magallanes (XII)
    "PUNTA ARENAS": "Magallanes", "PUERTO NATALES": "Magallanes",
}


def resolve_region(comuna: Any) -> str:
    """Resuelve la Región de Chile a partir del nombre de la comuna."""
    if not comuna or not isinstance(comuna, str):
        return "Metropolitana (RM)"
    clean_key = CarDataPipeline.strip_accents(comuna.strip().upper())
    if clean_key in COMUNA_TO_REGION:
        return COMUNA_TO_REGION[clean_key]
    for k, reg in COMUNA_TO_REGION.items():
        if k in clean_key:
            return reg
    return "Metropolitana (RM)"


def format_clp(value: float) -> str:
    """Formatea un monto numérico como Pesos Chilenos ($ 12.450.000)."""
    try:
        return f"$ {int(round(float(value))):,}".replace(",", ".")
    except Exception:
        return "$ 0"


def resolve_portal_info(id_aviso: Any) -> Tuple[str, str, str]:
    """
    Identifica el portal de origen según el prefijo de 'id_aviso'.
    Retorna: (nombre_corto, etiqueta_con_emoji, css_class)
    """
    uid = str(id_aviso or "").strip().upper()
    if uid.startswith("MLC"):
        return "MercadoLibre", "🟡 MercadoLibre", "portal-badge-mlc"
    if uid.startswith("AC-"):
        return "Autocosmos", "🔵 Autocosmos", "portal-badge-ac"
    if uid.startswith("YAPO-"):
        return "Yapo.cl", "🟠 Yapo.cl", "portal-badge-yapo"
    return "Chileautos", "🟢 Chileautos", "portal-badge-cl"


def get_vehicle_emoji(marca: str, modelo: str) -> str:
    """Retorna un emoji acorde al segmento del vehículo."""
    mod = f"{marca} {modelo}".upper()
    if any(k in mod for k in ("HILUX", "L200", "RANGER", "NP300", "BT-50", "SILVERADO")):
        return "🛻"
    if any(
        k in mod
        for k in (
            "RAV4", "CX-5", "TIGGO", "SPORTAGE", "TUCSON", "QASHQAI",
            "XV", "3008", "CR-V", "HR-V", "PILOT", "XC60", "XC40", "XC90",
            "X1", "X3", "GLA", "GLC",
        )
    ):
        return "🚙"
    if any(k in mod for k in ("MORNING", "SPARK", "MARCH", "SWIFT", "YARIS", "BALENO", "208", "GOL", "RIO", "V40", "118I", "A200")):
        return "🚗"
    return "🚘"


def normalize_model_input(raw_model: Optional[str]) -> str:
    """Normaliza variantes y errores tipográficos de modelos según config.MODEL_ALIASES."""
    if not raw_model:
        return ""
    clean_m = re.sub(r"\s+", " ", str(raw_model).strip().upper())
    if clean_m in config.MODEL_ALIASES:
        return config.MODEL_ALIASES[clean_m]
    compact = clean_m.replace("-", "").replace(" ", "")
    if compact in config.MODEL_ALIASES:
        return config.MODEL_ALIASES[compact]
    return clean_m


def _assign_portal_and_region_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega vectorialmente las columnas 'portal', 'portal_badge' y 'region'."""
    if df.empty:
        return df
    uids = df["id_aviso"].fillna("").astype(str).str.strip().str.upper()
    conds = [uids.str.startswith("MLC"), uids.str.startswith("AC-"), uids.str.startswith("YAPO-")]
    df["portal"] = np.select(conds, ["MercadoLibre", "Autocosmos", "Yapo.cl"], default="Chileautos")
    df["portal_badge"] = np.select(
        conds,
        ["🟡 MercadoLibre", "🔵 Autocosmos", "🟠 Yapo.cl"],
        default="🟢 Chileautos",
    )
    if "comuna" in df.columns:
        unique_comunas = df["comuna"].fillna("").astype(str).unique()
        reg_map = {c: resolve_region(c) for c in unique_comunas}
        df["region"] = df["comuna"].fillna("").astype(str).map(reg_map).fillna("Metropolitana (RM)")
    return df


_WATCHED_BACKEND_MODULES = (
    "config",
    "database",
    "pipeline",
    "train",
    "monitor",
    "scraper",
    "main",
)


def _get_watched_module_mtimes() -> Dict[str, float]:
    """Almacena los mtimes iniciales en 'sys' para sobrevivir a st.cache_resource.clear()."""
    mtimes: Optional[Dict[str, float]] = getattr(sys, "_arbitraje_module_mtimes", None)
    if mtimes is None:
        mtimes = {}
        base_dir = Path(config.BASE_DIR)
        for mod_name in _WATCHED_BACKEND_MODULES:
            mod_path = base_dir / f"{mod_name}.py"
            if mod_name in sys.modules and mod_path.exists():
                mtimes[mod_name] = mod_path.stat().st_mtime
        setattr(sys, "_arbitraje_module_mtimes", mtimes)
    return mtimes


def _get_fresh_orchestrator():
    """
    Carga diferida de 'main.py' para acelerar el arranque en frío de la UI y recarga
    únicamente los módulos backend cuyo archivo .py haya sido modificado en disco.
    """
    mtimes = _get_watched_module_mtimes()
    base_dir = Path(config.BASE_DIR)
    for mod_name in _WATCHED_BACKEND_MODULES:
        mod_path = base_dir / f"{mod_name}.py"
        disk_mtime = mod_path.stat().st_mtime if mod_path.exists() else 0.0
        if mod_name not in sys.modules:
            importlib.import_module(mod_name)
            mtimes[mod_name] = disk_mtime
        elif disk_mtime > mtimes.get(mod_name, 0.0):
            importlib.reload(sys.modules[mod_name])
            mtimes[mod_name] = disk_mtime
    return sys.modules["main"]


def extract_nlp_tags(row: pd.Series) -> str:
    """Convierte las banderas binarias NLP en texto legible para las tarjetas."""
    tags = [label for col, label in NLP_LABELS_MAP.items() if row.get(col, 0) == 1]
    return " • ".join(tags) if tags else "Equipamiento Estándar"


def check_system_health() -> Tuple[bool, str]:
    """Verifica que 'data/chileautos_arbitraje.db' y 'models/model.pkl' estén operativos."""
    if not config.DB_PATH.exists():
        return False, "No se encontró la base de datos SQLite ('data/chileautos_arbitraje.db')."

    try:
        with closing(sqlite3.connect(str(config.DB_PATH))) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='autos_historico'"
            )
            if cur.fetchone() is None:
                return False, "La tabla 'autos_historico' aún no existe en la base de datos."
            cur.execute("SELECT COUNT(*) FROM autos_historico")
            if cur.fetchone()[0] == 0:
                return False, "La base de datos está vacía (0 vehículos registrados)."
    except Exception as exc:
        return False, f"Error verificando SQLite: {exc}"

    if not config.MODEL_PATH.exists():
        return False, "No se encontró el artefacto del modelo IA ('models/model.pkl')."

    return True, "OK"


@st.cache_resource(show_spinner=False)
def load_cached_predictor(model_mtime: float) -> Tuple[CarPricePredictor, Dict[str, Any]]:
    """Carga el modelo LightGBM desde 'models/model.pkl' y devuelve (predictor, metadata)."""
    _ = model_mtime
    setattr(sys.modules["__main__"], "HierarchicalCarEncoder", HierarchicalCarEncoder)
    predictor = CarPricePredictor(model_path=config.MODEL_PATH)
    try:
        metadata = predictor.load_artifact()
    except Exception:
        # En servidores cloud con distinta version de Python/sklearn regeneramos el artefacto compatible
        metadata = predictor.train_and_evaluate()
    return predictor, metadata


@st.cache_resource(show_spinner=False)
def load_dashboard_datasets(
    db_mtime: float,
    model_mtime: float,
    clean_csv_mtime: float,
    opp_csv_mtime: float,
) -> Dict[str, Any]:
    """
    Carga y enriquece todos los datasets del sistema en una sola pasada (SQLite histórico,
    activos deduplicados multiplataforma, autos_clean.csv, oportunidades_detectadas.csv
    y rebajas de precio_historial). Comparte DataFrames zero-copy en memoria (@st.cache_resource);
    los consumidores no deben mutar los DataFrames retornados sin copiar.
    """
    _ = (db_mtime, model_mtime, clean_csv_mtime, opp_csv_mtime)
    db_manager = DatabaseManager(db_path=config.DB_PATH)
    pipeline = CarDataPipeline(db_path=config.DB_PATH)
    predictor, model_meta = load_cached_predictor(model_mtime)

    today_prefix = datetime.now().strftime("%Y-%m-%d")
    with closing(sqlite3.connect(str(config.DB_PATH))) as conn:
        full_df = pd.read_sql_query(
            "SELECT * FROM autos_historico ORDER BY fecha_ultima_vista DESC", conn
        )
        try:
            price_history_df = pd.read_sql_query(
                """
                SELECT ph.id, ph.id_aviso, a.marca, a.modelo, a.anio, a.kilometraje,
                       a.comuna, a.estado, ph.precio_antiguo, ph.precio_nuevo,
                       (ph.precio_nuevo - ph.precio_antiguo) AS variacion_clp,
                       ph.fecha_cambio, a.url
                FROM precio_historial ph
                LEFT JOIN autos_historico a ON ph.id_aviso = a.id_aviso
                ORDER BY ph.fecha_cambio DESC
                LIMIT 400
                """,
                conn,
            )
            if not price_history_df.empty:
                price_history_df["variacion_pct"] = np.where(
                    price_history_df["precio_antiguo"] > 0,
                    (
                        (price_history_df["variacion_clp"] / price_history_df["precio_antiguo"]) * 100.0
                    ).round(1),
                    0.0,
                )
                price_history_df = _assign_portal_and_region_columns(price_history_df)
        except Exception:
            price_history_df = pd.DataFrame()

        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM autos_historico WHERE estado = 'vendido' AND fecha_venta LIKE ?",
            (f"{today_prefix}%",),
        )
        sold_today = int(cur.fetchone()[0])

    summary_stats = db_manager.get_summary_stats()
    summary_stats["vendidos_hoy"] = sold_today
    summary_stats["duplicados_multiplataforma"] = 0

    def enrich_with_ia_and_portal(df_in: pd.DataFrame, only_valid_predict: bool = False) -> pd.DataFrame:
        if df_in.empty:
            return df_in.copy()
        df_out = pipeline.engineer_features(df_in)
        df_out = _assign_portal_and_region_columns(df_out)
        df_out["precio_justo_ia"] = 0.0
        if only_valid_predict:
            valid_mask = df_out["precio"] >= config.MIN_VALID_PRICE_CLP
            if valid_mask.any():
                df_out.loc[valid_mask, "precio_justo_ia"] = predictor.predict_fair_price(
                    df_out.loc[valid_mask]
                )
        else:
            df_out["precio_justo_ia"] = predictor.predict_fair_price(df_out)

        df_out["margen_ganancia_clp"] = df_out["precio_justo_ia"] - df_out["precio"]
        df_out["descuento_pct"] = np.where(
            df_out["precio_justo_ia"] > 0,
            (df_out["margen_ganancia_clp"] / df_out["precio_justo_ia"]).round(4),
            0.0,
        )
        df_out["roi_pct"] = np.where(
            df_out["precio"] > 0,
            ((df_out["margen_ganancia_clp"] / df_out["precio"]) * 100.0).round(1),
            0.0,
        )
        disc_vals = df_out["descuento_pct"].values
        df_out["categoria_arbitraje"] = np.select(
            [
                disc_vals >= config.SUSPICIOUS_DISCOUNT_THRESHOLD,
                disc_vals >= config.OPPORTUNITY_DISCOUNT_THRESHOLD,
            ],
            ["POSIBLE ESTAFA / ERROR DE TIPEO", "OPORTUNIDAD"],
            default="PRECIO DE MERCADO",
        )
        return df_out

    # 1. Enriquecer full_df en UNA sola pasada
    if not full_df.empty:
        full_enriched = enrich_with_ia_and_portal(full_df, only_valid_predict=True)
        full_enriched["multi_portal_info"] = "Publicación única"

        active_raw = full_enriched[
            (full_enriched["estado"] == "activo")
            & (full_enriched["precio"] >= config.MIN_VALID_PRICE_CLP)
        ].copy()
        active_featured, merged_dups = CarDataPipeline.deduplicate_cross_platform_df(active_raw)
        summary_stats["duplicados_multiplataforma"] = int(merged_dups)

        if not active_featured.empty and merged_dups > 0:
            merged_map = active_featured.loc[
                active_featured["multi_portal_info"] != "Publicación única",
                ["id_aviso", "multi_portal_info"],
            ].set_index("id_aviso")["multi_portal_info"].to_dict()
            if merged_map:
                full_enriched["multi_portal_info"] = (
                    full_enriched["id_aviso"].map(merged_map).fillna("Publicación única")
                )
    else:
        full_enriched = pd.DataFrame()
        active_featured = pd.DataFrame()

    # 2. Derivar 'clean_df' sin re-ejecutar LightGBM ni engineer_features cuando coincide con SQLite
    discarded_id_set = (
        set(full_enriched.loc[full_enriched["estado"] == "descartado", "id_aviso"].astype(str))
        if not full_enriched.empty
        else set()
    )
    active_id_set = (
        set(full_enriched.loc[full_enriched["estado"] == "activo", "id_aviso"].astype(str))
        if not full_enriched.empty
        else set()
    )

    clean_csv_path = config.DATA_DIR / "autos_clean.csv"
    if clean_csv_path.exists() and not active_featured.empty:
        clean_ids_df = pd.read_csv(clean_csv_path, usecols=["id_aviso"], encoding="utf-8")
        clean_id_set = set(clean_ids_df["id_aviso"].astype(str)) - discarded_id_set
        clean_df = active_featured[active_featured["id_aviso"].astype(str).isin(clean_id_set)].copy()
        if len(clean_df) < int(len(clean_id_set) * 0.99):
            non_discarded = full_enriched[full_enriched["estado"] != "descartado"]
            clean_df = non_discarded[non_discarded["id_aviso"].astype(str).isin(clean_id_set)].copy()
            if len(clean_df) < int(len(clean_id_set) * 0.99):
                raw_clean = pd.read_csv(clean_csv_path, encoding="utf-8")
                if discarded_id_set:
                    raw_clean = raw_clean[~raw_clean["id_aviso"].astype(str).isin(discarded_id_set)]
                clean_df = enrich_with_ia_and_portal(raw_clean)
    elif clean_csv_path.exists():
        raw_clean = pd.read_csv(clean_csv_path, encoding="utf-8")
        if discarded_id_set:
            raw_clean = raw_clean[~raw_clean["id_aviso"].astype(str).isin(discarded_id_set)]
        clean_df = enrich_with_ia_and_portal(raw_clean)
    else:
        raw_clean, _ = pipeline.run_pipeline(only_active=True, save_output=True)
        clean_df = enrich_with_ia_and_portal(raw_clean)

    # 3. Cargar y pre-enriquecer 'data/oportunidades_detectadas.csv'
    opp_csv_path = config.DATA_DIR / "oportunidades_detectadas.csv"
    if opp_csv_path.exists():
        try:
            opp_csv_df = pd.read_csv(opp_csv_path, encoding="utf-8")
            if not opp_csv_df.empty and active_id_set:
                opp_csv_df = opp_csv_df[
                    opp_csv_df["id_aviso"].astype(str).isin(active_id_set)
                ].copy()
            if not opp_csv_df.empty:
                opp_csv_df["marca"] = opp_csv_df["marca"].astype(str).str.strip().str.upper()
                opp_csv_df["modelo"] = opp_csv_df["modelo"].astype(str).str.strip().str.upper()
                opp_csv_df = _assign_portal_and_region_columns(opp_csv_df)
                opp_csv_df["roi_pct"] = (
                    (opp_csv_df["margen_ganancia_clp"] / np.maximum(1.0, opp_csv_df["precio"])) * 100.0
                ).round(1)
                if not full_enriched.empty:
                    extra_cols = [
                        c for c in full_enriched.columns if c not in opp_csv_df.columns or c == "id_aviso"
                    ]
                    if len(extra_cols) > 1:
                        opp_csv_df = opp_csv_df.merge(
                            full_enriched[extra_cols].drop_duplicates(subset=["id_aviso"]),
                            on="id_aviso",
                            how="left",
                        )
        except Exception:
            opp_csv_df = pd.DataFrame()
    else:
        opp_csv_df = pd.DataFrame()

    # 4. Precalcular tabla agregada del Treemap
    if not clean_df.empty:
        tree_df = (
            clean_df.groupby(["marca", "modelo", "generacion"], as_index=False)
            .agg(
                unidades=("id_aviso", "count"),
                precio_medio=("precio", "mean"),
                descuento_max_pct=("descuento_pct", lambda s: round(float(s.max()) * 100.0, 1)),
            )
        )
    else:
        tree_df = pd.DataFrame()

    portal_counts = (
        active_featured["portal"].value_counts().to_dict() if not active_featured.empty else {}
    )
    favorites_map = db_manager.get_favorites_map()

    return {
        "summary": summary_stats,
        "portal_counts": portal_counts,
        "model_meta": model_meta,
        "full_df": full_enriched,
        "price_history_df": price_history_df,
        "active_featured": active_featured,
        "clean_df": clean_df,
        "opp_csv_df": opp_csv_df,
        "tree_df": tree_df,
        "favorites_map": favorites_map,
    }


def execute_backend_cycle(
    force_train: bool = False,
    marca_filter: Optional[str] = None,
    modelo_filter: Optional[str] = None,
    pages_per_portal: int = 2,
    max_telegram_alerts: int = 5,
    anio_min: Optional[int] = None,
    anio_max: Optional[int] = None,
    precio_min: Optional[int] = None,
    precio_max: Optional[int] = None,
    km_max: Optional[int] = None,
    deep: bool = False,
) -> Tuple[bool, str]:
    """Invoca 'run_autonomous_cycle' de 'main.py' y refresca todas las cachés de la UI."""
    try:
        marca_clean = (
            marca_filter.strip().upper()
            if marca_filter and marca_filter.strip() and marca_filter != "TODAS"
            else None
        )
        raw_mod = (
            modelo_filter.strip().upper()
            if modelo_filter and modelo_filter.strip() and modelo_filter != "TODOS"
            else None
        )
        modelo_clean = normalize_model_input(raw_mod) if raw_mod else None

        orch = _get_fresh_orchestrator()
        stats = orch.run_autonomous_cycle(
            force_train=force_train,
            force_fallback_scraper=False,
            recheck_all_alerts=True,
            marca_filter=marca_clean,
            modelo_filter=modelo_clean,
            pages_per_portal=int(pages_per_portal),
            max_telegram_alerts=int(max_telegram_alerts),
            anio_min=anio_min,
            anio_max=anio_max,
            precio_min=precio_min,
            precio_max=precio_max,
            km_max=km_max,
            deep=deep,
        )
        st.cache_data.clear()
        st.cache_resource.clear()

        filter_details = []
        if anio_min or anio_max:
            filter_details.append(f"Años: {anio_min or 'min'}..{anio_max or 'max'}")
        if precio_min or precio_max:
            p_lo = f"${precio_min:,}" if precio_min else "$0"
            p_hi = f"${precio_max:,}" if precio_max else "max"
            filter_details.append(f"Precio: {p_lo}–{p_hi}".replace(",", "."))
        if km_max:
            filter_details.append(f"Km ≤ {km_max:,}".replace(",", "."))
        if deep:
            filter_details.append("Chileautos Profundo Año x Año")
        details_str = f" ({', '.join(filter_details)})" if filter_details else ""

        portal_str = ""
        if isinstance(stats, dict) and stats.get("desglose_portales"):
            desglose = stats["desglose_portales"]
            non_zero = [f"{p}: {c}" for p, c in desglose.items() if c > 0]
            portal_summary = " · ".join(non_zero) if non_zero else "0 avisos nuevos"
            portal_str = (
                f" | Captura: [{portal_summary}] | Nuevos: {stats.get('nuevos_insertados', 0)} | "
                f"Oportunidades: {stats.get('oportunidades_detectadas', 0)} "
                f"({stats.get('elapsed_seconds', 0):.1f}s)"
            )

        if marca_clean:
            st.session_state["focused_marca"] = marca_clean
            st.session_state["focused_modelo"] = modelo_clean
            return (
                True,
                f"✅ Búsqueda en vivo completada para {marca_clean} {modelo_clean or 'TODOS'}{details_str}{portal_str}.",
            )
        return (
            True,
            f"✅ Barrido general multi-portal completado{details_str}{portal_str}.",
        )
    except Exception as exc:
        return False, f"❌ Error durante la ejecución del ciclo autónomo: {exc}"


# ==============================================================================
# 3. PANTALLA DE RESILIENCIA (SI NO EXISTE DB O MODEL.PKL)
# ==============================================================================
def render_resilience_onboarding(reason: str) -> None:
    """Muestra pantalla amigable de inicialización cuando falta DB o modelo IA."""
    inject_theme_css(is_light_theme())
    st.markdown(
        _clean_html(
            """
            <div class="hero-banner">
                <h1 class="hero-title">🏎️ Sistema Multi-Portal de Arbitraje Automotriz — Inicialización</h1>
                <p class="hero-subtitle">
                    MercadoLibre Autos • Autocosmos Chile • Yapo.cl • Chileautos | Motor LightGBM IA
                </p>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )
    st.warning(f"ℹ️ **Diagnóstico del Sistema:** {reason}")

    col_info, col_action = st.columns([2, 1])
    with col_info:
        st.markdown(
            """
            ### 🚀 Inicialización Rápida del Sistema
            Presiona el botón de activación para ejecutar el primer barrido en vivo:
            1. **Módulo 1 (`scraper.py`):** Extrae avisos reales desde **MercadoLibre (`MLC-`)**, **Autocosmos (`AC-`)** y **Yapo (`YAPO-`)**.
            2. **Módulo 2 (`pipeline.py`):** Filtra outliers con IQR/Z-Score, asigna generaciones (`MODEL_GENERATIONS`), nivel socioeconómico comunal (`1–5`) y extrae variables NLP.
            3. **Módulo 3 (`train.py`):** Entrena el regresor híbrido **LightGBM + Target Encoding Jerárquico** y genera `models/model.pkl`.
            4. **Módulo 4 (`monitor.py`):** Detecta oportunidades con descuento $\\ge 15\\%$ y envía alertas a Telegram.
            """
        )
    with col_action:
        st.markdown("#### Panel de Activación")
        if st.button(
            "⚡ Inicializar y Ejecutar Primer Escaneo Ahora",
            type="primary",
            width="stretch",
        ):
            with st.spinner("Ejecutando adquisición multi-portal, limpieza e IA..."):
                ok, msg = execute_backend_cycle(force_train=True, pages_per_portal=2)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


# ==============================================================================
# 4. BARRA LATERAL (KPIS, BÚSQUEDA RÁPIDA Y GESTOR DE CATÁLOGO)
# ==============================================================================
def render_sidebar(data_bundle: Dict[str, Any]) -> None:
    """Renderiza el Sidebar con KPIs de SQLite, buscador rápido, gestor de catálogo y métricas IA."""
    summary = data_bundle["summary"]
    portal_counts = data_bundle["portal_counts"]
    model_meta = data_bundle["model_meta"]
    metrics = model_meta.get("metricas_test_80_20", {})
    clean_df = data_bundle["clean_df"]
    favorites_count = len(data_bundle.get("favorites_map", {}))

    raw_train_date = model_meta.get("fecha_entrenamiento", "")
    try:
        dt_parsed = datetime.fromisoformat(raw_train_date)
        formatted_train_date = dt_parsed.strftime("%d/%m/%Y %H:%M")
    except Exception:
        if config.MODEL_PATH.exists():
            dt_mtime = datetime.fromtimestamp(config.MODEL_PATH.stat().st_mtime)
            formatted_train_date = dt_mtime.strftime("%d/%m/%Y %H:%M")
        else:
            formatted_train_date = "No disponible"

    with st.sidebar:
        st.markdown("## 🏎️ Arbitraje IA Chile")
        st.caption("Motor Multi-Portal: MercadoLibre • Autocosmos • Yapo • Chileautos")
        theme_kwargs = (
            {"default": THEME_MODES[0]} if "ui_theme_mode" not in st.session_state else {}
        )
        st.segmented_control(
            "🎨 Tema de Interfaz:",
            options=THEME_MODES,
            selection_mode="single",
            required=True,
            width="stretch",
            key="ui_theme_mode",
            **theme_kwargs,
        )
        st.divider()

        db_hdr1, db_hdr2 = st.columns([2.1, 1.0])
        with db_hdr1:
            st.markdown("### 📊 Estado de BD")
        with db_hdr2:
            if st.button("🔄", key="sb_refresh_cache_btn", help="Recargar datos y cachés desde disco sin escanear portales"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                label="Total en BD",
                value=f"{summary.get('total_historicos', 0):,}".replace(",", "."),
                help="Total histórico de vehículos almacenados en SQLite.",
            )
        with c2:
            st.metric(
                label="Autos Activos",
                value=f"{summary.get('activos', 0):,}".replace(",", "."),
                help="Publicaciones activas disponibles hoy.",
            )

        c3, c4 = st.columns(2)
        with c3:
            st.metric(
                label="Vendidos Hoy",
                value=f"{summary.get('vendidos_hoy', 0):,}".replace(",", "."),
                delta=f"{summary.get('vendidos', 0)} hist.",
            )
        with c4:
            st.metric(
                label="Autos Limpios",
                value=f"{len(clean_df):,}".replace(",", "."),
                help="Vehículos validados en data/autos_clean.csv tras filtro de outliers.",
            )

        if portal_counts:
            portal_str = " | ".join(f"**{k}:** {v}" for k, v in portal_counts.items())
            st.caption(f"🌐 **Activos (dedup):** {portal_str}")
        dups_merged = int(summary.get("duplicados_multiplataforma", 0))
        if dups_merged > 0:
            st.caption(f"💎 **{dups_merged} avisos duplicados** consolidados al portal más barato.")
        st.caption(f"⭐ **Favoritos guardados:** {favorites_count}")

        st.divider()

        st.markdown("### 🎯 Búsqueda Focalizada Rápida")
        all_segments = config.get_all_target_segments()
        target_marcas = sorted({m for m, _ in all_segments})
        db_marcas = sorted(clean_df["marca"].unique().tolist()) if not clean_df.empty else []
        all_marcas = sorted(set(target_marcas + db_marcas))

        default_marca = st.session_state.get("focused_marca") or "MAZDA"
        marca_idx = all_marcas.index(default_marca) if default_marca in all_marcas else 0

        sb_marca_choice = st.selectbox(
            "Marca a escanear:",
            options=all_marcas + ["➕ OTRA MARCA (Escribir nueva)"],
            index=marca_idx,
            key="sb_focused_marca",
        )
        if sb_marca_choice == "➕ OTRA MARCA (Escribir nueva)":
            sb_marca = (
                st.text_input("Escribe la nueva Marca (ej. AUDI, PORSCHE):", value="AUDI", key="sb_custom_marca_txt")
                .strip()
                .upper()
            )
        else:
            sb_marca = sb_marca_choice

        target_mods = [mod for m, mod in all_segments if m == sb_marca]
        db_mods = (
            clean_df[clean_df["marca"] == sb_marca]["modelo"].unique().tolist()
            if not clean_df.empty
            else []
        )
        known_mods = sorted({str(x) for x in (target_mods + db_mods) if str(x).strip()})
        mod_options = ["TODOS (Toda la Marca)"] + known_mods + ["➕ OTRO MODELO (Escribir manualmente)"]

        default_mod = st.session_state.get("focused_modelo")
        mod_idx = mod_options.index(default_mod) if default_mod in mod_options else (1 if known_mods else 0)

        sb_mod_choice = st.selectbox(
            "Modelo a escanear:",
            options=mod_options,
            index=mod_idx,
            key="sb_focused_modelo_choice",
        )
        if sb_mod_choice == "➕ OTRO MODELO (Escribir manualmente)":
            sb_modelo = normalize_model_input(
                st.text_input("Escribe el Modelo (ej. A3, XC60, CIVIC):", value="A3", key="sb_custom_mod_txt")
            )
        elif sb_mod_choice == "TODOS (Toda la Marca)":
            sb_modelo = "TODOS"
        else:
            sb_modelo = sb_mod_choice

        sb_pages = st.slider(
            "Páginas por portal:",
            min_value=1,
            max_value=5,
            value=2,
            key="sb_pages_slider",
        )
        sb_alerts = st.slider(
            "Máx. alertas a Telegram:",
            min_value=0,
            max_value=10,
            value=3,
            help="0 = Actualizar dashboard sin enviar alertas a Telegram.",
            key="sb_alerts_slider",
        )

        with st.expander("🎯 Filtros Granulares (Años, Precios, Km)", expanded=False):
            sb_apply_years = st.toggle("Restringir por rango de años", value=False, key="sb_apply_years")
            sb_yr_c1, sb_yr_c2 = st.columns(2)
            with sb_yr_c1:
                sb_anio_min = st.number_input(
                    "Año Mín:",
                    min_value=config.MIN_VALID_YEAR,
                    max_value=config.MAX_VALID_YEAR,
                    value=2016,
                    step=1,
                    disabled=not sb_apply_years,
                    key="sb_anio_min",
                )
            with sb_yr_c2:
                sb_anio_max = st.number_input(
                    "Año Máx:",
                    min_value=config.MIN_VALID_YEAR,
                    max_value=config.MAX_VALID_YEAR,
                    value=2024,
                    step=1,
                    disabled=not sb_apply_years,
                    key="sb_anio_max",
                )

            sb_p_max = st.number_input(
                "Precio Máximo (CLP):",
                min_value=0,
                max_value=config.MAX_VALID_PRICE_CLP,
                value=0,
                step=1_000_000,
                help="0 = Sin límite de precio",
                key="sb_precio_max",
            )
            sb_km_max = st.number_input(
                "Km Máximo:",
                min_value=0,
                max_value=config.MAX_VALID_KM,
                value=0,
                step=10_000,
                help="0 = Sin límite de kilometraje",
                key="sb_km_max",
            )
            sb_deep = st.checkbox(
                "⚡ Chileautos Profundo (Año x Año)",
                value=True,
                help="Consulta cada año individualmente en Google para Chileautos, multiplicando la captura de avisos.",
                key="sb_deep_chileautos",
            )

        if st.button(
            "🔎 Buscar y Analizar en Vivo",
            key="sb_run_focused_btn",
            type="primary",
            width="stretch",
        ):
            if not sb_marca:
                st.warning("Debes ingresar una Marca válida.")
            elif sb_apply_years and sb_anio_min > sb_anio_max:
                st.warning("Año Mínimo no puede ser mayor que Año Máximo.")
            else:
                norm_sb_mod = None if sb_modelo in ("TODOS", "") else normalize_model_input(sb_modelo)
                if norm_sb_mod:
                    config.add_custom_segment(sb_marca, norm_sb_mod)
                with st.spinner(
                    f"Escaneando portales en vivo para {sb_marca} {norm_sb_mod or 'TODOS'} ({sb_pages} págs/portal)..."
                ):
                    ok, msg = execute_backend_cycle(
                        force_train=False,
                        marca_filter=sb_marca,
                        modelo_filter=norm_sb_mod,
                        pages_per_portal=sb_pages,
                        max_telegram_alerts=sb_alerts,
                        anio_min=int(sb_anio_min) if sb_apply_years else None,
                        anio_max=int(sb_anio_max) if sb_apply_years else None,
                        precio_max=int(sb_p_max) if sb_p_max > 0 else None,
                        km_max=int(sb_km_max) if sb_km_max > 0 else None,
                        deep=sb_deep,
                    )
                if ok:
                    st.session_state["last_scan_message"] = msg
                    st.rerun()
                else:
                    st.error(msg)

        with st.expander("➕ Agregar Marca / Modelo al Catálogo", expanded=False):
            st.caption("Añade permanentemente nuevas marcas y modelos al sistema:")
            new_brand_input = st.text_input(
                "Nueva Marca:", placeholder="Ej: HONDA, BMW, AUDI", key="add_cat_marca"
            )
            new_model_input = st.text_input(
                "Nuevo Modelo:", placeholder="Ej: ACCORD, X5, A4", key="add_cat_modelo"
            )
            if st.button("💾 Guardar en Catálogo", key="btn_save_catalog", width="stretch"):
                norm_new_mod = normalize_model_input(new_model_input)
                ok_cat, msg_cat = config.add_custom_segment(new_brand_input, norm_new_mod)
                if ok_cat:
                    st.session_state["focused_marca"] = new_brand_input.strip().upper()
                    st.session_state["focused_modelo"] = norm_new_mod
                    st.success(msg_cat)
                    st.rerun()
                else:
                    st.warning(msg_cat)

        st.divider()
        st.markdown("### 🤖 Modelo LightGBM IA")
        st.info(
            f"🗓️ **Entrenado:** `{formatted_train_date}`\n\n"
            f"📦 **Muestras Train/Total:** `{model_meta.get('registros_entrenamiento_total', 0)}`"
        )
        if metrics:
            m1, m2 = st.columns(2)
            with m1:
                r2_val = float(metrics.get("R2_Score", 0.0))
                st.metric("R² Score", f"{r2_val * 100:.1f}%")
            with m2:
                mape_val = float(metrics.get("MAPE_Pct", 0.0))
                st.metric("Error MAPE", f"{mape_val:.1f}%")
            st.caption(
                f"MAE: **{format_clp(metrics.get('MAE_CLP', 0))}** | "
                f"RMSE: **{format_clp(metrics.get('RMSE_CLP', 0))}**"
            )

        if st.button(
            f"🔄 Barrido General ({len(all_segments)} Segmentos)",
            key="sb_full_sweep_btn",
            width="stretch",
        ):
            with st.spinner("Ejecutando barrido multi-portal en todos los segmentos del catálogo..."):
                ok, msg = execute_backend_cycle(
                    force_train=False,
                    marca_filter=None,
                    modelo_filter=None,
                    pages_per_portal=config.SCRAPER_PAGES_PER_SEGMENT,
                    max_telegram_alerts=6,
                )
            if ok:
                st.session_state["last_scan_message"] = msg
                st.rerun()
            else:
                st.error(msg)


# ==============================================================================
# 5. COMPONENTE REUTILIZABLE DE FILTRADO AVANZADO MULTIFACÉTICO
# ==============================================================================
def _reset_multifacet_filters(key_prefix: str) -> None:
    """Callback seguro que limpia las claves de filtro asociadas antes del rerun."""
    prefix = f"{key_prefix}_"
    for k in [key for key in list(st.session_state.keys()) if key.startswith(prefix)]:
        del st.session_state[k]
    if key_prefix == "opp_flt":
        st.session_state.pop("opp_card_page", None)


def _sanitize_multiselect_state(key: str, valid_options: List[str]) -> None:
    """Mantiene en session_state solo opciones válidas dentro del dataset actual."""
    if key in st.session_state:
        valid_set = set(valid_options)
        cur = st.session_state[key]
        if isinstance(cur, list):
            filtered = [x for x in cur if x in valid_set]
            if filtered != cur:
                st.session_state[key] = filtered


def _sanitize_slider_state(key: str, min_v: int, max_v: int) -> None:
    """Acota una tupla de slider en session_state dentro de [min_v, max_v]."""
    if key in st.session_state:
        cur = st.session_state[key]
        if isinstance(cur, (tuple, list)) and len(cur) == 2:
            lo = max(min_v, min(int(cur[0]), max_v))
            hi = max(min_v, min(int(cur[1]), max_v))
            if lo > hi:
                lo, hi = min_v, max_v
            if (lo, hi) != (cur[0], cur[1]):
                st.session_state[key] = (lo, hi)


def apply_multifacet_filters(
    df: pd.DataFrame,
    key_prefix: str,
    default_only_opportunities: bool = False,
    allow_category_filter: bool = True,
) -> pd.DataFrame:
    """
    Renderiza una barra de filtros avanzados unificados y los aplica sobre una vista del DataFrame:
    - Marcas (multi-select), Modelos (multi-select dinámicos según marca), Portales (multi-select)
    - Rango de Precios (CLP), Rango de Años, Rango de Kilometraje (km)
    - Rango de % de Descuento frente a IA y Margen Mínimo en CLP
    - Transmisión, Combustible, Comunas, Tier Socioeconómico (1 a 5)
    - Atributos NLP obligatorios (Único dueño, Mantenciones al día, Cuero, etc.)
    - Ordenamiento dinámico y Búsqueda textual segura (regex=False, multi-palabra).
    """
    if df.empty:
        return df.copy()

    work = df

    # Límites dinámicos seguros
    valid_prices = work.loc[work["precio"] > 0, "precio"]
    min_p_db = int(valid_prices.min()) if not valid_prices.empty else 0
    max_p_db = int(valid_prices.max()) if not valid_prices.empty else 120_000_000
    slider_p_max = max(max_p_db, min_p_db + 1_000_000, 1_000_000)

    valid_years = work["anio"].dropna()
    min_y_db = int(valid_years.min()) if not valid_years.empty else config.MIN_VALID_YEAR
    max_y_db = int(valid_years.max()) if not valid_years.empty else config.MAX_VALID_YEAR
    slider_y_max = max(max_y_db, min_y_db + 1)

    valid_kms = work["kilometraje"].dropna()
    max_km_db = (
        max(int(min(valid_kms.max(), config.MAX_VALID_KM)), 50_000)
        if not valid_kms.empty
        else 300_000
    )

    reg_col = work["region"] if "region" in work.columns else work["comuna"].apply(resolve_region)

    marcas_disp = sorted(work["marca"].dropna().astype(str).unique().tolist())
    regiones_disp = sorted(reg_col.dropna().astype(str).unique().tolist())
    portales_disp = sorted(work["portal"].dropna().astype(str).unique().tolist())
    trans_disp = sorted(work["transmision"].dropna().astype(str).unique().tolist())
    comb_disp = sorted(work["combustible"].dropna().astype(str).unique().tolist())

    _sanitize_multiselect_state(f"{key_prefix}_marcas", marcas_disp)
    _sanitize_multiselect_state(f"{key_prefix}_regiones", regiones_disp)
    _sanitize_multiselect_state(f"{key_prefix}_portales", portales_disp)
    _sanitize_multiselect_state(f"{key_prefix}_trans", trans_disp)
    _sanitize_multiselect_state(f"{key_prefix}_comb", comb_disp)

    # Fila 1: Marcas, Modelos, Regiones, Portales, Ordenamiento y Reset
    r1_c1, r1_c2, r1_c3, r1_c4, r1_c5, r1_c6 = st.columns([1.15, 1.15, 1.2, 1.05, 1.15, 0.7])
    with r1_c1:
        sel_marcas = st.multiselect(
            "🚗 Marcas:",
            options=marcas_disp,
            placeholder="Todas las marcas...",
            key=f"{key_prefix}_marcas",
            **({} if f"{key_prefix}_marcas" in st.session_state else {"default": []}),
        )
    with r1_c2:
        if sel_marcas:
            modelos_disp = sorted(
                work.loc[work["marca"].isin(sel_marcas), "modelo"].dropna().astype(str).unique().tolist()
            )
        else:
            modelos_disp = sorted(work["modelo"].dropna().astype(str).unique().tolist())
        _sanitize_multiselect_state(f"{key_prefix}_modelos", modelos_disp)
        sel_modelos = st.multiselect(
            "🏷️ Modelos:",
            options=modelos_disp,
            placeholder="Todos los modelos...",
            key=f"{key_prefix}_modelos",
            **({} if f"{key_prefix}_modelos" in st.session_state else {"default": []}),
        )
    with r1_c3:
        sel_regiones = st.multiselect(
            "🗺️ Región:",
            options=regiones_disp,
            placeholder="Todas las regiones...",
            help="Filtra las oportunidades detectadas por Región de Chile (ej. Metropolitana, Valparaíso, Biobío, Los Lagos...).",
            key=f"{key_prefix}_regiones",
            **({} if f"{key_prefix}_regiones" in st.session_state else {"default": []}),
        )
    with r1_c4:
        sel_portales = st.multiselect(
            "🌐 Portales:",
            options=portales_disp,
            key=f"{key_prefix}_portales",
            **({} if f"{key_prefix}_portales" in st.session_state else {"default": portales_disp}),
        )
    with r1_c5:
        sort_by = st.selectbox(
            "🔃 Ordenar por:",
            options=[
                "Mayor Margen Ganancia (CLP)",
                "Mayor % Descuento vs IA",
                "Mayor ROI (%)",
                "Menor Precio Publicado",
                "Mayor Precio Publicado",
                "Menor Kilometraje",
                "Año más Reciente",
            ],
            index=0,
            key=f"{key_prefix}_sort",
        )
    with r1_c6:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        st.button(
            "🧹 Limpiar",
            key=f"{key_prefix}_reset_btn",
            help="Restablece todos los filtros de esta sección",
            on_click=_reset_multifacet_filters,
            args=(key_prefix,),
            width="stretch",
        )

    if sel_regiones:
        comunas_disp = sorted(
            work.loc[reg_col.isin(sel_regiones), "comuna"].dropna().astype(str).unique().tolist()
        )
    else:
        comunas_disp = sorted(work["comuna"].dropna().astype(str).unique().tolist())
    _sanitize_multiselect_state(f"{key_prefix}_comunas", comunas_disp)

    # Sanitizar rangos de sliders ante cambios de fuente
    _sanitize_slider_state(f"{key_prefix}_price_rng", 0, slider_p_max)
    _sanitize_slider_state(f"{key_prefix}_year_rng", min_y_db, slider_y_max)
    _sanitize_slider_state(f"{key_prefix}_km_rng", 0, max_km_db)

    # Fila 2: Sliders numéricos (Precio, Años, Kilometraje, Descuento/Margen)
    r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
    with r2_c1:
        price_range = st.slider(
            "💰 Rango de Precio (CLP):",
            min_value=0,
            max_value=slider_p_max,
            step=250_000,
            format="$ %d",
            key=f"{key_prefix}_price_rng",
            **(
                {}
                if f"{key_prefix}_price_rng" in st.session_state
                else {"value": (min_p_db, max_p_db)}
            ),
        )
    with r2_c2:
        year_range = st.slider(
            "📅 Rango de Años:",
            min_value=min_y_db,
            max_value=slider_y_max,
            step=1,
            key=f"{key_prefix}_year_rng",
            **(
                {}
                if f"{key_prefix}_year_rng" in st.session_state
                else {"value": (min_y_db, slider_y_max)}
            ),
        )
    with r2_c3:
        km_range = st.slider(
            "🛣️ Rango de Kilometraje (km):",
            min_value=0,
            max_value=max_km_db,
            step=5_000,
            format="%d km",
            key=f"{key_prefix}_km_rng",
            **({} if f"{key_prefix}_km_rng" in st.session_state else {"value": (0, max_km_db)}),
        )
    with r2_c4:
        default_min_disc = 15 if default_only_opportunities else -50
        disc_range = st.slider(
            "📉 % Descuento vs Precio Justo IA:",
            min_value=-50,
            max_value=80,
            value=(default_min_disc, 55 if default_only_opportunities else 80),
            step=1,
            format="%d%%",
            help="Filtra qué tan por debajo del Precio Justo IA está publicado el vehículo (ej. 15% a 55% = Oportunidad Real).",
            key=f"{key_prefix}_disc_rng",
        )

    # Fila 3 Desplegable: Filtros Técnicos, Geográficos (Comuna/Tier), NLP y Búsqueda Textual
    with st.expander(
        "⚙️ **Más Filtros Avanzados (Transmisión, Combustible, Comunas, Tier Socioeconómico y Equipamiento NLP)**",
        expanded=False,
    ):
        r3_c1, r3_c2, r3_c3, r3_c4 = st.columns(4)
        with r3_c1:
            sel_trans = st.multiselect(
                "⚙️ Transmisión:",
                options=trans_disp,
                key=f"{key_prefix}_trans",
                **({} if f"{key_prefix}_trans" in st.session_state else {"default": trans_disp}),
            )
        with r3_c2:
            sel_comb = st.multiselect(
                "⛽ Combustible:",
                options=comb_disp,
                key=f"{key_prefix}_comb",
                **({} if f"{key_prefix}_comb" in st.session_state else {"default": comb_disp}),
            )
        with r3_c3:
            sel_comunas = st.multiselect(
                "📍 Comunas Específicas:",
                options=comunas_disp,
                placeholder="Todas las comunas...",
                key=f"{key_prefix}_comunas",
                **({} if f"{key_prefix}_comunas" in st.session_state else {"default": []}),
            )
        with r3_c4:
            tier_range = st.slider(
                "🏙️ Tier Socioeconómico Comunal (1–5):",
                min_value=1,
                max_value=5,
                value=(1, 5),
                key=f"{key_prefix}_tier",
            )

        r4_c1, r4_c2, r4_c3 = st.columns([1.6, 1.2, 1.2])
        with r4_c1:
            sel_nlp_labels = st.multiselect(
                "✨ Exigir Atributos NLP en el Aviso:",
                options=list(NLP_LABELS_MAP.values()),
                default=[],
                placeholder="Ej: Único Dueño, Mant. al Día, Tapiz Cuero...",
                key=f"{key_prefix}_nlp",
            )
        with r4_c2:
            min_margin_clp = st.number_input(
                "💵 Margen Mínimo Ganancia (CLP):",
                min_value=-50_000_000,
                max_value=50_000_000,
                value=0,
                step=250_000,
                key=f"{key_prefix}_min_margin",
            )
        with r4_c3:
            free_query = st.text_input(
                "🔎 Palabra Clave / ID Aviso / Generación:",
                value="",
                placeholder="Ej: GEN4, MLC-, sunroof, 4x4...",
                key=f"{key_prefix}_free_q",
            )

        if allow_category_filter and "categoria_arbitraje" in work.columns:
            cats_disp = sorted(work["categoria_arbitraje"].dropna().astype(str).unique().tolist())
            _sanitize_multiselect_state(f"{key_prefix}_cats", cats_disp)
            default_cats = (
                ["OPORTUNIDAD"]
                if (default_only_opportunities and "OPORTUNIDAD" in cats_disp)
                else cats_disp
            )
            sel_cats = st.multiselect(
                "🏷️ Categoría de Arbitraje:",
                options=cats_disp,
                key=f"{key_prefix}_cats",
                **({} if f"{key_prefix}_cats" in st.session_state else {"default": default_cats}),
            )
        else:
            sel_cats = []

    # Combinar todas las máscaras booleanas antes de indexar (1 sola copia al final)
    mask = (
        (work["precio"] >= price_range[0])
        & (work["precio"] <= price_range[1])
        & (work["anio"] >= year_range[0])
        & (work["anio"] <= year_range[1])
        & (work["kilometraje"] >= km_range[0])
        & (work["kilometraje"] <= km_range[1])
        & ((work["descuento_pct"] * 100.0) >= disc_range[0])
        & ((work["descuento_pct"] * 100.0) <= disc_range[1])
        & (work["margen_ganancia_clp"] >= min_margin_clp)
    )

    if "comuna_tier" in work.columns:
        mask &= (work["comuna_tier"] >= tier_range[0]) & (work["comuna_tier"] <= tier_range[1])
    if sel_marcas:
        mask &= work["marca"].isin(sel_marcas)
    if sel_modelos:
        mask &= work["modelo"].astype(str).isin(sel_modelos)
    if sel_regiones:
        mask &= reg_col.isin(sel_regiones)
    if sel_portales:
        mask &= work["portal"].isin(sel_portales)
    if sel_trans:
        mask &= work["transmision"].isin(sel_trans)
    if sel_comb:
        mask &= work["combustible"].isin(sel_comb)
    if sel_comunas:
        mask &= work["comuna"].isin(sel_comunas)
    if allow_category_filter and sel_cats:
        mask &= work["categoria_arbitraje"].isin(sel_cats)

    inv_nlp_map = {v: k for k, v in NLP_LABELS_MAP.items()}
    for label in sel_nlp_labels:
        col_name = inv_nlp_map.get(label)
        if col_name and col_name in work.columns:
            mask &= work[col_name] == 1

    filtered = work.loc[mask].copy()

    # Búsqueda textual multi-término segura (regex=False evita crash con paréntesis/corchetes)
    terms = [t for t in free_query.strip().lower().split() if t]
    if terms and not filtered.empty:
        search_blob = (
            filtered["id_aviso"].fillna("").astype(str)
            + " "
            + filtered["marca"].fillna("").astype(str)
            + " "
            + filtered["modelo"].fillna("").astype(str)
            + " "
            + filtered.get("generacion", pd.Series("", index=filtered.index)).fillna("").astype(str)
            + " "
            + filtered.get("titulo", pd.Series("", index=filtered.index)).fillna("").astype(str)
            + " "
            + filtered.get("descripcion", pd.Series("", index=filtered.index)).fillna("").astype(str)
        ).str.lower()
        txt_mask = pd.Series(True, index=filtered.index)
        for term in terms:
            txt_mask &= search_blob.str.contains(term, na=False, regex=False)
        filtered = filtered.loc[txt_mask].copy()

    # Resumen compacto de filtros activos
    active_pills: List[str] = []
    if sel_marcas:
        active_pills.append(f"Marcas: {', '.join(sel_marcas)}")
    if sel_modelos:
        active_pills.append(f"Modelos: {', '.join(sel_modelos)}")
    if sel_regiones:
        active_pills.append(f"Regiones: {len(sel_regiones)}")
    if sel_portales and len(sel_portales) < len(portales_disp):
        active_pills.append(f"Portales: {', '.join(sel_portales)}")
    if terms:
        active_pills.append(f"Texto: '{free_query.strip()}'")
    if sel_nlp_labels:
        active_pills.append(f"NLP: {', '.join(sel_nlp_labels)}")
    if active_pills:
        st.caption(f"🔎 **Filtros activos:** {' | '.join(active_pills)}")

    # Ordenamiento dinámico
    sort_map = {
        "Mayor Margen Ganancia (CLP)": ("margen_ganancia_clp", False),
        "Mayor % Descuento vs IA": ("descuento_pct", False),
        "Mayor ROI (%)": ("roi_pct", False),
        "Menor Precio Publicado": ("precio", True),
        "Mayor Precio Publicado": ("precio", False),
        "Menor Kilometraje": ("kilometraje", True),
        "Año más Reciente": ("anio", False),
    }
    sort_col, ascending = sort_map.get(sort_by, ("margen_ganancia_clp", False))
    if sort_col in filtered.columns and not filtered.empty:
        filtered = filtered.sort_values(by=sort_col, ascending=ascending)

    return filtered


# ==============================================================================
# 6. PESTAÑA 1: 🔥 OPORTUNIDADES (BÚSQUEDA PARAMETRIZADA + FILTROS + GRÁFICOS)
# ==============================================================================
def _highlight_profit_cell(_val: Any) -> str:
    return "background-color: rgba(16, 185, 129, 0.22); color: #34d399; font-weight: bold;"


def _update_listing_status_and_refresh(
    id_aviso: str, new_status: str, label: str
) -> None:
    """Actualiza el estado del aviso en SQLite, limpia caché y registra acción para deshacer."""
    db_mgr = DatabaseManager(db_path=config.DB_PATH)
    ok = db_mgr.update_listing_status(id_aviso, new_status)
    if ok:
        load_dashboard_datasets.clear()
        if new_status == "activo":
            st.session_state.pop("last_dismissed_action", None)
        else:
            st.session_state["last_dismissed_action"] = {
                "id_aviso": str(id_aviso),
                "status": new_status,
                "label": label,
            }


def _toggle_favorite_and_refresh(id_aviso: str) -> bool:
    """Agrega o quita un aviso de Favoritos en SQLite y limpia caché del dashboard."""
    db_mgr = DatabaseManager(db_path=config.DB_PATH)
    is_fav = db_mgr.toggle_favorite(id_aviso)
    load_dashboard_datasets.clear()
    return is_fav


def _render_quick_edit_popover(row: pd.Series, key_prefix: str) -> None:
    """Popover compacto para corregir km, transmisión, combustible, año o precio y recalcular IA."""
    raw_id = str(row["id_aviso"])
    trans_opts = ["Automática", "Manual"]
    comb_opts = ["Bencina", "Diésel", "Híbrido", "Eléctrico"]
    cur_trans = str(row.get("transmision") or "Manual")
    cur_comb = str(row.get("combustible") or "Bencina")
    trans_idx = trans_opts.index(cur_trans) if cur_trans in trans_opts else 0
    comb_idx = comb_opts.index(cur_comb) if cur_comb in comb_opts else 0

    with st.popover("✏️ Corregir", width="stretch"):
        st.caption(f"Corregir datos de `{raw_id}` (protegido contra futuros escaneos)")
        ed_km = st.number_input(
            "Kilometraje (km):",
            min_value=0,
            max_value=config.MAX_VALID_KM,
            value=min(config.MAX_VALID_KM, max(0, int(row.get("kilometraje") or 0))),
            step=1_000,
            key=f"{key_prefix}_ed_km_{raw_id}",
        )
        ed_c1, ed_c2 = st.columns(2)
        with ed_c1:
            ed_trans = st.selectbox(
                "Transmisión:",
                options=trans_opts,
                index=trans_idx,
                key=f"{key_prefix}_ed_trans_{raw_id}",
            )
        with ed_c2:
            ed_comb = st.selectbox(
                "Combustible:",
                options=comb_opts,
                index=comb_idx,
                key=f"{key_prefix}_ed_comb_{raw_id}",
            )
        ed_c3, ed_c4 = st.columns(2)
        with ed_c3:
            ed_anio = st.number_input(
                "Año:",
                min_value=config.MIN_VALID_YEAR,
                max_value=config.MAX_VALID_YEAR,
                value=min(
                    config.MAX_VALID_YEAR,
                    max(config.MIN_VALID_YEAR, int(row.get("anio") or 2020)),
                ),
                step=1,
                key=f"{key_prefix}_ed_anio_{raw_id}",
            )
        with ed_c4:
            ed_precio = st.number_input(
                "Precio (CLP):",
                min_value=config.MIN_VALID_PRICE_CLP,
                max_value=config.MAX_VALID_PRICE_CLP,
                value=min(
                    config.MAX_VALID_PRICE_CLP,
                    max(
                        config.MIN_VALID_PRICE_CLP,
                        int(row.get("precio") or config.MIN_VALID_PRICE_CLP),
                    ),
                ),
                step=100_000,
                key=f"{key_prefix}_ed_precio_{raw_id}",
            )
        if st.button(
            "💾 Guardar y recalcular IA",
            key=f"{key_prefix}_ed_save_{raw_id}",
            type="primary",
            width="stretch",
        ):
            ok = DatabaseManager(db_path=config.DB_PATH).update_listing_specs(
                id_aviso=raw_id,
                kilometraje=int(ed_km),
                transmision=ed_trans,
                combustible=ed_comb,
                anio=int(ed_anio),
                precio=int(ed_precio),
            )
            if ok:
                load_dashboard_datasets.clear()
                km_str = f"{int(ed_km):,}".replace(",", ".")
                st.session_state["last_spec_edit_msg"] = (
                    f"✅ Datos de **{row['marca']} {row['modelo']} ({int(ed_anio)})** (`{raw_id}`) "
                    f"actualizados: **{km_str} km | {ed_trans} ({ed_comb})**. Precio Justo IA recalculado."
                )
                st.rerun(scope="app")


@st.fragment
def _render_opportunities_explorer(
    active_df: pd.DataFrame,
    opp_csv_df: pd.DataFrame,
    favorites_map: Dict[str, Dict[str, str]],
) -> None:
    """Fragmento aislado para explorar, filtrar y paginar oportunidades sin recargar toda la app."""
    if st.session_state.get("last_spec_edit_msg"):
        st.success(st.session_state.pop("last_spec_edit_msg"))
    last_act = st.session_state.get("last_dismissed_action")
    if isinstance(last_act, dict) and last_act.get("id_aviso"):
        st_lbl = (
            "No disponible (vendido)"
            if last_act.get("status") == "vendido"
            else "Link erróneo (descartado)"
        )
        u_col1, u_col2 = st.columns([4.2, 1.0])
        with u_col1:
            st.info(
                f"🗂️ Aviso **{last_act.get('label', '')}** (`{last_act['id_aviso']}`) marcado como **{st_lbl}** y removido de Oportunidades."
            )
        with u_col2:
            if st.button("↩️ Deshacer", key="opp_undo_dismiss_btn", width="stretch"):
                _update_listing_status_and_refresh(
                    last_act["id_aviso"], "activo", last_act.get("label", "")
                )
                st.rerun(scope="app")

    st.markdown("### 🎛️ Filtros de Exploración sobre Oportunidades y Base Activa")
    src_col1, src_col2 = st.columns([2.2, 1.8])
    with src_col1:
        source_mode = st.radio(
            "Conjunto de Datos a Filtrar:",
            options=[
                "🌐 Toda la Base de Datos Activa en SQLite (Permite ajustar % Descuento libremente)",
                "🎯 Solo el Reporte del Último Escaneo (data/oportunidades_detectadas.csv)",
            ],
            index=0,
            horizontal=True,
            key="opp_source_radio",
        )
    with src_col2:
        view_layout = st.radio(
            "Formato de Visualización:",
            options=["Tarjetas + Tabla + Gráficos ROI", "Solo Tabla + Gráficos ROI", "Solo Tarjetas"],
            index=0,
            horizontal=True,
            key="opp_view_layout",
        )

    base_source_df = (
        opp_csv_df
        if (source_mode.startswith("🎯") and not opp_csv_df.empty)
        else active_df
    )

    if base_source_df.empty:
        st.info("No hay vehículos evaluados disponibles. Ejecuta una búsqueda en vivo arriba.")
        return

    # Aplicar el motor de filtrado multifacético (por defecto muestra Oportunidades con 15%-55% dcto)
    filtered_opps = apply_multifacet_filters(
        base_source_df,
        key_prefix="opp_flt",
        default_only_opportunities=True,
        allow_category_filter=True,
    )

    scam_df = base_source_df.loc[
        base_source_df["categoria_arbitraje"] == "POSIBLE ESTAFA / ERROR DE TIPEO"
    ].sort_values(by="descuento_pct", ascending=False)

    # 3. KPIS DINÁMICOS DEL SUBCONJUNTO FILTRADO
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("🔥 Vehículos Filtrados", len(filtered_opps))
    with k2:
        top_margin = filtered_opps["margen_ganancia_clp"].max() if not filtered_opps.empty else 0
        st.metric("💰 Mayor Margen (CLP)", format_clp(top_margin))
    with k3:
        avg_margin = filtered_opps["margen_ganancia_clp"].mean() if not filtered_opps.empty else 0
        st.metric("📊 Margen Promedio", format_clp(avg_margin))
    with k4:
        avg_dcto = (filtered_opps["descuento_pct"].mean() * 100.0) if not filtered_opps.empty else 0.0
        st.metric("📉 Descuento Medio IA", f"-{avg_dcto:.1f}%")
    with k5:
        avg_roi = filtered_opps["roi_pct"].mean() if not filtered_opps.empty else 0.0
        st.metric("📈 ROI Medio Estimado", f"+{avg_roi:.1f}%")

    st.markdown("---")

    if filtered_opps.empty:
        st.warning(
            "No hay vehículos que cumplan simultáneamente con todos los filtros seleccionados. "
            "Prueba ampliando el rango de precio, reduciendo el % de descuento mínimo o pulsando '🧹 Limpiar'."
        )
        return

    # 4. GRÁFICOS INTERACTIVOS DE RENTABILIDAD EN PESTAÑA DE OPORTUNIDADES
    if "Gráficos ROI" in view_layout and len(filtered_opps) >= 2:
        with st.expander(
            "📊 **Visualización Gráfica de Oportunidades: Mapa de Rentabilidad (ROI) y Distribución por Marca/Portal**",
            expanded=True,
        ):
            g_col1, g_col2 = st.columns([1.35, 1.0])
            with g_col1:
                portal_palette = ["#38bdf8", "#a855f7", "#22c55e", "#f97316", "#facc15", "#f472b6"]
                fig_roi = go.Figure()
                for idx_p, (p_name, grp) in enumerate(filtered_opps.groupby("portal", sort=False)):
                    fig_roi.add_trace(
                        go.Scatter(
                            x=grp["precio"],
                            y=grp["margen_ganancia_clp"],
                            mode="markers",
                            name=str(p_name),
                            marker=dict(
                                size=np.clip(grp["descuento_pct"] * 38.0, 7.0, 22.0),
                                color=portal_palette[idx_p % len(portal_palette)],
                                opacity=0.82,
                                line=dict(width=1, color="rgba(255,255,255,0.25)"),
                            ),
                            customdata=np.stack(
                                [
                                    grp["id_aviso"],
                                    grp["marca"],
                                    grp["modelo"],
                                    grp["anio"],
                                    grp["kilometraje"],
                                    grp["precio_justo_ia"],
                                    grp["roi_pct"],
                                    grp["comuna"],
                                ],
                                axis=-1,
                            ),
                            hovertemplate=(
                                "<b>%{customdata[0]} — %{customdata[1]} %{customdata[2]} (%{customdata[3]})</b><br>"
                                "Comuna: %{customdata[7]} | Km: %{customdata[4]:,.0f} km<br>"
                                "Precio Publicado: <b>$ %{x:,.0f}</b><br>"
                                "Precio Justo IA: $ %{customdata[5]:,.0f}<br>"
                                "Margen IA: <b>$ %{y:,.0f}</b> (ROI: +%{customdata[6]:.1f}%)<extra></extra>"
                            ),
                        )
                    )
                fig_roi.update_layout(
                    title="🎯 Matriz Capital Requerido (Precio Publicado) vs Margen de Ganancia IA",
                    xaxis_title="Capital Requerido / Precio Publicado (CLP)",
                    yaxis_title="Margen Potencial de Ganancia (CLP)",
                    legend_title_text="Portal",
                    template=get_plotly_template(),
                    height=370,
                    xaxis=dict(tickprefix="$ ", separatethousands=True),
                    yaxis=dict(tickprefix="$ ", separatethousands=True),
                    margin=dict(l=20, r=20, t=45, b=20),
                )
                st.plotly_chart(fig_roi, width="stretch")

            with g_col2:
                brand_summary = (
                    filtered_opps.groupby("marca", as_index=False)
                    .agg(
                        cantidad=("id_aviso", "count"),
                        margen_medio=("margen_ganancia_clp", "mean"),
                    )
                    .sort_values("cantidad", ascending=True)
                )
                fig_brand = go.Figure(
                    go.Bar(
                        x=brand_summary["cantidad"],
                        y=brand_summary["marca"],
                        orientation="h",
                        text=[
                            f"{cnt} autos (Prom: {format_clp(m)})"
                            for cnt, m in zip(brand_summary["cantidad"], brand_summary["margen_medio"])
                        ],
                        textposition="auto",
                        marker=dict(
                            color=brand_summary["margen_medio"],
                            colorscale="Viridis",
                            showscale=False,
                        ),
                    )
                )
                fig_brand.update_layout(
                    title="🏆 Oportunidades y Margen Medio por Marca",
                    xaxis_title="Cantidad de Vehículos Filtrados",
                    yaxis_title="Marca",
                    template=get_plotly_template(),
                    height=370,
                    margin=dict(l=20, r=20, t=45, b=20),
                )
                st.plotly_chart(fig_brand, width="stretch")

    # 5. TARJETAS ESTILIZADAS CON PAGINACIÓN Y ESCAPADO HTML
    if "Tarjetas" in view_layout:
        card_page_sizes = [6, 12, 24]
        if (
            "opp_max_cards" in st.session_state
            and st.session_state["opp_max_cards"] not in card_page_sizes
        ):
            st.session_state["opp_max_cards"] = 12

        c_title, c_limit, c_page = st.columns([2.4, 0.8, 0.8])
        with c_title:
            st.markdown("### 🏆 Tarjetas de Inversión Automotriz (Enlaces Directos)")
        with c_limit:
            max_cards = st.selectbox(
                "Tarjetas / pág:",
                options=card_page_sizes,
                key="opp_max_cards",
                **({} if "opp_max_cards" in st.session_state else {"index": 1}),
            )

        total_rows = len(filtered_opps)
        total_pages = max(1, (total_rows + int(max_cards) - 1) // int(max_cards))
        if "opp_card_page" in st.session_state:
            clamped_page = min(max(1, int(st.session_state["opp_card_page"])), total_pages)
            if clamped_page != st.session_state["opp_card_page"]:
                st.session_state["opp_card_page"] = clamped_page
        with c_page:
            cur_page = st.number_input(
                f"Página (1–{total_pages}):",
                min_value=1,
                max_value=total_pages,
                step=1,
                key="opp_card_page",
                **({} if "opp_card_page" in st.session_state else {"value": 1}),
            )

        start_idx = (int(cur_page) - 1) * int(max_cards)
        end_idx = start_idx + int(max_cards)
        cards_subset = filtered_opps.iloc[start_idx:end_idx]
        rows_list = [r for _, r in cards_subset.iterrows()]

        for i in range(0, len(rows_list), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(rows_list):
                    row = rows_list[i + j]
                    marca_s = html.escape(str(row["marca"]))
                    modelo_s = html.escape(str(row["modelo"]))
                    emoji = get_vehicle_emoji(str(row["marca"]), str(row["modelo"]))
                    portal_name, portal_label, portal_css = resolve_portal_info(row["id_aviso"])
                    id_s = html.escape(str(row["id_aviso"]))
                    portal_name_s = html.escape(portal_name)
                    portal_label_s = html.escape(portal_label)

                    raw_title = str(row.get("titulo") or f"{row['marca']} {row['modelo']} {int(row['anio'])}").strip()
                    short_title = (raw_title[:55] + "...") if len(raw_title) > 58 else raw_title
                    title_sub_s = html.escape(short_title)

                    comuna_s = html.escape(str(row.get("comuna") or "Santiago"))
                    region_s = html.escape(str(row.get("region") or resolve_region(row.get("comuna"))))
                    tier_val = int(row.get("comuna_tier") or 2)
                    trans_s = html.escape(str(row.get("transmision") or "Manual"))
                    comb_s = html.escape(str(row.get("combustible") or "Bencina"))
                    gen_s = html.escape(str(row.get("generacion") or ""))
                    nlp_s = html.escape(extract_nlp_tags(row))
                    url_s = html.escape(str(row.get("url") or "#"), quote=True)

                    precio_pub_str = format_clp(row["precio"])
                    precio_ia_str = format_clp(row["precio_justo_ia"])
                    margen_str = format_clp(row["margen_ganancia_clp"])
                    dcto_str = f"-{float(row['descuento_pct']) * 100:.1f}%"
                    roi_str = f"+{float(row.get('roi_pct', 0.0)):.1f}% ROI"
                    km_str = f"{int(row['kilometraje']):,} km".replace(",", ".")

                    dias_pub = int(row.get("dias_publicado") or 0)
                    dias_label = "Nuevo hoy" if dias_pub == 0 else f"🕒 {dias_pub}d"

                    drop_html = ""
                    cambios_p = int(row.get("cambios_precio") or 0)
                    precio_ant = int(row.get("precio_anterior") or 0)
                    if cambios_p > 0 and precio_ant > int(row["precio"]):
                        drop_clp = precio_ant - int(row["precio"])
                        drop_html = f'<span class="drop-badge">📉 Bajó {format_clp(drop_clp)}</span>'

                    multi_raw = str(row.get("multi_portal_info") or "")
                    multi_html = ""
                    if multi_raw.startswith("Más barato"):
                        multi_html = f'<div class="multi-badge">💎 {html.escape(multi_raw)}</div>'

                    card_html = _clean_html(
                        f"""
                        <div class="opp-card">
                            <div class="car-header">
                                <span class="car-title">{emoji} {marca_s} {modelo_s} ({int(row['anio'])})</span>
                                <span class="{portal_css}">{portal_label_s}</span>
                            </div>
                            <div class="car-subtitle-trim" title="{html.escape(raw_title, quote=True)}">{title_sub_s}</div>
                            <div class="car-meta">
                                🗺️ <b>{region_s}</b> — 📍 <b>{comuna_s}</b> (Tier {tier_val}/5) &nbsp;<span class="days-pill">{dias_label}</span><br>
                                🛣️ {km_str} &nbsp;|&nbsp; ⚙️ {trans_s} ({comb_s}) &nbsp;|&nbsp; 🧬 <code>{gen_s}</code><br>
                                ✨ {nlp_s}
                                {drop_html}
                                {multi_html}
                            </div>
                            <div class="price-grid">
                                <div>
                                    <div class="price-label">Precio Publicado</div>
                                    <div class="price-val-pub">{precio_pub_str}</div>
                                </div>
                                <div>
                                    <div class="price-label">Precio Justo IA</div>
                                    <div class="price-val-ia">{precio_ia_str}</div>
                                </div>
                            </div>
                            <div class="profit-banner">
                                <span>🟢 +{margen_str} ({dcto_str})</span>
                                <span>{roi_str}</span>
                            </div>
                            <a href="{url_s}" target="_blank" rel="noopener noreferrer" class="btn-portal">
                                🔗 Abrir Aviso en {portal_name_s} ({id_s}) ↗
                            </a>
                        </div>
                        """
                    )
                    card_label = f"{row['marca']} {row['modelo']} ({int(row['anio'])})"
                    raw_id_aviso = str(row["id_aviso"])
                    is_fav = raw_id_aviso in favorites_map
                    with col:
                        st.markdown(card_html, unsafe_allow_html=True)
                        act_r1_c1, act_r1_c2 = st.columns(2)
                        with act_r1_c1:
                            if st.button(
                                "💛 Guardado" if is_fav else "⭐ Guardar",
                                key=f"opp_fav_{i + j}_{raw_id_aviso}",
                                help="Guarda o quita esta oferta de tu ventana de ⭐ Favoritos.",
                                width="stretch",
                            ):
                                _toggle_favorite_and_refresh(raw_id_aviso)
                                st.rerun(scope="app")
                        with act_r1_c2:
                            _render_quick_edit_popover(row, key_prefix=f"opp_{i + j}")

                        act_r2_c1, act_r2_c2 = st.columns(2)
                        with act_r2_c1:
                            if st.button(
                                "🚫 No disponible",
                                key=f"opp_sold_{i + j}_{raw_id_aviso}",
                                help="Marca este aviso como vendido / no disponible para quitarlo de Oportunidades.",
                                width="stretch",
                            ):
                                _update_listing_status_and_refresh(
                                    raw_id_aviso, "vendido", card_label
                                )
                                st.rerun(scope="app")
                        with act_r2_c2:
                            if st.button(
                                "⚠️ Link erróneo",
                                key=f"opp_bad_{i + j}_{raw_id_aviso}",
                                help="Descarta permanentemente este aviso por link roto o vehículo distinto (no volverá a aparecer).",
                                width="stretch",
                            ):
                                _update_listing_status_and_refresh(
                                    raw_id_aviso, "descartado", card_label
                                )
                                st.rerun(scope="app")

    # 6. TABLA INTERACTIVA Y BOTÓN DE DESCARGA CSV LAZY
    if "Tabla" in view_layout:
        t_head1, t_head2 = st.columns([3, 1])
        with t_head1:
            st.markdown("### 📋 Tabla Detallada de Oportunidades Filtradas")
        with t_head2:
            st.download_button(
                label="📥 Descargar Filtro en CSV",
                data=lambda df_exp=filtered_opps: df_exp.to_csv(index=False).encode("utf-8"),
                file_name=f"oportunidades_filtradas_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                on_click="ignore",
                key="opp_download_csv_btn",
                width="stretch",
            )

        multi_col = (
            filtered_opps["multi_portal_info"]
            if "multi_portal_info" in filtered_opps.columns
            else "Publicación única"
        )
        reg_series = (
            filtered_opps["region"]
            if "region" in filtered_opps.columns
            else filtered_opps["comuna"].apply(resolve_region)
        )
        table_df = pd.DataFrame(
            {
                "Portal": filtered_opps["portal_badge"],
                "ID Aviso": filtered_opps["id_aviso"],
                "Marca": filtered_opps["marca"],
                "Modelo": filtered_opps["modelo"],
                "Año": filtered_opps["anio"].astype(int),
                "Generación": filtered_opps["generacion"],
                "Kilometraje": filtered_opps["kilometraje"].astype(int),
                "Transmisión": filtered_opps["transmision"],
                "Combustible": filtered_opps["combustible"],
                "Región": reg_series,
                "Comuna": filtered_opps["comuna"],
                "Precio Publicado (CLP)": filtered_opps["precio"].astype(int),
                "Precio Justo IA (CLP)": filtered_opps["precio_justo_ia"].round(0).astype(int),
                "Margen Ganancia (CLP)": filtered_opps["margen_ganancia_clp"].round(0).astype(int),
                "Descuento (%)": (filtered_opps["descuento_pct"] * 100.0).round(1),
                "ROI (%)": filtered_opps["roi_pct"].round(1),
                "Multi-Portal": multi_col,
                "Publicación Real": filtered_opps["url"],
            }
        )

        col_cfg = {
            "Año": st.column_config.NumberColumn("Año", format="%d"),
            "Kilometraje": st.column_config.NumberColumn("Kilometraje", format="%d km"),
            "Precio Publicado (CLP)": st.column_config.NumberColumn(
                "Precio Publicado", format="$ %d"
            ),
            "Precio Justo IA (CLP)": st.column_config.NumberColumn(
                "Precio Justo IA", format="$ %d"
            ),
            "Margen Ganancia (CLP)": st.column_config.NumberColumn(
                "🟢 Margen Ganancia", format="+ $ %d"
            ),
            "ROI (%)": st.column_config.NumberColumn("ROI", format="+%.1f %%"),
            "Publicación Real": st.column_config.LinkColumn(
                "Enlace Directo",
                display_text="🔗 Ver Publicación",
            ),
        }

        if len(table_df) <= 300:
            col_cfg["Descuento (%)"] = st.column_config.NumberColumn("Dcto. IA", format="-%.1f %%")
            df_display: Any = table_df.style.map(
                _highlight_profit_cell,
                subset=["Margen Ganancia (CLP)", "Descuento (%)", "ROI (%)"],
            )
        else:
            col_cfg["Descuento (%)"] = st.column_config.ProgressColumn(
                "Dcto. IA",
                min_value=0,
                max_value=60,
                format="-%.1f %%",
            )
            df_display = table_df

        st.dataframe(
            df_display,
            column_config=col_cfg,
            hide_index=True,
            width="stretch",
        )

    # 7. ALERTAS MODERADAS ANTI-ESTAFA
    if not scam_df.empty:
        with st.expander(
            f"⚠️ Alertas Moderadas: Posibles Estafas / Errores de Tipeo / Prenda ({len(scam_df)} detectadas)",
            expanded=False,
        ):
            scam_view = pd.DataFrame(
                {
                    "Portal": scam_df["portal_badge"],
                    "ID Aviso": scam_df["id_aviso"],
                    "Marca": scam_df["marca"],
                    "Modelo": scam_df["modelo"],
                    "Año": scam_df["anio"].astype(int),
                    "Kilometraje": scam_df["kilometraje"].astype(int),
                    "Comuna": scam_df["comuna"],
                    "Precio Publicado": scam_df["precio"].astype(int),
                    "Precio Justo IA": scam_df["precio_justo_ia"].round(0).astype(int),
                    "Descuento (%)": (scam_df["descuento_pct"] * 100.0).round(1),
                    "URL": scam_df["url"],
                }
            )
            st.dataframe(
                scam_view,
                column_config={
                    "Precio Publicado": st.column_config.NumberColumn(format="$ %d"),
                    "Precio Justo IA": st.column_config.NumberColumn(format="$ %d"),
                    "Descuento (%)": st.column_config.NumberColumn(format="-%.1f %%"),
                    "URL": st.column_config.LinkColumn("Enlace", display_text="🔗 Inspeccionar"),
                },
                hide_index=True,
                width="stretch",
            )


def render_tab_opportunities(data_bundle: Dict[str, Any]) -> None:
    """
    Pestaña 1 mejorada:
    1. Panel de Búsqueda en Vivo con Parámetros Objetivo (Marca, Modelo, Páginas, Rango de Año,
       Precio Mín/Máx, Km Máx, Chileautos Profundo, Alertas Telegram y Re-entrenamiento).
    2. Fragmento interactivo de exploración sobre Oportunidades / Base Activa.
    """
    active_df = data_bundle["active_featured"]
    opp_csv_df = data_bundle["opp_csv_df"]
    clean_df = data_bundle["clean_df"]

    if st.session_state.get("last_scan_message"):
        st.success(st.session_state["last_scan_message"])

    # 1. PANEL DE BÚSQUEDA EN VIVO CON FILTROS OBJETIVO DE CAPTURA
    with st.expander(
        "🎯 **Panel de Búsqueda en Vivo Multi-Portal + Filtros Objetivo de Captura**",
        expanded=True,
    ):
        all_segments = config.get_all_target_segments()
        target_marcas = sorted({m for m, _ in all_segments})
        db_marcas = sorted(clean_df["marca"].unique().tolist()) if not clean_df.empty else []
        all_marcas = sorted(set(target_marcas + db_marcas))

        curr_marca = st.session_state.get("focused_marca") or "MAZDA"
        m_idx = (all_marcas.index(curr_marca) + 1) if curr_marca in all_marcas else 1

        b_col1, b_col2, b_col3, b_col4 = st.columns([1.35, 1.35, 1.15, 1.35])
        with b_col1:
            marca_sel = st.selectbox(
                "1️⃣ Marca a Buscar en Portales:",
                options=[f"TODAS (Barrido {len(all_segments)} Modelos)"] + all_marcas + ["➕ OTRA MARCA (Escribir nueva)"],
                index=m_idx,
                key="main_search_marca",
            )
            if marca_sel == "➕ OTRA MARCA (Escribir nueva)":
                input_marca = (
                    st.text_input("Nueva Marca:", value="AUDI", key="main_custom_marca")
                    .strip()
                    .upper()
                )
            elif marca_sel.startswith("TODAS"):
                input_marca = "TODAS"
            else:
                input_marca = marca_sel

        with b_col2:
            if input_marca == "TODAS":
                suggested_models = []
            else:
                suggested_models = sorted(
                    {str(mod) for m, mod in all_segments if m == input_marca}.union(
                        set(
                            clean_df.loc[clean_df["marca"] == input_marca, "modelo"]
                            .astype(str)
                            .unique()
                            if not clean_df.empty
                            else []
                        )
                    )
                )
            mod_opts = (
                ["TODOS (Toda la Marca)"]
                + suggested_models
                + ["➕ OTRO MODELO (Escribir nuevo)"]
            )
            default_mod_str = (
                st.session_state.get("focused_modelo")
                if st.session_state.get("focused_marca") == input_marca
                and st.session_state.get("focused_modelo")
                else (suggested_models[0] if suggested_models else "TODOS (Toda la Marca)")
            )
            mod_sel_idx = mod_opts.index(default_mod_str) if default_mod_str in mod_opts else 0
            mod_sel = st.selectbox(
                "2️⃣ Modelo a Buscar:",
                options=mod_opts,
                index=mod_sel_idx,
                key="main_search_modelo_select",
            )
            if mod_sel == "➕ OTRO MODELO (Escribir nuevo)":
                input_modelo = normalize_model_input(
                    st.text_input(
                        "Nuevo Modelo:",
                        value="",
                        placeholder="Ej: CIVIC, XC60, 320I",
                        key="main_custom_modelo",
                    )
                )
            elif mod_sel == "TODOS (Toda la Marca)":
                input_modelo = "TODOS"
            else:
                input_modelo = mod_sel

        with b_col3:
            input_pages = st.slider(
                "3️⃣ Páginas/Portal:",
                min_value=1,
                max_value=5,
                value=2,
                key="main_search_pages",
            )

        with b_col4:
            input_alerts = st.slider(
                "📩 Alertas Telegram:",
                min_value=0,
                max_value=10,
                value=3,
                help="0 = Solo actualizar dashboard sin enviar mensajes a Telegram.",
                key="main_search_alerts",
            )
            force_retrain_chk = st.checkbox(
                "🧠 Re-entrenar IA tras búsqueda",
                value=False,
                key="main_search_retrain",
            )

        st.markdown("##### ⚙️ Parámetros y Filtros de Extracción (Chileautos + MercadoLibre + Autocosmos + Yapo)")
        t1_apply_years = st.toggle(
            "📅 Restringir rango de años",
            value=False,
            help="Actívalo para acotar la descarga en portales y consultar Chileautos año por año dentro del rango.",
            key="t1_apply_years_chk",
        )
        f_row1, f_row2, f_row3, f_row4, f_row5 = st.columns(5)
        with f_row1:
            t1_anio_min = st.number_input(
                "Año Mínimo:",
                min_value=config.MIN_VALID_YEAR,
                max_value=config.MAX_VALID_YEAR,
                value=2016,
                step=1,
                disabled=not t1_apply_years,
                key="t1_search_anio_min",
            )
        with f_row2:
            t1_anio_max = st.number_input(
                "Año Máximo:",
                min_value=config.MIN_VALID_YEAR,
                max_value=config.MAX_VALID_YEAR,
                value=2024,
                step=1,
                disabled=not t1_apply_years,
                key="t1_search_anio_max",
            )
        with f_row3:
            t1_precio_min = st.number_input(
                "Precio Mín (CLP):",
                min_value=0,
                max_value=config.MAX_VALID_PRICE_CLP,
                value=0,
                step=1_000_000,
                help="0 = Sin mínimo",
                key="t1_search_precio_min",
            )
        with f_row4:
            t1_precio_max = st.number_input(
                "Precio Tope (CLP):",
                min_value=0,
                max_value=config.MAX_VALID_PRICE_CLP,
                value=0,
                step=1_000_000,
                help="0 = Sin límite de precio",
                key="t1_search_precio_max",
            )
        with f_row5:
            t1_km_max = st.number_input(
                "Km Máximo:",
                min_value=0,
                max_value=config.MAX_VALID_KM,
                value=0,
                step=10_000,
                help="0 = Sin límite de kilometraje",
                key="t1_search_km_max",
            )

        f_sub1, f_sub2 = st.columns([1.8, 1.2])
        with f_sub1:
            t1_deep = st.checkbox(
                "⚡ Búsqueda Profunda en Chileautos (Año x Año)",
                value=True,
                help="Barre año por año en Google para extraer decenas de autos adicionales de Chileautos.",
                key="t1_deep_chk",
            )
        with f_sub2:
            launch_search = st.button(
                "🚀 Buscar y Analizar en Vivo",
                key="main_search_btn",
                type="primary",
                width="stretch",
            )

        if launch_search:
            norm_modelo = None if input_modelo in ("TODOS", "") else normalize_model_input(input_modelo)
            if input_marca != "TODAS" and not input_marca:
                st.warning("Ingresa una Marca válida para iniciar la búsqueda.")
            elif t1_apply_years and t1_anio_min > t1_anio_max:
                st.warning("Año mínimo no puede ser mayor al año máximo.")
            elif t1_precio_min > 0 and t1_precio_max > 0 and t1_precio_min > t1_precio_max:
                st.warning("Precio Mínimo no puede ser mayor al Precio Tope.")
            else:
                if input_marca != "TODAS" and norm_modelo:
                    config.add_custom_segment(input_marca, norm_modelo)
                with st.spinner(
                    f"Escaneando MercadoLibre, Autocosmos, Yapo y Chileautos en vivo ({input_marca} {norm_modelo or 'TODOS'})..."
                ):
                    ok, msg = execute_backend_cycle(
                        force_train=force_retrain_chk,
                        marca_filter=None if input_marca == "TODAS" else input_marca,
                        modelo_filter=norm_modelo,
                        pages_per_portal=input_pages,
                        max_telegram_alerts=input_alerts,
                        anio_min=int(t1_anio_min) if t1_apply_years else None,
                        anio_max=int(t1_anio_max) if t1_apply_years else None,
                        precio_min=int(t1_precio_min) if t1_precio_min > 0 else None,
                        precio_max=int(t1_precio_max) if t1_precio_max > 0 else None,
                        km_max=int(t1_km_max) if t1_km_max > 0 else None,
                        deep=t1_deep,
                    )
                if ok:
                    if input_marca != "TODAS":
                        st.session_state["opp_flt_marcas"] = [input_marca]
                    if norm_modelo:
                        st.session_state["opp_flt_modelos"] = [norm_modelo]
                    else:
                        st.session_state.pop("opp_flt_modelos", None)
                    st.session_state["opp_card_page"] = 1
                    st.session_state["last_scan_message"] = msg
                    st.rerun(scope="app")
                else:
                    st.error(msg)

    # 2. EXPLORACIÓN DE OPORTUNIDADES DENTRO DE FRAGMENTO AISLADO
    _render_opportunities_explorer(
        active_df, opp_csv_df, data_bundle.get("favorites_map", {})
    )


# ==============================================================================
# 6B. PESTAÑA DE FAVORITOS: ⭐ OFERTAS GUARDADAS + NOTAS PERSONALES
# ==============================================================================
@st.fragment
def _render_favorites_fragment(
    full_df: pd.DataFrame, favorites_map: Dict[str, Dict[str, str]]
) -> None:
    """Fragmento aislado para visualizar, ordenar, anotar y gestionar ofertas favoritas."""
    if st.session_state.get("last_spec_edit_msg"):
        st.success(st.session_state.pop("last_spec_edit_msg"))

    st.markdown("### ⭐ Mis Ofertas Favoritas Guardadas")
    st.caption(
        "Aquí se guardan de forma persistente las ofertas que marcaste con **⭐ Guardar** en la pestaña "
        "de Oportunidades. Puedes agregar notas personales, exportarlas o controlar si bajaron de precio o se vendieron."
    )

    if not favorites_map or full_df.empty:
        st.info(
            "Todavía no tienes vehículos guardados en Favoritos. "
            "Ve a **🔥 Oportunidades** y pulsa **⭐ Guardar** debajo de cualquier tarjeta que te interese."
        )
        return

    fav_ids = list(favorites_map.keys())
    fav_df = full_df[full_df["id_aviso"].astype(str).isin(fav_ids)].copy()
    if fav_df.empty:
        st.info("Los avisos guardados ya no se encuentran en la base de datos local.")
        return

    fav_df["nota_personal"] = fav_df["id_aviso"].astype(str).map(
        lambda x: favorites_map.get(x, {}).get("nota", "")
    )
    fav_df["fecha_guardado"] = fav_df["id_aviso"].astype(str).map(
        lambda x: favorites_map.get(x, {}).get("fecha_guardado", "")
    )

    # Controles superiores: solo activos, orden y exportación CSV
    ctrl1, ctrl2, ctrl3 = st.columns([1.4, 1.6, 1.2])
    with ctrl1:
        only_active_favs = st.toggle(
            "🟢 Mostrar solo avisos activos",
            value=False,
            help="Actívalo para ocultar favoritos que ya fueron marcados como vendidos o descartados.",
            key="fav_only_active_toggle",
        )
    with ctrl2:
        fav_sort = st.selectbox(
            "🔃 Ordenar favoritos por:",
            options=[
                "Guardado más reciente",
                "Mayor Margen Ganancia (CLP)",
                "Mayor % Descuento vs IA",
                "Menor Precio Publicado",
                "Año más Reciente",
            ],
            index=0,
            key="fav_sort_select",
        )

    view_df = (
        fav_df.loc[fav_df["estado"].astype(str).str.lower() == "activo"].copy()
        if only_active_favs
        else fav_df.copy()
    )

    sort_cfg = {
        "Guardado más reciente": ("fecha_guardado", False),
        "Mayor Margen Ganancia (CLP)": ("margen_ganancia_clp", False),
        "Mayor % Descuento vs IA": ("descuento_pct", False),
        "Menor Precio Publicado": ("precio", True),
        "Año más Reciente": ("anio", False),
    }
    s_col, s_asc = sort_cfg.get(fav_sort, ("fecha_guardado", False))
    if s_col in view_df.columns and not view_df.empty:
        view_df = view_df.sort_values(by=s_col, ascending=s_asc)

    with ctrl3:
        st.markdown("<div style='height: 1.65rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Exportar Favoritos CSV",
            data=lambda df_exp=view_df: df_exp.to_csv(index=False).encode("utf-8"),
            file_name=f"favoritos_arbitraje_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            on_click="ignore",
            key="fav_download_csv_btn",
            width="stretch",
        )

    # KPIs de Favoritos
    active_fav_count = int((fav_df["estado"].astype(str).str.lower() == "activo").sum())
    fk1, fk2, fk3, fk4, fk5 = st.columns(5)
    with fk1:
        st.metric("⭐ Total Guardados", len(fav_df))
    with fk2:
        st.metric("🟢 Activos", active_fav_count)
    with fk3:
        max_m = view_df["margen_ganancia_clp"].max() if not view_df.empty else 0
        st.metric("💰 Mayor Margen", format_clp(max_m))
    with fk4:
        avg_d = (view_df["descuento_pct"].mean() * 100.0) if not view_df.empty else 0.0
        st.metric("📉 Descuento Medio", f"-{avg_d:.1f}%")
    with fk5:
        avg_p = view_df["precio"].mean() if not view_df.empty else 0
        st.metric("💵 Precio Promedio", format_clp(avg_p))

    st.markdown("---")

    if view_df.empty:
        st.warning("No hay favoritos que coincidan con el filtro actual (prueba desactivar 'Mostrar solo avisos activos').")
        return

    rows_list = [r for _, r in view_df.iterrows()]
    for i in range(0, len(rows_list), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(rows_list):
                row = rows_list[i + j]
                raw_id = str(row["id_aviso"])
                marca_s = html.escape(str(row["marca"]))
                modelo_s = html.escape(str(row["modelo"]))
                emoji = get_vehicle_emoji(str(row["marca"]), str(row["modelo"]))
                portal_name, portal_label, portal_css = resolve_portal_info(raw_id)
                id_s = html.escape(raw_id)
                portal_name_s = html.escape(portal_name)
                portal_label_s = html.escape(portal_label)

                raw_title = str(row.get("titulo") or f"{row['marca']} {row['modelo']} {int(row['anio'])}").strip()
                short_title = (raw_title[:55] + "...") if len(raw_title) > 58 else raw_title
                title_sub_s = html.escape(short_title)

                comuna_s = html.escape(str(row.get("comuna") or "Santiago"))
                region_s = html.escape(str(row.get("region") or resolve_region(row.get("comuna"))))
                tier_val = int(row.get("comuna_tier") or 2)
                trans_s = html.escape(str(row.get("transmision") or "Manual"))
                comb_s = html.escape(str(row.get("combustible") or "Bencina"))
                gen_s = html.escape(str(row.get("generacion") or ""))
                nlp_s = html.escape(extract_nlp_tags(row))
                url_s = html.escape(str(row.get("url") or "#"), quote=True)

                precio_pub_str = format_clp(row["precio"])
                precio_ia_str = format_clp(row["precio_justo_ia"])
                margen_str = format_clp(row["margen_ganancia_clp"])
                dcto_val = float(row.get("descuento_pct", 0.0)) * 100.0
                dcto_str = f"{-dcto_val:+.1f}%" if dcto_val < 0 else f"-{dcto_val:.1f}%"
                roi_str = f"{float(row.get('roi_pct', 0.0)):+.1f}% ROI"
                km_str = f"{int(row['kilometraje']):,} km".replace(",", ".")

                estado_val = str(row.get("estado") or "activo").lower()
                if estado_val == "activo":
                    estado_badge = '<span class="portal-badge-cl">🟢 ACTIVO</span>'
                elif estado_val == "vendido":
                    estado_badge = '<span class="drop-badge">🔴 NO DISPONIBLE</span>'
                else:
                    estado_badge = '<span class="drop-badge">⚠️ DESCARTADO</span>'

                saved_raw = str(row.get("fecha_guardado") or "")
                try:
                    saved_fmt = datetime.fromisoformat(saved_raw).strftime("%d/%m %H:%M")
                except Exception:
                    saved_fmt = saved_raw[:16]

                drop_html = ""
                cambios_p = int(row.get("cambios_precio") or 0)
                precio_ant = int(row.get("precio_anterior") or 0)
                if cambios_p > 0 and precio_ant > int(row["precio"]):
                    drop_clp = precio_ant - int(row["precio"])
                    drop_html = f'<span class="drop-badge">📉 Bajó {format_clp(drop_clp)}</span>'

                fav_card_html = _clean_html(
                    f"""
                    <div class="opp-card" style="border-left-color: #f59e0b;">
                        <div class="car-header">
                            <span class="car-title">{emoji} {marca_s} {modelo_s} ({int(row['anio'])})</span>
                            <span class="{portal_css}">{portal_label_s}</span>
                        </div>
                        <div class="car-subtitle-trim" title="{html.escape(raw_title, quote=True)}">{title_sub_s}</div>
                        <div class="car-meta">
                            {estado_badge} &nbsp;<span class="days-pill">⭐ Guardado: {html.escape(saved_fmt)}</span><br>
                            🗺️ <b>{region_s}</b> — 📍 <b>{comuna_s}</b> (Tier {tier_val}/5)<br>
                            🛣️ {km_str} &nbsp;|&nbsp; ⚙️ {trans_s} ({comb_s}) &nbsp;|&nbsp; 🧬 <code>{gen_s}</code><br>
                            ✨ {nlp_s}
                            {drop_html}
                        </div>
                        <div class="price-grid">
                            <div>
                                <div class="price-label">Precio Publicado</div>
                                <div class="price-val-pub">{precio_pub_str}</div>
                            </div>
                            <div>
                                <div class="price-label">Precio Justo IA</div>
                                <div class="price-val-ia">{precio_ia_str}</div>
                            </div>
                        </div>
                        <div class="profit-banner">
                            <span>🟢 {margen_str} ({dcto_str})</span>
                            <span>{roi_str}</span>
                        </div>
                        <a href="{url_s}" target="_blank" rel="noopener noreferrer" class="btn-portal">
                            🔗 Abrir Aviso en {portal_name_s} ({id_s}) ↗
                        </a>
                    </div>
                    """
                )

                card_label = f"{row['marca']} {row['modelo']} ({int(row['anio'])})"
                with col:
                    st.markdown(fav_card_html, unsafe_allow_html=True)
                    note_col, save_col = st.columns([2.6, 1.0])
                    with note_col:
                        note_val = st.text_input(
                            "Nota personal",
                            value=str(row.get("nota_personal") or ""),
                            placeholder="Ej: Llamar hoy, ofrecer $10.5M...",
                            label_visibility="collapsed",
                            key=f"fav_note_input_{raw_id}",
                        )
                    with save_col:
                        if st.button("💾 Nota", key=f"fav_save_note_{raw_id}", width="stretch"):
                            DatabaseManager(db_path=config.DB_PATH).update_favorite_note(raw_id, note_val)
                            load_dashboard_datasets.clear()
                            st.rerun(scope="app")

                    fa1, fa2, fa3 = st.columns(3)
                    with fa1:
                        _render_quick_edit_popover(row, key_prefix="fav")
                    with fa2:
                        if st.button(
                            "🗑️ Quitar",
                            key=f"fav_remove_{raw_id}",
                            help="Quita este vehículo de tu lista de Favoritos.",
                            width="stretch",
                        ):
                            _toggle_favorite_and_refresh(raw_id)
                            st.rerun(scope="app")
                    with fa3:
                        if estado_val == "activo":
                            if st.button(
                                "🚫 No disp.",
                                key=f"fav_sold_{raw_id}",
                                help="Marca el aviso como vendido / no disponible.",
                                width="stretch",
                            ):
                                _update_listing_status_and_refresh(raw_id, "vendido", card_label)
                                st.rerun(scope="app")
                        else:
                            if st.button(
                                "↩️ Reactivar",
                                key=f"fav_restore_{raw_id}",
                                help="Vuelve a marcar el aviso como activo.",
                                width="stretch",
                            ):
                                _update_listing_status_and_refresh(raw_id, "activo", card_label)
                                st.rerun(scope="app")


def render_tab_favorites(data_bundle: Dict[str, Any]) -> None:
    """Ventana dedicada de Favoritos persistentes en SQLite."""
    _render_favorites_fragment(
        data_bundle["full_df"],
        data_bundle.get("favorites_map", {}),
    )


# ==============================================================================
# 7. PESTAÑA 2: 📈 CURVAS DE MERCADO Y VISUALIZACIONES COMPLEJAS (2D, 3D, BOXPLOT, HEATMAP, TREEMAP)
# ==============================================================================
@st.cache_data(show_spinner=False)
def _build_treemap_figure(tree_df: pd.DataFrame, plotly_template: str) -> go.Figure:
    """Construye y cachea el gráfico Treemap global del mercado automotriz."""
    fig_tree = px.treemap(
        tree_df,
        path=[px.Constant("Mercado Chileno"), "marca", "modelo", "generacion"],
        values="unidades",
        color="precio_medio",
        color_continuous_scale="Turbo",
        hover_data={"precio_medio": ":$,.0f", "descuento_max_pct": True},
        title="🗺️ Mapa Jerárquico del Parque Automotriz Analizado (Tamaño = Volumen | Color = Precio Medio)",
        template=plotly_template,
    )
    fig_tree.update_layout(height=460, margin=dict(l=10, r=10, t=45, b=10))
    return fig_tree


@st.fragment
def _render_curves_fragment(
    clean_df: pd.DataFrame,
    active_df: pd.DataFrame,
    model_meta: Dict[str, Any],
    tree_df: pd.DataFrame,
) -> None:
    """Fragmento aislado para curvas, gráficos 3D, BoxPlots, Heatmaps y Treemap."""
    all_segments = config.get_all_target_segments()
    target_marcas = {m for m, _ in all_segments}
    db_marcas = set(clean_df["marca"].unique().tolist()) if not clean_df.empty else set()
    marcas_list = sorted(target_marcas.union(db_marcas))

    preferred_marca = st.session_state.get("focused_marca") or "MAZDA"
    default_marca_idx = marcas_list.index(preferred_marca) if preferred_marca in marcas_list else 0

    c_sel1, c_sel2, c_sel3, c_sel4 = st.columns([1.3, 1.3, 1.3, 1.3])
    with c_sel1:
        selected_marca = st.selectbox(
            "1️⃣ Marca:",
            options=marcas_list,
            index=default_marca_idx,
            key="curve_tab_marca",
        )

    target_mods_for_marca = {str(mod) for m, mod in all_segments if m == selected_marca}
    db_mods_for_marca = set(
        clean_df.loc[clean_df["marca"] == selected_marca, "modelo"].astype(str).unique().tolist()
    )
    modelos_list = ["TODOS LOS MODELOS DE LA MARCA"] + sorted(
        target_mods_for_marca.union(db_mods_for_marca)
    )

    preferred_modelo = normalize_model_input(st.session_state.get("focused_modelo")) or "3"
    if "curve_tab_modelo" in st.session_state and st.session_state["curve_tab_modelo"] not in modelos_list:
        fallback_mod = modelos_list[1] if len(modelos_list) > 1 else modelos_list[0]
        if fallback_mod != st.session_state["curve_tab_modelo"]:
            st.session_state["curve_tab_modelo"] = fallback_mod

    default_mod_idx = (
        modelos_list.index(preferred_modelo)
        if preferred_modelo in modelos_list
        else (1 if len(modelos_list) > 1 else 0)
    )

    with c_sel2:
        selected_modelo = st.selectbox(
            "2️⃣ Modelo:",
            options=modelos_list,
            key="curve_tab_modelo",
            **({} if "curve_tab_modelo" in st.session_state else {"index": default_mod_idx}),
        )

    with c_sel3:
        x_axis_mode = st.selectbox(
            "3️⃣ Variable del Eje X (Curva 2D):",
            options=[
                "Año de Fabricación",
                "Kilometraje (km)",
                "Intensidad de Uso (km/año)",
                "Antigüedad (años)",
            ],
            index=0,
            key="curve_tab_xaxis",
        )

    with c_sel4:
        color_grouping = st.selectbox(
            "4️⃣ Colorear Puntos Normales por:",
            options=[
                "Categoría de Arbitraje",
                "Transmisión (Manual vs Automática)",
                "Portal de Origen",
                "Generación del Modelo",
                "Combustible",
            ],
            index=0,
            key="curve_tab_color",
        )

    # Filtros adicionales para refinar los datos de los gráficos
    min_y_avail = int(min(1990, clean_df["anio"].min())) if not clean_df.empty else 1990
    with st.expander(
        "🎚️ **Filtros Específicos para los Gráficos de este Modelo (Transmisión, Portal, Año, Km)**",
        expanded=False,
    ):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            g_trans = st.multiselect(
                "Transmisión:",
                options=["Automática", "Manual"],
                default=["Automática", "Manual"],
                key="curve_flt_trans",
            )
        with fc2:
            g_portales = st.multiselect(
                "Portales:",
                options=["MercadoLibre", "Autocosmos", "Yapo.cl", "Chileautos"],
                default=["MercadoLibre", "Autocosmos", "Yapo.cl", "Chileautos"],
                key="curve_flt_portales",
            )
        with fc3:
            _sanitize_slider_state("curve_flt_years", min_y_avail, config.MAX_VALID_YEAR)
            g_year_rng = st.slider(
                "Rango de Años:",
                min_value=min_y_avail,
                max_value=config.MAX_VALID_YEAR,
                value=(max(min_y_avail, 2005), config.MAX_VALID_YEAR),
                key="curve_flt_years",
            )
        with fc4:
            g_max_km = st.slider(
                "Kilometraje Máximo:",
                min_value=20_000,
                max_value=config.MAX_VALID_KM,
                value=min(400_000, config.MAX_VALID_KM),
                step=10_000,
                key="curve_flt_km",
            )

    # Filtrar dataset limpio y dataset activo según selección
    if selected_modelo == "TODOS LOS MODELOS DE LA MARCA":
        sub_clean_base = clean_df.loc[clean_df["marca"] == selected_marca]
        sub_active_base = active_df.loc[active_df["marca"] == selected_marca]
        model_title_label = f"{selected_marca} (Todos los Modelos)"
    else:
        sub_clean_base = clean_df.loc[
            (clean_df["marca"] == selected_marca)
            & (clean_df["modelo"].astype(str) == str(selected_modelo))
        ]
        sub_active_base = active_df.loc[
            (active_df["marca"] == selected_marca)
            & (active_df["modelo"].astype(str) == str(selected_modelo))
        ]
        model_title_label = f"{selected_marca} {selected_modelo}"

    if not sub_clean_base.empty:
        sub_clean = sub_clean_base.loc[
            sub_clean_base["transmision"].isin(g_trans)
            & sub_clean_base["portal"].isin(g_portales)
            & (sub_clean_base["anio"] >= g_year_rng[0])
            & (sub_clean_base["anio"] <= g_year_rng[1])
            & (sub_clean_base["kilometraje"] <= g_max_km)
        ].copy()
    else:
        sub_clean = pd.DataFrame()

    if not sub_active_base.empty:
        sub_active = sub_active_base.loc[
            sub_active_base["transmision"].isin(g_trans)
            & sub_active_base["portal"].isin(g_portales)
            & (sub_active_base["anio"] >= g_year_rng[0])
            & (sub_active_base["anio"] <= g_year_rng[1])
            & (sub_active_base["kilometraje"] <= g_max_km)
        ].copy()
    else:
        sub_active = pd.DataFrame()

    if sub_clean.empty and sub_active.empty:
        mod_for_scan = None if selected_modelo == "TODOS LOS MODELOS DE LA MARCA" else str(selected_modelo)
        st.info(
            f"📡 El segmento **{model_title_label}** no tiene anuncios con estos filtros en la base local "
            "(o aún no ha sido descargado desde los portales web)."
        )
        if st.button(
            f"🚀 Escanear Portales en Vivo y Construir Curva para {model_title_label} Ahora",
            key="curve_empty_fetch_btn",
            type="primary",
        ):
            with st.spinner(
                f"Descargando avisos reales de {model_title_label} desde MercadoLibre, Autocosmos, Yapo y Chileautos..."
            ):
                ok, msg = execute_backend_cycle(
                    force_train=False,
                    marca_filter=selected_marca,
                    modelo_filter=mod_for_scan,
                    pages_per_portal=2,
                    max_telegram_alerts=5,
                    deep=True,
                )
            if ok:
                st.session_state["last_scan_message"] = msg
                st.rerun(scope="app")
            else:
                st.error(msg)
        return

    # Navegación lazy entre sub-vistas gráficas (solo renderiza el gráfico seleccionado)
    subview_kwargs = (
        {"default": CURVE_SUBVIEWS[0]}
        if "active_curve_subtab" not in st.session_state
        else {}
    )
    active_subview = st.segmented_control(
        "Visualización:",
        options=CURVE_SUBVIEWS,
        selection_mode="single",
        required=True,
        width="stretch",
        label_visibility="collapsed",
        key="active_curve_subtab",
        **subview_kwargs,
    )
    if not active_subview:
        active_subview = CURVE_SUBVIEWS[0]

    x_col_map = {
        "Año de Fabricación": ("anio", "Año de Fabricación"),
        "Kilometraje (km)": ("kilometraje", "Kilometraje (km)"),
        "Intensidad de Uso (km/año)": ("km_por_anio", "Kilometraje por Año (km/año)"),
        "Antigüedad (años)": ("antiguedad", "Antigüedad del Vehículo (años)"),
    }
    x_col, x_label = x_col_map.get(x_axis_mode, ("anio", "Año de Fabricación"))

    normal_pts = (
        sub_clean.loc[sub_clean["categoria_arbitraje"] == "PRECIO DE MERCADO"]
        if not sub_clean.empty
        else pd.DataFrame()
    )
    opp_pts = (
        sub_active.loc[sub_active["categoria_arbitraje"] == "OPORTUNIDAD"]
        if not sub_active.empty
        else pd.DataFrame()
    )
    scam_pts = (
        sub_active.loc[sub_active["categoria_arbitraje"] == "POSIBLE ESTAFA / ERROR DE TIPEO"]
        if not sub_active.empty
        else pd.DataFrame()
    )

    # --------------------------------------------------------------------------
    # SUB-VISTA 1: CURVA 2D MULTI-CAPA + BANDA DE PRECIO JUSTO IA
    # --------------------------------------------------------------------------
    if active_subview == CURVE_SUBVIEWS[0]:
        fig = go.Figure()

        if not normal_pts.empty:
            if color_grouping == "Categoría de Arbitraje":
                fig.add_trace(
                    go.Scatter(
                        x=normal_pts[x_col],
                        y=normal_pts["precio"],
                        mode="markers",
                        name="Mercado Normal (Precio Regular)",
                        marker=dict(
                            size=9,
                            color="#38bdf8",
                            opacity=0.76,
                            line=dict(width=1, color="#0284c7"),
                        ),
                        customdata=np.stack(
                            [
                                normal_pts["id_aviso"],
                                normal_pts["portal_badge"],
                                normal_pts["anio"],
                                normal_pts["kilometraje"],
                                normal_pts["comuna"],
                                normal_pts["precio_justo_ia"],
                                normal_pts["generacion"],
                                normal_pts["transmision"],
                            ],
                            axis=-1,
                        ),
                        hovertemplate=(
                            f"<b>{model_title_label} (%{{customdata[2]}})</b><br>"
                            "Portal: %{customdata[1]} (%{customdata[0]})<br>"
                            "Generación: %{customdata[6]} | %{customdata[7]}<br>"
                            "Kilometraje: %{customdata[3]:,.0f} km | Comuna: %{customdata[4]}<br>"
                            "Precio Publicado: <b>$ %{y:,.0f}</b><br>"
                            "Precio Justo IA: $ %{customdata[5]:,.0f}<extra></extra>"
                        ),
                    )
                )
            else:
                group_col_map = {
                    "Transmisión (Manual vs Automática)": "transmision",
                    "Portal de Origen": "portal",
                    "Generación del Modelo": "generacion",
                    "Combustible": "combustible",
                }
                g_col = group_col_map.get(color_grouping, "transmision")
                palette = ["#38bdf8", "#f472b6", "#facc15", "#60a5fa", "#c084fc", "#fb923c"]
                for idx_g, (g_val, g_df) in enumerate(normal_pts.groupby(g_col)):
                    fig.add_trace(
                        go.Scatter(
                            x=g_df[x_col],
                            y=g_df["precio"],
                            mode="markers",
                            name=f"{g_val}",
                            marker=dict(
                                size=9,
                                color=palette[idx_g % len(palette)],
                                opacity=0.8,
                            ),
                            customdata=np.stack(
                                [
                                    g_df["id_aviso"],
                                    g_df["portal_badge"],
                                    g_df["anio"],
                                    g_df["kilometraje"],
                                    g_df["comuna"],
                                    g_df["precio_justo_ia"],
                                ],
                                axis=-1,
                            ),
                            hovertemplate=(
                                f"<b>{model_title_label} - {g_val} (%{{customdata[2]}})</b><br>"
                                "Portal: %{customdata[1]} (%{customdata[0]})<br>"
                                "Kilometraje: %{customdata[3]:,.0f} km | %{customdata[4]}<br>"
                                "Precio Publicado: <b>$ %{y:,.0f}</b><br>"
                                "Precio Justo IA: $ %{customdata[5]:,.0f}<extra></extra>"
                            ),
                        )
                    )

        curve_base = sub_clean if not sub_clean.empty else sub_active
        if len(curve_base) >= 2:
            x_line = None
            y_line = None
            if x_col == "anio":
                ia_curve = (
                    curve_base.groupby("anio", as_index=False)
                    .agg(mediana_ia=("precio_justo_ia", "median"))
                    .sort_values("anio")
                )
                if len(ia_curve) >= 2:
                    x_line = ia_curve["anio"].values
                    y_line = ia_curve["mediana_ia"].values
            else:
                x_vals = curve_base[x_col].astype(float).values
                y_vals = curve_base["precio_justo_ia"].astype(float).values
                unique_x_cnt = len(np.unique(x_vals))
                if unique_x_cnt >= 2:
                    x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 60)
                    deg = 2 if unique_x_cnt >= 4 else 1
                    try:
                        coeffs = np.polyfit(x_vals, np.log(np.maximum(y_vals, 500_000)), deg=deg)
                        y_line = np.exp(np.polyval(coeffs, x_line))
                    except Exception:
                        try:
                            coeffs = np.polyfit(x_vals, y_vals, deg=1)
                            y_line = np.polyval(coeffs, x_line)
                        except Exception:
                            x_line, y_line = None, None
                else:
                    st.caption("ℹ️ Todos los vehículos filtrados comparten el mismo valor en el eje X; se requiere ≥ 2 valores distintos para trazar curva continua.")

            if x_line is not None and y_line is not None:
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_line, x_line[::-1]]),
                        y=np.concatenate([y_line * 1.08, (y_line * 0.92)[::-1]]),
                        fill="toself",
                        fillcolor="rgba(168, 85, 247, 0.12)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        name="Banda Normal IA (±8%)",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines+markers" if x_col == "anio" else "lines",
                        name="🤖 Curva Precio Justo IA",
                        line=dict(color="#a855f7", width=3.5, dash="solid"),
                        marker=dict(size=6, color="#c084fc"),
                        hovertemplate="Precio Justo IA: <b>$ %{y:,.0f}</b><extra></extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line * (1.0 - config.OPPORTUNITY_DISCOUNT_THRESHOLD),
                        mode="lines",
                        name="Umbral Oportunidad (-15% bajo IA)",
                        line=dict(color="#10b981", width=2.2, dash="dot"),
                        hovertemplate="Umbral -15% IA: <b>$ %{y:,.0f}</b><extra></extra>",
                    )
                )

        if not opp_pts.empty:
            fig.add_trace(
                go.Scatter(
                    x=opp_pts[x_col],
                    y=opp_pts["precio"],
                    mode="markers",
                    name="Halo Oportunidad",
                    showlegend=False,
                    hoverinfo="skip",
                    marker=dict(
                        size=24,
                        color="rgba(16, 185, 129, 0.25)",
                        line=dict(width=1.5, color="rgba(16, 185, 129, 0.7)"),
                    ),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=opp_pts[x_col],
                    y=opp_pts["precio"],
                    mode="markers+text",
                    name="🟢 OPORTUNIDAD REAL (15%–55% Dcto)",
                    text=[f"-{r['descuento_pct']*100:.0f}%" for _, r in opp_pts.iterrows()],
                    textposition="bottom center",
                    textfont=dict(color="#34d399", size=11, family="Arial Black"),
                    marker=dict(
                        size=13,
                        color="#10b981",
                        symbol="diamond",
                        line=dict(width=2, color="#ffffff"),
                    ),
                    customdata=np.stack(
                        [
                            opp_pts["id_aviso"],
                            opp_pts["portal_badge"],
                            opp_pts["anio"],
                            opp_pts["kilometraje"],
                            opp_pts["comuna"],
                            opp_pts["precio_justo_ia"],
                            opp_pts["margen_ganancia_clp"],
                            opp_pts["descuento_pct"] * 100.0,
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        f"<b>🟢 OPORTUNIDAD: {model_title_label} (%{{customdata[2]}})</b><br>"
                        "Portal: %{customdata[1]} (%{customdata[0]})<br>"
                        "Kilometraje: %{customdata[3]:,.0f} km | Comuna: %{customdata[4]}<br>"
                        "Precio Publicado: <b>$ %{y:,.0f}</b><br>"
                        "Precio Justo IA: <b>$ %{customdata[5]:,.0f}</b><br>"
                        "Margen Ganancia: <b>+ $ %{customdata[6]:,.0f} (-%{customdata[7]:.1f}%)</b><extra></extra>"
                    ),
                )
            )

        if not scam_pts.empty:
            fig.add_trace(
                go.Scatter(
                    x=scam_pts[x_col],
                    y=scam_pts["precio"],
                    mode="markers+text",
                    name="⚠️ POSIBLE ESTAFA / ERROR (>=55% Dcto)",
                    text=[f"⚠️ -{r['descuento_pct']*100:.0f}%" for _, r in scam_pts.iterrows()],
                    textposition="bottom center",
                    textfont=dict(color="#fbbf24", size=11),
                    marker=dict(
                        size=13,
                        color="#ef4444",
                        symbol="triangle-up",
                        line=dict(width=2, color="#fbbf24"),
                    ),
                    customdata=np.stack(
                        [
                            scam_pts["id_aviso"],
                            scam_pts["portal_badge"],
                            scam_pts["anio"],
                            scam_pts["kilometraje"],
                            scam_pts["comuna"],
                            scam_pts["precio_justo_ia"],
                            scam_pts["descuento_pct"] * 100.0,
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        f"<b>⚠️ SOSPECHOSO: {model_title_label} (%{{customdata[2]}})</b><br>"
                        "Portal: %{customdata[1]} (%{customdata[0]})<br>"
                        "Precio Publicado: <b>$ %{y:,.0f}</b><br>"
                        "Precio Justo IA: $ %{customdata[5]:,.0f} (-%{customdata[6]:.1f}%)<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            title=dict(
                text=f"Curva de Depreciación Multi-Capa: {model_title_label} ({x_label} vs CLP)",
                font=dict(size=17),
            ),
            xaxis_title=x_label,
            yaxis_title="Precio en Pesos Chilenos (CLP)",
            yaxis=dict(tickprefix="$ ", separatethousands=True),
            xaxis=dict(separatethousands=True if x_col in ("kilometraje", "km_por_anio") else False),
            template=get_plotly_template(),
            height=530,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=30, t=70, b=40),
        )
        st.plotly_chart(fig, width="stretch")

    # --------------------------------------------------------------------------
    # SUB-VISTA 2: SUPERFICIE / DISPERSIÓN TRIDIMENSIONAL 3D (AÑO × KM × PRECIO)
    # --------------------------------------------------------------------------
    elif active_subview == CURVE_SUBVIEWS[1]:
        st.caption(
            "Rotación interactiva en 3 dimensiones (**Eje X = Año**, **Eje Y = Kilometraje**, **Eje Z = Precio CLP**). "
            "Los diamantes verdes debajo de la nube principal corresponden a oportunidades de arbitraje."
        )
        plot3d_df = sub_active if not sub_active.empty else sub_clean
        if plot3d_df.empty:
            st.info("No hay suficientes vehículos para construir la nube 3D con los filtros actuales.")
        else:
            fig_3d = px.scatter_3d(
                plot3d_df,
                x="anio",
                y="kilometraje",
                z="precio",
                color="categoria_arbitraje",
                color_discrete_map={
                    "OPORTUNIDAD": "#10b981",
                    "PRECIO DE MERCADO": "#38bdf8",
                    "POSIBLE ESTAFA / ERROR DE TIPEO": "#ef4444",
                },
                symbol="transmision",
                hover_name="id_aviso",
                hover_data={
                    "marca": True,
                    "modelo": True,
                    "generacion": True,
                    "precio": ":$,.0f",
                    "precio_justo_ia": ":$,.0f",
                    "margen_ganancia_clp": ":$,.0f",
                    "comuna": True,
                    "portal": True,
                },
                title=f"🧊 Espacio Tridimensional de Precios: {model_title_label} (Año × Kilometraje × Precio)",
                template=get_plotly_template(),
            )
            fig_3d.update_layout(
                height=580,
                scene=dict(
                    xaxis_title="Año",
                    yaxis_title="Kilometraje (km)",
                    zaxis_title="Precio (CLP)",
                ),
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig_3d, width="stretch")

    # --------------------------------------------------------------------------
    # SUB-VISTA 3: BOXPLOT POR GENERACIÓN Y COMPARATIVA ENTRE PORTALES
    # --------------------------------------------------------------------------
    elif active_subview == CURVE_SUBVIEWS[2]:
        box_df = sub_clean if not sub_clean.empty else sub_active
        if box_df.empty:
            st.info("No hay datos disponibles para generar los diagramas de dispersión.")
        else:
            pts_mode = "outliers" if len(box_df) > 250 else "all"
            b1, b2 = st.columns(2)
            with b1:
                fig_box_gen = px.box(
                    box_df,
                    x="generacion",
                    y="precio",
                    color="generacion",
                    points=pts_mode,
                    hover_name="id_aviso",
                    hover_data=["anio", "kilometraje", "comuna", "portal"],
                    title=f"📦 Dispersión de Precios por Generación ({model_title_label})",
                    labels={"generacion": "Generación", "precio": "Precio Publicado (CLP)"},
                    template=get_plotly_template(),
                )
                fig_box_gen.update_layout(
                    height=460,
                    showlegend=False,
                    yaxis=dict(tickprefix="$ ", separatethousands=True),
                )
                st.plotly_chart(fig_box_gen, width="stretch")

            with b2:
                fig_box_portal = px.violin(
                    box_df,
                    x="portal",
                    y="precio",
                    color="transmision",
                    box=True,
                    points=pts_mode,
                    hover_name="id_aviso",
                    title=f"🌐 Comparativa de Precios por Portal y Transmisión ({model_title_label})",
                    labels={"portal": "Portal Web", "precio": "Precio Publicado (CLP)"},
                    template=get_plotly_template(),
                )
                fig_box_portal.update_layout(
                    height=460,
                    yaxis=dict(tickprefix="$ ", separatethousands=True),
                )
                st.plotly_chart(fig_box_portal, width="stretch")

    # --------------------------------------------------------------------------
    # SUB-VISTA 4: HEATMAP DE PRECIOS Y DESCUENTOS POR AÑO Y TRANSMISIÓN
    # --------------------------------------------------------------------------
    elif active_subview == CURVE_SUBVIEWS[3]:
        hm_base = sub_clean if not sub_clean.empty else sub_active
        hm1, hm2 = st.columns(2)
        with hm1:
            if hm_base.empty:
                st.info("Sin registros suficientes para construir el Mapa de Calor de precios.")
            else:
                pivot_price = hm_base.pivot_table(
                    index="transmision",
                    columns="anio",
                    values="precio",
                    aggfunc="median",
                )
                if pivot_price.empty:
                    st.info("Sin matriz válida de Año × Transmisión.")
                else:
                    fig_hm_price = px.imshow(
                        pivot_price,
                        text_auto=".2s",
                        aspect="auto",
                        color_continuous_scale="Tealgrn",
                        title=f"🔥 Mapa de Calor: Precio Mediano (CLP) por Año y Transmisión ({model_title_label})",
                        labels=dict(x="Año", y="Transmisión", color="Precio Mediano"),
                        template=get_plotly_template(),
                    )
                    fig_hm_price.update_layout(height=420)
                    st.plotly_chart(fig_hm_price, width="stretch")

        with hm2:
            if sub_active.empty:
                st.info("Sin avisos activos para comparar márgenes entre portales y generaciones.")
            else:
                pivot_disc = sub_active.pivot_table(
                    index="generacion",
                    columns="portal",
                    values="margen_ganancia_clp",
                    aggfunc="max",
                )
                if pivot_disc.empty:
                    st.info("Sin matriz válida de Generación × Portal.")
                else:
                    fig_hm_disc = px.imshow(
                        pivot_disc,
                        text_auto=".2s",
                        aspect="auto",
                        color_continuous_scale="RdYlGn",
                        title=f"💰 Mayor Margen Detectado (CLP) por Generación y Portal ({model_title_label})",
                        labels=dict(x="Portal", y="Generación", color="Máx Margen CLP"),
                        template=get_plotly_template(),
                    )
                    fig_hm_disc.update_layout(height=420)
                    st.plotly_chart(fig_hm_disc, width="stretch")

    # --------------------------------------------------------------------------
    # SUB-VISTA 5: TREEMAP GLOBAL DEL MERCADO E IMPORTANCIA DE VARIABLES IA
    # --------------------------------------------------------------------------
    else:
        tm_col1, tm_col2 = st.columns([1.35, 1.0])
        with tm_col1:
            if tree_df.empty:
                st.info("Sin datos agregados para generar el Mapa Jerárquico.")
            else:
                fig_tree = _build_treemap_figure(tree_df, get_plotly_template())
                st.plotly_chart(fig_tree, width="stretch")

        with tm_col2:
            top_features = model_meta.get("top_10_features", {})
            if top_features:
                feat_names = list(top_features.keys())[::-1]
                feat_vals = list(top_features.values())[::-1]
                fig_imp = go.Figure(
                    go.Bar(
                        x=feat_vals,
                        y=feat_names,
                        orientation="h",
                        marker=dict(color="#10b981"),
                    )
                )
                fig_imp.update_layout(
                    title="🧠 Top 10 Variables del Modelo LightGBM",
                    xaxis_title="Importancia (Splits en Árboles)",
                    yaxis_title="Variable",
                    template=get_plotly_template(),
                    height=460,
                    margin=dict(l=20, r=20, t=45, b=20),
                )
                st.plotly_chart(fig_imp, width="stretch")

    # KPIs inferiores del modelo seleccionado
    kpi_base = sub_clean if not sub_clean.empty else sub_active
    m_c1, m_c2, m_c3, m_c4, m_c5 = st.columns(5)
    with m_c1:
        st.metric("Muestras en Vista", len(kpi_base))
    with m_c2:
        st.metric(
            "Precio Mediano Mercado",
            format_clp(kpi_base["precio"].median() if not kpi_base.empty else 0),
        )
    with m_c3:
        st.metric(
            "Precio Justo IA Medio",
            format_clp(kpi_base["precio_justo_ia"].mean() if not kpi_base.empty else 0),
        )
    with m_c4:
        st.metric("🟢 Oportunidades en Modelo", len(opp_pts))
    with m_c5:
        st.metric("⚠️ Alertas Anti-Estafa", len(scam_pts))


def render_tab_market_curves(data_bundle: Dict[str, Any]) -> None:
    """
    Suite Gráfica Avanzada:
    Renderiza en un fragmento aislado con sub-vistas lazy las curvas 2D, 3D,
    BoxPlots/Violines, Heatmaps y Treemap + Feature Importance.
    """
    clean_df = data_bundle["clean_df"]
    active_df = data_bundle["active_featured"]
    model_meta = data_bundle["model_meta"]
    tree_df = data_bundle.get("tree_df", pd.DataFrame())

    if clean_df.empty:
        st.info("No hay registros en 'data/autos_clean.csv' para graficar curvas.")
        return

    st.markdown("### 📈 Laboratorio Gráfico de Curvas de Depreciación y Analítica Avanzada")
    _render_curves_fragment(clean_df, active_df, model_meta, tree_df)


# ==============================================================================
# 8. PESTAÑA 3: 🧮 TASADOR MANUAL IA EN VIVO
# ==============================================================================
@st.fragment
def _render_appraiser_fragment(clean_df: pd.DataFrame, model_mtime: float) -> None:
    """Fragmento aislado del Tasador IA: recalcula predicción y comparables en <50ms."""
    predictor, _ = load_cached_predictor(model_mtime)

    all_segments = config.get_all_target_segments()
    target_marcas = sorted({m for m, _ in all_segments})
    db_marcas = sorted(clean_df["marca"].unique().tolist()) if not clean_df.empty else []
    all_marcas = sorted(set(target_marcas + db_marcas))
    comunas_list = sorted(
        {c.title() for c in config.COMUNA_WEALTH_TIER.keys()}.union(
            {"Santiago", "Providencia", "Las Condes"}
        )
    )

    col_form, col_result = st.columns([1.45, 1.15])

    with col_form:
        f_c1, f_c2 = st.columns(2)
        with f_c1:
            marca_options = all_marcas + ["➕ OTRA MARCA (Escribir manualmente)"]
            t_marca_sel = st.selectbox(
                "Marca:",
                options=marca_options,
                index=all_marcas.index("MAZDA") if "MAZDA" in all_marcas else 0,
                key="tasador_marca",
            )
            if t_marca_sel == "➕ OTRA MARCA (Escribir manualmente)":
                t_marca = (
                    st.text_input("Escribe la Marca:", value="AUDI", key="tasador_custom_marca")
                    .strip()
                    .upper()
                ) or "AUDI"
            else:
                t_marca = t_marca_sel
        with f_c2:
            avail_models = sorted(
                {str(mod) for m, mod in all_segments if m == t_marca}.union(
                    set(
                        clean_df.loc[clean_df["marca"] == t_marca, "modelo"].astype(str).unique()
                        if not clean_df.empty
                        else []
                    )
                )
            )
            mod_opts_tas = avail_models + ["➕ OTRO MODELO (Escribir manualmente)"]
            if (
                "tasador_modelo" in st.session_state
                and st.session_state["tasador_modelo"] not in mod_opts_tas
                and mod_opts_tas[0] != st.session_state["tasador_modelo"]
            ):
                st.session_state["tasador_modelo"] = mod_opts_tas[0]

            t_modelo_sel = st.selectbox(
                "Modelo:",
                options=mod_opts_tas,
                key="tasador_modelo",
                **({} if "tasador_modelo" in st.session_state else {"index": 0}),
            )
            if t_modelo_sel == "➕ OTRO MODELO (Escribir manualmente)":
                t_modelo_raw = st.text_input(
                    "Escribe el Modelo:", value="A4", key="tasador_custom_modelo"
                )
                t_modelo_norm = normalize_model_input(t_modelo_raw) or "GENERICO"
            else:
                t_modelo_norm = normalize_model_input(t_modelo_sel) or str(t_modelo_sel)

        f_c3, f_c4, f_c5 = st.columns(3)
        with f_c3:
            t_anio = st.number_input(
                "Año de Fabricación:",
                min_value=config.MIN_VALID_YEAR,
                max_value=config.MAX_VALID_YEAR,
                value=2020,
                step=1,
                key="tasador_anio",
            )
        with f_c4:
            t_km = st.number_input(
                "Kilometraje (km):",
                min_value=0,
                max_value=config.MAX_VALID_KM,
                value=65_000,
                step=5_000,
                key="tasador_km",
            )
        with f_c5:
            t_precio_oferta = st.number_input(
                "Precio Publicado / Oferta (CLP):",
                min_value=config.MIN_VALID_PRICE_CLP,
                max_value=config.MAX_VALID_PRICE_CLP,
                value=10_500_000,
                step=250_000,
                key="tasador_precio_oferta",
            )

        f_c6, f_c7, f_c8 = st.columns(3)
        with f_c6:
            t_trans = st.selectbox("Transmisión:", ["Automática", "Manual"], key="tasador_trans")
        with f_c7:
            t_comb = st.selectbox(
                "Combustible:", ["Bencina", "Diésel", "Híbrido", "Eléctrico"], key="tasador_comb"
            )
        with f_c8:
            t_comuna = st.selectbox(
                "Comuna:",
                options=comunas_list,
                index=comunas_list.index("Providencia") if "Providencia" in comunas_list else 0,
                key="tasador_comuna",
            )

        st.markdown("#### ✨ Equipamiento y Atributos Detectables por NLP")
        chk1, chk2, chk3 = st.columns(3)
        with chk1:
            c_unico = st.checkbox("Único dueño", value=True, key="tasador_chk_unico")
            c_mant = st.checkbox("Mantenciones al día", value=True, key="tasador_chk_mant")
            c_cuero = st.checkbox("Asientos de cuero", value=False, key="tasador_chk_cuero")
        with chk2:
            c_laminas = st.checkbox("Láminas de seguridad", value=False, key="tasador_chk_laminas")
            c_neum = st.checkbox("Neumáticos nuevos", value=False, key="tasador_chk_neum")
            c_sunroof = st.checkbox("Sunroof / Techo panorámico", value=False, key="tasador_chk_sunroof")
        with chk3:
            c_full = st.checkbox("Full equipo / Top de línea", value=True, key="tasador_chk_full")
            c_inmac = st.checkbox("Inmaculado / Impecable", value=True, key="tasador_chk_inmac")
            c_apuro = st.checkbox("Vende por apuro / Conversable", value=False, key="tasador_chk_apuro")

    nlp_phrases: List[str] = []
    if c_unico:
        nlp_phrases.append("único dueño")
    if c_mant:
        nlp_phrases.append("mantenciones al día")
    if c_cuero:
        nlp_phrases.append("asientos de cuero")
    if c_laminas:
        nlp_phrases.append("láminas de seguridad")
    if c_neum:
        nlp_phrases.append("neumáticos nuevos")
    if c_sunroof:
        nlp_phrases.append("techo panorámico sunroof")
    if c_full:
        nlp_phrases.append("full equipo top de línea")
    if c_inmac:
        nlp_phrases.append("impecable como nuevo")
    if c_apuro:
        nlp_phrases.append("vende por apuro conversable")

    synthetic_title = f"{t_marca} {t_modelo_norm} {t_anio}"
    synthetic_desc = ", ".join(nlp_phrases) if nlp_phrases else "vehículo estándar"

    df_single = pd.DataFrame(
        [
            {
                "marca": t_marca,
                "modelo": str(t_modelo_norm),
                "anio": int(t_anio),
                "kilometraje": int(t_km),
                "combustible": t_comb,
                "transmision": t_trans,
                "comuna": t_comuna,
                "titulo": synthetic_title,
                "descripcion": synthetic_desc,
                "precio": int(t_precio_oferta),
            }
        ]
    )

    precio_justo_ia = float(predictor.predict_fair_price(df_single)[0])
    margen_clp = precio_justo_ia - float(t_precio_oferta)
    descuento_pct = (margen_clp / precio_justo_ia) if precio_justo_ia > 0 else 0.0
    precio_objetivo_arbitraje = precio_justo_ia * (1.0 - config.OPPORTUNITY_DISCOUNT_THRESHOLD)
    gen_code = CarDataPipeline.resolve_generation(t_marca, str(t_modelo_norm), int(t_anio))
    comuna_tier = CarDataPipeline.resolve_comuna_tier(t_comuna)

    with col_result:
        st.markdown("#### 🤖 Resultado de Valoración Instantánea")
        km_fmt = f"{int(t_km):,}".replace(",", ".")
        st.markdown(
            _clean_html(
                f"""
                <div class="opp-card" style="border-left-color: #0284c7;">
                    <div class="car-header">
                        <span class="car-title">{get_vehicle_emoji(t_marca, str(t_modelo_norm))} {html.escape(t_marca)} {html.escape(str(t_modelo_norm))} ({int(t_anio)})</span>
                        <span class="comuna-pill">📍 {html.escape(str(t_comuna))} (Tier {comuna_tier}/5)</span>
                    </div>
                    <div class="car-meta">
                        🧬 Generación detectada: <code>{html.escape(gen_code)}</code><br>
                        🛣️ {km_fmt} km &nbsp;|&nbsp; ⚙️ {html.escape(str(t_trans))} ({html.escape(str(t_comb))})
                    </div>
                    <div class="price-grid">
                        <div>
                            <div class="price-label">🤖 Precio Justo IA (Mercado)</div>
                            <div class="price-val-ia" style="font-size: 1.35rem;">{format_clp(precio_justo_ia)}</div>
                        </div>
                        <div>
                            <div class="price-label">🎯 Precio Máx. Compra (-15%)</div>
                            <div class="price-val-target">{format_clp(precio_objetivo_arbitraje)}</div>
                        </div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        r1, r2 = st.columns(2)
        with r1:
            st.metric(
                "Diferencia vs Precio Oferta",
                format_clp(margen_clp),
                delta=f"{-descuento_pct * 100:.1f}% vs IA",
            )
        with r2:
            st.metric("Precio Oferta Evaluado", format_clp(t_precio_oferta))

        if descuento_pct >= config.SUSPICIOUS_DISCOUNT_THRESHOLD:
            st.error(
                f"🚨 **ALERTA DE POSIBLE ESTAFA O ERROR ({descuento_pct*100:.1f}% bajo IA):** "
                "El precio ofertado es excesivamente bajo. Verificar prenda, clonación o error de tipeo."
            )
        elif descuento_pct >= config.OPPORTUNITY_DISCOUNT_THRESHOLD:
            st.success(
                f"🔥 **¡OPORTUNIDAD DE ARBITRAJE DETECTADA! ({descuento_pct*100:.1f}% bajo IA):** "
                f"Margen bruto estimado de **{format_clp(margen_clp)}** respecto al valor justo de mercado."
            )
        elif descuento_pct >= -0.05:
            st.info(
                f"✅ **PRECIO DE MERCADO REGULAR ({abs(descuento_pct)*100:.1f}% de variación):** "
                f"Para que sea oportunidad de inversión (>=15% descuento), negocia bajo **{format_clp(precio_objetivo_arbitraje)}**."
            )
        else:
            st.warning(
                f"📈 **SOBREPRECIO DETECTADO (+{abs(descuento_pct)*100:.1f}% sobre IA):** "
                f"El vehículo está publicado **{format_clp(abs(margen_clp))}** por encima de su Precio Justo IA."
            )

    # COMPARABLES REALES EN LA BASE DE DATOS
    st.markdown("---")
    st.markdown(f"#### 🔍 Comparables Reales en el Mercado ({t_marca} {t_modelo_norm} · ±2 años)")
    if clean_df.empty:
        st.info("La base limpia está vacía; la valoración se calculó con la curva entrenada del modelo.")
        return

    same_model_df = clean_df.loc[
        (clean_df["marca"] == t_marca) & (clean_df["modelo"] == str(t_modelo_norm))
    ]
    comp_df = same_model_df.loc[same_model_df["anio"].sub(int(t_anio)).abs() <= 2].copy()
    range_label = f"±2 años ({int(t_anio) - 2}–{int(t_anio) + 2})"
    if len(comp_df) < 3 and len(same_model_df) > len(comp_df):
        comp_df = same_model_df.copy()
        range_label = "Todos los años del modelo"

    if comp_df.empty:
        st.info(
            f"Aún no hay unidades limpias de **{t_marca} {t_modelo_norm}** en la BD local; "
            "la tasación utiliza la curva jerárquica de marca/segmento."
        )
        return

    pct_cheaper_than = float((comp_df["precio"] > int(t_precio_oferta)).mean() * 100.0)
    comp_median = float(comp_df["precio"].median())
    comp_min = int(comp_df["precio"].min())
    comp_max = int(comp_df["precio"].max())

    cm1, cm2, cm3, cm4 = st.columns(4)
    with cm1:
        st.metric("Unidades Similares", len(comp_df), delta=range_label)
    with cm2:
        st.metric("Mediana Mercado", format_clp(comp_median))
    with cm3:
        st.metric(
            "Rango Mín – Máx",
            f"${comp_min / 1_000_000:.1f}M – ${comp_max / 1_000_000:.1f}M",
            help=f"Mín: {format_clp(comp_min)} | Máx: {format_clp(comp_max)}",
        )
    with cm4:
        st.metric(
            "Tu Percentil Oferta",
            f"{pct_cheaper_than:.0f}% más caros",
            help="Porcentaje de comparables publicados a un precio superior al evaluado.",
        )

    cg1, cg2 = st.columns([1.05, 1.25])
    with cg1:
        fig_comp = px.scatter(
            comp_df,
            x="kilometraje",
            y="precio",
            color="transmision",
            hover_name="id_aviso",
            hover_data={
                "anio": True,
                "portal": True,
                "comuna": True,
                "precio": ":$,.0f",
                "precio_justo_ia": ":$,.0f",
            },
            title=f"Comparables: {t_marca} {t_modelo_norm} (Km vs Precio)",
            labels={"kilometraje": "Kilometraje (km)", "precio": "Precio (CLP)"},
            template=get_plotly_template(),
        )
        fig_comp.add_hline(
            y=precio_justo_ia,
            line_dash="dot",
            line_color="#a855f7",
            annotation_text=f"IA: {format_clp(precio_justo_ia)}",
            annotation_position="top right",
        )
        fig_comp.add_trace(
            go.Scatter(
                x=[int(t_km)],
                y=[int(t_precio_oferta)],
                mode="markers",
                name="⭐ Tu Oferta",
                marker=dict(
                    symbol="star",
                    size=18,
                    color="#facc15",
                    line=dict(width=1.5, color="#ffffff"),
                ),
                hovertemplate=(
                    f"<b>⭐ Tu Oferta ({int(t_anio)})</b><br>"
                    f"Km: {km_fmt} km<br>"
                    f"Precio: <b>{format_clp(t_precio_oferta)}</b><extra></extra>"
                ),
            )
        )
        fig_comp.update_layout(
            height=340,
            yaxis=dict(tickprefix="$ ", separatethousands=True),
            xaxis=dict(separatethousands=True),
            margin=dict(l=15, r=15, t=40, b=15),
        )
        st.plotly_chart(fig_comp, width="stretch")

    with cg2:
        comp_ranked = comp_df.assign(
            _dist=(comp_df["anio"] - int(t_anio)).abs() * 15_000
            + (comp_df["kilometraje"] - int(t_km)).abs()
        ).nsmallest(10, "_dist")
        comp_table = pd.DataFrame(
            {
                "Portal": comp_ranked["portal_badge"],
                "Año": comp_ranked["anio"].astype(int),
                "Kilometraje": comp_ranked["kilometraje"].astype(int),
                "Transmisión": comp_ranked["transmision"],
                "Comuna": comp_ranked["comuna"],
                "Precio": comp_ranked["precio"].astype(int),
                "Precio Justo IA": comp_ranked["precio_justo_ia"].round(0).astype(int),
                "Enlace": comp_ranked["url"],
            }
        )
        st.dataframe(
            comp_table,
            column_config={
                "Año": st.column_config.NumberColumn(format="%d"),
                "Kilometraje": st.column_config.NumberColumn(format="%d km"),
                "Precio": st.column_config.NumberColumn(format="$ %d"),
                "Precio Justo IA": st.column_config.NumberColumn(format="$ %d"),
                "Enlace": st.column_config.LinkColumn("Enlace", display_text="🔗 Ver"),
            },
            hide_index=True,
            width="stretch",
            height=340,
        )


def render_tab_manual_appraiser(data_bundle: Dict[str, Any]) -> None:
    """
    Pestaña 3: Tasador interactivo con IA + comparables reales de mercado.
    """
    clean_df = data_bundle["clean_df"]
    model_mtime = config.MODEL_PATH.stat().st_mtime

    st.markdown("### 🧮 Tasador Interactivo en Vivo con Inteligencia Artificial")
    st.caption(
        "Ingresa los datos de cualquier vehículo que estés evaluando comprar o vender. "
        "El motor **LightGBM + Target Encoding Jerárquico** calculará al instante su **Precio Justo IA**, "
        "su generación, su nivel socioeconómico comunal y lo comparará contra unidades reales en la BD."
    )
    _render_appraiser_fragment(clean_df, model_mtime)


# ==============================================================================
# 9. PESTAÑA 4: 🔍 EXPLORADOR GENERAL SQLITE + RADAR DE REBAJAS RECIENTES
# ==============================================================================
@st.fragment
def _render_explorer_table_fragment(full_df: pd.DataFrame) -> None:
    """Fragmento aislado para filtrar y explorar 'autos_historico' sin recargar toda la UI."""
    est_col1, est_col2 = st.columns([2, 2])
    with est_col1:
        estados_disp = sorted(full_df["estado"].dropna().astype(str).unique().tolist())
        sel_estados = st.multiselect(
            "📌 Estado en Base de Datos:",
            options=estados_disp,
            default=estados_disp,
            key="exp_estados_flt",
        )

    sub_full = full_df.loc[full_df["estado"].isin(sel_estados)] if sel_estados else full_df

    filtered = apply_multifacet_filters(
        sub_full,
        key_prefix="exp_flt",
        default_only_opportunities=False,
        allow_category_filter=True,
    )

    with est_col2:
        st.markdown("<div style='height: 1.65rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            label=f"📥 Exportar {len(filtered)} Vehículos Filtrados a CSV",
            data=lambda df_exp=filtered: df_exp.to_csv(index=False).encode("utf-8"),
            file_name=f"base_sqlite_filtrada_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            on_click="ignore",
            key="exp_download_csv_btn",
            width="stretch",
        )

    st.markdown(
        f"**Mostrando `{len(filtered)}` de `{len(full_df)}` vehículos registrados en SQLite.**"
    )

    multi_col = (
        filtered["multi_portal_info"]
        if "multi_portal_info" in filtered.columns
        else "Publicación única"
    )
    explorer_view = pd.DataFrame(
        {
            "Portal": filtered["portal_badge"],
            "ID Aviso": filtered["id_aviso"],
            "Categoría IA": filtered["categoria_arbitraje"],
            "Estado": filtered["estado"].str.upper(),
            "Marca": filtered["marca"],
            "Modelo": filtered["modelo"],
            "Año": filtered["anio"],
            "Generación": filtered["generacion"],
            "Kilometraje": filtered["kilometraje"],
            "Comuna": filtered["comuna"],
            "Transmisión": filtered["transmision"],
            "Combustible": filtered["combustible"],
            "Precio Actual (CLP)": filtered["precio"],
            "Precio Justo IA (CLP)": filtered["precio_justo_ia"].round(0).astype(int),
            "Margen IA (CLP)": filtered["margen_ganancia_clp"].round(0).astype(int),
            "Dcto. IA (%)": (filtered["descuento_pct"] * 100.0).round(1),
            "Cambios Precio": filtered["cambios_precio"],
            "Multi-Portal": multi_col,
            "URL": filtered["url"],
        }
    )

    st.dataframe(
        explorer_view,
        column_config={
            "Año": st.column_config.NumberColumn("Año", format="%d"),
            "Kilometraje": st.column_config.NumberColumn("Kilometraje", format="%d km"),
            "Precio Actual (CLP)": st.column_config.NumberColumn("Precio Actual", format="$ %d"),
            "Precio Justo IA (CLP)": st.column_config.NumberColumn("Precio Justo IA", format="$ %d"),
            "Margen IA (CLP)": st.column_config.NumberColumn("Margen IA", format="$ %d"),
            "Dcto. IA (%)": st.column_config.NumberColumn("Dcto. IA", format="%.1f %%"),
            "URL": st.column_config.LinkColumn("Enlace", display_text="🔗 Abrir"),
        },
        hide_index=True,
        width="stretch",
        height=460,
    )

    # Panel de gestión para restaurar avisos descartados/vendidos o purgar publicidad patrocinada
    inactive_df = full_df.loc[full_df["estado"].isin(["descartado", "vendido"])].copy()
    with st.expander(
        f"🗂️ **Gestionar / Restaurar Avisos Descartados o Vendidos ({len(inactive_df)})**",
        expanded=False,
    ):
        g_col1, g_col2 = st.columns([2.4, 1.0])
        with g_col1:
            if inactive_df.empty:
                st.caption("No hay avisos descartados ni vendidos actualmente.")
            else:
                options_map = {
                    f"[{str(r['estado']).upper()}] {r['id_aviso']} — {r['marca']} {r['modelo']} ({int(r['anio'])}) | {format_clp(r['precio'])}": str(r["id_aviso"])
                    for _, r in inactive_df.head(200).iterrows()
                }
                sel_restore_label = st.selectbox(
                    "Selecciona un aviso para restaurar a Activo:",
                    options=list(options_map.keys()),
                    key="exp_restore_select",
                )
                if st.button(
                    "↩️ Restaurar aviso seleccionado a Activo",
                    key="exp_restore_btn",
                    type="primary",
                ):
                    restore_id = options_map[sel_restore_label]
                    _update_listing_status_and_refresh(restore_id, "activo", sel_restore_label)
                    st.rerun(scope="app")
        with g_col2:
            st.markdown("<div style='height: 1.65rem;'></div>", unsafe_allow_html=True)
            if st.button(
                "🧹 Limpiar avisos patrocinados",
                key="exp_cleanup_ads_btn",
                help="Descarta automáticamente anuncios publicitarios patrocinados (ej. click1.mercadolibre.cl).",
                width="stretch",
            ):
                cleaned_n = DatabaseManager(db_path=config.DB_PATH).cleanup_sponsored_ad_listings()
                if cleaned_n > 0:
                    load_dashboard_datasets.clear()
                    st.success(f"Se descartaron {cleaned_n} anuncios patrocinados.")
                    st.rerun(scope="app")
                else:
                    st.info("No se encontraron anuncios patrocinados activos en la BD.")


@st.fragment
def _render_price_drops_and_logs_fragment(price_history_df: pd.DataFrame) -> None:
    """Fragmento dedicado al Radar de Rebajas Recientes y visor de logs en tiempo real."""
    drops_count = 0
    if not price_history_df.empty and "variacion_clp" in price_history_df.columns:
        drops_mask = price_history_df["variacion_clp"] < 0
        if "estado" in price_history_df.columns:
            drops_mask &= price_history_df["estado"].astype(str).str.lower() == "activo"
        drops_count = int(drops_mask.sum())

    with st.expander(
        f"📉 Radar de Rebajas y Cambios de Precio ({len(price_history_df)} movimientos | {drops_count} rebajas activas)",
        expanded=True,
    ):
        if price_history_df.empty:
            st.caption("Aún no se registran cambios de precio en la tabla 'precio_historial'.")
        else:
            only_active_drops = st.toggle(
                "Solo rebajas activas (bajas de precio en vehículos vigentes)",
                value=True,
                key="exp_only_active_drops",
            )
            ph_work = price_history_df
            if only_active_drops and "variacion_clp" in ph_work.columns:
                mask_d = ph_work["variacion_clp"] < 0
                if "estado" in ph_work.columns:
                    mask_d &= ph_work["estado"].fillna("").astype(str).str.lower() == "activo"
                ph_work = ph_work.loc[mask_d]

            if ph_work.empty:
                st.info("No hay rebajas activas con los filtros seleccionados (desactiva el switch para ver todos los movimientos).")
            else:
                ph_view = pd.DataFrame(
                    {
                        "Portal": ph_work.get("portal_badge", ph_work["id_aviso"]),
                        "ID Aviso": ph_work["id_aviso"],
                        "Marca": ph_work.get("marca", ""),
                        "Modelo": ph_work.get("modelo", ""),
                        "Año": pd.to_numeric(ph_work.get("anio", 0), errors="coerce").fillna(0).astype(int),
                        "Estado": ph_work.get("estado", pd.Series("", index=ph_work.index)).fillna("").astype(str).str.upper(),
                        "Precio Antiguo": pd.to_numeric(ph_work["precio_antiguo"], errors="coerce").fillna(0).astype(int),
                        "Precio Nuevo": pd.to_numeric(ph_work["precio_nuevo"], errors="coerce").fillna(0).astype(int),
                        "Variación (CLP)": pd.to_numeric(ph_work.get("variacion_clp", 0), errors="coerce").fillna(0).astype(int),
                        "Variación (%)": pd.to_numeric(ph_work.get("variacion_pct", 0.0), errors="coerce").fillna(0.0).round(1),
                        "Fecha Cambio": ph_work["fecha_cambio"],
                        "URL": ph_work.get("url", ""),
                    }
                )
                st.dataframe(
                    ph_view,
                    column_config={
                        "Año": st.column_config.NumberColumn(format="%d"),
                        "Precio Antiguo": st.column_config.NumberColumn(format="$ %d"),
                        "Precio Nuevo": st.column_config.NumberColumn(format="$ %d"),
                        "Variación (CLP)": st.column_config.NumberColumn("Variación", format="$ %d"),
                        "Variación (%)": st.column_config.NumberColumn("Var %", format="%.1f %%"),
                        "URL": st.column_config.LinkColumn("Enlace", display_text="🔗 Ver"),
                    },
                    hide_index=True,
                    width="stretch",
                    height=320,
                )

    with st.expander(
        "📜 Registro de Eventos en Tiempo Real (`logs/system.log`)", expanded=False
    ):
        log_c1, log_c2 = st.columns([4, 1])
        with log_c2:
            if st.button("🔄 Refrescar Log", key="exp_refresh_log_btn", width="stretch"):
                st.rerun(scope="fragment")
        with log_c1:
            st.caption("Mostrando las últimas 60 líneas registradas en logs/system.log")
        if config.LOG_FILE_PATH.exists():
            try:
                lines = config.LOG_FILE_PATH.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()
                st.code("\n".join(lines[-60:]), language="text")
            except Exception as exc:
                st.caption(f"No se pudo leer system.log: {exc}")
        else:
            st.caption("El archivo logs/system.log aún no existe.")


def render_tab_explorer(data_bundle: Dict[str, Any]) -> None:
    """
    Explorador histórico completo de 'autos_historico' y Radar de Rebajas 'precio_historial' en SQLite.
    """
    full_df = data_bundle["full_df"]
    price_history_df = data_bundle["price_history_df"]

    head_col1, head_col2, head_col3 = st.columns([2.2, 1.1, 1.1])
    with head_col1:
        st.markdown("### 🔍 Explorador General de Base de Datos SQLite (`autos_historico`)")
        st.caption(
            "Filtra, ordena y audita todos los vehículos capturados en **MercadoLibre (`MLC-`)**, "
            "**Autocosmos (`AC-`)**, **Yapo (`YAPO-`)** y **Chileautos (`CL-`)**."
        )
    with head_col2:
        if st.button(
            "🔄 Ejecutar Escaneo Diario Ahora",
            key="exp_scan_now_btn",
            type="primary",
            width="stretch",
        ):
            with st.spinner("Ejecutando barrido autónomo completo (main.py)..."):
                ok, msg = execute_backend_cycle(
                    force_train=False,
                    pages_per_portal=config.SCRAPER_PAGES_PER_SEGMENT,
                    max_telegram_alerts=6,
                )
            if ok:
                st.session_state["last_scan_message"] = msg
                st.rerun(scope="app")
            else:
                st.error(msg)
    with head_col3:
        if st.button(
            "🧠 Forzar Re-Entrenamiento IA",
            key="exp_retrain_btn",
            width="stretch",
        ):
            with st.spinner("Re-entrenando LightGBM sobre data/autos_clean.csv..."):
                predictor = CarPricePredictor(model_path=config.MODEL_PATH)
                predictor.train_and_evaluate()
                st.cache_data.clear()
                st.cache_resource.clear()
            st.success("✅ Modelo LightGBM re-entrenado y actualizado en 'models/model.pkl'.")
            st.rerun(scope="app")

    st.markdown("---")

    if full_df.empty:
        st.info("La base de datos no contiene registros aún.")
        return

    _render_explorer_table_fragment(full_df)
    st.markdown("---")
    _render_price_drops_and_logs_fragment(price_history_df)


# ==============================================================================
# 10. PUNTO DE ENTRADA PRINCIPAL DE LA APLICACIÓN STREAMLIT
# ==============================================================================
def main() -> None:
    is_healthy, health_msg = check_system_health()
    if not is_healthy:
        render_resilience_onboarding(health_msg)
        return

    try:
        db_mtime = config.DB_PATH.stat().st_mtime
        model_mtime = config.MODEL_PATH.stat().st_mtime
        clean_csv_path = config.DATA_DIR / "autos_clean.csv"
        opp_csv_path = config.DATA_DIR / "oportunidades_detectadas.csv"
        clean_mtime = clean_csv_path.stat().st_mtime if clean_csv_path.exists() else 0.0
        opp_mtime = opp_csv_path.stat().st_mtime if opp_csv_path.exists() else 0.0

        data_bundle = load_dashboard_datasets(
            db_mtime=db_mtime,
            model_mtime=model_mtime,
            clean_csv_mtime=clean_mtime,
            opp_csv_mtime=opp_mtime,
        )
    except Exception as exc:
        render_resilience_onboarding(
            f"Se requiere reinicializar los artefactos del sistema ({exc})."
        )
        return

    render_sidebar(data_bundle)
    inject_theme_css(is_light_theme())

    st.markdown(
        _clean_html(
            """
            <div class="hero-banner">
                <h1 class="hero-title">🏎️ Suite de Arbitraje Automotriz IA Chile — Radar Multi-Portal</h1>
                <p class="hero-subtitle">
                    Extracción en vivo desde <b>MercadoLibre Autos</b>, <b>Autocosmos Chile</b>, <b>Yapo.cl</b> y <b>Chileautos</b>
                    | Filtrado Multifacético, Analítica 3D y Regresión Híbrida <b>LightGBM</b>.
                </p>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # Navegación lazy de alto rendimiento: solo renderiza la sección seleccionada
    nav_kwargs = (
        {"default": MAIN_SECTIONS[0]}
        if "active_main_section" not in st.session_state
        else {}
    )
    active_section = st.segmented_control(
        "Sección:",
        options=MAIN_SECTIONS,
        selection_mode="single",
        required=True,
        width="stretch",
        label_visibility="collapsed",
        key="active_main_section",
        **nav_kwargs,
    )
    if not active_section:
        active_section = MAIN_SECTIONS[0]

    if active_section == MAIN_SECTIONS[0]:
        render_tab_opportunities(data_bundle)
    elif active_section == MAIN_SECTIONS[1]:
        render_tab_favorites(data_bundle)
    elif active_section == MAIN_SECTIONS[2]:
        render_tab_market_curves(data_bundle)
    elif active_section == MAIN_SECTIONS[3]:
        render_tab_manual_appraiser(data_bundle)
    else:
        render_tab_explorer(data_bundle)


if __name__ == "__main__":
    main()
