"""
ui/domain.py - Utilidades de dominio automotriz chileno, portales y formato monetario.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import config

NLP_LABELS_MAP: Dict[str, str] = {
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

COMUNA_TO_REGION: Dict[str, str] = {
    # Región de Arica y Parinacota (XV)
    "ARICA": "Arica y Parinacota",
    "PUTRE": "Arica y Parinacota",
    # Región de Tarapacá (I)
    "IQUIQUE": "Tarapacá",
    "ALTO HOSPICIO": "Tarapacá",
    "POZO ALMONTE": "Tarapacá",
    # Región de Antofagasta (II)
    "ANTOFAGASTA": "Antofagasta",
    "CALAMA": "Antofagasta",
    "TOCOPILLA": "Antofagasta",
    "MEJILLONES": "Antofagasta",
    "SAN PEDRO DE ATACAMA": "Antofagasta",
    # Región de Atacama (III)
    "COPIAPO": "Atacama",
    "VALLENAR": "Atacama",
    "CALDERA": "Atacama",
    "CHANARAL": "Atacama",
    # Región de Coquimbo (IV)
    "LA SERENA": "Coquimbo",
    "COQUIMBO": "Coquimbo",
    "OVALLE": "Coquimbo",
    "ILLAPEL": "Coquimbo",
    "SALAMANCA": "Coquimbo",
    "VICUNA": "Coquimbo",
    # Región de Valparaíso (V)
    "VALPARAISO": "Valparaíso",
    "VINA DEL MAR": "Valparaíso",
    "CONCON": "Valparaíso",
    "QUILPUE": "Valparaíso",
    "VILLA ALEMANA": "Valparaíso",
    "LIMACHE": "Valparaíso",
    "QUILLOTA": "Valparaíso",
    "LA CALERA": "Valparaíso",
    "SAN ANTONIO": "Valparaíso",
    "SANTO DOMINGO": "Valparaíso",
    "CARTAGENA": "Valparaíso",
    "EL QUISCO": "Valparaíso",
    "EL TABO": "Valparaíso",
    "ALGARROBO": "Valparaíso",
    "CASABLANCA": "Valparaíso",
    "LOS ANDES": "Valparaíso",
    "SAN FELIPE": "Valparaíso",
    "CALLE LARGA": "Valparaíso",
    "LA LIGUA": "Valparaíso",
    "ZAPALLAR": "Valparaíso",
    "PAPUDO": "Valparaíso",
    "PUCHUNCAVI": "Valparaíso",
    "QUINTERO": "Valparaíso",
    # Región Metropolitana de Santiago (RM)
    "SANTIAGO": "Metropolitana (RM)",
    "LAS CONDES": "Metropolitana (RM)",
    "VITACURA": "Metropolitana (RM)",
    "LO BARNECHEA": "Metropolitana (RM)",
    "PROVIDENCIA": "Metropolitana (RM)",
    "NUNOA": "Metropolitana (RM)",
    "LA REINA": "Metropolitana (RM)",
    "MACUL": "Metropolitana (RM)",
    "PENALOLEN": "Metropolitana (RM)",
    "LA FLORIDA": "Metropolitana (RM)",
    "PUENTE ALTO": "Metropolitana (RM)",
    "PIRQUE": "Metropolitana (RM)",
    "SAN JOSE DE MAIPO": "Metropolitana (RM)",
    "MAIPU": "Metropolitana (RM)",
    "CERRILLOS": "Metropolitana (RM)",
    "ESTACION CENTRAL": "Metropolitana (RM)",
    "PUDAHUEL": "Metropolitana (RM)",
    "LO PRADO": "Metropolitana (RM)",
    "PRADO": "Metropolitana (RM)",
    "CERRO NAVIA": "Metropolitana (RM)",
    "QUINTA NORMAL": "Metropolitana (RM)",
    "RENCA": "Metropolitana (RM)",
    "QUILICURA": "Metropolitana (RM)",
    "HUECHURABA": "Metropolitana (RM)",
    "CONCHALI": "Metropolitana (RM)",
    "RECOLETA": "Metropolitana (RM)",
    "INDEPENDENCIA": "Metropolitana (RM)",
    "SAN MIGUEL": "Metropolitana (RM)",
    "SAN JOAQUIN": "Metropolitana (RM)",
    "PEDRO AGUIRRE CERDA": "Metropolitana (RM)",
    "LO ESPEJO": "Metropolitana (RM)",
    "LA CISTERNA": "Metropolitana (RM)",
    "EL BOSQUE": "Metropolitana (RM)",
    "SAN RAMON": "Metropolitana (RM)",
    "LA GRANJA": "Metropolitana (RM)",
    "LA PINTANA": "Metropolitana (RM)",
    "SAN BERNARDO": "Metropolitana (RM)",
    "BUIN": "Metropolitana (RM)",
    "PAINE": "Metropolitana (RM)",
    "CALERA DE TANGO": "Metropolitana (RM)",
    "COLINA": "Metropolitana (RM)",
    "CHICUREO": "Metropolitana (RM)",
    "LAMPA": "Metropolitana (RM)",
    "TILTIL": "Metropolitana (RM)",
    "TALAGANTE": "Metropolitana (RM)",
    "PENAFLOR": "Metropolitana (RM)",
    "PADRE HURTADO": "Metropolitana (RM)",
    "ISLA DE MAIPO": "Metropolitana (RM)",
    "EL MONTE": "Metropolitana (RM)",
    "MELIPILLA": "Metropolitana (RM)",
    "CURACAVI": "Metropolitana (RM)",
    # Región de O'Higgins (VI)
    "RANCAGUA": "O'Higgins",
    "MACHALI": "O'Higgins",
    "RENGO": "O'Higgins",
    "SAN FERNANDO": "O'Higgins",
    "SANTA CRUZ": "O'Higgins",
    "PICHILEMU": "O'Higgins",
    "MOSTAZAL": "O'Higgins",
    "GRANEROS": "O'Higgins",
    "SAN VICENTE": "O'Higgins",
    "CHEPICA": "O'Higgins",
    # Región del Maule (VII)
    "TALCA": "Maule",
    "CURICO": "Maule",
    "LINARES": "Maule",
    "CAUQUENES": "Maule",
    "CONSTITUCION": "Maule",
    "MOLINA": "Maule",
    "PARRAL": "Maule",
    "SAN CLEMENTE": "Maule",
    "SAN JAVIER": "Maule",
    "VILLA ALEGRE": "Maule",
    "BUCALEMU": "O'Higgins",
    # Región de Ñuble (XVI)
    "CHILLAN": "Ñuble",
    "CHILLAN VIEJO": "Ñuble",
    "SAN CARLOS": "Ñuble",
    "BULNES": "Ñuble",
    "PORTEZUELO": "Ñuble",
    "QUIRIHUE": "Ñuble",
    # Región del Biobío (VIII)
    "CONCEPCION": "Biobío",
    "TALCAHUANO": "Biobío",
    "SAN PEDRO DE LA PAZ": "Biobío",
    "CHIGUAYANTE": "Biobío",
    "HUALPEN": "Biobío",
    "CORONEL": "Biobío",
    "LOTA": "Biobío",
    "TOME": "Biobío",
    "PENCO": "Biobío",
    "LOS ANGELES": "Biobío",
    "ARAUCO": "Biobío",
    "CANETE": "Biobío",
    "CURANILAHUE": "Biobío",
    "CABRERO": "Biobío",
    "YUMBEL": "Biobío",
    "LAJA": "Biobío",
    "SANTA JUANA": "Biobío",
    "MULCHEN": "Biobío",
    # Región de La Araucanía (IX)
    "TEMUCO": "La Araucanía",
    "PADRE LAS CASAS": "La Araucanía",
    "VILLARRICA": "La Araucanía",
    "PUCON": "La Araucanía",
    "ANGOL": "La Araucanía",
    "VICTORIA": "La Araucanía",
    "LAUTARO": "La Araucanía",
    "PITRUFQUEN": "La Araucanía",
    "CUNCO": "La Araucanía",
    "GORBEA": "La Araucanía",
    "NUEVA IMPERIAL": "La Araucanía",
    # Región de Los Ríos (XIV)
    "VALDIVIA": "Los Ríos",
    "LA UNION": "Los Ríos",
    "RIO BUENO": "Los Ríos",
    "PANGUIPULLI": "Los Ríos",
    "FUTRONO": "Los Ríos",
    "CORRAL": "Los Ríos",
    # Región de Los Lagos (X)
    "PUERTO MONTT": "Los Lagos",
    "PUERTO VARAS": "Los Lagos",
    "OSORNO": "Los Lagos",
    "CASTRO": "Los Lagos",
    "ANCUD": "Los Lagos",
    "QUELLON": "Los Lagos",
    "FRUTILLAR": "Los Lagos",
    "LLANQUIHUE": "Los Lagos",
    "CALBUCO": "Los Lagos",
    "LOS MUERMOS": "Los Lagos",
    "PURRANQUE": "Los Lagos",
    # Región de Aysén (XI)
    "COYHAIQUE": "Aysén",
    "PUERTO AYSEN": "Aysén",
    # Región de Magallanes (XII)
    "PUNTA ARENAS": "Magallanes",
    "PUERTO NATALES": "Magallanes",
}


def _strip_accents(text: str) -> str:
    """Elimina marcas diacríticas para normalización geográfica sin depender de pipeline."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def resolve_region(comuna: Any) -> str:
    """Resuelve la Región de Chile a partir del nombre de la comuna."""
    if not comuna or not isinstance(comuna, str):
        return "Metropolitana (RM)"
    clean_key = _strip_accents(comuna.strip().upper())
    if clean_key in COMUNA_TO_REGION:
        return COMUNA_TO_REGION[clean_key]
    for k, reg in COMUNA_TO_REGION.items():
        if k in clean_key:
            return reg
    return "Metropolitana (RM)"


def format_clp(val: Any) -> str:
    """Formatea un monto numérico como Pesos Chilenos ($ 12.450.000)."""
    try:
        if val is None or pd.isna(val):
            return "$ 0"
        num = float(val)
        if not math.isfinite(num):
            return "$ 0"
        rounded = int(round(num))
        if rounded < 0:
            return f"- $ {abs(rounded):,}".replace(",", ".")
        return f"$ {rounded:,}".replace(",", ".")
    except Exception:
        return "$ 0"


def resolve_portal_info(id_aviso: Any) -> Tuple[str, str, str]:
    """
    Identifica el portal de origen según el prefijo de 'id_aviso'.
    Retorna: (nombre_corto, etiqueta_limpia, css_class)
    """
    uid = str(id_aviso or "").strip().upper()
    if uid.startswith("MLC"):
        return "MercadoLibre", "MercadoLibre", "portal-badge-mlc"
    if uid.startswith("AC-"):
        return "Autocosmos", "Autocosmos", "portal-badge-ac"
    if uid.startswith("YAPO-"):
        return "Yapo.cl", "Yapo.cl", "portal-badge-yapo"
    return "Chileautos", "Chileautos", "portal-badge-cl"


def get_vehicle_emoji(marca: Any, modelo: Any) -> str:
    """Retorna un indicador compacto acorde al segmento del vehículo."""
    mod = f"{marca or ''} {modelo or ''}".upper()
    if any(k in mod for k in ("HILUX", "L200", "RANGER", "NP300", "BT-50", "SILVERADO")):
        return "🛻"
    if any(
        k in mod
        for k in (
            "RAV4",
            "CX-5",
            "TIGGO",
            "SPORTAGE",
            "TUCSON",
            "QASHQAI",
            "X-TRAIL",
            "XV",
            "3008",
            "CR-V",
            "HR-V",
            "PILOT",
            "XC60",
            "XC40",
            "XC90",
            "X1",
            "X3",
            "GLA",
            "GLC",
        )
    ):
        return "🚙"
    if any(
        k in mod
        for k in (
            "MORNING",
            "SPARK",
            "MARCH",
            "SWIFT",
            "YARIS",
            "BALENO",
            "208",
            "GOL",
            "RIO",
            "V40",
            "118I",
            "A200",
        )
    ):
        return "🚗"
    return "🚘"


def normalize_model_input(raw_modelo: Any) -> str:
    """Normaliza variantes y errores tipográficos de modelos según config.MODEL_ALIASES."""
    if raw_modelo is None:
        return ""
    try:
        if pd.isna(raw_modelo):
            return ""
    except Exception:
        pass
    clean_m = re.sub(r"\s+", " ", str(raw_modelo).strip().upper())
    if not clean_m:
        return ""
    aliases = getattr(config, "MODEL_ALIASES", {})
    if clean_m in aliases:
        return aliases[clean_m]
    compact = clean_m.replace("-", "").replace(" ", "")
    if compact in aliases:
        return aliases[compact]
    return clean_m


def assign_portal_and_region_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega vectorialmente las columnas 'portal', 'portal_badge' y 'region' sobre una copia."""
    out = df.copy()
    if out.empty:
        for col in ("portal", "portal_badge", "region"):
            if col not in out.columns:
                out[col] = pd.Series(dtype="object")
        return out

    uids = (
        out["id_aviso"].fillna("").astype(str).str.strip().str.upper()
        if "id_aviso" in out.columns
        else pd.Series("", index=out.index)
    )
    conds = [
        uids.str.startswith("MLC"),
        uids.str.startswith("AC-"),
        uids.str.startswith("YAPO-"),
    ]
    portal_names = ["MercadoLibre", "Autocosmos", "Yapo.cl"]
    out["portal"] = np.select(conds, portal_names, default="Chileautos")
    out["portal_badge"] = np.select(conds, portal_names, default="Chileautos")

    if "comuna" in out.columns:
        unique_comunas = out["comuna"].fillna("").astype(str).unique()
        reg_map = {c: resolve_region(c) for c in unique_comunas}
        out["region"] = (
            out["comuna"].fillna("").astype(str).map(reg_map).fillna("Metropolitana (RM)")
        )
    else:
        out["region"] = "Metropolitana (RM)"
    return out


def extract_nlp_tags(row: pd.Series) -> str:
    """Convierte las banderas binarias NLP en texto legible para las tarjetas."""
    tags: List[str] = []
    for col, label in NLP_LABELS_MAP.items():
        val = row.get(col, 0)
        try:
            if val is not None and not pd.isna(val) and int(float(val)) == 1:
                tags.append(label)
        except Exception:
            if str(val).strip().lower() in ("1", "true", "si", "sí"):
                tags.append(label)
    return " · ".join(tags) if tags else "Equipamiento Estándar"
