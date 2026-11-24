"""
config.py - Configuración centralizada del Sistema de Arbitraje Automotriz (Chile)
Contiene rutas, parámetros del scraper, diccionarios de dominio chileno (comunas,
generaciones, NLP), hiperparámetros de Machine Learning y credenciales de alertas.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

# ==============================================================================
# 1. RUTAS DEL SISTEMA Y ARCHIVOS LOCALES
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent


def _load_local_dotenv() -> None:
    """Carga variables desde '.env' local sin sobrescribir variables ya presentes."""
    dotenv_path = BASE_DIR / ".env"
    if not dotenv_path.exists():
        return
    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            if key:
                os.environ.setdefault(key, val)
    except Exception:
        pass


def _get_secret(key: str, default: str = "") -> str:
    """Lee secretos desde entorno/.env o desde st.secrets en despliegue cloud."""
    env_val = os.getenv(key)
    if env_val:
        return env_val
    st_mod = sys.modules.get("streamlit")
    if st_mod is not None:
        try:
            secret_val = st_mod.secrets.get(key, default)
            if secret_val is not None and str(secret_val) != "":
                return str(secret_val)
        except Exception:
            pass
    return default


_load_local_dotenv()

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Crear directorios automáticamente si no existen
for directory in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "chileautos_arbitraje.db"
DATABASE_URL = _get_secret("DATABASE_URL", "")
MODEL_PATH = MODELS_DIR / "model.pkl"
CLEAN_DATA_PATH = DATA_DIR / "autos_clean.parquet"
LOG_FILE_PATH = LOGS_DIR / "system.log"
CHILEAUTOS_IMPORT_DIR = DATA_DIR / "import_chileautos"
CHILEAUTOS_IMPORT_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_SEGMENTS_PATH = DATA_DIR / "custom_segments.json"


def setup_logger(name: str) -> logging.Logger:
    """Configura un logger de módulo sin crear FileHandlers huérfanos ni duplicados."""
    root = logging.getLogger()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")

    had_any_handler = bool(root.handlers)
    target_log = str(LOG_FILE_PATH.resolve())
    has_file_handler = any(
        isinstance(h, logging.FileHandler)
        and getattr(h, "baseFilename", None)
        and str(Path(h.baseFilename).resolve()) == target_log
        for h in root.handlers
    )
    if not has_file_handler:
        fh = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    if not had_any_handler:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)
        root.setLevel(logging.INFO)

    mod_logger = logging.getLogger(name)
    mod_logger.setLevel(logging.INFO)
    return mod_logger

# ==============================================================================
# 2. CONFIGURACIÓN DEL WEB SCRAPER MULTI-PORTAL Y SEGMENTADO (MÓDULO 1)
# ==============================================================================
# Portales públicos de autos usados en Chile habilitados
CHILEAUTOS_BASE_URL = "https://www.chileautos.cl/vehiculos/autos-veh%C3%ADculo/"
AUTOCOSMOS_BASE_URL = "https://www.autocosmos.cl/auto/usado"
YAPO_BASE_URL = "https://www.yapo.cl/autos-usados"
MERCADOLIBRE_BASE_URL = "https://autos.mercadolibre.cl"

# Segmentos clave (Marca, Modelo) para barrido estructurado y curvas específicas
TARGET_SEGMENTS = [
    # Generalistas Líderes en Chile
    ("MAZDA", "3"),
    ("MAZDA", "CX-5"),
    ("TOYOTA", "YARIS"),
    ("TOYOTA", "RAV4"),
    ("TOYOTA", "COROLLA"),
    ("TOYOTA", "HILUX"),
    ("SUZUKI", "SWIFT"),
    ("SUZUKI", "BALENO"),
    ("KIA", "RIO"),
    ("KIA", "MORNING"),
    ("KIA", "SPORTAGE"),
    ("HYUNDAI", "ACCENT"),
    ("HYUNDAI", "TUCSON"),
    ("CHEVROLET", "SAIL"),
    ("PEUGEOT", "208"),
    ("PEUGEOT", "3008"),
    # Nissan
    ("NISSAN", "QASHQAI"),
    ("NISSAN", "VERSA"),
    ("NISSAN", "X-TRAIL"),
    ("SUBARU", "XV"),
    ("VOLKSWAGEN", "GOL"),
    ("FORD", "RANGER"),
    ("MITSUBISHI", "L200"),
    # Honda
    ("HONDA", "CIVIC"),
    ("HONDA", "CR-V"),
    ("HONDA", "HR-V"),
    ("HONDA", "PILOT"),
    # Volvo
    ("VOLVO", "XC60"),
    ("VOLVO", "XC40"),
    ("VOLVO", "V40"),
    ("VOLVO", "XC90"),
    # BMW
    ("BMW", "320I"),
    ("BMW", "118I"),
    ("BMW", "X1"),
    ("BMW", "X3"),
    # Mercedes-Benz
    ("MERCEDES-BENZ", "A200"),
    ("MERCEDES-BENZ", "C200"),
    ("MERCEDES-BENZ", "GLA"),
    ("MERCEDES-BENZ", "GLC"),
]


def load_custom_segments() -> List[Tuple[str, str]]:
    """Lee las marcas y modelos personalizados agregados desde la interfaz UI."""
    if not CUSTOM_SEGMENTS_PATH.exists():
        return []
    try:
        raw = json.loads(CUSTOM_SEGMENTS_PATH.read_text(encoding="utf-8"))
        out: List[Tuple[str, str]] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                m, mod = str(item[0]).strip().upper(), str(item[1]).strip().upper()
                if m and mod:
                    out.append((m, mod))
        return out
    except Exception:
        return []


def get_all_target_segments() -> List[Tuple[str, str]]:
    """Devuelve la unión sin duplicados de TARGET_SEGMENTS + custom_segments.json."""
    combined = list(TARGET_SEGMENTS)
    existing = set(combined)
    for pair in load_custom_segments():
        if pair not in existing:
            combined.append(pair)
            existing.add(pair)
    return combined


# Aliases y normalizaciones de nombres de modelos para portales
MODEL_ALIASES = {
    "XTRAIL": "X-TRAIL",
    "XTRAILL": "X-TRAIL",
    "X TRAIL": "X-TRAIL",
    "X-TRAIL": "X-TRAIL",
    "CX5": "CX-5",
    "CX 5": "CX-5",
    "CX3": "CX-3",
    "CX30": "CX-30",
    "CRV": "CR-V",
    "CR V": "CR-V",
    "HRV": "HR-V",
    "HR V": "HR-V",
    "RAV 4": "RAV4",
    "SERIE 3": "320I",
    "SERIE 1": "118I",
    "CLASE C": "C200",
    "CLASE A": "A200",
}


def add_custom_segment(marca: str, modelo: str) -> Tuple[bool, str]:
    """Agrega una nueva dupla (Marca, Modelo) desde la UI y la persiste en disco."""
    m = str(marca or "").strip().upper()
    mod = str(modelo or "").strip().upper()
    if not m or not mod:
        return False, "Debes ingresar tanto la Marca como el Modelo."

    pair = (m, mod)
    current_all = set(get_all_target_segments())
    custom_list = load_custom_segments()
    if pair not in set(custom_list) and pair not in set(TARGET_SEGMENTS):
        custom_list.append(pair)
        CUSTOM_SEGMENTS_PATH.write_text(
            json.dumps(custom_list, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if pair not in TARGET_SEGMENTS:
        TARGET_SEGMENTS.append(pair)

    if pair in current_all:
        return True, f"El modelo {m} {mod} ya está activo en el catálogo."
    return True, f"Marca/Modelo '{m} {mod}' añadido exitosamente al catálogo."


# Fusionar automáticamente segmentos personalizados al cargar config.py
for _custom_pair in load_custom_segments():
    if _custom_pair not in TARGET_SEGMENTS:
        TARGET_SEGMENTS.append(_custom_pair)

SCRAPER_MAX_PAGES = int(os.getenv("SCRAPER_MAX_PAGES", "5"))
SCRAPER_PAGES_PER_SEGMENT = int(os.getenv("SCRAPER_PAGES_PER_SEGMENT", "2"))
SCRAPER_ITEMS_PER_PAGE = 12
SCRAPER_TIMEOUT_SECONDS = 20
SCRAPER_MIN_DELAY_SEC = 0.8
SCRAPER_MAX_DELAY_SEC = 1.8
SCRAPER_MAX_RETRIES = 3

# Si un anuncio activo no ha sido visto en un barrido completo o lleva más de X días
# sin aparecer en los barridos completos, se marca como vendido.
STALE_DAYS_TO_MARK_SOLD = 2

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0",
]

# ==============================================================================
# 3. FILTROS DE OUTLIERS Y LIMPIEZA (MÓDULO 2)
# ==============================================================================
MIN_VALID_PRICE_CLP = 500_000          # Excluir < $500.000 CLP o "Consultar" ($0)
MAX_VALID_PRICE_CLP = 180_000_000      # Techo absoluto de cordura para mercado general
MIN_VALID_YEAR = 1995
MAX_VALID_YEAR = 2027
MIN_VALID_KM = 0
MAX_VALID_KM = 600_000

# Parámetros estadísticos para filtro de Outliers por grupo (Marca/Modelo/Año)
IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
MIN_GROUP_SIZE_FOR_STATS = 4           # Mínimo de muestras en grupo para aplicar IQR/Z-Score

# ==============================================================================
# 4. ENRIQUECIMIENTO: GENERACIONES Y PODER ADQUISITIVO COMUNAL (MÓDULO 2)
# ==============================================================================
# Diccionario de generaciones de modelos populares en Chile:
# Formato: ("MARCA", "MODELO"): [(año_inicio, año_fin, "Código_Generación"), ...]
MODEL_GENERATIONS = {
    ("MAZDA", "3"): [
        (2004, 2009, "GEN1_BK"),
        (2010, 2013, "GEN2_BL"),
        (2014, 2018, "GEN3_BM_BN"),
        (2019, 2027, "GEN4_BP"),
    ],
    ("MAZDA", "CX-5"): [
        (2012, 2016, "GEN1_KE"),
        (2017, 2027, "GEN2_KF"),
    ],
    ("TOYOTA", "RAV4"): [
        (2006, 2012, "GEN3_XA30"),
        (2013, 2018, "GEN4_XA40"),
        (2019, 2027, "GEN5_XA50"),
    ],
    ("TOYOTA", "YARIS"): [
        (2006, 2013, "GEN2_XP90"),
        (2014, 2020, "GEN3_XP150"),
        (2021, 2027, "GEN4_AC100"),
    ],
    ("TOYOTA", "COROLLA"): [
        (2008, 2013, "GEN10_E140"),
        (2014, 2019, "GEN11_E170"),
        (2020, 2027, "GEN12_E210"),
    ],
    ("TOYOTA", "HILUX"): [
        (2005, 2015, "GEN7_AN10"),
        (2016, 2027, "GEN8_AN120"),
    ],
    ("SUZUKI", "SWIFT"): [
        (2005, 2010, "GEN1_RS"),
        (2011, 2016, "GEN2_AZG"),
        (2017, 2023, "GEN3_A2L"),
        (2024, 2027, "GEN4_AOL"),
    ],
    ("SUZUKI", "BALENO"): [
        (2016, 2022, "GEN1_WB"),
        (2023, 2027, "GEN2_WB_FL"),
    ],
    ("KIA", "RIO"): [
        (2006, 2011, "GEN2_JB"),
        (2012, 2016, "GEN3_UB"),
        (2017, 2024, "GEN4_YB"),
    ],
    ("KIA", "MORNING"): [
        (2008, 2011, "GEN1_SA"),
        (2012, 2016, "GEN2_TA"),
        (2017, 2027, "GEN3_JA"),
    ],
    ("KIA", "SPORTAGE"): [
        (2011, 2015, "GEN3_SL"),
        (2016, 2021, "GEN4_QL"),
        (2022, 2027, "GEN5_NQ5"),
    ],
    ("HYUNDAI", "ACCENT"): [
        (2006, 2010, "GEN3_MC"),
        (2011, 2017, "GEN4_RB"),
        (2018, 2022, "GEN5_HC"),
        (2023, 2027, "GEN6_BN7"),
    ],
    ("HYUNDAI", "TUCSON"): [
        (2010, 2015, "GEN2_LM"),
        (2016, 2020, "GEN3_TL"),
        (2021, 2027, "GEN4_NX4"),
    ],
    ("CHEVROLET", "SAIL"): [
        (2011, 2015, "GEN2_SGM"),
        (2016, 2023, "GEN3_SGM300"),
        (2024, 2027, "GEN4_NEW_SAIL"),
    ],
    ("NISSAN", "QASHQAI"): [
        (2008, 2013, "GEN1_J10"),
        (2014, 2021, "GEN2_J11"),
        (2022, 2027, "GEN3_J12"),
    ],
    ("NISSAN", "X-TRAIL"): [
        (2002, 2007, "GEN1_T30"),
        (2008, 2014, "GEN2_T31"),
        (2015, 2022, "GEN3_T32"),
        (2023, 2027, "GEN4_T33_EPOWER"),
    ],
    ("NISSAN", "XTRAIL"): [
        (2002, 2007, "GEN1_T30"),
        (2008, 2014, "GEN2_T31"),
        (2015, 2022, "GEN3_T32"),
        (2023, 2027, "GEN4_T33_EPOWER"),
    ],
    ("NISSAN", "VERSA"): [
        (2012, 2019, "GEN1_N17"),
        (2020, 2027, "GEN2_N18"),
    ],
    ("SUBARU", "XV"): [
        (2012, 2017, "GEN1_GP"),
        (2018, 2023, "GEN2_GT"),
        (2024, 2027, "GEN3_CROSSTREK"),
    ],
    ("VOLKSWAGEN", "GOL"): [
        (2009, 2012, "GEN5_G5"),
        (2013, 2016, "GEN6_G6"),
        (2017, 2023, "GEN7_G7_G8"),
    ],
    ("PEUGEOT", "208"): [
        (2013, 2019, "GEN1_A9"),
        (2020, 2027, "GEN2_P21"),
    ],
    ("PEUGEOT", "3008"): [
        (2010, 2016, "GEN1_T84"),
        (2017, 2023, "GEN2_P84"),
        (2024, 2027, "GEN3_P64"),
    ],
    ("FORD", "RANGER"): [
        (2012, 2015, "T6_PRE_FL"),
        (2016, 2022, "T6_FACELIFT"),
        (2023, 2027, "NEXT_GEN_P703"),
    ],
    ("MITSUBISHI", "L200"): [
        (2007, 2015, "GEN4_KB"),
        (2016, 2023, "GEN5_KJ_KK_KL"),
        (2024, 2027, "GEN6_MV"),
    ],
    # --- HONDA ---
    ("HONDA", "CIVIC"): [
        (2006, 2011, "GEN8_FA_FD"),
        (2012, 2015, "GEN9_FB"),
        (2016, 2021, "GEN10_FC_FK"),
        (2022, 2027, "GEN11_FE_FL"),
    ],
    ("HONDA", "CR-V"): [
        (2007, 2011, "GEN3_RE"),
        (2012, 2016, "GEN4_RM"),
        (2017, 2022, "GEN5_RW"),
        (2023, 2027, "GEN6_RS"),
    ],
    ("HONDA", "HR-V"): [
        (2015, 2022, "GEN2_RU"),
        (2023, 2027, "GEN3_RV"),
    ],
    ("HONDA", "PILOT"): [
        (2009, 2015, "GEN2_YF3"),
        (2016, 2022, "GEN3_YF5"),
        (2023, 2027, "GEN4_YG"),
    ],
    # --- VOLVO ---
    ("VOLVO", "XC60"): [
        (2009, 2017, "GEN1_Y20"),
        (2018, 2021, "GEN2_SPA_PRE_FL"),
        (2022, 2027, "GEN2_SPA_B_SERIES"),
    ],
    ("VOLVO", "XC40"): [
        (2018, 2022, "GEN1_CMA_T_SERIES"),
        (2023, 2027, "GEN1_CMA_FACELIFT"),
    ],
    ("VOLVO", "V40"): [
        (2013, 2016, "GEN1_Y55_PRE_FL"),
        (2017, 2020, "GEN1_Y55_THOR_FL"),
    ],
    ("VOLVO", "XC90"): [
        (2003, 2014, "GEN1_P2"),
        (2015, 2027, "GEN2_SPA"),
    ],
    # --- BMW ---
    ("BMW", "320I"): [
        (2006, 2011, "GEN5_E90"),
        (2012, 2018, "GEN6_F30"),
        (2019, 2027, "GEN7_G20"),
    ],
    ("BMW", "SERIE 3"): [
        (2006, 2011, "GEN5_E90"),
        (2012, 2018, "GEN6_F30"),
        (2019, 2027, "GEN7_G20"),
    ],
    ("BMW", "118I"): [
        (2005, 2011, "GEN1_E87"),
        (2012, 2019, "GEN2_F20_RWD"),
        (2020, 2027, "GEN3_F40_FWD"),
    ],
    ("BMW", "SERIE 1"): [
        (2005, 2011, "GEN1_E87"),
        (2012, 2019, "GEN2_F20_RWD"),
        (2020, 2027, "GEN3_F40_FWD"),
    ],
    ("BMW", "X1"): [
        (2010, 2015, "GEN1_E84"),
        (2016, 2022, "GEN2_F48"),
        (2023, 2027, "GEN3_U11"),
    ],
    ("BMW", "X3"): [
        (2011, 2017, "GEN2_F25"),
        (2018, 2024, "GEN3_G01"),
        (2025, 2027, "GEN4_G45"),
    ],
    # --- MERCEDES-BENZ ---
    ("MERCEDES-BENZ", "A200"): [
        (2013, 2018, "GEN3_W176"),
        (2019, 2027, "GEN4_W177"),
    ],
    ("MERCEDES-BENZ", "CLASE A"): [
        (2013, 2018, "GEN3_W176"),
        (2019, 2027, "GEN4_W177"),
    ],
    ("MERCEDES-BENZ", "C200"): [
        (2008, 2014, "GEN3_W204"),
        (2015, 2021, "GEN4_W205"),
        (2022, 2027, "GEN5_W206"),
    ],
    ("MERCEDES-BENZ", "CLASE C"): [
        (2008, 2014, "GEN3_W204"),
        (2015, 2021, "GEN4_W205"),
        (2022, 2027, "GEN5_W206"),
    ],
    ("MERCEDES-BENZ", "GLA"): [
        (2014, 2020, "GEN1_X156"),
        (2021, 2027, "GEN2_H247"),
    ],
    ("MERCEDES-BENZ", "GLC"): [
        (2016, 2022, "GEN1_X253"),
        (2023, 2027, "GEN2_X254"),
    ],
}

# Nivel Socioeconómico / Poder Adquisitivo por Comuna (Escala 1 a 5)
# 5: Sector Oriente / Alto poder adquisitivo
# 4: Residencial Medio-Alto / Polos urbanos consolidados
# 3: Medio / Capitales regionales principales
# 2: Medio-Bajo / Periferia urbana / Ciudades intermedias
# 1: Rural / Otras comunas (default = 2)
COMUNA_WEALTH_TIER = {
    # Tier 5
    "VITACURA": 5, "LAS CONDES": 5, "LO BARNECHEA": 5, "LA REINA": 5,
    "PROVIDENCIA": 5, "CONCON": 5, "ZAPALLAR": 5, "PUERTO VARAS": 5,
    # Tier 4
    "NUNOA": 4, "ÑUÑOA": 4, "PENALOLEN": 4, "PEÑALOLÉN": 4, "COLINA": 4,
    "CHICUREO": 4, "HUECHURABA": 4, "SAN MIGUEL": 4, "LA FLORIDA": 4,
    "MACUL": 4, "VINA DEL MAR": 4, "VIÑA DEL MAR": 4, "CONCEPCION": 4,
    "CONCEPCIÓN": 4, "SAN PEDRO DE LA PAZ": 4, "ANTOFAGASTA": 4, "MACHALI": 4,
    "TEMUCO": 4, "LA SERENA": 4, "RANCAGUA": 4, "CHILLAN": 4, "VALDIVIA": 4,
    # Tier 3
    "SANTIAGO": 3, "MAIPU": 3, "MAIPÚ": 3, "PUENTE ALTO": 3, "QUILICURA": 3,
    "ESTACION CENTRAL": 3, "INDEPENDENCIA": 3, "RECOLETA": 3, "SAN BERNARDO": 3,
    "BUIN": 3, "PAINE": 3, "TALAGANTE": 3, "MELIPILLA": 3, "QUILPUE": 3,
    "QUILPUÉ": 3, "VILLA ALEMANA": 3, "VALPARAISO": 3, "VALPARAÍSO": 3,
    "COQUIMBO": 3, "TALCA": 3, "CURICO": 3, "CURICÓ": 3, "OSORNO": 3,
    "PUERTO MONTT": 3, "IQUIQUE": 3, "CALAMA": 3, "COPIAPO": 3, "PUNTA ARENAS": 3,
    # Tier 2
    "PUDAHUEL": 2, "CERRILLOS": 2, "EL BOSQUE": 2, "LA CISTERNA": 2,
    "CONCHALI": 2, "CONCHALÍ": 2, "QUINTA NORMAL": 2, "PRADO": 2,
    "RENCA": 2, "SAN JOAQUIN": 2, "LA GRANJA": 2, "LA PINTANA": 2,
    "CERRO NAVIA": 2, "LO ESPEJO": 2, "PEDRO AGUIRRE CERDA": 2, "SAN RAMON": 2,
    "CORONEL": 2, "TALCAHUANO": 2, "LOS ANGELES": 2, "OVALLE": 2, "ARICA": 2,
}
DEFAULT_COMUNA_TIER = 2

# Keywords de NLP Básico sobre Título y Descripción
NLP_FEATURE_PATTERNS = {
    "nlp_unico_dueno": [
        r"unico\s+dueno", r"único\s+dueño", r"1\s+dueno", r"1\s+dueño",
        r"primer\s+dueno", r"primer\s+dueño", r"unica\s+duena", r"única\s+dueña",
    ],
    "nlp_mantenciones_al_dia": [
        r"mantenciones\s+al\s+dia", r"mantenciones\s+al\s+día",
        r"mantencion\s+al\s+dia", r"mantenciones\s+en\s+la\s+marca",
        r"mantenciones\s+concesionario", r"papeles\s+al\s+dia", r"todo\s+al\s+dia",
    ],
    "nlp_cuero": [
        r"\bcuero\b", r"asientos\s+de\s+cuero", r"tapiz\s+de\s+cuero", r"ecocuero",
    ],
    "nlp_sellado_inmaculado": [
        r"\bsellado\b", r"\binmaculado\b", r"como\s+nuevo", r"impecable",
        r"sin\s+detalles", r"flamante", r"igual\s+a\s+nuevo",
    ],
    "nlp_techo_panoramico": [
        r"\bsunroof\b", r"techo\s+panoramico", r"techo\s+panorámico", r"techo\s+electrico",
    ],
    "nlp_full_equipo": [
        r"full\s+equipo", r"top\s+de\s+linea", r"top\s+de\s+línea", r"version\s+full",
    ],
    "nlp_urgencia_conversable": [
        r"vende\s+por\s+apuro", r"por\s+viaje", r"conversable", r"precio\s+conversable",
        r"liquido", r"oferta",
    ],
    "nlp_laminas_seguridad": [
        r"laminas\s+de\s+seguridad", r"láminas\s+de\s+seguridad", r"lamina\s+de\s+seguridad",
        r"vidrios\s+blindados", r"polarizado\s+certificado",
    ],
    "nlp_neumaticos_nuevos": [
        r"neumaticos\s+nuevos", r"neumáticos\s+nuevos", r"neumaticos\s+recien\s+cambiados",
        r"llantas\s+y\s+neumaticos\s+nuevos",
    ],
}

# ==============================================================================
# 5. HIPERPARÁMETROS DE MACHINE LEARNING (MÓDULO 3)
# ==============================================================================
MODEL_RETRAIN_DAYS = 7                 # Re-entrenar automáticamente cada 7 días
RANDOM_SEED = 42
TEST_SPLIT_SIZE = 0.15

LGBM_PARAMS = {
    "objective": "regression_l1",       # MAE/Huber-style robustez frente a residuos extremos
    "metric": "rmse",
    "boosting_type": "gbdt",
    "n_estimators": 600,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "max_depth": 7,
    "min_child_samples": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# ==============================================================================
# 6. SISTEMA DE ARBITRAJE Y ALERTAS TELEGRAM (MÓDULO 4)
# ==============================================================================
# Si el precio publicado es al menos un 15% menor al predicho -> OPORTUNIDAD
OPPORTUNITY_DISCOUNT_THRESHOLD = 0.15  # 15% de descuento mínimo frente al Precio Justo IA

# Si el descuento supera el 55% (ej. vale 10M y está a < 4.5M), se clasifica como
# "POSIBLE ESTAFA / ERROR DE TIPEO / PRENDA / CHOCADO" en alerta moderada separada.
SUSPICIOUS_DISCOUNT_THRESHOLD = 0.55   # 55% o más de descuento es sospechoso

# Credenciales de Telegram y buscadores (leídas desde '.env', variables de entorno o st.secrets)
TELEGRAM_BOT_TOKEN = _get_secret("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = _get_secret("TELEGRAM_CHAT_ID", "")
TELEGRAM_API_TIMEOUT = 15

# (Opcional) Credenciales de Google Custom Search JSON API o Serper.dev (Google Search API)
# para consultar fichas de Chileautos indexadas por Google sin tocar su firewall
GOOGLE_CSE_API_KEY = _get_secret("GOOGLE_CSE_API_KEY", "")
GOOGLE_CSE_CX = _get_secret("GOOGLE_CSE_CX", "")
SERPER_API_KEY = _get_secret("SERPER_API_KEY", "")
# Páginas de 10 resultados por consulta (cada página consume 1 crédito de Serper)
SERPER_PAGES_PER_QUERY = int(_get_secret("SERPER_PAGES_PER_QUERY", "2") or "2")
