"""
app.py - AutoPulse IA Chile: Suite SaaS Multi-Portal de Arbitraje Automotriz.
Integra:
- Caché desacoplado de mercado (@st.cache_resource) y favoritos/CRM por espacio (@st.cache_data).
- Sistema de diseño SaaS responsivo (ui.theme, ui.domain, ui.components).
- Servicios comerciales y financieros netos, PDF, Auth/Workspaces y Tendencias (services.*).
- 5 vistas comerciales aisladas con @st.fragment: Oportunidades, Pipeline CRM, Mercado, Tasador IA e Inventario.
"""

from contextlib import closing
from datetime import datetime
import html
import importlib
from pathlib import Path
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from database import CRM_STAGES, DatabaseManager
from pipeline import CarDataPipeline
from services import (
    calculate_net_deal_metrics,
    check_scan_cooldown,
    compute_market_executive_summary,
    enrich_dataframe_with_net_deal_and_confidence,
    evaluate_radar_matches,
    generate_valuation_pdf_bytes,
    get_current_workspace_id,
    is_admin_unlocked,
    lock_admin,
    record_scan_triggered,
    sanitize_user_text,
    set_current_workspace_id,
    verify_admin_code,
)
from train import CarPricePredictor, HierarchicalCarEncoder
from ui.components import (
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
    format_clp,
    get_vehicle_emoji,
    normalize_model_input,
    resolve_portal_info,
    resolve_region,
)
from ui.theme import (
    THEME_MODES,
    get_plotly_template,
    inject_theme_css,
    is_light_theme,
)

# Compatibilidad de deserialización para joblib en el namespace de Streamlit
setattr(sys.modules["__main__"], "HierarchicalCarEncoder", HierarchicalCarEncoder)

# ==============================================================================
# 1. CONFIGURACIÓN DE PÁGINA, NAVEGACIÓN Y ESTADO DE SESIÓN
# ==============================================================================
st.set_page_config(
    page_title="AutoPulse IA Chile · Inteligencia de Arbitraje Automotriz",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

MAIN_SECTIONS: List[str] = [
    "Oportunidades",
    "Pipeline CRM",
    "Mercado",
    "Tasador IA",
    "Inventario",
]

CRM_SUBVIEWS: List[str] = [
    "Seguimiento CRM",
    "Radares de Alerta",
]

MARKET_SUBVIEWS: List[str] = [
    "Curva 2D",
    "Nube 3D",
    "Por Generación",
    "Mapa de Calor",
    "Panorama Ejecutivo",
]

SORT_OPTIONS_LABELS: List[str] = [
    "Mayor Margen Neto",
    "Mayor ROI Neto",
    "Mayor Confianza IA",
    "Menor Precio",
    "Mayor % Dcto",
]

_WATCHED_BACKEND_MODULES = (
    "config",
    "database",
    "pipeline",
    "train",
    "monitor",
    "scraper",
    "main",
)


def _init_session_defaults() -> None:
    """Inicializa las claves mínimas de estado en sesión sin colisionar con widgets."""
    defaults: Dict[str, Any] = {
        "focused_marca": None,
        "focused_modelo": None,
        "last_scan_message": None,
        "_fav_nonce": 0,
        "_radar_nonce": 0,
        "_segments_nonce": 0,
        "_db_cache_nonce": 0,
        "_status_overrides": {},
        "_spec_overrides": {},
        "auth_workspace_id": "default",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _queue_toast(message: str, icon: Optional[str] = None) -> None:
    """Encola un toast para que sobreviva a reruns inmediatos."""
    st.session_state["_pending_toast"] = (str(message), icon)


def _flush_pending_toast() -> None:
    """Muestra cualquier notificación toast encolada tras un rerun."""
    pending = st.session_state.pop("_pending_toast", None)
    if isinstance(pending, tuple) and len(pending) == 2:
        msg, ico = pending
        if ico:
            st.toast(msg, icon=ico)
        else:
            st.toast(msg)


def _safe_rerun_fragment() -> None:
    """Re-ejecuta el fragmento activo o hace fallback seguro a rerun completo si no hay fragmento en curso."""
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

        ctx = get_script_run_ctx(suppress_warning=True)
        if ctx is not None and getattr(ctx, "fragment_ids_this_run", None):
            _safe_rerun_fragment()
            return
    except st.errors.StreamlitAPIException:
        pass
    st.rerun(scope="app")


def _fmt_int(value: Any) -> str:
    """Formatea enteros con separador de miles chileno."""
    try:
        return f"{int(round(float(value))):,}".replace(",", ".")
    except Exception:
        return "0"


@st.cache_resource(show_spinner=False)
def _get_cached_db_manager(db_path_str: str, database_url: str) -> DatabaseManager:
    """Instancia singleton cacheada de DatabaseManager para evitar init_db() repetido."""
    return DatabaseManager(db_path=Path(db_path_str), database_url=database_url)


def _get_db_manager() -> DatabaseManager:
    """Retorna el gestor de base de datos cacheado respetando DATABASE_URL o DB_PATH."""
    db_url = str(getattr(config, "DATABASE_URL", "") or "").strip()
    return _get_cached_db_manager(str(Path(config.DB_PATH).resolve()), db_url)


def _get_watched_module_mtimes() -> Dict[str, float]:
    """Almacena mtimes de módulos backend en sys para sobrevivir a limpiezas de caché."""
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


def _get_fresh_orchestrator() -> Any:
    """Carga diferida del orquestador y recarga módulos modificados en disco."""
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


# ==============================================================================
# 2. SALUD DEL SISTEMA, CACHÉ DESACOPLADA Y ANULACIONES EN MEMORIA
# ==============================================================================
@st.cache_data(show_spinner=False, ttl=120)
def _check_system_health_cached(
    db_url_active: bool,
    db_mtime: float,
    model_exists: bool,
    model_mtime: float,
) -> Tuple[bool, str]:
    """Ejecuta y cachea la verificación de salud cuando el sistema está operativo."""
    _ = (db_mtime, model_mtime)
    if db_url_active:
        try:
            stats = _get_db_manager().get_summary_stats()
            if int(stats.get("total_historicos") or 0) == 0:
                return False, "El repositorio de mercado en la nube aún no tiene publicaciones registradas."
        except Exception:
            return False, "No se pudo conectar al servicio de datos en la nube. Verifica tu configuración."
    else:
        if not config.DB_PATH.exists():
            return False, "El repositorio local de vehículos aún no ha sido inicializado."

        try:
            with closing(sqlite3.connect(str(config.DB_PATH))) as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='autos_historico'"
                )
                if cur.fetchone() is None:
                    return False, "La estructura inicial del repositorio de datos aún no está creada."
                cur.execute("SELECT COUNT(*) FROM autos_historico")
                if int(cur.fetchone()[0] or 0) == 0:
                    return False, "El repositorio local de datos está vacío (0 vehículos registrados)."
        except Exception:
            return False, "No se pudo verificar el estado del repositorio local de vehículos."

    if not model_exists:
        return False, "El modelo predictivo IA aún no ha sido entrenado."

    return True, "OK"


def check_system_health() -> Tuple[bool, str]:
    """Verifica la disponibilidad del repositorio de datos y del modelo IA sin exponer rutas."""
    db_url_active = bool(str(getattr(config, "DATABASE_URL", "") or "").strip())
    db_mtime = (
        0.0
        if db_url_active
        else (config.DB_PATH.stat().st_mtime if config.DB_PATH.exists() else -1.0)
    )
    model_exists = config.MODEL_PATH.exists()
    model_mtime = config.MODEL_PATH.stat().st_mtime if model_exists else -1.0
    ok, msg = _check_system_health_cached(db_url_active, db_mtime, model_exists, model_mtime)
    if not ok:
        _check_system_health_cached.clear()
    return ok, msg


@st.cache_resource(show_spinner=False)
def load_cached_predictor(model_mtime: float) -> Tuple[CarPricePredictor, Dict[str, Any]]:
    """Carga el modelo LightGBM serializado y devuelve (predictor, metadata)."""
    _ = model_mtime
    setattr(sys.modules["__main__"], "HierarchicalCarEncoder", HierarchicalCarEncoder)
    predictor = CarPricePredictor(model_path=config.MODEL_PATH)
    try:
        metadata = predictor.load_artifact()
    except Exception:
        metadata = predictor.train_and_evaluate()
    return predictor, metadata


def _enrich_with_ia_and_portal(
    df_in: pd.DataFrame,
    pipeline: CarDataPipeline,
    predictor: CarPricePredictor,
    only_valid_predict: bool = False,
) -> pd.DataFrame:
    """Aplica ingeniería de características, portales/regiones e inferencia IA sobre copia."""
    if df_in.empty:
        return df_in.copy()

    df_out = pipeline.engineer_features(df_in)
    df_out = assign_portal_and_region_columns(df_out)
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
    disc_vals = df_out["descuento_pct"].to_numpy(dtype=float)
    df_out["categoria_arbitraje"] = np.select(
        [
            disc_vals >= config.SUSPICIOUS_DISCOUNT_THRESHOLD,
            disc_vals >= config.OPPORTUNITY_DISCOUNT_THRESHOLD,
        ],
        ["POSIBLE ESTAFA / ERROR DE TIPEO", "OPORTUNIDAD"],
        default="PRECIO DE MERCADO",
    )
    return df_out


@st.cache_resource(show_spinner=False, ttl=120)
def load_market_datasets(
    db_mtime: float,
    model_mtime: float,
    clean_csv_mtime: float,
) -> Dict[str, Any]:
    """
    Carga datasets de mercado en una sola pasada sin acoplarse a favoritos de usuario
    ni a reportes CSV heredados. Los consumidores no deben mutar los DataFrames sin copiar.
    """
    _ = (db_mtime, model_mtime, clean_csv_mtime)
    db_manager = _get_db_manager()
    pipeline = CarDataPipeline(db_path=config.DB_PATH)
    predictor, model_meta = load_cached_predictor(model_mtime)

    today_prefix = datetime.now().strftime("%Y-%m-%d")
    full_df = db_manager.read_dataframe(
        "SELECT * FROM autos_historico ORDER BY fecha_ultima_vista DESC"
    )
    try:
        price_history_df = db_manager.read_dataframe(
            """
            SELECT ph.id, ph.id_aviso, a.marca, a.modelo, a.anio, a.kilometraje,
                   a.comuna, a.estado, ph.precio_antiguo, ph.precio_nuevo,
                   (ph.precio_nuevo - ph.precio_antiguo) AS variacion_clp,
                   ph.fecha_cambio, a.url
            FROM precio_historial ph
            LEFT JOIN autos_historico a ON ph.id_aviso = a.id_aviso
            ORDER BY ph.fecha_cambio DESC
            LIMIT 400
            """
        )
        if not price_history_df.empty:
            price_history_df["variacion_pct"] = np.where(
                pd.to_numeric(price_history_df["precio_antiguo"], errors="coerce").fillna(0.0) > 0,
                (
                    (
                        pd.to_numeric(price_history_df["variacion_clp"], errors="coerce").fillna(0.0)
                        / pd.to_numeric(price_history_df["precio_antiguo"], errors="coerce").replace(0, np.nan)
                    )
                    * 100.0
                ).fillna(0.0).round(1),
                0.0,
            )
            price_history_df = assign_portal_and_region_columns(price_history_df)
    except Exception:
        price_history_df = pd.DataFrame()

    sold_today = db_manager.get_sold_today_count(today_prefix)
    summary_stats = db_manager.get_summary_stats()
    summary_stats["vendidos_hoy"] = sold_today
    summary_stats["duplicados_multiplataforma"] = 0

    if not full_df.empty:
        full_enriched = _enrich_with_ia_and_portal(
            full_df, pipeline, predictor, only_valid_predict=True
        )
        full_enriched["multi_portal_info"] = "Publicación única"

        active_raw = full_enriched[
            (full_enriched["estado"] == "activo")
            & (full_enriched["precio"] >= config.MIN_VALID_PRICE_CLP)
        ].copy()
        active_featured, merged_dups = CarDataPipeline.deduplicate_cross_platform_df(active_raw)
        summary_stats["duplicados_multiplataforma"] = int(merged_dups)

        if not active_featured.empty and merged_dups > 0:
            merged_map = (
                active_featured.loc[
                    active_featured["multi_portal_info"] != "Publicación única",
                    ["id_aviso", "multi_portal_info"],
                ]
                .set_index("id_aviso")["multi_portal_info"]
                .to_dict()
            )
            if merged_map:
                full_enriched["multi_portal_info"] = (
                    full_enriched["id_aviso"].map(merged_map).fillna("Publicación única")
                )
    else:
        full_enriched = pd.DataFrame()
        active_featured = pd.DataFrame()

    discarded_id_set = (
        set(full_enriched.loc[full_enriched["estado"] == "descartado", "id_aviso"].astype(str))
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
                clean_df = _enrich_with_ia_and_portal(raw_clean, pipeline, predictor)
    elif clean_csv_path.exists():
        raw_clean = pd.read_csv(clean_csv_path, encoding="utf-8")
        if discarded_id_set:
            raw_clean = raw_clean[~raw_clean["id_aviso"].astype(str).isin(discarded_id_set)]
        clean_df = _enrich_with_ia_and_portal(raw_clean, pipeline, predictor)
    elif not active_featured.empty:
        clean_df, _ = pipeline.filter_outliers(active_featured)
    else:
        clean_df = pd.DataFrame()

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

    return {
        "summary": summary_stats,
        "portal_counts": portal_counts,
        "model_meta": model_meta,
        "full_df": full_enriched,
        "price_history_df": price_history_df,
        "active_featured": active_featured,
        "clean_df": clean_df,
        "tree_df": tree_df,
    }


def _assign_override_field(df_target: pd.DataFrame, mask: pd.Series, col: str, val: Any) -> None:
    """Asigna un valor en DataFrame respetando el dtype de la columna en pandas 3."""
    if col not in df_target.columns or not mask.any():
        return
    dtype = df_target[col].dtype
    try:
        if pd.api.types.is_integer_dtype(dtype):
            df_target.loc[mask, col] = int(round(float(val)))
        elif pd.api.types.is_float_dtype(dtype):
            df_target.loc[mask, col] = float(val)
        else:
            df_target.loc[mask, col] = str(val)
    except Exception:
        df_target[col] = df_target[col].astype(object)
        df_target.loc[mask, col] = val


def apply_session_overrides(data_bundle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aplica en memoria las anulaciones rápidas de sesión (_status_overrides y _spec_overrides)
    sin invalidar load_market_datasets, manteniendo respuestas <100ms.
    """
    status_ov: Dict[str, str] = st.session_state.get("_status_overrides") or {}
    spec_ov: Dict[str, Dict[str, Any]] = st.session_state.get("_spec_overrides") or {}
    if not status_ov and not spec_ov:
        return data_bundle

    out = dict(data_bundle)
    full_df = data_bundle["full_df"].copy() if not data_bundle["full_df"].empty else data_bundle["full_df"]
    active_df = (
        data_bundle["active_featured"].copy()
        if not data_bundle["active_featured"].empty
        else data_bundle["active_featured"]
    )
    clean_df = data_bundle["clean_df"].copy() if not data_bundle["clean_df"].empty else data_bundle["clean_df"]

    if spec_ov:
        for df_target in (full_df, active_df, clean_df):
            if df_target.empty or "id_aviso" not in df_target.columns:
                continue
            id_series = df_target["id_aviso"].astype(str)
            for rid, fields in spec_ov.items():
                mask = id_series == str(rid)
                if mask.any():
                    for col, val in fields.items():
                        _assign_override_field(df_target, mask, col, val)

    if status_ov and not full_df.empty and "id_aviso" in full_df.columns:
        full_ids = full_df["id_aviso"].astype(str)
        for rid, new_st in status_ov.items():
            _assign_override_field(full_df, full_ids == str(rid), "estado", new_st)

        removed_ids = {str(rid) for rid, st_val in status_ov.items() if st_val != "activo"}
        restored_ids = {str(rid) for rid, st_val in status_ov.items() if st_val == "activo"}

        if removed_ids:
            if not active_df.empty:
                active_df = active_df[~active_df["id_aviso"].astype(str).isin(removed_ids)].copy()
            if not clean_df.empty:
                clean_df = clean_df[~clean_df["id_aviso"].astype(str).isin(removed_ids)].copy()

        if restored_ids:
            existing_active = (
                set(active_df["id_aviso"].astype(str)) if not active_df.empty else set()
            )
            missing_active = restored_ids - existing_active
            if missing_active:
                add_rows = full_df[
                    full_df["id_aviso"].astype(str).isin(missing_active)
                    & (pd.to_numeric(full_df["precio"], errors="coerce").fillna(0) >= config.MIN_VALID_PRICE_CLP)
                ].copy()
                if not add_rows.empty:
                    active_df = (
                        pd.concat([add_rows, active_df], ignore_index=True)
                        if not active_df.empty
                        else add_rows
                    )
                    existing_clean = (
                        set(clean_df["id_aviso"].astype(str)) if not clean_df.empty else set()
                    )
                    add_clean = add_rows[~add_rows["id_aviso"].astype(str).isin(existing_clean)]
                    if not add_clean.empty:
                        clean_df = (
                            pd.concat([add_clean, clean_df], ignore_index=True)
                            if not clean_df.empty
                            else add_clean
                        )

    summary = dict(data_bundle.get("summary", {}))
    if not full_df.empty and "estado" in full_df.columns:
        estados_lower = full_df["estado"].fillna("").astype(str).str.lower()
        summary["activos"] = int((estados_lower == "activo").sum())
        summary["vendidos"] = int((estados_lower == "vendido").sum())

    if not clean_df.empty:
        out["tree_df"] = (
            clean_df.groupby(["marca", "modelo", "generacion"], as_index=False)
            .agg(
                unidades=("id_aviso", "count"),
                precio_medio=("precio", "mean"),
                descuento_max_pct=("descuento_pct", lambda s: round(float(s.max()) * 100.0, 1)),
            )
        )
    else:
        out["tree_df"] = pd.DataFrame()

    out["full_df"] = full_df
    out["active_featured"] = active_df
    out["clean_df"] = clean_df
    out["summary"] = summary
    out["portal_counts"] = (
        active_df["portal"].value_counts().to_dict() if not active_df.empty else {}
    )
    return out


@st.cache_data(show_spinner=False, ttl=120)
def _get_cached_target_segments(segments_nonce: int) -> List[Tuple[str, str]]:
    """Cachea el catálogo de marcas y modelos para evitar consultas de red en reruns."""
    _ = segments_nonce
    return config.get_all_target_segments()


@st.cache_data(show_spinner=False, ttl=30)
def load_user_favorites(workspace_id: str, fav_nonce: int) -> Dict[str, Dict[str, Any]]:
    """Carga los favoritos y pipeline CRM del workspace activo sin recargar el inventario."""
    _ = fav_nonce
    return _get_db_manager().get_favorites_map(owner_id=workspace_id)


@st.cache_data(show_spinner=False, ttl=30)
def load_user_radars(workspace_id: str, radar_nonce: int) -> List[Dict[str, Any]]:
    """Carga los radares de alerta configurados para el workspace activo."""
    _ = radar_nonce
    return _get_db_manager().list_radar_alerts(owner_id=workspace_id)


@st.cache_data(show_spinner=False, ttl=30)
def load_listing_price_history_cached(id_aviso: str, db_mtime: float) -> List[Dict[str, Any]]:
    """Consulta el historial de variaciones de precio de un aviso específico."""
    _ = db_mtime
    return _get_db_manager().get_listing_price_history(id_aviso)


def _get_current_mtimes() -> Tuple[float, float, float]:
    """Obtiene marcas de modificación para llaves de caché desacopladas."""
    db_mtime = (
        float(st.session_state.get("_db_cache_nonce", 0.0))
        if config.DATABASE_URL
        else config.DB_PATH.stat().st_mtime
    )
    model_mtime = config.MODEL_PATH.stat().st_mtime if config.MODEL_PATH.exists() else 0.0
    clean_csv_path = config.DATA_DIR / "autos_clean.csv"
    clean_mtime = clean_csv_path.stat().st_mtime if clean_csv_path.exists() else 0.0
    return db_mtime, model_mtime, clean_mtime


def _clear_all_market_caches() -> None:
    """Limpia explícitamente cachés de mercado y anulaciones de sesión tras rescans/refresh."""
    st.session_state["_status_overrides"] = {}
    st.session_state["_spec_overrides"] = {}
    st.session_state["_db_cache_nonce"] = float(st.session_state.get("_db_cache_nonce", 0.0)) + 1.0
    st.session_state["_fav_nonce"] = int(st.session_state.get("_fav_nonce", 0)) + 1
    st.session_state["_radar_nonce"] = int(st.session_state.get("_radar_nonce", 0)) + 1
    st.session_state["_segments_nonce"] = int(st.session_state.get("_segments_nonce", 0)) + 1
    load_market_datasets.clear()
    load_user_favorites.clear()
    load_user_radars.clear()
    load_listing_price_history_cached.clear()
    _get_cached_target_segments.clear()
    _check_system_health_cached.clear()


# ==============================================================================
# 3. ACCIONES INSTANTÁNEAS (<100ms), PDF Y ORQUESTACIÓN DE ESCANEOS
# ==============================================================================
def _toggle_favorite_instant(id_aviso: str, workspace_id: str, label: str = "") -> bool:
    """Alterna un vehículo en Pipeline CRM y solo incrementa _fav_nonce."""
    is_fav = _get_db_manager().toggle_favorite(id_aviso, owner_id=workspace_id)
    st.session_state["_fav_nonce"] = int(st.session_state.get("_fav_nonce", 0)) + 1
    display = label or id_aviso
    if is_fav:
        _queue_toast(f"{display} guardado en Pipeline CRM.", icon="⭐")
    else:
        _queue_toast(f"{display} removido de Pipeline CRM.", icon="🗑️")
    return is_fav


def _add_favorite_if_missing(id_aviso: str, workspace_id: str, label: str = "") -> bool:
    """Guarda un vehículo en Pipeline CRM si aún no está guardado en el espacio."""
    fav_nonce = int(st.session_state.get("_fav_nonce", 0))
    fav_map = load_user_favorites(workspace_id, fav_nonce)
    clean_id = str(id_aviso or "").strip()
    if not clean_id:
        return False
    if clean_id in fav_map:
        _queue_toast(f"{label or clean_id} ya está en Pipeline CRM.", icon="💛")
        return False
    _get_db_manager().toggle_favorite(clean_id, owner_id=workspace_id)
    st.session_state["_fav_nonce"] = fav_nonce + 1
    _queue_toast(f"{label or clean_id} agregado a Pipeline CRM.", icon="⭐")
    return True


def _update_listing_status_instant(id_aviso: str, new_status: str, label: str) -> bool:
    """Actualiza el estado en BD y aplica anulación instantánea en memoria."""
    ok = _get_db_manager().update_listing_status(id_aviso, new_status)
    if ok:
        overrides = dict(st.session_state.get("_status_overrides") or {})
        overrides[str(id_aviso)] = str(new_status).strip().lower()
        st.session_state["_status_overrides"] = overrides
        if new_status == "activo":
            st.session_state.pop("last_dismissed_action", None)
            _queue_toast(f"{label} restaurado a inventario activo.", icon="↩️")
        else:
            st.session_state["last_dismissed_action"] = {
                "id_aviso": str(id_aviso),
                "status": new_status,
                "label": label,
            }
            tag = "vendido" if new_status == "vendido" else "descartado"
            _queue_toast(f"{label} marcado como {tag}.", icon="🗂️")
    return ok


def _update_listing_specs_instant(
    row: pd.Series,
    kilometraje: int,
    transmision: str,
    combustible: str,
    anio: int,
    precio: int,
) -> bool:
    """Actualiza especificaciones en BD y recalcula valoración IA en memoria para esa fila."""
    raw_id = str(row["id_aviso"])
    ok = _get_db_manager().update_listing_specs(
        id_aviso=raw_id,
        kilometraje=int(kilometraje),
        transmision=transmision,
        combustible=combustible,
        anio=int(anio),
        precio=int(precio),
    )
    if not ok:
        return False

    _, model_mtime, _ = _get_current_mtimes()
    predictor, _ = load_cached_predictor(model_mtime)
    pipeline = CarDataPipeline(db_path=config.DB_PATH)

    row_dict = row.to_dict()
    row_dict["kilometraje"] = int(kilometraje)
    row_dict["transmision"] = transmision
    row_dict["combustible"] = combustible
    row_dict["anio"] = int(anio)
    row_dict["precio"] = int(precio)
    single_df = pd.DataFrame([row_dict])
    enriched_one = _enrich_with_ia_and_portal(single_df, pipeline, predictor)
    r0 = enriched_one.iloc[0]

    spec_overrides = dict(st.session_state.get("_spec_overrides") or {})
    spec_overrides[raw_id] = {
        "kilometraje": int(r0["kilometraje"]),
        "transmision": str(r0["transmision"]),
        "combustible": str(r0["combustible"]),
        "anio": int(r0["anio"]),
        "precio": int(r0["precio"]),
        "generacion": str(r0["generacion"]),
        "antiguedad": int(r0["antiguedad"]),
        "km_por_anio": float(r0["km_por_anio"]),
        "precio_justo_ia": float(r0["precio_justo_ia"]),
        "margen_ganancia_clp": float(r0["margen_ganancia_clp"]),
        "descuento_pct": float(r0["descuento_pct"]),
        "roi_pct": float(r0["roi_pct"]),
        "categoria_arbitraje": str(r0["categoria_arbitraje"]),
        "datos_corregidos": 1,
    }
    st.session_state["_spec_overrides"] = spec_overrides
    km_str = _fmt_int(kilometraje)
    st.session_state["last_spec_edit_msg"] = (
        f"Ficha de **{row['marca']} {row['modelo']} ({int(anio)})** actualizada: "
        f"**{km_str} km · {transmision} ({combustible})**. Valoración IA recalculada."
    )
    _queue_toast("Ficha corregida y valoración IA recalculada.", icon="✅")
    return True


def _build_comparables_for_row(row: pd.Series, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Extrae hasta 10 comparables reales cercanos en año y kilometraje."""
    if clean_df.empty:
        return pd.DataFrame()
    marca_u = str(row.get("marca") or "").strip().upper()
    modelo_u = str(row.get("modelo") or "").strip().upper()
    anio_i = int(row.get("anio") or 2020)
    km_i = int(row.get("kilometraje") or 60_000)
    raw_id = str(row.get("id_aviso") or "")

    same_model = clean_df[
        (clean_df["marca"].astype(str).str.upper() == marca_u)
        & (clean_df["modelo"].astype(str).str.upper() == modelo_u)
    ]
    if raw_id and "id_aviso" in same_model.columns:
        same_model = same_model[same_model["id_aviso"].astype(str) != raw_id]
    if same_model.empty:
        return pd.DataFrame()

    comp = same_model[same_model["anio"].sub(anio_i).abs() <= 2].copy()
    if len(comp) < 3:
        comp = same_model.copy()
    comp = comp.assign(
        _dist=(comp["anio"] - anio_i).abs() * 15_000 + (comp["kilometraje"] - km_i).abs()
    ).nsmallest(10, "_dist")
    return comp.drop(columns=["_dist"], errors="ignore")


def _build_pdf_bytes_for_row(
    row: pd.Series,
    clean_df: pd.DataFrame,
    deal_costs: Dict[str, Any],
    brand_name: str = "AutoPulse AI Chile",
) -> bytes:
    """Genera el dossier PDF ejecutivo de una publicación específica."""
    vehicle_dict = row.to_dict()
    p_pub = float(row.get("precio") or 0.0)
    p_ia = float(row.get("precio_justo_ia") or 0.0)
    metrics = calculate_net_deal_metrics(p_pub, p_ia, costs=deal_costs)
    comp_df = _build_comparables_for_row(row, clean_df)
    clean_brand = sanitize_user_text(brand_name, max_len=60) or "AutoPulse AI Chile"
    return generate_valuation_pdf_bytes(
        vehicle=vehicle_dict,
        deal_metrics=metrics,
        comparables_df=comp_df,
        brand_name=clean_brand,
    )


def _render_quick_crm_save_bar(
    df_candidates: pd.DataFrame,
    workspace_id: str,
    key_prefix: str,
    heading: str = "Guardar unidad destacada en Pipeline CRM",
) -> None:
    """Selector rápido reutilizable para enviar unidades destacadas directamente a Pipeline CRM."""
    if df_candidates.empty or "id_aviso" not in df_candidates.columns:
        return

    work = df_candidates.copy()
    if "descuento_pct" in work.columns:
        work = work.sort_values(by="descuento_pct", ascending=False)
    work = work.head(40)

    options_map: Dict[str, Tuple[str, str]] = {}
    for _, r in work.iterrows():
        rid = str(r.get("id_aviso") or "").strip()
        if not rid:
            continue
        portal_name, _, _ = resolve_portal_info(rid)
        dcto = float(r.get("descuento_pct", 0.0)) * 100.0
        dcto_txt = f"-{dcto:.1f}%" if dcto >= 0 else f"+{abs(dcto):.1f}%"
        lbl = (
            f"{r.get('marca', '')} {r.get('modelo', '')} ({int(r.get('anio') or 0)}) · "
            f"{format_clp(r.get('precio', 0))} ({dcto_txt} IA) · {portal_name} [{rid}]"
        )
        short_lbl = f"{r.get('marca', '')} {r.get('modelo', '')} ({int(r.get('anio') or 0)})"
        options_map[lbl] = (rid, short_lbl)

    if not options_map:
        return

    opt_keys = list(options_map.keys())
    sel_key = f"{key_prefix}_quick_crm_sel"
    if sel_key in st.session_state and st.session_state[sel_key] not in opt_keys:
        st.session_state[sel_key] = opt_keys[0]

    st.markdown(f"##### ⭐ {heading}")
    qc1, qc2 = st.columns([3.4, 1.1])
    with qc1:
        selected_lbl = st.selectbox(
            "Selecciona una unidad:",
            options=opt_keys,
            label_visibility="collapsed",
            key=sel_key,
            **({} if sel_key in st.session_state else {"index": 0}),
        )
    with qc2:
        if st.button("⭐ Guardar en CRM", key=f"{key_prefix}_quick_crm_btn", width="stretch"):
            rid, short_lbl = options_map[selected_lbl]
            _add_favorite_if_missing(rid, workspace_id, short_lbl)
            _safe_rerun_fragment()


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
    status_box: Any = None,
) -> Tuple[bool, str]:
    """Invoca el orquestador multi-portal y actualiza cachés de mercado."""
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

        if status_box is not None:
            status_box.write("1/4 · Preparando orquestador multi-portal y parámetros de búsqueda...")
        orch = _get_fresh_orchestrator()

        if status_box is not None:
            status_box.write(
                "2/4 · Consultando portales en vivo (MercadoLibre, Autocosmos, Yapo y Chileautos)..."
            )
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

        if status_box is not None:
            status_box.write("3/4 · Recalculando valoraciones IA y sincronizando cachés de sesión...")
        _clear_all_market_caches()

        filter_details = []
        if anio_min or anio_max:
            filter_details.append(f"Años: {anio_min or 'min'}..{anio_max or 'max'}")
        if precio_min or precio_max:
            p_lo = format_clp(precio_min) if precio_min else "$ 0"
            p_hi = format_clp(precio_max) if precio_max else "max"
            filter_details.append(f"Precio: {p_lo}–{p_hi}")
        if km_max:
            filter_details.append(f"Km <= {_fmt_int(km_max)}")
        if deep:
            filter_details.append("Búsqueda profunda")
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

        if status_box is not None:
            status_box.write("4/4 · Escaneo completado y tablero actualizado.")

        if marca_clean:
            st.session_state["focused_marca"] = marca_clean
            st.session_state["focused_modelo"] = modelo_clean
            return (
                True,
                f"Búsqueda en vivo completada para {marca_clean} {modelo_clean or 'TODOS'}{details_str}{portal_str}.",
            )
        return (
            True,
            f"Barrido general multi-portal completado{details_str}{portal_str}.",
        )
    except Exception:
        return False, "No se pudo completar el escaneo multi-portal en este momento. Revisa tu conexión e intenta nuevamente."


# ==============================================================================
# 4. PANTALLA DE RESILIENCIA (CUANDO FALTA BASE DE DATOS O MODELO IA)
# ==============================================================================
def render_resilience_onboarding(reason: str) -> None:
    """Muestra pantalla limpia de inicialización cuando falta DB o modelo IA."""
    inject_theme_css(is_light_theme())
    render_hero_header(
        workspace_id=get_current_workspace_id(),
        active_count=0,
        opp_count=0,
    )
    render_empty_state(
        title="Sistema pendiente de inicialización",
        subtitle=reason,
        icon="🚀",
    )

    col_info, col_action = st.columns([2, 1])
    with col_info:
        st.markdown(
            """
            ### Activación del Radar Multi-Portal
            Presiona el botón de activación para ejecutar el ciclo inicial:
            1. **Captura Multi-Portal:** Extrae publicaciones reales desde MercadoLibre, Autocosmos, Yapo y Chileautos.
            2. **Limpieza y Enriquecimiento:** Filtra anomalías de precio, asigna generaciones, nivel comunal y atributos NLP.
            3. **Entrenamiento IA:** Calibra el modelo híbrido LightGBM con codificación jerárquica.
            4. **Detección de Oportunidades:** Identifica vehículos con descuento relevante frente al valor justo de mercado.
            """
        )
    with col_action:
        st.markdown("#### Activación Operativa")
        if not is_admin_unlocked():
            code_in = st.text_input(
                "Código de operador:", type="password", key="onb_admin_code"
            )
            if st.button("🔓 Desbloquear Operador", key="onb_unlock_btn", width="stretch"):
                if verify_admin_code(code_in):
                    _queue_toast("Acceso operador desbloqueado.", icon="🔓")
                    st.rerun()
                else:
                    st.error("Código de operador inválido.")
            return

        if st.button(
            "⚡ Inicializar y Ejecutar Primer Escaneo",
            type="primary",
            width="stretch",
            key="onb_init_btn",
        ):
            allowed, wait_sec = check_scan_cooldown()
            if not allowed:
                st.warning(f"Enfriamiento activo: espera {wait_sec}s antes de reintentar.")
                return
            record_scan_triggered()
            with st.status("Inicializando sistema multi-portal e IA...", expanded=True) as status_box:
                ok, msg = execute_backend_cycle(
                    force_train=True, pages_per_portal=2, status_box=status_box
                )
                status_box.update(
                    label="Sistema inicializado correctamente" if ok else "Error en inicialización",
                    state="complete" if ok else "error",
                    expanded=False,
                )
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


# ==============================================================================
# 5. BARRA LATERAL LIMPIA Y MULTI-ESPACIO
# ==============================================================================
def render_sidebar(
    data_bundle: Dict[str, Any],
    workspace_id: str,
    favorites_map: Dict[str, Dict[str, Any]],
) -> None:
    """Renderiza el sidebar SaaS: tema, workspace, resumen de mercado, catálogo y operador."""
    summary = data_bundle["summary"]
    portal_counts = data_bundle["portal_counts"]
    model_meta = data_bundle["model_meta"]
    metrics = model_meta.get("metricas_test_80_20", {})
    clean_df = data_bundle["clean_df"]
    favorites_count = len(favorites_map)

    raw_train_date = model_meta.get("fecha_entrenamiento", "")
    try:
        formatted_train_date = datetime.fromisoformat(raw_train_date).strftime("%d/%m/%Y %H:%M")
    except Exception:
        if config.MODEL_PATH.exists():
            formatted_train_date = datetime.fromtimestamp(
                config.MODEL_PATH.stat().st_mtime
            ).strftime("%d/%m/%Y %H:%M")
        else:
            formatted_train_date = "No disponible"

    with st.sidebar:
        st.markdown("## 🚘 AutoPulse IA")
        st.caption("Inteligencia Automotriz Multi-Portal · Chile")

        theme_kwargs = (
            {"default": THEME_MODES[0]} if "ui_theme_mode" not in st.session_state else {}
        )
        st.segmented_control(
            "Tema de Interfaz:",
            options=THEME_MODES,
            selection_mode="single",
            required=True,
            width="stretch",
            key="ui_theme_mode",
            **theme_kwargs,
        )

        st.divider()
        st.markdown("### 👤 Espacio de Trabajo / Cuenta")
        st.caption(f"Espacio activo: **`{workspace_id}`**")
        if "sb_workspace_input" not in st.session_state:
            st.session_state["sb_workspace_input"] = workspace_id
        ws_col1, ws_col2 = st.columns([2.2, 1.1])
        with ws_col1:
            ws_candidate = st.text_input(
                "ID Espacio / Automotora:",
                label_visibility="collapsed",
                placeholder="ej. automotora_sur",
                key="sb_workspace_input",
            )
        with ws_col2:
            if st.button("Cambiar", key="sb_workspace_apply_btn", width="stretch"):
                new_ws = set_current_workspace_id(ws_candidate)
                st.session_state["_fav_nonce"] = int(st.session_state.get("_fav_nonce", 0)) + 1
                st.session_state["_radar_nonce"] = int(st.session_state.get("_radar_nonce", 0)) + 1
                _queue_toast(f"Espacio activo cambiado a '{new_ws}'.", icon="👤")
                st.rerun()

        st.divider()
        inv_hdr1, inv_hdr2 = st.columns([2.2, 1.0])
        with inv_hdr1:
            st.markdown("### 📊 Inventario de Mercado")
        with inv_hdr2:
            if st.button(
                "🔄",
                key="sb_refresh_cache_btn",
                help="Refrescar datos de mercado sin ejecutar escaneo externo",
                width="stretch",
            ):
                _clear_all_market_caches()
                _queue_toast("Datos de mercado actualizados.", icon="🔄")
                st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            st.metric(
                label="Total Histórico",
                value=_fmt_int(summary.get("total_historicos", 0)),
                help="Publicaciones históricas monitoreadas en la plataforma.",
            )
        with c2:
            st.metric(
                label="Autos Activos",
                value=_fmt_int(summary.get("activos", 0)),
                help="Publicaciones vigentes disponibles hoy.",
            )

        c3, c4 = st.columns(2)
        with c3:
            st.metric(
                label="Vendidos Hoy",
                value=_fmt_int(summary.get("vendidos_hoy", 0)),
                delta=f"{_fmt_int(summary.get('vendidos', 0))} hist.",
            )
        with c4:
            st.metric(
                label="Muestra Validada",
                value=_fmt_int(len(clean_df)),
                help="Publicaciones depuradas tras control de valores atípicos.",
            )

        if portal_counts:
            portal_str = " · ".join(f"**{k}:** {v}" for k, v in portal_counts.items())
            st.caption(f"Portales activos: {portal_str}")
        dups_merged = int(summary.get("duplicados_multiplataforma", 0))
        if dups_merged > 0:
            st.caption(f"💎 {dups_merged} publicaciones multi-portal consolidadas.")
        st.caption(f"⭐ Unidades en Pipeline CRM: **{favorites_count}**")

        st.divider()
        with st.expander("➕ Catálogo Personalizado", expanded=False):
            st.caption("Añade marcas y modelos para seguimiento permanente:")
            new_brand_input = st.text_input(
                "Marca:", placeholder="Ej: HONDA, BMW, AUDI", key="add_cat_marca"
            )
            new_model_input = st.text_input(
                "Modelo:", placeholder="Ej: ACCORD, X5, A4", key="add_cat_modelo"
            )
            if st.button("💾 Guardar Segmento", key="btn_save_catalog", width="stretch"):
                norm_new_mod = normalize_model_input(new_model_input)
                ok_cat, msg_cat = config.add_custom_segment(new_brand_input, norm_new_mod)
                if ok_cat:
                    st.session_state["focused_marca"] = new_brand_input.strip().upper()
                    st.session_state["focused_modelo"] = norm_new_mod
                    st.session_state["_segments_nonce"] = (
                        int(st.session_state.get("_segments_nonce", 0)) + 1
                    )
                    _get_cached_target_segments.clear()
                    _queue_toast(msg_cat, icon="✅")
                    st.rerun()
                else:
                    st.warning(msg_cat)

        has_admin_code = bool(str(getattr(config, "ADMIN_ACCESS_CODE", "") or "").strip())
        if has_admin_code:
            st.divider()
            st.markdown("### 🔐 Acceso Operador")
            if is_admin_unlocked():
                st.success("Modo operador habilitado")
                if st.button("🔒 Bloquear Acceso", key="sb_lock_admin_btn", width="stretch"):
                    lock_admin()
                    _queue_toast("Acceso operador bloqueado.", icon="🔒")
                    st.rerun()
            else:
                admin_input = st.text_input(
                    "Código de operador:",
                    type="password",
                    placeholder="Ingresa código de operador...",
                    key="sb_admin_code_input",
                )
                if st.button("🔓 Desbloquear", key="sb_unlock_admin_btn", width="stretch"):
                    if verify_admin_code(admin_input):
                        _queue_toast("Acceso operador desbloqueado.", icon="🔓")
                        st.rerun()
                    else:
                        st.error("Código incorrecto.")

        st.divider()
        st.markdown("### 🤖 Modelo de Tasación IA")
        st.caption(
            f"Última calibración: **`{formatted_train_date}`** · "
            f"Muestras: **`{_fmt_int(model_meta.get('registros_entrenamiento_total', 0))}`**"
        )
        if metrics:
            m1, m2 = st.columns(2)
            with m1:
                st.metric("R² Score", f"{float(metrics.get('R2_Score', 0.0)) * 100:.1f}%")
            with m2:
                st.metric("Error MAPE", f"{float(metrics.get('MAPE_Pct', 0.0)):.1f}%")
            st.caption(
                f"MAE: **{format_clp(metrics.get('MAE_CLP', 0))}** · "
                f"RMSE: **{format_clp(metrics.get('RMSE_CLP', 0))}**"
            )


# ==============================================================================
# 6. COMPONENTE UNIFICADO DE FILTROS MULTIFACÉTICOS
# ==============================================================================
def _reset_multifacet_filters(key_prefix: str) -> None:
    """Limpia de forma segura las claves de filtro asociadas antes del rerun."""
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


def _apply_saas_filters(
    df: pd.DataFrame,
    key_prefix: str,
    default_only_opportunities: bool = False,
    allow_category_filter: bool = True,
    clean_reference_df: Optional[pd.DataFrame] = None,
    deal_costs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Renderiza 4 filtros principales (Marcas, Modelos, Región, Ordenar) + popover '🎛️ Más Filtros',
    enriquece con métricas netas/confianza IA y retorna el DataFrame filtrado y ordenado.
    """
    if df.empty:
        return enrich_dataframe_with_net_deal_and_confidence(
            df.copy(), reference_df=clean_reference_df, costs=deal_costs
        )

    work = df
    valid_prices = pd.to_numeric(work["precio"], errors="coerce").dropna()
    valid_prices = valid_prices[valid_prices > 0]
    min_p_db = int(valid_prices.min()) if not valid_prices.empty else 0
    max_p_db = int(valid_prices.max()) if not valid_prices.empty else 120_000_000
    slider_p_max = max(max_p_db, min_p_db + 1_000_000, 1_000_000)

    valid_years = pd.to_numeric(work["anio"], errors="coerce").dropna()
    min_y_db = int(valid_years.min()) if not valid_years.empty else config.MIN_VALID_YEAR
    max_y_db = int(valid_years.max()) if not valid_years.empty else config.MAX_VALID_YEAR
    slider_y_max = max(max_y_db, min_y_db + 1)

    valid_kms = pd.to_numeric(work["kilometraje"], errors="coerce").dropna()
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

    # 4 Columnas Principales
    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with c1:
        sel_marcas = st.multiselect(
            "Marcas:",
            options=marcas_disp,
            placeholder="Todas las marcas...",
            key=f"{key_prefix}_marcas",
            **({} if f"{key_prefix}_marcas" in st.session_state else {"default": []}),
        )
    with c2:
        if sel_marcas:
            modelos_disp = sorted(
                work.loc[work["marca"].isin(sel_marcas), "modelo"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        else:
            modelos_disp = sorted(work["modelo"].dropna().astype(str).unique().tolist())
        _sanitize_multiselect_state(f"{key_prefix}_modelos", modelos_disp)
        sel_modelos = st.multiselect(
            "Modelos:",
            options=modelos_disp,
            placeholder="Todos los modelos...",
            key=f"{key_prefix}_modelos",
            **({} if f"{key_prefix}_modelos" in st.session_state else {"default": []}),
        )
    with c3:
        sel_regiones = st.multiselect(
            "Región:",
            options=regiones_disp,
            placeholder="Todas las regiones...",
            key=f"{key_prefix}_regiones",
            **({} if f"{key_prefix}_regiones" in st.session_state else {"default": []}),
        )
    with c4:
        sort_key = f"{key_prefix}_sort"
        if sort_key in st.session_state and st.session_state[sort_key] not in SORT_OPTIONS_LABELS:
            st.session_state[sort_key] = SORT_OPTIONS_LABELS[0]
        sort_by = st.selectbox(
            "Ordenar por:",
            options=SORT_OPTIONS_LABELS,
            key=sort_key,
            **({} if sort_key in st.session_state else {"index": 0}),
        )

    if sel_regiones:
        comunas_disp = sorted(
            work.loc[reg_col.isin(sel_regiones), "comuna"].dropna().astype(str).unique().tolist()
        )
    else:
        comunas_disp = sorted(work["comuna"].dropna().astype(str).unique().tolist())
    _sanitize_multiselect_state(f"{key_prefix}_comunas", comunas_disp)

    _sanitize_slider_state(f"{key_prefix}_price_rng", 0, slider_p_max)
    _sanitize_slider_state(f"{key_prefix}_year_rng", min_y_db, slider_y_max)
    _sanitize_slider_state(f"{key_prefix}_km_rng", 0, max_km_db)
    _sanitize_slider_state(f"{key_prefix}_disc_rng", -50, 80)
    _sanitize_slider_state(f"{key_prefix}_tier", 1, 5)

    default_min_disc = 15 if default_only_opportunities else -50
    default_max_disc = 55 if default_only_opportunities else 80

    pop_col, summary_col = st.columns([1.2, 3.6])
    with pop_col:
        with st.popover("🎛️ Más Filtros", width="stretch"):
            p1, p2 = st.columns(2)
            with p1:
                price_range = st.slider(
                    "Precio (CLP):",
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
            with p2:
                year_range = st.slider(
                    "Año:",
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

            p3, p4 = st.columns(2)
            with p3:
                km_range = st.slider(
                    "Kilometraje (km):",
                    min_value=0,
                    max_value=max_km_db,
                    step=5_000,
                    format="%d km",
                    key=f"{key_prefix}_km_rng",
                    **(
                        {}
                        if f"{key_prefix}_km_rng" in st.session_state
                        else {"value": (0, max_km_db)}
                    ),
                )
            with p4:
                disc_range = st.slider(
                    "% Descuento vs IA:",
                    min_value=-50,
                    max_value=80,
                    step=1,
                    format="%d%%",
                    key=f"{key_prefix}_disc_rng",
                    **(
                        {}
                        if f"{key_prefix}_disc_rng" in st.session_state
                        else {"value": (default_min_disc, default_max_disc)}
                    ),
                )

            p5, p6 = st.columns(2)
            with p5:
                min_margin_clp = st.number_input(
                    "Margen Neto Mínimo (CLP):",
                    min_value=-50_000_000,
                    max_value=50_000_000,
                    step=250_000,
                    key=f"{key_prefix}_min_margin",
                    **(
                        {}
                        if f"{key_prefix}_min_margin" in st.session_state
                        else {"value": 0}
                    ),
                )
            with p6:
                tier_range = st.slider(
                    "Tier Comunal (1–5):",
                    min_value=1,
                    max_value=5,
                    key=f"{key_prefix}_tier",
                    **({} if f"{key_prefix}_tier" in st.session_state else {"value": (1, 5)}),
                )

            p7, p8 = st.columns(2)
            with p7:
                sel_trans = st.multiselect(
                    "Transmisión:",
                    options=trans_disp,
                    key=f"{key_prefix}_trans",
                    **({} if f"{key_prefix}_trans" in st.session_state else {"default": trans_disp}),
                )
            with p8:
                sel_comb = st.multiselect(
                    "Combustible:",
                    options=comb_disp,
                    key=f"{key_prefix}_comb",
                    **({} if f"{key_prefix}_comb" in st.session_state else {"default": comb_disp}),
                )

            p9, p10 = st.columns(2)
            with p9:
                sel_comunas = st.multiselect(
                    "Comunas:",
                    options=comunas_disp,
                    placeholder="Todas las comunas...",
                    key=f"{key_prefix}_comunas",
                    **({} if f"{key_prefix}_comunas" in st.session_state else {"default": []}),
                )
            with p10:
                sel_portales = st.multiselect(
                    "Portales:",
                    options=portales_disp,
                    key=f"{key_prefix}_portales",
                    **(
                        {}
                        if f"{key_prefix}_portales" in st.session_state
                        else {"default": portales_disp}
                    ),
                )

            nlp_key = f"{key_prefix}_nlp"
            nlp_opts = list(NLP_LABELS_MAP.values())
            _sanitize_multiselect_state(nlp_key, nlp_opts)
            sel_nlp_labels = st.multiselect(
                "Atributos NLP:",
                options=nlp_opts,
                placeholder="Único Dueño, Mant. al Día, Cuero...",
                key=nlp_key,
                **({} if nlp_key in st.session_state else {"default": []}),
            )

            free_q_key = f"{key_prefix}_free_q"
            free_query = st.text_input(
                "Búsqueda textual / ID / Generación:",
                placeholder="Ej: GEN4, MLC-, sunroof, 4x4...",
                key=free_q_key,
                **({} if free_q_key in st.session_state else {"value": ""}),
            )

            if allow_category_filter and "categoria_arbitraje" in work.columns:
                cats_disp = sorted(
                    work["categoria_arbitraje"].dropna().astype(str).unique().tolist()
                )
                cats_key = f"{key_prefix}_cats"
                _sanitize_multiselect_state(cats_key, cats_disp)
                default_cats = (
                    ["OPORTUNIDAD"]
                    if (default_only_opportunities and "OPORTUNIDAD" in cats_disp)
                    else cats_disp
                )
                sel_cats = st.multiselect(
                    "Categoría IA:",
                    options=cats_disp,
                    key=cats_key,
                    **({} if cats_key in st.session_state else {"default": default_cats}),
                )
            else:
                sel_cats = []

            st.button(
                "🧹 Limpiar Filtros",
                key=f"{key_prefix}_reset_btn",
                on_click=_reset_multifacet_filters,
                args=(key_prefix,),
                width="stretch",
            )

    mask = (
        (work["precio"] >= price_range[0])
        & (work["precio"] <= price_range[1])
        & (work["anio"] >= year_range[0])
        & (work["anio"] <= year_range[1])
        & (work["kilometraje"] >= km_range[0])
        & (work["kilometraje"] <= km_range[1])
        & ((work["descuento_pct"] * 100.0) >= disc_range[0])
        & ((work["descuento_pct"] * 100.0) <= disc_range[1])
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

    # Búsqueda textual multi-término segura (regex=False evita crashes con paréntesis/corchetes)
    terms = [t for t in str(free_query or "").strip().lower().split() if t]
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

    # Enriquecer con Margen Neto Real, ROI Neto y Confianza IA
    enriched = enrich_dataframe_with_net_deal_and_confidence(
        filtered,
        reference_df=clean_reference_df,
        costs=deal_costs,
    )
    if not enriched.empty and int(min_margin_clp) != 0:
        enriched = enriched.loc[enriched["margen_neto_clp"] >= int(min_margin_clp)].copy()

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
        active_pills.append(f"Texto: '{str(free_query).strip()}'")
    if sel_nlp_labels:
        active_pills.append(f"NLP: {', '.join(sel_nlp_labels)}")
    if int(min_margin_clp) > 0:
        active_pills.append(f"Margen >= {format_clp(min_margin_clp)}")

    with summary_col:
        st.markdown("<div style='height: 0.45rem;'></div>", unsafe_allow_html=True)
        if active_pills:
            st.caption(f"🔎 **Filtros activos:** {' | '.join(active_pills)}")
        else:
            st.caption(
                f"Mostrando **{len(enriched)}** vehículos filtrados · Orden: **{sort_by}**"
            )

    sort_map = {
        "Mayor Margen Neto": ("margen_neto_clp", False),
        "Mayor ROI Neto": ("roi_neto_pct", False),
        "Mayor Confianza IA": ("confianza_ia_score", False),
        "Menor Precio": ("precio", True),
        "Mayor % Dcto": ("descuento_pct", False),
    }
    sort_col, ascending = sort_map.get(sort_by, ("margen_neto_clp", False))
    if sort_col in enriched.columns and not enriched.empty:
        enriched = enriched.sort_values(by=sort_col, ascending=ascending)

    return enriched


# ==============================================================================
# 7. VISTA 1: OPORTUNIDADES DE ARBITRAJE
# ==============================================================================
def _highlight_profit_cell(_val: Any) -> str:
    return "background-color: rgba(16, 185, 129, 0.18); color: #34d399; font-weight: 700;"


def _render_live_capture_expander(clean_df: pd.DataFrame) -> None:
    """Buscador colapsable de captura en vivo protegido por operador y cooldown."""
    with st.expander("📡 Captura en Vivo Multi-Portal (Escaneo Focalizado)", expanded=False):
        if not is_admin_unlocked():
            st.info(
                "🔒 El escaneo en vivo sobre portales está protegido. "
                "Desbloquea el **Acceso Operador** en la barra lateral para ejecutar capturas bajo demanda."
            )
            return

        all_segments = _get_cached_target_segments(
            int(st.session_state.get("_segments_nonce", 0))
        )
        target_marcas = sorted({m for m, _ in all_segments})
        db_marcas = sorted(clean_df["marca"].unique().tolist()) if not clean_df.empty else []
        all_marcas = sorted(set(target_marcas + db_marcas))

        curr_marca = st.session_state.get("focused_marca") or "MAZDA"
        m_idx = (all_marcas.index(curr_marca) + 1) if curr_marca in all_marcas else (1 if all_marcas else 0)
        marca_opts = [f"TODAS ({len(all_segments)} Segmentos)"] + all_marcas + ["➕ OTRA MARCA"]

        b_c1, b_c2, b_c3, b_c4 = st.columns([1.3, 1.3, 1.1, 1.3])
        with b_c1:
            marca_sel = st.selectbox(
                "Marca objetivo:",
                options=marca_opts,
                key="main_search_marca",
                **({} if "main_search_marca" in st.session_state else {"index": m_idx}),
            )
            if marca_sel == "➕ OTRA MARCA":
                input_marca = (
                    st.text_input("Escribe la Marca:", value="AUDI", key="main_custom_marca")
                    .strip()
                    .upper()
                )
            elif marca_sel.startswith("TODAS"):
                input_marca = "TODAS"
            else:
                input_marca = marca_sel

        with b_c2:
            if input_marca == "TODAS":
                suggested_models: List[str] = []
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
            mod_opts = ["TODOS (Toda la Marca)"] + suggested_models + ["➕ OTRO MODELO"]
            mod_sel_key = "main_search_modelo_select"
            if mod_sel_key in st.session_state and st.session_state[mod_sel_key] not in mod_opts:
                st.session_state[mod_sel_key] = mod_opts[0]
            default_mod_str = (
                st.session_state.get("focused_modelo")
                if st.session_state.get("focused_marca") == input_marca
                and st.session_state.get("focused_modelo")
                else (suggested_models[0] if suggested_models else "TODOS (Toda la Marca)")
            )
            mod_sel_idx = mod_opts.index(default_mod_str) if default_mod_str in mod_opts else 0
            mod_sel = st.selectbox(
                "Modelo objetivo:",
                options=mod_opts,
                key=mod_sel_key,
                **({} if mod_sel_key in st.session_state else {"index": mod_sel_idx}),
            )
            if mod_sel == "➕ OTRO MODELO":
                input_modelo = normalize_model_input(
                    st.text_input("Escribe el Modelo:", placeholder="Ej: CIVIC, XC60", key="main_custom_modelo")
                )
            elif mod_sel == "TODOS (Toda la Marca)":
                input_modelo = "TODOS"
            else:
                input_modelo = mod_sel

        with b_c3:
            input_pages = st.slider(
                "Páginas/Portal:",
                min_value=1,
                max_value=5,
                key="main_search_pages",
                **({} if "main_search_pages" in st.session_state else {"value": 2}),
            )

        with b_c4:
            input_alerts = st.slider(
                "Alertas Telegram:",
                min_value=0,
                max_value=10,
                help="0 = Actualizar tablero sin enviar alertas externas.",
                key="main_search_alerts",
                **({} if "main_search_alerts" in st.session_state else {"value": 3}),
            )
            force_retrain_chk = st.checkbox(
                "Re-calibrar IA tras captura",
                key="main_search_retrain",
                **({} if "main_search_retrain" in st.session_state else {"value": False}),
            )

        t1_apply_years = st.toggle(
            "Restringir rango de años y captura profunda",
            key="t1_apply_years_chk",
            **({} if "t1_apply_years_chk" in st.session_state else {"value": False}),
        )
        f_r1, f_r2, f_r3, f_r4, f_r5 = st.columns(5)
        with f_r1:
            t1_anio_min = st.number_input(
                "Año Mín:",
                min_value=config.MIN_VALID_YEAR,
                max_value=config.MAX_VALID_YEAR,
                step=1,
                disabled=not t1_apply_years,
                key="t1_search_anio_min",
                **({} if "t1_search_anio_min" in st.session_state else {"value": 2016}),
            )
        with f_r2:
            t1_anio_max = st.number_input(
                "Año Máx:",
                min_value=config.MIN_VALID_YEAR,
                max_value=config.MAX_VALID_YEAR,
                step=1,
                disabled=not t1_apply_years,
                key="t1_search_anio_max",
                **({} if "t1_search_anio_max" in st.session_state else {"value": 2024}),
            )
        with f_r3:
            t1_precio_min = st.number_input(
                "Precio Mín (CLP):",
                min_value=0,
                max_value=config.MAX_VALID_PRICE_CLP,
                step=1_000_000,
                key="t1_search_precio_min",
                **({} if "t1_search_precio_min" in st.session_state else {"value": 0}),
            )
        with f_r4:
            t1_precio_max = st.number_input(
                "Precio Máx (CLP):",
                min_value=0,
                max_value=config.MAX_VALID_PRICE_CLP,
                step=1_000_000,
                key="t1_search_precio_max",
                **({} if "t1_search_precio_max" in st.session_state else {"value": 0}),
            )
        with f_r5:
            t1_km_max = st.number_input(
                "Km Máximo:",
                min_value=0,
                max_value=config.MAX_VALID_KM,
                step=10_000,
                key="t1_search_km_max",
                **({} if "t1_search_km_max" in st.session_state else {"value": 0}),
            )

        f_sub1, f_sub2 = st.columns([1.8, 1.2])
        with f_sub1:
            t1_deep = st.checkbox(
                "⚡ Búsqueda profunda año por año (Chileautos)",
                key="t1_deep_chk",
                **({} if "t1_deep_chk" in st.session_state else {"value": True}),
            )
        with f_sub2:
            launch_search = st.button(
                "🚀 Buscar y Analizar en Vivo",
                key="main_search_btn",
                type="primary",
                width="stretch",
            )

        if launch_search:
            allowed, wait_sec = check_scan_cooldown()
            if not allowed:
                st.warning(
                    f"⏳ Enfriamiento entre escaneos activo: espera **{wait_sec}s** antes de iniciar otro barrido."
                )
                return

            norm_modelo = None if input_modelo in ("TODOS", "") else normalize_model_input(input_modelo)
            if input_marca != "TODAS" and not input_marca:
                st.warning("Ingresa una Marca válida para iniciar el escaneo.")
                return
            if t1_apply_years and t1_anio_min > t1_anio_max:
                st.warning("Año mínimo no puede ser mayor que año máximo.")
                return
            if t1_precio_min > 0 and t1_precio_max > 0 and t1_precio_min > t1_precio_max:
                st.warning("Precio mínimo no puede ser mayor que precio máximo.")
                return

            record_scan_triggered()
            if input_marca != "TODAS" and norm_modelo:
                config.add_custom_segment(input_marca, norm_modelo)
                st.session_state["_segments_nonce"] = (
                    int(st.session_state.get("_segments_nonce", 0)) + 1
                )
                _get_cached_target_segments.clear()

            target_lbl = f"{input_marca} {norm_modelo or 'TODOS'}"
            with st.status(f"Escaneando portales en vivo ({target_lbl})...", expanded=True) as status_box:
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
                    status_box=status_box,
                )
                status_box.update(
                    label=f"Captura completada ({target_lbl})" if ok else "Error en captura en vivo",
                    state="complete" if ok else "error",
                    expanded=False,
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
                _queue_toast("Escaneo en vivo completado.", icon="✅")
                st.rerun(scope="app")
            else:
                st.error(msg)


def _render_edit_spec_expander(row: pd.Series, key_prefix: str) -> None:
    """Contenedor de edición rápida de ficha técnica dentro del menú Acciones."""
    raw_id = str(row["id_aviso"])
    trans_opts = ["Automática", "Manual"]
    comb_opts = ["Bencina", "Diésel", "Híbrido", "Eléctrico"]
    cur_trans = str(row.get("transmision") or "Manual")
    cur_comb = str(row.get("combustible") or "Bencina")
    trans_idx = trans_opts.index(cur_trans) if cur_trans in trans_opts else 0
    comb_idx = comb_opts.index(cur_comb) if cur_comb in comb_opts else 0

    with st.expander("✏️ Corregir ficha", expanded=False):
        st.caption(f"Ajustar ficha técnica de `{raw_id}`")
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
                value=min(config.MAX_VALID_YEAR, max(config.MIN_VALID_YEAR, int(row.get("anio") or 2020))),
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
                    max(config.MIN_VALID_PRICE_CLP, int(row.get("precio") or config.MIN_VALID_PRICE_CLP)),
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
            if _update_listing_specs_instant(
                row,
                kilometraje=int(ed_km),
                transmision=ed_trans,
                combustible=ed_comb,
                anio=int(ed_anio),
                precio=int(ed_precio),
            ):
                st.rerun(scope="app")


@st.fragment
def _render_opportunities_fragment(data_bundle: Dict[str, Any]) -> None:
    """Vista 1 aislada: costos flipping, captura colapsable, filtros SaaS, KPIs y tarjetas."""
    _flush_pending_toast()

    effective_bundle = apply_session_overrides(data_bundle)
    active_df = effective_bundle["active_featured"]
    clean_df = effective_bundle["clean_df"]
    workspace_id = get_current_workspace_id()
    favorites_map = load_user_favorites(workspace_id, int(st.session_state.get("_fav_nonce", 0)))

    if st.session_state.get("last_scan_message"):
        st.success(st.session_state.pop("last_scan_message"))
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
                f"🗂️ **{last_act.get('label', '')}** (`{last_act['id_aviso']}`) marcado como **{st_lbl}**."
            )
        with u_col2:
            if st.button("↩️ Deshacer", key="opp_undo_dismiss_btn", width="stretch"):
                _update_listing_status_instant(
                    last_act["id_aviso"], "activo", last_act.get("label", "")
                )
                st.rerun(scope="app")

    # Barra compacta superior: Costos Flipping + Layout + Captura en Vivo colapsable
    bar_c1, bar_c2, bar_c3 = st.columns([1.15, 1.55, 2.3])
    with bar_c1:
        deal_costs = render_deal_cost_controls("opp")
    with bar_c2:
        layout_key = "opp_view_layout"
        layout_opts = ["Tarjetas + Tabla", "Tarjetas + Tabla + Gráficos", "Solo Tarjetas"]
        if layout_key in st.session_state and st.session_state[layout_key] not in layout_opts:
            st.session_state[layout_key] = layout_opts[0]
        view_layout = st.selectbox(
            "Vista:",
            options=layout_opts,
            label_visibility="collapsed",
            key=layout_key,
            **({} if layout_key in st.session_state else {"index": 0}),
        )
    with bar_c3:
        st.caption(
            "Costos netos aplicados en tiempo real (transferencia 1,5% + RVM, Autofact, reacond. y capital)."
        )

    _render_live_capture_expander(clean_df)

    if active_df.empty:
        render_empty_state(
            title="Sin publicaciones activas disponibles",
            subtitle="Ejecuta un escaneo en vivo para capturar oportunidades actualizadas desde los portales.",
            icon="🚘",
        )
        return

    filtered_opps = _apply_saas_filters(
        active_df,
        key_prefix="opp_flt",
        default_only_opportunities=True,
        allow_category_filter=True,
        clean_reference_df=clean_df,
        deal_costs=deal_costs,
    )

    # Strip KPI responsivo (5 métricas clave)
    best_net = float(filtered_opps["margen_neto_clp"].max()) if not filtered_opps.empty else 0.0
    avg_net = float(filtered_opps["margen_neto_clp"].mean()) if not filtered_opps.empty else 0.0
    avg_roi = float(filtered_opps["roi_neto_pct"].mean()) if not filtered_opps.empty else 0.0
    avg_conf = float(filtered_opps["confianza_ia_score"].mean()) if not filtered_opps.empty else 0.0

    render_kpi_strip(
        [
            {
                "label": "Oportunidades",
                "value": _fmt_int(len(filtered_opps)),
                "sub": "Publicaciones filtradas",
                "tone": "neutral",
            },
            {
                "label": "Mejor Margen Neto",
                "value": format_clp(best_net),
                "sub": "Después de gastos",
                "tone": "emerald",
            },
            {
                "label": "Margen Neto Medio",
                "value": format_clp(avg_net),
                "sub": "Promedio del filtro",
                "tone": "emerald" if avg_net >= 0 else "amber",
            },
            {
                "label": "ROI Neto Medio",
                "value": f"{avg_roi:+.1f}%",
                "sub": "Sobre inversión total",
                "tone": "blue",
            },
            {
                "label": "Confianza IA Media",
                "value": f"{avg_conf:.0f}/100",
                "sub": "Respaldo por comparables",
                "tone": "amber" if avg_conf < 60 else "blue",
            },
        ]
    )

    if filtered_opps.empty:
        render_empty_state(
            title="Ningún vehículo coincide con estos filtros",
            subtitle="Amplía el rango de precio, ajusta el % de descuento mínimo o limpia los filtros activos.",
            icon="🔎",
        )
        return

    # Gráficos de Rentabilidad Neta
    if "Gráficos" in view_layout and len(filtered_opps) >= 2:
        with st.expander("📊 Matriz de Rentabilidad Neta y Distribución por Marca", expanded=False):
            g_col1, g_col2 = st.columns([1.35, 1.0])
            with g_col1:
                portal_palette = ["#10b981", "#38bdf8", "#f59e0b", "#a855f7", "#fb7185"]
                fig_roi = go.Figure()
                for idx_p, (p_name, grp) in enumerate(filtered_opps.groupby("portal", sort=False)):
                    fig_roi.add_trace(
                        go.Scatter(
                            x=grp["precio"],
                            y=grp["margen_neto_clp"],
                            mode="markers",
                            name=str(p_name),
                            marker=dict(
                                size=np.clip(grp["descuento_pct"] * 38.0, 7.0, 22.0),
                                color=portal_palette[idx_p % len(portal_palette)],
                                opacity=0.84,
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
                                    grp["roi_neto_pct"],
                                    grp["confianza_ia_score"],
                                ],
                                axis=-1,
                            ),
                            hovertemplate=(
                                "<b>%{customdata[0]} — %{customdata[1]} %{customdata[2]} (%{customdata[3]})</b><br>"
                                "Km: %{customdata[4]:,.0f} km | Confianza IA: %{customdata[7]}/100<br>"
                                "Precio Publicado: <b>$ %{x:,.0f}</b><br>"
                                "Precio Justo IA: $ %{customdata[5]:,.0f}<br>"
                                "Margen Neto: <b>$ %{y:,.0f}</b> (ROI Neto: %{customdata[6]:+.1f}%)<extra></extra>"
                            ),
                        )
                    )
                fig_roi.update_layout(
                    title="Capital de Compra vs Margen Neto Real (CLP)",
                    xaxis_title="Precio Publicado (CLP)",
                    yaxis_title="Margen Neto Proyectado (CLP)",
                    legend_title_text="Portal",
                    template=get_plotly_template(),
                    height=360,
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
                        margen_medio=("margen_neto_clp", "mean"),
                    )
                    .sort_values("cantidad", ascending=True)
                )
                fig_brand = go.Figure(
                    go.Bar(
                        x=brand_summary["cantidad"],
                        y=brand_summary["marca"],
                        orientation="h",
                        text=[
                            f"{cnt} uds. ({format_clp(m)})"
                            for cnt, m in zip(brand_summary["cantidad"], brand_summary["margen_medio"])
                        ],
                        textposition="auto",
                        marker=dict(
                            color=brand_summary["margen_medio"],
                            colorscale="Tealgrn",
                            showscale=False,
                        ),
                    )
                )
                fig_brand.update_layout(
                    title="Oportunidades y Margen Neto Medio por Marca",
                    xaxis_title="Unidades",
                    yaxis_title="Marca",
                    template=get_plotly_template(),
                    height=360,
                    margin=dict(l=20, r=20, t=45, b=20),
                )
                st.plotly_chart(fig_brand, width="stretch")

    # Tarjetas Paginadas con SOLO 2 botones abajo (Guardar/En CRM + Menú ⋯ Acciones)
    if "Tarjetas" in view_layout:
        card_page_sizes = [6, 12, 24]
        if (
            "opp_max_cards" in st.session_state
            and st.session_state["opp_max_cards"] not in card_page_sizes
        ):
            st.session_state["opp_max_cards"] = 6

        c_title, c_limit, c_page = st.columns([2.4, 0.8, 0.8])
        with c_title:
            st.markdown("### Oportunidades Detectadas")
        with c_limit:
            max_cards = st.selectbox(
                "Tarjetas / pág:",
                options=card_page_sizes,
                key="opp_max_cards",
                **({} if "opp_max_cards" in st.session_state else {"index": 0}),
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
                    raw_id_aviso = str(row["id_aviso"])
                    is_fav = raw_id_aviso in favorites_map
                    card_label = f"{row['marca']} {row['modelo']} ({int(row['anio'])})"
                    card_html = clean_html(
                        render_vehicle_card_html(row, mode="opportunity", is_saved=is_fav)
                    )

                    with col:
                        st.markdown(card_html, unsafe_allow_html=True)
                        act_c1, act_c2 = st.columns(2)
                        with act_c1:
                            if st.button(
                                "💛 En CRM" if is_fav else "⭐ Guardar",
                                key=f"opp_fav_{i + j}_{raw_id_aviso}",
                                width="stretch",
                            ):
                                _toggle_favorite_instant(raw_id_aviso, workspace_id, card_label)
                                _safe_rerun_fragment()
                        with act_c2:
                            with st.popover("⋯ Acciones", width="stretch"):
                                st.caption(f"Gestionar `{raw_id_aviso}`")
                                st.download_button(
                                    label="📄 Descargar Dossier PDF",
                                    data=lambda r_snap=row.copy(), c_snap=deal_costs: _build_pdf_bytes_for_row(
                                        r_snap, clean_df, c_snap
                                    ),
                                    file_name=f"dossier_{raw_id_aviso}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf",
                                    on_click="ignore",
                                    key=f"opp_pdf_{i + j}_{raw_id_aviso}",
                                    width="stretch",
                                )
                                _render_edit_spec_expander(row, key_prefix=f"opp_{i + j}")
                                if st.button(
                                    "🚫 Ya se vendió",
                                    key=f"opp_sold_{i + j}_{raw_id_aviso}",
                                    width="stretch",
                                ):
                                    _update_listing_status_instant(
                                        raw_id_aviso, "vendido", card_label
                                    )
                                    st.rerun(scope="app")
                                if st.button(
                                    "⚠️ Link erróneo",
                                    key=f"opp_bad_{i + j}_{raw_id_aviso}",
                                    width="stretch",
                                ):
                                    _update_listing_status_instant(
                                        raw_id_aviso, "descartado", card_label
                                    )
                                    st.rerun(scope="app")

    # Tabla Detallada + Exportación CSV
    if "Tabla" in view_layout:
        t_head1, t_head2 = st.columns([3, 1])
        with t_head1:
            st.markdown("### Tabla Detallada de Oportunidades")
        with t_head2:
            st.download_button(
                label="📥 Descargar CSV",
                data=lambda df_exp=filtered_opps: df_exp.to_csv(index=False).encode("utf-8"),
                file_name=f"oportunidades_autopulse_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                on_click="ignore",
                key="opp_download_csv_btn",
                width="stretch",
            )

        table_df = pd.DataFrame(
            {
                "Portal": filtered_opps["portal_badge"],
                "ID Aviso": filtered_opps["id_aviso"],
                "Marca": filtered_opps["marca"],
                "Modelo": filtered_opps["modelo"],
                "Año": filtered_opps["anio"].astype(int),
                "Kilometraje": filtered_opps["kilometraje"].astype(int),
                "Comuna": filtered_opps["comuna"],
                "Precio Publicado": filtered_opps["precio"].astype(int),
                "Precio Justo IA": filtered_opps["precio_justo_ia"].round(0).astype(int),
                "Margen Neto": filtered_opps["margen_neto_clp"].round(0).astype(int),
                "ROI Neto (%)": filtered_opps["roi_neto_pct"].round(1),
                "Dcto (%)": (filtered_opps["descuento_pct"] * 100.0).round(1),
                "Confianza IA": filtered_opps["confianza_ia_score"].astype(int),
                "Tier": filtered_opps["deal_tier"],
                "Enlace": filtered_opps["url"],
            }
        )
        col_cfg = {
            "Año": st.column_config.NumberColumn(format="%d"),
            "Kilometraje": st.column_config.NumberColumn(format="%d km"),
            "Precio Publicado": st.column_config.NumberColumn(format="$ %d"),
            "Precio Justo IA": st.column_config.NumberColumn(format="$ %d"),
            "Margen Neto": st.column_config.NumberColumn(format="+ $ %d"),
            "ROI Neto (%)": st.column_config.NumberColumn("ROI Neto", format="%+.1f %%"),
            "Dcto (%)": st.column_config.NumberColumn("Dcto. IA", format="-%.1f %%"),
            "Confianza IA": st.column_config.NumberColumn("Confianza", format="%d/100"),
            "Enlace": st.column_config.LinkColumn("Aviso", display_text="Abrir ↗"),
        }
        df_display: Any = (
            table_df.style.map(
                _highlight_profit_cell,
                subset=["Margen Neto", "ROI Neto (%)", "Dcto (%)"],
            )
            if len(table_df) <= 300
            else table_df
        )
        st.dataframe(df_display, column_config=col_cfg, hide_index=True, width="stretch")

    # Alertas de Riesgo / Revisión
    scam_df = active_df.loc[
        active_df["categoria_arbitraje"] == "POSIBLE ESTAFA / ERROR DE TIPEO"
    ].sort_values(by="descuento_pct", ascending=False)
    if not scam_df.empty:
        with st.expander(
            f"⚠️ Publicaciones con Descuento Extremo / Revisar ({len(scam_df)} detectadas)",
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
                    "URL": st.column_config.LinkColumn("Enlace", display_text="Inspeccionar ↗"),
                },
                hide_index=True,
                width="stretch",
            )


def render_tab_opportunities(data_bundle: Dict[str, Any]) -> None:
    """Vista 1: Oportunidades de Arbitraje."""
    _render_opportunities_fragment(data_bundle)


# ==============================================================================
# 8. VISTA 2: PIPELINE CRM Y RADARES DE ALERTA
# ==============================================================================
def _render_crm_tracking_subview(
    full_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    workspace_id: str,
) -> None:
    """Sub-vista Seguimiento CRM: etapas, oferta, nota personal, PDF e historial de precios."""
    favorites_map = load_user_favorites(workspace_id, int(st.session_state.get("_fav_nonce", 0)))

    if not favorites_map or full_df.empty:
        render_empty_state(
            title="Tu Pipeline CRM está vacío",
            subtitle=(
                "Guarda vehículos desde Oportunidades, Mercado, Tasador IA o Rebajas "
                "para hacer seguimiento de ofertas, notas e historial de precios."
            ),
            icon="⭐",
        )
        return

    fav_ids = list(favorites_map.keys())
    fav_df = full_df[full_df["id_aviso"].astype(str).isin(fav_ids)].copy()
    if fav_df.empty:
        render_empty_state(
            title="Sin registros activos asociados a tus favoritos",
            subtitle="Los avisos guardados en este espacio aún no figuran en el inventario sincronizado.",
            icon="📋",
        )
        return

    cost_col, f_col1, f_col2, f_col3, f_col4 = st.columns([1.15, 1.5, 1.0, 1.35, 1.0])
    with cost_col:
        deal_costs = render_deal_cost_controls("crm")

    fav_df = enrich_dataframe_with_net_deal_and_confidence(
        fav_df, reference_df=clean_df, costs=deal_costs
    )
    fav_df["nota_personal"] = fav_df["id_aviso"].astype(str).map(
        lambda x: str(favorites_map.get(x, {}).get("nota", "") or "")
    )
    fav_df["etapa_crm"] = fav_df["id_aviso"].astype(str).map(
        lambda x: str(favorites_map.get(x, {}).get("etapa_crm", "Interesado") or "Interesado")
    )
    fav_df["oferta_clp"] = fav_df["id_aviso"].astype(str).map(
        lambda x: int(favorites_map.get(x, {}).get("oferta_clp", 0) or 0)
    )
    fav_df["fecha_guardado"] = fav_df["id_aviso"].astype(str).map(
        lambda x: str(favorites_map.get(x, {}).get("fecha_guardado", "") or "")
    )

    stage_flt_key = "crm_stage_filter"
    _sanitize_multiselect_state(stage_flt_key, CRM_STAGES)
    with f_col1:
        sel_stages = st.multiselect(
            "Filtrar por Etapa CRM:",
            options=CRM_STAGES,
            placeholder="Todas las etapas...",
            key=stage_flt_key,
            **({} if stage_flt_key in st.session_state else {"default": []}),
        )
    with f_col2:
        only_active_crm = st.toggle(
            "Solo activos",
            key="crm_only_active_toggle",
            **({} if "crm_only_active_toggle" in st.session_state else {"value": False}),
        )
    with f_col3:
        crm_sort_opts = [
            "Guardado más reciente",
            "Mayor Margen Neto",
            "Mayor ROI Neto",
            "Menor Precio",
            "Año más Reciente",
        ]
        crm_sort_key = "crm_sort_select"
        if crm_sort_key in st.session_state and st.session_state[crm_sort_key] not in crm_sort_opts:
            st.session_state[crm_sort_key] = crm_sort_opts[0]
        crm_sort = st.selectbox(
            "Ordenar por:",
            options=crm_sort_opts,
            key=crm_sort_key,
            **({} if crm_sort_key in st.session_state else {"index": 0}),
        )

    view_df = fav_df.copy()
    if sel_stages:
        view_df = view_df[view_df["etapa_crm"].isin(sel_stages)].copy()
    if only_active_crm:
        view_df = view_df[view_df["estado"].astype(str).str.lower() == "activo"].copy()

    sort_cfg = {
        "Guardado más reciente": ("fecha_guardado", False),
        "Mayor Margen Neto": ("margen_neto_clp", False),
        "Mayor ROI Neto": ("roi_neto_pct", False),
        "Menor Precio": ("precio", True),
        "Año más Reciente": ("anio", False),
    }
    s_col, s_asc = sort_cfg.get(crm_sort, ("fecha_guardado", False))
    if s_col in view_df.columns and not view_df.empty:
        view_df = view_df.sort_values(by=s_col, ascending=s_asc)

    with f_col4:
        st.download_button(
            label="📥 Exportar CRM",
            data=lambda df_exp=view_df: df_exp.to_csv(index=False).encode("utf-8"),
            file_name=f"pipeline_crm_{workspace_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            on_click="ignore",
            key="crm_download_csv_btn",
            width="stretch",
        )

    # KPIs del portafolio guardado
    active_cnt = int((fav_df["estado"].astype(str).str.lower() == "activo").sum())
    negotiation_cnt = int(
        fav_df["etapa_crm"].isin(["Contactado", "Inspección", "Oferta Enviada"]).sum()
    )
    best_margin = float(view_df["margen_neto_clp"].max()) if not view_df.empty else 0.0
    total_offers = int(fav_df["oferta_clp"].sum())

    render_kpi_strip(
        [
            {"label": "Unidades en CRM", "value": _fmt_int(len(fav_df)), "sub": f"Espacio: {workspace_id}", "tone": "neutral"},
            {"label": "Vigentes Activas", "value": _fmt_int(active_cnt), "sub": "Disponibles hoy", "tone": "emerald"},
            {"label": "En Gestión / Oferta", "value": _fmt_int(negotiation_cnt), "sub": "Contactado a Oferta", "tone": "blue"},
            {"label": "Mejor Margen Neto", "value": format_clp(best_margin), "sub": "Del subconjunto visible", "tone": "emerald"},
            {"label": "Capital Ofertado", "value": format_clp(total_offers), "sub": "Suma de ofertas CRM", "tone": "amber"},
        ]
    )

    if view_df.empty:
        render_empty_state(
            title="Sin vehículos en esta etapa o estado",
            subtitle="Ajusta los filtros de Etapa CRM o desactiva 'Solo activos' para ver todo tu portafolio.",
            icon="🗂️",
        )
        return

    db_mtime, _, _ = _get_current_mtimes()
    rows_list = [r for _, r in view_df.iterrows()]

    for i in range(0, len(rows_list), 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            if i + j < len(rows_list):
                row = rows_list[i + j]
                raw_id = str(row["id_aviso"])
                card_label = f"{row['marca']} {row['modelo']} ({int(row['anio'])})"
                estado_val = str(row.get("estado") or "activo").lower()
                card_html = clean_html(render_vehicle_card_html(row, mode="crm", is_saved=True))

                with col:
                    st.markdown(card_html, unsafe_allow_html=True)

                    cur_stage = str(row.get("etapa_crm") or CRM_STAGES[0])
                    stage_idx = CRM_STAGES.index(cur_stage) if cur_stage in CRM_STAGES else 0
                    stage_key = f"crm_stage_{raw_id}"
                    if stage_key in st.session_state and st.session_state[stage_key] not in CRM_STAGES:
                        st.session_state[stage_key] = cur_stage

                    offer_key = f"crm_offer_{raw_id}"
                    note_key = f"crm_note_{raw_id}"

                    c_st, c_of = st.columns(2)
                    with c_st:
                        sel_stage = st.selectbox(
                            "Etapa CRM:",
                            options=CRM_STAGES,
                            key=stage_key,
                            **({} if stage_key in st.session_state else {"index": stage_idx}),
                        )
                    with c_of:
                        sel_offer = st.number_input(
                            "Tu Oferta (CLP):",
                            min_value=0,
                            max_value=config.MAX_VALID_PRICE_CLP,
                            step=100_000,
                            key=offer_key,
                            **(
                                {}
                                if offer_key in st.session_state
                                else {"value": max(0, int(row.get("oferta_clp") or 0))}
                            ),
                        )

                    note_val = st.text_input(
                        "Nota personal:",
                        placeholder="Ej: Contactar hoy, ofrecer cierre contado...",
                        key=note_key,
                        **(
                            {}
                            if note_key in st.session_state
                            else {"value": str(row.get("nota_personal") or "")}
                        ),
                    )

                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("💾 Guardar CRM", key=f"crm_save_{raw_id}", type="primary", width="stretch"):
                            clean_note = sanitize_user_text(note_val, max_len=280)
                            _get_db_manager().update_favorite_note(
                                raw_id,
                                clean_note,
                                etapa_crm=sel_stage,
                                oferta_clp=int(sel_offer),
                                owner_id=workspace_id,
                            )
                            st.session_state["_fav_nonce"] = (
                                int(st.session_state.get("_fav_nonce", 0)) + 1
                            )
                            _queue_toast(f"CRM actualizado para {card_label} ({sel_stage}).", icon="💾")
                            _safe_rerun_fragment()
                    with b2:
                        st.download_button(
                            label="📄 Dossier PDF",
                            data=lambda r_snap=row.copy(), c_snap=deal_costs: _build_pdf_bytes_for_row(
                                r_snap, clean_df, c_snap
                            ),
                            file_name=f"dossier_{raw_id}_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf",
                            on_click="ignore",
                            key=f"crm_pdf_{raw_id}",
                            width="stretch",
                        )

                    b3, b4 = st.columns(2)
                    with b3:
                        if st.button("🗑️ Quitar", key=f"crm_remove_{raw_id}", width="stretch"):
                            _toggle_favorite_instant(raw_id, workspace_id, card_label)
                            _safe_rerun_fragment()
                    with b4:
                        if estado_val == "activo":
                            if st.button("🚫 No disp.", key=f"crm_sold_{raw_id}", width="stretch"):
                                _update_listing_status_instant(raw_id, "vendido", card_label)
                                st.rerun(scope="app")
                        else:
                            if st.button("↩️ Reactivar", key=f"crm_restore_{raw_id}", width="stretch"):
                                _update_listing_status_instant(raw_id, "activo", card_label)
                                st.rerun(scope="app")

                    with st.expander("📉 Historial de Precios del Aviso", expanded=False):
                        hist_rows = load_listing_price_history_cached(raw_id, db_mtime)
                        if not hist_rows:
                            st.caption("Sin variaciones de precio registradas para este aviso.")
                        else:
                            hist_df = pd.DataFrame(hist_rows)
                            hist_df["variacion_clp"] = hist_df["precio_nuevo"] - hist_df["precio_antiguo"]
                            st.dataframe(
                                hist_df,
                                column_config={
                                    "precio_antiguo": st.column_config.NumberColumn("Anterior", format="$ %d"),
                                    "precio_nuevo": st.column_config.NumberColumn("Nuevo", format="$ %d"),
                                    "variacion_clp": st.column_config.NumberColumn("Variación", format="$ %d"),
                                    "fecha_cambio": st.column_config.TextColumn("Fecha"),
                                },
                                hide_index=True,
                                width="stretch",
                            )
                            if len(hist_df) >= 2:
                                fig_ph = go.Figure(
                                    go.Scatter(
                                        x=hist_df["fecha_cambio"],
                                        y=hist_df["precio_nuevo"],
                                        mode="lines+markers",
                                        line=dict(color="#10b981", width=2.5),
                                    )
                                )
                                fig_ph.update_layout(
                                    height=180,
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    yaxis=dict(tickprefix="$ ", separatethousands=True),
                                    template=get_plotly_template(),
                                )
                                st.plotly_chart(fig_ph, width="stretch")


def _render_radars_subview(active_df: pd.DataFrame, workspace_id: str) -> None:
    """Sub-vista Radares de Alerta: creación persistente, evaluación en vivo y borrado."""
    st.caption(
        "Configura reglas de vigilancia para detectar automáticamente oportunidades que cumplan tus criterios de compra."
    )

    all_segments = _get_cached_target_segments(
        int(st.session_state.get("_segments_nonce", 0))
    )
    all_marcas = sorted({m for m, _ in all_segments}.union(
        set(active_df["marca"].dropna().astype(str).unique()) if not active_df.empty else set()
    ))
    all_regions = sorted(set(COMUNA_TO_REGION.values()))

    with st.expander("➕ Crear Nuevo Radar de Alerta", expanded=True):
        r_c1, r_c2, r_c3 = st.columns([1.4, 1.0, 1.0])
        with r_c1:
            rad_name = st.text_input(
                "Nombre del Radar:",
                placeholder="Ej: Mazda 3 bajo $11M en RM",
                key="rad_form_name",
            )
        with r_c2:
            rad_marca_sel = st.selectbox(
                "Marca:",
                options=["TODAS"] + all_marcas,
                key="rad_form_marca",
            )
        with r_c3:
            if rad_marca_sel == "TODAS":
                rad_mod_opts = ["TODOS"]
            else:
                seg_mods = {str(mod) for m, mod in all_segments if m == rad_marca_sel}
                db_mods = (
                    set(active_df.loc[active_df["marca"] == rad_marca_sel, "modelo"].astype(str).unique())
                    if not active_df.empty
                    else set()
                )
                rad_mod_opts = ["TODOS"] + sorted(seg_mods.union(db_mods)) + ["➕ OTRO"]
            if (
                "rad_form_modelo" in st.session_state
                and st.session_state["rad_form_modelo"] not in rad_mod_opts
            ):
                st.session_state["rad_form_modelo"] = rad_mod_opts[0]
            rad_mod_sel = st.selectbox(
                "Modelo:",
                options=rad_mod_opts,
                key="rad_form_modelo",
                **({} if "rad_form_modelo" in st.session_state else {"index": 0}),
            )
            if rad_mod_sel == "➕ OTRO":
                rad_modelo_val = normalize_model_input(
                    st.text_input("Especificar modelo:", key="rad_form_custom_mod")
                )
            elif rad_mod_sel == "TODOS":
                rad_modelo_val = ""
            else:
                rad_modelo_val = rad_mod_sel

        r_c4, r_c5, r_c6, r_c7 = st.columns(4)
        with r_c4:
            rad_anio_min = st.number_input(
                "Año mínimo (0 = sin mín):",
                min_value=0,
                max_value=config.MAX_VALID_YEAR,
                value=2018,
                step=1,
                key="rad_form_anio_min",
            )
        with r_c5:
            rad_precio_max = st.number_input(
                "Precio máx CLP (0 = sin tope):",
                min_value=0,
                max_value=config.MAX_VALID_PRICE_CLP,
                value=15_000_000,
                step=500_000,
                key="rad_form_precio_max",
            )
        with r_c6:
            rad_km_max = st.number_input(
                "Km máx (0 = sin límite):",
                min_value=0,
                max_value=config.MAX_VALID_KM,
                value=120_000,
                step=10_000,
                key="rad_form_km_max",
            )
        with r_c7:
            rad_disc_min = st.slider(
                "Descuento mínimo (%):",
                min_value=0.0,
                max_value=60.0,
                value=15.0,
                step=1.0,
                key="rad_form_disc_min",
            )

        r_c8, r_c9, r_c10 = st.columns([1.2, 1.4, 1.0])
        with r_c8:
            rad_region = st.selectbox(
                "Región:",
                options=["TODAS"] + all_regions,
                key="rad_form_region",
            )
        with r_c9:
            rad_canal = st.text_input(
                "Canal / Etiqueta destino:",
                placeholder="Ej: Telegram · Mesa Compras",
                key="rad_form_canal",
            )
        with r_c10:
            st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
            if st.button("➕ Crear Radar", key="rad_form_submit", type="primary", width="stretch"):
                clean_name = sanitize_user_text(rad_name, max_len=100)
                if not clean_name:
                    st.warning("Ingresa un nombre descriptivo para el radar.")
                else:
                    _get_db_manager().create_radar_alert(
                        nombre=clean_name,
                        marca=None if rad_marca_sel == "TODAS" else rad_marca_sel,
                        modelo=rad_modelo_val or None,
                        anio_min=int(rad_anio_min) if int(rad_anio_min) > 0 else None,
                        precio_max=int(rad_precio_max) if int(rad_precio_max) > 0 else None,
                        km_max=int(rad_km_max) if int(rad_km_max) > 0 else None,
                        descuento_min_pct=float(rad_disc_min),
                        region=None if rad_region == "TODAS" else rad_region,
                        canal_destino=sanitize_user_text(rad_canal, max_len=100),
                        owner_id=workspace_id,
                    )
                    st.session_state["_radar_nonce"] = (
                        int(st.session_state.get("_radar_nonce", 0)) + 1
                    )
                    _queue_toast(f"Radar '{clean_name}' creado.", icon="📡")
                    _safe_rerun_fragment()

    radars = load_user_radars(workspace_id, int(st.session_state.get("_radar_nonce", 0)))
    if not radars:
        render_empty_state(
            title="No tienes radares configurados en este espacio",
            subtitle="Crea tu primer radar arriba para monitorear en vivo coincidencias con tu criterio de compra.",
            icon="📡",
        )
        return

    matches_map = evaluate_radar_matches(active_df, radars)
    st.markdown(f"### Radares Activos ({len(radars)})")

    for radar in radars:
        rid = int(radar["id"])
        matched_df = matches_map.get(rid, pd.DataFrame())
        crit_parts = [
            f"Marca: {radar.get('marca') or 'Todas'}",
            f"Modelo: {radar.get('modelo') or 'Todos'}",
            f"Año >= {radar['anio_min']}" if radar.get("anio_min") else "Año: Todos",
            f"Precio <= {format_clp(radar['precio_max'])}" if radar.get("precio_max") else "Sin tope CLP",
            f"Km <= {_fmt_int(radar['km_max'])}" if radar.get("km_max") else "Sin tope Km",
            f"Dcto >= {float(radar.get('descuento_min_pct') or 15.0):.0f}%",
            f"Región: {radar.get('region') or 'Todas'}",
        ]
        canal_lbl = f" · Canal: {radar['canal_destino']}" if radar.get("canal_destino") else ""

        with st.container(border=True):
            rc1, rc2, rc3 = st.columns([3.0, 1.2, 0.8])
            with rc1:
                st.markdown(f"**📡 {radar['nombre']}**{canal_lbl}")
                st.caption(" · ".join(crit_parts))
            with rc2:
                st.metric("Coincidencias Hoy", _fmt_int(len(matched_df)))
            with rc3:
                if st.button("🗑️ Eliminar", key=f"rad_del_{rid}", width="stretch"):
                    _get_db_manager().delete_radar_alert(rid, owner_id=workspace_id)
                    st.session_state["_radar_nonce"] = (
                        int(st.session_state.get("_radar_nonce", 0)) + 1
                    )
                    _queue_toast(f"Radar '{radar['nombre']}' eliminado.", icon="🗑️")
                    _safe_rerun_fragment()

            with st.expander(f"👁️ Vista rápida ({len(matched_df)} oportunidades activas)", expanded=False):
                if matched_df.empty:
                    st.caption("Ninguna publicación activa cumple hoy con este radar.")
                else:
                    preview_df = pd.DataFrame(
                        {
                            "Portal": matched_df.get("portal_badge", matched_df["id_aviso"]),
                            "ID Aviso": matched_df["id_aviso"],
                            "Marca": matched_df["marca"],
                            "Modelo": matched_df["modelo"],
                            "Año": matched_df["anio"].astype(int),
                            "Km": matched_df["kilometraje"].astype(int),
                            "Comuna": matched_df["comuna"],
                            "Precio": matched_df["precio"].astype(int),
                            "Valor IA": matched_df["precio_justo_ia"].round(0).astype(int),
                            "Dcto (%)": (matched_df["descuento_pct"] * 100.0).round(1),
                            "URL": matched_df["url"],
                        }
                    ).head(25)
                    st.dataframe(
                        preview_df,
                        column_config={
                            "Año": st.column_config.NumberColumn(format="%d"),
                            "Km": st.column_config.NumberColumn(format="%d km"),
                            "Precio": st.column_config.NumberColumn(format="$ %d"),
                            "Valor IA": st.column_config.NumberColumn(format="$ %d"),
                            "Dcto (%)": st.column_config.NumberColumn(format="-%.1f %%"),
                            "URL": st.column_config.LinkColumn("Enlace", display_text="Abrir ↗"),
                        },
                        hide_index=True,
                        width="stretch",
                    )
                    _render_quick_crm_save_bar(
                        matched_df,
                        workspace_id=workspace_id,
                        key_prefix=f"rad_{rid}",
                        heading="Guardar coincidencia en Pipeline CRM",
                    )


@st.fragment
def _render_pipeline_crm_fragment(data_bundle: Dict[str, Any]) -> None:
    """Vista 2 aislada: Pipeline CRM y Radares de Alerta."""
    _flush_pending_toast()
    effective_bundle = apply_session_overrides(data_bundle)
    workspace_id = get_current_workspace_id()

    crm_sub_kwargs = (
        {"default": CRM_SUBVIEWS[0]} if "active_crm_subtab" not in st.session_state else {}
    )
    active_sub = st.segmented_control(
        "Sub-vista CRM:",
        options=CRM_SUBVIEWS,
        selection_mode="single",
        required=True,
        width="stretch",
        label_visibility="collapsed",
        key="active_crm_subtab",
        **crm_sub_kwargs,
    )
    if not active_sub:
        active_sub = CRM_SUBVIEWS[0]

    if active_sub == CRM_SUBVIEWS[0]:
        _render_crm_tracking_subview(
            effective_bundle["full_df"],
            effective_bundle["clean_df"],
            workspace_id,
        )
    else:
        _render_radars_subview(effective_bundle["active_featured"], workspace_id)


def render_tab_pipeline_crm(data_bundle: Dict[str, Any]) -> None:
    """Vista 2: Pipeline CRM y Radares de Alerta."""
    _render_pipeline_crm_fragment(data_bundle)


# ==============================================================================
# 9. VISTA 3: INTELIGENCIA DE MERCADO Y CURVAS
# ==============================================================================
@st.cache_data(show_spinner=False)
def _build_treemap_figure(tree_df: pd.DataFrame, plotly_template: str) -> go.Figure:
    """Construye y cachea el gráfico Treemap jerárquico del mercado."""
    fig_tree = px.treemap(
        tree_df,
        path=[px.Constant("Mercado Chileno"), "marca", "modelo", "generacion"],
        values="unidades",
        color="precio_medio",
        color_continuous_scale="Turbo",
        hover_data={"precio_medio": ":$,.0f", "descuento_max_pct": True},
        title="Mapa Jerárquico del Parque Automotriz (Volumen y Precio Medio)",
        template=plotly_template,
    )
    fig_tree.update_layout(height=440, margin=dict(l=10, r=10, t=45, b=10))
    return fig_tree


@st.fragment
def _render_market_fragment(data_bundle: Dict[str, Any]) -> None:
    """Vista 3 aislada: Curvas 2D, Nube 3D, Generación, Mapa de Calor y Panorama Ejecutivo."""
    _flush_pending_toast()

    effective_bundle = apply_session_overrides(data_bundle)
    clean_df = effective_bundle["clean_df"]
    active_df = effective_bundle["active_featured"]
    price_history_df = effective_bundle["price_history_df"]
    model_meta = effective_bundle["model_meta"]
    tree_df = effective_bundle.get("tree_df", pd.DataFrame())
    workspace_id = get_current_workspace_id()

    if clean_df.empty and active_df.empty:
        render_empty_state(
            title="Sin datos suficientes de mercado",
            subtitle="Ejecuta un escaneo en vivo en la pestaña Oportunidades para generar las curvas y métricas de mercado.",
            icon="📈",
        )
        return

    all_segments = _get_cached_target_segments(
        int(st.session_state.get("_segments_nonce", 0))
    )
    target_marcas = {m for m, _ in all_segments}
    db_marcas = set(clean_df["marca"].unique().tolist()) if not clean_df.empty else set()
    act_marcas = set(active_df["marca"].unique().tolist()) if not active_df.empty else set()
    marcas_list = sorted(target_marcas.union(db_marcas).union(act_marcas))

    preferred_marca = st.session_state.get("focused_marca") or "MAZDA"
    if "curve_tab_marca" in st.session_state and st.session_state["curve_tab_marca"] not in marcas_list:
        st.session_state["curve_tab_marca"] = marcas_list[0]
    default_marca_idx = marcas_list.index(preferred_marca) if preferred_marca in marcas_list else 0

    c_sel1, c_sel2, c_sel3, c_sel4 = st.columns([1.3, 1.3, 1.3, 1.3])
    with c_sel1:
        selected_marca = st.selectbox(
            "Marca:",
            options=marcas_list,
            key="curve_tab_marca",
            **({} if "curve_tab_marca" in st.session_state else {"index": default_marca_idx}),
        )

    target_mods = {str(mod) for m, mod in all_segments if m == selected_marca}
    db_mods = (
        set(clean_df.loc[clean_df["marca"] == selected_marca, "modelo"].astype(str).unique())
        if not clean_df.empty
        else set()
    )
    act_mods = (
        set(active_df.loc[active_df["marca"] == selected_marca, "modelo"].astype(str).unique())
        if not active_df.empty
        else set()
    )
    modelos_list = ["TODOS LOS MODELOS DE LA MARCA"] + sorted(
        target_mods.union(db_mods).union(act_mods)
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
            "Modelo:",
            options=modelos_list,
            key="curve_tab_modelo",
            **({} if "curve_tab_modelo" in st.session_state else {"index": default_mod_idx}),
        )

    xaxis_opts = [
        "Año de Fabricación",
        "Kilometraje (km)",
        "Intensidad de Uso (km/año)",
        "Antigüedad (años)",
    ]
    with c_sel3:
        x_axis_mode = st.selectbox(
            "Eje X (Curva 2D):",
            options=xaxis_opts,
            key="curve_tab_xaxis",
            **({} if "curve_tab_xaxis" in st.session_state else {"index": 0}),
        )

    color_opts = [
        "Categoría de Arbitraje",
        "Transmisión (Manual vs Automática)",
        "Portal de Origen",
        "Generación del Modelo",
        "Combustible",
    ]
    with c_sel4:
        color_grouping = st.selectbox(
            "Color por:",
            options=color_opts,
            key="curve_tab_color",
            **({} if "curve_tab_color" in st.session_state else {"index": 0}),
        )

    min_y_avail = int(min(1990, clean_df["anio"].min())) if not clean_df.empty else 1990
    with st.expander("🎚️ Filtros Específicos del Segmento", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        trans_all = ["Automática", "Manual"]
        portal_all = ["MercadoLibre", "Autocosmos", "Yapo.cl", "Chileautos"]
        _sanitize_multiselect_state("curve_flt_trans", trans_all)
        _sanitize_multiselect_state("curve_flt_portales", portal_all)
        _sanitize_slider_state("curve_flt_years", min_y_avail, config.MAX_VALID_YEAR)

        with fc1:
            g_trans = st.multiselect(
                "Transmisión:",
                options=trans_all,
                key="curve_flt_trans",
                **({} if "curve_flt_trans" in st.session_state else {"default": trans_all}),
            )
        with fc2:
            g_portales = st.multiselect(
                "Portales:",
                options=portal_all,
                key="curve_flt_portales",
                **({} if "curve_flt_portales" in st.session_state else {"default": portal_all}),
            )
        with fc3:
            g_year_rng = st.slider(
                "Rango de Años:",
                min_value=min_y_avail,
                max_value=config.MAX_VALID_YEAR,
                key="curve_flt_years",
                **(
                    {}
                    if "curve_flt_years" in st.session_state
                    else {"value": (max(min_y_avail, 2005), config.MAX_VALID_YEAR)}
                ),
            )
        with fc4:
            g_max_km = st.slider(
                "Kilometraje Máximo:",
                min_value=20_000,
                max_value=config.MAX_VALID_KM,
                step=10_000,
                key="curve_flt_km",
                **(
                    {}
                    if "curve_flt_km" in st.session_state
                    else {"value": min(400_000, config.MAX_VALID_KM)}
                ),
            )

    if selected_modelo == "TODOS LOS MODELOS DE LA MARCA":
        sub_clean_base = clean_df.loc[clean_df["marca"] == selected_marca] if not clean_df.empty else pd.DataFrame()
        sub_active_base = active_df.loc[active_df["marca"] == selected_marca] if not active_df.empty else pd.DataFrame()
        model_title_label = f"{selected_marca} (Todos)"
    else:
        sub_clean_base = (
            clean_df.loc[
                (clean_df["marca"] == selected_marca)
                & (clean_df["modelo"].astype(str) == str(selected_modelo))
            ]
            if not clean_df.empty
            else pd.DataFrame()
        )
        sub_active_base = (
            active_df.loc[
                (active_df["marca"] == selected_marca)
                & (active_df["modelo"].astype(str) == str(selected_modelo))
            ]
            if not active_df.empty
            else pd.DataFrame()
        )
        model_title_label = f"{selected_marca} {selected_modelo}"

    sub_clean = (
        sub_clean_base.loc[
            sub_clean_base["transmision"].isin(g_trans)
            & sub_clean_base["portal"].isin(g_portales)
            & (sub_clean_base["anio"] >= g_year_rng[0])
            & (sub_clean_base["anio"] <= g_year_rng[1])
            & (sub_clean_base["kilometraje"] <= g_max_km)
        ].copy()
        if not sub_clean_base.empty
        else pd.DataFrame()
    )
    sub_active = (
        sub_active_base.loc[
            sub_active_base["transmision"].isin(g_trans)
            & sub_active_base["portal"].isin(g_portales)
            & (sub_active_base["anio"] >= g_year_rng[0])
            & (sub_active_base["anio"] <= g_year_rng[1])
            & (sub_active_base["kilometraje"] <= g_max_km)
        ].copy()
        if not sub_active_base.empty
        else pd.DataFrame()
    )

    subview_kwargs = (
        {"default": MARKET_SUBVIEWS[0]}
        if "active_curve_subtab" not in st.session_state
        else {}
    )
    active_subview = st.segmented_control(
        "Sub-vista de Mercado:",
        options=MARKET_SUBVIEWS,
        selection_mode="single",
        required=True,
        width="stretch",
        label_visibility="collapsed",
        key="active_curve_subtab",
        **subview_kwargs,
    )
    if not active_subview:
        active_subview = MARKET_SUBVIEWS[0]

    # Permitir Panorama Ejecutivo aunque el modelo puntual seleccionado esté vacío
    if sub_clean.empty and sub_active.empty and active_subview != MARKET_SUBVIEWS[4]:
        render_empty_state(
            title=f"Sin registros para {model_title_label}",
            subtitle="No hay unidades que coincidan con los filtros actuales para este segmento.",
            icon="📉",
        )
        if is_admin_unlocked():
            mod_for_scan = (
                None if selected_modelo == "TODOS LOS MODELOS DE LA MARCA" else str(selected_modelo)
            )
            if st.button(
                f"🚀 Capturar {model_title_label} en Portales Ahora",
                key="curve_empty_fetch_btn",
                type="primary",
            ):
                allowed, wait_sec = check_scan_cooldown()
                if not allowed:
                    st.warning(f"Enfriamiento activo: espera {wait_sec}s.")
                    return
                record_scan_triggered()
                with st.status(f"Capturando {model_title_label}...", expanded=True) as status_box:
                    ok, msg = execute_backend_cycle(
                        force_train=False,
                        marca_filter=selected_marca,
                        modelo_filter=mod_for_scan,
                        pages_per_portal=2,
                        max_telegram_alerts=5,
                        deep=True,
                        status_box=status_box,
                    )
                    status_box.update(
                        label="Captura completada" if ok else "Error en captura",
                        state="complete" if ok else "error",
                        expanded=False,
                    )
                if ok:
                    st.session_state["last_scan_message"] = msg
                    st.rerun(scope="app")
                else:
                    st.error(msg)
        return

    x_col_map = {
        "Año de Fabricación": ("anio", "Año de Fabricación"),
        "Kilometraje (km)": ("kilometraje", "Kilometraje (km)"),
        "Intensidad de Uso (km/año)": ("km_por_anio", "Kilometraje por Año (km/año)"),
        "Antigüedad (años)": ("antiguedad", "Antigüedad (años)"),
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

    if active_subview == MARKET_SUBVIEWS[0]:  # Curva 2D
        fig = go.Figure()
        if not normal_pts.empty:
            if color_grouping == "Categoría de Arbitraje":
                fig.add_trace(
                    go.Scatter(
                        x=normal_pts[x_col],
                        y=normal_pts["precio"],
                        mode="markers",
                        name="Precio Regular",
                        marker=dict(size=9, color="#38bdf8", opacity=0.78, line=dict(width=1, color="#0284c7")),
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
                            "Gen: %{customdata[6]} | %{customdata[7]}<br>"
                            "Km: %{customdata[3]:,.0f} km | %{customdata[4]}<br>"
                            "Precio: <b>$ %{y:,.0f}</b> | IA: $ %{customdata[5]:,.0f}<extra></extra>"
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
                            name=str(g_val),
                            marker=dict(size=9, color=palette[idx_g % len(palette)], opacity=0.82),
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
                                "Km: %{customdata[3]:,.0f} km | %{customdata[4]}<br>"
                                "Precio: <b>$ %{y:,.0f}</b> | IA: $ %{customdata[5]:,.0f}<extra></extra>"
                            ),
                        )
                    )

        curve_base = sub_clean if not sub_clean.empty else sub_active
        if len(curve_base) >= 2:
            x_line, y_line = None, None
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
                if len(np.unique(x_vals)) >= 2:
                    x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 60)
                    deg = 2 if len(np.unique(x_vals)) >= 4 else 1
                    try:
                        coeffs = np.polyfit(x_vals, np.log(np.maximum(y_vals, 500_000)), deg=deg)
                        y_line = np.exp(np.polyval(coeffs, x_line))
                    except Exception:
                        try:
                            coeffs = np.polyfit(x_vals, y_vals, deg=1)
                            y_line = np.polyval(coeffs, x_line)
                        except Exception:
                            x_line, y_line = None, None
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
                        name="Curva Precio Justo IA",
                        line=dict(color="#a855f7", width=3.2),
                        hovertemplate="Precio Justo IA: <b>$ %{y:,.0f}</b><extra></extra>",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line * (1.0 - config.OPPORTUNITY_DISCOUNT_THRESHOLD),
                        mode="lines",
                        name="Umbral Oportunidad (-15% IA)",
                        line=dict(color="#10b981", width=2.0, dash="dot"),
                        hovertemplate="Umbral -15%: <b>$ %{y:,.0f}</b><extra></extra>",
                    )
                )
        elif len(curve_base) == 1:
            st.caption("Muestra de 1 unidad en el segmento: mostrando punto individual.")

        if not opp_pts.empty:
            fig.add_trace(
                go.Scatter(
                    x=opp_pts[x_col],
                    y=opp_pts["precio"],
                    mode="markers+text",
                    name="Oportunidad (15%–55% Dcto)",
                    text=[f"-{r['descuento_pct']*100:.0f}%" for _, r in opp_pts.iterrows()],
                    textposition="bottom center",
                    textfont=dict(color="#34d399", size=11),
                    marker=dict(size=13, color="#10b981", symbol="diamond", line=dict(width=1.5, color="#ffffff")),
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
                        f"<b>OPORTUNIDAD: {model_title_label} (%{{customdata[2]}})</b><br>"
                        "Portal: %{customdata[1]} (%{customdata[0]})<br>"
                        "Km: %{customdata[3]:,.0f} km | %{customdata[4]}<br>"
                        "Precio: <b>$ %{y:,.0f}</b> | IA: $ %{customdata[5]:,.0f}<br>"
                        "Margen Bruto: +$ %{customdata[6]:,.0f} (-%{customdata[7]:.1f}%)<extra></extra>"
                    ),
                )
            )

        if not scam_pts.empty:
            fig.add_trace(
                go.Scatter(
                    x=scam_pts[x_col],
                    y=scam_pts["precio"],
                    mode="markers",
                    name="Revisar (>=55% Dcto)",
                    marker=dict(size=12, color="#ef4444", symbol="triangle-up"),
                )
            )

        fig.update_layout(
            title=f"Curva de Depreciación: {model_title_label} ({x_label} vs CLP)",
            xaxis_title=x_label,
            yaxis_title="Precio (CLP)",
            yaxis=dict(tickprefix="$ ", separatethousands=True),
            xaxis=dict(separatethousands=x_col in ("kilometraje", "km_por_anio")),
            template=get_plotly_template(),
            height=500,
            hovermode="closest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=35, r=25, t=65, b=35),
        )
        st.plotly_chart(fig, width="stretch")

    elif active_subview == MARKET_SUBVIEWS[1]:  # Nube 3D
        plot3d_df = sub_active if not sub_active.empty else sub_clean
        if plot3d_df.empty:
            render_empty_state(
                "Sin datos para nube 3D",
                "Ajusta los filtros de segmento para visualizar la dispersión tridimensional.",
                icon="🧊",
            )
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
                    "comuna": True,
                    "portal": True,
                },
                title=f"Espacio 3D: {model_title_label} (Año × Km × Precio)",
                template=get_plotly_template(),
            )
            fig_3d.update_layout(height=540, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_3d, width="stretch")

    elif active_subview == MARKET_SUBVIEWS[2]:  # Por Generación
        box_df = sub_clean if not sub_clean.empty else sub_active
        if box_df.empty:
            render_empty_state(
                "Sin datos por generación",
                "No hay unidades suficientes para comparar generaciones o portales.",
                icon="📦",
            )
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
                    title=f"Dispersión por Generación ({model_title_label})",
                    labels={"generacion": "Generación", "precio": "Precio (CLP)"},
                    template=get_plotly_template(),
                )
                fig_box_gen.update_layout(
                    height=440,
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
                    title=f"Precios por Portal y Transmisión ({model_title_label})",
                    labels={"portal": "Portal", "precio": "Precio (CLP)"},
                    template=get_plotly_template(),
                )
                fig_box_portal.update_layout(
                    height=440,
                    yaxis=dict(tickprefix="$ ", separatethousands=True),
                )
                st.plotly_chart(fig_box_portal, width="stretch")

    elif active_subview == MARKET_SUBVIEWS[3]:  # Mapa de Calor
        hm_base = sub_clean if not sub_clean.empty else sub_active
        hm1, hm2 = st.columns(2)
        with hm1:
            pivot_price = (
                hm_base.pivot_table(
                    index="transmision",
                    columns="anio",
                    values="precio",
                    aggfunc="median",
                )
                .dropna(how="all")
                .dropna(axis=1, how="all")
                if not hm_base.empty
                else pd.DataFrame()
            )
            if pivot_price.empty:
                render_empty_state(
                    "Matriz de precios vacía",
                    "Se requieren más combinaciones de Año y Transmisión.",
                    icon="🔥",
                )
            else:
                fig_hm_price = px.imshow(
                    pivot_price,
                    text_auto=".2s",
                    aspect="auto",
                    color_continuous_scale="Tealgrn",
                    title=f"Precio Mediano por Año y Transmisión ({model_title_label})",
                    labels=dict(x="Año", y="Transmisión", color="Precio Mediano"),
                    template=get_plotly_template(),
                )
                fig_hm_price.update_layout(height=410)
                st.plotly_chart(fig_hm_price, width="stretch")
        with hm2:
            pivot_disc = (
                sub_active.pivot_table(
                    index="generacion",
                    columns="portal",
                    values="margen_ganancia_clp",
                    aggfunc="max",
                )
                .dropna(how="all")
                .dropna(axis=1, how="all")
                if not sub_active.empty
                else pd.DataFrame()
            )
            if pivot_disc.empty:
                render_empty_state(
                    "Matriz de márgenes vacía",
                    "Sin avisos activos suficientes por Generación × Portal.",
                    icon="💰",
                )
            else:
                fig_hm_disc = px.imshow(
                    pivot_disc,
                    text_auto=".2s",
                    aspect="auto",
                    color_continuous_scale="RdYlGn",
                    title=f"Mayor Margen por Generación y Portal ({model_title_label})",
                    labels=dict(x="Portal", y="Generación", color="Máx Margen CLP"),
                    template=get_plotly_template(),
                )
                fig_hm_disc.update_layout(height=410)
                st.plotly_chart(fig_hm_disc, width="stretch")

    else:  # Panorama Ejecutivo
        exec_summary = compute_market_executive_summary(clean_df, active_df, price_history_df)
        ek = exec_summary["kpis"]
        render_kpi_strip(
            [
                {"label": "Activos en Mercado", "value": _fmt_int(ek["total_activos"]), "sub": f"{ek['marcas_activas']} marcas / {ek['modelos_activos']} modelos", "tone": "neutral"},
                {"label": "Oportunidades Activas", "value": _fmt_int(ek["total_oportunidades"]), "sub": f"Tasa: {ek['tasa_oportunidad_pct']:.1f}%", "tone": "emerald"},
                {"label": "Precio Mediano", "value": format_clp(ek["precio_mediano_mercado_clp"]), "sub": f"Dcto medio opp: -{ek['descuento_medio_oportunidades_pct']:.1f}%", "tone": "blue"},
                {"label": "Rebajas Detectadas", "value": _fmt_int(ek["rebajas_detectadas"]), "sub": f"Prom: {format_clp(ek['rebaja_promedio_clp'])}", "tone": "amber"},
                {"label": "Modelo Más Líquido", "value": str(ek["modelo_mas_liquido"]), "sub": f"Portal líder: {ek['portal_lider']}", "tone": "emerald"},
            ]
        )

        tm_col1, tm_col2 = st.columns([1.35, 1.0])
        with tm_col1:
            if tree_df.empty:
                render_empty_state("Sin datos para Treemap", "No hay muestra suficiente.", icon="🗺️")
            else:
                st.plotly_chart(
                    _build_treemap_figure(tree_df, get_plotly_template()), width="stretch"
                )
        with tm_col2:
            top_features = model_meta.get("top_10_features", {})
            if top_features:
                feat_names = list(top_features.keys())[::-1]
                feat_vals = list(top_features.values())[::-1]
                fig_imp = go.Figure(
                    go.Bar(x=feat_vals, y=feat_names, orientation="h", marker=dict(color="#10b981"))
                )
                fig_imp.update_layout(
                    title="Top 10 Variables del Modelo IA",
                    xaxis_title="Importancia",
                    yaxis_title="Variable",
                    template=get_plotly_template(),
                    height=440,
                    margin=dict(l=20, r=20, t=45, b=20),
                )
                st.plotly_chart(fig_imp, width="stretch")

        t1, t2, t3 = st.columns(3)
        with t1:
            st.markdown("#### 🔥 Modelos Más Líquidos")
            st.dataframe(
                exec_summary["top_liquid_models"],
                column_config={
                    "precio_medio": st.column_config.NumberColumn("P. Medio", format="$ %d"),
                    "precio_mediano": st.column_config.NumberColumn("Mediana", format="$ %d"),
                    "descuento_medio_pct": st.column_config.NumberColumn("Dcto %", format="%.1f %%"),
                },
                hide_index=True,
                width="stretch",
                height=300,
            )
        with t2:
            st.markdown("#### 🏆 Ranking por Marca")
            st.dataframe(
                exec_summary["brand_opportunity_ranking"],
                column_config={
                    "tasa_oportunidad_pct": st.column_config.NumberColumn("Tasa %", format="%.1f %%"),
                    "margen_potencial_medio_clp": st.column_config.NumberColumn("Margen Med.", format="$ %d"),
                    "descuento_medio_pct": st.column_config.NumberColumn("Dcto %", format="%.1f %%"),
                },
                hide_index=True,
                width="stretch",
                height=300,
            )
        with t3:
            st.markdown("#### 🌐 Benchmark por Portal")
            st.dataframe(
                exec_summary["portal_benchmark"],
                column_config={
                    "participacion_pct": st.column_config.NumberColumn("Share %", format="%.1f %%"),
                    "precio_promedio_clp": st.column_config.NumberColumn("P. Prom.", format="$ %d"),
                    "descuento_medio_pct": st.column_config.NumberColumn("Dcto %", format="%.1f %%"),
                },
                hide_index=True,
                width="stretch",
                height=300,
            )

    # KPIs del segmento seleccionado y selector rápido hacia Pipeline CRM
    if active_subview != MARKET_SUBVIEWS[4]:
        kpi_base = sub_clean if not sub_clean.empty else sub_active
        render_kpi_strip(
            [
                {"label": "Muestras en Vista", "value": _fmt_int(len(kpi_base)), "tone": "neutral"},
                {"label": "Precio Mediano", "value": format_clp(kpi_base["precio"].median() if not kpi_base.empty else 0), "tone": "blue"},
                {"label": "Valor Justo IA Medio", "value": format_clp(kpi_base["precio_justo_ia"].mean() if not kpi_base.empty else 0), "tone": "blue"},
                {"label": "Oportunidades", "value": _fmt_int(len(opp_pts)), "tone": "emerald"},
                {"label": "Alertas Extremas", "value": _fmt_int(len(scam_pts)), "tone": "amber"},
            ]
        )

    quick_pool = sub_active if not sub_active.empty else active_df
    _render_quick_crm_save_bar(
        quick_pool,
        workspace_id=workspace_id,
        key_prefix="mkt",
        heading="Guardar unidad destacada del segmento en Pipeline CRM",
    )


def render_tab_market(data_bundle: Dict[str, Any]) -> None:
    """Vista 3: Inteligencia de Mercado y Curvas."""
    _render_market_fragment(data_bundle)


# ==============================================================================
# 10. VISTA 4: TASADOR IA Y DOSSIER PDF
# ==============================================================================
@st.fragment
def _render_appraiser_fragment(data_bundle: Dict[str, Any]) -> None:
    """Vista 4 aislada: Tasador IA con desglose neto, Dossier PDF y comparables."""
    _flush_pending_toast()

    effective_bundle = apply_session_overrides(data_bundle)
    clean_df = effective_bundle["clean_df"]
    workspace_id = get_current_workspace_id()
    _, model_mtime, _ = _get_current_mtimes()
    predictor, _ = load_cached_predictor(model_mtime)

    all_segments = _get_cached_target_segments(
        int(st.session_state.get("_segments_nonce", 0))
    )
    target_marcas = sorted({m for m, _ in all_segments})
    db_marcas = sorted(clean_df["marca"].unique().tolist()) if not clean_df.empty else []
    all_marcas = sorted(set(target_marcas + db_marcas))
    comunas_list = sorted(
        {c.title() for c in config.COMUNA_WEALTH_TIER.keys()}.union(
            {"Santiago", "Providencia", "Las Condes"}
        )
    )

    hdr1, hdr2 = st.columns([3.0, 1.2])
    with hdr1:
        st.markdown("### 🧮 Tasador IA en Vivo y Dossier Financiero")
        st.caption(
            "Evalúa el valor justo de mercado, la rentabilidad neta real tras gastos de traspaso/preparación y descarga un dossier PDF listo para negociar."
        )
    with hdr2:
        deal_costs = render_deal_cost_controls("tas")

    col_form, col_result = st.columns([1.4, 1.2])

    with col_form:
        f_c1, f_c2 = st.columns(2)
        with f_c1:
            marca_options = all_marcas + ["➕ OTRA MARCA"]
            t_marca_idx = marca_options.index("MAZDA") if "MAZDA" in marca_options else 0
            t_marca_sel = st.selectbox(
                "Marca:",
                options=marca_options,
                key="tasador_marca",
                **({} if "tasador_marca" in st.session_state else {"index": t_marca_idx}),
            )
            if t_marca_sel == "➕ OTRA MARCA":
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
            mod_opts_tas = avail_models + ["➕ OTRO MODELO"]
            if (
                "tasador_modelo" in st.session_state
                and st.session_state["tasador_modelo"] not in mod_opts_tas
            ):
                st.session_state["tasador_modelo"] = mod_opts_tas[0]

            t_modelo_sel = st.selectbox(
                "Modelo:",
                options=mod_opts_tas,
                key="tasador_modelo",
                **({} if "tasador_modelo" in st.session_state else {"index": 0}),
            )
            if t_modelo_sel == "➕ OTRO MODELO":
                t_modelo_raw = st.text_input(
                    "Escribe el Modelo:", value="A4", key="tasador_custom_modelo"
                )
                t_modelo_norm = normalize_model_input(t_modelo_raw) or "GENERICO"
            else:
                t_modelo_norm = normalize_model_input(t_modelo_sel) or str(t_modelo_sel)

        f_c3, f_c4, f_c5 = st.columns(3)
        with f_c3:
            t_anio = st.number_input(
                "Año:",
                min_value=config.MIN_VALID_YEAR,
                max_value=config.MAX_VALID_YEAR,
                step=1,
                key="tasador_anio",
                **({} if "tasador_anio" in st.session_state else {"value": 2020}),
            )
        with f_c4:
            t_km = st.number_input(
                "Kilometraje (km):",
                min_value=0,
                max_value=config.MAX_VALID_KM,
                step=5_000,
                key="tasador_km",
                **({} if "tasador_km" in st.session_state else {"value": 65_000}),
            )
        with f_c5:
            t_precio_oferta = st.number_input(
                "Precio Oferta (CLP):",
                min_value=config.MIN_VALID_PRICE_CLP,
                max_value=config.MAX_VALID_PRICE_CLP,
                step=250_000,
                key="tasador_precio_oferta",
                **({} if "tasador_precio_oferta" in st.session_state else {"value": 10_500_000}),
            )

        f_c6, f_c7, f_c8 = st.columns(3)
        with f_c6:
            t_trans = st.selectbox(
                "Transmisión:",
                ["Automática", "Manual"],
                key="tasador_trans",
                **({} if "tasador_trans" in st.session_state else {"index": 0}),
            )
        with f_c7:
            t_comb = st.selectbox(
                "Combustible:",
                ["Bencina", "Diésel", "Híbrido", "Eléctrico"],
                key="tasador_comb",
                **({} if "tasador_comb" in st.session_state else {"index": 0}),
            )
        with f_c8:
            prov_idx = comunas_list.index("Providencia") if "Providencia" in comunas_list else 0
            t_comuna = st.selectbox(
                "Comuna:",
                options=comunas_list,
                key="tasador_comuna",
                **({} if "tasador_comuna" in st.session_state else {"index": prov_idx}),
            )

        st.markdown("#### Equipamiento y Atributos NLP")
        chk1, chk2, chk3 = st.columns(3)
        with chk1:
            c_unico = st.checkbox("Único dueño", value=True, key="tasador_chk_unico")
            c_mant = st.checkbox("Mantenciones al día", value=True, key="tasador_chk_mant")
            c_cuero = st.checkbox("Asientos de cuero", value=False, key="tasador_chk_cuero")
        with chk2:
            c_laminas = st.checkbox("Láminas de seguridad", value=False, key="tasador_chk_laminas")
            c_neum = st.checkbox("Neumáticos nuevos", value=False, key="tasador_chk_neum")
            c_sunroof = st.checkbox("Sunroof / Panorámico", value=False, key="tasador_chk_sunroof")
        with chk3:
            c_full = st.checkbox("Full equipo", value=True, key="tasador_chk_full")
            c_inmac = st.checkbox("Impecable", value=True, key="tasador_chk_inmac")
            c_apuro = st.checkbox("Vende por apuro", value=False, key="tasador_chk_apuro")

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
    deal_metrics = calculate_net_deal_metrics(
        precio_publicado=float(t_precio_oferta),
        precio_justo_ia=precio_justo_ia,
        costs=deal_costs,
    )
    margen_bruto = float(deal_metrics["margen_bruto_clp"])
    descuento_pct = (margen_bruto / precio_justo_ia) if precio_justo_ia > 0 else 0.0
    gen_code = CarDataPipeline.resolve_generation(t_marca, str(t_modelo_norm), int(t_anio))
    comuna_tier = CarDataPipeline.resolve_comuna_tier(t_comuna)
    km_fmt = _fmt_int(t_km)

    # Preparar comparables antes para informar PDF y tarjeta
    same_model_df = (
        clean_df.loc[
            (clean_df["marca"] == t_marca) & (clean_df["modelo"] == str(t_modelo_norm))
        ]
        if not clean_df.empty
        else pd.DataFrame()
    )
    comp_df = (
        same_model_df.loc[same_model_df["anio"].sub(int(t_anio)).abs() <= 2].copy()
        if not same_model_df.empty
        else pd.DataFrame()
    )
    range_label = f"±2 años ({int(t_anio) - 2}–{int(t_anio) + 2})"
    if len(comp_df) < 3 and len(same_model_df) > len(comp_df):
        comp_df = same_model_df.copy()
        range_label = "Todos los años del modelo"

    comp_ranked = (
        comp_df.assign(
            _dist=(comp_df["anio"] - int(t_anio)).abs() * 15_000
            + (comp_df["kilometraje"] - int(t_km)).abs()
        ).nsmallest(10, "_dist")
        if not comp_df.empty
        else pd.DataFrame()
    )

    tas_enriched = enrich_dataframe_with_net_deal_and_confidence(
        pd.DataFrame(
            [
                {
                    "id_aviso": "",
                    "marca": t_marca,
                    "modelo": str(t_modelo_norm),
                    "anio": int(t_anio),
                    "kilometraje": int(t_km),
                    "precio": int(t_precio_oferta),
                    "precio_justo_ia": precio_justo_ia,
                    "descuento_pct": descuento_pct,
                }
            ]
        ),
        reference_df=clean_df,
        costs=deal_costs,
    )
    tas_meta = tas_enriched.iloc[0]

    with col_result:
        st.markdown("#### Resultado de Valoración IA")
        conf_lbl_raw = str(tas_meta["confianza_ia_label"])
        conf_lbl_lower = conf_lbl_raw.lower()
        conf_cls = (
            "conf-high"
            if "alta" in conf_lbl_lower
            else ("conf-med" if "media" in conf_lbl_lower else "conf-low")
        )
        val_html = clean_html(
            f"""
            <div class="opp-card tier-good">
                <div class="car-header">
                    <span class="car-title">{get_vehicle_emoji(t_marca, str(t_modelo_norm))} {html.escape(t_marca)} {html.escape(str(t_modelo_norm))} ({int(t_anio)})</span>
                    <div class="car-badges">
                        <span class="comuna-pill">{html.escape(str(t_comuna))} (Tier {comuna_tier}/5)</span>
                        <span class="conf-badge {conf_cls}">Confianza {html.escape(conf_lbl_raw)}: {int(tas_meta['confianza_ia_score'])}/100</span>
                    </div>
                </div>
                <div class="car-meta">
                    <div class="car-meta-row">{km_fmt} km · {html.escape(str(t_trans))} · {html.escape(str(t_comb))} · <code>{html.escape(gen_code)}</code></div>
                    <div class="car-meta-row">{html.escape(str(tas_meta['liquidez_label']))} · {int(tas_meta['comparables_count'])} comparables</div>
                </div>
                <div class="price-grid">
                    <div>
                        <div class="price-label">Precio Justo IA (Venta)</div>
                        <div class="price-val-ia">{format_clp(precio_justo_ia)}</div>
                    </div>
                    <div>
                        <div class="price-label">Compra Máx. Sugerida</div>
                        <div class="price-val-target">{format_clp(deal_metrics['precio_max_compra_sugerido_clp'])}</div>
                    </div>
                </div>
                <div class="{'profit-banner' if deal_metrics['es_rentable_neto'] else 'profit-banner is-negative'}">
                    <span>Margen Neto: {format_clp(deal_metrics['margen_neto_clp'])}</span>
                    <span>ROI Neto: {deal_metrics['roi_neto_pct']:+.1f}%</span>
                </div>
            </div>
            """
        )
        st.markdown(val_html, unsafe_allow_html=True)

        # Desglose financiero neto completo
        with st.expander("💵 Desglose Financiero Neto Completo", expanded=True):
            breakdown_df = pd.DataFrame(
                [
                    {"Concepto": "Valor Estimado Venta (IA)", "Monto (CLP)": deal_metrics["precio_venta_estimado"]},
                    {"Concepto": "Precio Oferta / Compra", "Monto (CLP)": -deal_metrics["precio_compra"]},
                    {"Concepto": "Margen Bruto", "Monto (CLP)": deal_metrics["margen_bruto_clp"]},
                    {"Concepto": "Transferencia + RVM", "Monto (CLP)": -deal_metrics["costo_transferencia_clp"]},
                    {"Concepto": "Certificado Autofact", "Monto (CLP)": -deal_metrics["costo_autofact_clp"]},
                    {"Concepto": "Reacondicionamiento", "Monto (CLP)": -deal_metrics["costo_reacondicionamiento_clp"]},
                    {"Concepto": "Costo Financiero Stock", "Monto (CLP)": -deal_metrics["costo_financiero_clp"]},
                    {"Concepto": "Comisión Venta", "Monto (CLP)": -deal_metrics["comision_venta_clp"]},
                    {"Concepto": "Inversión Total Requerida", "Monto (CLP)": deal_metrics["inversion_total_clp"]},
                    {"Concepto": "Margen Neto Final", "Monto (CLP)": deal_metrics["margen_neto_clp"]},
                ]
            )
            st.dataframe(
                breakdown_df,
                column_config={
                    "Monto (CLP)": st.column_config.NumberColumn(format="$ %d"),
                },
                hide_index=True,
                width="stretch",
            )

        if descuento_pct >= config.SUSPICIOUS_DISCOUNT_THRESHOLD:
            st.error(
                f"🚨 Alerta de revisión ({descuento_pct * 100:.1f}% bajo IA): validar documentación, prenda o kilometraje."
            )
        elif deal_metrics["es_rentable_neto"] and descuento_pct >= config.OPPORTUNITY_DISCOUNT_THRESHOLD:
            st.success(
                f"🔥 Oportunidad rentable: margen neto proyectado de **{format_clp(deal_metrics['margen_neto_clp'])}** ({deal_metrics['roi_neto_pct']:+.1f}% ROI neto)."
            )
        elif deal_metrics["es_rentable_neto"]:
            st.info(
                f"Margen neto positivo ajustado ({format_clp(deal_metrics['margen_neto_clp'])}). Compra máxima sugerida: **{format_clp(deal_metrics['precio_max_compra_sugerido_clp'])}**."
            )
        else:
            st.warning(
                f"Operación sin margen neto positivo con los costos actuales. Negocia bajo **{format_clp(deal_metrics['precio_max_compra_sugerido_clp'])}**."
            )

        brand_pdf = st.text_input(
            "Nombre / Automotora en el PDF:",
            key="tas_pdf_brand_name",
            **(
                {}
                if "tas_pdf_brand_name" in st.session_state
                else {"value": "AutoPulse AI Chile"}
            ),
        )
        vehicle_payload = {
            "marca": t_marca,
            "modelo": str(t_modelo_norm),
            "anio": int(t_anio),
            "kilometraje": int(t_km),
            "transmision": t_trans,
            "combustible": t_comb,
            "comuna": t_comuna,
            "region": resolve_region(t_comuna),
            "generacion": gen_code,
            "titulo": synthetic_title,
            "precio": int(t_precio_oferta),
            "precio_justo_ia": precio_justo_ia,
            "descuento_pct": descuento_pct,
            "confianza_ia_label": str(tas_meta["confianza_ia_label"]),
            "confianza_ia_score": int(tas_meta["confianza_ia_score"]),
            "comparables_count": int(tas_meta["comparables_count"]),
            "liquidez_label": str(tas_meta["liquidez_label"]),
            "deal_tier": str(tas_meta["deal_tier"]),
        }
        st.download_button(
            label="📄 Descargar Informe PDF de Tasación",
            data=lambda v_snap=vehicle_payload, m_snap=deal_metrics, c_snap=comp_ranked, b_snap=brand_pdf: generate_valuation_pdf_bytes(
                vehicle=v_snap,
                deal_metrics=m_snap,
                comparables_df=c_snap if not c_snap.empty else None,
                brand_name=sanitize_user_text(b_snap, max_len=60) or "AutoPulse AI Chile",
            ),
            file_name=f"tasacion_{t_marca}_{t_modelo_norm}_{int(t_anio)}.pdf".replace(" ", "_"),
            mime="application/pdf",
            type="primary",
            on_click="ignore",
            key="tas_download_pdf_btn",
            width="stretch",
        )

    # Comparables Reales ±2 años
    st.markdown("---")
    st.markdown(f"#### Comparables Reales ({t_marca} {t_modelo_norm} · {range_label})")
    if comp_df.empty:
        render_empty_state(
            title=f"Sin comparables directos para {t_marca} {t_modelo_norm}",
            subtitle="La tasación se calculó usando la curva jerárquica de marca y segmento del modelo IA.",
            icon="🔍",
        )
        return

    pct_cheaper_than = float((comp_df["precio"] > int(t_precio_oferta)).mean() * 100.0)
    comp_median = float(comp_df["precio"].median())
    comp_min = int(comp_df["precio"].min())
    comp_max = int(comp_df["precio"].max())

    render_kpi_strip(
        [
            {"label": "Comparables", "value": _fmt_int(len(comp_df)), "sub": range_label, "tone": "neutral"},
            {"label": "Mediana Mercado", "value": format_clp(comp_median), "tone": "blue"},
            {"label": "Rango Mín – Máx", "value": f"${comp_min / 1_000_000:.1f}M – ${comp_max / 1_000_000:.1f}M", "tone": "neutral"},
            {"label": "Tu Posición", "value": f"{pct_cheaper_than:.0f}% más caros", "sub": "Frente a tu oferta", "tone": "emerald"},
        ]
    )

    cg1, cg2 = st.columns([1.05, 1.25])
    with cg1:
        fig_comp = px.scatter(
            comp_df,
            x="kilometraje",
            y="precio",
            color="transmision",
            hover_name="id_aviso",
            hover_data={"anio": True, "portal": True, "comuna": True, "precio": ":$,.0f"},
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
                name="Tu Oferta",
                marker=dict(symbol="star", size=17, color="#facc15", line=dict(width=1.5, color="#ffffff")),
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
        comp_table = pd.DataFrame(
            {
                "Portal": comp_ranked["portal_badge"],
                "Año": comp_ranked["anio"].astype(int),
                "Kilometraje": comp_ranked["kilometraje"].astype(int),
                "Transmisión": comp_ranked["transmision"],
                "Comuna": comp_ranked["comuna"],
                "Precio": comp_ranked["precio"].astype(int),
                "Valor IA": comp_ranked["precio_justo_ia"].round(0).astype(int),
                "Enlace": comp_ranked["url"],
            }
        )
        st.dataframe(
            comp_table,
            column_config={
                "Año": st.column_config.NumberColumn(format="%d"),
                "Kilometraje": st.column_config.NumberColumn(format="%d km"),
                "Precio": st.column_config.NumberColumn(format="$ %d"),
                "Valor IA": st.column_config.NumberColumn(format="$ %d"),
                "Enlace": st.column_config.LinkColumn("Enlace", display_text="Abrir ↗"),
            },
            hide_index=True,
            width="stretch",
            height=340,
        )

    _render_quick_crm_save_bar(
        comp_ranked,
        workspace_id=workspace_id,
        key_prefix="tas_comp",
        heading="Guardar comparable en Pipeline CRM",
    )


def render_tab_appraiser(data_bundle: Dict[str, Any]) -> None:
    """Vista 4: Tasador IA."""
    _render_appraiser_fragment(data_bundle)


# ==============================================================================
# 11. VISTA 5: INVENTARIO COMPLETO, REBAJAS Y PANEL DE OPERADOR
# ==============================================================================
@st.fragment
def _render_inventory_fragment(data_bundle: Dict[str, Any]) -> None:
    """Vista 5 aislada: explorador unificado de inventario, rebajas y acciones de operador."""
    _flush_pending_toast()

    effective_bundle = apply_session_overrides(data_bundle)
    full_df = effective_bundle["full_df"]
    clean_df = effective_bundle["clean_df"]
    price_history_df = effective_bundle["price_history_df"]
    workspace_id = get_current_workspace_id()

    if full_df.empty:
        render_empty_state(
            title="Inventario sin registros",
            subtitle="No hay vehículos almacenados todavía en el repositorio de mercado.",
            icon="📦",
        )
        return

    est_col1, est_col2, est_col3 = st.columns([2.0, 1.1, 1.1])
    with est_col1:
        estados_disp = sorted(full_df["estado"].dropna().astype(str).unique().tolist())
        est_key = "exp_estados_flt"
        _sanitize_multiselect_state(est_key, estados_disp)
        sel_estados = st.multiselect(
            "Estado del Aviso:",
            options=estados_disp,
            key=est_key,
            **({} if est_key in st.session_state else {"default": estados_disp}),
        )
    with est_col2:
        deal_costs = render_deal_cost_controls("inv")

    sub_full = full_df.loc[full_df["estado"].isin(sel_estados)] if sel_estados else full_df

    filtered = _apply_saas_filters(
        sub_full,
        key_prefix="exp_flt",
        default_only_opportunities=False,
        allow_category_filter=True,
        clean_reference_df=clean_df,
        deal_costs=deal_costs,
    )

    with est_col3:
        st.download_button(
            label=f"📥 Exportar CSV ({len(filtered)})",
            data=lambda df_exp=filtered: df_exp.to_csv(index=False).encode("utf-8"),
            file_name=f"inventario_autopulse_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            on_click="ignore",
            key="exp_download_csv_btn",
            width="stretch",
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
            "Categoría": filtered["categoria_arbitraje"],
            "Estado": filtered["estado"].str.upper(),
            "Marca": filtered["marca"],
            "Modelo": filtered["modelo"],
            "Año": filtered["anio"].astype(int),
            "Kilometraje": filtered["kilometraje"].astype(int),
            "Comuna": filtered["comuna"],
            "Precio Actual": filtered["precio"].astype(int),
            "Valor IA": filtered["precio_justo_ia"].round(0).astype(int),
            "Margen Neto": filtered["margen_neto_clp"].round(0).astype(int),
            "Dcto (%)": (filtered["descuento_pct"] * 100.0).round(1),
            "Multi-Portal": multi_col,
            "URL": filtered["url"],
        }
    )

    st.dataframe(
        explorer_view,
        column_config={
            "Año": st.column_config.NumberColumn(format="%d"),
            "Kilometraje": st.column_config.NumberColumn(format="%d km"),
            "Precio Actual": st.column_config.NumberColumn(format="$ %d"),
            "Valor IA": st.column_config.NumberColumn(format="$ %d"),
            "Margen Neto": st.column_config.NumberColumn(format="$ %d"),
            "Dcto (%)": st.column_config.NumberColumn(format="%.1f %%"),
            "URL": st.column_config.LinkColumn("Enlace", display_text="Abrir ↗"),
        },
        hide_index=True,
        width="stretch",
        height=440,
    )

    st.markdown("---")
    drops_count = 0
    if not price_history_df.empty and "variacion_clp" in price_history_df.columns:
        drops_mask = pd.to_numeric(price_history_df["variacion_clp"], errors="coerce").fillna(0) < 0
        if "estado" in price_history_df.columns:
            drops_mask &= price_history_df["estado"].fillna("").astype(str).str.lower() == "activo"
        drops_count = int(drops_mask.sum())

    with st.expander(
        f"📉 Radar de Rebajas Recientes ({len(price_history_df)} movimientos | {drops_count} rebajas activas)",
        expanded=True,
    ):
        if price_history_df.empty:
            render_empty_state(
                title="Sin rebajas registradas",
                subtitle="Aún no se han detectado cambios de precio en el historial reciente.",
                icon="📉",
            )
        else:
            only_active_drops = st.toggle(
                "Solo rebajas activas en publicaciones vigentes",
                key="exp_only_active_drops",
                **({} if "exp_only_active_drops" in st.session_state else {"value": True}),
            )
            ph_work = price_history_df.copy()
            if only_active_drops and "variacion_clp" in ph_work.columns:
                mask_d = pd.to_numeric(ph_work["variacion_clp"], errors="coerce").fillna(0) < 0
                if "estado" in ph_work.columns:
                    mask_d &= ph_work["estado"].fillna("").astype(str).str.lower() == "activo"
                ph_work = ph_work.loc[mask_d].copy()

            if ph_work.empty:
                render_empty_state(
                    title="Sin rebajas activas con este filtro",
                    subtitle="Desactiva el switch superior para revisar todos los cambios históricos de precio.",
                    icon="📉",
                )
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
                        "Variación": pd.to_numeric(ph_work.get("variacion_clp", 0), errors="coerce").fillna(0).astype(int),
                        "Var (%)": pd.to_numeric(ph_work.get("variacion_pct", 0.0), errors="coerce").fillna(0.0).round(1),
                        "Fecha": ph_work["fecha_cambio"],
                        "URL": ph_work.get("url", ""),
                    }
                )
                st.dataframe(
                    ph_view,
                    column_config={
                        "Año": st.column_config.NumberColumn(format="%d"),
                        "Precio Antiguo": st.column_config.NumberColumn(format="$ %d"),
                        "Precio Nuevo": st.column_config.NumberColumn(format="$ %d"),
                        "Variación": st.column_config.NumberColumn(format="$ %d"),
                        "Var (%)": st.column_config.NumberColumn(format="%.1f %%"),
                        "URL": st.column_config.LinkColumn("Enlace", display_text="Abrir ↗"),
                    },
                    hide_index=True,
                    width="stretch",
                    height=310,
                )

                # Enriquecer IDs rebajados desde active/full para guardar rápido en CRM
                drop_ids = set(ph_work["id_aviso"].astype(str))
                drop_candidates = full_df[full_df["id_aviso"].astype(str).isin(drop_ids)].copy()
                _render_quick_crm_save_bar(
                    drop_candidates,
                    workspace_id=workspace_id,
                    key_prefix="inv_drops",
                    heading="Guardar unidad rebajada en Pipeline CRM",
                )

    # Controles de Operador protegidos por is_admin_unlocked()
    if is_admin_unlocked():
        st.markdown("---")
        with st.expander("🛠️ Controles de Operador y Mantenimiento", expanded=False):
            op_c1, op_c2, op_c3 = st.columns(3)
            with op_c1:
                if st.button(
                    "🔄 Ejecutar Barrido Completo",
                    key="exp_scan_now_btn",
                    type="primary",
                    width="stretch",
                ):
                    allowed, wait_sec = check_scan_cooldown()
                    if not allowed:
                        st.warning(f"Enfriamiento activo: espera {wait_sec}s.")
                    else:
                        record_scan_triggered()
                        with st.status("Ejecutando barrido general multi-portal...", expanded=True) as status_box:
                            ok, msg = execute_backend_cycle(
                                force_train=False,
                                pages_per_portal=config.SCRAPER_PAGES_PER_SEGMENT,
                                max_telegram_alerts=6,
                                status_box=status_box,
                            )
                            status_box.update(
                                label="Barrido completado" if ok else "Error en barrido",
                                state="complete" if ok else "error",
                                expanded=False,
                            )
                        if ok:
                            st.session_state["last_scan_message"] = msg
                            st.rerun(scope="app")
                        else:
                            st.error(msg)
            with op_c2:
                if st.button(
                    "🧠 Re-entrenar Modelo IA",
                    key="exp_retrain_btn",
                    width="stretch",
                ):
                    with st.status("Re-calibrando modelo IA sobre muestra limpia...", expanded=True) as sbox:
                        predictor = CarPricePredictor(model_path=config.MODEL_PATH)
                        predictor.train_and_evaluate()
                        _clear_all_market_caches()
                        sbox.update(label="Modelo IA actualizado", state="complete", expanded=False)
                    _queue_toast("Modelo IA re-entrenado correctamente.", icon="✅")
                    st.rerun(scope="app")
            with op_c3:
                if st.button(
                    "🧹 Limpiar Avisos Patrocinados",
                    key="exp_cleanup_ads_btn",
                    width="stretch",
                ):
                    cleaned_n = _get_db_manager().cleanup_sponsored_ad_listings()
                    if cleaned_n > 0:
                        _clear_all_market_caches()
                        _queue_toast(f"Se descartaron {cleaned_n} avisos patrocinados.", icon="🧹")
                        st.rerun(scope="app")
                    else:
                        st.info("No se encontraron avisos patrocinados activos.")

            inactive_df = full_df.loc[full_df["estado"].isin(["descartado", "vendido"])].copy()
            st.markdown(f"#### Restaurar Avisos Inactivos ({len(inactive_df)})")
            if inactive_df.empty:
                st.caption("No hay avisos descartados ni vendidos actualmente.")
            else:
                options_map = {
                    f"[{str(r['estado']).upper()}] {r['id_aviso']} — {r['marca']} {r['modelo']} ({int(r['anio'])}) | {format_clp(r['precio'])}": str(
                        r["id_aviso"]
                    )
                    for _, r in inactive_df.head(200).iterrows()
                }
                r1, r2 = st.columns([3.0, 1.2])
                with r1:
                    sel_restore_label = st.selectbox(
                        "Aviso a restaurar:",
                        options=list(options_map.keys()),
                        key="exp_restore_select",
                    )
                with r2:
                    st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
                    if st.button(
                        "↩️ Restaurar a Activo",
                        key="exp_restore_btn",
                        type="primary",
                        width="stretch",
                    ):
                        restore_id = options_map[sel_restore_label]
                        _update_listing_status_instant(restore_id, "activo", sel_restore_label)
                        st.rerun(scope="app")

            st.markdown("#### Registro de Eventos del Sistema")
            l1, l2 = st.columns([4, 1])
            with l1:
                st.caption("Últimas 60 líneas de actividad del motor operativo.")
            with l2:
                if st.button("🔄 Refrescar", key="exp_refresh_log_btn", width="stretch"):
                    _safe_rerun_fragment()
            if config.LOG_FILE_PATH.exists():
                try:
                    lines = config.LOG_FILE_PATH.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()
                    st.code("\n".join(lines[-60:]), language="text")
                except Exception:
                    st.caption("No se pudo leer el registro de eventos.")
            else:
                st.caption("Aún no existen eventos registrados.")
    else:
        st.caption(
            "🔒 Las herramientas de mantenimiento y administración operativa están ocultas hasta desbloquear Acceso Operador en la barra lateral."
        )


def render_tab_inventory(data_bundle: Dict[str, Any]) -> None:
    """Vista 5: Inventario y Rebajas."""
    _render_inventory_fragment(data_bundle)


# ==============================================================================
# 12. PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
def main() -> None:
    """Orquesta la carga desacoplada, tema SaaS, sidebar multi-espacio y 5 vistas."""
    _init_session_defaults()
    _flush_pending_toast()

    is_healthy, health_msg = check_system_health()
    if not is_healthy:
        render_resilience_onboarding(health_msg)
        return

    try:
        db_mtime, model_mtime, clean_mtime = _get_current_mtimes()
        raw_bundle = load_market_datasets(
            db_mtime=db_mtime,
            model_mtime=model_mtime,
            clean_csv_mtime=clean_mtime,
        )
    except Exception:
        render_resilience_onboarding(
            "No fue posible cargar los artefactos de mercado en este momento."
        )
        return

    data_bundle = apply_session_overrides(raw_bundle)
    workspace_id = get_current_workspace_id()
    favorites_map = load_user_favorites(workspace_id, int(st.session_state.get("_fav_nonce", 0)))

    render_sidebar(data_bundle, workspace_id, favorites_map)
    inject_theme_css(is_light_theme())

    active_df = data_bundle["active_featured"]
    opp_count = (
        int((active_df["categoria_arbitraje"] == "OPORTUNIDAD").sum())
        if not active_df.empty and "categoria_arbitraje" in active_df.columns
        else 0
    )
    render_hero_header(
        workspace_id=workspace_id,
        active_count=len(active_df),
        opp_count=opp_count,
    )

    nav_kwargs = (
        {"default": MAIN_SECTIONS[0]} if "active_main_section" not in st.session_state else {}
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
        render_tab_pipeline_crm(data_bundle)
    elif active_section == MAIN_SECTIONS[2]:
        render_tab_market(data_bundle)
    elif active_section == MAIN_SECTIONS[3]:
        render_tab_appraiser(data_bundle)
    else:
        render_tab_inventory(data_bundle)


if __name__ == "__main__":
    main()
