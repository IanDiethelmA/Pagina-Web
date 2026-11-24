"""
services/market_trends.py - Inteligencia de mercado ejecutiva y evaluación de radares.
Calcula resúmenes agregados por modelo, marca y portal, analiza rebajas de precio
históricas y evalúa coincidencias de radares de alerta sobre oportunidades activas.
"""

import unicodedata
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import config

_TOP_LIQUID_COLS = [
    "marca",
    "modelo",
    "unidades_activas",
    "precio_medio",
    "precio_mediano",
    "descuento_medio_pct",
    "oportunidades",
    "dias_promedio",
    "liquidez",
]

_BRAND_RANKING_COLS = [
    "marca",
    "total_unidades",
    "oportunidades",
    "tasa_oportunidad_pct",
    "margen_potencial_medio_clp",
    "descuento_medio_pct",
]

_PORTAL_BENCHMARK_COLS = [
    "portal",
    "publicaciones",
    "participacion_pct",
    "precio_promedio_clp",
    "oportunidades",
    "descuento_medio_pct",
]


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text or ""))
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _normalize_model(raw_model: Any) -> str:
    clean_m = " ".join(str(raw_model or "").strip().upper().split())
    aliases = getattr(config, "MODEL_ALIASES", {})
    if clean_m in aliases:
        return aliases[clean_m]
    compact = clean_m.replace("-", "").replace(" ", "")
    if compact in aliases:
        return aliases[compact]
    return clean_m


def _ensure_portal_series(df: pd.DataFrame) -> pd.Series:
    if "portal" in df.columns and df["portal"].notna().any():
        return df["portal"].fillna("Chileautos").astype(str)
    if "id_aviso" in df.columns:
        uids = df["id_aviso"].fillna("").astype(str).str.strip().str.upper()
        conds = [
            uids.str.startswith("MLC"),
            uids.str.startswith("AC-"),
            uids.str.startswith("YAPO-"),
        ]
        return pd.Series(
            np.select(conds, ["MercadoLibre", "Autocosmos", "Yapo.cl"], default="Chileautos"),
            index=df.index,
        )
    return pd.Series("Chileautos", index=df.index)


def _ensure_discount_fraction(df: pd.DataFrame) -> pd.Series:
    if "descuento_pct" in df.columns:
        return pd.to_numeric(df["descuento_pct"], errors="coerce").fillna(0.0)
    if "precio_justo_ia" in df.columns and "precio" in df.columns:
        p = pd.to_numeric(df["precio"], errors="coerce").fillna(0.0)
        ia = pd.to_numeric(df["precio_justo_ia"], errors="coerce").fillna(0.0)
        return pd.Series(
            np.where(ia > 0.0, (ia - p) / np.maximum(1.0, ia), 0.0),
            index=df.index,
        )
    return pd.Series(0.0, index=df.index)


def _ensure_margin_series(df: pd.DataFrame) -> pd.Series:
    if "margen_neto_clp" in df.columns:
        return pd.to_numeric(df["margen_neto_clp"], errors="coerce").fillna(0.0)
    if "margen_ganancia_clp" in df.columns:
        return pd.to_numeric(df["margen_ganancia_clp"], errors="coerce").fillna(0.0)
    if "precio_justo_ia" in df.columns and "precio" in df.columns:
        p = pd.to_numeric(df["precio"], errors="coerce").fillna(0.0)
        ia = pd.to_numeric(df["precio_justo_ia"], errors="coerce").fillna(0.0)
        return pd.Series(np.where((ia > 0.0) & (p > 0.0), ia - p, 0.0), index=df.index)
    return pd.Series(0.0, index=df.index)


def compute_market_executive_summary(
    clean_df: pd.DataFrame,
    active_df: pd.DataFrame,
    price_history_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calcula tablas ejecutivas de liquidez, ranking por marca, benchmark entre portales
    y KPIs globales del mercado. Soporta DataFrames vacíos o columnas opcionales faltantes.
    """
    opp_thresh = float(getattr(config, "OPPORTUNITY_DISCOUNT_THRESHOLD", 0.15) or 0.15)
    susp_thresh = float(getattr(config, "SUSPICIOUS_DISCOUNT_THRESHOLD", 0.55) or 0.55)

    clean_safe = clean_df if isinstance(clean_df, pd.DataFrame) else pd.DataFrame()
    active_safe = active_df if isinstance(active_df, pd.DataFrame) else pd.DataFrame()
    hist_safe = price_history_df if isinstance(price_history_df, pd.DataFrame) else pd.DataFrame()

    source_df = active_safe if not active_safe.empty else clean_safe

    empty_top = pd.DataFrame(columns=_TOP_LIQUID_COLS)
    empty_brands = pd.DataFrame(columns=_BRAND_RANKING_COLS)
    empty_portals = pd.DataFrame(columns=_PORTAL_BENCHMARK_COLS)

    # Analítica de rebajas en historial de precios
    rebajas_count = 0
    rebaja_promedio_clp = 0
    if not hist_safe.empty and {"precio_antiguo", "precio_nuevo"}.issubset(hist_safe.columns):
        p_old = pd.to_numeric(hist_safe["precio_antiguo"], errors="coerce").fillna(0.0)
        p_new = pd.to_numeric(hist_safe["precio_nuevo"], errors="coerce").fillna(0.0)
        drops = (p_old - p_new)[(p_old > 0) & (p_new > 0) & (p_new < p_old)]
        rebajas_count = int(len(drops))
        rebaja_promedio_clp = int(round(float(drops.mean()))) if rebajas_count > 0 else 0

    if source_df.empty:
        return {
            "top_liquid_models": empty_top,
            "brand_opportunity_ranking": empty_brands,
            "portal_benchmark": empty_portals,
            "kpis": {
                "total_activos": int(len(active_safe)),
                "total_limpios": int(len(clean_safe)),
                "total_oportunidades": 0,
                "tasa_oportunidad_pct": 0.0,
                "precio_mediano_mercado_clp": 0,
                "descuento_medio_oportunidades_pct": 0.0,
                "rebajas_detectadas": rebajas_count,
                "rebaja_promedio_clp": rebaja_promedio_clp,
                "modelo_mas_liquido": "N/A",
                "portal_lider": "N/A",
                "marcas_activas": 0,
                "modelos_activos": 0,
            },
        }

    work = pd.DataFrame(index=source_df.index)
    work["marca"] = (
        source_df["marca"].fillna("DESCONOCIDA").astype(str).str.strip().str.upper()
        if "marca" in source_df.columns
        else "DESCONOCIDA"
    )
    work["modelo"] = (
        source_df["modelo"].fillna("DESCONOCIDO").astype(str).str.strip().str.upper()
        if "modelo" in source_df.columns
        else "DESCONOCIDO"
    )
    work["precio"] = (
        pd.to_numeric(source_df["precio"], errors="coerce").fillna(0.0)
        if "precio" in source_df.columns
        else 0.0
    )
    work["dias_publicado"] = (
        pd.to_numeric(source_df["dias_publicado"], errors="coerce").fillna(0.0)
        if "dias_publicado" in source_df.columns
        else 0.0
    )
    work["portal"] = _ensure_portal_series(source_df)
    work["descuento_pct"] = _ensure_discount_fraction(source_df)
    work["margen_clp"] = _ensure_margin_series(source_df)
    work["es_opp"] = (work["descuento_pct"] >= opp_thresh) & (work["descuento_pct"] < susp_thresh)

    # 1. Modelos más líquidos / ofertados
    top_liquid = (
        work.groupby(["marca", "modelo"], as_index=False)
        .agg(
            unidades_activas=("precio", "count"),
            precio_medio=("precio", "mean"),
            precio_mediano=("precio", "median"),
            descuento_medio_pct=("descuento_pct", lambda s: round(float(s.mean()) * 100.0, 1)),
            oportunidades=("es_opp", "sum"),
            dias_promedio=("dias_publicado", lambda s: round(float(s.mean()), 1)),
        )
        .sort_values(by=["unidades_activas", "oportunidades"], ascending=[False, False])
        .head(20)
        .reset_index(drop=True)
    )
    top_liquid["precio_medio"] = top_liquid["precio_medio"].round(0).astype(int)
    top_liquid["precio_mediano"] = top_liquid["precio_mediano"].round(0).astype(int)
    top_liquid["oportunidades"] = top_liquid["oportunidades"].astype(int)
    top_liquid["liquidez"] = np.select(
        [top_liquid["unidades_activas"] >= 15, top_liquid["unidades_activas"] >= 5],
        ["Alta Rotación", "Rotación Media"],
        default="Nicho",
    )
    top_liquid = top_liquid[_TOP_LIQUID_COLS]

    # 2. Ranking de marcas por oportunidad
    brand_rank = (
        work.groupby("marca", as_index=False)
        .agg(
            total_unidades=("precio", "count"),
            oportunidades=("es_opp", "sum"),
            margen_potencial_medio_clp=("margen_clp", "mean"),
            descuento_medio_pct=("descuento_pct", lambda s: round(float(s.mean()) * 100.0, 1)),
        )
    )
    brand_rank["oportunidades"] = brand_rank["oportunidades"].astype(int)
    brand_rank["tasa_oportunidad_pct"] = (
        (brand_rank["oportunidades"] / np.maximum(1, brand_rank["total_unidades"])) * 100.0
    ).round(1)
    brand_rank["margen_potencial_medio_clp"] = (
        brand_rank["margen_potencial_medio_clp"].round(0).astype(int)
    )
    brand_rank = (
        brand_rank.sort_values(
            by=["oportunidades", "tasa_oportunidad_pct", "total_unidades"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )[_BRAND_RANKING_COLS]

    # 3. Benchmark entre portales
    total_units = max(1, len(work))
    portal_bench = (
        work.groupby("portal", as_index=False)
        .agg(
            publicaciones=("precio", "count"),
            precio_promedio_clp=("precio", "mean"),
            oportunidades=("es_opp", "sum"),
            descuento_medio_pct=("descuento_pct", lambda s: round(float(s.mean()) * 100.0, 1)),
        )
        .sort_values(by="publicaciones", ascending=False)
        .reset_index(drop=True)
    )
    portal_bench["participacion_pct"] = (
        (portal_bench["publicaciones"] / total_units) * 100.0
    ).round(1)
    portal_bench["precio_promedio_clp"] = (
        portal_bench["precio_promedio_clp"].round(0).astype(int)
    )
    portal_bench["oportunidades"] = portal_bench["oportunidades"].astype(int)
    portal_bench = portal_bench[_PORTAL_BENCHMARK_COLS]

    # 4. KPIs ejecutivos
    total_opps = int(work["es_opp"].sum())
    opp_subset = work.loc[work["es_opp"]]
    avg_opp_disc = (
        round(float(opp_subset["descuento_pct"].mean()) * 100.0, 1)
        if not opp_subset.empty
        else 0.0
    )
    valid_prices = work.loc[work["precio"] > 0, "precio"]
    median_price = int(round(float(valid_prices.median()))) if not valid_prices.empty else 0

    top_model_str = (
        f"{top_liquid.iloc[0]['marca']} {top_liquid.iloc[0]['modelo']}"
        if not top_liquid.empty
        else "N/A"
    )
    top_portal_str = (
        str(portal_bench.iloc[0]["portal"])
        if not portal_bench.empty
        else "N/A"
    )

    kpis: Dict[str, Any] = {
        "total_activos": int(len(active_safe)),
        "total_limpios": int(len(clean_safe)),
        "total_oportunidades": total_opps,
        "tasa_oportunidad_pct": round((total_opps / total_units) * 100.0, 1),
        "precio_mediano_mercado_clp": median_price,
        "descuento_medio_oportunidades_pct": avg_opp_disc,
        "rebajas_detectadas": rebajas_count,
        "rebaja_promedio_clp": rebaja_promedio_clp,
        "modelo_mas_liquido": top_model_str,
        "portal_lider": top_portal_str,
        "marcas_activas": int(work["marca"].nunique()),
        "modelos_activos": int(work[["marca", "modelo"]].drop_duplicates().shape[0]),
    }

    return {
        "top_liquid_models": top_liquid,
        "brand_opportunity_ranking": brand_rank,
        "portal_benchmark": portal_bench,
        "kpis": kpis,
    }


def _resolve_region_series(df: pd.DataFrame) -> pd.Series:
    """Obtiene una serie normalizada de región para evaluar radares geográficos."""
    if "region" in df.columns and df["region"].notna().any():
        return df["region"].fillna("").astype(str).map(lambda x: _strip_accents(x).strip().upper())

    if "comuna" in df.columns:
        try:
            from ui.domain import resolve_region

            return df["comuna"].fillna("").astype(str).map(
                lambda c: _strip_accents(resolve_region(c)).strip().upper()
            )
        except Exception:
            return df["comuna"].fillna("").astype(str).map(lambda x: _strip_accents(x).strip().upper())

    return pd.Series("", index=df.index)


def evaluate_radar_matches(
    active_df: pd.DataFrame,
    radars: List[Dict[str, Any]],
) -> Dict[int, pd.DataFrame]:
    """
    Evalúa cada radar guardado contra `active_df` según marca, modelo, anio_min,
    precio_max, km_max, descuento_min_pct (escala 0..100 contra descuento_pct * 100) y region.
    Retorna {int(radar["id"]): df_matches}.
    """
    if not radars:
        return {}

    has_df = isinstance(active_df, pd.DataFrame) and not active_df.empty
    base_cols = active_df.columns if isinstance(active_df, pd.DataFrame) else []
    if not has_df:
        return {
            int(r["id"]): pd.DataFrame(columns=base_cols)
            for r in radars
            if isinstance(r, dict) and r.get("id") is not None
        }

    work = active_df.copy()
    if "estado" in work.columns:
        work = work[work["estado"].fillna("activo").astype(str).str.strip().str.lower() == "activo"].copy()
        if work.empty:
            return {
                int(r["id"]): pd.DataFrame(columns=base_cols)
                for r in radars
                if isinstance(r, dict) and r.get("id") is not None
            }

    marca_s = (
        work["marca"].fillna("").astype(str).str.strip().str.upper()
        if "marca" in work.columns
        else pd.Series("", index=work.index)
    )
    modelo_s = (
        work["modelo"].fillna("").astype(str).map(_normalize_model)
        if "modelo" in work.columns
        else pd.Series("", index=work.index)
    )
    anio_s = (
        pd.to_numeric(work["anio"], errors="coerce").fillna(0).astype(int)
        if "anio" in work.columns
        else pd.Series(0, index=work.index, dtype=int)
    )
    precio_s = (
        pd.to_numeric(work["precio"], errors="coerce").fillna(0.0)
        if "precio" in work.columns
        else pd.Series(0.0, index=work.index)
    )
    km_s = (
        pd.to_numeric(work["kilometraje"], errors="coerce").fillna(0).astype(int)
        if "kilometraje" in work.columns
        else pd.Series(0, index=work.index, dtype=int)
    )
    disc_pct_100 = (_ensure_discount_fraction(work) * 100.0).round(4)
    region_s = _resolve_region_series(work)

    wildcard_tokens = {"", "ALL", "TODAS", "TODOS", "*", "ANY", "CUALQUIERA"}
    results: Dict[int, pd.DataFrame] = {}

    for radar in radars:
        if not isinstance(radar, dict) or radar.get("id") is None:
            continue
        rid = int(radar["id"])
        mask = pd.Series(True, index=work.index)

        r_marca = str(radar.get("marca") or "").strip().upper()
        if r_marca not in wildcard_tokens:
            mask &= marca_s == r_marca

        r_modelo_raw = str(radar.get("modelo") or "").strip().upper()
        if r_modelo_raw not in wildcard_tokens:
            r_modelo = _normalize_model(r_modelo_raw)
            mask &= (modelo_s == r_modelo) | (modelo_s == r_modelo_raw)

        r_anio_min = radar.get("anio_min")
        if r_anio_min is not None and str(r_anio_min).strip() != "":
            try:
                anio_min_int = int(r_anio_min)
                if anio_min_int > 0:
                    mask &= anio_s >= anio_min_int
            except (TypeError, ValueError):
                pass

        r_precio_max = radar.get("precio_max")
        if r_precio_max is not None and str(r_precio_max).strip() != "":
            try:
                precio_max_val = float(r_precio_max)
                if precio_max_val > 0:
                    mask &= (precio_s > 0) & (precio_s <= precio_max_val)
            except (TypeError, ValueError):
                pass

        r_km_max = radar.get("km_max")
        if r_km_max is not None and str(r_km_max).strip() != "":
            try:
                km_max_int = int(r_km_max)
                if km_max_int > 0:
                    mask &= km_s <= km_max_int
            except (TypeError, ValueError):
                pass

        r_disc_min = radar.get("descuento_min_pct")
        if r_disc_min is not None and str(r_disc_min).strip() != "":
            try:
                disc_min_val = float(r_disc_min)
                mask &= disc_pct_100 >= disc_min_val
            except (TypeError, ValueError):
                pass

        r_region = _strip_accents(str(radar.get("region") or "")).strip().upper()
        if r_region not in wildcard_tokens:
            mask &= region_s == r_region

        matched = work.loc[mask].copy()
        if not matched.empty:
            for sort_col in ("margen_neto_clp", "margen_ganancia_clp", "descuento_pct"):
                if sort_col in matched.columns:
                    matched = matched.sort_values(by=sort_col, ascending=False)
                    break

        results[rid] = matched

    return results
