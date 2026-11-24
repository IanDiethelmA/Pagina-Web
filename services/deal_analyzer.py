"""
services/deal_analyzer.py - Motor financiero neto, confianza IA y clasificación de deals.
Calcula costos reales de flipping automotriz en Chile (transferencia 1.5% + arancel
Registro Civil, Autofact, reacondicionamiento, costo de capital y comisión de venta),
estima la densidad de comparables reales y asigna niveles de confianza, liquidez y tier.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import config

DEFAULT_DEAL_COSTS: Dict[str, Any] = {
    "transferencia_pct": 0.015,
    "arancel_registro_civil_clp": 36030,
    "autofact_clp": 9990,
    "reacondicionamiento_clp": 350000,
    "dias_stock_estimados": 30,
    "costo_capital_mensual_pct": 0.009,
    "comision_venta_pct": 0.0,
}

_ENRICH_OUTPUT_COLUMNS = (
    "costos_estimados_clp",
    "margen_neto_clp",
    "roi_neto_pct",
    "comparables_count",
    "confianza_ia_score",
    "confianza_ia_label",
    "liquidez_label",
    "deal_tier",
)


def _safe_nonneg_float(value: Any, default: float = 0.0) -> float:
    """Convierte un valor a float no negativo finito de forma segura."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(num):
        return default
    return max(0.0, num)


def _merge_deal_costs(costs: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """Combina parámetros de costos provistos por el usuario sobre DEFAULT_DEAL_COSTS."""
    merged: Dict[str, float] = {
        k: _safe_nonneg_float(v) for k, v in DEFAULT_DEAL_COSTS.items()
    }
    if isinstance(costs, dict):
        for key in DEFAULT_DEAL_COSTS:
            if key in costs and costs[key] is not None:
                merged[key] = _safe_nonneg_float(costs[key], merged[key])
    return merged


def calculate_net_deal_metrics(
    precio_publicado: float,
    precio_justo_ia: float,
    costs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Calcula métricas financieras netas de una oportunidad de compra/reventa automotriz.
    Incluye transferencia vehicular (1.5% + arancel Registro Civil), Autofact,
    reacondicionamiento, costo financiero del capital en stock y comisión de venta.
    """
    c = _merge_deal_costs(costs)
    p_pub = _safe_nonneg_float(precio_publicado)
    p_ia = _safe_nonneg_float(precio_justo_ia)

    precio_compra = int(round(p_pub))
    precio_venta_estimado = int(round(p_ia))

    target_margin_pct = _safe_nonneg_float(
        getattr(config, "OPPORTUNITY_DISCOUNT_THRESHOLD", 0.15), 0.15
    )
    var_rate = c["transferencia_pct"] + c["costo_capital_mensual_pct"] * (
        c["dias_stock_estimados"] / 30.0
    )
    fixed_for_target = (
        c["arancel_registro_civil_clp"]
        + c["autofact_clp"]
        + c["reacondicionamiento_clp"]
        + p_ia * c["comision_venta_pct"]
    )
    if p_ia > 0.0:
        max_buy_raw = (p_ia * (1.0 - target_margin_pct) - fixed_for_target) / max(
            1e-9, 1.0 + var_rate
        )
        precio_max_compra_sugerido_clp = int(round(max(0.0, max_buy_raw)))
    else:
        precio_max_compra_sugerido_clp = 0

    if p_pub <= 0.0:
        return {
            "precio_compra": 0,
            "precio_venta_estimado": precio_venta_estimado,
            "costo_transferencia_clp": 0,
            "costo_autofact_clp": 0,
            "costo_reacondicionamiento_clp": 0,
            "costo_financiero_clp": 0,
            "comision_venta_clp": 0,
            "costos_totales_clp": 0,
            "inversion_total_clp": 0,
            "margen_bruto_clp": 0,
            "margen_neto_clp": 0,
            "roi_bruto_pct": 0.0,
            "roi_neto_pct": 0.0,
            "precio_max_compra_sugerido_clp": precio_max_compra_sugerido_clp,
            "es_rentable_neto": False,
        }

    costo_transferencia_clp = int(
        round(p_pub * c["transferencia_pct"] + c["arancel_registro_civil_clp"])
    )
    costo_autofact_clp = int(round(c["autofact_clp"]))
    costo_reacondicionamiento_clp = int(round(c["reacondicionamiento_clp"]))
    costo_financiero_clp = int(
        round(
            p_pub
            * c["costo_capital_mensual_pct"]
            * (c["dias_stock_estimados"] / 30.0)
        )
    )
    comision_venta_clp = int(round(p_ia * c["comision_venta_pct"]))

    costos_totales_clp = (
        costo_transferencia_clp
        + costo_autofact_clp
        + costo_reacondicionamiento_clp
        + costo_financiero_clp
        + comision_venta_clp
    )
    inversion_total_clp = precio_compra + costos_totales_clp

    if p_ia > 0.0:
        margen_bruto_clp = precio_venta_estimado - precio_compra
        margen_neto_clp = precio_venta_estimado - inversion_total_clp
        roi_bruto_pct = round((margen_bruto_clp / precio_compra) * 100.0, 1)
        roi_neto_pct = (
            round((margen_neto_clp / inversion_total_clp) * 100.0, 1)
            if inversion_total_clp > 0
            else 0.0
        )
    else:
        margen_bruto_clp = 0
        margen_neto_clp = -costos_totales_clp
        roi_bruto_pct = 0.0
        roi_neto_pct = 0.0

    es_rentable_neto = bool(
        precio_compra > 0 and precio_venta_estimado > 0 and margen_neto_clp > 0
    )

    return {
        "precio_compra": precio_compra,
        "precio_venta_estimado": precio_venta_estimado,
        "costo_transferencia_clp": costo_transferencia_clp,
        "costo_autofact_clp": costo_autofact_clp,
        "costo_reacondicionamiento_clp": costo_reacondicionamiento_clp,
        "costo_financiero_clp": costo_financiero_clp,
        "comision_venta_clp": comision_venta_clp,
        "costos_totales_clp": costos_totales_clp,
        "inversion_total_clp": inversion_total_clp,
        "margen_bruto_clp": margen_bruto_clp,
        "margen_neto_clp": margen_neto_clp,
        "roi_bruto_pct": roi_bruto_pct,
        "roi_neto_pct": roi_neto_pct,
        "precio_max_compra_sugerido_clp": precio_max_compra_sugerido_clp,
        "es_rentable_neto": es_rentable_neto,
    }


def _build_comparables_lookup(
    out: pd.DataFrame,
    reference_df: Optional[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula vectorialmente para cada fila de `out`:
    (comparables_count en ±2 años, model_count total del modelo, mediana precio cohorte).
    """
    n_rows = len(out)
    ref = reference_df if (reference_df is not None and not reference_df.empty) else out
    if ref.empty or "marca" not in ref.columns or "modelo" not in ref.columns:
        return (
            np.zeros(n_rows, dtype=int),
            np.zeros(n_rows, dtype=int),
            np.zeros(n_rows, dtype=float),
        )

    ref_m = ref["marca"].fillna("").astype(str).str.strip().str.upper()
    ref_mod = ref["modelo"].fillna("").astype(str).str.strip().str.upper()
    ref_y = (
        pd.to_numeric(ref["anio"], errors="coerce").fillna(0).astype(int)
        if "anio" in ref.columns
        else pd.Series(0, index=ref.index, dtype=int)
    )
    ref_p = (
        pd.to_numeric(ref["precio"], errors="coerce").fillna(0.0)
        if "precio" in ref.columns
        else pd.Series(0.0, index=ref.index, dtype=float)
    )

    ref_work = pd.DataFrame(
        {"_m": ref_m, "_mod": ref_mod, "_y": ref_y, "_p": ref_p}
    )
    valid_ref = ref_work[ref_work["_p"] > 0]
    if valid_ref.empty:
        valid_ref = ref_work

    model_counts_map = valid_ref.groupby(["_m", "_mod"]).size().to_dict()
    year_stats = (
        valid_ref.groupby(["_m", "_mod", "_y"])["_p"]
        .agg(["count", "sum"])
        .to_dict("index")
    )

    out_m = (
        out["marca"].fillna("").astype(str).str.strip().str.upper()
        if "marca" in out.columns
        else pd.Series("", index=out.index)
    )
    out_mod = (
        out["modelo"].fillna("").astype(str).str.strip().str.upper()
        if "modelo" in out.columns
        else pd.Series("", index=out.index)
    )
    out_y = (
        pd.to_numeric(out["anio"], errors="coerce").fillna(0).astype(int)
        if "anio" in out.columns
        else pd.Series(0, index=out.index, dtype=int)
    )

    unique_keys = set(zip(out_m, out_mod, out_y))
    cohort_count_map: Dict[tuple[str, str, int], int] = {}
    cohort_mean_map: Dict[tuple[str, str, int], float] = {}

    for m, mod, yr in unique_keys:
        c_sum = 0
        p_sum = 0.0
        if yr > 0:
            for delta in (-2, -1, 0, 1, 2):
                stat = year_stats.get((m, mod, yr + delta))
                if stat is not None:
                    c_sum += int(stat["count"])
                    p_sum += float(stat["sum"])
        else:
            c_sum = int(model_counts_map.get((m, mod), 0))
        cohort_count_map[(m, mod, yr)] = c_sum
        cohort_mean_map[(m, mod, yr)] = (p_sum / c_sum) if c_sum > 0 else 0.0

    key_tuples = list(zip(out_m, out_mod, out_y))
    model_tuples = list(zip(out_m, out_mod))

    raw_cohort_counts = np.array(
        [cohort_count_map[k] for k in key_tuples], dtype=int
    )
    model_counts = np.array(
        [int(model_counts_map.get(mk, 0)) for mk in model_tuples], dtype=int
    )
    cohort_means = np.array(
        [cohort_mean_map[k] for k in key_tuples], dtype=float
    )

    if reference_df is None or reference_df.empty:
        self_subtract = np.ones(n_rows, dtype=int)
    elif "id_aviso" in out.columns and "id_aviso" in ref.columns:
        ref_ids = set(ref["id_aviso"].fillna("").astype(str).str.strip())
        self_subtract = (
            out["id_aviso"].fillna("").astype(str).str.strip().isin(ref_ids).astype(int).to_numpy()
        )
    else:
        self_subtract = np.zeros(n_rows, dtype=int)

    comparables_count = np.maximum(0, raw_cohort_counts - self_subtract)
    return comparables_count, model_counts, cohort_means


def enrich_dataframe_with_net_deal_and_confidence(
    df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame] = None,
    costs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Enriquece vectorialmente una copia del DataFrame con métricas financieras netas,
    conteo de comparables, score/etiqueta de confianza IA, liquidez y deal_tier.
    No muta el DataFrame de entrada y soporta DataFrames vacíos.
    """
    out = df.copy()
    if out.empty:
        int_cols = {"costos_estimados_clp", "margen_neto_clp", "comparables_count", "confianza_ia_score"}
        for col in _ENRICH_OUTPUT_COLUMNS:
            if col not in out.columns:
                if col in int_cols:
                    out[col] = pd.Series(dtype="int64")
                elif col == "roi_neto_pct":
                    out[col] = pd.Series(dtype="float64")
                else:
                    out[col] = pd.Series(dtype="object")
        return out

    c = _merge_deal_costs(costs)

    precio_s = (
        pd.to_numeric(out["precio"], errors="coerce").fillna(0.0).clip(lower=0.0)
        if "precio" in out.columns
        else pd.Series(0.0, index=out.index)
    )
    ia_s = (
        pd.to_numeric(out["precio_justo_ia"], errors="coerce").fillna(0.0).clip(lower=0.0)
        if "precio_justo_ia" in out.columns
        else pd.Series(0.0, index=out.index)
    )

    precio_arr = precio_s.to_numpy(dtype=float)
    ia_arr = ia_s.to_numpy(dtype=float)
    valid_p = precio_arr > 0.0
    valid_ia = ia_arr > 0.0

    # Costos vectorizados (alineados con calculate_net_deal_metrics)
    c_trans = np.where(
        valid_p,
        np.round(precio_arr * c["transferencia_pct"] + c["arancel_registro_civil_clp"]),
        0.0,
    )
    c_auto = np.where(valid_p, round(c["autofact_clp"]), 0.0)
    c_reac = np.where(valid_p, round(c["reacondicionamiento_clp"]), 0.0)
    c_fin = np.where(
        valid_p,
        np.round(
            precio_arr
            * c["costo_capital_mensual_pct"]
            * (c["dias_stock_estimados"] / 30.0)
        ),
        0.0,
    )
    c_com = np.where(valid_p, np.round(ia_arr * c["comision_venta_pct"]), 0.0)

    costos_arr = c_trans + c_auto + c_reac + c_fin + c_com
    precio_compra_round = np.round(precio_arr)
    ia_round = np.round(ia_arr)
    inversion_arr = precio_compra_round + costos_arr

    margen_neto_arr = np.where(
        valid_p & valid_ia,
        ia_round - inversion_arr,
        np.where(valid_p, -costos_arr, 0.0),
    )
    roi_neto_arr = np.where(
        valid_p & valid_ia & (inversion_arr > 0.0),
        np.round((margen_neto_arr / np.maximum(1.0, inversion_arr)) * 100.0, 1),
        0.0,
    )

    computed_disc = np.where(valid_ia, (ia_arr - precio_arr) / np.maximum(1.0, ia_arr), 0.0)
    if "descuento_pct" in out.columns:
        raw_disc = pd.to_numeric(out["descuento_pct"], errors="coerce")
        descuento_arr = np.where(
            raw_disc.notna().to_numpy(),
            raw_disc.fillna(0.0).to_numpy(dtype=float),
            computed_disc,
        )
    else:
        descuento_arr = computed_disc

    # Comparables, densidad de mercado y coherencia
    comp_count, model_count, cohort_means = _build_comparables_lookup(out, reference_df)

    anio_arr = (
        pd.to_numeric(out["anio"], errors="coerce").fillna(0).to_numpy(dtype=int)
        if "anio" in out.columns
        else np.zeros(len(out), dtype=int)
    )
    km_arr = (
        pd.to_numeric(out["kilometraje"], errors="coerce").fillna(-1).to_numpy(dtype=float)
        if "kilometraje" in out.columns
        else np.full(len(out), 60_000.0)
    )

    min_valid_p = _safe_nonneg_float(getattr(config, "MIN_VALID_PRICE_CLP", 500_000), 500_000)
    min_valid_y = int(getattr(config, "MIN_VALID_YEAR", 1995))
    max_valid_y = int(getattr(config, "MAX_VALID_YEAR", 2027))
    max_valid_km = _safe_nonneg_float(getattr(config, "MAX_VALID_KM", 600_000), 600_000)
    opp_thresh = _safe_nonneg_float(getattr(config, "OPPORTUNITY_DISCOUNT_THRESHOLD", 0.15), 0.15)
    susp_thresh = _safe_nonneg_float(getattr(config, "SUSPICIOUS_DISCOUNT_THRESHOLD", 0.55), 0.55)

    comp_pts = np.minimum(55.0, 14.5 * np.log1p(comp_count))
    family_pts = np.minimum(15.0, 4.0 * np.log1p(model_count))

    valid_specs = (
        (precio_arr >= min_valid_p)
        & valid_ia
        & (anio_arr >= min_valid_y)
        & (anio_arr <= max_valid_y)
        & (km_arr >= 0.0)
        & (km_arr <= max_valid_km)
    )
    spec_pts = np.where(valid_specs, 18.0, 4.0)

    has_cohort_mean = (cohort_means > 0.0) & valid_ia
    rel_diff = np.where(
        has_cohort_mean,
        np.abs(ia_arr - cohort_means) / np.maximum(1.0, cohort_means),
        1.0,
    )
    coherence_pts = np.select(
        [~has_cohort_mean, rel_diff <= 0.20, rel_diff <= 0.40],
        [6.0, 12.0, 7.0],
        default=2.0,
    )

    anomaly_penalty = np.select(
        [descuento_arr >= susp_thresh, descuento_arr <= -0.60],
        [-30.0, -10.0],
        default=0.0,
    )

    raw_conf = comp_pts + family_pts + spec_pts + coherence_pts + anomaly_penalty
    conf_score = np.clip(np.round(raw_conf), 0, 100).astype(int)
    conf_label = np.select(
        [conf_score >= 70, conf_score >= 45],
        ["Alta", "Media"],
        default="Baja",
    )

    liquidez_label = np.select(
        [(model_count >= 15) | (comp_count >= 8), (model_count >= 5) | (comp_count >= 3)],
        ["Alta Rotación", "Rotación Media"],
        default="Nicho",
    )

    deal_tier = np.select(
        [
            (~valid_p) | (~valid_ia) | (descuento_arr >= susp_thresh) | ((descuento_arr >= opp_thresh) & (conf_score < 30)),
            (descuento_arr >= 0.22) & (margen_neto_arr > 0.0) & (roi_neto_arr >= 15.0) & (conf_score >= 60),
            (descuento_arr >= opp_thresh) & (margen_neto_arr > 0.0) & (roi_neto_arr >= 5.0),
            (descuento_arr >= 0.08) | (margen_neto_arr > 0.0),
        ],
        [
            "Revisar / Riesgo",
            "Oportunidad Élite",
            "Buena Compra",
            "Margen Ajustado",
        ],
        default="Precio Mercado",
    )

    out["costos_estimados_clp"] = costos_arr.astype(np.int64)
    out["margen_neto_clp"] = margen_neto_arr.astype(np.int64)
    out["roi_neto_pct"] = roi_neto_arr
    out["comparables_count"] = comp_count.astype(int)
    out["confianza_ia_score"] = conf_score
    out["confianza_ia_label"] = conf_label
    out["liquidez_label"] = liquidez_label
    out["deal_tier"] = deal_tier

    return out
