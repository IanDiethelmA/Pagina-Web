"""
ui/components.py - Componentes UI/UX SaaS responsivos y seguros contra CommonMark/XSS.
"""

from __future__ import annotations

from datetime import datetime
import html
import math
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
import streamlit as st

from ui.domain import (
    extract_nlp_tags,
    format_clp,
    resolve_portal_info,
    resolve_region,
)

_FALLBACK_DEAL_COSTS: Dict[str, Any] = {
    "transferencia_pct": 0.015,
    "arancel_registro_civil_clp": 36030,
    "autofact_clp": 9990,
    "reacondicionamiento_clp": 350000,
    "dias_stock_estimados": 30,
    "costo_capital_mensual_pct": 0.009,
    "comision_venta_pct": 0.0,
}

_VALID_KPI_TONES = {"emerald", "blue", "amber", "neutral"}


def clean_html(raw_html: str) -> str:
    """
    Elimina sangría inicial por línea y líneas vacías para evitar que
    el parser CommonMark de Streamlit interprete bloques HTML como código.
    """
    return "\n".join(line.strip() for line in str(raw_html).splitlines() if line.strip())


def _safe_float(val: Any, default: Optional[float] = 0.0) -> Optional[float]:
    """Convierte un valor escalar a float finito de forma segura."""
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
        num = float(val)
        return num if math.isfinite(num) else default
    except Exception:
        return default


def _safe_int(val: Any, default: Optional[int] = 0) -> Optional[int]:
    """Convierte un valor escalar a int de forma segura."""
    num = _safe_float(val, default=None)
    if num is None:
        return default
    return int(round(num))


def _safe_str(val: Any, default: str = "") -> str:
    """Convierte un valor escalar a str limpio evitando 'nan' o 'None'."""
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    text = str(val).strip()
    if text.lower() in ("none", "nan", "<na>"):
        return default
    return text or default


def _sanitize_url(raw_url: Any) -> str:
    """Permite únicamente URLs http/https válidas o fallback seguro '#'."""
    url = _safe_str(raw_url, "#")
    lower = url.lower()
    if lower.startswith("https://") or lower.startswith("http://"):
        return url
    return "#"


def _get_default_deal_costs() -> Dict[str, Any]:
    """Obtiene DEFAULT_DEAL_COSTS desde services.deal_analyzer con fallback seguro."""
    try:
        from services.deal_analyzer import DEFAULT_DEAL_COSTS

        merged = dict(_FALLBACK_DEAL_COSTS)
        merged.update(DEFAULT_DEAL_COSTS)
        return merged
    except Exception:
        return dict(_FALLBACK_DEAL_COSTS)


def build_hero_header_html(workspace_id: str, active_count: int, opp_count: int) -> str:
    """Construye el HTML limpio del encabezado ejecutivo SaaS."""
    ws_s = html.escape(_safe_str(workspace_id, "default"))
    act_str = f"{max(0, int(active_count)):,}".replace(",", ".")
    opp_str = f"{max(0, int(opp_count)):,}".replace(",", ".")

    raw_html = f"""
    <div class="hero-banner">
        <h1 class="hero-title">AutoPulse IA Chile · Inteligencia de Arbitraje Automotriz</h1>
        <p class="hero-subtitle">
            Radar multi-portal en tiempo real, valoración LightGBM con confianza estadística y análisis de margen neto flipping.
        </p>
        <div class="hero-pill-row">
            <span class="hero-pill">Espacio activo: {ws_s}</span>
            <span class="hero-pill">{act_str} vehículos analizados</span>
            <span class="hero-pill">{opp_str} oportunidades activas</span>
        </div>
    </div>
    """
    return clean_html(raw_html)


def render_hero_header(workspace_id: str, active_count: int, opp_count: int) -> None:
    """Renderiza el encabezado ejecutivo SaaS."""
    st.markdown(
        build_hero_header_html(workspace_id=workspace_id, active_count=active_count, opp_count=opp_count),
        unsafe_allow_html=True,
    )


def build_kpi_strip_html(items: List[Dict[str, str]]) -> str:
    """Construye el HTML de la franja KPI responsiva."""
    if not items:
        return ""
    cards_html: List[str] = []
    for item in items:
        label_s = html.escape(_safe_str(item.get("label"), ""))
        val_s = html.escape(_safe_str(item.get("value"), "—"))
        sub_raw = _safe_str(item.get("sub"), "")
        tone_raw = _safe_str(item.get("tone"), "neutral").lower()
        tone = tone_raw if tone_raw in _VALID_KPI_TONES else "neutral"
        sub_html = f'<div class="saas-kpi-sub">{html.escape(sub_raw)}</div>' if sub_raw else ""
        cards_html.append(
            f'<div class="saas-kpi-card tone-{tone}">'
            f'<div class="saas-kpi-label">{label_s}</div>'
            f'<div class="saas-kpi-value">{val_s}</div>'
            f"{sub_html}"
            f"</div>"
        )
    return clean_html(f'<div class="saas-kpi-grid">{"".join(cards_html)}</div>')


def render_kpi_strip(items: List[Dict[str, str]]) -> None:
    """Renderiza un strip KPI HTML 100% responsivo (2 cols en móvil, N en escritorio)."""
    html_str = build_kpi_strip_html(items)
    if html_str:
        st.markdown(html_str, unsafe_allow_html=True)


def build_empty_state_html(title: str, subtitle: str, icon: str = "🔎") -> str:
    """Construye el HTML limpio de tarjeta para estados vacíos."""
    icon_s = html.escape(_safe_str(icon, "🔎"))
    title_s = html.escape(_safe_str(title, "Sin resultados"))
    sub_s = html.escape(_safe_str(subtitle, ""))
    return clean_html(
        f"""
        <div class="empty-state-box">
            <div class="empty-state-icon">{icon_s}</div>
            <div class="empty-state-title">{title_s}</div>
            <div class="empty-state-subtitle">{sub_s}</div>
        </div>
        """
    )


def render_empty_state(title: str, subtitle: str, icon: str = "🔎") -> None:
    """Renderiza una tarjeta limpia y elegante de estado vacío."""
    st.markdown(
        build_empty_state_html(title=title, subtitle=subtitle, icon=icon),
        unsafe_allow_html=True,
    )


def _sync_cost_widget_keys(key_prefix: str, cfg: Dict[str, Any]) -> None:
    """Sincroniza claves de widgets existentes antes de instanciarlos (sin warnings de Session State)."""
    mappings = {
        f"{key_prefix}_cost_transfer_pct": round(float(cfg["transferencia_pct"]) * 100.0, 2),
        f"{key_prefix}_cost_autofact_clp": int(cfg["autofact_clp"]),
        f"{key_prefix}_cost_reacond_clp": int(cfg["reacondicionamiento_clp"]),
        f"{key_prefix}_cost_dias_stock": int(cfg["dias_stock_estimados"]),
        f"{key_prefix}_cost_comision_pct": round(float(cfg["comision_venta_pct"]) * 100.0, 2),
    }
    for wkey, wval in mappings.items():
        if wkey in st.session_state:
            st.session_state[wkey] = wval


def _reset_deal_costs_callback(key_prefix: str) -> None:
    """Restaura valores estándar de costos flipping de manera segura antes del rerun."""
    defaults = _get_default_deal_costs()
    st.session_state["deal_costs_cfg"] = dict(defaults)
    new_rev = int(st.session_state.get("_deal_costs_rev", 0)) + 1
    st.session_state["_deal_costs_rev"] = new_rev
    _sync_cost_widget_keys(key_prefix, defaults)
    st.session_state[f"_{key_prefix}_deal_costs_rev"] = new_rev


def render_deal_cost_controls(key_prefix: str = "global") -> Dict[str, Any]:
    """
    Popover compacto '⚙️ Costos Flipping' sincronizado con st.session_state['deal_costs_cfg'].
    Retorna un diccionario compatible con services.deal_analyzer.DEFAULT_DEAL_COSTS.
    """
    defaults = _get_default_deal_costs()
    if "deal_costs_cfg" not in st.session_state or not isinstance(
        st.session_state["deal_costs_cfg"], dict
    ):
        st.session_state["deal_costs_cfg"] = dict(defaults)
    else:
        for k, v in defaults.items():
            st.session_state["deal_costs_cfg"].setdefault(k, v)

    cfg = dict(st.session_state["deal_costs_cfg"])
    global_rev = int(st.session_state.get("_deal_costs_rev", 0))
    prefix_rev_key = f"_{key_prefix}_deal_costs_rev"

    k_trans = f"{key_prefix}_cost_transfer_pct"
    k_auto = f"{key_prefix}_cost_autofact_clp"
    k_reac = f"{key_prefix}_cost_reacond_clp"
    k_dias = f"{key_prefix}_cost_dias_stock"
    k_com = f"{key_prefix}_cost_comision_pct"

    if st.session_state.get(prefix_rev_key) != global_rev:
        _sync_cost_widget_keys(key_prefix, cfg)
        st.session_state[prefix_rev_key] = global_rev

    with st.popover("⚙️ Costos Flipping", width="stretch"):
        st.caption("Ajusta los parámetros financieros para calcular margen y ROI neto real.")
        transfer_pct_ui = st.slider(
            "Transferencia (%):",
            min_value=0.0,
            max_value=5.0,
            step=0.1,
            format="%.1f%%",
            key=k_trans,
            **({} if k_trans in st.session_state else {"value": round(float(cfg["transferencia_pct"]) * 100.0, 2)}),
        )
        c1, c2 = st.columns(2)
        with c1:
            autofact_clp = st.number_input(
                "Autofact (CLP):",
                min_value=0,
                max_value=150_000,
                step=1_000,
                key=k_auto,
                **({} if k_auto in st.session_state else {"value": int(cfg["autofact_clp"])}),
            )
        with c2:
            reacond_clp = st.number_input(
                "Reacond. (CLP):",
                min_value=0,
                max_value=5_000_000,
                step=50_000,
                key=k_reac,
                **({} if k_reac in st.session_state else {"value": int(cfg["reacondicionamiento_clp"])}),
            )
        dias_stock = st.slider(
            "Días en stock:",
            min_value=5,
            max_value=180,
            step=5,
            key=k_dias,
            **({} if k_dias in st.session_state else {"value": int(cfg["dias_stock_estimados"])}),
        )
        comision_pct_ui = st.slider(
            "Comisión venta (%):",
            min_value=0.0,
            max_value=10.0,
            step=0.25,
            format="%.2f%%",
            key=k_com,
            **({} if k_com in st.session_state else {"value": round(float(cfg["comision_venta_pct"]) * 100.0, 2)}),
        )
        arancel_str = format_clp(cfg.get("arancel_registro_civil_clp", defaults["arancel_registro_civil_clp"]))
        cap_pct = float(cfg.get("costo_capital_mensual_pct", defaults["costo_capital_mensual_pct"])) * 100.0
        st.caption(f"RVM fijo: {arancel_str} · Costo capital: {cap_pct:.1f}%/mes")
        st.button(
            "Restaurar estándar",
            key=f"{key_prefix}_cost_reset_btn",
            on_click=_reset_deal_costs_callback,
            args=(key_prefix,),
            width="stretch",
        )

    updated_cfg: Dict[str, Any] = {
        "transferencia_pct": round(float(transfer_pct_ui) / 100.0, 6),
        "arancel_registro_civil_clp": int(cfg.get("arancel_registro_civil_clp", defaults["arancel_registro_civil_clp"])),
        "autofact_clp": int(autofact_clp),
        "reacondicionamiento_clp": int(reacond_clp),
        "dias_stock_estimados": int(dias_stock),
        "costo_capital_mensual_pct": float(cfg.get("costo_capital_mensual_pct", defaults["costo_capital_mensual_pct"])),
        "comision_venta_pct": round(float(comision_pct_ui) / 100.0, 6),
    }

    if updated_cfg != st.session_state["deal_costs_cfg"]:
        st.session_state["deal_costs_cfg"] = dict(updated_cfg)
        new_rev = global_rev + 1
        st.session_state["_deal_costs_rev"] = new_rev
        st.session_state[prefix_rev_key] = new_rev

    return dict(updated_cfg)


def render_vehicle_card_html(
    row: Mapping[str, Any] | pd.Series,
    mode: str = "opportunity",
    is_saved: bool = False,
) -> str:
    """
    Construye una tarjeta HTML limpia, responsiva y escapada con html.escape()
    para oportunidades de arbitraje o seguimiento en Pipeline CRM.
    """
    marca_raw = _safe_str(row.get("marca"), "Vehículo")
    modelo_raw = _safe_str(row.get("modelo"), "")
    anio_i = _safe_int(row.get("anio"), 0) or 0
    id_raw = _safe_str(row.get("id_aviso"), "")

    marca_s = html.escape(marca_raw)
    modelo_s = html.escape(modelo_raw)
    id_s = html.escape(id_raw)
    _, portal_label, portal_css = resolve_portal_info(id_raw)
    portal_label_s = html.escape(portal_label)

    raw_title = _safe_str(row.get("titulo"), f"{marca_raw} {modelo_raw} {anio_i}".strip())
    short_title = (raw_title[:57] + "...") if len(raw_title) > 60 else raw_title
    title_attr_s = html.escape(raw_title, quote=True)
    title_sub_s = html.escape(short_title)

    comuna_raw = _safe_str(row.get("comuna"), "Santiago")
    region_raw = _safe_str(row.get("region"), "") or resolve_region(comuna_raw)
    comuna_s = html.escape(comuna_raw)
    region_s = html.escape(region_raw)
    tier_val = _safe_int(row.get("comuna_tier"), 2) or 2

    km_i = _safe_int(row.get("kilometraje"), 0) or 0
    km_str = f"{km_i:,} km".replace(",", ".")
    trans_s = html.escape(_safe_str(row.get("transmision"), "Manual"))
    comb_s = html.escape(_safe_str(row.get("combustible"), "Bencina"))
    gen_raw = _safe_str(row.get("generacion"), "")
    gen_html = f" · <code>{html.escape(gen_raw)}</code>" if gen_raw else ""

    dias_pub = max(0, _safe_int(row.get("dias_publicado"), 0) or 0)
    dias_label = "Hoy" if dias_pub == 0 else f"{dias_pub}d publicado"

    # Modificador visual de la tarjeta y pills de cabecera
    deal_tier_raw = _safe_str(row.get("deal_tier"), "")
    dcto_ratio = _safe_float(row.get("descuento_pct"), 0.0) or 0.0
    if mode == "crm":
        card_modifier = "tier-crm"
    elif "élite" in deal_tier_raw.lower() or "elite" in deal_tier_raw.lower() or (not deal_tier_raw and dcto_ratio >= 0.25):
        card_modifier = "tier-elite"
    elif "buena" in deal_tier_raw.lower() or (not deal_tier_raw and dcto_ratio >= 0.15):
        card_modifier = "tier-good"
    else:
        card_modifier = ""

    header_badges: List[str] = [f'<span class="{portal_css}">{portal_label_s}</span>']
    if mode == "crm":
        etapa_crm = _safe_str(row.get("etapa_crm"), "Interesado")
        header_badges.append(f'<span class="crm-pill">{html.escape(etapa_crm)}</span>')
        estado_val = _safe_str(row.get("estado"), "activo").lower()
        if estado_val == "vendido":
            header_badges.append('<span class="drop-badge">No disponible</span>')
        elif estado_val == "descartado":
            header_badges.append('<span class="drop-badge">Descartado</span>')
    else:
        if deal_tier_raw:
            header_badges.append(f'<span class="tier-pill">{html.escape(deal_tier_raw)}</span>')
        if is_saved:
            header_badges.append('<span class="crm-pill">En CRM</span>')

    # Confianza IA y Liquidez
    conf_label = _safe_str(row.get("confianza_ia_label"), "")
    conf_score = _safe_int(row.get("confianza_ia_score"), None)
    comp_count = _safe_int(row.get("comparables_count"), None)
    conf_html = ""
    if conf_label or conf_score is not None:
        lbl_lower = conf_label.lower()
        if "alta" in lbl_lower or (not conf_label and (conf_score or 0) >= 75):
            conf_cls = "conf-high"
            conf_lbl_show = conf_label or "Alta"
        elif "media" in lbl_lower or (not conf_label and (conf_score or 0) >= 45):
            conf_cls = "conf-med"
            conf_lbl_show = conf_label or "Media"
        else:
            conf_cls = "conf-low"
            conf_lbl_show = conf_label or "Baja"

        parts = [f"Confianza IA: {html.escape(conf_lbl_show)}"]
        if conf_score is not None:
            parts.append(f"{conf_score}/100")
        if comp_count is not None and comp_count >= 0:
            parts.append(f"({comp_count} comp.)")
        conf_html = f'<span class="conf-badge {conf_cls}">{" · ".join(parts)}</span>'

    liq_label = _safe_str(row.get("liquidez_label"), "")
    liq_html = f'<span class="liq-pill">{html.escape(liq_label)}</span>' if liq_label else ""

    # Bajadas de precio y multi-portal
    precio_pub = _safe_float(row.get("precio"), 0.0) or 0.0
    precio_ia = _safe_float(row.get("precio_justo_ia"), 0.0) or 0.0
    cambios_p = _safe_int(row.get("cambios_precio"), 0) or 0
    precio_ant = _safe_float(row.get("precio_anterior"), 0.0) or 0.0
    drop_html = ""
    if cambios_p > 0 and precio_ant > precio_pub and precio_pub > 0:
        drop_clp = precio_ant - precio_pub
        drop_html = f'<span class="drop-badge">Bajó {format_clp(drop_clp)}</span>'

    multi_raw = _safe_str(row.get("multi_portal_info"), "")
    multi_html = ""
    if multi_raw and multi_raw.lower() != "publicación única":
        if multi_raw.startswith("Más barato") or "portal" in multi_raw.lower() or "publica" in multi_raw.lower():
            multi_html = f'<span class="multi-badge">{html.escape(multi_raw)}</span>'

    # Fecha guardado / oferta CRM
    crm_meta_html = ""
    if mode == "crm":
        saved_raw = _safe_str(row.get("fecha_guardado"), "")
        if saved_raw:
            try:
                saved_fmt = datetime.fromisoformat(saved_raw).strftime("%d/%m %H:%M")
            except Exception:
                saved_fmt = saved_raw[:16]
            crm_meta_html += f'<span class="days-pill">Guardado: {html.escape(saved_fmt)}</span>'

    badges_row_items = [x for x in (conf_html, liq_html, drop_html, multi_html, crm_meta_html) if x]
    badges_row_html = (
        f'<div class="car-meta-row">{"".join(badges_row_items)}</div>' if badges_row_items else ""
    )

    nlp_s = html.escape(extract_nlp_tags(row))
    url_s = html.escape(_sanitize_url(row.get("url")), quote=True)

    # Precios y margen financiero (Neto Real o fallback Bruto)
    precio_pub_str = format_clp(precio_pub)
    precio_ia_str = format_clp(precio_ia)

    margen_neto = _safe_float(row.get("margen_neto_clp"), None)
    roi_neto = _safe_float(row.get("roi_neto_pct"), None)
    if margen_neto is not None:
        margen_val = margen_neto
        margen_label = "Margen Neto"
        roi_val = roi_neto if roi_neto is not None else (_safe_float(row.get("roi_pct"), 0.0) or 0.0)
        roi_text = f"{roi_val:+.1f}% ROI Neto"
    else:
        margen_val = _safe_float(row.get("margen_ganancia_clp"), 0.0) or 0.0
        margen_label = "Margen Bruto IA"
        roi_val = _safe_float(row.get("roi_pct"), 0.0) or 0.0
        roi_text = f"{roi_val:+.1f}% ROI"

    dcto_pct = dcto_ratio * 100.0
    dcto_str = f"-{dcto_pct:.1f}%" if dcto_pct >= 0 else f"+{abs(dcto_pct):.1f}%"
    margen_sign = "+" if margen_val >= 0 else ""
    profit_cls = "profit-banner is-negative" if margen_val < 0 else "profit-banner"

    costos_est = _safe_float(row.get("costos_estimados_clp"), None)
    oferta_crm = _safe_int(row.get("oferta_clp"), 0) or 0
    breakdown_parts: List[str] = []
    if costos_est is not None and costos_est > 0:
        breakdown_parts.append(f"Costos est.: {format_clp(costos_est)}")
    if mode == "crm" and oferta_crm > 0:
        breakdown_parts.append(f"Oferta CRM: {format_clp(oferta_crm)}")
    if id_s:
        breakdown_parts.append(f"ID: {id_s}")

    breakdown_html = ""
    if breakdown_parts:
        left_part = html.escape(breakdown_parts[0])
        right_part = html.escape(" · ".join(breakdown_parts[1:])) if len(breakdown_parts) > 1 else ""
        breakdown_html = (
            f'<div class="net-breakdown-row"><span>{left_part}</span><span>{right_part}</span></div>'
        )

    card_cls = f"opp-card {card_modifier}".strip()
    anio_label = f" ({anio_i})" if anio_i > 0 else ""
    raw_card = f"""
    <div class="{card_cls}">
        <div class="car-header">
            <span class="car-title">{marca_s} {modelo_s}{anio_label}</span>
            <div class="car-badges">{"".join(header_badges)}</div>
        </div>
        <div class="car-subtitle-trim" title="{title_attr_s}">{title_sub_s}</div>
        <div class="car-meta">
            <div><span class="comuna-pill">{region_s} · {comuna_s} (Tier {tier_val}/5)</span> <span class="days-pill">{dias_label}</span></div>
            <div class="car-meta-row">{km_str} · {trans_s} · {comb_s}{gen_html}</div>
            <div class="car-meta-row">{nlp_s}</div>
            {badges_row_html}
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
        <div class="{profit_cls}">
            <span>{margen_label}: {margen_sign}{format_clp(margen_val)} ({dcto_str})</span>
            <span>{roi_text}</span>
        </div>
        {breakdown_html}
        <a href="{url_s}" target="_blank" rel="noopener noreferrer" class="btn-portal">
            Abrir Publicación ↗
        </a>
    </div>
    """
    return clean_html(raw_card)
