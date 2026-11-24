"""
services/pdf_report.py - Generador de Dossier Ejecutivo PDF para tasaciones y
oportunidades de arbitraje automotriz usando fpdf2 con normalización segura Latin-1.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import unicodedata

from fpdf import FPDF
from fpdf.enums import XPos, YPos
import numpy as np
import pandas as pd

from services.deal_analyzer import calculate_net_deal_metrics

_CHAR_REPLACEMENTS = {
    "—": "-",
    "–": "-",
    "―": "-",
    "•": "*",
    "·": "*",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "…": "...",
    "≥": ">=",
    "≤": "<=",
    "±": "+/-",
    "↗": "->",
    "→": "->",
    "←": "<-",
    "×": "x",
    "€": "EUR",
    "\u00a0": " ",
}


def _pdf_safe_text(text: Any, max_len: Optional[int] = None) -> str:
    """
    Normaliza texto Unicode para fuentes core Latin-1 de fpdf2:
    conserva tildes, diéresis y ñ en español, reemplaza puntuación tipográfica
    y descarta emojis/símbolos incompatibles sin lanzar excepciones.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    for src, dst in _CHAR_REPLACEMENTS.items():
        s = s.replace(src, dst)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = s.encode("latin-1", errors="ignore").decode("latin-1")
    s = " ".join(s.split())
    if max_len is not None and max_len > 3 and len(s) > max_len:
        s = s[: max_len - 3].rstrip() + "..."
    return s


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        num = float(val)
    except (TypeError, ValueError):
        return default
    return num if np.isfinite(num) else default


def _safe_int(val: Any, default: int = 0) -> int:
    return int(round(_safe_float(val, float(default))))


def _fmt_clp(val: Any) -> str:
    num = _safe_int(val, 0)
    if num < 0:
        return f"-$ {abs(num):,}".replace(",", ".")
    return f"$ {num:,}".replace(",", ".")


def _fmt_km(val: Any) -> str:
    num = max(0, _safe_int(val, 0))
    return f"{num:,} km".replace(",", ".")


def _resolve_portal_label(id_aviso: Any) -> str:
    uid = str(id_aviso or "").strip().upper()
    if uid.startswith("MLC"):
        return "MercadoLibre"
    if uid.startswith("AC-"):
        return "Autocosmos"
    if uid.startswith("YAPO-"):
        return "Yapo.cl"
    if uid:
        return "Chileautos"
    return "Evaluación Directa"


class _ValuationDossierPDF(FPDF):
    """Documento PDF corporativo con pie de página paginado."""

    def __init__(self, brand_name: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self._brand_name = _pdf_safe_text(brand_name or "AutoPulse AI Chile", max_len=60)
        self.alias_nb_pages()
        self.set_auto_page_break(auto=True, margin=16)
        self.set_margins(left=12, top=12, right=12)

    def header(self) -> None:
        if self.page_no() > 1:
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(100, 116, 139)
            self.cell(
                0,
                5,
                _pdf_safe_text(f"{self._brand_name} | Dossier Ejecutivo de Valoración"),
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            self.set_draw_color(226, 232, 240)
            self.line(12, self.get_y(), 198, self.get_y())
            self.ln(3)

    def footer(self) -> None:
        self.set_y(-13)
        self.set_draw_color(226, 232, 240)
        self.line(12, self.get_y(), 198, self.get_y())
        self.ln(1.5)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(100, 116, 139)
        self.cell(
            152,
            4,
            _pdf_safe_text(
                "Documento referencial generado por motor estadístico IA. No reemplaza inspección mecánica ni Autofact."
            ),
        )
        self.cell(
            0,
            4,
            _pdf_safe_text(f"Página {self.page_no()} de {{nb}}"),
            align="R",
        )


def _draw_section_title(pdf: FPDF, title: str) -> None:
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 6, _pdf_safe_text(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    y = pdf.get_y()
    pdf.set_draw_color(16, 185, 129)
    pdf.set_line_width(0.5)
    pdf.line(12, y, 42, y)
    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.2)
    pdf.line(42, y, 198, y)
    pdf.ln(2.5)


def _build_negotiation_bullets(
    vehicle: Dict[str, Any],
    metrics: Dict[str, Any],
    comparables_df: Optional[pd.DataFrame],
    descuento_pct: float,
) -> List[str]:
    bullets: List[str] = []

    p_compra = _safe_int(metrics.get("precio_compra"), 0)
    p_ia = _safe_int(metrics.get("precio_venta_estimado"), 0)
    p_max = _safe_int(metrics.get("precio_max_compra_sugerido_clp"), 0)
    costos_tot = _safe_int(metrics.get("costos_totales_clp"), 0)
    margen_neto = _safe_int(metrics.get("margen_neto_clp"), 0)

    if p_ia > 0 and p_compra > 0:
        if p_compra <= p_max and p_max > 0:
            bullets.append(
                f"Precio competitivo: la oferta ({_fmt_clp(p_compra)}) se ubica bajo el techo de compra sugerido ({_fmt_clp(p_max)}), dejando un margen neto proyectado de {_fmt_clp(margen_neto)} tras costos operativos."
            )
        elif p_compra <= p_ia:
            brecha = p_compra - p_max if p_max > 0 else 0
            bullets.append(
                f"Margen de negociación objetivo: aunque el aviso está {descuento_pct * 100:.1f}% bajo el valor IA ({_fmt_clp(p_ia)}), se sugiere contra-ofertar cerca de {_fmt_clp(p_max)} (ajuste de {_fmt_clp(brecha)}) para asegurar rentabilidad neta >= 15%."
            )
        else:
            sobreprecio = p_compra - p_ia
            bullets.append(
                f"Sobreprecio frente a mercado: el precio pedido excede en {_fmt_clp(sobreprecio)} la valoración justa IA ({_fmt_clp(p_ia)}). El precio máximo recomendado de cierre es {_fmt_clp(p_max)}."
            )

    bullets.append(
        f"Costos inmediatos de traspaso y preparación: el comprador asume {_fmt_clp(costos_tot)} en transferencia (1,5% + arancel RVM), Autofact, reacondicionamiento y capital en stock."
    )

    if isinstance(comparables_df, pd.DataFrame) and not comparables_df.empty and "precio" in comparables_df.columns:
        valid_comp_prices = pd.to_numeric(comparables_df["precio"], errors="coerce").dropna()
        valid_comp_prices = valid_comp_prices[valid_comp_prices > 0]
        if not valid_comp_prices.empty:
            med_p = float(valid_comp_prices.median())
            min_p = float(valid_comp_prices.min())
            max_p = float(valid_comp_prices.max())
            bullets.append(
                f"Referencia directa de mercado ({len(valid_comp_prices)} comparables): la mediana de unidades equivalentes es {_fmt_clp(med_p)} (rango {_fmt_clp(min_p)} a {_fmt_clp(max_p)})."
            )

    dias_pub = _safe_int(vehicle.get("dias_publicado"), 0)
    cambios_p = _safe_int(vehicle.get("cambios_precio"), 0)
    if dias_pub >= 15 or cambios_p > 0:
        detalles = []
        if dias_pub >= 15:
            detalles.append(f"{dias_pub} días publicado")
        if cambios_p > 0:
            detalles.append(f"{cambios_p} rebaja(s) de precio registradas")
        bullets.append(
            f"Palanca temporal de negociación: la unidad registra {' y '.join(detalles)}, lo que respalda solicitar un descuento adicional por cierre rápido al contado."
        )

    bullets.append(
        "Debida diligencia obligatoria antes del cierre: validar certificado Autofact (multas TAG/JPL, prendas, encargo por robo, remates y consistencia de kilometraje) junto a inspección mecánica."
    )
    return bullets


def _draw_comparables_table_header(pdf: FPDF, col_widths: List[float], headers: List[str]) -> None:
    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 7.5)
    for w, hdr in zip(col_widths, headers):
        pdf.cell(w, 6.5, _pdf_safe_text(hdr), border=0, align="L", fill=True)
    pdf.ln(6.5)


def generate_valuation_pdf_bytes(
    vehicle: Dict[str, Any],
    deal_metrics: Dict[str, Any],
    comparables_df: Optional[pd.DataFrame] = None,
    brand_name: str = "AutoPulse AI Chile",
) -> bytes:
    """
    Genera un dossier ejecutivo en formato PDF (bytes válidos %PDF) con ficha técnica,
    valoración IA vs oferta, desglose financiero neto, argumentos de negociación y
    hasta 10 comparables reales de mercado.
    """
    veh = dict(vehicle or {})
    raw_metrics = dict(deal_metrics or {})

    precio_pub = _safe_float(
        raw_metrics.get("precio_compra", veh.get("precio", veh.get("precio_publicado", 0)))
    )
    precio_ia = _safe_float(
        raw_metrics.get("precio_venta_estimado", veh.get("precio_justo_ia", 0))
    )

    computed_metrics = calculate_net_deal_metrics(precio_pub, precio_ia)
    for key, val in computed_metrics.items():
        if key not in raw_metrics or raw_metrics[key] is None:
            raw_metrics[key] = val

    marca = _pdf_safe_text(veh.get("marca") or "N/D", max_len=28).upper()
    modelo = _pdf_safe_text(veh.get("modelo") or "N/D", max_len=32).upper()
    anio = _safe_int(veh.get("anio"), 0)
    anio_txt = str(anio) if anio > 0 else "N/D"
    km_txt = _fmt_km(veh.get("kilometraje", 0))
    trans_txt = _pdf_safe_text(veh.get("transmision") or "N/D", max_len=24)
    comb_txt = _pdf_safe_text(veh.get("combustible") or "N/D", max_len=24)
    comuna_txt = _pdf_safe_text(veh.get("comuna") or "Santiago", max_len=30)
    region_txt = _pdf_safe_text(veh.get("region") or "", max_len=30)
    ubicacion_txt = f"{comuna_txt}, {region_txt}" if region_txt else comuna_txt
    gen_txt = _pdf_safe_text(veh.get("generacion") or "Estándar", max_len=34)
    id_aviso = _pdf_safe_text(veh.get("id_aviso") or "", max_len=32)
    portal_txt = _pdf_safe_text(veh.get("portal") or _resolve_portal_label(id_aviso), max_len=24)
    titulo_txt = _pdf_safe_text(veh.get("titulo") or f"{marca} {modelo} {anio_txt}", max_len=110)

    dias_pub = _safe_int(veh.get("dias_publicado"), 0)
    conf_label = _pdf_safe_text(veh.get("confianza_ia_label") or "", max_len=16)
    conf_score = _safe_int(veh.get("confianza_ia_score"), 0)
    comp_cnt = _safe_int(
        veh.get("comparables_count", len(comparables_df) if isinstance(comparables_df, pd.DataFrame) else 0),
        0,
    )
    liquidez_txt = _pdf_safe_text(veh.get("liquidez_label") or "", max_len=24)
    deal_tier_txt = _pdf_safe_text(veh.get("deal_tier") or "", max_len=26)

    if "descuento_pct" in veh and veh["descuento_pct"] is not None:
        descuento_pct = _safe_float(veh["descuento_pct"], 0.0)
    elif precio_ia > 0 and precio_pub > 0:
        descuento_pct = (precio_ia - precio_pub) / precio_ia
    else:
        descuento_pct = 0.0

    pdf = _ValuationDossierPDF(brand_name=brand_name)
    pdf.add_page()

    # 1. Encabezado Corporativo
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(12, 12, 186, 22, style="F")
    pdf.set_fill_color(16, 185, 129)
    pdf.rect(12, 34, 186, 1.4, style="F")

    pdf.set_xy(16, 15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(120, 7, _pdf_safe_text(brand_name or "AutoPulse AI Chile", max_len=48))

    fecha_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(203, 213, 225)
    pdf.cell(58, 7, _pdf_safe_text(f"Emitido: {fecha_str}"), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_x(16)
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(167, 243, 208)
    pdf.cell(120, 6, _pdf_safe_text("Dossier Ejecutivo de Valoración y Arbitraje Automotriz"))
    ref_code = f"Ref: {id_aviso}" if id_aviso else "Tasación IA"
    pdf.set_text_color(226, 232, 240)
    pdf.cell(58, 6, _pdf_safe_text(ref_code), align="R", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(38)

    # 2. Ficha del Vehículo
    _draw_section_title(pdf, "1. Ficha Técnica del Vehículo Evaluado")
    box_y = pdf.get_y()
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(12, box_y, 186, 28, style="FD")

    rows_specs = [
        ("Vehículo:", f"{marca} {modelo} ({anio_txt})", "Generación:", gen_txt),
        ("Kilometraje:", km_txt, "Transmisión / Comb.:", f"{trans_txt} / {comb_txt}"),
        ("Ubicación:", ubicacion_txt, "Fuente / Antigüedad:", f"{portal_txt} ({dias_pub} días)"),
    ]
    pdf.set_y(box_y + 2)
    for lbl1, val1, lbl2, val2 in rows_specs:
        pdf.set_x(15)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(24, 5.5, _pdf_safe_text(lbl1))
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(67, 5.5, _pdf_safe_text(val1, max_len=36))

        pdf.set_font("Helvetica", "B", 8.5)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(33, 5.5, _pdf_safe_text(lbl2))
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(56, 5.5, _pdf_safe_text(val2, max_len=32), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(24, 5.5, _pdf_safe_text("Aviso / Detalle:"))
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(155, 5.5, titulo_txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_y(box_y + 30)

    # 3. Valoración IA vs Oferta
    _draw_section_title(pdf, "2. Resumen de Valoración IA vs Precio Oferta")
    cards_y = pdf.get_y()
    card_w = 60.0
    gap = 3.0

    conf_sub = (
        f"Confianza {conf_label} ({conf_score}/100)"
        if conf_label
        else f"Comparables evaluados: {comp_cnt}"
    )
    dcto_sign = f"-{descuento_pct * 100:.1f}%" if descuento_pct >= 0 else f"+{abs(descuento_pct) * 100:.1f}%"
    summary_cards = [
        ("PRECIO OFERTA / PUBLICADO", _fmt_clp(raw_metrics["precio_compra"]), f"Diferencia IA: {dcto_sign}", (248, 250, 252), (15, 23, 42)),
        ("VALOR JUSTO MERCADO (IA)", _fmt_clp(raw_metrics["precio_venta_estimado"]), conf_sub, (240, 249, 255), (2, 132, 199)),
        ("COMPRA MÁX. SUGERIDA", _fmt_clp(raw_metrics["precio_max_compra_sugerido_clp"]), "Objetivo >= 15% margen neto", (236, 253, 245), (5, 150, 105)),
    ]

    for idx, (c_title, c_val, c_sub, bg_rgb, accent_rgb) in enumerate(summary_cards):
        cx = 12 + idx * (card_w + gap)
        pdf.set_fill_color(*bg_rgb)
        pdf.set_draw_color(203, 213, 225)
        pdf.rect(cx, cards_y, card_w, 21, style="FD")

        pdf.set_xy(cx + 3, cards_y + 2.5)
        pdf.set_font("Helvetica", "B", 7.2)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(card_w - 6, 4, _pdf_safe_text(c_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_x(cx + 3)
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*accent_rgb)
        pdf.cell(card_w - 6, 7.5, _pdf_safe_text(c_val), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.set_x(cx + 3)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(card_w - 6, 4.5, _pdf_safe_text(c_sub, max_len=36))

    pdf.set_y(cards_y + 23)
    badges_line = []
    if deal_tier_txt:
        badges_line.append(f"Clasificación: {deal_tier_txt}")
    if liquidez_txt:
        badges_line.append(f"Liquidez: {liquidez_txt}")
    if comp_cnt > 0:
        badges_line.append(f"Muestra comparable: {comp_cnt} uds.")
    if badges_line:
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(51, 65, 85)
        pdf.cell(0, 5, _pdf_safe_text("  |  ".join(badges_line)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # 4. Tabla de Desglose Financiero Neto
    _draw_section_title(pdf, "3. Desglose Financiero Neto de Operación")
    fin_rows = [
        ("(+) Valor Estimado de Venta en Mercado (IA)", "Referencia LightGBM", _fmt_clp(raw_metrics["precio_venta_estimado"]), False),
        ("(-) Precio de Compra / Oferta Evaluada", "Capital inicial unidad", _fmt_clp(-_safe_int(raw_metrics["precio_compra"])), False),
        ("(=) Margen Bruto Estimado", f"ROI Bruto: {raw_metrics['roi_bruto_pct']:+.1f}%", _fmt_clp(raw_metrics["margen_bruto_clp"]), True),
        ("(-) Transferencia Vehicular (1,5% + Arancel RVM)", "Registro Civil e Impuesto", _fmt_clp(-_safe_int(raw_metrics["costo_transferencia_clp"])), False),
        ("(-) Certificado Autofact / Historial", "Verificación documental", _fmt_clp(-_safe_int(raw_metrics["costo_autofact_clp"])), False),
        ("(-) Reacondicionamiento y Estética", "Mantención / Detailing", _fmt_clp(-_safe_int(raw_metrics["costo_reacondicionamiento_clp"])), False),
        ("(-) Costo Financiero / Capital en Stock", "Costo de oportunidad", _fmt_clp(-_safe_int(raw_metrics["costo_financiero_clp"])), False),
        ("(-) Comisión de Venta", "Corretaje / Consignación", _fmt_clp(-_safe_int(raw_metrics["comision_venta_clp"])), False),
        ("(=) Costos Totales de Operación", "Suma de gastos directos", _fmt_clp(raw_metrics["costos_totales_clp"]), True),
        ("(=) Inversión Total Requerida", "Compra + Costos totales", _fmt_clp(raw_metrics["inversion_total_clp"]), True),
    ]

    pdf.set_fill_color(15, 23, 42)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 8)
    pdf.cell(96, 6, _pdf_safe_text("Concepto Financiero"), fill=True)
    pdf.cell(50, 6, _pdf_safe_text("Detalle"), fill=True)
    pdf.cell(40, 6, _pdf_safe_text("Monto (CLP)"), align="R", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    for idx, (concepto, detalle, monto_str, is_sub) in enumerate(fin_rows):
        if is_sub:
            pdf.set_fill_color(241, 245, 249)
            pdf.set_font("Helvetica", "B", 8)
        elif idx % 2 == 1:
            pdf.set_fill_color(248, 250, 252)
            pdf.set_font("Helvetica", "", 8)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(96, 5.4, _pdf_safe_text(concepto, max_len=52), fill=True)
        pdf.set_text_color(71, 85, 105)
        pdf.cell(50, 5.4, _pdf_safe_text(detalle, max_len=30), fill=True)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(40, 5.4, _pdf_safe_text(monto_str), align="R", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    margen_neto_val = _safe_int(raw_metrics["margen_neto_clp"], 0)
    roi_neto_val = _safe_float(raw_metrics["roi_neto_pct"], 0.0)
    is_prof = margen_neto_val > 0
    if is_prof:
        pdf.set_fill_color(209, 250, 229)
        pdf.set_text_color(6, 95, 70)
    else:
        pdf.set_fill_color(254, 226, 226)
        pdf.set_text_color(153, 27, 27)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(96, 7, _pdf_safe_text("(=) MARGEN NETO PROYECTADO"), fill=True)
    pdf.cell(50, 7, _pdf_safe_text(f"ROI Neto: {roi_neto_val:+.1f}%"), fill=True)
    pdf.cell(40, 7, _pdf_safe_text(_fmt_clp(margen_neto_val)), align="R", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    # 5. Argumentos Objetivos de Negociación
    _draw_section_title(pdf, "4. Argumentos Objetivos de Negociación")
    bullets = _build_negotiation_bullets(veh, raw_metrics, comparables_df, descuento_pct)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_text_color(30, 41, 59)
    for bullet in bullets:
        pdf.set_x(13)
        pdf.multi_cell(
            184,
            4.5,
            _pdf_safe_text(f"* {bullet}"),
            align="L",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.ln(0.8)

    # 6. Comparables Reales de Mercado (hasta 10 unidades)
    if pdf.get_y() > 215:
        pdf.add_page()
    _draw_section_title(pdf, "5. Comparables Reales de Mercado (Top 10 Referencias)")

    if isinstance(comparables_df, pd.DataFrame) and not comparables_df.empty:
        comp_subset = comparables_df.head(10).copy()
        col_widths = [25.0, 12.0, 23.0, 23.0, 31.0, 26.0, 26.0, 20.0]
        headers = ["Portal", "Año", "Kilometraje", "Transmisión", "Comuna", "Oferta", "Valor IA", "Dcto %"]
        _draw_comparables_table_header(pdf, col_widths, headers)

        for idx, (_, crow) in enumerate(comp_subset.iterrows()):
            if pdf.get_y() > 268:
                pdf.add_page()
                _draw_comparables_table_header(pdf, col_widths, headers)

            c_portal = _pdf_safe_text(
                crow.get("portal") or _resolve_portal_label(crow.get("id_aviso")),
                max_len=13,
            )
            c_anio = str(_safe_int(crow.get("anio"), 0) or "N/D")
            c_km = _fmt_km(crow.get("kilometraje", 0))
            c_trans = _pdf_safe_text(crow.get("transmision") or "N/D", max_len=12)
            c_comuna = _pdf_safe_text(crow.get("comuna") or "Santiago", max_len=17)
            c_precio_num = _safe_float(crow.get("precio"), 0.0)
            c_ia_num = _safe_float(crow.get("precio_justo_ia"), 0.0)
            c_precio = _fmt_clp(c_precio_num)
            c_ia = _fmt_clp(c_ia_num) if c_ia_num > 0 else "N/D"

            if "descuento_pct" in crow and pd.notna(crow["descuento_pct"]):
                d_val = _safe_float(crow["descuento_pct"], 0.0) * 100.0
            elif c_ia_num > 0 and c_precio_num > 0:
                d_val = ((c_ia_num - c_precio_num) / c_ia_num) * 100.0
            else:
                d_val = 0.0
            c_dcto = f"{-d_val:+.1f}%" if c_ia_num > 0 else "N/D"

            if idx % 2 == 1:
                pdf.set_fill_color(248, 250, 252)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(15, 23, 42)
            pdf.set_font("Helvetica", "", 7.5)

            cells = [c_portal, c_anio, c_km, c_trans, c_comuna, c_precio, c_ia, c_dcto]
            aligns = ["L", "L", "R", "L", "L", "R", "R", "R"]
            for w, val, al in zip(col_widths, cells, aligns):
                pdf.cell(w, 5.2, _pdf_safe_text(val), align=al, fill=True)
            pdf.ln(5.2)
    else:
        pdf.set_font("Helvetica", "", 8.5)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(
            186,
            5,
            _pdf_safe_text(
                "No se adjuntaron comparables locales para este modelo/año en la muestra actual; la valoración utiliza la curva jerárquica entrenada del mercado chileno."
            ),
            align="L",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

    return bytes(pdf.output())
