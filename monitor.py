"""
monitor.py - MÓDULO 4: Monitor Diario de Arbitraje y Sistema de Alertas (Telegram Bot).
Implementa:
1. Carga instantánea del artefacto 'models/model.pkl' y enriquecimiento de avisos activos
   mediante 'CarDataPipeline' (con soporte para filtro focalizado --marca y --modelo).
2. Detección de Arbitraje Multi-Portal (MercadoLibre, Autocosmos, Yapo, Chileautos):
   Compara el precio publicado contra el 'Precio Justo' predicho por la IA.
   Si el descuento está entre 15% y <55%, lo clasifica como 'OPORTUNIDAD'.
3. Exclusión de Falsos Positivos / Filtro Anti-Estafas: Si el descuento es >= 55% respecto
   al Precio Justo estimado por la IA, lo clasifica como 'POSIBLE ESTAFA / ERROR DE TIPEO'.
4. Integración con Bot de Telegram vía 'requests' con sanitización Markdown y control de
   ráfaga (envía las mejores oportunidades ordenadas por margen y exporta el total a CSV).
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import requests

import config
from database import DatabaseManager
from pipeline import CarDataPipeline
from train import CarPricePredictor

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = config.setup_logger("Modulo4.Monitor")


class ArbitrageMonitor:
    """Motor detector de oportunidades de compra y enviador de alertas a Telegram."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        pipeline: Optional[CarDataPipeline] = None,
        predictor: Optional[CarPricePredictor] = None,
    ) -> None:
        self.db = db_manager or DatabaseManager()
        self.pipeline = pipeline or CarDataPipeline()
        self.predictor = predictor or CarPricePredictor()

    @staticmethod
    def format_clp(amount: float) -> str:
        """Formatea un monto numérico como pesos chilenos (ej. $ 12.450.000)."""
        return f"$ {int(round(amount)):,}".replace(",", ".")

    @staticmethod
    def _clean_md(text: Any) -> str:
        """Escapa caracteres que podrían romper el parseo Markdown clásico de Telegram."""
        if text is None:
            return ""
        s = str(text)
        for ch in ("*", "_", "`", "["):
            s = s.replace(ch, " ")
        return " ".join(s.split())

    @staticmethod
    def _resolve_portal_name(id_aviso: str) -> str:
        """Identifica el portal de origen a partir del prefijo del ID."""
        uid = str(id_aviso).upper()
        if uid.startswith("MLC"):
            return "MercadoLibre Autos Chile"
        if uid.startswith("AC-"):
            return "Autocosmos Chile"
        if uid.startswith("YAPO-"):
            return "Yapo.cl Autos"
        return "Chileautos"

    @staticmethod
    def _extract_active_badges(row: pd.Series) -> str:
        """Devuelve una lista legible de los atributos positivos detectados por NLP."""
        badges = []
        if row.get("nlp_unico_dueno", 0) == 1:
            badges.append("Único Dueño")
        if row.get("nlp_mantenciones_al_dia", 0) == 1:
            badges.append("Mantenciones al Día")
        if row.get("nlp_cuero", 0) == 1:
            badges.append("Tapiz Cuero")
        if row.get("nlp_sellado_inmaculado", 0) == 1:
            badges.append("Inmaculado")
        if row.get("nlp_full_equipo", 0) == 1:
            badges.append("Full Equipo")
        if row.get("nlp_laminas_seguridad", 0) == 1:
            badges.append("Láminas Seguridad")
        if row.get("nlp_neumaticos_nuevos", 0) == 1:
            badges.append("Neumáticos Nuevos")
        if row.get("nlp_urgencia_conversable", 0) == 1:
            badges.append("Vende por Apuro/Conversable")
        return ", ".join(badges) if badges else "Equipamiento Estándar"

    def build_alert_message(self, row: pd.Series) -> str:
        """Construye el mensaje Markdown enriquecido para Telegram y Consola."""
        categoria = row["categoria_arbitraje"]
        portal = self._resolve_portal_name(row["id_aviso"])
        marca_modelo = self._clean_md(f"{row['marca']} {row['modelo']} ({row['anio']})")
        titulo_clean = self._clean_md(row.get("titulo", marca_modelo))
        generacion = self._clean_md(row["generacion"])
        comuna = self._clean_md(row["comuna"])
        km_str = f"{int(row['kilometraje']):,} km".replace(",", ".")
        precio_pub = self.format_clp(row["precio"])
        precio_ia = self.format_clp(row["precio_justo_ia"])
        margen_clp = self.format_clp(row["margen_ganancia_clp"])
        descuento_pct = f"{row['descuento_pct'] * 100:.1f}%"
        badges = self._extract_active_badges(row)
        multi_info = self._clean_md(row.get("multi_portal_info", "Publicación única"))
        url = str(row["url"]).strip()

        if categoria == "OPORTUNIDAD":
            header = "🔥 *¡OPORTUNIDAD DE ARBITRAJE DETECTADA!* 🔥"
            footer_note = "✅ *Acción recomendada:* Revisar historial/documentación y contactar al vendedor."
        else:
            header = "⚠️ *[ALERTA MODERADA] POSIBLE ESTAFA / ERROR DE TIPEO* ⚠️"
            footer_note = (
                "🚨 *Precaución:* El precio está más de un 55% bajo el mercado. "
                "Verificar posible prenda, vehículo siniestrado, valor pie o error de digitación."
            )

        msg = (
            f"{header}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌐 *Portal Más Barato:* {portal} (`{row['id_aviso']}`)\n"
            f"🔄 *Multi-Plataforma:* {multi_info}\n"
            f"🚗 *Vehículo:* {marca_modelo}\n"
            f"📋 *Aviso:* {titulo_clean}\n"
            f"🧬 *Generación:* `{generacion}`\n"
            f"📍 *Ubicación:* {comuna} (Tier {row['comuna_tier']}/5)\n"
            f"🛣️ *Kilometraje:* {km_str} | ⚙️ {row['transmision']} ({row['combustible']})\n"
            f"✨ *Extras NLP:* {badges}\n\n"
            f"💰 *Precio Publicado:* *{precio_pub}*\n"
            f"🤖 *Precio Justo IA:* `{precio_ia}`\n"
            f"📈 *Margen Potencial:* *{margen_clp}* (*-{descuento_pct}* bajo mercado)\n\n"
            f"🔗 [Abrir Publicación Real en {portal}]({url})\n"
            f"{footer_note}\n"
            f"━━━━━━━━━━━━━━━━━━━━━━"
        )
        return msg

    def send_telegram_or_console_alert(self, message: str, row: pd.Series) -> bool:
        """Envía la alerta a Telegram usando requests (con respaldo sin formato si falla Markdown)."""
        token = (config.TELEGRAM_BOT_TOKEN or "").strip()
        chat_id = (config.TELEGRAM_CHAT_ID or "").strip()

        if not token or not chat_id or token.startswith("TU_"):
            print("\n" + message)
            return True

        api_url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": False,
        }

        try:
            resp = requests.post(api_url, json=payload, timeout=config.TELEGRAM_API_TIMEOUT)
            if resp.status_code == 200:
                logger.info("Alerta enviada exitosamente a Telegram para aviso %s", row["id_aviso"])
                time.sleep(0.35)
                return True

            # Si algún caracter especial de la URL molesta al parser Markdown de Telegram, reintentar en texto plano
            payload.pop("parse_mode", None)
            resp_fallback = requests.post(api_url, json=payload, timeout=config.TELEGRAM_API_TIMEOUT)
            if resp_fallback.status_code == 200:
                logger.info("Alerta enviada (texto plano) a Telegram para aviso %s", row["id_aviso"])
                return True

            logger.warning("Telegram API devolvió HTTP %s: %s", resp.status_code, resp.text[:200])
            print("\n" + message)
            return False
        except requests.RequestException as exc:
            logger.error("Error de red con Telegram API: %s. Mostrando en consola.", exc)
            print("\n" + message)
            return False

    def run_daily_monitor(
        self,
        include_already_alerted: bool = False,
        marca_filter: Optional[str] = None,
        modelo_filter: Optional[str] = None,
        max_telegram_alerts: int = 8,
        anio_min: Optional[int] = None,
        anio_max: Optional[int] = None,
        precio_min: Optional[int] = None,
        precio_max: Optional[int] = None,
        km_max: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta el escaneo de arbitraje sobre las publicaciones activas (general o filtrado por marca/modelo/años/precios/km).
        """
        logger.info("=" * 72)
        logger.info("INICIANDO MÓDULO 4: MONITOR DIARIO DE ARBITRAJE Y ALERTAS")
        logger.info("=" * 72)

        if not config.MODEL_PATH.exists():
            logger.info("No se encontró %s. Entrenando modelo automáticamente...", config.MODEL_PATH)
            self.predictor.train_and_evaluate()
        else:
            self.predictor.load_artifact()

        active_df = self.pipeline.load_from_db(only_active=True)
        if active_df.empty:
            logger.warning("No hay vehículos activos en la base de datos para evaluar.")
            return {"evaluados": 0, "oportunidades": 0, "sospechosos": 0}

        # Deduplicar y fusionar publicaciones multiplataforma conservando siempre la más barata
        active_df, merged_dups = self.pipeline.deduplicate_cross_platform_df(active_df)
        if merged_dups > 0:
            logger.info(
                "Deduplicación Multi-Plataforma en Monitor: %d avisos repetidos consolidados en su opción más barata.",
                merged_dups,
            )

        if marca_filter:
            active_df = active_df[active_df["marca"].astype(str).str.upper() == marca_filter.strip().upper()].copy()
        if modelo_filter:
            active_df = active_df[active_df["modelo"].astype(str).str.upper() == modelo_filter.strip().upper()].copy()

        if not include_already_alerted and "alerta_enviada" in active_df.columns:
            candidate_df = active_df[active_df["alerta_enviada"].fillna(0).astype(int) == 0].copy()
        else:
            candidate_df = active_df.copy()

        if anio_min is not None:
            candidate_df = candidate_df[candidate_df["anio"] >= anio_min].copy()
        if anio_max is not None:
            candidate_df = candidate_df[candidate_df["anio"] <= anio_max].copy()
        if precio_min is not None:
            candidate_df = candidate_df[candidate_df["precio"] >= precio_min].copy()
        if precio_max is not None:
            candidate_df = candidate_df[candidate_df["precio"] <= precio_max].copy()
        if km_max is not None:
            candidate_df = candidate_df[candidate_df["kilometraje"] <= km_max].copy()

        valid_candidates = candidate_df[
            (candidate_df["precio"] >= config.MIN_VALID_PRICE_CLP)
            & (candidate_df["anio"] >= config.MIN_VALID_YEAR)
            & (candidate_df["kilometraje"] <= config.MAX_VALID_KM)
        ].copy()

        if valid_candidates.empty:
            logger.info("No hay publicaciones nuevas válidas pendientes de evaluación.")
            return {"evaluados": 0, "oportunidades": 0, "sospechosos": 0}

        featured_df = self.pipeline.engineer_features(valid_candidates)
        featured_df["precio_justo_ia"] = self.predictor.predict_fair_price(featured_df)
        featured_df["margen_ganancia_clp"] = featured_df["precio_justo_ia"] - featured_df["precio"]
        featured_df["descuento_pct"] = (
            featured_df["margen_ganancia_clp"] / featured_df["precio_justo_ia"]
        ).round(4)

        def classify_listing(discount: float) -> str:
            if discount >= config.SUSPICIOUS_DISCOUNT_THRESHOLD:
                return "POSIBLE ESTAFA / ERROR DE TIPEO"
            if discount >= config.OPPORTUNITY_DISCOUNT_THRESHOLD:
                return "OPORTUNIDAD"
            return "PRECIO DE MERCADO"

        featured_df["categoria_arbitraje"] = featured_df["descuento_pct"].apply(classify_listing)

        # Ordenar oportunidades por un balance de % de descuento y margen en CLP
        opportunities_df = featured_df[featured_df["categoria_arbitraje"] == "OPORTUNIDAD"].sort_values(
            by=["descuento_pct", "margen_ganancia_clp"], ascending=[False, False]
        )
        suspicious_df = featured_df[
            featured_df["categoria_arbitraje"] == "POSIBLE ESTAFA / ERROR DE TIPEO"
        ].sort_values(by="descuento_pct", ascending=False)

        logger.info(
            "Evaluados %d vehículos activos válidos -> %d OPORTUNIDADES reales | %d POSIBLES ESTAFAS/ERRORES.",
            len(featured_df),
            len(opportunities_df),
            len(suspicious_df),
        )

        # Enviar a Telegram las mejores oportunidades (hasta max_telegram_alerts para evitar spam masivo)
        sent_opps = 0
        for _, row in opportunities_df.iterrows():
            if sent_opps < max_telegram_alerts:
                msg = self.build_alert_message(row)
                self.send_telegram_or_console_alert(msg, row)
                sent_opps += 1
            self.db.mark_listing_alerted(str(row["id_aviso"]), "OPORTUNIDAD")

        sent_susp = 0
        for _, row in suspicious_df.iterrows():
            if sent_susp < max(2, max_telegram_alerts // 3):
                msg = self.build_alert_message(row)
                self.send_telegram_or_console_alert(msg, row)
                sent_susp += 1
            self.db.mark_listing_alerted(str(row["id_aviso"]), "POSIBLE_ESTAFA_O_ERROR")

        alerts_combined = pd.concat([opportunities_df, suspicious_df], ignore_index=True)
        report_path = config.DATA_DIR / "oportunidades_detectadas.csv"
        if not alerts_combined.empty:
            cols_export = [
                "id_aviso", "categoria_arbitraje", "marca", "modelo", "anio",
                "generacion", "kilometraje", "transmision", "combustible", "comuna",
                "precio", "precio_justo_ia", "margen_ganancia_clp", "descuento_pct",
                "multi_portal_info", "url"
            ]
            cols_export = [c for c in cols_export if c in alerts_combined.columns]
            alerts_combined[cols_export].to_csv(report_path, index=False, encoding="utf-8")
            logger.info("Reporte completo (%d alertas) exportado en: %s", len(alerts_combined), report_path)

        summary = {
            "fecha_monitoreo": datetime.now().isoformat(timespec="seconds"),
            "filtro_focalizado": {"marca": marca_filter, "modelo": modelo_filter},
            "vehiculos_evaluados": len(featured_df),
            "oportunidades_detectadas_total": len(opportunities_df),
            "oportunidades_enviadas_telegram": sent_opps,
            "posibles_estafas_o_errores": len(suspicious_df),
            "archivo_reporte": str(report_path),
        }
        logger.info("Resumen del Monitor Módulo 4: %s", json.dumps(summary, ensure_ascii=False))
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Módulo 4: Monitor de Arbitraje y Alertas Telegram")
    parser.add_argument("--recheck-all", action="store_true", help="Re-evaluar todos los vehículos activos")
    parser.add_argument("--marca", type=str, default=None, help="Filtrar por Marca específica (ej. MAZDA)")
    parser.add_argument("--modelo", type=str, default=None, help="Filtrar por Modelo específico (ej. 3)")
    parser.add_argument("--anio-min", type=int, default=None, help="Año mínimo (ej. 2018)")
    parser.add_argument("--anio-max", type=int, default=None, help="Año máximo (ej. 2022)")
    parser.add_argument("--precio-min", type=int, default=None, help="Precio mínimo en CLP")
    parser.add_argument("--precio-max", type=int, default=None, help="Precio máximo en CLP")
    parser.add_argument("--km-max", type=int, default=None, help="Kilometraje máximo (ej. 80000)")
    parser.add_argument("--max-alerts", type=int, default=8, help="Máximo de alertas a enviar a Telegram por corrida")
    args = parser.parse_args()

    monitor = ArbitrageMonitor()
    monitor.run_daily_monitor(
        include_already_alerted=args.recheck_all,
        marca_filter=args.marca,
        modelo_filter=args.modelo,
        max_telegram_alerts=args.max_alerts,
        anio_min=args.anio_min,
        anio_max=args.anio_max,
        precio_min=args.precio_min,
        precio_max=args.precio_max,
        km_max=args.km_max,
    )


if __name__ == "__main__":
    main()
