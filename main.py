"""
main.py - Orquestador Autónomo del Sistema Multi-Portal y Segmentado de Arbitraje Automotriz.
Soporta dos modos de operación:
1. BARRIDO MULTI-SEGMENTO AUTOMÁTICO (python main.py):
   Recorre las marcas y modelos más transados en Chile en MercadoLibre Autos,
   Autocosmos Chile y Yapo Autos, construyendo curvas específicas por modelo,
   re-entrenando el modelo semanalmente (o cuando crece la base real) y enviando
   las oportunidades reales a Telegram.
2. BÚSQUEDA FOCALIZADA BAJO DEMANDA (python main.py --marca MAZDA --modelo 3):
   Escanea en vivo los portales específicamente para esa Marca y Modelo sin saturar
   los servidores, actualiza su curva en la base de datos y envía a Telegram las
   mejores oportunidades de ese auto en particular.
"""

import argparse
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import config
from database import DatabaseManager
from monitor import ArbitrageMonitor
from pipeline import CarDataPipeline
from scraper import ChileautosScraper
from train import CarPricePredictor

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = config.setup_logger("Orquestador.Main")


def run_autonomous_cycle(
    force_train: bool = False,
    force_fallback_scraper: bool = False,
    recheck_all_alerts: bool = False,
    marca_filter: Optional[str] = None,
    modelo_filter: Optional[str] = None,
    pages_per_portal: int = 2,
    max_telegram_alerts: int = 8,
    anio_min: Optional[int] = None,
    anio_max: Optional[int] = None,
    precio_min: Optional[int] = None,
    precio_max: Optional[int] = None,
    km_max: Optional[int] = None,
    deep: bool = False,
) -> Dict[str, Any]:
    """Ejecuta el flujo autónomo de extremo a extremo (Módulos 1 -> 2 -> 3 -> 4) con filtros avanzados."""
    start_time = datetime.now()
    logger.info("#" * 76)
    if marca_filter:
        filter_str = f"MARCA='{marca_filter.upper()}' | MODELO='{(modelo_filter or 'TODOS').upper()}'"
        if anio_min or anio_max:
            filter_str += f" | Años={anio_min or 'min'}..{anio_max or 'max'}"
        if precio_min or precio_max:
            filter_str += f" | Precio=${precio_min or 0:,}..${precio_max or 'max':,}".replace(",", ".")
        if km_max:
            filter_str += f" | Km<={km_max:,}".replace(",", ".")
        if deep:
            filter_str += " | MODO DEEP CHILEAUTOS ACTIVO"
        logger.info(" INICIANDO BÚSQUEDA FOCALIZADA MULTI-PORTAL: %s", filter_str)
    else:
        logger.info(" INICIANDO BARRIDO MULTI-PORTAL Y MULTI-SEGMENTO AUTOMÁTICO (CHILE)")
    logger.info("#" * 76)

    db_manager = DatabaseManager()

    # 1. Paso 1: Adquisición Multi-Portal (MercadoLibre + Autocosmos + Yapo + Chileautos)
    scraper = ChileautosScraper(db_manager=db_manager)
    scraper_stats = scraper.run_daily_acquisition(
        max_pages=pages_per_portal,
        force_fallback=force_fallback_scraper,
        marca_filter=marca_filter,
        modelo_filter=modelo_filter,
        segmented_sweep=(marca_filter is None),
        anio_min=anio_min,
        anio_max=anio_max,
        precio_min=precio_min,
        precio_max=precio_max,
        km_max=km_max,
        deep=deep,
    )

    # 2. Paso 2: Limpieza, Filtro de Outliers e Ingeniería de Características (Módulo 2)
    pipeline = CarDataPipeline(db_path=config.DB_PATH)
    clean_df, _ = pipeline.run_pipeline(only_active=True, save_output=True)

    # 3. Paso 3: Verificación de vigencia de 'models/model.pkl' y Entrenamiento IA (Módulo 3)
    predictor = CarPricePredictor(model_path=config.MODEL_PATH)
    need_retrain = force_train or predictor.should_retrain(
        model_path=config.MODEL_PATH, max_age_days=config.MODEL_RETRAIN_DAYS
    )

    if not need_retrain:
        train_metadata = predictor.load_artifact()
        prev_count = int(train_metadata.get("registros_entrenamiento_total", 0))
        # Si la base limpia creció en más de un 25% con nuevos avisos reales, re-entrenamos
        # automáticamente para que las curvas por modelo incorporen la nueva información
        if len(clean_df) >= 30 and abs(len(clean_df) - prev_count) > max(30, int(prev_count * 0.25)):
            logger.info(
                "El volumen de datos reales cambió significativamente (%d -> %d autos limpios). "
                "Re-calibrando curvas del modelo LightGBM...",
                prev_count,
                len(clean_df),
            )
            need_retrain = True

    if need_retrain and len(clean_df) >= 20:
        train_metadata = predictor.train_and_evaluate()
    else:
        train_metadata = predictor.load_artifact()
        logger.info(
            "Modelo IA vigente (< %d días | Muestras=%s | R²=%s).",
            config.MODEL_RETRAIN_DAYS,
            train_metadata.get("registros_entrenamiento_total", "N/A"),
            train_metadata.get("metricas_test_80_20", {}).get("R2_Score", "N/A"),
        )

    # 4. Paso 4: Monitor Diario de Arbitraje y Envío de Alertas (Módulo 4)
    monitor = ArbitrageMonitor(
        db_manager=db_manager,
        pipeline=pipeline,
        predictor=predictor,
    )
    monitor_summary = monitor.run_daily_monitor(
        include_already_alerted=recheck_all_alerts,
        marca_filter=marca_filter,
        modelo_filter=modelo_filter,
        max_telegram_alerts=max_telegram_alerts,
        anio_min=anio_min,
        anio_max=anio_max,
        precio_min=precio_min,
        precio_max=precio_max,
        km_max=km_max,
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("#" * 76)
    logger.info(
        " CICLO COMPLETADO EN %.2f SEGUNDOS | Evaluados: %d | Oportunidades: %d (Enviadas Telegram: %d)",
        elapsed,
        monitor_summary.get("vehiculos_evaluados", 0),
        monitor_summary.get("oportunidades_detectadas_total", 0),
        monitor_summary.get("oportunidades_enviadas_telegram", 0),
    )
    logger.info("#" * 76)

    return {
        "elapsed_seconds": round(elapsed, 2),
        "desglose_portales": scraper_stats.get("desglose_portales", {}),
        "procesados": scraper_stats.get("procesados_en_corrida", 0),
        "nuevos_insertados": scraper_stats.get("nuevos_insertados", 0),
        "precios_actualizados": scraper_stats.get("precios_actualizados", 0),
        "vehiculos_evaluados": monitor_summary.get("vehiculos_evaluados", 0),
        "oportunidades_detectadas": monitor_summary.get("oportunidades_detectadas_total", 0),
        "oportunidades_enviadas_telegram": monitor_summary.get("oportunidades_enviadas_telegram", 0),
        "posibles_estafas": monitor_summary.get("posibles_estafas_o_errores", 0),
        "reentrenado": bool(need_retrain and len(clean_df) >= 20),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orquestador Multi-Portal y Buscador Focalizado por Marca/Modelo (Chile)"
    )
    parser.add_argument("--marca", type=str, default=None, help="Búsqueda focalizada por Marca (ej. MAZDA, TOYOTA, SUBARU, NISSAN)")
    parser.add_argument("--modelo", type=str, default=None, help="Búsqueda focalizada por Modelo (ej. 3, RAV4, XV, X-TRAIL)")
    parser.add_argument("--anio-min", type=int, default=None, help="Año mínimo (ej. 2018)")
    parser.add_argument("--anio-max", type=int, default=None, help="Año máximo (ej. 2022)")
    parser.add_argument("--precio-min", type=int, default=None, help="Precio mínimo en CLP (ej. 10000000)")
    parser.add_argument("--precio-max", type=int, default=None, help="Precio máximo en CLP (ej. 17000000)")
    parser.add_argument("--km-max", type=int, default=None, help="Kilometraje máximo (ej. 80000)")
    parser.add_argument("--deep", action="store_true", help="Búsqueda profunda año por año en Google / Chileautos")
    parser.add_argument("--pages", type=int, default=2, help="Páginas por portal en búsqueda focalizada")
    parser.add_argument("--max-alerts", type=int, default=6, help="Máximo de alertas a enviar a Telegram por corrida")
    parser.add_argument("--force-train", action="store_true", help="Forzar reentrenamiento del modelo IA")
    parser.add_argument("--fallback", action="store_true", help="Usar generador de respaldo offline")
    parser.add_argument("--recheck-all", action="store_true", help="Re-evaluar todos los avisos activos")
    args = parser.parse_args()

    run_autonomous_cycle(
        force_train=args.force_train,
        force_fallback_scraper=args.fallback,
        recheck_all_alerts=args.recheck_all,
        marca_filter=args.marca,
        modelo_filter=args.modelo,
        pages_per_portal=args.pages,
        max_telegram_alerts=args.max_alerts,
        anio_min=args.anio_min,
        anio_max=args.anio_max,
        precio_min=args.precio_min,
        precio_max=args.precio_max,
        km_max=args.km_max,
        deep=args.deep,
    )


if __name__ == "__main__":
    main()
