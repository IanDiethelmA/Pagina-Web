"""
pipeline.py - MÓDULO 2: Ingeniería de Características y Filtrado de Outliers (Data Pipelines).
Implementa:
1. Lectura de registros desde SQLite ('data/chileautos_arbitraje.db') en Pandas.
2. Filtro de Outliers en dos etapas:
   - Reglas duras de negocio: Exclusión de precio $0 ("Consultar"), < $500.000 CLP,
     o fuera de límites físicos de año/kilometraje.
   - Filtro estadístico IQR (Rango Intercuartílico) y Z-Score Robusto agrupado por
     [Marca / Modelo / Año] (con respaldo jerárquico a [Marca / Modelo] ajustado).
3. Enriquecimiento de Generaciones: Mapeo de cada vehículo a su generación estética/tecnológica
   según su Marca, Modelo y Año en el mercado chileno.
4. Procesamiento de Texto (NLP Básico): Extracción de variables binarias (1 o 0) desde
   título y descripción ("único dueño", "mantenciones al día", "cuero",
   "láminas de seguridad", "neumáticos nuevos", etc.) y codificación socioeconómica comunal.
"""

import json
import logging
import re
import sqlite3
import sys
import unicodedata
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config

logger = config.setup_logger("Modulo2.Pipeline")


class CarDataPipeline:
    """Pipeline de limpieza, filtrado de outliers e ingeniería de características en Pandas."""

    _COMPILED_NLP_REGEX: Optional[Dict[str, str]] = None

    def __init__(self, db_path: Path = config.DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.current_year = datetime.now().year

    @staticmethod
    def strip_accents(text: str) -> str:
        """Elimina tildes y normaliza texto para comparaciones robustas."""
        if not isinstance(text, str):
            return ""
        normalized = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")

    def load_from_db(self, only_active: bool = True) -> pd.DataFrame:
        """
        Lee los registros desde la tabla 'autos_historico' en SQLite usando Pandas.
        Por defecto carga los anuncios con estado = 'activo'.
        """
        query = "SELECT * FROM autos_historico"
        if only_active:
            query += " WHERE estado = 'activo'"

        use_pg = bool(config.DATABASE_URL) and (
            self.db_path.resolve() == Path(config.DB_PATH).resolve()
        )
        if use_pg:
            from database import DatabaseManager

            df = DatabaseManager(database_url=config.DATABASE_URL).read_dataframe(query)
        else:
            if not self.db_path.exists():
                raise FileNotFoundError(f"No se encontró la base de datos en {self.db_path}")
            with closing(sqlite3.connect(str(self.db_path))) as conn:
                df = pd.read_sql_query(query, conn)

        logger.info(
            "Cargados %d registros desde SQLite (%s).",
            len(df),
            "solo activos" if only_active else "histórico completo",
        )
        return df

    @classmethod
    def resolve_generation(cls, marca: str, modelo: str, anio: int) -> str:
        """
        Determina la Generación del vehículo según Marca, Modelo y Año de fabricación
        usando las reglas de config.MODEL_GENERATIONS. Si el modelo no tiene rango
        explícito para ese año, agrupa en cohortes tecnológicas de 4 años.
        """
        marca_clean = str(marca).strip().upper()
        modelo_clean = str(modelo).strip().upper()
        try:
            anio_int = int(anio)
        except Exception:
            anio_int = 2018

        ranges = config.MODEL_GENERATIONS.get((marca_clean, modelo_clean), [])
        for start_yr, end_yr, gen_code in ranges:
            if start_yr <= anio_int <= end_yr:
                return f"{marca_clean}_{modelo_clean}_{gen_code}"

        # Fallback estructurado por bloque generacional de 4 años
        cohort_start = (anio_int // 4) * 4
        return f"{marca_clean}_{modelo_clean}_GEN_{cohort_start}_{cohort_start + 3}"

    @classmethod
    def resolve_comuna_tier(cls, comuna: str) -> int:
        """
        Codifica la comuna en un índice numérico de poder adquisitivo (1 a 5).
        Normaliza mayúsculas y acentos para evitar fallos de calce.
        """
        if not comuna or not isinstance(comuna, str):
            return config.DEFAULT_COMUNA_TIER

        raw_upper = comuna.strip().upper()
        if raw_upper in config.COMUNA_WEALTH_TIER:
            return config.COMUNA_WEALTH_TIER[raw_upper]

        no_accents = cls.strip_accents(raw_upper)
        if no_accents in config.COMUNA_WEALTH_TIER:
            return config.COMUNA_WEALTH_TIER[no_accents]

        return config.DEFAULT_COMUNA_TIER

    @classmethod
    def _get_compiled_nlp_patterns(cls) -> Dict[str, str]:
        """Devuelve el diccionario cacheado de expresiones regulares NLP sin tildes."""
        if cls._COMPILED_NLP_REGEX is None:
            cls._COMPILED_NLP_REGEX = {
                feature_name: "|".join(f"(?:{cls.strip_accents(p.lower())})" for p in patterns)
                for feature_name, patterns in config.NLP_FEATURE_PATTERNS.items()
            }
        return cls._COMPILED_NLP_REGEX

    @classmethod
    def extract_nlp_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Escanea el título y la descripción buscando términos diferenciadores
        ("único dueño", "mantenciones al día", "cuero", "láminas de seguridad",
        "neumáticos nuevos", etc.) y los convierte en columnas booleanas numéricas (1 o 0).
        """
        df = df.copy()
        combined_text = (
            df["titulo"].fillna("").astype(str) + " " + df["descripcion"].fillna("").astype(str)
        )
        # Normalizamos a minúsculas y versión sin tildes para mayor cobertura regex
        combined_clean = pd.Series(
            [cls.strip_accents(s) for s in combined_text.str.lower()],
            index=df.index,
        )

        compiled_patterns = cls._get_compiled_nlp_patterns()
        nlp_cols: List[str] = []
        for feature_name, combined_regex in compiled_patterns.items():
            df[feature_name] = combined_clean.str.contains(
                combined_regex, regex=True, na=False
            ).astype(int)
            nlp_cols.append(feature_name)

        # Score compuesto de atributos positivos detectados en texto (excluyendo urgencia)
        positive_nlp_cols = [c for c in nlp_cols if c != "nlp_urgencia_conversable"]
        df["nlp_extras_count"] = df[positive_nlp_cols].sum(axis=1)

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica toda la ingeniería de características (normalización, generaciones,
        variables de uso y NLP) sobre cualquier DataFrame de entrada.
        """
        if df.empty:
            return df.copy()

        out = df.copy()

        # 1. Normalización de tipos y strings categóricos
        out["marca"] = out["marca"].fillna("DESCONOCIDA").astype(str).str.strip().str.upper()
        out["modelo"] = out["modelo"].fillna("DESCONOCIDO").astype(str).str.strip().str.upper()
        out["combustible"] = out["combustible"].fillna("Bencina").astype(str).str.strip().str.title()
        out["transmision"] = out["transmision"].fillna("Manual").astype(str).str.strip().str.title()
        out["comuna"] = out["comuna"].fillna("Santiago").astype(str).str.strip().str.title()

        out["anio"] = pd.to_numeric(out["anio"], errors="coerce").fillna(2018).astype(int)
        out["kilometraje"] = pd.to_numeric(out["kilometraje"], errors="coerce").fillna(60_000).astype(int)
        out["precio"] = pd.to_numeric(out["precio"], errors="coerce").fillna(0).astype(int)

        # 2. Enriquecimiento por Generaciones (Marca + Modelo + Año)
        out["generacion"] = [
            self.resolve_generation(m, mod, y)
            for m, mod, y in zip(out["marca"], out["modelo"], out["anio"])
        ]

        # 3. Codificación de Comuna por Nivel Socioeconómico / Poder Adquisitivo (1 a 5) via mapa único
        unique_comunas = out["comuna"].unique()
        comuna_map = {c: self.resolve_comuna_tier(c) for c in unique_comunas}
        out["comuna_tier"] = out["comuna"].map(comuna_map).fillna(config.DEFAULT_COMUNA_TIER).astype(int)

        # 4. Variables físicas de depreciación y uso intensivo
        out["antiguedad"] = np.maximum(0, self.current_year - out["anio"])
        out["km_por_anio"] = np.round(
            out["kilometraje"] / np.maximum(1, out["antiguedad"]), 1
        )

        # 5. Procesamiento de Texto (NLP Básico -> variables binarias 1/0)
        out = self.extract_nlp_features(out)

        return out

    @staticmethod
    def deduplicate_cross_platform_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Identifica vehículos físicos publicados simultáneamente en varios portales
        usando una huella ultra-estricta para NUNCA eliminar autos distintos que
        simplemente coincidan en año o kilometraje aproximado/redondo:
        1. Kilometraje NO redondo (km % 100 != 0) e idéntico al último dígito (km_1 == km_2).
        2. Misma Marca, Modelo y Año exacto.
        3. Misma Transmisión, Combustible y Comuna.
        4. Diferencia de precio entre avisos <= 12%.
        Cuando se cumplen las 4 condiciones, conserva el aviso más barato y suma sus descripciones.
        """
        if df.empty or len(df) <= 1:
            df_copy = df.copy()
            df_copy["multi_portal_info"] = "Publicación única"
            return df_copy, 0

        work = df.copy()
        work["precio"] = pd.to_numeric(work["precio"], errors="coerce").fillna(0).astype(int)
        work["kilometraje"] = pd.to_numeric(work["kilometraje"], errors="coerce").fillna(0).astype(int)
        work["anio"] = pd.to_numeric(work["anio"], errors="coerce").fillna(2018).astype(int)

        fingerprints = [
            f"UNIQUE_CAR|{uid}|{idx_row}" if (km <= 0 or (km % 100 == 0))
            else f"EXACT_ODOMETER|{m}|{mod}|{yr}|{km}|{tr}|{cb}|{com}"
            for idx_row, (uid, m, mod, yr, km, tr, cb, com) in enumerate(
                zip(
                    work["id_aviso"].astype(str),
                    work["marca"].astype(str).str.upper().str.strip(),
                    work["modelo"].astype(str).str.upper().str.strip(),
                    work["anio"],
                    work["kilometraje"],
                    work["transmision"].astype(str).str.title().str.strip(),
                    work["combustible"].astype(str).str.title().str.strip(),
                    work["comuna"].astype(str).str.upper().str.strip(),
                )
            )
        ]

        work["_fingerprint"] = fingerprints
        work["_valid_price_rank"] = np.where(work["precio"] > 0, work["precio"], 999_999_999)

        # Separar vectorialmente filas únicas de colisiones reales para evitar miles de iteraciones singleton
        dup_mask = work["_fingerprint"].duplicated(keep=False)
        unique_part = work.loc[~dup_mask].copy()
        unique_part["multi_portal_info"] = "Publicación única"

        duplicates_merged = 0
        kept_dup_rows: List[Dict[str, Any]] = []

        if dup_mask.any():
            dup_part = work.loc[dup_mask].sort_values(
                by=["_fingerprint", "_valid_price_rank"], ascending=[True, True]
            )
            for _, group in dup_part.groupby("_fingerprint", sort=False):
                min_p = int(group["precio"].min())
                max_p = int(group["precio"].max())
                # Si la diferencia de precio supera el 12%, son versiones o condiciones distintas -> conservar todos
                if min_p <= 0 or ((max_p - min_p) / max(1, min_p) > 0.12):
                    for _, r in group.iterrows():
                        r_dict = r.to_dict()
                        r_dict["multi_portal_info"] = "Publicación única"
                        kept_dup_rows.append(r_dict)
                    continue

                cheapest = group.iloc[0].to_dict()
                others = group.iloc[1:]
                duplicates_merged += len(others)

                comp_notes = []
                all_descs = [str(cheapest.get("descripcion") or "")]
                for _, other in others.iterrows():
                    diff = int(other["precio"]) - int(cheapest["precio"])
                    other_p_fmt = f"${int(other['precio']):,}".replace(",", ".")
                    diff_fmt = f"+${diff:,}".replace(",", ".") if diff > 0 else "mismo precio"
                    comp_notes.append(f"{other['id_aviso']} ({other_p_fmt}, {diff_fmt})")
                    o_desc = str(other.get("descripcion") or "")
                    if o_desc and o_desc not in all_descs:
                        all_descs.append(o_desc)

                cheapest["descripcion"] = " // ".join(all_descs)
                cheapest["multi_portal_info"] = f"Más barato aquí vs {', '.join(comp_notes[:3])}"
                kept_dup_rows.append(cheapest)

        if kept_dup_rows:
            dedup_df = pd.concat([unique_part, pd.DataFrame(kept_dup_rows)], ignore_index=True)
        else:
            dedup_df = unique_part.reset_index(drop=True)

        dedup_df = dedup_df.drop(columns=["_fingerprint", "_valid_price_rank"], errors="ignore")
        return dedup_df, duplicates_merged

    def filter_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Filtra registros inválidos y outliers estadísticos para el entrenamiento:
        1. Excluye precio == 0 ("Consultar precio") y precio < $500.000 CLP.
        2. Excluye kilometrajes o años fuera de rango físico.
        3. Aplica método de Rango Intercuartílico (IQR) y Z-Score agrupado por
           [Marca, Modelo, Año] (con respaldo jerárquico a [Marca, Modelo] ajustado por
           depreciación anual cuando el subgrupo exacto tiene pocas observaciones).
        """
        initial_count = len(df)
        work_df = df.copy()

        # Etapa 1: Reglas duras de negocio ($0 / Consultar y < $500.000 CLP)
        mask_zero = work_df["precio"] <= 0
        removed_zero = int(mask_zero.sum())
        work_df = work_df[~mask_zero].copy()

        mask_low_price = work_df["precio"] < config.MIN_VALID_PRICE_CLP
        removed_low_price = int(mask_low_price.sum())
        work_df = work_df[~mask_low_price].copy()

        mask_high_abs = work_df["precio"] > config.MAX_VALID_PRICE_CLP
        removed_high_abs = int(mask_high_abs.sum())
        work_df = work_df[~mask_high_abs].copy()

        mask_invalid_meta = (
            (work_df["anio"] < config.MIN_VALID_YEAR)
            | (work_df["anio"] > config.MAX_VALID_YEAR)
            | (work_df["kilometraje"] < config.MIN_VALID_KM)
            | (work_df["kilometraje"] > config.MAX_VALID_KM)
        )
        removed_invalid_meta = int(mask_invalid_meta.sum())
        work_df = work_df[~mask_invalid_meta].copy()

        if work_df.empty:
            stats = {
                "registros_iniciales": initial_count,
                "excluidos_precio_cero_consultar": removed_zero,
                "excluidos_menor_500k": removed_low_price,
                "excluidos_techo_absoluto": removed_high_abs,
                "excluidos_anio_km_invalido": removed_invalid_meta,
                "excluidos_iqr_zscore_grupo": 0,
                "registros_limpios_finales": 0,
            }
            return work_df, stats

        # Etapa 2: Filtro Estadístico IQR + Z-Score por grupo
        # Para que grupos pequeños de (Marca, Modelo, Año) también detecten valores
        # ridículamente altos (ej. un auto de 12M publicado en 48M), normalizamos el
        # precio equivalente a año base 2024 dentro de cada (Marca, Modelo) además de
        # evaluar el grupo exacto (Marca, Modelo, Año) cuando tiene >= MIN_GROUP_SIZE_FOR_STATS.
        age_years = np.maximum(0, 2025 - work_df["anio"])
        work_df["_precio_norm_edad"] = work_df["precio"] / (0.932 ** age_years)

        is_statistical_outlier = pd.Series(False, index=work_df.index)

        # 2A. Evaluación por subgrupo exacto [Marca, Modelo, Año] cuando hay tamaño suficiente
        for _, group in work_df.groupby(["marca", "modelo", "anio"]):
            if len(group) >= config.MIN_GROUP_SIZE_FOR_STATS:
                prices = group["precio"]
                q1 = prices.quantile(0.25)
                q3 = prices.quantile(0.75)
                iqr = max(q3 - q1, 500_000)
                upper_iqr = q3 + config.IQR_MULTIPLIER * iqr
                lower_iqr = max(config.MIN_VALID_PRICE_CLP, q1 - config.IQR_MULTIPLIER * iqr)

                mean_p = prices.mean()
                std_p = max(prices.std(ddof=0), 500_000)
                z_scores = np.abs((prices - mean_p) / std_p)

                group_outliers = (prices > upper_iqr) | (prices < lower_iqr) | (z_scores > config.ZSCORE_THRESHOLD)
                is_statistical_outlier.loc[group.index] |= group_outliers

        # 2B. Evaluación por familia [Marca, Modelo] sobre el precio normalizado por año
        # Esto captura con precisión quirúrgica outliers extremos en años con 1-3 autos publicados
        # sin eliminar oportunidades legítimas (límite inferior permisivo para oportunidades reales,
        # límite superior estricto por IQR y Z-Score robusto para precios inflados/errores de tipeo).
        for _, group in work_df.groupby(["marca", "modelo"]):
            if len(group) >= 3:
                norm_p = group["_precio_norm_edad"]
                q1 = norm_p.quantile(0.25)
                q3 = norm_p.quantile(0.75)
                iqr = max(q3 - q1, 1_000_000)

                median_p = norm_p.median()
                mad = np.median(np.abs(norm_p - median_p))
                robust_std = max(mad * 1.4826, 1_000_000)
                robust_z = (norm_p - median_p) / robust_std

                # Filtramos del entrenamiento tanto valores ridículamente altos (IQR superior / Z > 3.0)
                # como valores absurdos por debajo del 40% del valor normalizado de la familia (estafas/errores)
                upper_bound = q3 + (config.IQR_MULTIPLIER * iqr)
                lower_scam_bound = median_p * 0.42

                family_outliers = (
                    (norm_p > upper_bound)
                    | (robust_z > config.ZSCORE_THRESHOLD)
                    | (norm_p < lower_scam_bound)
                )
                is_statistical_outlier.loc[group.index] |= family_outliers

        removed_stat_outliers = int(is_statistical_outlier.sum())
        clean_df = work_df[~is_statistical_outlier].drop(columns=["_precio_norm_edad"]).copy()

        stats = {
            "registros_iniciales": initial_count,
            "excluidos_precio_cero_consultar": removed_zero,
            "excluidos_menor_500k": removed_low_price,
            "excluidos_techo_absoluto": removed_high_abs,
            "excluidos_anio_km_invalido": removed_invalid_meta,
            "excluidos_iqr_zscore_grupo": removed_stat_outliers,
            "registros_limpios_finales": len(clean_df),
        }

        return clean_df, stats

    def run_pipeline(self, only_active: bool = False, save_output: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Ejecuta el flujo completo del Módulo 2:
        1. Carga los registros de SQLite (puede incluir tanto activos como vendidos recientes
           para maximizar la riqueza estadística del entrenamiento, o solo activos).
        2. Aplica ingeniería de características (Generaciones, Comuna Tier, NLP Binario).
        3. Aplica el filtro de Outliers ($0, <$500k, IQR/Z-Score por grupo).
        4. Guarda el dataset procesado listo para el Módulo 3.
        """
        logger.info("=" * 72)
        logger.info("INICIANDO MÓDULO 2: INGENIERÍA DE CARACTERÍSTICAS Y FILTRADO DE OUTLIERS")
        logger.info("=" * 72)

        raw_df = self.load_from_db(only_active=only_active)
        dedup_raw_df, merged_count = self.deduplicate_cross_platform_df(raw_df)
        if merged_count > 0:
            logger.info(
                "Fusión Multi-Plataforma en Pipeline: %d publicaciones repetidas consolidadas (conservando siempre el portal más barato).",
                merged_count,
            )
        featured_df = self.engineer_features(dedup_raw_df)
        clean_df, filter_stats = self.filter_outliers(featured_df)
        filter_stats["duplicados_multiplataforma_fusionados"] = merged_count

        if save_output and not clean_df.empty:
            csv_path = config.DATA_DIR / "autos_clean.csv"
            clean_df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info("Dataset limpio guardado en: %s (%d filas)", csv_path, len(clean_df))

        nlp_cols = list(config.NLP_FEATURE_PATTERNS.keys())
        nlp_summary = {col: int(clean_df[col].sum()) for col in nlp_cols if col in clean_df.columns}

        summary = {
            "filtro_outliers": filter_stats,
            "generaciones_unicas_detectadas": int(clean_df["generacion"].nunique()) if not clean_df.empty else 0,
            "marcas_unicas": int(clean_df["marca"].nunique()) if not clean_df.empty else 0,
            "modelos_unicos": int(clean_df["modelo"].nunique()) if not clean_df.empty else 0,
            "frecuencia_variables_nlp": nlp_summary,
        }

        logger.info("Resumen del Pipeline Módulo 2: %s", json.dumps(summary, ensure_ascii=False))
        return clean_df, summary


def main() -> None:
    pipeline = CarDataPipeline()
    # Cargamos los datos activos de la base de datos según lo solicitado
    clean_df, summary = pipeline.run_pipeline(only_active=True, save_output=True)
    print("\n--- MUESTRA DE COLUMNAS TRANSFORMADAS (PRIMERAS 5 FILAS) ---")
    cols_to_show = [
        "id_aviso", "marca", "modelo", "anio", "generacion", "kilometraje",
        "comuna", "comuna_tier", "nlp_unico_dueno", "nlp_mantenciones_al_dia",
        "nlp_cuero", "nlp_laminas_seguridad", "nlp_neumaticos_nuevos", "precio"
    ]
    existing_cols = [c for c in cols_to_show if c in clean_df.columns]
    print(clean_df[existing_cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
