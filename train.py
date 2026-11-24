"""
train.py - MÓDULO 3: Inteligencia Artificial (Machine Learning Core).
Implementa:
1. Lectura del dataset limpio 'data/autos_clean.csv' generado por el Módulo 2.
2. Preprocesamiento híbrido de variables categóricas:
   - Target Encoding Suavizado (Empirical Bayes) y Anclaje de Depreciación por
     Marca, Modelo y Generación (calculado estrictamente en Train para evitar Data Leakage).
   - One-Hot Encoding para Transmisión, Combustible y Marca.
3. Entrenamiento de un regresor avanzado basado en LightGBM (LGBMRegressor) apoyado
   por calibración de curva base para máxima precisión en precios de vehículos usados.
4. Validación Train/Test Split (80/20) reportando MAE (CLP), RMSE (CLP), MAPE (%) y R².
5. Serialización del artefacto completo (modelo + encoders + metadatos + control de
   caducidad semanal de 7 días) en 'models/model.pkl'.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

import config
from pipeline import CarDataPipeline

logger = config.setup_logger("Modulo3.Train")


class HierarchicalCarEncoder:
    """
    Codificador especializado para el mercado automotriz que combina:
    1. Target Encoding Suavizado (M-estimate) para Marca, Modelo y Generación.
    2. Target Encoding Normalizado por Depreciación Anual (estima el valor base 0 km
       implícito de cada Modelo/Generación y lo proyecta según la antigüedad del auto).
    3. One-Hot Encoding alineado para Combustible, Transmisión y Marca.
    """

    def __init__(self, smoothing: float = 3.0, annual_depreciation_rate: float = 0.068) -> None:
        self.smoothing = smoothing
        self.retention_rate = 1.0 - annual_depreciation_rate
        self.global_mean_price: float = 12_000_000.0
        self.global_base_price: float = 16_000_000.0

        self.marca_raw_map: Dict[str, float] = {}
        self.modelo_raw_map: Dict[Tuple[str, str], float] = {}
        self.gen_raw_map: Dict[str, float] = {}

        self.marca_base_map: Dict[str, float] = {}
        self.modelo_base_map: Dict[Tuple[str, str], float] = {}
        self.gen_base_map: Dict[str, float] = {}

        self.onehot_columns: List[str] = []
        self.feature_columns: List[str] = []

    def _compute_smoothed_map(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        target_col: str,
        prior_map: Optional[Dict[Any, float]] = None,
        default_prior: float = 0.0,
    ) -> Dict[Any, float]:
        """Calcula media suavizada (M-estimate) por grupo para evitar sobreajuste."""
        stats = df.groupby(group_cols)[target_col].agg(["count", "mean"])
        result: Dict[Any, float] = {}
        for key, row in stats.iterrows():
            cnt = float(row["count"])
            mean_val = float(row["mean"])
            if prior_map is not None:
                # Si la clave es una tupla (marca, modelo), su prior es la marca
                parent_key = key[0] if isinstance(key, tuple) else key
                prior = prior_map.get(parent_key, default_prior)
            else:
                prior = default_prior
            smoothed = (cnt * mean_val + self.smoothing * prior) / (cnt + self.smoothing)
            result[key] = smoothed
        return result

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "HierarchicalCarEncoder":
        """Ajusta los mapas de Target Encoding y columnas One-Hot usando solo datos de Train."""
        work = df.copy()
        work["_target_precio"] = y.values
        antiguedad = np.maximum(0, work["antiguedad"].values)
        # Precio equivalente llevado a año base (edad = 0) según curva media del mercado
        work["_target_base_0km"] = work["_target_precio"] / (self.retention_rate ** antiguedad)

        self.global_mean_price = float(work["_target_precio"].mean())
        self.global_base_price = float(work["_target_base_0km"].mean())

        # 1. Target Encoding sobre Precio Directo (Marca -> Modelo -> Generación)
        self.marca_raw_map = self._compute_smoothed_map(
            work, ["marca"], "_target_precio", default_prior=self.global_mean_price
        )
        self.modelo_raw_map = self._compute_smoothed_map(
            work, ["marca", "modelo"], "_target_precio",
            prior_map=self.marca_raw_map, default_prior=self.global_mean_price
        )
        self.gen_raw_map = self._compute_smoothed_map(
            work, ["generacion"], "_target_precio", default_prior=self.global_mean_price
        )

        # 2. Target Encoding Jerárquico sobre Precio Base 0km (captura la gama exacta del modelo)
        self.marca_base_map = self._compute_smoothed_map(
            work, ["marca"], "_target_base_0km", default_prior=self.global_base_price
        )
        self.modelo_base_map = self._compute_smoothed_map(
            work, ["marca", "modelo"], "_target_base_0km",
            prior_map=self.marca_base_map, default_prior=self.global_base_price
        )
        # Para generación, usamos el modelo como prior
        gen_stats = work.groupby(["marca", "modelo", "generacion"])["_target_base_0km"].agg(["count", "mean"])
        self.gen_base_map = {}
        for (m, mod, gen), row in gen_stats.iterrows():
            cnt = float(row["count"])
            mean_val = float(row["mean"])
            prior = self.modelo_base_map.get((m, mod), self.marca_base_map.get(m, self.global_base_price))
            self.gen_base_map[gen] = (cnt * mean_val + 2.0 * prior) / (cnt + 2.0)

        # Construir representación transformada para fijar las columnas finales
        transformed = self._build_features(df, is_fitting=True)
        self.feature_columns = list(transformed.columns)
        return self

    def _build_features(self, df: pd.DataFrame, is_fitting: bool = False) -> pd.DataFrame:
        """Transforma un DataFrame enriquecido en la matriz numérica X para LightGBM."""
        work = df.copy()
        antiguedad = np.maximum(0, work["antiguedad"].astype(float))
        depreciation_factor = self.retention_rate ** antiguedad

        # Mapear codificaciones por Marca, Modelo y Generación
        te_marca_raw = work["marca"].map(self.marca_raw_map).fillna(self.global_mean_price)
        te_modelo_raw = [
            self.modelo_raw_map.get((m, mod), self.marca_raw_map.get(m, self.global_mean_price))
            for m, mod in zip(work["marca"], work["modelo"])
        ]
        te_gen_raw = [
            self.gen_raw_map.get(gen, mod_val)
            for gen, mod_val in zip(work["generacion"], te_modelo_raw)
        ]

        # Mapear codificaciones de gama base proyectadas a la antigüedad exacta del vehículo
        base_marca = work["marca"].map(self.marca_base_map).fillna(self.global_base_price)
        base_modelo = np.array([
            self.modelo_base_map.get((m, mod), self.marca_base_map.get(m, self.global_base_price))
            for m, mod in zip(work["marca"], work["modelo"])
        ], dtype=float)
        base_gen = np.array([
            self.gen_base_map.get(gen, bm)
            for gen, bm in zip(work["generacion"], base_modelo)
        ], dtype=float)

        # Curva ancla de depreciación física por modelo y kilometraje
        km_delta = work["kilometraje"].astype(float) - (antiguedad * 14_000.0)
        km_factor = np.clip(1.0 - (km_delta / 450_000.0), 0.70, 1.12)

        nlp_cols = [c for c in config.NLP_FEATURE_PATTERNS.keys() if c in work.columns]

        features = pd.DataFrame(index=work.index)
        features["anio"] = work["anio"].astype(float)
        features["antiguedad"] = antiguedad
        features["kilometraje"] = work["kilometraje"].astype(float)
        features["km_por_anio"] = work["km_por_anio"].astype(float)
        features["comuna_tier"] = work["comuna_tier"].astype(float)

        # Target Encodings
        features["te_marca_precio"] = te_marca_raw.values
        features["te_modelo_precio"] = te_modelo_raw
        features["te_generacion_precio"] = te_gen_raw
        features["te_modelo_base_0km"] = base_modelo
        features["te_generacion_base_0km"] = base_gen
        features["anchor_precio_depreciado"] = base_modelo * depreciation_factor
        features["anchor_precio_km_ajustado"] = base_modelo * depreciation_factor * km_factor
        features["anchor_gen_km_ajustado"] = base_gen * depreciation_factor * km_factor

        # Variables binarias de NLP y contador de extras
        for col in nlp_cols:
            features[col] = work[col].astype(float)
        if "nlp_extras_count" in work.columns:
            features["nlp_extras_count"] = work["nlp_extras_count"].astype(float)

        # One-Hot Encoding para Transmisión, Combustible y Marca
        dummies = pd.get_dummies(
            work[["transmision", "combustible", "marca"]],
            prefix=["trans", "comb", "marca"],
            dtype=float,
        )
        # Limpiar nombres de columnas para LightGBM (sin caracteres especiales JSON)
        dummies.columns = [
             CarDataPipeline.strip_accents(c).replace(" ", "_").upper()
            for c in dummies.columns
        ]

        if is_fitting:
            self.onehot_columns = list(dummies.columns)
        else:
            for col in self.onehot_columns:
                if col not in dummies.columns:
                    dummies[col] = 0.0
            dummies = dummies[self.onehot_columns]

        features = pd.concat([features, dummies], axis=1)

        if not is_fitting and self.feature_columns:
            for col in self.feature_columns:
                if col not in features.columns:
                    features[col] = 0.0
            features = features[self.feature_columns]

        return features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma datos nuevos o de validación usando los encoders ajustados."""
        return self._build_features(df, is_fitting=False)

    def fit_transform_oof(self, df: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> pd.DataFrame:
        """
        Genera matriz de entrenamiento usando Out-Of-Fold (OOF) Target Encoding para
        que LightGBM no sobreajuste a las medias de entrenamiento y aprenda el peso real
        de todas las variables (año, kilometraje, comuna_tier, transmisión y NLP).
        Finalmente ajusta el encoder completo sobre todo el dataset de entrenamiento.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
        # Primero ajustamos en todo el set para definir todas las columnas One-Hot posibles
        self.fit(df, y)
        oof_features = pd.DataFrame(0.0, index=df.index, columns=self.feature_columns)

        for train_idx, val_idx in kf.split(df):
            fold_encoder = HierarchicalCarEncoder(smoothing=self.smoothing)
            fold_encoder.fit(df.iloc[train_idx], y.iloc[train_idx])
            val_transformed = fold_encoder.transform(df.iloc[val_idx])
            # Alinear columnas exactas
            for col in self.feature_columns:
                if col not in val_transformed.columns:
                    val_transformed[col] = 0.0
            oof_features.iloc[val_idx] = val_transformed[self.feature_columns].values

        # Re-ajustar el encoder definitivo con el 100% del subconjunto entregado
        self.fit(df, y)
        return oof_features

    def to_state_dict(self) -> Dict[str, Any]:
        """Serializa los parámetros del encoder a tipos nativos (independiente de __main__)."""
        return {
            "smoothing": self.smoothing,
            "retention_rate": self.retention_rate,
            "global_mean_price": self.global_mean_price,
            "global_base_price": self.global_base_price,
            "marca_raw_map": self.marca_raw_map,
            "modelo_raw_map": self.modelo_raw_map,
            "gen_raw_map": self.gen_raw_map,
            "marca_base_map": self.marca_base_map,
            "modelo_base_map": self.modelo_base_map,
            "gen_base_map": self.gen_base_map,
            "onehot_columns": self.onehot_columns,
            "feature_columns": self.feature_columns,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "HierarchicalCarEncoder":
        """Reconstruye el encoder desde su diccionario de estado nativo."""
        enc = cls(smoothing=state.get("smoothing", 2.5))
        enc.retention_rate = state.get("retention_rate", 1.0 - 0.068)
        enc.global_mean_price = state.get("global_mean_price", 12_000_000.0)
        enc.global_base_price = state.get("global_base_price", 16_000_000.0)
        enc.marca_raw_map = state.get("marca_raw_map", {})
        enc.modelo_raw_map = state.get("modelo_raw_map", {})
        enc.gen_raw_map = state.get("gen_raw_map", {})
        enc.marca_base_map = state.get("marca_base_map", {})
        enc.modelo_base_map = state.get("modelo_base_map", {})
        enc.gen_base_map = state.get("gen_base_map", {})
        enc.onehot_columns = state.get("onehot_columns", [])
        enc.feature_columns = state.get("feature_columns", [])
        return enc


class CarPricePredictor:
    """
    Motor de Inteligencia Artificial (Módulo 3) para predecir el Precio Justo de mercado
    en Chile usando LightGBM + Codificación Jerárquica.
    """

    def __init__(self, model_path: Path = config.MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.encoder = HierarchicalCarEncoder(smoothing=2.5)
        self.model: Optional[lgb.LGBMRegressor] = None
        self.metadata: Dict[str, Any] = {}

    @staticmethod
    def should_retrain(model_path: Path = config.MODEL_PATH, max_age_days: int = config.MODEL_RETRAIN_DAYS) -> bool:
        """
        Verifica si es necesario entrenar o reentrenar el modelo semanalmente:
        - Retorna True si 'models/model.pkl' no existe o si tiene >= max_age_days días.
        """
        if not model_path.exists():
            return True
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        age_days = (datetime.now() - mtime).days
        return age_days >= max_age_days

    @staticmethod
    def _build_lgbm() -> lgb.LGBMRegressor:
        """Configura el regresor LightGBM optimizado para valoración automotriz."""
        params = dict(config.LGBM_PARAMS)
        params.update(
            {
                "objective": "regression",
                "metric": "rmse",
                "n_estimators": 350,
                "learning_rate": 0.035,
                "num_leaves": 15,
                "max_depth": 5,
                "min_child_samples": 4,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_alpha": 0.05,
                "reg_lambda": 0.5,
                "random_state": config.RANDOM_SEED,
                "verbose": -1,
            }
        )
        return lgb.LGBMRegressor(**params)

    def train_and_evaluate(self, clean_csv_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        1. Carga 'data/autos_clean.csv'.
        2. Realiza un Train/Test Split de 80/20 para medir MAE, RMSE, MAPE y R².
        3. Re-entrena el modelo final con el 100% de los datos limpios y serializa
           el artefacto completo en 'models/model.pkl'.
        """
        logger.info("=" * 72)
        logger.info("INICIANDO MÓDULO 3: ENTRENAMIENTO Y VALIDACIÓN DE IA (LightGBM)")
        logger.info("=" * 72)

        csv_path = clean_csv_path or (config.DATA_DIR / "autos_clean.csv")
        if not csv_path.exists():
            logger.info("No se encontró %s. Ejecutando Módulo 2 automáticamente...", csv_path)
            pipeline = CarDataPipeline()
            df, _ = pipeline.run_pipeline(only_active=True, save_output=True)
        else:
            df = pd.read_csv(csv_path, encoding="utf-8")
            logger.info("Dataset limpio cargado desde %s (%d registros).", csv_path, len(df))

        if len(df) < 20:
            raise ValueError(f"Se requieren al menos 20 registros limpios para entrenar (actual: {len(df)}).")

        # Excluir de la evaluación y entrenamiento las pocas oportunidades extremas ("vende por apuro")
        # para que el modelo aprenda el "Precio Justo de Mercado Normal" sin sesgarse hacia abajo
        if "nlp_urgencia_conversable" in df.columns:
            normal_market_df = df[df["nlp_urgencia_conversable"] == 0].copy()
            if len(normal_market_df) >= 50:
                logger.info(
                    "Entrenando sobre %d vehículos de mercado regular (excluyendo %d avisos de liquidación urgente para no sesgar el Precio Justo).",
                    len(normal_market_df),
                    len(df) - len(normal_market_df),
                )
                df = normal_market_df

        y = df["precio"].astype(float)

        # División Train/Test 80/20 estratificada por cuantil de precio
        price_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
        df_train, df_test, y_train, y_test = train_test_split(
            df,
            y,
            test_size=0.20,
            random_state=config.RANDOM_SEED,
            stratify=price_bins,
        )

        # 1. Codificación Out-Of-Fold en Train y transformación limpia en Test (0% Data Leakage)
        eval_encoder = HierarchicalCarEncoder(smoothing=2.5)
        X_train = eval_encoder.fit_transform_oof(df_train, y_train, n_splits=5)
        X_test = eval_encoder.transform(df_test)

        # Entrenamos LightGBM sobre los residuos respecto al ancla física para lograr
        # interpolación perfecta por año/km y corrección no lineal por extras NLP/comuna/transmisión
        anchor_train = X_train["anchor_precio_km_ajustado"].values
        anchor_test = X_test["anchor_precio_km_ajustado"].values

        residual_train = y_train.values - anchor_train

        eval_model = self._build_lgbm()
        eval_model.fit(X_train, residual_train)

        # Predicción en Test (80/20)
        pred_residual_test = eval_model.predict(X_test)
        y_pred_test = anchor_test + pred_residual_test

        # Cálculo de Métricas Oficiales en Test Set (20% Hold-out)
        mae = float(mean_absolute_error(y_test, y_pred_test))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        r2 = float(r2_score(y_test, y_pred_test))
        mape = float(np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100.0)
        rel_rmse_pct = float((rmse / y_test.mean()) * 100.0)

        # 2. Re-entrenamiento sobre el 100% del dataset limpio para máxima cobertura en producción
        self.encoder = HierarchicalCarEncoder(smoothing=2.5)
        X_full = self.encoder.fit_transform_oof(df, y, n_splits=5)
        anchor_full = X_full["anchor_precio_km_ajustado"].values
        residual_full = y.values - anchor_full

        self.model = self._build_lgbm()
        self.model.fit(X_full, residual_full)

        # Importancia de variables (Top 10)
        importances = pd.Series(
            self.model.feature_importances_, index=self.encoder.feature_columns
        ).sort_values(ascending=False)
        top_features = {k: int(v) for k, v in importances.head(10).items()}

        self.metadata = {
            "fecha_entrenamiento": datetime.now().isoformat(timespec="seconds"),
            "algoritmo": "LightGBM Hybrid Residual Regressor (LGBMRegressor + Hierarchical Target Encoding)",
            "registros_entrenamiento_total": len(df),
            "split_train_size": len(df_train),
            "split_test_size": len(df_test),
            "metricas_test_80_20": {
                "MAE_CLP": round(mae, 0),
                "RMSE_CLP": round(rmse, 0),
                "RMSE_Relativo_Pct": round(rel_rmse_pct, 2),
                "MAPE_Pct": round(mape, 2),
                "R2_Score": round(r2, 4),
            },
            "top_10_features": top_features,
        }

        self.save_artifact()

        logger.info("Métricas de Validación (Split 80/20): %s", json.dumps(self.metadata["metricas_test_80_20"]))
        logger.info("Artefacto serializado exitosamente en: %s", self.model_path)
        return self.metadata

    def save_artifact(self) -> None:
        """Guarda el modelo entrenado, sus encoders y metadatos en 'models/model.pkl'."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "encoder_state": self.encoder.to_state_dict(),
            "metadata": self.metadata,
            "version": "1.1.0",
        }
        joblib.dump(artifact, self.model_path)

    def load_artifact(self) -> Dict[str, Any]:
        """Carga 'models/model.pkl' desde disco para inferencia instantánea."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"El modelo no existe en {self.model_path}. Ejecute train.py primero.")
        # Asegurar compatibilidad en __main__ por si se cargan artefactos antiguos
        setattr(sys.modules["__main__"], "HierarchicalCarEncoder", HierarchicalCarEncoder)
        artifact = joblib.load(self.model_path)
        self.model = artifact["model"]
        if "encoder_state" in artifact:
            self.encoder = HierarchicalCarEncoder.from_state_dict(artifact["encoder_state"])
        else:
            self.encoder = artifact["encoder"]
        self.metadata = artifact.get("metadata", {})
        return self.metadata

    def predict_fair_price(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predice el 'Precio Justo' (CLP) para un DataFrame de vehículos.
        Si el DataFrame aún no tiene las columnas enriquecidas del Módulo 2,
        ejecuta automáticamente engineer_features().
        """
        if self.model is None:
            self.load_artifact()

        work_df = df.copy()
        if "generacion" not in work_df.columns or "comuna_tier" not in work_df.columns:
            pipeline = CarDataPipeline()
            work_df = pipeline.engineer_features(work_df)

        X = self.encoder.transform(work_df)
        anchor = X["anchor_precio_km_ajustado"].values
        pred_residual = self.model.predict(X)
        fair_prices = anchor + pred_residual
        # Redondear a múltiplos de $10.000 CLP y asegurar piso mínimo lógico
        return np.maximum(config.MIN_VALID_PRICE_CLP, np.round(fair_prices / 10_000.0) * 10_000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Módulo 3: Entrenamiento de Modelo Predictivo LightGBM")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forzar reentrenamiento aunque model.pkl tenga menos de 7 días",
    )
    args = parser.parse_args()

    predictor = CarPricePredictor()
    if args.force or predictor.should_retrain():
        metadata = predictor.train_and_evaluate()
    else:
        metadata = predictor.load_artifact()
        logger.info("El modelo actual en %s está vigente (< 7 días).", config.MODEL_PATH)

    metrics = metadata["metricas_test_80_20"]
    print("\n" + "=" * 68)
    print("  REPORTE DE CALIDAD DEL MODELO PREDICTIVO (TEST SPLIT 80/20)")
    print("=" * 68)
    print(f"  Algoritmo             : {metadata['algoritmo']}")
    print(f"  Muestras (Train/Test) : {metadata['split_train_size']} train / {metadata['split_test_size']} test (Total: {metadata['registros_entrenamiento_total']})")
    print(f"  MAE (Error Abs. Medio): $ {int(metrics['MAE_CLP']):,}".replace(",", "."))
    print(f"  RMSE (Error Cuad. Med): $ {int(metrics['RMSE_CLP']):,} ({metrics['RMSE_Relativo_Pct']}% relativo)".replace(",", "."))
    print(f"  MAPE (Error % Medio)  : {metrics['MAPE_Pct']} %")
    print(f"  Coeficiente R²        : {metrics['R2_Score']:.4f} ({metrics['R2_Score'] * 100:.2f}% del mercado explicado)")
    print(f"  Artefacto Guardado    : {config.MODEL_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()
