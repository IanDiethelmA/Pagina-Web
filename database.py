"""
database.py - Gestor de Base de Datos SQLite para el Sistema de Arbitraje Automotriz.
Maneja la tabla 'autos_historico' con clave primaria única (id_aviso),
actualización de precios, control de estado ('activo' vs 'vendido') y cálculo
de días publicados.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import config

logger = logging.getLogger(__name__)


def _adapt_sql_to_pg(sql: str) -> str:
    """Convierte placeholders '?' de SQLite a '%s' para psycopg (PostgreSQL)."""
    return sql.replace("?", "%s")


class _PgCursorProxy:
    """Wrapper mínimo sobre cursor psycopg para aceptar placeholders '?'."""

    def __init__(self, cursor: Any) -> None:
        self._cur = cursor

    def execute(self, sql: str, params: Any = None) -> "_PgCursorProxy":
        pg_sql = _adapt_sql_to_pg(sql)
        if params is None:
            self._cur.execute(pg_sql)
        else:
            self._cur.execute(pg_sql, params)
        return self

    def executemany(self, sql: str, seq_of_params: Iterable[Any]) -> "_PgCursorProxy":
        self._cur.executemany(_adapt_sql_to_pg(sql), seq_of_params)
        return self

    def fetchone(self) -> Any:
        return self._cur.fetchone()

    def fetchall(self) -> List[Any]:
        return self._cur.fetchall()

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    @property
    def description(self) -> Any:
        return self._cur.description


class _PgConnectionProxy:
    """Wrapper mínimo sobre conexión psycopg para exponer la misma interfaz que sqlite3."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def cursor(self) -> _PgCursorProxy:
        return _PgCursorProxy(self._conn.cursor())

    def execute(self, sql: str, params: Any = None) -> _PgCursorProxy:
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()


class DatabaseManager:
    """Gestor transaccional dual (Supabase PostgreSQL / SQLite WAL) para anuncios de vehículos."""

    def __init__(
        self,
        db_path: Path = config.DB_PATH,
        database_url: Optional[str] = None,
    ) -> None:
        self.db_path = Path(db_path)
        if database_url is not None:
            self.database_url = str(database_url).strip()
        elif self.db_path.resolve() == Path(config.DB_PATH).resolve():
            self.database_url = str(getattr(config, "DATABASE_URL", "") or "").strip()
        else:
            self.database_url = ""

        self.is_postgres = bool(self.database_url)
        if not self.is_postgres:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self):
        """Context manager que entrega una conexión segura a Supabase PostgreSQL o SQLite WAL."""
        if self.is_postgres:
            import psycopg
            from psycopg.rows import dict_row

            raw_conn = psycopg.connect(
                self.database_url,
                row_factory=dict_row,
                connect_timeout=15,
            )
            conn: Any = _PgConnectionProxy(raw_conn)
            try:
                yield conn
                conn.commit()
            except Exception as exc:
                conn.rollback()
                logger.error("Error en transacción PostgreSQL/Supabase: %s", exc, exc_info=True)
                raise
            finally:
                conn.close()
        else:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA foreign_keys=ON;")
                yield conn
                conn.commit()
            except Exception as exc:
                conn.rollback()
                logger.error("Error en transacción SQLite: %s", exc, exc_info=True)
                raise
            finally:
                conn.close()

    def init_db(self) -> None:
        """Crea las tablas principales ('autos_historico', 'precio_historial', 'favoritos') e índices."""
        if self.is_postgres:
            pg_ddl = """
            CREATE TABLE IF NOT EXISTS public.autos_historico (
                id_aviso TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                marca TEXT,
                modelo TEXT,
                anio INTEGER,
                kilometraje INTEGER,
                combustible TEXT,
                transmision TEXT,
                comuna TEXT,
                titulo TEXT,
                descripcion TEXT,
                precio BIGINT,
                precio_inicial BIGINT,
                precio_anterior BIGINT,
                cambios_precio INTEGER DEFAULT 0,
                estado TEXT DEFAULT 'activo',
                fecha_creacion TEXT NOT NULL,
                fecha_ultima_vista TEXT NOT NULL,
                fecha_venta TEXT,
                dias_publicado INTEGER DEFAULT 0,
                alerta_enviada INTEGER DEFAULT 0,
                categoria_alerta TEXT,
                datos_corregidos INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS public.precio_historial (
                id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                id_aviso TEXT NOT NULL REFERENCES public.autos_historico(id_aviso) ON DELETE CASCADE,
                precio_antiguo BIGINT,
                precio_nuevo BIGINT,
                fecha_cambio TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS public.favoritos (
                id_aviso TEXT PRIMARY KEY,
                nota TEXT DEFAULT '',
                fecha_guardado TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_autos_estado ON public.autos_historico(estado);
            CREATE INDEX IF NOT EXISTS idx_autos_marca_modelo ON public.autos_historico(marca, modelo, anio);
            CREATE INDEX IF NOT EXISTS idx_autos_fecha_vista ON public.autos_historico(fecha_ultima_vista);
            CREATE INDEX IF NOT EXISTS idx_autos_alerta ON public.autos_historico(alerta_enviada);
            CREATE INDEX IF NOT EXISTS idx_precio_hist_fecha ON public.precio_historial(fecha_cambio DESC);
            """
            with self.get_connection() as conn:
                conn.execute(pg_ddl)
            return
        create_autos_table = """
        CREATE TABLE IF NOT EXISTS autos_historico (
            id_aviso TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            marca TEXT,
            modelo TEXT,
            anio INTEGER,
            kilometraje INTEGER,
            combustible TEXT,
            transmision TEXT,
            comuna TEXT,
            titulo TEXT,
            descripcion TEXT,
            precio INTEGER,
            precio_inicial INTEGER,
            precio_anterior INTEGER,
            cambios_precio INTEGER DEFAULT 0,
            estado TEXT DEFAULT 'activo',
            fecha_creacion TEXT NOT NULL,
            fecha_ultima_vista TEXT NOT NULL,
            fecha_venta TEXT,
            dias_publicado INTEGER DEFAULT 0,
            alerta_enviada INTEGER DEFAULT 0,
            categoria_alerta TEXT
        );
        """

        create_price_history_table = """
        CREATE TABLE IF NOT EXISTS precio_historial (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            id_aviso TEXT NOT NULL,
            precio_antiguo INTEGER,
            precio_nuevo INTEGER,
            fecha_cambio TEXT NOT NULL,
            FOREIGN KEY (id_aviso) REFERENCES autos_historico(id_aviso)
        );
        """

        create_favorites_table = """
        CREATE TABLE IF NOT EXISTS favoritos (
            id_aviso TEXT PRIMARY KEY,
            nota TEXT DEFAULT '',
            fecha_guardado TEXT NOT NULL
        );
        """

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_autos_estado ON autos_historico(estado);",
            "CREATE INDEX IF NOT EXISTS idx_autos_marca_modelo ON autos_historico(marca, modelo, anio);",
            "CREATE INDEX IF NOT EXISTS idx_autos_fecha_vista ON autos_historico(fecha_ultima_vista);",
            "CREATE INDEX IF NOT EXISTS idx_autos_alerta ON autos_historico(alerta_enviada);",
        ]

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_autos_table)
            cursor.execute(create_price_history_table)
            cursor.execute(create_favorites_table)
            for idx_sql in indexes:
                cursor.execute(idx_sql)
            # Migración segura: columna para proteger correcciones manuales del usuario
            cols = {
                row["name"]
                for row in cursor.execute("PRAGMA table_info(autos_historico)").fetchall()
            }
            if "datos_corregidos" not in cols:
                cursor.execute(
                    "ALTER TABLE autos_historico ADD COLUMN datos_corregidos INTEGER DEFAULT 0"
                )

    @staticmethod
    def _calculate_days_between(start_iso: str, end_iso: str) -> int:
        """Calcula los días transcurridos entre dos fechas ISO-8601 (mínimo 0)."""
        try:
            dt_start = datetime.fromisoformat(start_iso)
            dt_end = datetime.fromisoformat(end_iso)
            delta = dt_end - dt_start
            return max(0, delta.days)
        except Exception:
            return 0

    def _upsert_listing_on_cursor(
        self, cursor: sqlite3.Cursor, item: Dict[str, Any], now_iso: str
    ) -> str:
        """Ejecuta el upsert de un aviso sobre un cursor activo sin abrir nueva conexión."""
        id_aviso = str(item["id_aviso"]).strip()
        nuevo_precio = int(item.get("precio") or 0)

        cursor.execute(
            """
            SELECT id_aviso, precio, precio_inicial, cambios_precio,
                   fecha_creacion, alerta_enviada, estado,
                   COALESCE(datos_corregidos, 0) AS datos_corregidos
            FROM autos_historico
            WHERE id_aviso = ?
            """,
            (id_aviso,),
        )
        existing = cursor.fetchone()

        if existing is None:
            cursor.execute(
                """
                INSERT INTO autos_historico (
                    id_aviso, url, marca, modelo, anio, kilometraje,
                    combustible, transmision, comuna, titulo, descripcion,
                    precio, precio_inicial, precio_anterior, cambios_precio,
                    estado, fecha_creacion, fecha_ultima_vista,
                    fecha_venta, dias_publicado, alerta_enviada
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'activo', ?, ?, NULL, 0, 0)
                """,
                (
                    id_aviso,
                    item.get("url", ""),
                    item.get("marca"),
                    item.get("modelo"),
                    item.get("anio"),
                    item.get("kilometraje"),
                    item.get("combustible"),
                    item.get("transmision"),
                    item.get("comuna"),
                    item.get("titulo"),
                    item.get("descripcion"),
                    nuevo_precio,
                    nuevo_precio,
                    nuevo_precio,
                    now_iso,
                    now_iso,
                ),
            )
            return "inserted"

        if existing["estado"] == "descartado":
            # Respetar avisos descartados manualmente por el usuario (link erróneo / patrocinado)
            return "updated_seen"

        precio_actual = int(existing["precio"] or 0)
        cambios_precio = int(existing["cambios_precio"] or 0)
        fecha_creacion = existing["fecha_creacion"]
        dias_pub = self._calculate_days_between(fecha_creacion, now_iso)
        alerta_enviada = int(existing["alerta_enviada"] or 0)
        is_corrected = int(existing["datos_corregidos"] or 0) == 1

        # Verificar si hubo cambio de precio válido
        if nuevo_precio > 0 and nuevo_precio != precio_actual:
            cambios_precio += 1
            # Si bajó de precio, permitimos re-evaluar alerta de arbitraje
            if nuevo_precio < precio_actual:
                alerta_enviada = 0

            cursor.execute(
                """
                INSERT INTO precio_historial (id_aviso, precio_antiguo, precio_nuevo, fecha_cambio)
                VALUES (?, ?, ?, ?)
                """,
                (id_aviso, precio_actual, nuevo_precio, now_iso),
            )

            cursor.execute(
                """
                UPDATE autos_historico
                SET url = COALESCE(NULLIF(?, ''), url),
                    marca = COALESCE(?, marca),
                    modelo = COALESCE(?, modelo),
                    anio = COALESCE(?, anio),
                    kilometraje = COALESCE(?, kilometraje),
                    combustible = COALESCE(?, combustible),
                    transmision = COALESCE(?, transmision),
                    comuna = COALESCE(?, comuna),
                    titulo = COALESCE(?, titulo),
                    descripcion = COALESCE(NULLIF(?, ''), descripcion),
                    precio_anterior = ?,
                    precio = ?,
                    cambios_precio = ?,
                    estado = 'activo',
                    fecha_ultima_vista = ?,
                    fecha_venta = NULL,
                    dias_publicado = ?,
                    alerta_enviada = ?
                WHERE id_aviso = ?
                """,
                (
                    item.get("url", ""),
                    item.get("marca"),
                    item.get("modelo"),
                    None if is_corrected else item.get("anio"),
                    None if is_corrected else item.get("kilometraje"),
                    None if is_corrected else item.get("combustible"),
                    None if is_corrected else item.get("transmision"),
                    item.get("comuna"),
                    item.get("titulo"),
                    item.get("descripcion", ""),
                    precio_actual,
                    nuevo_precio,
                    cambios_precio,
                    now_iso,
                    dias_pub,
                    alerta_enviada,
                    id_aviso,
                ),
            )
            return "updated_price"

        # Si el precio no cambió, solo actualizamos estado a 'activo', fecha_ultima_vista y días
        cursor.execute(
            """
            UPDATE autos_historico
            SET url = COALESCE(NULLIF(?, ''), url),
                kilometraje = COALESCE(?, kilometraje),
                comuna = COALESCE(?, comuna),
                descripcion = COALESCE(NULLIF(?, ''), descripcion),
                estado = 'activo',
                fecha_ultima_vista = ?,
                fecha_venta = NULL,
                dias_publicado = ?
            WHERE id_aviso = ?
            """,
            (
                item.get("url", ""),
                None if is_corrected else item.get("kilometraje"),
                item.get("comuna"),
                item.get("descripcion", ""),
                now_iso,
                dias_pub,
                id_aviso,
            ),
        )
        return "updated_seen"

    def upsert_listing(self, item: Dict[str, Any], seen_timestamp: Optional[str] = None) -> str:
        """
        Inserta un auto nuevo o actualiza uno existente en 'autos_historico'.
        - Si no existe: lo inserta como 'activo' con dias_publicado = 0.
        - Si ya existe: actualiza su precio (registrando si cambió), lo marca 'activo',
          actualiza fecha_ultima_vista y recalcula dias_publicado.
        Retorna: 'inserted' | 'updated_price' | 'updated_seen'
        """
        now_iso = seen_timestamp or datetime.now().isoformat(timespec="seconds")
        with self.get_connection() as conn:
            return self._upsert_listing_on_cursor(conn.cursor(), item, now_iso)

    def _upsert_batch_postgres(
        self, items: Iterable[Dict[str, Any]], now_iso: str
    ) -> Dict[str, int]:
        """Upsert masivo optimizado para Supabase PostgreSQL en pocos round-trips de red."""
        stats = {"inserted": 0, "updated_price": 0, "updated_seen": 0, "errors": 0}
        dedup_items: Dict[str, Dict[str, Any]] = {}
        for item in items:
            raw_id = str(item.get("id_aviso") or "").strip()
            if not raw_id:
                continue
            if item.get("url") is None:
                stats["errors"] += 1
                continue
            dedup_items[raw_id] = item

        if not dedup_items:
            return stats

        ids_list = list(dedup_items.keys())
        to_insert: List[Tuple[Any, ...]] = []
        to_price_hist: List[Tuple[Any, ...]] = []
        to_update_price: List[Tuple[Any, ...]] = []
        to_update_seen: List[Tuple[Any, ...]] = []

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id_aviso, precio, precio_inicial, cambios_precio,
                       fecha_creacion, alerta_enviada, estado,
                       COALESCE(datos_corregidos, 0) AS datos_corregidos
                FROM autos_historico
                WHERE id_aviso = ANY(?)
                """,
                (ids_list,),
            )
            existing_map = {str(r["id_aviso"]): r for r in cursor.fetchall()}

            for id_aviso, item in dedup_items.items():
                nuevo_precio = int(item.get("precio") or 0)
                existing = existing_map.get(id_aviso)
                if existing is None:
                    to_insert.append(
                        (
                            id_aviso,
                            item.get("url", ""),
                            item.get("marca"),
                            item.get("modelo"),
                            item.get("anio"),
                            item.get("kilometraje"),
                            item.get("combustible"),
                            item.get("transmision"),
                            item.get("comuna"),
                            item.get("titulo"),
                            item.get("descripcion"),
                            nuevo_precio,
                            nuevo_precio,
                            nuevo_precio,
                            now_iso,
                            now_iso,
                        )
                    )
                    continue

                if existing["estado"] == "descartado":
                    stats["updated_seen"] += 1
                    continue

                precio_actual = int(existing["precio"] or 0)
                cambios_precio = int(existing["cambios_precio"] or 0)
                fecha_creacion = existing["fecha_creacion"]
                dias_pub = self._calculate_days_between(fecha_creacion, now_iso)
                alerta_enviada = int(existing["alerta_enviada"] or 0)
                is_corrected = int(existing["datos_corregidos"] or 0) == 1

                if nuevo_precio > 0 and nuevo_precio != precio_actual:
                    cambios_precio += 1
                    if nuevo_precio < precio_actual:
                        alerta_enviada = 0
                    to_price_hist.append((id_aviso, precio_actual, nuevo_precio, now_iso))
                    to_update_price.append(
                        (
                            item.get("url", ""),
                            item.get("marca"),
                            item.get("modelo"),
                            None if is_corrected else item.get("anio"),
                            None if is_corrected else item.get("kilometraje"),
                            None if is_corrected else item.get("combustible"),
                            None if is_corrected else item.get("transmision"),
                            item.get("comuna"),
                            item.get("titulo"),
                            item.get("descripcion", ""),
                            precio_actual,
                            nuevo_precio,
                            cambios_precio,
                            now_iso,
                            dias_pub,
                            alerta_enviada,
                            id_aviso,
                        )
                    )
                else:
                    to_update_seen.append(
                        (
                            item.get("url", ""),
                            None if is_corrected else item.get("kilometraje"),
                            item.get("comuna"),
                            item.get("descripcion", ""),
                            now_iso,
                            dias_pub,
                            id_aviso,
                        )
                    )

            if to_insert:
                cursor.executemany(
                    """
                    INSERT INTO autos_historico (
                        id_aviso, url, marca, modelo, anio, kilometraje,
                        combustible, transmision, comuna, titulo, descripcion,
                        precio, precio_inicial, precio_anterior, cambios_precio,
                        estado, fecha_creacion, fecha_ultima_vista,
                        fecha_venta, dias_publicado, alerta_enviada
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'activo', ?, ?, NULL, 0, 0)
                    ON CONFLICT (id_aviso) DO NOTHING
                    """,
                    to_insert,
                )
                stats["inserted"] += len(to_insert)

            if to_price_hist:
                cursor.executemany(
                    """
                    INSERT INTO precio_historial (id_aviso, precio_antiguo, precio_nuevo, fecha_cambio)
                    VALUES (?, ?, ?, ?)
                    """,
                    to_price_hist,
                )

            if to_update_price:
                cursor.executemany(
                    """
                    UPDATE autos_historico
                    SET url = COALESCE(NULLIF(?, ''), url),
                        marca = COALESCE(?, marca),
                        modelo = COALESCE(?, modelo),
                        anio = COALESCE(?, anio),
                        kilometraje = COALESCE(?, kilometraje),
                        combustible = COALESCE(?, combustible),
                        transmision = COALESCE(?, transmision),
                        comuna = COALESCE(?, comuna),
                        titulo = COALESCE(?, titulo),
                        descripcion = COALESCE(NULLIF(?, ''), descripcion),
                        precio_anterior = ?,
                        precio = ?,
                        cambios_precio = ?,
                        estado = 'activo',
                        fecha_ultima_vista = ?,
                        fecha_venta = NULL,
                        dias_publicado = ?,
                        alerta_enviada = ?
                    WHERE id_aviso = ?
                    """,
                    to_update_price,
                )
                stats["updated_price"] += len(to_update_price)

            if to_update_seen:
                cursor.executemany(
                    """
                    UPDATE autos_historico
                    SET url = COALESCE(NULLIF(?, ''), url),
                        kilometraje = COALESCE(?, kilometraje),
                        comuna = COALESCE(?, comuna),
                        descripcion = COALESCE(NULLIF(?, ''), descripcion),
                        estado = 'activo',
                        fecha_ultima_vista = ?,
                        fecha_venta = NULL,
                        dias_publicado = ?
                    WHERE id_aviso = ?
                    """,
                    to_update_seen,
                )
                stats["updated_seen"] += len(to_update_seen)

        return stats

    def upsert_batch(self, items: Iterable[Dict[str, Any]], seen_timestamp: Optional[str] = None) -> Dict[str, int]:
        """Procesa un lote de anuncios en una sola transacción (Postgres masivo o SQLite con SAVEPOINT)."""
        now_iso = seen_timestamp or datetime.now().isoformat(timespec="seconds")
        if self.is_postgres:
            return self._upsert_batch_postgres(items, now_iso)

        stats = {"inserted": 0, "updated_price": 0, "updated_seen": 0, "errors": 0}
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for item in items:
                if not item.get("id_aviso"):
                    continue
                try:
                    cursor.execute("SAVEPOINT sp_upsert_item")
                    res = self._upsert_listing_on_cursor(cursor, item, now_iso)
                    cursor.execute("RELEASE SAVEPOINT sp_upsert_item")
                    stats[res] = stats.get(res, 0) + 1
                except Exception as exc:
                    try:
                        cursor.execute("ROLLBACK TO SAVEPOINT sp_upsert_item")
                        cursor.execute("RELEASE SAVEPOINT sp_upsert_item")
                    except Exception:
                        pass
                    stats["errors"] += 1
                    logger.error("Error haciendo upsert de aviso %s: %s", item.get("id_aviso"), exc)
        return stats

    def mark_missing_as_sold(
        self,
        active_seen_ids: Set[str],
        reference_timestamp: Optional[str] = None,
        marca_filter: Optional[str] = None,
        modelo_filter: Optional[str] = None,
    ) -> int:
        """
        Marca como 'vendido' todos los anuncios que estaban 'activo' pero no
        aparecieron en el conjunto de IDs vistos hoy (active_seen_ids),
        guardando fecha_venta y calculando los días que estuvieron publicados.
        Si se especifica marca_filter / modelo_filter, restringe el alcance a ese segmento.
        """
        now_iso = reference_timestamp or datetime.now().isoformat(timespec="seconds")
        sold_count = 0

        query = "SELECT id_aviso, fecha_creacion, fecha_ultima_vista FROM autos_historico WHERE estado = 'activo'"
        params: List[Any] = []
        if marca_filter:
            query += " AND UPPER(marca) = ?"
            params.append(marca_filter.strip().upper())
        if modelo_filter:
            query += " AND UPPER(modelo) = ?"
            params.append(modelo_filter.strip().upper())

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, tuple(params))
            currently_active = cursor.fetchall()

            for row in currently_active:
                id_aviso = row["id_aviso"]
                if id_aviso not in active_seen_ids:
                    # Una búsqueda parcial no cubre todas las páginas: solo se marca como vendido
                    # si el aviso lleva al menos STALE_DAYS_TO_MARK_SOLD días sin ser visto.
                    dias_sin_ver = self._calculate_days_between(row["fecha_ultima_vista"], now_iso)
                    if dias_sin_ver < config.STALE_DAYS_TO_MARK_SOLD:
                        continue
                    dias_pub = self._calculate_days_between(row["fecha_creacion"], now_iso)
                    cursor.execute(
                        """
                        UPDATE autos_historico
                        SET estado = 'vendido',
                            fecha_venta = ?,
                            dias_publicado = ?
                        WHERE id_aviso = ?
                        """,
                        (now_iso, dias_pub, id_aviso),
                    )
                    sold_count += 1

        return sold_count

    def mark_listing_alerted(self, id_aviso: str, categoria_alerta: str) -> None:
        """Marca un anuncio como ya alertado para no enviar spam repetido."""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE autos_historico
                SET alerta_enviada = 1,
                    categoria_alerta = ?
                WHERE id_aviso = ?
                """,
                (categoria_alerta, str(id_aviso)),
            )

    def update_listing_status(self, id_aviso: str, new_status: str) -> bool:
        """
        Actualiza manualmente el estado de un aviso ('activo', 'vendido' o 'descartado')
        desde la interfaz para descartar publicaciones vendidas o links erróneos.
        """
        allowed = {"activo", "vendido", "descartado"}
        status_clean = str(new_status).strip().lower()
        if status_clean not in allowed:
            raise ValueError(f"Estado inválido '{new_status}'. Permitidos: {sorted(allowed)}")

        now_iso = datetime.now().isoformat(timespec="seconds")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if status_clean == "activo":
                cursor.execute(
                    """
                    UPDATE autos_historico
                    SET estado = 'activo',
                        fecha_venta = NULL
                    WHERE id_aviso = ?
                    """,
                    (str(id_aviso).strip(),),
                )
            else:
                cursor.execute(
                    """
                    UPDATE autos_historico
                    SET estado = ?,
                        fecha_venta = ?
                    WHERE id_aviso = ?
                    """,
                    (status_clean, now_iso, str(id_aviso).strip()),
                )
            return cursor.rowcount > 0

    def update_listing_specs(
        self,
        id_aviso: str,
        kilometraje: int,
        transmision: str,
        combustible: str,
        anio: Optional[int] = None,
        precio: Optional[int] = None,
    ) -> bool:
        """
        Corrige manualmente kilometraje, transmisión, combustible, año o precio de un aviso
        y marca datos_corregidos=1 para que futuros escaneos no sobrescriban la corrección.
        """
        clean_id = str(id_aviso).strip()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE autos_historico
                SET kilometraje = ?,
                    transmision = ?,
                    combustible = ?,
                    anio = COALESCE(?, anio),
                    precio = COALESCE(?, precio),
                    datos_corregidos = 1
                WHERE id_aviso = ?
                """,
                (
                    int(kilometraje),
                    str(transmision).strip(),
                    str(combustible).strip(),
                    int(anio) if anio is not None else None,
                    int(precio) if precio is not None else None,
                    clean_id,
                ),
            )
            return cursor.rowcount > 0

    def cleanup_sponsored_ad_listings(self) -> int:
        """
        Marca como 'descartado' anuncios publicitarios patrocinados (ej. click1.mercadolibre.cl)
        que enlazan a otros vehículos distintos al segmento consultado.
        """
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE autos_historico
                SET estado = 'descartado',
                    fecha_venta = ?
                WHERE estado = 'activo'
                  AND (
                      LOWER(COALESCE(url, '')) LIKE '%click1.mercadolibre%'
                      OR LOWER(COALESCE(url, '')) LIKE '%/brand_ads/%'
                      OR LOWER(COALESCE(url, '')) LIKE '%/mclics/%'
                      OR LOWER(COALESCE(url, '')) LIKE '%/publicidad/%'
                  )
                """,
                (now_iso,),
            )
            return cursor.rowcount

    def toggle_favorite(self, id_aviso: str, nota: str = "") -> bool:
        """
        Alterna un aviso en la tabla 'favoritos'.
        Retorna True si quedó guardado en favoritos, o False si fue removido.
        """
        clean_id = str(id_aviso).strip()
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM favoritos WHERE id_aviso = ?", (clean_id,))
            if cursor.fetchone() is not None:
                cursor.execute("DELETE FROM favoritos WHERE id_aviso = ?", (clean_id,))
                return False
            cursor.execute(
                """
                INSERT INTO favoritos (id_aviso, nota, fecha_guardado)
                VALUES (?, ?, ?)
                """,
                (clean_id, str(nota or "").strip(), now_iso),
            )
            return True

    def update_favorite_note(self, id_aviso: str, nota: str) -> bool:
        """Actualiza la nota personal de un aviso guardado en favoritos."""
        clean_id = str(id_aviso).strip()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE favoritos SET nota = ? WHERE id_aviso = ?",
                (str(nota or "").strip(), clean_id),
            )
            return cursor.rowcount > 0

    def get_favorites_map(self) -> Dict[str, Dict[str, str]]:
        """Devuelve un diccionario {id_aviso: {'nota': ..., 'fecha_guardado': ...}}."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id_aviso, COALESCE(nota, '') AS nota, fecha_guardado FROM favoritos ORDER BY fecha_guardado DESC"
            )
            return {
                str(r["id_aviso"]): {
                    "nota": str(r["nota"] or ""),
                    "fecha_guardado": str(r["fecha_guardado"] or ""),
                }
                for r in cursor.fetchall()
            }

    def purge_simulated_records(self) -> int:
        """
        Elimina registros sintéticos de prueba ('CL-AD-20260%') cuando ingresan datos reales.
        Protege los avisos reales de Chileautos (que pueden compartir prefijo de ID) exigiendo
        que la descripción no provenga del índice real de Chileautos.
        """
        sim_filter = (
            "id_aviso LIKE 'CL-AD-20260%' "
            "AND COALESCE(descripcion, '') NOT LIKE '%Chileautos via Google Index%'"
        )
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM precio_historial WHERE id_aviso IN (SELECT id_aviso FROM autos_historico WHERE {sim_filter})"
            )
            cursor.execute(f"DELETE FROM autos_historico WHERE {sim_filter}")
            return cursor.rowcount

    def get_summary_stats(self) -> Dict[str, Any]:
        """Devuelve métricas generales de la base de datos."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS total FROM autos_historico")
            total = cursor.fetchone()["total"]

            cursor.execute("SELECT COUNT(*) AS activos FROM autos_historico WHERE estado = 'activo'")
            activos = cursor.fetchone()["activos"]

            cursor.execute("SELECT COUNT(*) AS vendidos FROM autos_historico WHERE estado = 'vendido'")
            vendidos = cursor.fetchone()["vendidos"]

            cursor.execute("SELECT COUNT(*) AS cambios FROM precio_historial")
            cambios = cursor.fetchone()["cambios"]

            return {
                "total_historicos": total,
                "activos": activos,
                "vendidos": vendidos,
                "cambios_de_precio_registrados": cambios,
            }

    def read_dataframe(self, query: str, params: Tuple[Any, ...] = ()) -> Any:
        """Ejecuta una consulta SELECT y devuelve un DataFrame de Pandas compatible con SQLite y Postgres."""
        import pandas as pd

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cols = [d[0] for d in cursor.description] if cursor.description else []
            if not rows:
                return pd.DataFrame(columns=cols)
            if isinstance(rows[0], dict):
                return pd.DataFrame(rows, columns=cols)
            return pd.DataFrame([dict(r) for r in rows], columns=cols)

    def get_sold_today_count(self, today_prefix: str) -> int:
        """Cuenta cuántos avisos pasaron a estado 'vendido' en la fecha dada (YYYY-MM-DD)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) AS cnt FROM autos_historico WHERE estado = 'vendido' AND fecha_venta LIKE ?",
                (f"{today_prefix}%",),
            )
            return int(cursor.fetchone()["cnt"] or 0)
