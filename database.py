"""
database.py - Gestor de Base de Datos SQLite para el Sistema de Arbitraje Automotriz.
Maneja la tabla 'autos_historico' con clave primaria única (id_aviso),
actualización de precios, control de estado ('activo' vs 'vendido') y cálculo
de días publicados.
"""

import atexit
import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import config

logger = logging.getLogger(__name__)

CRM_STAGES: List[str] = [
    "Interesado",
    "Contactado",
    "Inspección",
    "Oferta Enviada",
    "Comprado",
    "Descartado",
]

_PG_POOL_CACHE: Dict[str, Any] = {}
_PG_POOL_LOCK = threading.Lock()


def _close_pg_pools() -> None:
    """Cierra ordenadamente todos los pools PostgreSQL cacheados al finalizar el proceso."""
    with _PG_POOL_LOCK:
        pools = list(_PG_POOL_CACHE.values())
        _PG_POOL_CACHE.clear()
    for pool in pools:
        try:
            pool.close(timeout=2.0)
        except Exception:
            pass


atexit.register(_close_pg_pools)


def _evict_pg_pool(database_url: str, pool: Any) -> None:
    """Elimina del caché un pool PostgreSQL roto para permitir reintentos limpios."""
    with _PG_POOL_LOCK:
        if _PG_POOL_CACHE.get(database_url) is pool:
            _PG_POOL_CACHE.pop(database_url, None)
    try:
        pool.close(timeout=1.0)
    except Exception:
        pass


def _get_pg_pool(database_url: str) -> Any:
    """Obtiene o inicializa un ConnectionPool psycopg_pool cacheado por URL."""
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    with _PG_POOL_LOCK:
        pool = _PG_POOL_CACHE.get(database_url)
        if pool is not None and not getattr(pool, "closed", False):
            return pool
        if pool is not None:
            _PG_POOL_CACHE.pop(database_url, None)

        pool = ConnectionPool(
            database_url,
            min_size=1,
            max_size=6,
            open=True,
            timeout=15.0,
            kwargs={"row_factory": dict_row, "connect_timeout": 15},
            check=ConnectionPool.check_connection,
        )
        _PG_POOL_CACHE[database_url] = pool
        return pool


def _adapt_sql_to_pg(sql: str) -> str:
    """Convierte placeholders '?' de SQLite a '%s' para psycopg (PostgreSQL)."""
    return sql.replace("?", "%s")


class _PgCursorProxy:
    """Wrapper mínimo sobre cursor psycopg para aceptar placeholders '?'."""

    def __init__(self, cursor: Any) -> None:
        self._cur = cursor

    def execute(self, sql: str, params: Any = None) -> "_PgCursorProxy":
        pg_sql = _adapt_sql_to_pg(sql)
        if params is None or (isinstance(params, (tuple, list, dict)) and len(params) == 0):
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

    def close(self) -> None:
        self._cur.close()

    def __enter__(self) -> "_PgCursorProxy":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @property
    def rowcount(self) -> int:
        return self._cur.rowcount

    @property
    def description(self) -> Any:
        return self._cur.description


class _PgConnectionProxy:
    """Wrapper mínimo sobre conexión psycopg para exponer la misma interfaz que sqlite3."""

    def __init__(self, conn: Any, pooled: bool = False) -> None:
        self._conn = conn
        self._pooled = pooled

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
        if not self._pooled:
            self._conn.close()


class DatabaseManager:
    """Gestor transaccional dual (Supabase PostgreSQL / SQLite WAL) para anuncios de vehículos."""

    CRM_STAGES = CRM_STAGES

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

        if self.database_url:
            # Escapa '%' literales en credenciales que no estén ya URL-encoded
            self.database_url = re.sub(r"%(?![0-9A-Fa-f]{2})", "%25", self.database_url)

        self.is_postgres = bool(self.database_url)
        if not self.is_postgres:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @staticmethod
    def _normalize_owner_id(owner_id: Optional[str]) -> str:
        """Normaliza el identificador de espacio de trabajo, con fallback a 'default'."""
        clean = str(owner_id or "default").strip()
        return clean or "default"

    @contextmanager
    def get_connection(self):
        """Context manager que entrega una conexión segura a Supabase PostgreSQL o SQLite WAL."""
        if self.is_postgres:
            import psycopg
            from psycopg.rows import dict_row

            pool = None
            pool_cm = None
            raw_conn = None
            try:
                pool = _get_pg_pool(self.database_url)
                pool_cm = pool.connection()
                raw_conn = pool_cm.__enter__()
            except Exception as pool_exc:
                if pool_cm is not None:
                    try:
                        pool_cm.__exit__(type(pool_exc), pool_exc, pool_exc.__traceback__)
                    except Exception:
                        pass
                if pool is not None:
                    _evict_pg_pool(self.database_url, pool)
                logger.warning(
                    "ConnectionPool PostgreSQL no disponible (%s); usando psycopg.connect directo.",
                    pool_exc,
                )
                pool = None
                pool_cm = None
                raw_conn = psycopg.connect(
                    self.database_url,
                    row_factory=dict_row,
                    connect_timeout=15,
                )

            conn: Any = _PgConnectionProxy(raw_conn, pooled=(pool_cm is not None))
            if pool_cm is not None:
                try:
                    yield conn
                    conn.commit()
                except Exception as exc:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    logger.error("Error en transacción PostgreSQL/Supabase: %s", exc, exc_info=True)
                    pool_cm.__exit__(type(exc), exc, exc.__traceback__)
                    raise
                else:
                    pool_cm.__exit__(None, None, None)
            else:
                try:
                    yield conn
                    conn.commit()
                except Exception as exc:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    logger.error("Error en transacción PostgreSQL/Supabase: %s", exc, exc_info=True)
                    raise
                finally:
                    raw_conn.close()
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
        """Crea las tablas principales, tablas SaaS multi-espacio, RLS e índices."""
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
            CREATE TABLE IF NOT EXISTS public.user_favoritos (
                owner_id TEXT NOT NULL,
                id_aviso TEXT NOT NULL,
                nota TEXT DEFAULT '',
                etapa_crm TEXT DEFAULT 'Interesado',
                oferta_clp BIGINT DEFAULT 0,
                fecha_guardado TEXT NOT NULL,
                PRIMARY KEY (owner_id, id_aviso)
            );
            CREATE TABLE IF NOT EXISTS public.catalogo_segmentos (
                marca TEXT NOT NULL,
                modelo TEXT NOT NULL,
                fecha_agregado TEXT NOT NULL,
                PRIMARY KEY (marca, modelo)
            );
            CREATE TABLE IF NOT EXISTS public.radares_alertas (
                id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
                owner_id TEXT NOT NULL DEFAULT 'default',
                nombre TEXT NOT NULL,
                marca TEXT,
                modelo TEXT,
                anio_min INTEGER,
                precio_max BIGINT,
                km_max INTEGER,
                descuento_min_pct REAL DEFAULT 15.0,
                region TEXT,
                canal_destino TEXT DEFAULT '',
                activo INTEGER DEFAULT 1,
                fecha_creacion TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_autos_estado ON public.autos_historico(estado);
            CREATE INDEX IF NOT EXISTS idx_autos_marca_modelo ON public.autos_historico(marca, modelo, anio);
            CREATE INDEX IF NOT EXISTS idx_autos_fecha_vista ON public.autos_historico(fecha_ultima_vista);
            CREATE INDEX IF NOT EXISTS idx_autos_alerta ON public.autos_historico(alerta_enviada);
            CREATE INDEX IF NOT EXISTS idx_precio_hist_fecha ON public.precio_historial(fecha_cambio DESC);
            CREATE INDEX IF NOT EXISTS idx_precio_hist_aviso ON public.precio_historial(id_aviso, fecha_cambio ASC);
            CREATE INDEX IF NOT EXISTS idx_user_fav_owner ON public.user_favoritos(owner_id, etapa_crm);
            CREATE INDEX IF NOT EXISTS idx_radares_owner ON public.radares_alertas(owner_id, activo);
            ALTER TABLE public.user_favoritos ENABLE ROW LEVEL SECURITY;
            ALTER TABLE public.catalogo_segmentos ENABLE ROW LEVEL SECURITY;
            ALTER TABLE public.radares_alertas ENABLE ROW LEVEL SECURITY;
            INSERT INTO public.user_favoritos (owner_id, id_aviso, nota, etapa_crm, oferta_clp, fecha_guardado)
            SELECT 'default', id_aviso, COALESCE(nota, ''), 'Interesado', 0, COALESCE(fecha_guardado, '')
            FROM public.favoritos
            ON CONFLICT (owner_id, id_aviso) DO NOTHING;
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

        create_user_favorites_table = """
        CREATE TABLE IF NOT EXISTS user_favoritos (
            owner_id TEXT NOT NULL,
            id_aviso TEXT NOT NULL,
            nota TEXT DEFAULT '',
            etapa_crm TEXT DEFAULT 'Interesado',
            oferta_clp BIGINT DEFAULT 0,
            fecha_guardado TEXT NOT NULL,
            PRIMARY KEY (owner_id, id_aviso)
        );
        """

        create_custom_segments_table = """
        CREATE TABLE IF NOT EXISTS catalogo_segmentos (
            marca TEXT NOT NULL,
            modelo TEXT NOT NULL,
            fecha_agregado TEXT NOT NULL,
            PRIMARY KEY (marca, modelo)
        );
        """

        create_radars_table = """
        CREATE TABLE IF NOT EXISTS radares_alertas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            owner_id TEXT NOT NULL DEFAULT 'default',
            nombre TEXT NOT NULL,
            marca TEXT,
            modelo TEXT,
            anio_min INTEGER,
            precio_max BIGINT,
            km_max INTEGER,
            descuento_min_pct REAL DEFAULT 15.0,
            region TEXT,
            canal_destino TEXT DEFAULT '',
            activo INTEGER DEFAULT 1,
            fecha_creacion TEXT NOT NULL
        );
        """

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_autos_estado ON autos_historico(estado);",
            "CREATE INDEX IF NOT EXISTS idx_autos_marca_modelo ON autos_historico(marca, modelo, anio);",
            "CREATE INDEX IF NOT EXISTS idx_autos_fecha_vista ON autos_historico(fecha_ultima_vista);",
            "CREATE INDEX IF NOT EXISTS idx_autos_alerta ON autos_historico(alerta_enviada);",
            "CREATE INDEX IF NOT EXISTS idx_precio_hist_aviso ON precio_historial(id_aviso, fecha_cambio ASC);",
            "CREATE INDEX IF NOT EXISTS idx_user_fav_owner ON user_favoritos(owner_id, etapa_crm);",
            "CREATE INDEX IF NOT EXISTS idx_radares_owner ON radares_alertas(owner_id, activo);",
        ]

        migrate_favorites_sql = """
        INSERT OR IGNORE INTO user_favoritos (owner_id, id_aviso, nota, etapa_crm, oferta_clp, fecha_guardado)
        SELECT 'default', id_aviso, COALESCE(nota, ''), 'Interesado', 0, COALESCE(fecha_guardado, '')
        FROM favoritos;
        """

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(create_autos_table)
            cursor.execute(create_price_history_table)
            cursor.execute(create_favorites_table)
            cursor.execute(create_user_favorites_table)
            cursor.execute(create_custom_segments_table)
            cursor.execute(create_radars_table)
            for idx_sql in indexes:
                cursor.execute(idx_sql)
            cursor.execute(migrate_favorites_sql)
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

    def toggle_favorite(self, id_aviso: str, nota: str = "", owner_id: str = "default") -> bool:
        """
        Alterna un aviso en la tabla 'user_favoritos' para el espacio owner_id.
        Mantiene sincronizada la tabla heredada 'favoritos' cuando owner_id == 'default'.
        Retorna True si quedó guardado, o False si fue removido.
        """
        clean_id = str(id_aviso or "").strip()
        if not clean_id:
            return False
        clean_owner = self._normalize_owner_id(owner_id)
        clean_nota = str(nota or "").strip()
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM user_favoritos WHERE owner_id = ? AND id_aviso = ?",
                (clean_owner, clean_id),
            )
            if cursor.fetchone() is not None:
                cursor.execute(
                    "DELETE FROM user_favoritos WHERE owner_id = ? AND id_aviso = ?",
                    (clean_owner, clean_id),
                )
                if clean_owner == "default":
                    cursor.execute("DELETE FROM favoritos WHERE id_aviso = ?", (clean_id,))
                return False
            cursor.execute(
                """
                INSERT INTO user_favoritos (
                    owner_id, id_aviso, nota, etapa_crm, oferta_clp, fecha_guardado
                )
                VALUES (?, ?, ?, 'Interesado', 0, ?)
                """,
                (clean_owner, clean_id, clean_nota, now_iso),
            )
            if clean_owner == "default":
                if self.is_postgres:
                    cursor.execute(
                        """
                        INSERT INTO favoritos (id_aviso, nota, fecha_guardado)
                        VALUES (?, ?, ?)
                        ON CONFLICT (id_aviso) DO NOTHING
                        """,
                        (clean_id, clean_nota, now_iso),
                    )
                else:
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO favoritos (id_aviso, nota, fecha_guardado)
                        VALUES (?, ?, ?)
                        """,
                        (clean_id, clean_nota, now_iso),
                    )
            return True

    def update_favorite_note(
        self,
        id_aviso: str,
        nota: str,
        etapa_crm: Optional[str] = None,
        oferta_clp: Optional[int] = None,
        owner_id: str = "default",
    ) -> bool:
        """Actualiza nota, etapa CRM y/o oferta CLP de un aviso guardado en 'user_favoritos'."""
        clean_id = str(id_aviso or "").strip()
        if not clean_id:
            return False
        clean_owner = self._normalize_owner_id(owner_id)
        clean_nota = str(nota or "").strip()

        set_clauses = ["nota = ?"]
        params: List[Any] = [clean_nota]
        if etapa_crm is not None and str(etapa_crm).strip():
            raw_stage = str(etapa_crm).strip()
            stage_lookup = {s.lower(): s for s in self.CRM_STAGES}
            set_clauses.append("etapa_crm = ?")
            params.append(stage_lookup.get(raw_stage.lower(), raw_stage))
        if oferta_clp is not None:
            set_clauses.append("oferta_clp = ?")
            params.append(max(0, int(oferta_clp)))

        params.extend([clean_owner, clean_id])

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE user_favoritos SET {', '.join(set_clauses)} WHERE owner_id = ? AND id_aviso = ?",
                tuple(params),
            )
            updated = cursor.rowcount > 0
            if updated and clean_owner == "default":
                cursor.execute(
                    "UPDATE favoritos SET nota = ? WHERE id_aviso = ?",
                    (clean_nota, clean_id),
                )
            return updated

    def get_favorites_map(self, owner_id: str = "default") -> Dict[str, Dict[str, Any]]:
        """Devuelve un diccionario {id_aviso: {'nota', 'fecha_guardado', 'etapa_crm', 'oferta_clp', 'owner_id'}}."""
        clean_owner = self._normalize_owner_id(owner_id)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT owner_id, id_aviso, COALESCE(nota, '') AS nota,
                       COALESCE(etapa_crm, 'Interesado') AS etapa_crm,
                       COALESCE(oferta_clp, 0) AS oferta_clp,
                       fecha_guardado
                FROM user_favoritos
                WHERE owner_id = ?
                ORDER BY fecha_guardado DESC
                """,
                (clean_owner,),
            )
            return {
                str(r["id_aviso"]): {
                    "nota": str(r["nota"] or ""),
                    "fecha_guardado": str(r["fecha_guardado"] or ""),
                    "etapa_crm": str(r["etapa_crm"] or "Interesado"),
                    "oferta_clp": int(r["oferta_clp"] or 0),
                    "owner_id": str(r["owner_id"] or clean_owner),
                }
                for r in cursor.fetchall()
            }

    def get_listing_price_history(self, id_aviso: str) -> List[Dict[str, Any]]:
        """Devuelve el historial de variaciones de precio de un aviso ordenado ASC por fecha_cambio."""
        clean_id = str(id_aviso or "").strip()
        if not clean_id:
            return []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT precio_antiguo, precio_nuevo, fecha_cambio
                FROM precio_historial
                WHERE id_aviso = ?
                ORDER BY fecha_cambio ASC, id ASC
                """,
                (clean_id,),
            )
            return [
                {
                    "precio_antiguo": int(r["precio_antiguo"] or 0),
                    "precio_nuevo": int(r["precio_nuevo"] or 0),
                    "fecha_cambio": str(r["fecha_cambio"] or ""),
                }
                for r in cursor.fetchall()
            ]

    def get_custom_segments(self) -> List[Tuple[str, str]]:
        """Devuelve los segmentos personalizados (marca, modelo) persistidos en catalogo_segmentos."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT marca, modelo
                FROM catalogo_segmentos
                ORDER BY fecha_agregado ASC, marca ASC, modelo ASC
                """
            )
            out: List[Tuple[str, str]] = []
            for r in cursor.fetchall():
                m = str(r["marca"] or "").strip().upper()
                mod = str(r["modelo"] or "").strip().upper()
                if m and mod:
                    out.append((m, mod))
            return out

    def add_custom_segment_db(self, marca: str, modelo: str) -> bool:
        """Inserta de forma idempotente un par (marca, modelo) en catalogo_segmentos."""
        m = str(marca or "").strip().upper()
        mod = str(modelo or "").strip().upper()
        if not m or not mod:
            return False
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute(
                    """
                    INSERT INTO catalogo_segmentos (marca, modelo, fecha_agregado)
                    VALUES (?, ?, ?)
                    ON CONFLICT (marca, modelo) DO NOTHING
                    """,
                    (m, mod, now_iso),
                )
            else:
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO catalogo_segmentos (marca, modelo, fecha_agregado)
                    VALUES (?, ?, ?)
                    """,
                    (m, mod, now_iso),
                )
            return True

    def create_radar_alert(
        self,
        nombre: str,
        marca: Optional[str] = None,
        modelo: Optional[str] = None,
        anio_min: Optional[int] = None,
        precio_max: Optional[int] = None,
        km_max: Optional[int] = None,
        descuento_min_pct: float = 15.0,
        region: Optional[str] = None,
        canal_destino: str = "",
        owner_id: str = "default",
    ) -> int:
        """Crea un radar de alerta personalizado en radares_alertas y devuelve su ID."""
        clean_nombre = str(nombre or "").strip()
        if not clean_nombre:
            raise ValueError("El nombre del radar de alerta no puede estar vacío.")
        clean_owner = self._normalize_owner_id(owner_id)
        clean_marca = str(marca).strip().upper() if marca and str(marca).strip() else None
        clean_modelo = str(modelo).strip().upper() if modelo and str(modelo).strip() else None
        clean_anio_min = int(anio_min) if anio_min is not None and int(anio_min) > 0 else None
        clean_precio_max = int(precio_max) if precio_max is not None and int(precio_max) > 0 else None
        clean_km_max = int(km_max) if km_max is not None and int(km_max) > 0 else None
        clean_descuento = float(descuento_min_pct) if descuento_min_pct is not None else 15.0
        clean_region = str(region).strip() if region and str(region).strip() else None
        clean_canal = str(canal_destino or "").strip()
        now_iso = datetime.now().isoformat(timespec="seconds")

        params = (
            clean_owner,
            clean_nombre,
            clean_marca,
            clean_modelo,
            clean_anio_min,
            clean_precio_max,
            clean_km_max,
            clean_descuento,
            clean_region,
            clean_canal,
            now_iso,
        )
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if self.is_postgres:
                cursor.execute(
                    """
                    INSERT INTO radares_alertas (
                        owner_id, nombre, marca, modelo, anio_min, precio_max, km_max,
                        descuento_min_pct, region, canal_destino, activo, fecha_creacion
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    RETURNING id
                    """,
                    params,
                )
                row = cursor.fetchone()
                return int(row["id"])
            cursor.execute(
                """
                INSERT INTO radares_alertas (
                    owner_id, nombre, marca, modelo, anio_min, precio_max, km_max,
                    descuento_min_pct, region, canal_destino, activo, fecha_creacion
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                """,
                params,
            )
            return int(cursor.lastrowid)

    def list_radar_alerts(self, owner_id: str = "default") -> List[Dict[str, Any]]:
        """Lista las reglas de radar configuradas para el espacio owner_id."""
        clean_owner = self._normalize_owner_id(owner_id)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, owner_id, nombre, marca, modelo, anio_min, precio_max, km_max,
                       descuento_min_pct, region, canal_destino, activo, fecha_creacion
                FROM radares_alertas
                WHERE owner_id = ?
                ORDER BY id DESC
                """,
                (clean_owner,),
            )
            return [
                {
                    "id": int(r["id"]),
                    "owner_id": str(r["owner_id"] or clean_owner),
                    "nombre": str(r["nombre"] or ""),
                    "marca": r["marca"],
                    "modelo": r["modelo"],
                    "anio_min": int(r["anio_min"]) if r["anio_min"] is not None else None,
                    "precio_max": int(r["precio_max"]) if r["precio_max"] is not None else None,
                    "km_max": int(r["km_max"]) if r["km_max"] is not None else None,
                    "descuento_min_pct": (
                        float(r["descuento_min_pct"])
                        if r["descuento_min_pct"] is not None
                        else 15.0
                    ),
                    "region": r["region"],
                    "canal_destino": str(r["canal_destino"] or ""),
                    "activo": int(r["activo"]) if r["activo"] is not None else 1,
                    "fecha_creacion": str(r["fecha_creacion"] or ""),
                }
                for r in cursor.fetchall()
            ]

    def delete_radar_alert(self, radar_id: int, owner_id: str = "default") -> bool:
        """Elimina una regla de radar perteneciente al espacio owner_id."""
        clean_owner = self._normalize_owner_id(owner_id)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM radares_alertas WHERE id = ? AND owner_id = ?",
                (int(radar_id), clean_owner),
            )
            return cursor.rowcount > 0

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
